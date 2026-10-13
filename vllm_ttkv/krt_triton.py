"""`krt` in Triton: post-rotary scores straight from a pre-rotary, packed slot.

THE ONLY SCORE KERNEL THE PRE-ROTARY PATH NEEDS. `qt` and `resid_add` belong to the absorbed path
and are dead here: absorption hoists the decoder into the query because the decoder is the same for
every row, but with a pre-rotary store the rotation is a function of the KEY's own position, so
`R_i` sits between the reconstruction and the contraction and cannot be hoisted out.

WHY TRITON AND NOT THE SHIPPED CUDA KERNEL
    `ttkv_kernels.cu` is hand-tuned (shared-memory layout, bank-conflict avoidance, a KMxKD thread
    tiling chosen because the reconstruct loop is INSTRUCTION bound) and it stays as the opt-in fast
    path. It is not the default because a vLLM plugin that ships an AOT extension forces every user
    to have a matching toolchain, CUDA version and arch at install time. Triton is a vLLM dependency
    already and JITs against whatever is present, so this is the portable default.

TWO TRITON LANDMINES, BOTH HIT WHILE BUILDING THIS, BOTH NOW CLOSED

    1. `do_not_specialize=["n", "r1"]` ON THE DECORATOR IS LOAD-BEARING.
       Triton specializes integer arguments that are divisible by 16 by compiling a separate
       variant for them, and on Triton 3.2 that variant ABORTS the compiler with an LLVM
       `ArrayRef<mlir::Value>::operator[]` assertion and no Triton diagnostic at all. Measured
       directly: r1 = 16, 32, 64 crash; r1 = 33, 61 compile and run. Since the mask on the `tl.dot`
       operand is `ar < r1`, a rank that happens to be a multiple of 16 is the trigger -- and real
       per-layer ranks are often round numbers, so this would fire intermittently in production
       rather than reproducibly in a test. Disabling the specialization removes the variant.
       If you touch the signature, re-run `tests/test_krt_triton.py`: its r1 sweep is what catches it.

    2. Nothing is factored into a `@triton.jit` device function.
       The first draft had two helpers (`_load_f16`, `_resid_half`); they are inlined here. Their
       removal was NOT what fixed the crash (the landmine above was), but the inline form is the one
       the bisect harness was reduced to and the one that has been exercised, so it stays until
       something needs to change.

WHAT IT READS -- one token's slot, laid out token-major by `layout.pack_k_slots`:
    [ coeff_q : r1 * qbits/8 bytes ][ coeff_scale : fp16 ][ head 0 row ][ head 1 row ]...
    one head's row is  [ bitmap : d/8 bytes ][ packed : PB bytes ][ scale : fp16 ]
    where PB = ceil(res_k * res_bits / 8).

    Two of those fields land on ODD byte offsets for real configs: r1=61 puts `coeff_scale` at byte
    61, and a per-head row stride of 50 puts the residual ruler at 111. Both are therefore assembled
    from two `uint8` loads and bitcast rather than read as a native fp16. That is a property of the
    format, not a defect in the kernel.

THE FOUR STEPS, and the third is the whole point:
    reconstruct   K_pre = (coeff * coeff_scale) @ decoder       -- tl.dot, not a thread loop
    patch         K_pre += residual code * per-row ruler         -- the lanes the fit missed
    rotate        K_post = RoPE(K_pre, THAT ROW's own position)  -- cannot be hoisted out
    contract      scores = q_post . K_post

THE FAILURE MODE THIS FILE IS SHAPED AROUND. Skipping or mis-applying the rotation produces finite,
correctly-shaped, plausible scores -- no crash, no shape mismatch, nothing to catch it. The project's
standing signal for a silent permutation is an ANTI-correlated metric rather than noise, so the test
compares against `vllm_ttkv.reference.raw_scores` (the direct observable of the rotation) and
carries two negative controls.

`G` is 4 on Llama-3.1, below `tl.dot`'s minimum dim, so the contraction is a broadcast-and-sum.
ASCII only.
"""
import torch
import triton
import triton.language as tl

from vllm_ttkv.layout import packed_row_bytes


@triton.jit(do_not_specialize=["n", "r1"])
def _krt_kernel(
    u8, i8, dec, cos_p, sin_p, q_p, s_p, n, r1,
    SLOT: tl.constexpr, ROW_OFF: tl.constexpr, ROW_STEP: tl.constexpr, CSCALE_OFF: tl.constexpr,
    PB: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, BM: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, BR: tl.constexpr,
    RES_BITS: tl.constexpr, HAS_RES: tl.constexpr,
):
    pid = tl.program_id(0)
    hv = tl.program_id(1)
    DSTR: tl.constexpr = H * D
    HALF_OFF: tl.constexpr = 1 << (RES_BITS - 1)
    it = pid * KT + tl.arange(0, KT)
    tm = it < n
    # One token's slot, then this kv head's row inside it. Token-major, so a row's address is a
    # plain function of the token and never of the sequence length.
    row = it * SLOT + ROW_OFF + hv * ROW_STEP
    lo = tl.arange(0, HALF)
    hi = lo + HALF

    # ---- 1. reconstruct. Two halves throughout, so the rotation below needs no register shuffle:
    #         both members of every (j, j+D/2) pair are already separate live tensors.
    cq_s = tl.load(u8 + it * SLOT + CSCALE_OFF).to(tl.int16)
    cq_s = (cq_s | (tl.load(u8 + it * SLOT + CSCALE_OFF + 1).to(tl.int16) << 8)
            ).to(tl.float16, bitcast=True).to(tl.float32)
    acc_lo = tl.zeros((KT, HALF), dtype=tl.float32)
    acc_hi = tl.zeros((KT, HALF), dtype=tl.float32)
    for r0 in range(0, r1, BR):
        ar = r0 + tl.arange(0, BR)
        rm = ar < r1
        g = tl.load(i8 + it[:, None] * SLOT + ar[None, :],
                    mask=tm[:, None] & rm[None, :], other=0).to(tl.float32) * cq_s[:, None]
        # `input_precision="ieee"` IS REQUIRED, not a nicety: Triton defaults fp32 `tl.dot` to TF32,
        # which carries 10 mantissa bits. Left at the default this measured 7.58e-04 against the
        # reference where the CUDA kernel's fp32 FMA path measures 4.36e-07 -- a 1700x regression
        # that still returns correctly-shaped, plausible scores. The reconstruction is the one step
        # where a compressed store can afford to be exact, so it is exact.
        acc_lo = tl.dot(g, tl.load(dec + ar[:, None] * DSTR + (hv * D + lo)[None, :],
                                   mask=rm[:, None], other=0.0), acc_lo, input_precision="ieee")
        acc_hi = tl.dot(g, tl.load(dec + ar[:, None] * DSTR + (hv * D + hi)[None, :],
                                   mask=rm[:, None], other=0.0), acc_hi, input_precision="ieee")

    # ---- 2. patch the lanes the fit missed. A high-half lane's rank has to carry the low half's
    #         total, because the codes are one MSB-first bit stream across the whole row.
    if HAS_RES:
        bm_lo = tl.load(u8 + row[:, None] + (lo // 8)[None, :],
                        mask=tm[:, None], other=0).to(tl.int32)
        is_lo = ((bm_lo >> (7 - (lo % 8))[None, :]) & 1) != 0
        rank_lo = tl.cumsum(is_lo.to(tl.int32), axis=1) - is_lo.to(tl.int32)
        total_lo = tl.sum(is_lo.to(tl.int32), axis=1)
        byte = (rank_lo * RES_BITS) // 8
        sh = (rank_lo * RES_BITS) % 8
        p0 = tl.load(u8 + row[:, None] + BM + byte, mask=tm[:, None], other=0).to(tl.int32)
        p1 = tl.load(u8 + row[:, None] + BM + byte + 1,
                     mask=tm[:, None] & (byte + 1 < PB), other=0).to(tl.int32)
        code = (((p0 << 8) | p1) >> (16 - sh - RES_BITS)) & ((1 << RES_BITS) - 1)
        rs = tl.load(u8 + row + BM + PB, mask=tm, other=0).to(tl.int16)
        rs = (rs | (tl.load(u8 + row + BM + PB + 1, mask=tm, other=0).to(tl.int16) << 8)
              ).to(tl.float16, bitcast=True).to(tl.float32)
        acc_lo += tl.where(is_lo, (code.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

        bm_hi = tl.load(u8 + row[:, None] + (hi // 8)[None, :],
                        mask=tm[:, None], other=0).to(tl.int32)
        is_hi = ((bm_hi >> (7 - (hi % 8))[None, :]) & 1) != 0
        rank_hi = total_lo[:, None] + tl.cumsum(is_hi.to(tl.int32), axis=1) - is_hi.to(tl.int32)
        byte2 = (rank_hi * RES_BITS) // 8
        sh2 = (rank_hi * RES_BITS) % 8
        q0 = tl.load(u8 + row[:, None] + BM + byte2, mask=tm[:, None], other=0).to(tl.int32)
        q1 = tl.load(u8 + row[:, None] + BM + byte2 + 1,
                     mask=tm[:, None] & (byte2 + 1 < PB), other=0).to(tl.int32)
        code2 = (((q0 << 8) | q1) >> (16 - sh2 - RES_BITS)) & ((1 << RES_BITS) - 1)
        acc_hi += tl.where(is_hi, (code2.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

    # ---- 3. rotate each row by ITS OWN position, `(j, j + D/2)` pairing.
    y_lo = (acc_lo * tl.load(cos_p + it[:, None] * D + lo[None, :], mask=tm[:, None], other=0.0)
            - acc_hi * tl.load(sin_p + it[:, None] * D + lo[None, :], mask=tm[:, None],
                               other=0.0))
    y_hi = (acc_hi * tl.load(cos_p + it[:, None] * D + hi[None, :], mask=tm[:, None], other=0.0)
            + acc_lo * tl.load(sin_p + it[:, None] * D + hi[None, :], mask=tm[:, None],
                               other=0.0))

    # ---- 4. contract. G < 16, below tl.dot's minimum, so broadcast-and-sum.
    gq = tl.arange(0, G)
    q_lo = tl.load(q_p + (hv * G + gq)[:, None] * D + lo[None, :]).to(tl.float32)
    q_hi = tl.load(q_p + (hv * G + gq)[:, None] * D + hi[None, :]).to(tl.float32)
    sc = tl.sum(y_lo[None, :, :] * q_lo[:, None, :] + y_hi[None, :, :] * q_hi[:, None, :], axis=2)
    tl.store(s_p + (hv * G + gq)[:, None] * n + it[None, :], sc, mask=tm[None, :])


@triton.jit(do_not_specialize=["n", "r1", "Q", "P0"])
def _krt_multi_kernel(
    u8, i8, dec, cos_p, sin_p, q_p, s_p, n, r1, Q, P0,
    SLOT: tl.constexpr, ROW_OFF: tl.constexpr, ROW_STEP: tl.constexpr, CSCALE_OFF: tl.constexpr,
    PB: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, BM: tl.constexpr,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    KT: tl.constexpr, BR: tl.constexpr, BQ: tl.constexpr,
    RES_BITS: tl.constexpr, HAS_RES: tl.constexpr, CAUSAL: tl.constexpr,
):
    """`_krt_kernel` with a QUERY BLOCK, so one reconstruction serves `BQ` queries.

    WHAT CHANGES AND WHY IT IS THE WHOLE POINT. Steps 1-3 (reconstruct, patch, rotate) depend on the
    stored row and NOT on the query. The single-query kernel therefore pays `r1*d` per row for a
    single query, which is what makes the read 27x dense's arithmetic. Here the grid carries a third
    axis over queries, so a row block is reconstructed once per (row block, head, query block) and
    reused for all `BQ` queries in it. With `BQ*GP` comfortably above `r1`, the reconstruction stops
    dominating -- it falls to roughly `r1/(BQ*G)` of the contraction.

    It does not vanish: a larger query block means more reuse, and the grid's query axis means the
    reconstruction is paid once per block rather than once per query. That is the standard attention
    tiling trade, made here in favour of the reconstruction because the reconstruction is the
    expensive part on this path.
    """
    pid = tl.program_id(0)
    hv = tl.program_id(1)
    jb = tl.program_id(2)
    DSTR: tl.constexpr = H * D
    HALF_OFF: tl.constexpr = 1 << (RES_BITS - 1)
    it = pid * KT + tl.arange(0, KT)
    tm = it < n
    row = it * SLOT + ROW_OFF + hv * ROW_STEP
    lo = tl.arange(0, HALF)
    hi = lo + HALF

    # ---- 1. reconstruct, identical to the single-query kernel: two halves throughout, so the
    #         rotation needs no register shuffle.
    cq_s = tl.load(u8 + it * SLOT + CSCALE_OFF).to(tl.int16)
    cq_s = (cq_s | (tl.load(u8 + it * SLOT + CSCALE_OFF + 1).to(tl.int16) << 8)
            ).to(tl.float16, bitcast=True).to(tl.float32)
    acc_lo = tl.zeros((KT, HALF), dtype=tl.float32)
    acc_hi = tl.zeros((KT, HALF), dtype=tl.float32)
    for r0 in range(0, r1, BR):
        ar = r0 + tl.arange(0, BR)
        rm = ar < r1
        g = tl.load(i8 + it[:, None] * SLOT + ar[None, :],
                    mask=tm[:, None] & rm[None, :], other=0).to(tl.float32) * cq_s[:, None]
        acc_lo = tl.dot(g, tl.load(dec + ar[:, None] * DSTR + (hv * D + lo)[None, :],
                                   mask=rm[:, None], other=0.0), acc_lo, input_precision="ieee")
        acc_hi = tl.dot(g, tl.load(dec + ar[:, None] * DSTR + (hv * D + hi)[None, :],
                                   mask=rm[:, None], other=0.0), acc_hi, input_precision="ieee")

    # ---- 2. patch the lanes the fit missed. A high-half lane's rank carries the low half's total.
    if HAS_RES:
        bm_lo = tl.load(u8 + row[:, None] + (lo // 8)[None, :],
                        mask=tm[:, None], other=0).to(tl.int32)
        is_lo = ((bm_lo >> (7 - (lo % 8))[None, :]) & 1) != 0
        rank_lo = tl.cumsum(is_lo.to(tl.int32), axis=1) - is_lo.to(tl.int32)
        total_lo = tl.sum(is_lo.to(tl.int32), axis=1)
        byte = (rank_lo * RES_BITS) // 8
        sh = (rank_lo * RES_BITS) % 8
        p0 = tl.load(u8 + row[:, None] + BM + byte, mask=tm[:, None], other=0).to(tl.int32)
        p1 = tl.load(u8 + row[:, None] + BM + byte + 1,
                     mask=tm[:, None] & (byte + 1 < PB), other=0).to(tl.int32)
        code = (((p0 << 8) | p1) >> (16 - sh - RES_BITS)) & ((1 << RES_BITS) - 1)
        rs = tl.load(u8 + row + BM + PB, mask=tm, other=0).to(tl.int16)
        rs = (rs | (tl.load(u8 + row + BM + PB + 1, mask=tm, other=0).to(tl.int16) << 8)
              ).to(tl.float16, bitcast=True).to(tl.float32)
        acc_lo += tl.where(is_lo, (code.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

        bm_hi = tl.load(u8 + row[:, None] + (hi // 8)[None, :],
                        mask=tm[:, None], other=0).to(tl.int32)
        is_hi = ((bm_hi >> (7 - (hi % 8))[None, :]) & 1) != 0
        rank_hi = total_lo[:, None] + tl.cumsum(is_hi.to(tl.int32), axis=1) - is_hi.to(tl.int32)
        byte2 = (rank_hi * RES_BITS) // 8
        sh2 = (rank_hi * RES_BITS) % 8
        q0 = tl.load(u8 + row[:, None] + BM + byte2, mask=tm[:, None], other=0).to(tl.int32)
        q1 = tl.load(u8 + row[:, None] + BM + byte2 + 1,
                     mask=tm[:, None] & (byte2 + 1 < PB), other=0).to(tl.int32)
        code2 = (((q0 << 8) | q1) >> (16 - sh2 - RES_BITS)) & ((1 << RES_BITS) - 1)
        acc_hi += tl.where(is_hi, (code2.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

    # ---- 3. rotate each row by ITS OWN position, `(j, j + D/2)` pairing.
    y_lo = (acc_lo * tl.load(cos_p + it[:, None] * D + lo[None, :], mask=tm[:, None], other=0.0)
            - acc_hi * tl.load(sin_p + it[:, None] * D + lo[None, :], mask=tm[:, None],
                               other=0.0))
    y_hi = (acc_hi * tl.load(cos_p + it[:, None] * D + hi[None, :], mask=tm[:, None], other=0.0)
            + acc_lo * tl.load(sin_p + it[:, None] * D + hi[None, :], mask=tm[:, None],
                               other=0.0))

    # ---- 4. contract the SAME reconstructed rows against a block of queries. `GP` pads the group
    #         axis to a power of two so `BQ*GP` clears `tl.dot`'s minimum of 16 without allocating a
    #         padded query: the padding rows are MASKED to zero, and a zero query scores zero.
    jj = jb * BQ + tl.arange(0, BQ)
    jm = jj < Q
    gq = tl.arange(0, GP)
    qm = gq < G
    qoff = ((hv * G + gq)[None, :, None] * Q + jj[:, None, None]) * D + lo[None, None, :]
    qlm = jm[:, None, None] & qm[None, :, None]
    q_lo = tl.load(q_p + qoff, mask=qlm, other=0.0).to(tl.float32)
    q_hi = tl.load(q_p + qoff + HALF, mask=qlm, other=0.0).to(tl.float32)
    sc = tl.dot(tl.reshape(q_lo, (BQ * GP, HALF)), tl.trans(y_lo), input_precision="ieee")
    sc = tl.dot(tl.reshape(q_hi, (BQ * GP, HALF)), tl.trans(y_hi), sc, input_precision="ieee")

    if CAUSAL:
        # Query `j` sits at `P0 + j` and may attend to rows `0 .. P0 + j`. `-inf` rather than zero:
        # zero is a legitimate score, and the caller's softmax would treat it as a real one.
        pos = P0 + jj
        allow = tl.broadcast_to((it[None, :] <= pos[:, None])[:, None, :], (BQ, GP, KT))
        sc = tl.where(tl.reshape(allow, (BQ * GP, KT)), sc, float("-inf"))

    soff = ((hv * G + gq)[None, :, None] * Q + jj[:, None, None]) * n + it[None, None, :]
    tl.store(s_p + soff, tl.reshape(sc, (BQ, GP, KT)),
             mask=qlm & tm[None, None, :])


def krt_multi_from_slots(lay, slots, decoder_f, cos, sin, q, kt: int = 32, br: int = 32,
                         bq: int = 16, p0: int | None = None, num_warps=None):
    """Multi-query `krt`: scores `[HQ, Q, n]` fp32 for `Q` query tokens, from the same store.

    `q` is `[HQ, Q, d]` POST-rotary, the frame `_prefill` already holds. `p0` is the absolute position
    of query 0 and turns on causal masking; leave it None for a fully-connected block (which is what
    a decode-shaped call with Q=1 wants, and it then agrees with `krt_from_slots`).

    `bq` is the QUERY BLOCK and it is the knob that matters: the reconstruction is paid once per
    (row block, head, query block), so `bq` sets how much of it is amortised. `bq*GP` must clear
    `tl.dot`'s minimum of 16.
    """
    n, H, D = slots.shape[0], lay.H, lay.d
    if q.dim() != 3:
        raise ValueError(
            f"the multi-query kernel takes q as [HQ, Q, d]; got {tuple(q.shape)}. The single-query "
            f"form is `krt_from_slots`.")
    HQ, Q = q.shape[0], q.shape[1]
    if HQ % H:
        raise ValueError(f"HQ={HQ} is not a multiple of H={H}")
    G = HQ // H
    GP = triton.next_power_of_2(G)
    if D % 32:
        raise ValueError(f"d={D} must be a multiple of 32 for this two-half lane split")
    if kt & (kt - 1) or br & (br - 1) or kt < 16 or br < 16:
        raise ValueError("kt and br must be powers of two, each >= 16")
    if bq < 1 or (bq * GP) < 16 or (bq & (bq - 1)):
        raise ValueError(
            f"bq={bq} with GP={GP} gives {bq * GP} rows in the contraction, and tl.dot needs a "
            f"power of two >= 16; bq must be a power of two with bq*GP >= 16")

    r1 = decoder_f.shape[0]
    pb = packed_row_bytes(lay.res_k, lay.res_bits)
    s = torch.empty(HQ, Q, n, device=slots.device, dtype=torch.float32)    # `num_stages=1` IS REQUIRED HERE, not a tuning choice. This kernel holds a `[KT, D]` fp32 key
    # tile and a `[BQ*GP, D]` fp32 query tile live at once, and Triton's default software pipelining
    # multiplies every `tl.dot` operand by the stage count -- at the defaults this asks for 160 KB of
    # shared memory against this card's 101 KB and fails at launch with `OutOfResources`. One stage
    # fits, and the loop it would have pipelined is the reconstruction over `r1`, which is short.
    _krt_multi_kernel[(triton.cdiv(n, kt), H, triton.cdiv(Q, bq))](
        slots, slots.view(torch.int8), decoder_f.contiguous(),
        cos.contiguous(), sin.contiguous(), q.contiguous(), s, n, r1, Q,
        0 if p0 is None else int(p0),
        SLOT=lay.k_slot_bytes, ROW_OFF=lay.coeff_bytes + 2, ROW_STEP=D // 8 + pb + 2,
        CSCALE_OFF=lay.coeff_bytes, PB=pb, D=D, HALF=D // 2, BM=D // 8,
        H=H, HQ=HQ, G=G, GP=GP, KT=kt, BR=br, BQ=bq, RES_BITS=lay.res_bits,
        HAS_RES=lay.res_k > 0, CAUSAL=p0 is not None,
        num_stages=1, **({} if num_warps is None else {"num_warps": num_warps}),
    )
    return s


def krt_from_slots(lay, slots, decoder_f, cos, sin, q, kt: int = 32, br: int = 16,
                   num_warps=None, num_stages=None):
    """Scores `[HQ, n]` fp32, UNSCALED -- the softmax scale is the caller's, matching the shipped
    kernel and `reference.raw_scores`.

    lay:        `layout.TTKLayout` for this layer (r1 is per layer, so each layer has its own).
    slots:      `[n, lay.k_slot_bytes]` uint8, as `pack_k_slots` produces.
    decoder_f:  `[r1, H*d]` fp32, ALREADY scaled (`PackedKLayer.decoder_f()`).
    cos, sin:   `[n, d]` fp32, in `rope_apply`'s frame, indexed by ABSOLUTE position.
    q:          `[HQ, d]` fp32, POST-rotary -- the kernel never rotates the query.

    THE TILE DEFAULTS ARE MEASURED, and the measurement is worth recording because two plausible
    guesses about this kernel turned out to be wrong. `tests/bench_krt_tune.py` sweeps it:

    * **SMALLER TILES WIN, which is backwards from the usual amortisation argument.** `kt` sets how
      many rows share one decoder load, so larger `kt` should help; instead n=16384 r1=110 goes
      1.295 ms at `kt=32` to 1.411 at the old `kt=64`, and `kt=256` does not launch at all
      (`OutOfResources: shared memory, Required: 131072, Hardware limit: 101312`). The two `[KT, HALF]`
      fp32 accumulators grow with `kt`, so register and shared pressure dominate the decoder reuse.
    * **The multi-query kernel is SLOWER at Q=1 despite using `tl.dot`.** This kernel contracts with
      broadcast-and-sum because `G=4` is under `tl.dot`'s minimum of 16, building a
      `[G, KT, HALF]` intermediate; the multi-query one pads the group axis and contracts properly.
      At n=2048 the honest comparison is 0.247 ms against 0.418 -- so the intermediate is NOT the
      problem and porting the dot formulation here would be a regression.

    What is left is a kernel at ~15% of this card's fp32 peak, and the reason is structural rather
    than a tile choice: `input_precision="ieee"` rules out tensor cores, and the rotation forces the
    two-half split, so the reconstruction is two dots of N=64 instead of one of N=128.
    """
    n, H, D = slots.shape[0], lay.H, lay.d
    HQ = q.shape[0]
    if HQ % H:
        raise ValueError(f"HQ={HQ} is not a multiple of H={H}")
    G = HQ // H

    # `tl.arange` requires a POWER-OF-TWO extent, so a group size that is not one cannot be handed
    # to the kernel as-is -- and that is not hypothetical: Qwen2.5-0.5B (the model `test_e2e_forward`
    # already drives) is 14 heads over 2 kv heads, i.e. G=7. Padding the query raises the group axis
    # to the next power of two WITHOUT touching the kernel: the extra rows are exactly zero, so they
    # contribute exactly zero scores, and they are sliced off on the way out.
    #
    # The padding must be a real ALLOCATION, not a view. The kernel reads `q` UNMASKED -- the load
    # is `q_p + (hv*G + gq)*D + lane`, with no `gq < G` test, because on the pow2 path every row it
    # names exists. A zero-extent view would leave those reads past the end of the tensor.
    GP = triton.next_power_of_2(G)
    if GP != G:
        qp = torch.zeros(H, GP, D, device=q.device, dtype=torch.float32)
        qp[:, :G] = q.reshape(H, G, D)
        s = _krt_pow2(lay, slots, decoder_f, cos, sin, qp.reshape(H * GP, D), kt, br,
                      num_warps, num_stages)
        return s.view(H, GP, n)[:, :G, :].reshape(HQ, n).contiguous()
    return _krt_pow2(lay, slots, decoder_f, cos, sin, q, kt, br, num_warps, num_stages)


def _krt_pow2(lay, slots, decoder_f, cos, sin, q, kt: int, br: int,
              num_warps=None, num_stages=None):
    """`krt_from_slots` for a power-of-two group size. See that function for the arguments.

    `num_warps`/`num_stages` are for `tests/bench_krt_tune.py` and default to Triton's own choice.
    They are worth having as knobs rather than as constants because this kernel holds TWO `[KT, HALF]`
    fp32 accumulators plus the reconstruction operand live at once, so the right warp count is a
    register-pressure question and not a style one.
    """
    n, H, D = slots.shape[0], lay.H, lay.d
    HQ = q.shape[0]
    G = HQ // H
    r1 = decoder_f.shape[0]
    if D % 32:
        raise ValueError(f"d={D} must be a multiple of 32 for this two-half lane split")
    if kt & (kt - 1) or br & (br - 1) or kt < 16 or br < 16:
        raise ValueError("kt and br must be powers of two, each >= 16")

    opts = {}
    if num_warps is not None:
        opts["num_warps"] = num_warps
    if num_stages is not None:
        opts["num_stages"] = num_stages

    pb = packed_row_bytes(lay.res_k, lay.res_bits)
    s = torch.empty(HQ, n, device=slots.device, dtype=torch.float32)
    _krt_kernel[(triton.cdiv(n, kt), H)](
        slots, slots.view(torch.int8), decoder_f.contiguous(),
        cos.contiguous(), sin.contiguous(), q.contiguous(), s, n, r1,
        SLOT=lay.k_slot_bytes, ROW_OFF=lay.coeff_bytes + 2, ROW_STEP=D // 8 + pb + 2,
        CSCALE_OFF=lay.coeff_bytes, PB=pb, D=D, HALF=D // 2, BM=D // 8,
        H=H, HQ=HQ, G=G, KT=kt, BR=br, RES_BITS=lay.res_bits, HAS_RES=lay.res_k > 0,
        **opts,
    )
    return s
