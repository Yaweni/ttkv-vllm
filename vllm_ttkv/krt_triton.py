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


def krt_from_slots(lay, slots, decoder_f, cos, sin, q, kt: int = 64, br: int = 32):
    """Scores `[HQ, n]` fp32, UNSCALED -- the softmax scale is the caller's, matching the shipped
    kernel and `reference.raw_scores`.

    lay:        `layout.TTKLayout` for this layer (r1 is per layer, so each layer has its own).
    slots:      `[n, lay.k_slot_bytes]` uint8, as `pack_k_slots` produces.
    decoder_f:  `[r1, H*d]` fp32, ALREADY scaled (`PackedKLayer.decoder_f()`).
    cos, sin:   `[n, d]` fp32, in `rope_apply`'s frame, indexed by ABSOLUTE position.
    q:          `[HQ, d]` fp32, POST-rotary -- the kernel never rotates the query.
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
        s = _krt_pow2(lay, slots, decoder_f, cos, sin, qp.reshape(H * GP, D), kt, br)
        return s.view(H, GP, n)[:, :G, :].reshape(HQ, n).contiguous()
    return _krt_pow2(lay, slots, decoder_f, cos, sin, q, kt, br)


def _krt_pow2(lay, slots, decoder_f, cos, sin, q, kt: int, br: int):
    """`krt_from_slots` for a power-of-two group size. See that function for the arguments."""
    n, H, D = slots.shape[0], lay.H, lay.d
    HQ = q.shape[0]
    G = HQ // H
    r1 = decoder_f.shape[0]
    if D % 32:
        raise ValueError(f"d={D} must be a multiple of 32 for this two-half lane split")
    if kt & (kt - 1) or br & (br - 1) or kt < 16 or br < 16:
        raise ValueError("kt and br must be powers of two, each >= 16")

    pb = packed_row_bytes(lay.res_k, lay.res_bits)
    s = torch.empty(HQ, n, device=slots.device, dtype=torch.float32)
    _krt_kernel[(triton.cdiv(n, kt), H)](
        slots, slots.view(torch.int8), decoder_f.contiguous(),
        cos.contiguous(), sin.contiguous(), q.contiguous(), s, n, r1,
        SLOT=lay.k_slot_bytes, ROW_OFF=lay.coeff_bytes + 2, ROW_STEP=D // 8 + pb + 2,
        CSCALE_OFF=lay.coeff_bytes, PB=pb, D=D, HALF=D // 2, BM=D // 8,
        H=H, HQ=HQ, G=G, KT=kt, BR=br, RES_BITS=lay.res_bits, HAS_RES=lay.res_k > 0,
    )
    return s
