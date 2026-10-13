"""`v_out` in Triton: the attention OUTPUT straight from a packed V slot.

THE OTHER HALF OF THE PRE-ROTARY PATH. `krt` produces the scores; this consumes the softmax
weights and never materialises V. That is the whole reason V can be stored as magnitude-selected
lane codes at all -- building the dense `[H*n, d]` block was measured at 44.4 ms/token of
`masked_scatter_` in the torch path, against a 22.9 ms/token dense decode, i.e. twice the budget
of the thing it exists to beat.

WHY THERE IS NO `decoder` HERE, unlike `krt`
    K needs a fitted decoder because it is reconstructed from coefficients in a basis. V has no
    basis -- it is per-token magnitude pruning (`build_v_layer`: "no shared basis, because V is
    high-rank per token and a shared basis destroys exactly the per-token distinctness"). So a V
    slot is self-contained: bitmap, codes, one ruler. Nothing to reconstruct against, and nothing
    that can go stale when the decoder is refitted.

    That is also why V is NOT rotated: RoPE applies to keys and queries only.

THE DECODE, which is `PackedRows.materialise` with the dense tensor never built:
    v[j] = (code_j - 2**(bits-1)) * row_scale   for j in the row's bitmap
         = 0                                    otherwise
and then, for each query head in the kv head's group, `out[g, :] = sum_t p[g, t] * v[t, :]`.

THE TWO-HALF SPLIT IS INHERITED FROM `krt`, AND IT IS NOT COSMETIC. The codes are ONE MSB-first
bit stream across the whole row, so a lane's code position depends on how many lanes precede it in
the ENTIRE row -- including lanes in the low half. Splitting D into halves therefore needs the
high half's ranks to continue from the low half's total, which is what `total_lo` below carries.
Getting this wrong shifts codes by a few bits and produces finite, plausible, wrong output.

THE `tl.dot` SHAPE CONSTRAINT, and what is done about it
    `tl.dot` needs every dimension >= 16. With GQA, `G = num_heads // num_kv_heads` is 4 or 8 -- so
    the weight matrix is `[G, KT]` and M would be 4. The kernel therefore pads the weight rows to
    `GP = max(16, next_pow2(G))`, masking the pad to exactly zero, and stores only rows `< G`.
    Padding to zero is what makes this exact rather than approximate: a zero weight contributes a
    zero row, and that row is never stored. The alternative -- a broadcast-and-sum, which is what
    `krt` does for the contraction -- needs a `[GP, KT, HALF]` intermediate, 128 KB at the shipped
    tile sizes, which is a register spill rather than a saving.

ASCII only.
"""
import torch
import triton
import triton.language as tl

from vllm_ttkv.layout import packed_row_bytes

# `tl.dot`'s minimum M. Not a tuning knob.
_DOT_M = 16


@triton.jit(do_not_specialize=["n"])
def _v_out_kernel(
    u8, p_p, o_p, n,
    SLOT: tl.constexpr, ROW_OFF: tl.constexpr, ROW_STEP: tl.constexpr,
    BM: tl.constexpr, PB: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, KT: tl.constexpr, V_BITS: tl.constexpr,
):
    hv = tl.program_id(0)
    HALF_OFF: tl.constexpr = 1 << (V_BITS - 1)
    lo = tl.arange(0, HALF)
    hi = lo + HALF
    gqa = tl.arange(0, GP)
    gm = gqa < G
    acc_lo = tl.zeros((GP, HALF), dtype=tl.float32)
    acc_hi = tl.zeros((GP, HALF), dtype=tl.float32)

    # One program owns a whole READOUT ROW SET, so the reduction over tokens is a loop here rather
    # than a grid dimension -- `krt` could tile the tokens out because scores are per token, but an
    # output row is a sum over all of them.
    for t0 in range(0, tl.cdiv(n, KT) * KT, KT):
        it = t0 + tl.arange(0, KT)
        tm = it < n
        # Token-major, so a row's address is a function of the token and never of the sequence
        # length. No coefficient preamble on a V slot, hence ROW_OFF = 0 at the call site.
        row = it * SLOT + ROW_OFF + hv * ROW_STEP

        # ---- low half: which lanes are present, and where each one's code starts
        bm = tl.load(u8 + row[:, None] + (lo // 8)[None, :], mask=tm[:, None], other=0).to(tl.int32)
        is_lo = ((bm >> (7 - (lo % 8))[None, :]) & 1) != 0
        rank_lo = tl.cumsum(is_lo.to(tl.int32), axis=1) - is_lo.to(tl.int32)
        total_lo = tl.sum(is_lo.to(tl.int32), axis=1)
        byte = (rank_lo * V_BITS) // 8
        sh = (rank_lo * V_BITS) % 8
        a0 = tl.load(u8 + row[:, None] + BM + byte, mask=tm[:, None], other=0).to(tl.int32)
        a1 = tl.load(u8 + row[:, None] + BM + byte + 1,
                     mask=tm[:, None] & (byte + 1 < PB), other=0).to(tl.int32)
        code = (((a0 << 8) | a1) >> (16 - sh - V_BITS)) & ((1 << V_BITS) - 1)
        # The ruler is fp16 at an ODD byte offset for real configs (row stride 50 puts it at 111),
        # so it is assembled from two uint8 loads and bitcast rather than read natively.
        rs = tl.load(u8 + row + BM + PB, mask=tm, other=0).to(tl.int16)
        rs = (rs | (tl.load(u8 + row + BM + PB + 1, mask=tm, other=0).to(tl.int16) << 8)
              ).to(tl.float16, bitcast=True).to(tl.float32)
        v_lo = tl.where(is_lo, (code.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

        # ---- high half: ranks continue from the low half's total (see the module docstring)
        bm2 = tl.load(u8 + row[:, None] + (hi // 8)[None, :], mask=tm[:, None], other=0).to(tl.int32)
        is_hi = ((bm2 >> (7 - (hi % 8))[None, :]) & 1) != 0
        rank_hi = total_lo[:, None] + tl.cumsum(is_hi.to(tl.int32), axis=1) - is_hi.to(tl.int32)
        byte2 = (rank_hi * V_BITS) // 8
        sh2 = (rank_hi * V_BITS) % 8
        b0 = tl.load(u8 + row[:, None] + BM + byte2, mask=tm[:, None], other=0).to(tl.int32)
        b1 = tl.load(u8 + row[:, None] + BM + byte2 + 1,
                     mask=tm[:, None] & (byte2 + 1 < PB), other=0).to(tl.int32)
        code2 = (((b0 << 8) | b1) >> (16 - sh2 - V_BITS)) & ((1 << V_BITS) - 1)
        v_hi = tl.where(is_hi, (code2.to(tl.float32) - HALF_OFF) * rs[:, None], 0.0)

        # ---- weights. `[GP, KT]`, and the padded rows read masked-to-zero rather than a real
        #      row of `p`, which is what keeps the padded output rows exactly zero.
        pg = tl.load(p_p + (hv * G + gqa)[:, None] * n + it[None, :],
                     mask=tm[None, :] & gm[:, None], other=0.0)
        # `input_precision="ieee"` for the same reason as in `krt`: Triton defaults fp32 `tl.dot`
        # to TF32, which carries 10 mantissa bits. The operands are 3-bit codes and softmax
        # weights, so TF32 would not be visible on THIS kernel -- but the convention is one rule,
        # and a rule with an exception is a rule that gets copied to where it matters.
        acc_lo = tl.dot(pg, v_lo, acc_lo, input_precision="ieee")
        acc_hi = tl.dot(pg, v_hi, acc_hi, input_precision="ieee")

    tl.store(o_p + (hv * G + gqa)[:, None] * D + lo[None, :], acc_lo, mask=gm[:, None])
    tl.store(o_p + (hv * G + gqa)[:, None] * D + hi[None, :], acc_hi, mask=gm[:, None])


def v_out_from_slots(lay, v_slots, p, kt: int = 64):
    """Attention output `[HQ, d]` fp32 from a `[n, v_slot_bytes]` V slot and softmax weights.

    lay:      `layout.TTKLayout` for this layer.
    v_slots:  `[n, lay.v_slot_bytes]` uint8, as `layout.pack_v_slots` produces.
    p:        `[HQ, n]` fp32, ALREADY softmaxed and already scaled -- this kernel applies no
              scale, matching `krt` and `reference.attend_raw_store`.

    Not a stub and not a fallback: this is the shipped readout. The dense path
    (`reference.dense_v_from_slots`) exists only as the test oracle.
    """
    n, H, D = v_slots.shape[0], lay.H, lay.d
    HQ = p.shape[0]
    if p.shape[1] != n:
        raise ValueError(f"weights are [HQ, {p.shape[1]}] but the store holds n={n} rows")
    if HQ % H:
        raise ValueError(f"HQ={HQ} is not a multiple of H={H}")
    if D % 32:
        raise ValueError(f"d={D} must be a multiple of 32 for this two-half lane split")
    if kt & (kt - 1) or kt < 16:
        raise ValueError("kt must be a power of two >= 16")
    G = HQ // H
    # See the module docstring: `tl.dot` needs M >= 16, so the group axis is padded.
    GP = max(_DOT_M, triton.next_power_of_2(G))
    if GP & (GP - 1) or GP < _DOT_M:
        raise ValueError(f"internal: padded group size {GP} is not a usable dot dimension")

    pb = packed_row_bytes(lay.v_lanes, lay.v_bits)
    out = torch.empty(HQ, D, device=v_slots.device, dtype=torch.float32)
    _v_out_kernel[(H,)](
        v_slots, p.contiguous(), out, n,
        SLOT=lay.v_slot_bytes, ROW_OFF=0, ROW_STEP=D // 8 + pb + 2,
        BM=D // 8, PB=pb, D=D, HALF=D // 2, G=G, GP=GP, KT=kt, V_BITS=lay.v_bits,
    )
    return out
