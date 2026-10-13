"""The lane SELECT and the bit PACK, fused, written straight into the cache slots.

WHAT THIS REPLACES, AND WHY IT IS ONE KERNEL PER STORE
    The torch path builds a row's three fields (`bitmap`, `packed`, `scale`) with about twenty small
    ops -- `abs`, `topk`, `scatter_`, `amax`, two `where`s, a `round`, a `clamp`, the bitmap's
    reshape/multiply/sum, the code stream's bit planes and Horner fold, the fp16 cast -- and then a
    Python loop over heads that slice-assigns those three fields into a token-major slot buffer.
    Measured on the real run: `proj.resid` 0.728 ms + `pack.k` 0.314 ms per layer for K, and
    `pack.v` 0.944 ms, i.e. **1.99 ms per layer**, 56 ms of a 113 ms decode step, spent in kernels
    whose actual arithmetic is a few hundred operations each. What costs the time is that every one
    of those ops is a separate launch depending on the previous one, so none of them overlap.

    Here one program owns one ROW and does the whole thing: load, select, quantise, bitmap, pack,
    address, store -- and the fields land in the paged cache directly instead of in a slot buffer
    that the next op copies them out of. A store becomes one launch.

    WHAT IT BOUGHT, measured with `tests/bench_write_path.py` at engine shapes with the GPU queue held
    deep: **1.794 -> 0.307 ms a layer-call, 5.84x**, and on a 1082-token chunk 2.196 -> 0.710, 3.09x.
    End to end on the 1082-token / 16-token run the decode step went 94 ms to 79 ms and the run 5.50 s
    to 5.32 s with the generated ids unchanged. Note the units: those per-layer numbers are the write's
    OWN cost, and `VLLM_TTKV_PROFILE` attributes about four times more to the same call, because on a
    deep step what a phase measures is the queue draining rather than the work being submitted.

THE EXACTNESS CONTRACT
    Byte-identical to `ttkv_packed_cache.PackedRows.build` + `layout.pack_{k,v}_slots`, asserted with
    `torch.equal` on a whole slot tensor by `tests/test_encode_triton.py`. Every convention that
    makes that true is named where it is used, because each is a silent-failure site rather than a
    crash: the bitmap is MSB-first, the code stream is MSB-first and ONE bit stream across the whole
    row, the codes are unsigned (offset by `2**(bits-1)`), the ruler is the row's own max over the
    KEPT lanes (the row's largest magnitude, not the selection threshold) and is
    `1 / (2**(bits-1) - 1)` when that max is zero, and a head's three fields are back to back at
    `d/8 + ceil(k*bits/8) + 2` bytes.

THE SELECTION IS EXACT, AND THE TIE IS THE ONE PLACE IT IS A CHOICE
    Top-k by magnitude, found here by bit-by-bit construction on the fp32 magnitude bits: a
    non-negative float's bit pattern is monotone in its value, so the k-th largest is one bit at a
    time -- try the next lower bit, keep it while at least k lanes are still covered. That is O(31)
    reductions rather than a sort, and it is EXACT: with distinct magnitudes the k-th largest value
    is unique, so this picks the same SET as `torch.topk` and the bytes match.

    Two kept lanes with the SAME magnitude are broken by the lowest lane index, and `torch.topk`
    makes no such promise -- it does not document a tie order at all. On a tie-heavy row the two can
    then select different lanes, which moves a bit in the bitmap but cannot change the
    reconstruction, because the two candidates have equal magnitude and therefore the same code.
    Both cases are tested: `torch.equal` on tie-free data, and a reconstruction comparison on rows
    that are all zeros.

THE COMPACTION, AND WHY THERE IS A SCRATCH
    The code stream's order is the row's LANE order -- rank 0 is the lowest lane index kept -- but
    the lanes holding the codes are scattered across the `[d]` block, so a lane's bit position comes
    from a prefix sum over the selection rather than from the lane index it already has. Writing each
    lane's bits straight at its stream position would need an atomic add per lane; instead each
    program STORES its `k` code bytes to a per-row scratch at `rank`, then, after a block barrier,
    LOADS them back by stream position and folds eight bits into each output byte. Both halves are in
    one program over one buffer, so nothing is shared between programs and no atomic is needed.

    The barrier is `tl.debug_barrier()` (`bar.sync`), which the CUDA model documents as making global
    writes by a block's threads visible to that block. It is not decoration: without it the pack
    phase can read a code byte before it is written, producing finite, plausible, wrong codes.

    The scratch is `torch.empty` and reused across layers rather than zeroed, because every byte a
    program reads it has just written -- the ranks of a row's kept lanes are exactly `0..k-1`, so the
    `k` bytes it stores are the `k` bytes it loads, and a load past `k` is masked off. The cache entry
    is keyed on `(device, rows, lanes)` and shared by every layer of a step, which is safe because
    kernels on one stream are serialized; two concurrent streams writing the same rows would not be,
    and that is the assumption to revisit if this ever runs under a graph or a second stream.

THE ROUNDING MODES ARE PART OF THE FORMAT
    `v/sc` is a real IEEE division in the torch path (a `[rows, d]` tensor over a `[rows, 1]` tensor,
    correctly rounded), so the kernel uses `div_rn`. The rounding is `rint` (half to EVEN), which is
    what `torch.round` does. Triton's `/` is not correctly rounded and its `round` breaks ties away
    from zero, so either would move a code by one step whenever a value sits within an ulp of a
    half-integer -- an error of one quantisation step, i.e. the kind that shows up as an accuracy
    delta instead of a failure.

    `sc` is NOT an IEEE division, and that is the counter-intuitive half. The codec divides by a
    PYTHON scalar (`/ (2**(bits-1) - 1)`), and torch's scalar division is a reciprocal MULTIPLY:
    measured over 2**20 random fp32 pairs, `tensor / 3` differed from the correctly-rounded quotient
    in 349427 of them and matched `tensor * float32(1/3)` in all of them. So `sc` is one ulp away
    from the honest quotient roughly a third of the time, and for coarse (bf16) activations that is
    often enough to move a code across a half-integer -- which is how this was found: the fp32
    fixtures matched byte for byte and the bf16 ones did not. The kernel therefore multiplies by
    `_INV_RULER[bits]`, the fp32 reciprocal, because byte-identical to the shipped store is the
    contract and the shipped store is what the decoder was measured against.

WHEN IT REFUSES
    `supported()` is the whole list, and each entry is a case where the arithmetic above would be
    wrong rather than slow: a head dim that is not a power of two (the block and the bitmap fold are
    sized by it), a non-float input (the reference quantises the incoming dtype), a lane count above
    `d`, a bit width outside 2..8 (1 has a zero divisor -- the codec's own ruler is `inf` there). The
    caller falls back to the torch path, which is still the definition of the format.

ASCII only.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:  # the libdevice module moved between Triton 3.2 and 3.7; both of these exist in both
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra import libdevice

from vllm_ttkv import prof
from vllm_ttkv.layout import packed_row_bytes

__all__ = ["ENV", "supported", "write_k_residual", "write_v_rows"]

ENV = "VLLM_TTKV_TRITON_PACK"

# The codec's ruler divisor, as the fp32 RECIPROCAL that torch's scalar division really applies --
# see "THE ROUNDING MODES ARE PART OF THE FORMAT" above for the measurement behind this. `bits = 1`
# is absent and not supported anywhere: its divisor is `2**0 - 1 = 0`, so the codec's own ruler is
# `inf` there, and matching that is not a thing worth doing.
_INV_RULER = {b: float(torch.tensor(1.0 / (2 ** (b - 1) - 1), dtype=torch.float32))
              for b in range(2, 9)}

# One scratch entry per (device, rows, lanes); cleared when it grows past this, so a workload that
# changes shape every step cannot keep an unbounded number of them alive.
_SCRATCH: dict[tuple, torch.Tensor] = {}
_SCRATCH_KEEP = 8


def _scratch(device, rows: int, lanes: int) -> torch.Tensor:
    key = (device.index, rows, lanes)
    buf = _SCRATCH.get(key)
    if buf is None:
        if len(_SCRATCH) >= _SCRATCH_KEEP:
            _SCRATCH.clear()
        buf = torch.empty(rows * lanes, dtype=torch.uint8, device=device)
        _SCRATCH[key] = buf
    return buf


def supported(x: torch.Tensor, d: int, lanes: int, bits: int) -> bool:
    """Whether `write_rows` can produce this geometry's bytes exactly.

    Deliberately silent: it is called on the hot path, and every False here has a working fallback,
    so a refusal is a performance decision rather than an error. `write_rows` itself refuses loudly.
    """
    if os.environ.get(ENV, "1") == "0":
        return False
    if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if x.dim() != 2 or x.shape[1] != d:
        return False
    if d < 8 or (d & (d - 1)) or d % 8:
        return False
    return 2 <= bits <= 8 and 1 <= lanes <= d


@triton.jit(do_not_specialize=["n_tok", "flat_stride", "field_off", "lanes"])
def _select_pack_kernel(
    x_p, flat_p, slot_p, codes_p,
    n_tok, flat_stride, field_off, lanes,
    D: tl.constexpr, ROW_STRIDE: tl.constexpr, BITS: tl.constexpr,
    PAD_P: tl.constexpr, P_BYTES: tl.constexpr, INV_RULER: tl.constexpr,
):
    """One program per row: `[rows, D]` magnitudes -> `D/8 + P_BYTES + 2` bytes of a cache slot.

    Row `r` is `h * n_tok + i` (the codec's (head, token) row order) and lands at token `i`'s slot,
    head `h`'s field group -- the same relocation `layout.pack_k_slots` performs with a Python loop
    over heads, done here by arithmetic on the program id.
    """
    r = tl.program_id(0).to(tl.int64)
    tok = r % n_tok
    head = r // n_tok
    # int64 throughout the addressing: the cache is gigabytes, so a 32-bit byte offset overflows.
    dst = tl.load(slot_p + tok).to(tl.int64) * flat_stride + field_off + head * ROW_STRIDE

    lane = tl.arange(0, D)
    x = tl.load(x_p + r * D + lane).to(tl.float32)
    # |x| of a non-negative float is monotone in its bit pattern, which is what makes the search
    # below a search on values. NaN has no ordering here; the torch path's amax would propagate one.
    u = tl.abs(x).to(tl.int32, bitcast=True)

    # The k-th largest magnitude, one bit at a time: bits 30..0 (31 is the sign and |x| clears it).
    t = tl.zeros([1], dtype=tl.int32)
    for i in tl.static_range(31):
        cand = t | (1 << (30 - i))
        cnt = tl.sum(tl.where(u >= cand, 1, 0))
        t = tl.where(cnt >= lanes, cand, t)

    # Exactly `lanes` lanes: everything above the threshold, then the lowest-index ties until full.
    n_above = tl.sum(tl.where(u > t, 1, 0))
    tie = u == t
    seen = tl.cumsum(tl.where(tie, 1, 0), axis=0)
    sel = (u > t) | (tie & (seen <= lanes - n_above))
    rank = tl.cumsum(tl.where(sel, 1, 0), axis=0) - 1

    # One ruler per row, from the row's own max over the kept lanes. It is NOT `t`: `t` is the
    # k-th largest, i.e. the SMALLEST of the kept, and the row's max is the largest -- which is
    # always kept when k >= 1 (the layout's `res_k` is at least 1, and `supported` refuses 0), so
    # the max over every lane is the max over the kept ones. The reciprocal is the codec's, not an
    # optimised form of it: see `_INV_RULER`.
    rmax = tl.max(u).to(tl.float32, bitcast=True)
    sc = tl.where(rmax == 0.0, 1.0, rmax) * INV_RULER
    q = libdevice.rint(tl.div_rn(tl.where(sel, x, 0.0), sc))
    q = tl.minimum(tl.maximum(q, -(2 ** (BITS - 1))), 2 ** (BITS - 1) - 1) + 2 ** (BITS - 1)

    # Compaction: this row's k code bytes, in lane order, at their rank in the row's own scratch.
    tl.store(codes_p + r * lanes + rank, q.to(tl.uint8), mask=sel)
    tl.debug_barrier()

    # Bitmap: MSB-first per byte, lane 8*j+b at bit 7-b of byte j.
    bits = tl.where(sel, 1, 0).to(tl.uint8).reshape(D // 8, 8)
    w = (1 << (7 - tl.arange(0, 8))).to(tl.uint8)
    bm = tl.sum((bits * w[None, :]).to(tl.int32), axis=1)
    tl.store(flat_p + dst + tl.arange(0, D // 8), bm.to(tl.uint8))

    # Codes: MSB-first, one bit stream across the row, each ROW padded to whole bytes. Stream bit p
    # is bit `bits-1-(p % bits)` of the code of rank `p // bits`, so a byte is a gather of eight
    # such bits -- which is exactly `pack_bits_rows`, with the padding reading as zeros.
    p8 = tl.arange(0, 8)
    pb = tl.arange(0, PAD_P)
    pos = pb[:, None] * 8 + p8[None, :]
    j = pos // BITS
    code = tl.load(codes_p + r * lanes + j, mask=j < lanes, other=0).to(tl.int32)
    bit = (code >> (BITS - 1 - (pos % BITS))) & 1
    byte = tl.sum(bit << (7 - p8[None, :]), axis=1)
    tl.store(flat_p + dst + D // 8 + pb, byte.to(tl.uint8), mask=pb < P_BYTES)

    # The ruler, as the two bytes of its fp16 bit pattern in little-endian order -- the same two
    # bytes `scale.view(torch.uint8).reshape(rows, 2)` writes.
    h16 = sc.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
    two = tl.arange(0, 2)
    rb = tl.where(two == 0, h16 & 0xFF, (h16 >> 8) & 0xFF)
    tl.store(flat_p + dst + D // 8 + P_BYTES + two, rb.to(tl.uint8))


def write_rows(flat, slots, x, *, field_off: int, row_stride: int, lanes: int, bits: int,
               heads: int, d: int) -> None:
    """Pack `x`'s rows into the cache's byte view as `heads` field groups per token.

    `flat` is `[n_slots, width]` uint8 with the layout's tokens as rows (the shape
    `decode.byte_rows` returns), `slots` the `n_tok` slot ids of this step, and `x` the `[H*n_tok, d]`
    values in (head, token) row order. The three fields a head owns are written at
    `field_off + head*row_stride`, at `+ d/8`, and at `+ d/8 + packed_row_bytes(lanes, bits)`.
    """
    if flat.dim() != 2 or flat.dtype != torch.uint8 or not flat.is_contiguous():
        raise ValueError(
            f"the cache's byte view is {tuple(flat.shape)} {flat.dtype} and has to be a contiguous "
            f"2-D uint8 tensor; anything else cannot be addressed as bytes by the kernel.")
    if x.dim() != 2 or not x.is_contiguous():
        raise ValueError(
            f"the values are {tuple(x.shape)} and have to be a contiguous 2-D tensor, because the "
            f"kernel addresses row r as `r*D` and a copy would silently be written instead of it.")
    n_tok = int(slots.numel())
    if heads < 1 or x.shape[0] != n_tok * heads:
        raise ValueError(
            f"{x.shape[0]} rows for {n_tok} tokens and {heads} heads: the rows are (head, token), so "
            f"there is exactly one per head per token. A mismatch would write a head's rows into "
            f"another token's slot.")
    if x.shape[1] != d or d % 8 or d < 8 or (d & (d - 1)):
        raise ValueError(
            f"head dim {x.shape[1]} against a layout d={d}: the kernel's block and the bitmap fold "
            f"are sized from d, which has to be a power of two and a whole number of bytes.")
    # THE SPLIT THAT DECIDES HOW TO MAKE THIS CHEAPER. Everything above is host work: the checks, the
    # arithmetic, and the scratch lookup. There is one launch below, and a bare Triton launch is
    # ~13 us. So if `pack.rows.prep` is most of `pack.rows.launch`, the fix is to do this prep ONCE
    # for a batch (or to cache it per layer) rather than to look for arithmetic to remove -- and if
    # the launch dominates, the fix is fewer launches. `tests/bench_write_path.py` reports both.
    with prof.phase("pack.rows.prep"):
        if not supported(x, d, lanes, bits):
            raise ValueError(
                f"lanes={lanes}, bits={bits}, dtype={x.dtype} is outside what this kernel reproduces; "
                f"`supported()` is the list, and the caller is expected to have consulted it.")
        p_bytes = packed_row_bytes(lanes, bits)
        need = field_off + (heads - 1) * row_stride + d // 8 + p_bytes + 2
        if need > flat.shape[1]:
            raise ValueError(
                f"a head's field group needs {need} bytes of a {flat.shape[1]}-byte slot row "
                f"(field_off={field_off}, row_stride={row_stride}, heads={heads}). The group is sized "
                f"from the layout, so a shortfall means the cache was allocated for another one -- and "
                f"the write would land in the next token's row.")
        scratch = _scratch(x.device, x.shape[0], lanes)
        pad_p = max(1, triton.next_power_of_2(p_bytes))
        inv_ruler = _INV_RULER[bits]

    with prof.phase("pack.rows.launch"):
        _select_pack_kernel[(x.shape[0],)](
            x, flat, slots, scratch,
            n_tok, flat.shape[1], field_off, lanes,
            D=d, ROW_STRIDE=row_stride, BITS=bits,
            PAD_P=pad_p, P_BYTES=p_bytes, INV_RULER=inv_ruler,
        )


def write_k_residual(flat, slots, R, lay) -> None:
    """The K residual rows of one store, into the cache. `R` is `[H*n, d]` fp32, (head, token)."""
    write_rows(flat, slots, R, field_off=lay.coeff_bytes + lay.coeff_scale_bytes,
               row_stride=lay.k_row_bytes, lanes=lay.res_k, bits=lay.res_bits,
               heads=lay.H, d=lay.d)


def write_v_rows(flat, slots, M, lay) -> None:
    """The V rows of one store, into the cache. `M` is `[H*n, d]`, (head, token)."""
    write_rows(flat, slots, M, field_off=lay.v_offset, row_stride=lay.v_row_bytes,
               lanes=lay.v_lanes, bits=lay.v_bits, heads=lay.H, d=lay.d)
