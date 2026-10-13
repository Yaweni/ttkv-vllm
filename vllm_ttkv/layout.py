"""What one vLLM cache slot holds, and why the head axis is folded into it.

vLLM's paged cache is a homogeneous page per layer,
`[num_blocks, block_size, num_head_slots, state_content_bytes]`, where the last two are
published by the backend through `AttentionSpec` (`vllm/v1/kv_cache_interface.py`).

TT-KV's K store is NOT homogeneous per head. The tensor-train coefficients are per TOKEN and
shared by every KV head, while the residual and V rows are per (head, token). The obvious
encoding -- one slot per head -- would replicate the coefficient block H times, and at
r1 ~ 128..600 coefficients are most of the K bytes. So the slot is made PER TOKEN and the head
axis is folded into its content:

    num_head_slots = 1
    state_content_bytes = K_SLOT_BYTES          (key cache)
                          V_SLOT_BYTES          (value cache)

and one token's K slot is `[r1 coefficients][coefficient ruler][H residual rows]`. The decoder
is per layer and CONSTANT, so it lives as a module buffer beside the attention weights -- the
way MLA keeps W_UK -- and never enters the cache at all.

Two open questions from the port plan dissolve under this encoding, which is the point of it:

  * "does the paged block size divide the TT row size" -- a slot is per TOKEN, so `block_size`
    counts tokens and the K page is exactly `block_size * K_SLOT_BYTES`. There is no row to
    divide, and no per-row padding to discover late.
  * "does the coefficient table need a second cache region" -- no. A true sibling region needs
    `KVCacheGroupSpec` / `is_index_group_leader` (the DSA indexer pattern,
    `vllm/v1/attention/backends/mla/index_group.py`) or a multi-shape spec (`MambaSpec`).
    Folding the head axis makes the coefficient table part of the slot instead, so one
    `AttentionSpec` describes the whole layer.

This is the only part of the plugin that is verifiable without a vLLM install, so it is also
the part that is actually tested: `tests/test_layout.py` round-trips it against the container's
own `PackedRows`, i.e. against the single definition of the codec rather than a copy of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def align_up(n: int, a: int) -> int:
    """Round `n` up to a multiple of `a`. TurboQuant's spec field is literally named
    `slot_size_aligned`, so vLLM expects a backend to publish an aligned slot width."""
    return n if a <= 1 else ((n + a - 1) // a) * a


def packed_row_bytes(k: int, bits: int) -> int:
    """Bytes one packed row occupies. `pack_bits_rows` pads each ROW to whole bytes -- that
    padding is what makes an append a `cat` -- so the width is ceil(k*bits/8), not k*bits/8."""
    return math.ceil(k * bits / 8)


@dataclass(frozen=True)
class TTKLayout:
    """Byte layout of one layer's TT-KV cache. Every size below is PER TOKEN.

    `r1` is per LAYER, not global: measured rank runs 93..616 across layers at a fixed
    tolerance, so a real integration carries one of these per layer and the layer's slot width
    is its own. (Shipped configs: `campaign4x` 64 lanes/side @4b residual 0.50; `safe` 81
    lanes/side @3b residual 0.633; `tight` K 32 / V 45 lanes @3b residual 0.25.)
    """

    H: int            # kv heads (8 for Llama-3.1-8B)
    d: int            # head dim (128 for Llama-3.1-8B; 64 / 96 / 256 also occur)
    r1: int           # TT rank of the first unfolding
    qbits: int        # coefficient width; 8 shipped
    res_k: int        # K residual lanes per head
    res_bits: int     # K residual code width
    v_lanes: int      # V lanes per head
    v_bits: int       # V code width
    align: int = 16   # pad each slot to this many bytes

    def __post_init__(self):
        if self.d % 8:
            raise ValueError(
                f"head dim {self.d} is not a multiple of 8; the lane bitmap is d/8 bytes and "
                f"the container is byte-aligned per row, so d must be a whole number of bytes")
        # The two lane knobs are NOT the same kind of quantity, and this is where that bites.
        # `res_k` comes from `res_frac` -- a FRACTION of d -- so it follows the geometry on its
        # own. `v_lanes` (`k_lanes` in the codec) is an ABSOLUTE count, tuned at d=128, so it
        # does not. A config carried to a narrower head keeps more lanes than the head has;
        # a config carried to a wider head silently becomes more aggressive (keeping 81 of 256
        # is a third of the row, not two thirds) and its ratio flatters in the process.
        if self.res_k > self.d:
            raise ValueError(
                f"res_k={self.res_k} exceeds d={self.d}. res_k is derived from res_frac, so "
                f"this should be impossible -- check the config was built by layout_for_config")
        if self.v_lanes > self.d:
            raise ValueError(
                f"v_lanes={self.v_lanes} exceeds d={self.d}. `k_lanes` is an ABSOLUTE lane "
                f"count, not a fraction: a config tuned at d=128 does not transfer to a "
                f"narrower head and has to be re-derived for this geometry")

    # ------------------------------------------------------------------ component sizes
    @property
    def coeff_bytes(self) -> int:
        """The whole point of the format: r1 bytes per token, SHARED by all H heads."""
        if (self.r1 * self.qbits) % 8:
            raise ValueError(
                f"r1={self.r1} at qbits={self.qbits} is not a whole number of bytes; the "
                f"coefficient row must be byte-aligned to be a contiguous slot field")
        return self.r1 * self.qbits // 8

    @property
    def k_row_bytes(self) -> int:
        """One KV head's K-residual row: bitmap + packed lanes + one fp16 ruler."""
        return self.d // 8 + packed_row_bytes(self.res_k, self.res_bits) + 2

    @property
    def v_row_bytes(self) -> int:
        return self.d // 8 + packed_row_bytes(self.v_lanes, self.v_bits) + 2

    # ------------------------------------------------------------------ slot sizes
    @property
    def coeff_scale_bytes(self) -> int:
        return 2                       # one fp16 ruler per token, not per head

    @property
    def v_offset(self) -> int:
        """Byte offset of the V slot inside a token's blob.

        K and V live in ONE cache region, concatenated, because that is what vLLM's
        `state_content_bytes` describes: `state_content_size_bytes` is the whole per-(head slot,
        state) cell, and it is `(head_size + head_size_v) * dtype_size` when a backend does not
        publish one (`kv_cache_interface.py`). The dense default is already K followed by V, so
        concatenating is the shape vLLM assumes rather than a trick.

        Why it matters that this is DERIVED and not written twice: the two slots have different
        widths (`k_slot_bytes != v_slot_bytes`), so a reader that assumes one width silently reads
        V's bytes as K's, or writes V over the next token's coefficients.
        """
        return self.k_slot_bytes

    @property
    def k_slot_bytes(self) -> int:
        """ONE TOKEN's K slot: coefficients once, then H residual rows."""
        return align_up(self.coeff_bytes + self.coeff_scale_bytes
                        + self.H * self.k_row_bytes, self.align)

    @property
    def v_slot_bytes(self) -> int:
        return align_up(self.H * self.v_row_bytes, self.align)

    # ------------------------------------------------------------------ accounting
    def dense_bytes_per_token(self) -> int:
        """fp16 K+V for one token: H heads x d x 2 B x 2 sides."""
        return 2 * self.H * self.d * 2

    def compressed_bytes_per_token(self) -> int:
        """Both slots. Aligned, so this is what the cache actually reserves."""
        return self.k_slot_bytes + self.v_slot_bytes

    def ratio(self) -> float:
        return self.dense_bytes_per_token() / self.compressed_bytes_per_token()


# ---------------------------------------------------------------- slot packing
def pack_k_slots(lay: TTKLayout, coeff_q: torch.Tensor, coeff_scale: torch.Tensor,
                 resid) -> torch.Tensor:
    """`[n, k_slot_bytes]` uint8, one slot per token, from a layer's K store.

    coeff_q: [n, r1] int8   coeff_scale: [n] fp16   resid: PackedRows over H*n rows

    The store's row order is (head, token) = h*n + i and that is NOT changed here -- this
    function only decides where those bytes live. The codec's lane-order invariant stays the
    codec's business, which is what keeps this a rearrangement rather than a second
    implementation of the format.
    """
    n = coeff_q.shape[0]
    m = lay.H * n
    if resid.rows != m:
        raise ValueError(f"residual holds {resid.rows} rows, expected H*n = {m}")
    out = torch.zeros(n, lay.k_slot_bytes, dtype=torch.uint8, device=coeff_q.device)

    off = 0
    out[:, off:off + lay.coeff_bytes] = coeff_q.contiguous().view(torch.uint8)
    off += lay.coeff_bytes
    out[:, off:off + 2] = coeff_scale.contiguous().view(torch.uint8).reshape(n, 2)
    off += 2
    for h in range(lay.H):
        rows = slice(h * n, (h + 1) * n)
        b = lay.d // 8
        out[:, off:off + b] = resid.bitmap[rows]
        off += b
        p = resid.packed.shape[1]
        out[:, off:off + p] = resid.packed[rows]
        off += p
        out[:, off:off + 2] = resid.scale[rows].contiguous().view(torch.uint8).reshape(n, 2)
        off += 2
    return out


def unpack_k_slots(lay: TTKLayout, slots: torch.Tensor):
    """Inverse of `pack_k_slots`. Returns `(coeff_q, coeff_scale, (bitmap, packed, scale))`."""
    n = slots.shape[0]
    off = 0
    coeff_q = slots[:, off:off + lay.coeff_bytes].contiguous().view(torch.int8)
    off += lay.coeff_bytes
    coeff_scale = slots[:, off:off + 2].contiguous().view(torch.float16).reshape(n)
    off += 2
    m = lay.H * n
    bitmap = torch.empty(m, lay.d // 8, dtype=torch.uint8, device=slots.device)
    packed = torch.empty(m, packed_row_bytes(lay.res_k, lay.res_bits), dtype=torch.uint8,
                         device=slots.device)
    scale = torch.empty(m, dtype=torch.float16, device=slots.device)
    for h in range(lay.H):
        rows = slice(h * n, (h + 1) * n)
        b = lay.d // 8
        bitmap[rows] = slots[:, off:off + b]
        off += b
        p = packed.shape[1]
        packed[rows] = slots[:, off:off + p]
        off += p
        scale[rows] = slots[:, off:off + 2].contiguous().view(torch.float16).reshape(n)
        off += 2
    return coeff_q, coeff_scale, (bitmap, packed, scale)


def pack_v_slots(lay: TTKLayout, rows) -> torch.Tensor:
    """`[n, v_slot_bytes]` uint8 from a V store's PackedRows over H*n rows."""
    n = rows.rows // lay.H
    if rows.rows % lay.H:
        raise ValueError(f"V store holds {rows.rows} rows, not a multiple of H={lay.H}")
    out = torch.zeros(n, lay.v_slot_bytes, dtype=torch.uint8, device=rows.bitmap.device)
    off = 0
    for h in range(lay.H):
        sl = slice(h * n, (h + 1) * n)
        b = lay.d // 8
        out[:, off:off + b] = rows.bitmap[sl]
        off += b
        p = rows.packed.shape[1]
        out[:, off:off + p] = rows.packed[sl]
        off += p
        out[:, off:off + 2] = rows.scale[sl].contiguous().view(torch.uint8).reshape(n, 2)
        off += 2
    return out


def unpack_v_slots(lay: TTKLayout, slots: torch.Tensor):
    """Inverse of `pack_v_slots`. Returns `(bitmap, packed, scale)` over H*n rows."""
    n = slots.shape[0]
    m = lay.H * n
    bitmap = torch.empty(m, lay.d // 8, dtype=torch.uint8, device=slots.device)
    packed = torch.empty(m, packed_row_bytes(lay.v_lanes, lay.v_bits), dtype=torch.uint8,
                         device=slots.device)
    scale = torch.empty(m, dtype=torch.float16, device=slots.device)
    off = 0
    for h in range(lay.H):
        sl = slice(h * n, (h + 1) * n)
        b = lay.d // 8
        bitmap[sl] = slots[:, off:off + b]
        off += b
        p = packed.shape[1]
        packed[sl] = slots[:, off:off + p]
        off += p
        scale[sl] = slots[:, off:off + 2].contiguous().view(torch.float16).reshape(n)
        off += 2
    return bitmap, packed, scale


def layout_for_config(cfg: dict, r1: int, H: int = 8, d: int = 128) -> TTKLayout:
    """Build a layout from a `ttkv_codec.CONFIGS` entry plus the layer's measured rank.

    `r1` is passed in rather than derived: it is a property of the layer's key block at a given
    tolerance and context length, not of the config. The `campaign` field on the config is a
    LABEL from the pre-rotary hook ledger and must not be used to predict this.
    """
    return TTKLayout(
        H=H, d=d, r1=r1, qbits=cfg["qbits"],
        res_k=max(int(d * cfg["res_frac"]), 1), res_bits=cfg["res_bits"],
        v_lanes=cfg["k_lanes"], v_bits=cfg["vbits"],
    )
