"""The vector-layout conventions the pre-rotary read path depends on, checked without a GPU.

These are the quietest failures in the whole port. RoPE's pairing convention and a cache's
half/straddle layout are both pure convention: getting either wrong still yields an ORTHOGONAL,
position-dependent transform, so every norm, energy and shape check still passes and the stored
tensor is silently mis-rotated. The codec's own `_rotate_half` docstring records the same trap.

So each function here asserts against a computation written out longhand, rather than against
another function that shares the assumption. Needs only torch and the codec, so it runs everywhere.
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from vllm_ttkv import decode as dc  # noqa: E402
from vllm_ttkv.layout import TTKLayout, packed_row_bytes  # noqa: E402

D = 128


def _vllm_cache(max_len: int, d: int = D, theta: float = 10000.0):
    """vLLM's `cos_sin_cache`: `[max_pos, d]`, cos in the first half and sin in the second."""
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def test_cos_sin_from_cache_reproduces_vllms_half_layout_rotation():
    """`krt` pairs `(j, j + d/2)`; vLLM pairs the two halves of the row. `cos_sin_from_cache` is the
    translation between the two spellings, so this asserts that the table it returns rotates a
    vector exactly the way vLLM's own layout says it should.

    A single wrong sign or a half-swap here mis-rotates half of every stored row, and the result is
    still orthogonal and still position dependent -- nothing downstream can notice.
    """
    n = 24
    cache = _vllm_cache(n)
    pos = torch.arange(n)
    cos, sin = dc.cos_sin_from_cache(cache, pos, D)

    torch.manual_seed(0)
    x = torch.randn(n, D)

    # Longhand, straight from the cache's own halves, in vLLM's rotate_half form.
    c, s = cache[:, :D // 2], cache[:, D // 2:]
    x1, x2 = x[:, :D // 2], x[:, D // 2:]
    want = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    got = pc.rope_apply(x, cos, sin)          # the codec's RoPE, on the translated table
    rel = ((got - want).abs().max() / want.abs().max()).item()
    print(f"  codec rope on the translated table vs vLLM's half layout: rel err {rel:.3e}")
    assert rel < 1e-6, (
        f"the translated angle table does not reproduce the model's rotation (rel err {rel:.3e}). "
        f"vLLM's rope op and `krt` spell the same rotation differently, and this translation is the "
        f"only place the two are reconciled.")

    # And the DROPPED-duplication mistake must be detectable, or the check above proves nothing.
    # Passing the cache's own halves straight through -- cos and sin both `cat([c, s])` -- is that
    # mistake, and it has to move the result a lot.
    mis = pc.rope_apply(x, cache, cache)
    spread = ((mis - want).abs().max() / want.abs().max()).item()
    assert spread > 0.1, (
        f"a deliberately mis-built table moved the result by only {spread:.3e}, so this fixture "
        f"does not distinguish a correct table from an incorrect one")


def test_cos_sin_from_cache_indexes_by_absolute_position():
    """The store holds position-free keys, so the position is supplied at READ time. Indexing the
    table by anything other than the row's own absolute position -- a relative offset, or the row's
    index within a chunk -- produces a rotation that is still orthogonal and still varies by
    position, i.e. exactly as plausible and wrong."""
    cache = _vllm_cache(64)
    cos_a, sin_a = dc.cos_sin_from_cache(cache, torch.tensor([5]), D)
    cos_b, sin_b = dc.cos_sin_from_cache(cache, torch.tensor([5]), D)
    cos_c, _ = dc.cos_sin_from_cache(cache, torch.tensor([6]), D)
    assert torch.equal(cos_a, cos_b), "the lookup is not deterministic in the position"
    assert not torch.allclose(cos_a, cos_c), (
        "positions 5 and 6 produced the same angle, so the table is not being indexed by position")
    # And it must be the ROW of the cache the position names, not a row that happens to differ.
    assert torch.allclose(cos_a[0, :D // 2], cache[5, :D // 2], atol=1e-6)
    print("  lookup is by absolute position and matches the cache row")


def test_cos_sin_from_cache_refuses_partial_rotary():
    """`krt` splits the row at d/2 and pairs `(j, j + d/2)`. That is only the rotation when the
    rotation spans the whole row, so a partial-rotary model needs a kernel with an explicit
    rotary_dim -- not a different table. Refusing beats rotating the wrong lanes."""
    cache = _vllm_cache(32, d=D)
    try:
        dc.cos_sin_from_cache(cache, torch.arange(4), D, rotary_dim=D // 2)
    except NotImplementedError as exc:
        assert "partial rotary" in str(exc)
        print(f"  refused: {str(exc).splitlines()[0]}")
        return
    raise AssertionError("a partial-rotary table was accepted, so it would be applied to whole rows")


def test_cos_sin_from_cache_refuses_a_position_past_the_table():
    """A store longer than the model's `max_model_len` would silently read the wrong angle (or here,
    not read at all). The table is sized for the model, so this is a serving-configuration error."""
    cache = _vllm_cache(16)
    dc.cos_sin_from_cache(cache, torch.arange(16), D)          # the last valid position
    try:
        dc.cos_sin_from_cache(cache, torch.tensor([16]), D)
    except IndexError as exc:
        assert "past the model" in str(exc)
        print(f"  refused: {str(exc).splitlines()[0]}")
        return
    raise AssertionError("a position past the end of the table was accepted")


def test_split_blob_refuses_a_blob_sized_for_a_different_layout():
    """The K and V slots have different widths, so a reader that assumed one width would walk off
    the end of every row and read the next token's bytes -- same shapes, wrong values. `split_blob`
    checks the blob width against the layout for exactly that reason."""
    lay = TTKLayout(H=8, d=D, r1=64, qbits=8, res_k=64, res_bits=4, v_lanes=64, v_bits=4)
    blob = torch.zeros(4, lay.compressed_bytes_per_token(), dtype=torch.uint8)
    k, v = dc.split_blob(lay, blob)
    assert k.shape == (4, lay.k_slot_bytes) and v.shape == (4, lay.v_slot_bytes)
    assert k.is_contiguous() and v.is_contiguous(), (
        "the kernels take their row stride as a compile-time constant equal to the slot width, so a "
        "non-contiguous slice would make them walk into the next row")
    print(f"  k={lay.k_slot_bytes}B v={lay.v_slot_bytes}B, offset {lay.v_offset}, both contiguous")
    try:
        dc.split_blob(lay, torch.zeros(4, lay.k_slot_bytes, dtype=torch.uint8))
    except ValueError as exc:
        assert "blob is" in str(exc)
        return
    raise AssertionError("a blob sized for K alone was accepted as a whole token blob")


def test_block_slots_matches_vllms_flat_indexing():
    """A slot id is `block * block_size + offset`. The store is written through vLLM's own
    `slot_mapping`, so the read path has to use vLLM's convention -- a private one would be
    self-consistent and wrong, reading positions that belong to other requests."""
    block_size = 16
    bt = torch.tensor([3, 7, 11], dtype=torch.int32)
    slots = dc.block_slots(bt, 40, block_size)
    assert slots.tolist()[:17] == [3 * 16 + i for i in range(16)] + [7 * 16]
    assert slots.tolist()[-1] == 11 * 16 + 7          # 40 tokens = 2 full blocks + 8
    assert len(slots) == 40
    print(f"  opening {slots[:3].tolist()} middle {slots[16:18].tolist()} closing {slots[-1].item()}")

    try:
        dc.block_slots(bt, 49, block_size)             # 49 tokens needs 4 blocks, table has 3
    except IndexError as exc:
        assert "blocks of" in str(exc)
        return
    raise AssertionError("a seq_len longer than the block table was accepted")


def test_layout_slot_arithmetic_is_byte_consistent():
    """K and V are concatenated into ONE region, and the region's width is what the KV cache spec
    publishes as `state_content_bytes`. So the sum the layout reports and the offset the reader uses
    have to agree, the V slot has to start where the K slot ends, and each slot's width has to be
    the ALIGNMENT of its content -- `align_up`, not the raw sum. Asserting the raw sum would let a
    change to the padding pass while the kernels still walk off the end of a row."""
    lay = TTKLayout(H=8, d=D, r1=64, qbits=8, res_k=64, res_bits=4, v_lanes=64, v_bits=4)
    a = lay.align

    def align_up(n):
        return -(-n // a) * a

    k_raw = lay.coeff_bytes + lay.coeff_scale_bytes + lay.H * lay.k_row_bytes
    v_raw = lay.H * lay.v_row_bytes
    assert lay.k_slot_bytes == align_up(k_raw), f"{lay.k_slot_bytes} != align_up({k_raw})"
    assert lay.v_slot_bytes == align_up(v_raw), f"{lay.v_slot_bytes} != align_up({v_raw})"
    assert lay.k_slot_bytes % a == 0 and lay.v_slot_bytes % a == 0
    assert lay.v_offset == lay.k_slot_bytes
    assert lay.v_offset % a == 0, "the V slot must start on an aligned boundary inside the blob"
    assert lay.compressed_bytes_per_token() == lay.k_slot_bytes + lay.v_slot_bytes
    print(f"  r1=64 -> K {lay.k_slot_bytes}B (raw {k_raw}) + V {lay.v_slot_bytes}B (raw {v_raw}) = "
          f"{lay.compressed_bytes_per_token()}B/token, ratio {lay.ratio():.2f}x")


def test_krt_slot_stride_matches_the_layout_it_is_read_with():
    """`krt` is handed `SLOT=lay.k_slot_bytes` as a compile-time row stride and a per-head row of
    `d/8 + PB + 2`. Those two constants have to be the layout's, or the kernel reads bitmaps as
    codes. Asserting the arithmetic here catches a change to either without a GPU."""
    for cfg, r1 in ((dict(qbits=8, res_frac=0.633, res_bits=3, k_lanes=81, vbits=3), 100),
                    (dict(qbits=8, res_frac=0.25, res_bits=3, k_lanes=45, vbits=3), 103),
                    (dict(qbits=8, res_frac=0.5, res_bits=4, k_lanes=64, vbits=4), 96)):
        lay = TTKLayout(H=8, d=D, r1=r1, qbits=cfg["qbits"], res_k=int(D * cfg["res_frac"]),
                        res_bits=cfg["res_bits"], v_lanes=cfg["k_lanes"], v_bits=cfg["vbits"])
        pb = packed_row_bytes(lay.res_k, lay.res_bits)
        row_stride = D // 8 + pb + 2
        coeff_end = lay.coeff_bytes + lay.coeff_scale_bytes
        assert coeff_end + lay.H * row_stride <= lay.k_slot_bytes <= coeff_end + lay.H * row_stride \
            + lay.align, "the K row walk does not fit the slot the layout sized"
        # `krt` reads the ruler two bytes past the packed codes, so PB must leave room for it.
        assert lay.k_row_bytes == D // 8 + pb + 2
        print(f"  r1={r1:<4d} res_k={lay.res_k:<3d} PB={pb:<3d} row={row_stride:<3d} "
              f"slot={lay.k_slot_bytes:<4d} ends at {coeff_end + lay.H * row_stride}")


if __name__ == "__main__":
    for fn in (test_cos_sin_from_cache_reproduces_vllms_half_layout_rotation,
               test_cos_sin_from_cache_indexes_by_absolute_position,
               test_cos_sin_from_cache_refuses_partial_rotary,
               test_cos_sin_from_cache_refuses_a_position_past_the_table,
               test_split_blob_refuses_a_blob_sized_for_a_different_layout,
               test_block_slots_matches_vllms_flat_indexing,
               test_layout_slot_arithmetic_is_byte_consistent,
               test_krt_slot_stride_matches_the_layout_it_is_read_with):
        print(f"{fn.__name__}:")
        fn()
    print("\nconvention validation complete")
