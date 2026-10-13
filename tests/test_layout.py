"""The one part of the plugin that can be checked without a vLLM install.

What is actually being asserted, and why it is the right thing to assert:

  * the byte arithmetic in `TTKLayout` is compared against the container's OWN `nbytes()`, so a
    drift between the layout and the codec shows up as a test failure rather than as a ratio
    claim nobody can reproduce;
  * the slot round-trip is compared against `PackedRows.materialise()`, so packing is checked
    to be a faithful REARRANGEMENT of the codec's bytes rather than a re-implementation that
    merely looks right.

Runs under pytest or as a plain script (`python tests/test_layout.py`).
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
# Support BOTH layouts: <root>/vllm_ttkv_plugin with the codec in <root>/integration (this project),
# and a flat checkout where the codec sits beside the plugin (the GPU box).
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.layout import (TTKLayout, layout_for_config, pack_k_slots,  # noqa: E402
                              pack_v_slots, unpack_k_slots, unpack_v_slots)

H, D, N = 8, 128, 64
SEED = 0


def _store(cfg, n=N, seed=SEED):
    torch.manual_seed(seed)
    K = torch.randn(H, n, D) * 0.2
    V = torch.randn(H, n, D) * 0.2
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    vl = pc.build_v_layer(V, cfg["k_lanes"], cfg["vbits"], d=D)
    return kl, vl


def test_slot_accounting_matches_the_container():
    """The layout's per-token byte count must equal what the container actually bills.

    The container bills, per (head, token) row, `bitmap + lanes + fp16 ruler`, plus the
    per-token coefficient row and its ruler. The layout must reproduce that exactly -- any
    disagreement means either the layout or the ratio table is wrong, and the ratio table is
    what the port would be claiming.
    """
    for name in ("safe", "tight", "campaign4x"):
        cfg = CONFIGS[name]
        kl, vl = _store(cfg)
        lay = layout_for_config(cfg, r1=kl.r1)

        # Container bytes, minus the per-layer constants the cache does not carry. The decoder
        # is per layer and CONSTANT, so it must not be inside the slot.
        layer_constant = int(kl.dec_q.numel()) + int(kl.dec_scale.numel()) * 2
        assert layer_constant > 0
        k_cache = kl.nbytes() - layer_constant

        k_content = lay.coeff_bytes + lay.coeff_scale_bytes + lay.H * lay.k_row_bytes
        v_content = lay.H * lay.v_row_bytes
        assert k_cache == N * k_content, \
            f"{name}: K billing disagrees: container {k_cache}, layout {N * k_content}"
        assert vl.nbytes() == N * v_content, \
            f"{name}: V billing disagrees: container {vl.nbytes()}, layout {N * v_content}"

        # And the slot is that content rounded UP to the alignment vLLM expects a backend to
        # publish (TurboQuant's field is literally `slot_size_aligned`). Up only, never down.
        assert lay.k_slot_bytes >= k_content and lay.k_slot_bytes % lay.align == 0
        assert lay.v_slot_bytes >= v_content and lay.v_slot_bytes % lay.align == 0
        print(f"  {name:11s} r1={lay.r1:3d} K={lay.k_slot_bytes:4d} B/token "
              f"V={lay.v_slot_bytes:4d} B/token  ratio={lay.ratio():5.2f}x")


def test_k_slots_round_trip():
    """Packing a token's K store into its slot and reading it back must be lossless."""
    cfg = CONFIGS["campaign4x"]
    kl, _ = _store(cfg)
    lay = layout_for_config(cfg, r1=kl.r1)

    slots = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
    assert slots.shape == (N, lay.k_slot_bytes)
    assert slots.dtype == torch.uint8
    # The slot is a fixed-width object per token: that is what makes block_size * slot the page.
    assert slots.stride(0) == lay.k_slot_bytes

    cq, csc, (bitmap, packed, scale) = unpack_k_slots(lay, slots)
    assert torch.equal(cq, kl.coeff_q), "coefficients did not survive the slot"
    assert torch.equal(csc, kl.coeff_scale), "coefficient rulers did not survive the slot"
    assert torch.equal(bitmap, kl.resid.bitmap)
    assert torch.equal(packed, kl.resid.packed)
    assert torch.equal(scale, kl.resid.scale)

    # And the reconstruction has to agree with the codec's own materialisation, not just with
    # the arrays we happened to copy back in.
    rebuilt = pc.PackedRows(lay.H * N, D, kl.resid.k, kl.resid.bits)
    rebuilt.bitmap, rebuilt.packed, rebuilt.scale = bitmap, packed, scale
    assert torch.equal(rebuilt.materialise(), kl.resid.materialise()), \
        "round-tripped residual does not materialise to the same dense rows"


def test_v_slots_round_trip():
    cfg = CONFIGS["tight"]
    _, vl = _store(cfg)
    lay = layout_for_config(cfg, r1=cfg.get("r1", 128))
    slots = pack_v_slots(lay, vl.rows)
    assert slots.shape == (N, lay.v_slot_bytes)
    bitmap, packed, scale = unpack_v_slots(lay, slots)
    assert torch.equal(bitmap, vl.rows.bitmap)
    assert torch.equal(packed, vl.rows.packed)
    assert torch.equal(scale, vl.rows.scale)

    rebuilt = pc.PackedRows(lay.H * N, D, vl.rows.k, vl.rows.bits)
    rebuilt.bitmap, rebuilt.packed, rebuilt.scale = bitmap, packed, scale
    assert torch.equal(rebuilt.materialise(), vl.rows.materialise())


def test_dense_equivalence_and_alignment():
    """The layout must be lossless against dense fp16 K/V, and every slot aligned.

    This is the check that would matter if the head axis were folded WRONG: a permutation
    still round-trips through pack/unpack, so the assertion has to be on the materialised
    dense block, not on the slots.
    """
    cfg = CONFIGS["safe"]
    kl, vl = _store(cfg)
    lay = layout_for_config(cfg, r1=kl.r1)

    Kd = kl.materialise(H)                          # [H, n, d], the codec's own reconstruction
    Vd = vl.materialise(H)

    cq, csc, (bitmap, packed, scale) = unpack_k_slots(lay, pack_k_slots(
        lay, kl.coeff_q, kl.coeff_scale, kl.resid))
    resid = pc.PackedRows(lay.H * N, D, kl.resid.k, kl.resid.bits)
    resid.bitmap, resid.packed, resid.scale = bitmap, packed, scale

    # (head, token) row order is h*n + i, so the residual must reshape head-major and the
    # reconstruction must place each head's rows under that head.
    g = cq.float() * csc.float().unsqueeze(1)
    kflat = g @ kl.decoder_f()                                   # [n, H*d]
    R = resid.materialise(torch.float32).reshape(H, N, D)
    got = (kflat.reshape(N, H, D).permute(1, 0, 2) + R)
    assert torch.equal(got.to(torch.float16), Kd), \
        "the slot layout permuted (head, token) -- this is the failure mode to watch for"
    assert torch.equal(unpack_v_slots(lay, pack_v_slots(lay, vl.rows))[0], vl.rows.bitmap)

    for width in (lay.k_slot_bytes, lay.v_slot_bytes):
        assert width % lay.align == 0, f"slot width {width} is not a multiple of {lay.align}"


def test_larger_rank_grows_the_slot_linearly():
    """Coefficient bytes are `r1 * qbits / 8` per token, so the slot grows linearly in r1.

    This is the arithmetic that turns post-RoPE rank inflation into bytes -- and the reason the
    pre-rotary form exists at all. Asserting it here keeps the claim anchored to the layout
    rather than to a modelling assumption.
    """
    cfg = CONFIGS["campaign4x"]
    a, b = layout_for_config(cfg, r1=64), layout_for_config(cfg, r1=320)
    assert b.coeff_bytes - a.coeff_bytes == (320 - 64) * cfg["qbits"] // 8
    assert b.k_row_bytes == a.k_row_bytes, "the residual row must not depend on the rank"
    assert b.k_slot_bytes > a.k_slot_bytes
    print(f"  r1 64 -> K {a.k_slot_bytes} B/token ({a.ratio():.2f}x); "
          f"r1 320 -> K {b.k_slot_bytes} B/token ({b.ratio():.2f}x)")


def test_per_layer_rank_gives_a_per_layer_slot():
    """`r1` varies per layer, so slot width does too. A port that assumes one global row size
    is the version that fails on the block-size question."""
    cfg = CONFIGS["campaign4x"]
    widths = {layout_for_config(cfg, r1=r).k_slot_bytes for r in (93, 279, 616)}
    assert len(widths) == 3, "per-layer ranks must give per-layer slot widths"


def test_ratio_is_nearly_invariant_to_the_head_dim_split():
    """The layout generalises across head dims; the CONFIG does not, and they are different claims.

    At a fixed kv-hidden (H*d) and a fixed lane FRACTION, the joint ratio moves only a few
    percent across d in {64, 128, 256, 512}: the bitmap is H*(d/8) = kv_hidden/8 and the
    residual lanes are H*d*res_frac = kv_hidden*res_frac, so both are invariant to how the
    hidden size is split. What is left is per-row byte padding and the 2-byte fp16 ruler per
    head, which is why more heads (smaller d) is marginally worse.

    The second half is the part that matters operationally: `k_lanes` is an ABSOLUTE count while
    `res_frac` is a fraction, so the same config is a different format at a different d -- and
    below d = k_lanes it is not a format at all.
    """
    cfg = CONFIGS["safe"]
    r1, kv_hidden = 128, 1024
    frac = 0.633

    def mk(H, d, v_lanes):
        return TTKLayout(H=H, d=d, r1=r1, qbits=cfg["qbits"],
                         res_k=max(int(d * frac), 1), res_bits=cfg["res_bits"],
                         v_lanes=v_lanes, v_bits=cfg["vbits"])

    ratios = {}
    for H, d in ((16, 64), (8, 128), (4, 256), (2, 512)):
        assert H * d == kv_hidden
        ratios[d] = mk(H, d, round(frac * d)).ratio()
    spread = max(ratios.values()) / min(ratios.values())
    assert spread < 1.10, f"ratio moved {spread:.3f}x across the head-dim split: {ratios}"

    # An absolute lane count does NOT transfer: at d=64 the shipped 81 lanes do not exist.
    try:
        mk(16, 64, 81)
    except ValueError as exc:
        assert "v_lanes" in str(exc)
    else:
        raise AssertionError("expected a d=64 layout with k_lanes=81 to be rejected")

    # And at a wider head the same absolute count is a coarser format, which flatters the ratio.
    shipped_at_128 = mk(8, 128, 81).ratio()
    shipped_at_256 = mk(4, 256, 81).ratio()
    assert shipped_at_256 > shipped_at_128, (
        "keeping 81 of 256 lanes is a smaller FRACTION than 81 of 128, so the ratio should read "
        "higher while the approximation is worse -- the number and the quality move apart")


if __name__ == "__main__":
    for fn in (test_slot_accounting_matches_the_container, test_k_slots_round_trip,
               test_v_slots_round_trip, test_dense_equivalence_and_alignment,
               test_larger_rank_grows_the_slot_linearly, test_per_layer_rank_gives_a_per_layer_slot,
               test_ratio_is_nearly_invariant_to_the_head_dim_split):
        print(f"{fn.__name__}:")
        fn()
    print("\nall layout tests passed")
