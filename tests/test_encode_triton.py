"""Is the fused write the SAME BYTES as the torch write, and does it still refuse to guess?

`select_triton` replaces about twenty launches a layer with one kernel, and the only thing that makes
that safe is that it has to reproduce `PackedRows.build` + `pack_{k,v}_slots` exactly. So the oracle
here is not a dense reconstruction -- it is the other implementation, compared with `torch.equal` on
a whole slot tensor. A reconstruction comparison would pass with the bitmap, the code stream or the
ruler wrong by one lane, which are exactly the errors that are cheap to make and expensive to find
later.

Four things are checked, and they are different in kind:

  * THE BYTES MATCH, for every shipped config, against the store the READ path will unpack. K and V
    are compared separately, because they are two different (lanes, bits) geometries over the same
    slot layout and a bug in one cannot be caught by the other's oracle.
  * THE FUSED PATH ACTUALLY RAN. `_fused_write` returns whether it took the fast branch, and every
    byte test asserts it returned True -- otherwise the comparisons would be true by construction and
    the file would pass while testing nothing.
  * THE FALLBACK STILL WORKS, on the two paths that must keep it: a FITTED store (whose residual the
    codec derives internally) and a geometry the kernel refuses (a head dim that is not a power of
    two). Both compare against the same oracle, so a refusal cannot quietly write garbage.
  * TIES ARE A CHOICE, NOT AN ERROR. `torch.topk` documents no tie order and the kernel picks the
    lowest lane index, so on a row whose magnitudes tie the two can disagree about the BITMAP. The
    claim that this cannot matter is checked rather than asserted: the reconstruction and the lane
    count are compared on tie-heavy data.

Needs the codec and a GPU, and skips cleanly without either.
"""
from __future__ import annotations

import os
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv import encode as enc  # noqa: E402
from vllm_ttkv import select_triton  # noqa: E402
from vllm_ttkv.layout import (layout_for_config, pack_k_slots, pack_v_slots,  # noqa: E402
                              unpack_v_slots)
from vllm_ttkv.reference import dense_k_from_slots, dense_v_from_slots  # noqa: E402

H, D = 8, 128


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device (the fused write is a Triton kernel)")
        return False
    return True


def _raw_k(h, n, d, rank=24, noise=0.02, seed=0):
    """`[H, n, d]` RAW keys: low rank along tokens plus a noise floor, so the fit neither saturates
    the budget nor reconstructs exactly -- which is what makes the residual patch live."""
    torch.manual_seed(seed)
    K = ((torch.randn(n, rank, device="cuda") * 0.2)
         @ (torch.randn(rank, h * d, device="cuda") * 0.2)
         ).reshape(n, h, d).permute(1, 0, 2).contiguous()
    return K + noise * torch.randn(h, n, d, device="cuda")


def _frozen(cfg, lay, n, seed=0, h=H, d=D, fit_on=64):
    """A FITTED FittedK for the first chunk, then a FROZEN one over the next `n` tokens.

    Deliberately the engine's own sequence -- `fit_into_budget` on a request's first chunk,
    `encode_frozen` on every step after it -- because the fused path is only offered to the frozen
    one.
    """
    K = _raw_k(h, fit_on + n, d, seed=seed)
    fitted = enc.fit_into_budget(K[:, :fit_on].contiguous(), lay, cfg)
    return enc.encode_frozen(K[:, fit_on:fit_on + n].contiguous(), fitted.decoder, lay), fitted


def _store(cfg, lay, n, seed=0, v_dtype=torch.float32, h=H, d=D):
    """The same store, written both ways: `(oracle_slots, fused_cache, fk, V, ran)`.

    The oracle is `pack_k_slots` + `pack_v_slots` over the SAME `FittedK` and the same V, which is
    what the torch path in `encode.pack_and_write` builds. `ran` is what `_fused_write` returned.
    """
    fk, _ = _frozen(cfg, lay, n, seed=seed, h=h, d=d)
    torch.manual_seed(seed + 1000)
    V = (torch.randn(h, n, d, device="cuda") * 0.3).to(v_dtype)
    slots = torch.arange(n, device="cuda", dtype=torch.int32)
    oracle = torch.cat([pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid),
                        pack_v_slots(lay, enc.encode_v(V, lay))], dim=-1)
    cache = torch.zeros(n, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")
    ran = enc._fused_write(cache, slots, fk, V, lay, check=True)
    return oracle, cache, fk, V, ran


# --------------------------------------------------------------------------- the bytes
def test_the_fused_write_is_byte_identical_to_the_torch_write_for_every_shipped_config():
    """The whole safety argument for the kernel in one assertion per config.

    K and V are two geometries over one slot layout: K's residual rows sit behind a coefficient
    block and a per-token ruler, V's sit at `v_offset`, and they have different lane counts and
    widths. Comparing the concatenated slot is what catches a field written at the wrong offset,
    which is the failure that still looks like a well-formed store.
    """
    if not _need_cuda():
        return
    for name, r1 in (("safe", 112), ("tight", 96), ("campaign4x", 64)):
        cfg = CONFIGS[name]
        lay = layout_for_config(cfg, r1=r1, H=H, d=D)
        oracle, cache, fk, V, ran = _store(cfg, lay, n=7, seed=r1)
        assert ran is True, (
            f"{name}: the fused path did not run, so this comparison would be true by construction")
        assert fk.resid_src is not None, f"{name}: a frozen store has no residual values"
        assert torch.equal(cache, oracle), (
            f"{name}: {int((cache != oracle).sum())} of {cache.numel()} bytes differ between the "
            f"fused write and pack_k_slots/pack_v_slots")
        print(f"  {name}: r1={r1} res_k={lay.res_k}@{lay.res_bits}b v_lanes={lay.v_lanes}"
              f"@{lay.v_bits}b, {cache.numel()} bytes identical")


def test_the_fused_write_survives_a_bf16_value_tensor():
    """What the engine actually passes. V arrives as the model's bf16 activation and the torch path
    casts it to fp32 (`as_fit_dtype`, because `build_v_layer` quantises in the incoming dtype) -- so
    the kernel has to load bf16 and do the same, and that is only equivalent because bf16 to fp32 is
    exact. If it were not, this is where it would show.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    oracle, cache, fk, V, ran = _store(cfg, lay, n=5, seed=7, v_dtype=torch.bfloat16)
    assert V.dtype == torch.bfloat16
    assert ran is True, "the fused path declined a bf16 value tensor"
    assert torch.equal(cache, oracle), (
        f"{int((cache != oracle).sum())} of {cache.numel()} bytes differ on the bf16 path")


def test_pack_and_write_and_the_kill_switch_agree():
    """`pack_and_write` is what the impl calls, and the env var is what selects the other arm. Both
    have to produce the same slot, or the two arms of a benchmark would be measuring two formats.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    oracle, cache, fk, V, ran = _store(cfg, lay, n=6, seed=11)
    assert ran is True

    slots = torch.arange(6, device="cuda", dtype=torch.int32)
    C = lay.compressed_bytes_per_token()
    live = torch.zeros(6, C, dtype=torch.uint8, device="cuda")
    enc.pack_and_write(live, slots, fk, V, lay)
    torch.cuda.synchronize()
    assert torch.equal(live, oracle), "pack_and_write did not take the fused path"

    os.environ[select_triton.ENV] = "0"
    try:
        off = torch.zeros(6, C, dtype=torch.uint8, device="cuda")
        assert select_triton.supported(fk.resid_src, lay.d, lay.res_k, lay.res_bits) is False
        enc.pack_and_write(off, slots, fk, V, lay)
        torch.cuda.synchronize()
    finally:
        del os.environ[select_triton.ENV]
    assert torch.equal(off, live), "the fallback and the fused write disagree"
    assert torch.equal(off, oracle)


# --------------------------------------------------------------------------- the fallbacks
def test_a_fitted_store_keeps_the_torch_packing_and_is_still_exact():
    """The FIT path is not offered the kernel, and the reason is a real difference rather than an
    oversight: `build_k_layer` derives its residual from the UNPADDED product, so its `PackedRows`
    is not recoverable from the store's own coefficients. What must hold is that declining changes
    nothing about the bytes.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    K = _raw_k(H, 40, D, seed=3)
    fk = enc.fit_into_budget(K, lay, cfg)
    assert fk.resid_src is None, "the fit path was offered a residual the kernel could pack"

    torch.manual_seed(4)
    V = torch.randn(H, 40, D, device="cuda") * 0.3
    slots = torch.arange(40, device="cuda", dtype=torch.int32)
    oracle = torch.cat([pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid),
                        pack_v_slots(lay, enc.encode_v(V, lay))], dim=-1)
    cache = torch.zeros(40, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")
    assert enc._fused_write(cache, slots, fk, V, lay, check=True) is False, (
        "the fit path took the fused branch")
    enc.pack_and_write(cache, slots, fk, V, lay)
    torch.cuda.synchronize()
    assert torch.equal(cache, oracle)


def test_a_head_dim_that_is_not_a_power_of_two_falls_back_byte_exactly():
    """`supported` refuses where the arithmetic would be wrong rather than slow: the kernel's block
    is `d` wide and the bitmap fold is `d/8` bytes, so a head dim of 96 (Mistral-7B-v0.1,
    Phi-3-small) has no block shape. The fallback has to be exact there too, since that is the only
    path those models have.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    d = 96
    lay = layout_for_config(cfg, r1=64, H=H, d=d)
    fk, _ = _frozen(cfg, lay, n=4, seed=5, d=d)
    torch.manual_seed(6)
    V = torch.randn(H, 4, d, device="cuda") * 0.3
    assert select_triton.supported(fk.resid_src, d, lay.res_k, lay.res_bits) is False, (
        "d=96 was accepted, which means the block shape is not what this test thinks")

    slots = torch.arange(4, device="cuda", dtype=torch.int32)
    oracle = torch.cat([pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid),
                        pack_v_slots(lay, enc.encode_v(V, lay))], dim=-1)
    cache = torch.zeros(4, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")
    assert enc._fused_write(cache, slots, fk, V, lay, check=True) is False
    enc.pack_and_write(cache, slots, fk, V, lay)
    torch.cuda.synchronize()
    assert torch.equal(cache, oracle), "the d=96 fallback disagrees with the oracle"


def test_the_slot_refusals_are_the_same_two_on_both_paths():
    """A fused write that skipped validation would be a silent corruption rather than a slow path:
    `PAD_SLOT_ID` is -1 and a slot past the end addresses another request's row. Both callers go
    through `check_slot_ids`, so this asserts it is reached, not merely present.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    fk, _ = _frozen(cfg, lay, n=3, seed=9)
    torch.manual_seed(10)
    V = torch.randn(H, 3, D, device="cuda") * 0.3
    cache = torch.zeros(8, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")

    for bad, exc in ((torch.tensor([0, -1, 2], dtype=torch.int32), RuntimeError),
                     (torch.tensor([0, 1, 99], dtype=torch.int32), IndexError)):
        try:
            enc._fused_write(cache, bad.cuda(), fk, V, lay, check=True)
        except exc as e:
            assert "slot" in str(e).lower(), f"the refusal does not name the slot: {e}"
        else:
            raise AssertionError(f"{bad.tolist()} was accepted by the fused write")

    # `check=False` is the step-level promise, and it is honoured by not reading the ids back.
    ok = enc._fused_write(cache, torch.tensor([0, 1, 2], dtype=torch.int32).cuda(), fk, V, lay,
                          check=False)
    assert ok is True


# --------------------------------------------------------------------------- the ties
def test_tied_magnitudes_are_a_bitmap_choice_and_not_a_different_value():
    """The one case where the kernel and `torch.topk` may pick different lanes.

    The kernel breaks a tie by the lowest lane index; `torch.topk` documents no tie order at all. So
    on a row whose magnitudes tie, the BITMAP can differ -- and the claim is that it cannot matter,
    because a tie means the two candidates have equal magnitude and therefore the same code. That is
    checked here directly instead of being asserted in prose: the reconstruction, the lane count and
    the multiset of decoded values all have to agree, while the byte comparison is deliberately not
    made (it is the tie-free data in the first test where byte equality is the right assertion).
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=96, H=H, d=D)
    fk, _ = _frozen(cfg, lay, n=6, seed=13)
    # Values from a three-element alphabet: most lanes of most rows tie with another lane.
    torch.manual_seed(14)
    V = torch.randint(0, 3, (H, 6, D), device="cuda").float() - 1.0

    slots = torch.arange(6, device="cuda", dtype=torch.int32)
    C = lay.compressed_bytes_per_token()
    cache = torch.zeros(6, C, dtype=torch.uint8, device="cuda")
    assert enc._fused_write(cache, slots, fk, V, lay, check=True) is True
    torch.cuda.synchronize()

    ref_v = pack_v_slots(lay, enc.encode_v(V, lay))
    got_v = cache[:, lay.v_offset:lay.v_offset + lay.v_slot_bytes]
    ref_dense = dense_v_from_slots(lay, ref_v)
    got_dense = dense_v_from_slots(lay, got_v)
    assert torch.allclose(got_dense, ref_dense, atol=1e-6), (
        "tied magnitudes decoded to different values, so the tie order is NOT harmless")
    assert ref_dense.abs().max() > 0.5, "the fixture's magnitudes are all zero, so this is vacuous"

    # The lane count itself is not a choice: every row keeps exactly `v_lanes`.
    bm = got_v[:, :D // 8]
    kept = pc.unpack_bitmap_rows(bm, D).sum(-1)
    assert int(kept.min()) == lay.v_lanes == int(kept.max()), (
        f"rows kept {int(kept.min())}..{int(kept.max())} lanes against v_lanes={lay.v_lanes}")


def test_all_zero_rows_decode_to_zero_on_both_paths():
    """The degenerate tie: every lane is 0, so every lane is an equally valid choice of the `k`
    largest. `torch.topk` will pick some set and the kernel picks the first `k` lanes, so the bitmaps
    differ by construction -- and the codes are all the offset, so both decode to zero.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=96, H=H, d=D)
    fk, _ = _frozen(cfg, lay, n=4, seed=15)
    V = torch.zeros(H, 4, D, device="cuda")
    slots = torch.arange(4, device="cuda", dtype=torch.int32)
    cache = torch.zeros(4, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")
    assert enc._fused_write(cache, slots, fk, V, lay, check=True) is True
    torch.cuda.synchronize()

    ref = pack_v_slots(lay, enc.encode_v(V, lay))
    got = cache[:, lay.v_offset:lay.v_offset + lay.v_slot_bytes]
    assert float(dense_v_from_slots(lay, got).abs().max()) == 0.0
    assert float(dense_v_from_slots(lay, ref).abs().max()) == 0.0
    # The ruler is `1 / (2**(bits-1) - 1)` on such a row -- the codec substitutes 1 for a zero max
    # BEFORE dividing, and that nonzero ruler is what keeps the zeros from being rescaled -- so the
    # two paths write the same two bytes here. This is the one all-tie case where the ruler is
    # checkable independently of the bitmap.
    want = float(torch.tensor(1.0 / (2 ** (lay.v_bits - 1) - 1), dtype=torch.float16))
    sc_fused = unpack_v_slots(lay, got)[2]
    assert float(sc_fused.min()) == want == float(sc_fused.max()), (
        f"the fused write wrote ruler {float(sc_fused.min())}..{float(sc_fused.max())} "
        f"against {want}")
    assert float(unpack_v_slots(lay, ref)[2].max()) == want


def test_dense_k_from_the_fused_store_matches_the_torch_store():
    """The byte equality above is the strong assertion; this one is the reason to care about it --
    a store the READ path turns into the same keys either way.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    fk, _ = _frozen(cfg, lay, n=5, seed=17)
    torch.manual_seed(18)
    V = torch.randn(H, 5, D, device="cuda") * 0.3
    slots = torch.arange(5, device="cuda", dtype=torch.int32)
    cache = torch.zeros(5, lay.compressed_bytes_per_token(), dtype=torch.uint8, device="cuda")
    assert enc._fused_write(cache, slots, fk, V, lay, check=True) is True
    torch.cuda.synchronize()

    fused = dense_k_from_slots(lay, cache[:, :lay.k_slot_bytes], fk.decoder.decoder_f)
    torch_store = dense_k_from_slots(
        lay, pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid), fk.decoder.decoder_f)
    assert torch.equal(fused, torch_store)
