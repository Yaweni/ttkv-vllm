"""Does the WRITE produce a store that holds the model's own keys?

The read path is gated elsewhere (`test_krt_triton`, `test_v_out_triton`, `test_impl_decode`). What
none of those reach is the encode: the TT-SVD fit, the rank budget, the frozen-decoder projection,
and the residual patch. A bug there is the quietest kind in this project -- a mis-padded slot, a
stale decoder or a wrong residual ruler all still produce a full store of correctly-shaped slots, and
the read path will happily attend over them.

Three assertions do the work, and they are different in kind:

  * PADDING IS EXACT. The budget is fixed at cache-allocation time while a request's fit lands at its
    own natural rank, so the fit is zero-padded up to the budget. That has to change nothing, and
    `test_the_fit_pads_without_changing_the_reconstruction` checks it against the unpadded fit at the
    byte level, because "changes nothing" is the only reason padding is safe.
  * THE FROZEN PATH IS THE ONE THAT WAS MEASURED. `tools/measure_fold_gap.py` measured the fold
    penalty and that measurement is what justifies the fold knob, so this pins the shipped encode to
    the tool's arithmetic. Otherwise the numbers would stop describing the shipped path the moment
    either side drifted, and nothing would say so.
  * A BINDING BUDGET IS THE ACCURACY COST, and it is the allocation decision rather than a bug.

Needs the codec and a GPU (the fit runs on device), and skips cleanly without either.
"""
from __future__ import annotations

import importlib.util
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
from vllm_ttkv.layout import layout_for_config, pack_k_slots, pack_v_slots  # noqa: E402
from vllm_ttkv.reference import dense_k_from_slots, dense_v_from_slots  # noqa: E402
from vllm_ttkv.rope_store import CONFIG_ENV  # noqa: E402

H, D = 8, 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device (the TT-SVD fit runs on device)")
        return False
    return True


def _k_block(n, rank=16, noise=0.02, seed=0, h=H, d=D):
    """`[H, n, d]` RAW keys: low rank along tokens plus a noise floor, so the fit is not exact and
    the residual patch is live -- the case `tests/test_krt_triton.py` uses for the same reason."""
    torch.manual_seed(seed)
    K = ((torch.randn(n, rank, device=DEV) * 0.2)
         @ (torch.randn(rank, h * d, device=DEV) * 0.2)
         ).reshape(n, h, d).permute(1, 0, 2).contiguous()
    return K + noise * torch.randn(h, n, d, device=DEV)


def _frozen_arm(K, dec_f, cfg):
    """The tool's own frozen encode: `[H, n, d]` -> its reconstruction `[H, n, d]`."""
    return _load_tool()._encode_frozen(K, dec_f, cfg["qbits"], cfg["res_frac"], cfg["res_bits"])


def _load_tool():
    """Import `tools/measure_fold_gap.py` by path. Not a package, and it must not be copied."""
    spec = importlib.util.spec_from_file_location("_ttkv_measure_fold_gap",
                                                 _HERE.parents[1] / "tools" / "measure_fold_gap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _roundtrip(lay, fk):
    """`FittedK` -> slots -> the dense reconstruction the READ path would see."""
    return dense_k_from_slots(lay, pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid),
                              fk.decoder.decoder_f)


# --------------------------------------------------------------------------- the budget
def test_the_fit_pads_without_changing_the_reconstruction():
    """The budget is fixed before any sequence exists; a request's natural rank is whatever the data
    gives. So the fit is zero-padded up to the budget, and THE PADDING HAS TO BE EXACT -- a padded
    fit and an unpadded one have to produce the same keys, or every rank decision in the calibration
    table would be entangled with a rounding artefact.

    Two things make it exact, and both are checked directly rather than inferred from the error:
    a zero coefficient times a zero decoder row is zero, and `pinv` of a decoder with zero ROWS has
    zero COLUMNS where those rows were -- so the frozen path later solves into the shared columns and
    leaves the pad at exactly zero instead of growing a ghost basis.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    K = _k_block(96)
    wide = layout_for_config(cfg, r1=256, H=H, d=D)
    fk = enc.fit_into_budget(K, wide, cfg)
    assert fk.fitted_rank < wide.r1, (
        f"the natural rank {fk.fitted_rank} filled the budget {wide.r1}, so this fixture never "
        f"pads and the test would pass vacuously")

    # The pad is exactly zero, on BOTH sides of the product.
    assert fk.coeff_q.shape == (96, wide.r1), f"coefficients are {tuple(fk.coeff_q.shape)}"
    assert int(fk.coeff_q[:, fk.fitted_rank:].abs().max()) == 0, "pad coefficients are not zero"
    assert float(fk.decoder.decoder_f[fk.fitted_rank:].abs().max()) == 0.0, "pad decoder is not zero"

    # The residual is budget-INDEPENDENT -- it is computed from the unpadded product in the codec --
    # so it must be bit-identical, which is a stronger statement than the reconstruction matching.
    narrow = layout_for_config(cfg, r1=fk.fitted_rank, H=H, d=D)
    ref = enc.fit_into_budget(K, narrow, cfg)
    assert ref.fitted_rank == fk.fitted_rank
    for field in ("bitmap", "packed", "scale"):
        assert torch.equal(getattr(fk.resid, field), getattr(ref.resid, field)), (
            f"the residual's {field} changed with the budget, but it is computed before padding "
            f"and must not depend on it")

    # And the reconstruction is the same keys.
    a, b = _roundtrip(wide, fk), _roundtrip(narrow, ref)
    assert torch.allclose(a, b, atol=1e-5, rtol=0), (
        f"padding to the budget changed the reconstruction by {_rel(a, b):.3e}; a padded store and "
        f"an unpadded one have to reconstruct the same keys")
    print(f"  natural rank {fk.fitted_rank} -> budget {wide.r1}: pad exact, "
          f"reconstruction identical (max delta {(a - b).abs().max():.2e})")


def test_a_binding_budget_is_the_accuracy_cost_and_it_is_monotone():
    """A cap that binds under-ranks the fit, and that is the ALLOCATION DECISION rather than a defect:
    measured, a cap of 64 under-ranks `tight` by 1.75x. The property worth asserting is that a
    bigger budget is never worse, because that is what makes the calibration table's per-layer rank a
    meaningful thing to spend bytes on.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    K = _k_block(256, rank=48, seed=3)
    errs = {}
    for budget in (16, 32, 64, 128, 256):
        lay = layout_for_config(cfg, r1=budget, H=H, d=D)
        fk = enc.fit_into_budget(K, lay, cfg)
        errs[budget] = _rel(_roundtrip(lay, fk), K)
        print(f"  budget r1={budget:<4d} natural {fk.fitted_rank:<4d}  rel-F {errs[budget]:.4f}")
    assert errs[256] <= min(errs.values()) + 1e-3, (
        f"a larger budget was worse: {errs}. The cap is not the only thing setting the rank, so the "
        f"calibration table would be spending bytes on nothing")
    assert errs[256] < errs[16], (
        f"a 16x smaller budget cost nothing ({errs}), so this fixture cannot see the cap at all")


def test_check_cfg_matches_layout_refuses_a_foreign_allocation():
    """The layout sizes the slot; the encoder fills it. A layout built from one allocation and a
    config from another puts every field after the coefficients at a different offset -- and the
    store still reads back as plausible numbers, because a residual bitmap is a valid byte string
    too. This is the check that makes the two readers of `$VLLM_TTKV_CONFIG` agree.
    """
    safe = layout_for_config(CONFIGS["safe"], r1=64, H=H, d=D)
    enc.check_cfg_matches_layout(safe, CONFIGS["safe"])           # must be silent
    try:
        enc.check_cfg_matches_layout(safe, CONFIGS["tight"])
    except RuntimeError as exc:
        assert "does not match the active allocation" in str(exc)
        print(f"  refused: {str(exc).splitlines()[0][:110]}")
        return
    raise AssertionError("a `tight` config was accepted against a `safe` layout")


def test_active_config_follows_the_env_var_and_refuses_an_unknown_one():
    """The backend sizes the cache from this env var and the encoder fills it, so the two have to
    read it through the same name -- which is why it lives in `rope_store`, the module neither of
    them owns."""
    if not _need_cuda():
        return
    old = os.environ.get(CONFIG_ENV)
    try:
        os.environ.pop(CONFIG_ENV, None)
        assert enc.active_config() is CONFIGS["safe"], "the default is not `safe`"
        os.environ[CONFIG_ENV] = "tight"
        assert enc.active_config() is CONFIGS["tight"], "the env var was ignored"
        os.environ[CONFIG_ENV] = "not-an-allocation"
        try:
            enc.active_config()
        except RuntimeError as exc:
            assert CONFIG_ENV in str(exc)
            print(f"  refused: {str(exc).splitlines()[0][:110]}")
        else:
            raise AssertionError("an unknown allocation was accepted")
    finally:
        os.environ.pop(CONFIG_ENV, None)
        if old is not None:
            os.environ[CONFIG_ENV] = old


# --------------------------------------------------------------------------- the frozen path
def test_the_frozen_encode_agrees_with_the_tool_that_measured_the_fold_gap():
    """THE PROVENANCE CHECK. The fold knob is justified by `tools/measure_fold_gap.py`'s numbers --
    the fold costs 1.13x at chunk 4096 and 1.61x at chunk 512 -- and those numbers only describe the
    shipped path while the two implementations agree. So a drift on either side fails here rather
    than silently leaving a README quoting arithmetic nobody runs.

    The scenario is the tool's own: fit on a PREFIX, project the WHOLE block onto that frozen basis.

    THE RESIDUAL AGREEMENT IS ~2e-04 AND THAT IS THE RULER, not a drift. The shipped format stores
    each row's ruler as fp16 (`PackedRows.scale`), while the tool keeps it in fp32 -- so the tool's
    fold numbers are marginally optimistic, by a factor of ~500x less than the errors it was
    measuring. Worth knowing rather than worth fixing: it means a measured fold penalty of "1.13x"
    describes the shipped path to within a few tenths of a percent.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=128, H=H, d=D)
    K = _k_block(128)

    fitted = enc.fit_into_budget(K[:, :48].contiguous(), lay, cfg)
    fk = enc.encode_frozen(K, fitted.decoder, lay)
    mine = _roundtrip(lay, fk)
    theirs = _frozen_arm(K, fitted.decoder.decoder_f, cfg)

    agree = _rel(mine, theirs)
    print(f"  prefix fit r1={fitted.fitted_rank}/{lay.r1}; shipped vs tool agree to {agree:.3e}")
    print(f"  both vs the true keys: shipped {_rel(mine, K):.4f}, "
          f"one-shot re-fit {_rel(_roundtrip(lay, enc.fit_into_budget(K, lay, cfg)), K):.4f}")
    assert agree < 1e-3, (
        f"the shipped frozen encode and the tool that measured the fold gap disagree by {agree:.3e}. "
        f"One of them changed, and the measured fold penalty -- the thing the knob was chosen from -- "
        f"no longer describes the shipped path.")


def test_the_frozen_encode_is_the_least_squares_fit_for_the_frozen_basis():
    """A property that must hold by construction, and is worth asserting because the failure is
    quiet: projecting onto a request's OWN decoder must be at least as good as the fit that produced
    it, since the fit's coefficients are quantised TT coefficients while the projection's are the
    least-squares optimum for the same basis. A transposed or mis-scaled projection would be worse
    and would still look like a store.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=128, H=H, d=D)
    K = _k_block(96, seed=5)
    fitted = enc.fit_into_budget(K, lay, cfg)
    projected = enc.encode_frozen(K, fitted.decoder, lay)

    as_fit = _rel(_roundtrip(lay, fitted), K)
    as_proj = _rel(_roundtrip(lay, projected), K)
    print(f"  same basis, whole block: fit {as_fit:.4f} vs least-squares projection {as_proj:.4f}")
    assert as_proj <= as_fit + 1e-3, (
        f"projecting onto the fit's own decoder was WORSE than the fit ({as_proj:.4f} vs "
        f"{as_fit:.4f}). The projection is the least-squares optimum for that basis, so this means "
        f"the two are not solving against the same decoder.")


def test_the_frozen_encode_refuses_a_decoder_of_the_wrong_rank():
    """The slot's coefficient field is `r1` wide, so a decoder at another rank writes a row that
    cannot be read back -- and `check_layout_matches_decoder` on the read side would catch it only
    after the bytes were already in the cache."""
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    K = _k_block(32)
    try:
        enc.encode_frozen(K, enc.KDecoder(torch.zeros(lay.r1 + 4, H * D, device=DEV)), lay)
    except ValueError as exc:
        assert "cannot be read at another" in str(exc)
        print(f"  refused: {str(exc).splitlines()[0][:110]}")
        return
    raise AssertionError("a decoder of the wrong rank was accepted")


def test_pinv_is_the_right_shape_and_is_computed_once():
    """`pinv` is `[H*d, r1]` -- the TRANSPOSE of the decoder, not its shape. The encode is `M @ pinv`
    with `M` being `[n, H*d]`; the other order gives an object that cannot be broadcast into a slot.
    Caching matters because this is a `pinv` of an `[r1, H*d]` matrix and recomputing it per token
    would swamp the 0.009 ms/token it exists to serve."""
    if not _need_cuda():
        return
    dec = enc.KDecoder(torch.randn(32, H * D, device=DEV) * 0.1)
    p1 = dec.pinv
    assert p1.shape == (H * D, 32), f"pinv is {tuple(p1.shape)}, expected {(H * D, 32)}"
    assert dec.pinv is p1, "pinv was recomputed on the second access"
    direct = torch.linalg.pinv(dec.decoder_f)
    assert torch.equal(p1, direct), "the cached pinv is not the plain pseudo-inverse"
    print(f"  pinv {tuple(p1.shape)} cached; matches torch.linalg.pinv exactly")


# --------------------------------------------------------------------------- the write
def test_write_slots_round_trips_through_the_cache():
    """The write and the read have to agree about slot ids, and the convention is vLLM's own
    (`block * block_size + offset`). A private convention would be self-consistent and would read
    another request's tokens."""
    if not _need_cuda():
        return
    from vllm_ttkv import decode as dc

    lay = layout_for_config(CONFIGS["campaign4x"], r1=64, H=H, d=D)
    C = lay.compressed_bytes_per_token()
    n, block = 100, 16
    nblk = (n + block - 1) // block
    cache = torch.zeros(nblk, 1, block, C, dtype=torch.uint8, device=DEV)
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV)
    slots = dc.block_slots(bt, n, block)

    torch.manual_seed(7)
    k_slots = torch.randint(0, 256, (n, lay.k_slot_bytes), dtype=torch.uint8, device=DEV)
    v_slots = torch.randint(0, 256, (n, lay.v_slot_bytes), dtype=torch.uint8, device=DEV)
    enc.write_slots(cache, slots, k_slots, v_slots, lay)

    blob = dc.gather_slots(cache, slots)
    k_back, v_back = dc.split_blob(lay, blob)
    assert torch.equal(k_back, k_slots), "the K slot did not come back from the cache"
    assert torch.equal(v_back, v_slots), "the V slot did not come back from the cache"
    print(f"  n={n} over {nblk} blocks of {block}: slots round-tripped byte-exact "
          f"(K {lay.k_slot_bytes}B + V {lay.v_slot_bytes}B)")


def test_write_slots_refuses_a_pad_slot_and_a_wrong_sized_cache():
    """A padded token has no cache slot, and decode reads exactly `seq_len` rows -- so a key that
    was never stored does not shorten the read, it silently leaves a stale row inside the softmax
    denominator. And a cache sized for another layout would put V over the next token's
    coefficients."""
    if not _need_cuda():
        return
    lay = layout_for_config(CONFIGS["campaign4x"], r1=64, H=H, d=D)
    C = lay.compressed_bytes_per_token()
    n = 8
    k_slots = torch.zeros(n, lay.k_slot_bytes, dtype=torch.uint8, device=DEV)
    v_slots = torch.zeros(n, lay.v_slot_bytes, dtype=torch.uint8, device=DEV)

    cache = torch.zeros(2, 1, 16, C, dtype=torch.uint8, device=DEV)
    try:
        enc.write_slots(cache, torch.tensor([0, 1, -1, 3, 4, 5, 6, 7]), k_slots, v_slots, lay)
    except RuntimeError as exc:
        assert "PAD_SLOT_ID" in str(exc)
        print(f"  refused pad slot: {str(exc).splitlines()[0][:100]}")
    else:
        raise AssertionError("a PAD_SLOT_ID of -1 was accepted and wrote to the last slot")

    try:
        enc.write_slots(cache, torch.tensor([0, 1, 2, 3, 4, 5, 6, 99]), k_slots, v_slots, lay)
    except IndexError as exc:
        assert "outside the cache" in str(exc)
    else:
        raise AssertionError("a slot past the end of the cache was accepted")

    narrow = torch.zeros(2, 1, 16, C - 16, dtype=torch.uint8, device=DEV)
    try:
        enc.write_slots(narrow, torch.arange(n), k_slots, v_slots, lay)
    except ValueError as exc:
        assert "allocated for a different layout" in str(exc)
        print(f"  refused wrong layout: {str(exc).splitlines()[0][:100]}")
        return
    raise AssertionError("a cache sized for a different layout was accepted")


def test_as_kv_heads_accepts_both_shapes_and_refuses_a_wrong_one():
    """vLLM's models hand attention a 3D key but `Attention.forward` reshapes with
    `.view(-1, num_kv_heads, head_size)`, so a flat `[T, H*d]` reaches the impl too. A wrong reshape
    here mis-assigns heads and produces attention over permuted heads and nothing else."""
    if not _need_cuda():
        return
    flat = torch.randn(5, H * D, device=DEV)
    threed = flat.reshape(5, H, D)
    assert torch.equal(enc.as_kv_heads(flat, H, D), threed)
    assert torch.equal(enc.as_kv_heads(threed, H, D), threed)
    for bad in (torch.randn(5, H * D + 1, device=DEV), torch.randn(5, H + 1, D, device=DEV),
                torch.randn(5, D, 1, device=DEV)):
        try:
            enc.as_kv_heads(bad, H, D)
        except ValueError:
            continue
        raise AssertionError(f"accepted {tuple(bad.shape)} for H={H}, d={D}")
    print("  both shapes accepted; three wrong ones refused")


def test_the_fit_accepts_the_serving_dtype_and_still_runs_in_fp32():
    """vLLM hands the write path bf16 activations, and THE FIT MUST NOT RUN IN bf16.

    Found on the first real vLLM run, not here: `tt_svd` calls `torch.linalg.svd`, which has no bf16
    CUDA kernel, so the engine died with `NotImplementedError: "svd_cuda_gesvdj" not implemented for
    'BFloat16'` on the first prefill. That failure was loud. The reason the cast is REQUIRED rather
    than convenient is the quieter one: the rank decision is "dropped energy <= tol^2", a numerical
    judgement on a singular spectrum, and the calibration table -- hence every `r1`, every slot
    width and every ratio quoted from them -- was measured in fp32. Deciding ranks from a bf16
    spectrum would answer a slightly different question and return a plausible rank.

    Pinned by comparing against the SAME data cast explicitly, so this tests the cast rather than
    bf16's rounding: `fit(bfloat16(K))` has to be bit-identical to `fit(bfloat16(K).float())`.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    K = _k_block(48).to(torch.bfloat16)

    implicit = enc.fit_into_budget(K, lay, cfg)
    explicit = enc.fit_into_budget(K.float(), lay, cfg)

    assert implicit.fitted_rank == explicit.fitted_rank
    assert torch.equal(implicit.coeff_q, explicit.coeff_q)
    assert torch.equal(implicit.coeff_scale, explicit.coeff_scale)
    assert torch.equal(implicit.decoder.decoder_f, explicit.decoder.decoder_f)
    for field in ("bitmap", "packed", "scale"):
        assert torch.equal(getattr(implicit.resid, field), getattr(explicit.resid, field))

    # And V's lane selection, which is magnitude-ordered, takes the same cast.
    V = (torch.randn(H, 48, D, device=DEV) * 0.3).to(torch.bfloat16)
    assert torch.equal(pack_v_slots(lay, enc.encode_v(V, lay)),
                       pack_v_slots(lay, enc.encode_v(V.float(), lay)))

    for bad in (torch.zeros(H, 8, D, dtype=torch.int8, device=DEV),):
        try:
            enc.encode_v(bad, lay)
        except TypeError:
            continue
        raise AssertionError("an integer tensor was accepted as an activation")
    print(f"  bf16 in -> fp32 fit: rank {implicit.fitted_rank}, coefficients bit-identical")


def test_the_cache_is_read_as_bytes_even_when_vllm_allocates_it_in_bf16():
    """vLLM allocates the page in the MODEL dtype, so its last dim counts ELEMENTS while
    `state_content_bytes` counts BYTES. On the real run the 816-byte blob arrived as a 408-wide bf16
    page, and reading that width as a byte count reported the allocation as half the size the layout
    needed. The write must translate; the read must translate back.

    Both directions are checked on one page, because a reader that translated while the writer did
    not (or vice versa) would put the payload half a row away and still return the right shape.
    """
    if not _need_cuda():
        return
    from vllm_ttkv import decode as dc

    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    C = lay.compressed_bytes_per_token()
    assert C % 2 == 0, f"this fixture assumes an even blob width, got {C}"
    block, nblk = 16, 4
    # A bf16 page: exactly what `customize_spec`'s `state_content_bytes` produces once vLLM
    # converts it to model-dtype elements.
    page = torch.zeros(nblk, 1, block, C // 2, dtype=torch.bfloat16, device=DEV)
    slots = torch.randperm(nblk * block, device=DEV)[:24]

    k_slots = torch.randint(0, 255, (24, lay.k_slot_bytes), dtype=torch.uint8, device=DEV)
    v_slots = torch.randint(0, 255, (24, lay.v_slot_bytes), dtype=torch.uint8, device=DEV)
    enc.write_slots(page, slots, k_slots, v_slots, lay)

    blob = dc.gather_slots(page, slots, C)
    assert blob.shape == (24, C) and blob.dtype == torch.uint8
    k_back, v_back = dc.split_blob(lay, blob)
    assert torch.equal(k_back, k_slots) and torch.equal(v_back, v_slots), (
        "the round trip through a model-dtype page changed the bytes")
    print(f"  {C}B blob round-tripped through a bf16 page of width {C // 2}")


def test_encode_v_uses_the_layouts_lane_count_not_the_configs():
    """`k_lanes` in the codec is an ABSOLUTE lane count, not a fraction of `d`, so the one that
    matters is the one that sized the slot. Taking it from the config instead would agree today and
    diverge the moment a layout was built for a different geometry -- which is the exact trap
    `layout.TTKLayout.__post_init__` refuses."""
    if not _need_cuda():
        return
    cfg = CONFIGS["campaign4x"]
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    V = torch.randn(H, 48, D, device=DEV) * 0.3
    rows = enc.encode_v(V, lay)
    assert rows.rows == H * 48, f"V store holds {rows.rows} rows, expected {H * 48}"
    assert rows.k == lay.v_lanes, f"kept {rows.k} lanes, but the layout reserved {lay.v_lanes}"
    back = dense_v_from_slots(lay, pack_v_slots(lay, rows))
    print(f"  {lay.v_lanes} of {D} lanes at {lay.v_bits} bits: V rel err {_rel(back, V):.4f}")
    assert _rel(back, V) < 0.75, "the V store does not approximate V at all"


if __name__ == "__main__":
    print(f"encode validation (CUDA: {torch.cuda.is_available()})")
    for fn in (test_the_fit_pads_without_changing_the_reconstruction,
               test_a_binding_budget_is_the_accuracy_cost_and_it_is_monotone,
               test_check_cfg_matches_layout_refuses_a_foreign_allocation,
               test_active_config_follows_the_env_var_and_refuses_an_unknown_one,
               test_the_frozen_encode_agrees_with_the_tool_that_measured_the_fold_gap,
               test_the_frozen_encode_is_the_least_squares_fit_for_the_frozen_basis,
               test_the_frozen_encode_refuses_a_decoder_of_the_wrong_rank,
               test_pinv_is_the_right_shape_and_is_computed_once,
               test_write_slots_round_trips_through_the_cache,
               test_write_slots_refuses_a_pad_slot_and_a_wrong_sized_cache,
               test_as_kv_heads_accepts_both_shapes_and_refuses_a_wrong_one,
               test_the_fit_accepts_the_serving_dtype_and_still_runs_in_fp32,
               test_the_cache_is_read_as_bytes_even_when_vllm_allocates_it_in_bf16,
               test_encode_v_uses_the_layouts_lane_count_not_the_configs):
        print(f"{fn.__name__}:")
        fn()
    print("\nencode validation complete")
