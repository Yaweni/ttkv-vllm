"""Does the MULTI-QUERY `krt` agree with the reference, and is the amortisation real?

The single-query kernel cannot express a prompt: it contracts one query row per (kv head, group) and
returns `[HQ, n]`. A prefill chunk has thousands of query tokens, which is why `_prefill` is still
the torch `attend_prefill`, and why batch sizes above 1 are refused (a continuation chunk needs the
same kernel). This is that kernel.

WHAT IS ACTUALLY NEW, and therefore what these tests have to be suspicious of: steps 1-3
(reconstruct, patch the residual, rotate by the row's own position) do not depend on the query, so a
row block is reconstructed ONCE and reused for every query in the block. That reuse is the entire
point -- it is what takes the reconstruction from 27x the contraction down to a few percent -- and it
is also where the indexing can go wrong. A row reused for the WRONG query, or a block boundary that
re-reads the wrong rows, still produces correctly shaped, plausible scores. So the tests are:

  * agreement with `reference.raw_scores_multi` over the finite entries, and an exact match on WHICH
    entries are finite, so a kernel that silently dropped the causal mask would fail here;
  * the causal pattern checked against `row <= p0 + j` directly, not merely "some values are -inf";
  * the tiling checked against ITSELF across `bq` and `kt` -- the answer must not depend on the block
    sizes, which is the property that catches a boundary error;
  * Q=1 must reproduce the single-query kernel, so the two kernels cannot disagree silently;
  * two negative controls: rolling the angle table has to move the output (the rotation is per row),
    and turning the mask off has to change it (the mask is doing something).

Skips cleanly without a CUDA device.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2],
              _HERE.parents[2] / "integration", pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.krt_triton import krt_from_slots, krt_multi_from_slots  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots, packed_row_bytes  # noqa: E402
from vllm_ttkv.reference import raw_scores, raw_scores_multi, rope_tables  # noqa: E402

H, D, HQ = 8, 128, 32                     # G=4, the Llama-3.1-8B geometry
CFGS = ("safe", "tight")
DEV = "cuda"
G = HQ // H


def _need_cuda() -> bool:
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def _store(cfgname, n, seed=0):
    """A realistic K store: low-rank along tokens plus a noise floor, so the residual path is live."""
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    K = ((torch.randn(n, 16) * 0.2) @ (torch.randn(16, D * H) * 0.2)
         ).reshape(n, H, D).permute(1, 0, 2).contiguous().to(DEV)
    K = K + 0.02 * torch.randn(H, n, D, device=DEV)
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    lay = layout_for_config(cfg, r1=kl.r1, H=H, d=D)
    return lay, pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid), kl.decoder_f()


def _rel(got, want):
    """Relative error over the entries BOTH tensors call finite, plus the mask agreement."""
    fin = torch.isfinite(want)
    assert bool((torch.isfinite(got) == fin).all()), (
        "the kernel and the reference disagree about which entries are masked, so the causal mask "
        "was dropped or applied to the wrong axis")
    if not bool(fin.any()):
        return 0.0
    a, b = got[fin], want[fin]
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("cfgname", CFGS)
@pytest.mark.parametrize("n,q_len", [(96, 5), (100, 17), (128, 64), (256, 32)])
def test_multi_query_agrees_with_the_reference(cfgname, n, q_len):
    """Scores for `q_len` query tokens against `n` stored rows, causal, against the torch oracle."""
    if not _need_cuda():
        return
    lay, ks, dec = _store(cfgname, n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, q_len, D, device=DEV) * 0.1
    p0 = n - q_len                                   # the chunk ends at the last stored row

    got = krt_multi_from_slots(lay, ks, dec, cos, sin, q, p0=p0)
    want = raw_scores_multi(lay, ks, dec, q, cos, sin, scale=1.0, p0=p0)

    assert tuple(got.shape) == (HQ, q_len, n), f"got {tuple(got.shape)}"
    rel = _rel(got, want)
    print(f"  {cfgname:<6s} n={n:<4d} Q={q_len:<3d} r1={lay.r1:<4d} res_k={lay.res_k:<3d} "
          f"PB={packed_row_bytes(lay.res_k, lay.res_bits):<3d}  rel err {rel:.3e}")
    assert rel < 1e-4, (
        f"{cfgname} at n={n}, Q={q_len}: the multi-query kernel disagrees with the reference by "
        f"{rel:.3e}. Reconstruct, residual, per-row rotation and the QUERY REUSE are each places "
        f"this can go wrong while still returning correctly shaped scores")


def test_the_causal_pattern_is_exactly_row_le_position():
    """Not "some entries are -inf" but the pattern itself: query j may see rows `0 .. p0 + j`."""
    if not _need_cuda():
        return
    n, q_len = 100, 17
    lay, ks, dec = _store("safe", n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, q_len, D, device=DEV) * 0.1
    p0 = n - q_len

    got = krt_multi_from_slots(lay, ks, dec, cos, sin, q, p0=p0)
    rows = torch.arange(n, device=DEV)[None, None, :]
    pos = (p0 + torch.arange(q_len, device=DEV))[None, :, None]
    allowed = (rows <= pos).expand(HQ, q_len, n)
    assert bool((torch.isfinite(got) == allowed).all()), (
        "the causal pattern does not match `row <= p0 + j`; a mask on the wrong axis would let a "
        "query attend its own future, which is finite and plausible and wrong")
    # And the newest query must see every row, or the mask is off by something.
    assert bool(torch.isfinite(got[:, -1, :]).all()), "the last query should see the whole store"
    print(f"  causal pattern exact for n={n}, Q={q_len}, p0={p0}; "
          f"last query sees all {n} rows")


def test_the_answer_does_not_depend_on_the_block_sizes():
    """The tiling must be invisible. A boundary error re-reads the wrong rows and changes the answer,
    and it is the failure a single (kt, bq) would not catch."""
    if not _need_cuda():
        return
    n, q_len = 200, 48
    lay, ks, dec = _store("safe", n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, q_len, D, device=DEV) * 0.1
    p0 = n - q_len

    ref = krt_multi_from_slots(lay, ks, dec, cos, sin, q, kt=32, bq=16, p0=p0)
    for kt, bq in ((16, 16), (64, 16), (32, 32), (16, 32)):
        got = krt_multi_from_slots(lay, ks, dec, cos, sin, q, kt=kt, bq=bq, p0=p0)
        fin = torch.isfinite(ref)
        assert bool((torch.isfinite(got) == fin).all()), f"kt={kt} bq={bq} changed the mask"
        rel = ((got[fin] - ref[fin]).abs().max() / ref[fin].abs().max()).item()
        print(f"  kt={kt:<4d} bq={bq:<3d} vs kt=32 bq=16: rel err {rel:.3e}")
        assert rel < 1e-5, (
            f"kt={kt}, bq={bq} disagrees with kt=32, bq=16 by {rel:.3e}, so the block sizes are "
            f"visible in the answer -- a tile is reading the wrong rows")


def test_one_query_reproduces_the_single_query_kernel():
    """The two kernels must not be able to disagree silently. With Q=1 and no mask, the multi-query
    kernel is doing the same work, so this is a cross-check of the shared steps 1-3 against a kernel
    that is already tested against `raw_scores`."""
    if not _need_cuda():
        return
    for cfgname in CFGS:
        n = 128
        lay, ks, dec = _store(cfgname, n, seed=3)
        cos, sin = rope_tables(torch.arange(n, device=DEV), D)
        q1 = torch.randn(HQ, D, device=DEV) * 0.1

        one = krt_from_slots(lay, ks, dec, cos, sin, q1)                  # [HQ, n]
        many = krt_multi_from_slots(lay, ks, dec, cos, sin, q1[:, None, :])[..., 0, :]
        rel = ((many - one).abs().max() / one.abs().max()).item()
        print(f"  {cfgname:<6s} multi-query Q=1 vs single-query kernel: rel err {rel:.3e}")
        assert rel < 1e-5, (
            f"{cfgname}: the two kernels disagree by {rel:.3e} on the same store and query, so one "
            f"of them is wrong and the prefill and decode paths would return different answers")


def test_rotation_is_still_per_row_under_a_query_block():
    """Negative control. If the kernel rotated every row by one angle, the agreement tests above
    would still pass -- so rolling the angle table has to move the output materially."""
    if not _need_cuda():
        return
    n, q_len = 128, 32
    lay, ks, dec = _store("safe", n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, q_len, D, device=DEV) * 0.1

    good = krt_multi_from_slots(lay, ks, dec, cos, sin, q)
    rolled = krt_multi_from_slots(lay, ks, dec, cos.roll(1, 0), sin.roll(1, 0), q)
    rel = ((good - rolled).abs().max() / good.abs().max()).item()
    print(f"  neighbour-position rotation moves the scores by {rel:.3f} (relative)")
    assert rel > 1e-2, (
        f"rolling the angle table moved the output by only {rel:.2e}, so the rotation is not being "
        f"applied per row and the agreement tests are vacuous")


def test_the_mask_changes_the_probabilities_not_just_the_masking():
    """The other negative control: if `p0` were ignored, the attention weights would spread over the
    future.

    THE FIRST VERSION OF THIS TEST WAS WRONG, and the way it was wrong is worth keeping. It compared
    the SCORES of the entries that stay finite -- and those are the same numbers either way, because
    the mask only turns FUTURE entries into `-inf`; it does not touch the ones that survive. So it
    measured zero and looked like the mask was not reaching the kernel, when in fact the causal
    pattern was already exact (see the test above). What the mask actually changes is the softmax
    DENOMINATOR, so the denominator is what this compares.
    """
    if not _need_cuda():
        return
    n, q_len = 128, 32
    lay, ks, dec = _store("safe", n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, q_len, D, device=DEV) * 0.1

    causal = krt_multi_from_slots(lay, ks, dec, cos, sin, q, p0=n - q_len)
    free = krt_multi_from_slots(lay, ks, dec, cos, sin, q)
    assert bool(torch.isfinite(free).all()), "the unmasked call should have no -inf"
    assert not bool(torch.isfinite(causal).all()), "the masked call should have some -inf"

    p_causal = torch.softmax(causal, dim=-1)
    p_free = torch.softmax(free, dim=-1)
    rel = ((p_causal - p_free).abs().max() / p_free.abs().max()).item()
    print(f"  masking moves the attention probabilities by {rel:.3f} (relative)")
    assert rel > 1e-2, (
        f"the mask changed the probabilities by only {rel:.2e}, so `p0` is not reaching the kernel")

    # And the future must carry exactly zero weight, which is what -inf has to mean once softmaxed.
    assert float(p_causal[~torch.isfinite(causal)].abs().max()) == 0.0, (
        "a masked entry carries non-zero probability, so the -inf was not preserved")
    # The probability mass has to go SOMEWHERE: normalised rows must still sum to one.
    assert bool(torch.allclose(p_causal.sum(dim=-1), torch.ones_like(p_causal.sum(dim=-1))))


def test_the_reconstruction_is_amortised_over_the_query_block():
    """The reason this kernel exists, measured rather than asserted in prose.

    A single query pays `r1*d` per row once; `Q` queries in blocks of `bq` pay it `ceil(Q/bq)` times
    rather than `Q` times, so the cost should grow with the NUMBER OF BLOCKS and not with `Q`. This
    asserts a loose bound (not a timing target) and prints the shape of the curve, because the point
    is the shape: if the reconstruction were being redone per query, the ratio would be ~Q.
    """
    if not _need_cuda():
        return
    n, r1_lay = 1024, None
    lay, ks, dec = _store("safe", n)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    torch.manual_seed(0)

    def timed(fn, warm=3, reps=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    base = timed(lambda: krt_from_slots(lay, ks, dec, cos, sin,
                                        torch.randn(HQ, D, device=DEV) * 0.1))
    print(f"  single query, n={n}: {base:.3f} ms")
    for q_len in (16, 64, 256):
        q = torch.randn(HQ, q_len, D, device=DEV) * 0.1
        ms = timed(lambda q=q: krt_multi_from_slots(lay, ks, dec, cos, sin, q))
        print(f"  Q={q_len:<4d} bq=16 -> {ms:7.3f} ms   {ms / base:6.2f}x one query"
              f"   ({q_len}x queries, {q_len // 16} blocks)")
        assert ms < base * q_len / 2, (
            f"Q={q_len} cost {ms / base:.2f}x a single query, close to {q_len}x, so the "
            f"reconstruction is being repeated per query instead of per query BLOCK")
