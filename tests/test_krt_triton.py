"""Does the TRITON `krt` agree with the reference decode, and is the rotation really per row?

The same two gates `tests/test_kernel_krt.py` applies to the shipped CUDA kernel, but driven through
the SLOT path -- `layout.pack_k_slots` -> `krt_triton.krt_from_slots` -- because the slot layout is
what the vLLM cache actually holds. The CUDA test reads the codec's own container; this one reads
what the plugin will read.

Why a SEPARATE file rather than a parameter on the existing one: the two kernels are compared
against the same oracle here, so the numbers are directly comparable and neither test can pass by
accident of the other's packing.

Skips cleanly without a CUDA device, so it is safe in a CPU-only run.
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.krt_triton import krt_from_slots  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots, packed_row_bytes  # noqa: E402
from vllm_ttkv.reference import raw_scores, rope_tables  # noqa: E402

H, D, HQ = 8, 128, 32
DEV = "cuda"


def _store(cfgname, n, dev=DEV, seed=0):
    """A realistic K store: low-rank along tokens plus a noise floor, so the residual path is live
    and the fit is not trivially exact."""
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    rank = 16
    K = ((torch.randn(n, rank) * 0.2) @ (torch.randn(rank, D * H) * 0.2)
         ).reshape(n, H, D).permute(1, 0, 2).contiguous().to(dev)
    K = K + 0.02 * torch.randn(H, n, D, device=dev)
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    return cfg, kl


def test_triton_matches_the_reference_for_every_shipped_config():
    """Every config, because they differ in exactly the fields the kernel parses: `res_k` sets PB
    and the row stride, `res_bits` sets the code width and whether the ruler lands on an odd byte."""
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return
    for cfgname in ("safe", "tight", "campaign4x"):
        for n in (96, 100, 4096):
            _cfg, kl = _store(cfgname, n)
            lay = layout_for_config(CONFIGS[cfgname], r1=kl.r1, H=H, d=D)
            ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
            cos, sin = rope_tables(torch.arange(n, device=DEV), D)
            q = torch.randn(HQ, D, device=DEV) * 0.1

            s_tri = krt_from_slots(lay, ks, kl.decoder_f(), cos, sin, q)
            s_ref = raw_scores(lay, ks, kl.decoder_f(), q, cos, sin, scale=1.0).T

            assert tuple(s_tri.shape) == (HQ, n), f"got {tuple(s_tri.shape)}, want {(HQ, n)}"
            rel = ((s_tri - s_ref).abs().max() / s_ref.abs().max()).item()
            print(f"  {cfgname:<11s} n={n:<5d} r1={kl.r1:<4d} res_k={lay.res_k:<3d} "
                  f"bits={lay.res_bits} PB={packed_row_bytes(lay.res_k, lay.res_bits):<3d} "
                  f"slot={lay.k_slot_bytes:<4d}  rel err {rel:.3e}")
            assert rel < 1e-4, (
                f"{cfgname} at n={n}: Triton disagrees with the reference by {rel:.3e}. Every step "
                f"(reconstruct, residual, per-row rotation, contract) is a place this can go wrong "
                f"while still returning correctly-shaped scores")


def test_triton_rotation_is_per_row_and_not_constant():
    """Negative control. If the kernel rotated every row by one angle, the test above would still
    pass -- so rolling the angle table has to move the output materially."""
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return
    n = 512
    _cfg, kl = _store("campaign4x", n)
    lay = layout_for_config(CONFIGS["campaign4x"], r1=kl.r1, H=H, d=D)
    ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, D, device=DEV) * 0.1

    good = krt_from_slots(lay, ks, kl.decoder_f(), cos, sin, q)
    rolled = krt_from_slots(lay, ks, kl.decoder_f(), cos.roll(1, 0), sin.roll(1, 0), q)
    rel = ((good - rolled).abs().max() / good.abs().max()).item()
    print(f"  neighbour-position rotation moves the scores by {rel:.3f} (relative)")
    assert rel > 1e-2, (
        f"rolling the angle table changed the output by only {rel:.2e}, so the rotation is not "
        f"being applied per row and the agreement test above is vacuous")


def test_triton_skipping_the_rotation_changes_the_answer():
    """The mistake `ttkv_attn._refuse_pre_storage` exists to prevent: contracting a post-rotary
    query against an UN-rotated key. It is finite, correctly shaped and wrong, so it is worth an
    explicit number."""
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return
    n = 512
    _cfg, kl = _store("campaign4x", n)
    lay = layout_for_config(CONFIGS["campaign4x"], r1=kl.r1, H=H, d=D)
    ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, D, device=DEV) * 0.1

    # cos=1, sin=0 is the identity rotation, so this arm contracts a post-rotary query against an
    # UN-rotated key -- exactly the mistake, and it needs no separate code path to produce.
    rotated = krt_from_slots(lay, ks, kl.decoder_f(), cos, sin, q)
    unrotated = krt_from_slots(lay, ks, kl.decoder_f(),
                               torch.ones(n, D, device=DEV), torch.zeros(n, D, device=DEV), q)
    rel = ((rotated - unrotated).abs().max() / rotated.abs().max()).item()
    print(f"  omitting the rotation changes the scores by {rel:.3f} (relative)")
    assert rel > 1e-2, f"skipping the rotation moved the scores by only {rel:.2e}"


def test_compiles_for_every_rank_class():
    """Triton compiles a SEPARATE VARIANT for integer arguments divisible by 16, and on Triton 3.2
    that variant aborted the compiler outright (LLVM ArrayRef assertion, no diagnostic). The failure
    is per-VALUE: r1=32 crashes while r1=33 is fine, so a single-rank test can pass green while a
    real run fires intermittently on whatever round number the layer's rank happens to be. That is
    why this sweep exists rather than relying on the store the other test happens to build.
    """
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return
    from vllm_ttkv.layout import TTKLayout
    for r1 in (16, 32, 33, 61, 64, 96, 128):
        lay = TTKLayout(H=H, d=D, r1=r1, qbits=8, res_k=64, res_bits=4, v_lanes=64, v_bits=4)
        n = 128
        slots = torch.zeros(n, lay.k_slot_bytes, dtype=torch.uint8, device=DEV)
        dec = torch.zeros(r1, H * D, device=DEV)
        cos, sin = rope_tables(torch.arange(n, device=DEV), D)
        s = krt_from_slots(lay, slots, dec, cos, sin, torch.zeros(HQ, D, device=DEV))
        assert tuple(s.shape) == (HQ, n), f"r1={r1}: got {tuple(s.shape)}"
        print(f"  r1={r1:<4d} slot={lay.k_slot_bytes:<4d} compiled and ran")


def _bench():
    if not torch.cuda.is_available():
        return
    n = 16384
    _cfg, kl = _store("campaign4x", n)
    lay = layout_for_config(CONFIGS["campaign4x"], r1=kl.r1, H=H, d=D)
    ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
    dec = kl.decoder_f()
    cos, sin = rope_tables(torch.arange(n, device=DEV), D)
    q = torch.randn(HQ, D, device=DEV) * 0.1

    for _ in range(5):
        krt_from_slots(lay, ks, dec, cos, sin, q)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        krt_from_slots(lay, ks, dec, cos, sin, q)
    torch.cuda.synchronize()
    print(f"  n={n} r1={kl.r1}: {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms/call")


if __name__ == "__main__":
    print(f"triton krt validation (CUDA available: {torch.cuda.is_available()})")
    for fn in (test_triton_matches_the_reference_for_every_shipped_config,
               test_triton_rotation_is_per_row_and_not_constant,
               test_triton_skipping_the_rotation_changes_the_answer,
               test_compiles_for_every_rank_class):
        print(f"{fn.__name__}:")
        fn()
    print("timing (one layer, one token step):")
    _bench()
    print("\ntriton kernel validation complete")
