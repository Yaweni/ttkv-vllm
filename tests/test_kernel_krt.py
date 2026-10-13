"""Does the SHIPPED pre-rotary kernel agree with the reference decode?

This is HANDOFF acceptance criterion 7, run against the real thing: `ttkv_krt` compiled by NVRTC
from the shipped `ttkv_kernels.cu`, compared against `vllm_ttkv.reference`. Both read the SAME
store, so agreement checks the kernel's whole chain -- reconstruct, patch the residual, rotate each
row by its own position, contract -- rather than any one step.

Why it matters for the port: `krt` is one of only two kernels the pre-rotary design needs (`qt` and
`resid_add` belong to the absorbed path and are dead). If it matches the reference here, then moving
its body into `csrc/` and swapping NVRTC for AOT nvcc is a delivery change, not a numerics change --
and the delivery change is the part that cannot be unit-tested anywhere.

Skips cleanly when there is no CUDA device or no `ttkv_cuda` binding, so it is safe in a CPU-only
run. It needs the source repo's modules importable (`ttkv_codec`, `ttkv_packed_cache`,
`ttkv_cuda`).
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
# Support BOTH layouts: <root>/vllm_ttkv_plugin with the codec in <root>/integration (the project),
# and a flat checkout where the codec sits beside the plugin (the GPU box).
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots  # noqa: E402
from vllm_ttkv.reference import raw_scores, rope_tables  # noqa: E402

try:
    import ttkv_cuda as tkc
except Exception:                                        # noqa: BLE001
    tkc = None

H, D, HQ, N = 8, 128, 32, 96
G = HQ // H
DEV = "cuda"


def available() -> bool:
    if tkc is None or not torch.cuda.is_available():
        return False
    try:
        return bool(tkc.kernels_available())
    except Exception:                                    # noqa: BLE001
        return False


def _store(dev=DEV, n=N, seed=0):
    """A realistic K store: low-rank along tokens plus a noise floor, on `dev`."""
    torch.manual_seed(seed)
    cfg = CONFIGS["campaign4x"]
    rank = 16
    K = ((torch.randn(n, rank) * 0.2) @ (torch.randn(rank, D * H) * 0.2)
         ).reshape(n, H, D).permute(1, 0, 2).contiguous().to(dev)
    K = K + 0.02 * torch.randn(H, n, D, device=dev)
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    return cfg, K, kl


def test_krt_matches_the_reference_decode():
    if not available():
        print("  SKIPPED: no CUDA device or no ttkv_cuda binding on this machine")
        return
    dev = DEV
    cfg, _K, kl = _store(dev)
    tkc.prime(HQ, D, G)

    cos, sin = rope_tables(torch.arange(N, device=dev), D)
    q = torch.randn(HQ, D, device=dev) * 0.1                  # post-rotary; the kernel takes it as-is

    # The kernel: reconstruct + residual + per-row rotation + contract, on device.
    s_kernel = tkc.krt(kl.coeff_q, kl.coeff_scale, kl.dec_q, kl.dec_scale, kl.resid,
                       cos, sin, q, N, HQ, G, D)              # [HQ, n]
    assert tuple(s_kernel.shape) == (HQ, N), f"krt returned {tuple(s_kernel.shape)}"

    # The reference, over the SAME store, reached through the slot layout so the packing is on the
    # critical path too. `scale=1.0` because `attend_container` applies the softmax scale OUTSIDE
    # the kernel, so the kernel's scores are unscaled.
    lay = layout_for_config(cfg, r1=kl.r1, H=H, d=D)
    ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
    s_ref = raw_scores(lay, ks, kl.decoder_f(), q, cos, sin, scale=1.0).T     # [HQ, n]

    rel = ((s_kernel - s_ref).abs().max() / s_ref.abs().max()).item()
    print(f"  krt: r1={kl.r1} max|score|={s_ref.abs().max():.4f}  rel err vs reference {rel:.3e}")
    assert rel < 1e-4, (
        f"the shipped kernel disagrees with the reference by {rel:.3e}. Every step of the chain "
        f"(reconstruct, residual, per-row rotation, contract) is a place this can go wrong while "
        f"still returning correctly-shaped scores")


def test_krt_rotation_is_per_row_and_not_constant():
    """Negative control on the DEVICE side: if `krt` rotated every row by the same angle, the test
    above would still pass. Rolling the angle table must change the kernel's output materially."""
    if not available():
        print("  SKIPPED: no CUDA device or no ttkv_cuda binding on this machine")
        return
    dev = DEV
    _cfg, _K, kl = _store(dev)
    tkc.prime(HQ, D, G)

    cos, sin = rope_tables(torch.arange(N, device=dev), D)
    q = torch.randn(HQ, D, device=dev) * 0.1

    def run(c, s):
        return tkc.krt(kl.coeff_q, kl.coeff_scale, kl.dec_q, kl.dec_scale, kl.resid,
                       c, s, q, N, HQ, G, D)

    good, rolled = run(cos, sin), run(cos.roll(1, 0), sin.roll(1, 0))
    rel = ((good - rolled).abs().max() / good.abs().max()).item()
    print(f"  neighbour-position rotation moves the kernel's scores by {rel:.3f} (relative)")
    assert rel > 1e-2, (
        f"rolling the angle table changed the kernel output by only {rel:.2e}, so the rotation is "
        f"not being applied per row and the agreement test above is vacuous")


if __name__ == "__main__":
    print(f"krt kernel validation (CUDA available: {torch.cuda.is_available()}, "
          f"binding usable: {available()})")
    for fn in (test_krt_matches_the_reference_decode, test_krt_rotation_is_per_row_and_not_constant):
        print(f"{fn.__name__}:")
        fn()
    print("\nkernel validation complete")
