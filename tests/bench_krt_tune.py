"""How fast is `krt`, really, and which launch shape gets it there?

`tests/bench_krt.py` answers a different question -- Triton against the shipped CUDA kernel, at the
real ranks. This one holds the math fixed and sweeps the LAUNCH SHAPE, because the reconstruction
runs at a fraction of what the card can do and the reason is a register-pressure question rather than
an arithmetic one.

WHAT THE KERNEL HOLDS LIVE AT ONCE, which is the whole story:

    acc_lo, acc_hi   [KT, HALF] fp32    two accumulators, the reconstruction result
    g                [KT, BR]   fp32    the coefficient block being multiplied in
    the contraction  [G, KT, HALF]      broadcast-and-sum, in the single-query kernel

That last one is 4 x KT x 64 elements -- at KT=64 it is 16,384 floats, 64 KB of live registers for
one program, against a card whose whole register file per SM is 256 KB. It is why the single-query
path is expected to lose to the multi-query one even at Q=1, since that one contracts with `tl.dot`
at `M = GP = 16` and never builds the intermediate. This script measures that rather than asserting
it.

REPORTED AS A FRACTION OF PEAK, because milliseconds on their own do not say whether there is
anything to win. The MAC count is exact and stated in the code below: `n * H * r1 * D` per call for
the reconstruction, which is `2 * n * H * r1 * D` FLOPs.

USAGE

    PYTHONPATH=/opt/ttkv/integration:/opt/ttkv/vllm_ttkv_plugin \
      python tests/bench_krt_tune.py --n 2048 16384 --r1 110
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv import encode  # noqa: E402
from vllm_ttkv.krt_triton import krt_from_slots, krt_multi_from_slots  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots  # noqa: E402

H, D, HQ = 8, 128, 32                      # G = 4, the Llama-3.x geometry
DEV = "cuda"
# A4000, fp32 FMA. Not a guess and not the card's marketing number: the kernel is fp32 through and
# through (`input_precision="ieee"`), so this is the ceiling it is measured against.
PEAK_TFLOPS = 19.2


def _store(cfgname: str, n: int, r1: int, seed: int = 0):
    """A real store: fitted decoder, packed coefficients, packed slots.

    Built through `fit_into_budget` rather than assembled by hand, because the kernel's addressing
    depends on the layout's own byte arithmetic (`coeff_bytes`, the residual stream, the odd-offset
    ruler) and a hand-built fixture would test a different format.
    """
    cfg = CONFIGS[cfgname]
    lay = layout_for_config(cfg, r1=r1, H=H, d=D)
    torch.manual_seed(seed)
    rank = 32
    K = ((torch.randn(n, rank, device=DEV) * 0.2)
         @ (torch.randn(rank, H * D, device=DEV) * 0.2)).reshape(n, H, D).permute(1, 0, 2)
    K = (K + 0.02 * torch.randn(H, n, D, device=DEV)).contiguous()
    fk = encode.fit_into_budget(K, lay, cfg)
    slots = pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid)
    return lay, slots, fk.decoder.decoder_f.contiguous()


def _tables(lay, n, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=DEV).float() / D))
    fr = torch.outer(torch.arange(n, device=DEV).float(), inv)
    cos = torch.cat([fr.cos(), fr.cos()], dim=-1)
    sin = torch.cat([fr.sin(), fr.sin()], dim=-1)
    return cos.contiguous(), sin.contiguous(), torch.randn(HQ, D, device=DEV, generator=g)


def _ms(fn, warm=3, reps=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    import time
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _peak_pct(ms, n, r1):
    flops = 2.0 * n * H * r1 * D
    return flops / (ms * 1e-3) / (PEAK_TFLOPS * 1e12) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[2048, 16384])
    ap.add_argument("--r1", type=int, nargs="+", default=[110])
    ap.add_argument("--cfg", default="safe")
    ap.add_argument("--kts", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--brs", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--warps", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()

    print(f"  {PEAK_TFLOPS} TFLOPS fp32 peak assumed; MACs = n*H*r1*d, two halves\n")
    for n, r1 in itertools.product(args.n, args.r1):
        lay, slots, dec = _store(args.cfg, n, r1)
        cos, sin, q = _tables(lay, n)
        q2 = q  # [HQ, d] for the single-query kernel
        qm = q.reshape(H, HQ // H, D).permute(1, 0, 2).reshape(HQ, 1, D).contiguous()  # [HQ,1,d]

        rows = []
        for kt, br, w in itertools.product(args.kts, args.brs, args.warps):
            if br > r1:
                continue
            try:
                t = _ms(lambda: krt_from_slots(lay, slots, dec, cos, sin, q2, kt=kt, br=br,
                                               num_warps=w))
                rows.append((t, f"single kt={kt:<3d} br={br:<3d} warps={w}", t))
            except Exception as e:
                print(f"    single kt={kt} br={br} warps={w}: {type(e).__name__} {str(e)[:70]}")
        # The multi-query kernel at Q=1: the same math through `tl.dot` instead of broadcast-and-sum.
        for kt, bq, w in itertools.product(args.kts, [16], args.warps):
            try:
                t = _ms(lambda: krt_multi_from_slots(lay, slots, dec, cos, sin, qm, kt=kt,
                                                     bq=bq, num_warps=w))
                rows.append((t, f"multi(Q=1) kt={kt:<3d} bq={bq:<3d} warps={w}", t))
            except Exception as e:
                print(f"    multi kt={kt} bq={bq} warps={w}: {type(e).__name__} {str(e)[:70]}")

        rows.sort()
        print(f"  === n={n} r1={r1} cfg={args.cfg} ===")
        for t, label, _ in rows[:12]:
            print(f"    {label:<30s} {t:8.4f} ms   {_peak_pct(t, n, r1):5.1f}% of peak")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
