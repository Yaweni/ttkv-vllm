"""Benchmark the shipped CUDA `krt` against the Triton `krt`, on IDENTICAL stores.

WHY r IS SWEPT RATHER THAN FIXED
    The reconstruction is a `[KT, r1] x [r1, d]` product per token tile, so its cost scales with r1 --
    and r1 is per layer and per corpus, measured 93..616 (mean 326) on prose at 16k. A benchmark at
    one rank says nothing about another. The first Triton number I took was 0.831 ms at r1=17, which
    is a rank-16 synthetic store and an order of magnitude below anything a real layer produces; the
    sweep is what stops that from being quoted as the comparison.

THREE THINGS THIS MEASURES, because they are not the same question
    1. AGREEMENT: both kernels against `reference.raw_scores`, and against each other, on one store.
       Two independent implementations (hand-written CUDA, Triton) agreeing with a torch reference is
       a much stronger claim than either matching alone.
    2. FULL CALL: `tkc.krt(...)` vs `krt_from_slots(...)`, i.e. what a serving loop would pay. This
       is the number that decides what ships.
    3. THE DECODER CONVERSION, reported separately because it is inside the CUDA call and not inside
       the Triton one: `tkc.krt` builds `dec_f = dec_q * dec_scale` on EVERY call (a transient, so
       that is not a leak, but it is work the Triton path does not do). Without that number the full
       call comparison is not interpretable.

ASCII only. `--gpu-architecture` is taken from the live device, so this does not hardcode sm_86.
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1] / "codec", _HERE.parent, _HERE.parent / "integration", pathlib.Path("/opt/ttkv/integration"),
           pathlib.Path("/opt/ttkv/vllm_ttkv_plugin")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
import ttkv_cuda as tkc  # noqa: E402
from vllm_ttkv.krt_triton import krt_from_slots  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots  # noqa: E402
from vllm_ttkv.reference import raw_scores, rope_tables  # noqa: E402

H, D, HQ = 8, 128, 32
G = HQ // H
N = 16384
CFG = "campaign4x"
RANKS = (16, 64, 128, 320)
WARM, REPS = 5, 20


def store(rank, n, cfgname=CFG, seed=0):
    """A store whose fitted rank is tunable: a rank-`rank` token subspace plus a noise floor."""
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    K = ((torch.randn(n, rank) * 0.2) @ (torch.randn(rank, D * H) * 0.2)
         ).reshape(n, H, D).permute(1, 0, 2).contiguous().cuda()
    K = K + 0.02 * torch.randn(H, n, D, device="cuda")
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    del K
    return cfg, kl


def timeit(fn, warm=WARM, reps=REPS):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    tkc.prime(HQ, D, G)
    cos, sin = rope_tables(torch.arange(N, device="cuda"), D)
    q = torch.randn(HQ, D, device="cuda") * 0.1

    print(f"n={N} cfg={CFG} H={H} d={D} HQ={HQ} G={G}  warm={WARM} reps={REPS}")
    print(f"{'r1':>5} {'CUDA ms':>9} {'Triton ms':>10} {'tri/cuda':>9} "
          f"{'CUDA relerr':>12} {'TRI relerr':>11} {'CUDA-TRI':>9} {'dec_f ms':>9}")

    for rank in RANKS:
        _cfg, kl = store(rank, N)
        lay = layout_for_config(CONFIGS[CFG], r1=kl.r1, H=H, d=D)
        ks = pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid)
        dec_f = kl.decoder_f()

        # ---- agreement, on one store, before any timing
        s_ref = raw_scores(lay, ks, dec_f, q, cos, sin, scale=1.0).T
        s_cuda = tkc.krt(kl.coeff_q, kl.coeff_scale, kl.dec_q, kl.dec_scale, kl.resid,
                         cos, sin, q, N, HQ, G, D)
        s_tri = krt_from_slots(lay, ks, dec_f, cos, sin, q)
        den = s_ref.abs().max()
        e_cuda = ((s_cuda - s_ref).abs().max() / den).item()
        e_tri = ((s_tri - s_ref).abs().max() / den).item()
        e_x = ((s_cuda - s_tri).abs().max() / den).item()

        t_cuda = timeit(lambda: tkc.krt(kl.coeff_q, kl.coeff_scale, kl.dec_q, kl.dec_scale,
                                        kl.resid, cos, sin, q, N, HQ, G, D))
        t_tri = timeit(lambda: krt_from_slots(lay, ks, dec_f, cos, sin, q))
        # the CUDA launcher's per-call decoder build, measured on its own
        t_dec = timeit(lambda: (kl.dec_q.to(torch.float32)
                                * kl.dec_scale.to(torch.float32).unsqueeze(1)).contiguous())

        print(f"{kl.r1:>5} {t_cuda:>9.3f} {t_tri:>10.3f} {t_tri / t_cuda:>9.2f} "
              f"{e_cuda:>12.3e} {e_tri:>11.3e} {e_x:>9.3e} {t_dec:>9.3f}", flush=True)
        del kl, ks, dec_f, s_ref, s_cuda, s_tri
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
