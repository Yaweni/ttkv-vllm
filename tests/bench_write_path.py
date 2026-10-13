"""What the WRITE PATH costs, both arms, at engine shapes -- and why `VLLM_TTKV_PROFILE` is not it.

`VLLM_TTKV_PROFILE=1` times host wall-clock per phase, and it earned its place: it is what showed the
decode's cost was ~6 device readbacks per layer-call rather than anything to do with kernels. But
once the step is deep enough that the GPU queue never drains, a phase's host time is the queue
draining, not the work being submitted -- measured here, the fused write is 0.34 ms a layer while the
engine's profile attributes 1.43 ms to the same call. Both numbers are true statements about
different things, and only the first one tells you what to optimise.

So this measures throughput the way a step experiences it: many calls back to back, ONE sync at the
end, and the arms compared in the same process on the same tensors. The cache is a bf16 page of the
shape vLLM allocates, so `decode.byte_rows` does its real translation rather than being handed a byte
tensor.

WHAT IT WAS USED FOR, and the numbers it produced on an A4000 (sm_86, WSL2), on a bf16 page:

    decode shapes (1 token, H=8, d=128, r1=112)
      fused (Triton)             0.307 ms/layer-call     8.6 ms/step over 28 layers
      torch (PackedRows + slots) 1.794 ms/layer-call    50.2 ms/step
      -> 5.84x, and the decode step went 94 ms to 79 ms (the 16-token run 5.50 s to 5.32 s)
    a 1082-token chunk, same arms: 0.710 against 2.196 ms/layer-call, 3.09x
      (that path is the frozen encode; a request's FIRST chunk is the fit, and it stays torch)

THE SHAPE OF THE REMAINING 0.307, which is what `_fixed_cost` measures and why it matters: at decode
the write runs once per token per layer, so 28 times a step, and the arithmetic it stands for is
~0.009 ms a call (`encode_frozen`, one token against the frozen decoder). If the rest is host work
then it is 28 x 0.3 ms of CPU a step against a step that is already host-bound, and the fix is to do
that host work once for a batch rather than to look for arithmetic to delete. Three controls settle
which it is: host wall against GPU time from events (CPU or card?), host wall with the queue EMPTY
(work or a wait?), and the per-phase split (which host code?). The last table is the one that decides
the design: ms/call against tokens/call gives the fixed part, and the fixed part is what batching --
across layers, by lagging one token -- would amortise.

Run it after touching `select_triton`, `encode.pack_and_write`, or the layout: a regression here is
invisible in the engine's profile.

USAGE
    PYTHONPATH=/opt/ttkv/integration python tests/bench_write_path.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv import encode as enc  # noqa: E402
from vllm_ttkv import prof  # noqa: E402
from vllm_ttkv import select_triton  # noqa: E402
from vllm_ttkv.layout import layout_for_config  # noqa: E402

H, D = 8, 128
LAYERS = 28
CALLS = 28 * 16                    # a decode step's layer-calls, times sixteen steps


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def _raw_k(h, n, d, rank=24, noise=0.02, seed=0):
    torch.manual_seed(seed)
    K = ((torch.randn(n, rank, device="cuda") * 0.2) @ (torch.randn(rank, h * d, device="cuda") * 0.2)
         ).reshape(n, h, d).permute(1, 0, 2).contiguous()
    return K + noise * torch.randn(h, n, d, device="cuda")


def _once(lay, cfg, n_tok, v_dtype=torch.bfloat16):
    """A fitted store plus one frozen chunk: the state a decode step's write actually has."""
    K = _raw_k(H, 64 + n_tok, D, seed=3)
    fitted = enc.fit_into_budget(K[:, :64].contiguous(), lay, cfg)
    chunk = K[:, 64:64 + n_tok].contiguous()
    fk = enc.encode_frozen(chunk, fitted.decoder, lay)
    torch.manual_seed(5)
    V = (torch.randn(H, n_tok, D, device="cuda") * 0.3).to(v_dtype)
    return fk, chunk, V


def _cache(n_slots, lay, dtype=torch.bfloat16):
    """A page as vLLM allocates it: the last dim counts MODEL-DTYPE ELEMENTS, not bytes."""
    n_bytes = lay.compressed_bytes_per_token()
    return torch.zeros(n_slots, n_bytes // 2, dtype=dtype, device="cuda")


def _timeit(label, fn, iters=CALLS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"  {label:36s} {ms:7.3f} ms/layer-call {ms * LAYERS:7.1f} ms/step over {LAYERS} layers")
    return ms


def _host_ms(call, iters):
    """Host wall per call with the queue held deep: max(host, GPU) over the loop."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _host_ms_empty(call, iters):
    """Per call with the queue EMPTY. Far below `_host_ms` means the number is a wait, not work."""
    acc = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        call()
        acc += time.perf_counter() - t
    return acc / iters * 1e3


def _gpu_ms(call, iters):
    """The GPU time the same calls occupy, from events on the stream."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _phases(call, iters):
    """Host ms per phase for this call, with the queue deep -- so these are work, not drains."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    prof.PHASES.clear()
    prof.ENABLED = True
    try:
        for _ in range(iters):
            call()
    finally:
        prof.ENABLED = False
    out = {k: sum(v) / len(v) * 1e3 for k, v in prof.PHASES.items() if v}
    prof.PHASES.clear()
    return out


def _fixed_cost(lay, cfg):
    """WHERE THE WRITE'S TIME GOES, and how much of it does not scale with tokens.

    The shape matters more than the total, because decode calls this ONCE PER TOKEN PER LAYER: the
    part that does not scale is paid 28 times a step, and the arithmetic it stands for is ~9 us
    (`encode.encode_frozen`, one token against the frozen decoder). Four measurements answer it -- host
    wall against GPU time (is the CPU or the card the limit?), the same host wall with the queue empty
    (is that number work or a wait?), the per-phase split (which host code it is), and ms/call against
    tokens per call (what a batch would amortise).
    """
    print("\n=== the write's shape, on 1 token (decode) ===")
    fk, _chunk, V = _once(lay, cfg, 1)
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    cache = _cache(4096, lay)
    call = lambda: enc.pack_and_write(cache, slots, fk, V, lay)      # noqa: E731
    host = _host_ms(call, CALLS)
    print(f"  {'host, queue deep':34s} {host:7.3f} ms/call {host * LAYERS:7.1f} ms/step")
    empty = _host_ms_empty(call, 64)
    print(f"  {'host, queue empty':34s} {empty:7.3f} ms/call"
          f"   {'-> work, not a wait' if empty > 0.6 * host else '-> mostly the queue'}")
    gpu = _gpu_ms(call, CALLS)
    print(f"  {'gpu, from events':34s} {gpu:7.3f} ms/call -> host share {host - gpu:7.3f}")
    for name, ms in sorted(_phases(call, CALLS).items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s} {ms:7.3f} ms/call")
    print(f"\n  what a batch would amortise, same store:\n  {'tokens/call':>12s} {'ms/call':>9s}"
          f" {'us/token':>9s}")
    for n_tok in (1, 8, 28, 64, 256, 1082):
        fk2, _c2, V2 = _once(lay, cfg, n_tok)
        sl = (torch.arange(n_tok, dtype=torch.int32, device="cuda") * 3) % 4000
        c2 = _cache(4096, lay)
        c = lambda fk2=fk2, V2=V2, sl=sl, c2=c2: enc.pack_and_write(c2, sl, fk2, V2, lay)  # noqa: E731
        ms = _host_ms(c, max(8, 2000 // n_tok))
        print(f"  {n_tok:12d} {ms:9.3f} {ms * 1e3 / n_tok:9.2f}")


def _arm(lay, cfg, n_tok, label, iters=CALLS):
    fk, chunk, V = _once(lay, cfg, n_tok)
    slots = (torch.arange(n_tok, device="cuda", dtype=torch.int32) * 3) % 4000
    cache = _cache(4096, lay)
    fused = _timeit(f"{label} fused (Triton)", lambda: enc.pack_and_write(cache, slots, fk, V, lay),
                    iters)
    os.environ[select_triton.ENV] = "0"
    try:
        torch_store = _timeit(f"{label} torch (PackedRows + slots)",
                              lambda: enc.pack_and_write(cache, slots, fk, V, lay), iters)
    finally:
        os.environ.pop(select_triton.ENV, None)
    print(f"  {'':36s} {torch_store / fused:6.2f}x")
    return fused, torch_store


def main() -> int:
    if not _need_cuda():
        return 0
    cfg = CONFIGS["safe"]
    lay = layout_for_config(cfg, r1=112, H=H, d=D)
    print(f"layout: r1={lay.r1} res_k={lay.res_k}@{lay.res_bits}b "
          f"v_lanes={lay.v_lanes}@{lay.v_bits}b slot={lay.compressed_bytes_per_token()} B")
    for n_tok, iters in ((1, CALLS), (1082, 20)):
        print(f"\n=== {n_tok} token(s) per step ===")
        _arm(lay, cfg, n_tok, f"{n_tok} token(s):", iters)
        fk, chunk, _ = _once(lay, cfg, n_tok)
        _timeit(f"{n_tok} token(s): encode_frozen",
                lambda: enc.encode_frozen(chunk, fk.decoder, lay), iters)
    _fixed_cost(lay, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
