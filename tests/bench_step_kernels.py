"""What does the GPU actually spend a step on? CUDA time by kernel, not host time by phase.

WHY THIS FILE EXISTS, AND WHY THE OBVIOUS ANSWER WAS WRONG TWICE. `VLLM_TTKV_PROFILE=1` reports host
wall-clock per phase, and this port has now been misread from it twice in one session. `write.project`
read 102.5 ms/step at batch 8 and looked like a launch storm; batching its launches 3.6x moved the
step by 14%, and sampling `nvidia-smi` during the run showed the GPU at 95-98%. A phase times the HOST
issuing or waiting on an async kernel group, and it cannot tell those two apart -- so on a saturated
device every phase is inflated by queue drain and the biggest number in the table is not the biggest
user of the GPU.

That makes "which kernel should I rewrite" a question only the DEVICE can answer. This script asks it:
it runs the engine in the vLLM arm, records a CUDA trace through vLLM's own profiler (the engine lives
in a separate process, so a profiler in this process would see nothing), and prints the kernels by
total CUDA time.

USAGE

    VLLM_TTKV_CALIBRATION=/opt/ttkv/vllm_ttkv_plugin/calibration/llama-3.2-3b-16384.json \
      python tests/bench_step_kernels.py --seqs 1 8 --new-tokens 24

STATUS ON THIS HARDWARE: THE EXPORT IS IMPRACTICALLY SLOW, SO THIS IS NOT THE ROUTE THAT WORKED.
A 20-step window at batch 8 produced a **100 MB gzipped** trace -- the profiler records the engine's
CPU ops as well as its kernels -- and exporting it pinned a core for minutes and grew the engine
process to 24 GB. A 2-step window was still exporting after twelve minutes. The numbers in this
file's docstring about where the step goes came from `bench_krt_tune.py` (the isolated kernel, with
CUDA synchronisation around it) plus the known step time, not from this script.

It is kept because the QUESTION is the right one and the mechanism is correct: if the profiler's
overhead is acceptable on another machine, this is still the only way to attribute a step to kernels
rather than to phases. Keep the window at one or two steps, and check the trace size before reaching
for a longer one.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

# The ttkv kernels are Triton, so their names carry the generated hash; the grouping below collapses
# that so one kernel does not appear as forty rows.
_KERNEL_GROUPS = (
    ("krt (reconstruct K)", ("krt",)),
    ("v_out (readout V)", ("v_out",)),
    ("select/pack (the write)", ("select_pack", "pack_k", "pack_v", "write_rows")),
    ("gather/split/copy", ("gather", "split", "index_select", "copy", "cat", "contiguous")),
    ("softmax", ("softmax",)),
    ("reduction (amax/sum)", ("amax", "sum", "max", "argmax")),
    ("elementwise (mul/add/cast)", ("mul", "add", "sub", "div", "where", "clamp", "round",
                                    "cast", "zero", "fill", "neg", "abs", "scatter")),
    ("matmul (cublas)", ("gemm", "matmul", "bmm", "sgemm", "cutlass")),
    ("topk/sort", ("topk", "sort",)),
)


def _group(name: str) -> str:
    low = name.lower()
    for label, keys in _KERNEL_GROUPS:
        if any(k in low for k in keys):
            return label
    return "other"


def _load_kernels(trace_dir: str) -> dict:
    """{kernel name: total cuda microseconds} summed across every rank's trace.

    THE TRACE IS GZIPPED AND IT IS BIG, which is why the window is a couple of steps rather than a
    generation. vLLM's torch profiler writes `rank0.<id>.pt.trace.json.gz`, and a 20-step window at
    batch 8 came out at 100 MB COMPRESSED -- the profiler records the engine's CPU ops as well as
    the kernels, and the export alone pinned a core for minutes and grew the engine to 24 GB. Two
    steps is plenty for "which kernel is the work"; it is not enough for a rate.
    """
    out: dict = {}
    files = sorted(glob.glob(os.path.join(trace_dir, "*.pt.trace.json*")))
    if not files:
        raise SystemExit(
            f"no trace under {trace_dir}. If the engine started but wrote nothing, this vLLM "
            f"build's profiler needs `profiler_config` at CONSTRUCTION (it refuses "
            f"`start_profile()` with 'Profiling is not enabled') -- `_build` in run_vllm_ttkv.py "
            f"takes `trace_dir` for that.")
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            events = json.load(fh).get("traceEvents", [])
        for e in events:
            if e.get("cat") != "kernel":
                continue
            out[e["name"]] = out.get(e["name"], 0.0) + float(e.get("dur", 0.0))
        print(f"  read {os.path.basename(f)} ({len(events)} events)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=8, help="concurrent requests; this is the batch")
    ap.add_argument("--context", type=int, default=2048)
    # TWO STEPS IS THE DEFAULT ON PURPOSE -- see `_load_kernels`. The question here is "which kernel
    # is the work", and the trace cost grows with the window while the answer does not.
    ap.add_argument("--new-tokens", type=int, default=6)
    ap.add_argument("--trace-dir", default="/opt/ttkv/trace")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    os.environ["VLLM_TORCH_PROFILER_DIR"] = args.trace_dir
    pathlib.Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    for stale in glob.glob(os.path.join(args.trace_dir, "*.pt.trace.json")):
        os.remove(stale)

    import types as _types

    from vllm import SamplingParams
    from run_vllm_ttkv import PROMPT_UNIT, _build  # noqa: E402
    from run_vllm_concurrency import _prompt        # noqa: E402

    print(f"  building the ttkv arm (batch {args.seqs})...", flush=True)
    llm = _build(_types.SimpleNamespace(arm="ttkv", graphs=False, chunked=False,
                                        prefix_cache=False, max_num_batched_tokens=16384,
                                        trace_dir=args.trace_dir))
    tok = llm.get_tokenizer()
    repeats = max(1, args.context // len(tok(PROMPT_UNIT)["input_ids"]))
    prompts = [_prompt(i, args.seqs, repeats) for i in range(args.seqs)]
    sp_warm = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.new_tokens, ignore_eos=True)

    # WARM FIRST. Triton JITs on first use, and a trace of the compile is a trace of the compiler.
    llm.generate(prompts, sp_warm)

    steps = args.new_tokens - 4
    print(f"  profiling {steps} decode steps at batch {args.seqs}...", flush=True)
    llm.start_profile()
    llm.generate(prompts, sp)
    llm.stop_profile()

    kern = _load_kernels(args.trace_dir)
    total_s = sum(kern.values()) / 1e6
    if not kern:
        raise SystemExit("the trace holds no kernel events")
    print(f"\n  {len(kern)} distinct kernels, {total_s:.3f} s of CUDA time over "
          f"{steps} steps = {total_s / steps * 1e3:.1f} ms/step")
    print(f"  {'kernel':<44s} {'ms total':>10s} {'ms/step':>9s} {'%':>6s}")

    grouped: dict = {}
    for name, us in kern.items():
        grouped[_group(name)] = grouped.get(_group(name), 0.0) + us
    for name, us in sorted(kern.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {name[:44]:<44s} {us / 1e3:>10.1f} {us / 1e3 / steps:>9.2f} "
              f"{us / sum(kern.values()) * 100:>5.1f}%")

    print(f"\n  {'-- by our grouping --':<44s} {'ms total':>10s} {'ms/step':>9s} {'%':>6s}")
    for label, us in sorted(grouped.items(), key=lambda kv: -kv[1]):
        if us / sum(kern.values()) < 0.005:
            continue
        print(f"  {label:<44s} {us / 1e3:>10.1f} {us / 1e3 / steps:>9.2f} "
              f"{us / sum(kern.values()) * 100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
