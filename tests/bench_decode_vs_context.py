"""Decode step time against context, in the ENGINE, one arm per process.

WHY THIS EXISTS. Two measurements of the read disagree by more than an order of magnitude and the
disagreement decides what to optimise:

  * the engine's phase table says `decode_total` is roughly FLAT in context (~0.5 ms a layer-call
    from 1082 to 4322 tokens);
  * `tests/bench_decode_read.py` says the same call grows LINEARLY at ~0.36 us a cached token a layer,
    which is ~33 ms a step at 4322 and ~330 at 32k.

Both cannot be right. A phase measures the host ISSUING an async kernel group, not the group running,
so a compute-bound read can look free in the table while its cost lands later in the step -- that is
the explanation the write path already demonstrated (its phase reads ~4x its measured throughput
cost). This script measures the thing neither of them measures: the wall clock of a decode step, as a
marginal over generation length so prefill cancels and only steps are timed.

THE METHOD. For each context, two generations of the same prompt: `max_tokens=1` (prefill plus one
step) and `max_tokens=1+GAP` (prefill plus GAP+1 steps). The difference over GAP is ms/step, and the
first run minus that is the prefill. Prefill on this port is 26x stock's, so any rate that includes it
is a measurement of prefill wearing a decode label -- hence the subtraction rather than a total.

USAGE
    VLLM_TTKV_CALIBRATION=.../calibration/llama-3.2-3b-16384.json \\
      python tests/bench_decode_vs_context.py --arm ttkv --repeats 60 240 480 800
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import types

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[1] / "codec",
              _HERE.parents[2], pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from run_vllm_ttkv import _build, _prompt  # noqa: E402

LAYERS = 28
GAP = 16
# The layer-calls and the steps the engine reports, so the ms/step below can be checked against the
# phase table instead of trusted: 28 layers per step means one step is 28 layer-calls.
def _measure(llm, prompt, gap: int) -> tuple[float, float, float]:
    """(prefill_s, ms_per_step, steps_measured) for one context.

    `ignore_eos=True` on both calls, and it is what makes the marginal mean anything. The second call
    asks for `1 + gap` tokens, but the prompt is repeated prose and greedy decoding can stop on EOS
    after one -- in which case the two calls cost the SAME, the difference is ~0, and `ms_per_step`
    comes out near zero instead of near 40. Forcing the full count also makes `steps_measured` true
    rather than assumed. `tests/bench_vllm_baseline.py` already did this; this instrument did not, and
    it is the one the decode table is built from.
    """
    from vllm import SamplingParams

    t0 = time.time()
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    one_s = time.time() - t0
    t0 = time.time()
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1 + gap, ignore_eos=True))
    long_s = time.time() - t0
    per_step = (long_s - one_s) / gap * 1e3
    return one_s - per_step / 1e3, per_step, gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ttkv", "baseline"), required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[60, 240, 480, 800])
    ap.add_argument("--gap", type=int, default=GAP)
    ap.add_argument("--out")
    args = ap.parse_args()

    t0 = time.time()
    llm = _build(types.SimpleNamespace(arm=args.arm, graphs=False))
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    kv = getattr(llm.llm_engine, "cache_config", None) or getattr(
        getattr(llm.llm_engine, "vllm_config", None), "cache_config", None)
    print(f"[{args.arm}] load {load_s:.1f}s  blocks={getattr(kv, 'num_gpu_blocks', None)}"
          f"  gap={args.gap} steps", flush=True)

    print(f"  {'tokens':>8s} {'prefill s':>10s} {'ms/step':>9s} {'tok/s':>8s}"
          f" {'us/token cached':>16s}", flush=True)
    recs, first = [], None
    for reps in args.repeats:
        prompt = _prompt(reps)
        n = len(tok(prompt)["input_ids"])
        prefill_s, per_step, _ = _measure(llm, prompt, args.gap)
        # The marginal per cached token, measured BETWEEN contexts rather than within one: the
        # model's own weight streaming does not depend on context, so it cancels in this slope.
        slope = None
        if first is not None:
            n0, ms0 = first
            slope = (per_step - ms0) / (n - n0) * 1e3
        else:
            first = (n, per_step)
        print(f"  {n:8d} {prefill_s:10.2f} {per_step:9.1f} {1e3 / per_step:8.1f}"
              f" {'' if slope is None else f'{slope:.3f}':>16s}", flush=True)
        recs.append({"repeats": reps, "prompt_tokens": n, "prefill_s": round(prefill_s, 2),
                     "ms_per_step": round(per_step, 2), "tok_per_s": round(1e3 / per_step, 1),
                     "us_per_cached_token": None if slope is None else round(slope, 4)})
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"arm": args.arm, "gap": args.gap, "load_s": round(load_s, 1), "contexts": recs},
            indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
