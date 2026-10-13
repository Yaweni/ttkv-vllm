"""Does the extra cache capacity turn into throughput? One arm per process, concurrency swept.

THE QUESTION, mechanically. In vLLM the KV cache IS the concurrency limit: blocks are cached tokens,
a request needs `ceil(len/16)` of them, and the pool is sized once from what is left after the
weights. So "1.74x more blocks" (7244 against 4159, measured on this box) means ~1.74x more resident
sequences at a fixed length -- and that is the ONLY route by which capacity becomes throughput,
because a decode step costs

    step = W + B * n * c

where `W` is streaming the model's weights and does not depend on the batch at all, `n` is the
context each request reads, `c` is our attention cost per cached token per request, and `B` is how
many requests are resident. Throughput is `B / step`, so it rises with `B` while `B*n*c << W` and
saturates after that. **Capacity converts only while the batch is still small enough that the weights
dominate**, and where that knee sits is a measurement, not an opinion: stock's `c` is ~0.6 us a token
and ours is ~10, so stock's knee is at `B*n ~ 33k` and ours at `B*n ~ 2k`.

WHY 8192 TOKENS AND seqs UP TO 12. 8192 tokens is 512 blocks a request, so stock's 4159 blocks admit
8 requests and our 7244 admit 14. Sweeping to 12 goes inside our pool and past stock's, which is
exactly the crossing the capacity claim needs. Contexts that fit in both pools would test nothing.

WHAT TO READ. If tokens/s keeps rising for us while stock's flattens, the extra cache is doing work.
If ours flattens at N=1-2 while stock keeps rising to 8, we are compute-bound -- our attention is 17x
stock's per cached token -- and the extra blocks sit idle. That second outcome is the same finding as
"the read is the project", reached from the serving side instead of from the kernel side.

`enable_chunked_prefill=False` (inherited from the arm's construction) means each prompt prefills in
one chunk, so prefill cost is real and included in `total`. Per-request `metrics` are printed when the
installed vLLM exposes them, because the interesting number is the DECODE window -- the span from the
first request's first token to the last request's last token -- not the total, which is dominated by
prefill on this port.

USAGE
    VLLM_TTKV_CALIBRATION=/opt/ttkv/vllm_ttkv_plugin/calibration/llama-3.2-3b-16384.json \
      python tests/run_vllm_concurrency.py --arm ttkv --context 8192 --seqs 1 2 4 8 12
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import types

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parent, _HERE.parents[1], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from run_vllm_ttkv import PROMPT_UNIT, _build  # noqa: E402

BLOCK = 16


def _prompt(i: int, n: int, repeats: int) -> str:
    """Distinct per request AND per sweep point, so nothing is shared through the prefix cache.

    Two traps this avoids, both learned from a crash. Identical prompts would let vLLM serve N
    requests off one cached prefix, so the block demand would collapse and the sweep would measure the
    prefix cache instead of the pool. And reusing a prompt across sweep points is worse: the engine
    recognises it, schedules only the new tokens as a continuation chunk, and a chunked prefill is
    something this impl REFUSES by design (the prefill branch needs a multi-query `krt`). That refusal
    is correct behaviour and it looks exactly like a concurrency bug.
    """
    return f"Note {i} of {n}: the following passage is repeated for length. " + PROMPT_UNIT * repeats


def _sweep_one(llm, prompts, short: int, long: int):
    """(prefill s, decode s, decode tokens) for one batch, as a MARGINAL over generation length.

    Two generations of the same prompts, `short` tokens and `long` tokens, and the difference between
    their times is the cost of `long - short` decode steps -- the prefill cancels. That matters here
    because this port's prefill is 26x stock's, so any rate that includes it is a measurement of
    prefill wearing a decode label. It also avoids needing the prefix cache to make a second call
    cheap, which is just as well: prefix caching is off on this arm by design.
    """
    from vllm import SamplingParams

    t0 = time.time()
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=short))
    first_s = time.time() - t0
    t0 = time.time()
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=long))
    second_s = time.time() - t0
    n = len(prompts)
    return first_s, second_s - first_s, n * (long - short)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ttkv", "baseline"), required=True)
    ap.add_argument("--context", type=int, default=8192, help="target prompt tokens")
    ap.add_argument("--seqs", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--chunked", action="store_true",
                    help="chunked prefill: splits a long prompt, which schedules continuation "
                         "chunks, which the multi-query kernel now handles")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="prefix caching. SEPARATE FROM --chunked because it breaks this script's "
                         "decode measurement: with caching on, the second generation re-uses the "
                         "first call's KV, so the two calls differ by a whole prefill and `tok/s "
                         "decode` comes out negative")
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384,
                    help="the token budget. This is what decides whether a batch splits at all: a "
                         "prompt set that fits one chunk never enters the continuation path, which "
                         "costs ~10x the first-chunk rate. Sweep both ways -- the fit tells you "
                         "about the format, the split tells you about the continuation")
    ap.add_argument("--gap", type=int, default=8,
                    help="decode steps between the two generations. The decode rate is a difference of "
                         "two runs, so the gap has to be wide enough that its time is above timing "
                         "noise -- 8 steps is fine for this port's ~79 ms/step, but the stock arm with "
                         "CUDA graphs runs fast enough that 8 steps is inside the noise, so widen it")
    ap.add_argument("--graphs", action="store_true",
                    help="baseline only: CUDA graphs + torch.compile, to price the integration cost")
    ap.add_argument("--out")
    args = ap.parse_args()

    from vllm import SamplingParams

    t0 = time.time()
    llm = _build(types.SimpleNamespace(arm=args.arm, graphs=args.graphs, chunked=args.chunked,
                                       prefix_cache=args.prefix_cache,
                                       max_num_batched_tokens=args.max_num_batched_tokens))
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    repeats = max(1, args.context // len(tok(PROMPT_UNIT)["input_ids"]))
    kv = getattr(llm.llm_engine, "cache_config", None) or getattr(
        getattr(llm.llm_engine, "vllm_config", None), "cache_config", None)
    n_blocks = getattr(kv, "num_gpu_blocks", None)
    probe = len(tok(_prompt(0, -1, repeats))["input_ids"])
    print(f"[{args.arm}] load {load_s:.1f}s  blocks={n_blocks}  prompt {probe} tokens"
          f"  {args.new_tokens} greedy new tokens", flush=True)

    short, long = args.new_tokens, args.new_tokens + args.gap
    print(f"  {'N':>3s} {'prefill s':>10s} {'decode s':>9s} {'out tok':>8s} {'tok/s decode':>13s}"
          f" {'blocks':>8s} {'pool':>6s}", flush=True)
    recs = []
    for n in args.seqs:
        prompts = [_prompt(i, n, repeats) for i in range(n)]
        prefill_s, decode_s, got = _sweep_one(llm, prompts, short, long)
        need = n * (-(-probe // BLOCK))
        rate = got / decode_s if decode_s else 0.0
        print(f"  {n:3d} {prefill_s:10.2f} {decode_s:9.2f} {got:8d} {rate:13.1f}"
              f" {need:8d} {str(n_blocks):>6s}"
              f"   {'fits' if n_blocks and need <= n_blocks else 'OVER POOL'}", flush=True)
        recs.append({"n": n, "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
                     "out_tokens": got, "tok_per_s_decode": round(rate, 2),
                     "blocks_needed": need, "blocks_pool": n_blocks})
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"arm": args.arm, "context": probe, "blocks": n_blocks, "load_s": load_s,
             "sweep": recs}, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
