"""First real run: TT-KV serving inside vLLM, and an apples-to-apples stock-vLLM arm.

WHY THIS SCRIPT EXISTS AND WHAT IT IS ALLOWED TO CONCLUDE
    Every layer below this has been verified against an oracle, and every one of those oracles reads
    the store this port writes. None of them proves that vLLM actually SELECTS our backend, that the
    builder is handed the metadata it expects, that the rotary wrapper intercepts the model's
    rotation, or that the cache the engine allocated is the cache the encoder fills. Those are
    integration facts, and the only way to check them is to serve.

    So this runs twice with the same prompt and greedy decoding: once with `--arm ttkv` and once
    with `--arm baseline` (stock vLLM, no plugin involvement). Agreement is then the signal, and it
    is a WEAK signal that must be reported honestly: TT-KV is lossy, so a disagreement does not
    indicate a bug and agreement does not exhaustively prove correctness. What it does catch is a
    gross integration failure -- a backend that never ran, a rotation applied twice, a store read at
    the wrong offsets -- all of which produce fluent-looking text that diverges immediately.

    The arm that is NOT affected by compression is `baseline`, and it is also the control for the
    script itself: if the two arms differ only in `attention_config`, then any difference in the
    generated ids came from the cache.

USAGE
    export VLLM_TTKV_CALIBRATION=$PWD/calibration/llama-3.2-3b-16384.json
    export VLLM_TTKV_CONFIG=safe
    export PYTHONPATH=/opt/ttkv/integration          # the codec, for the encode
    python tests/run_vllm_ttkv.py --arm ttkv     --out /tmp/ttkv.json
    python tests/run_vllm_ttkv.py --arm baseline --out /tmp/base.json
    python tests/run_vllm_ttkv.py --compare /tmp/ttkv.json /tmp/base.json

    # prefill only, both arms: --new-tokens 1 makes generate_s prefill + one step
    python tests/run_vllm_ttkv.py --arm ttkv     --new-tokens 1 --out /tmp/ttkv_p.json
    python tests/run_vllm_ttkv.py --arm baseline --new-tokens 1 --out /tmp/base_p.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


_HERE = pathlib.Path(__file__).resolve()
# The codec, for the encode. Vendored at `<repo>/codec` and imported as a top-level module, which is
# how the original project laid it out; a clean checkout works without setting PYTHONPATH.
for _c in (_HERE.parents[1], _HERE.parents[1] / "codec"):
    if _c.is_dir() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

MODEL = "meta-llama/Llama-3.2-3B"
PROMPT_UNIT = "The history of the Roman Empire is a long one, and it is recorded in detail. "
NEW_TOKENS = 16


def _prompt(repeats: int) -> str:
    return PROMPT_UNIT * repeats


def _build(args):
    from vllm import LLM
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # `--graphs` is a BASELINE-ONLY probe, and it exists to price one integration cost rather than to
    # be a setting: the stock arm can be CUDA-graph captured, and stock's own `enforce_eager=False`
    # path also runs `torch.compile`, so the difference between the two baseline runs is the value of
    # everything this impl gives up by being host-interactive (`TTKVAttentionMetadataBuilder` inherits
    # `AttentionCGSupport.NEVER`). The ttkv arm cannot capture a graph at all -- it splits the batch
    # with `.tolist()` and looks decoders up in a Python registry, so a captured graph would replay
    # one batch's shapes against another's data -- so asking for it there is refused rather than
    # silently ignored.
    graphs = bool(getattr(args, "graphs", False))
    chunked = bool(getattr(args, "chunked", False))
    # CHUNKED PREFILL AND PREFIX CACHING ARE DECOUPLED ON PURPOSE, and the reason is a measurement
    # one rather than a capability one. Both can schedule a continuation chunk, but prefix caching
    # ALSO makes the harness's second generation re-use the first call's KV -- so with it on, the
    # two calls differ by a whole prefill and the decode rate (a DIFFERENCE of two runs) comes out
    # NEGATIVE. Measured: -1.4, -1.5, -0.1, -0.1 tok/s at N=1,2,4,8, against -prefill_s almost
    # exactly. So `--chunked` alone is the setting that exercises the continuation path AND leaves
    # the decode number meaningful.
    prefix = bool(getattr(args, "prefix_cache", False))
    # THE TOKEN BUDGET IS WHAT DECIDES WHETHER A CONTINUATION HAPPENS AT ALL, and a continuation is
    # ~8x the per-token cost of a first chunk on the ttkv arm (measured: 1070 tok/s at N=2 where the
    # batch fits one chunk, 108 tok/s at N=8 where it does not). So it is a flag rather than a
    # constant: a sweep at a batch that does not fit tells you about the continuation path, and one
    # at a budget that does fit tells you about the format. Both are wanted, at different times.
    budget = int(getattr(args, "max_num_batched_tokens", 0) or 16384)
    if budget < 16384:
        raise SystemExit(
            f"--max-num-batched-tokens={budget} is below max_model_len=16384, which pydantic "
            f"refuses: a smaller batched-token budget would cap the sequence length below what "
            f"`max_model_len` was asked for.")
    if graphs and args.arm == "ttkv":
        raise SystemExit(
            "--graphs cannot apply to the ttkv arm: the impl is host-interactive (`.tolist()` batch "
            "split, Python decoder registry) and a captured graph would replay the wrong batch "
            "against the wrong request lengths. This flag measures what that costs, on the arm that "
            "can use it.")

    kw = {}
    if args.arm == "ttkv":
        table = os.environ.get("VLLM_TTKV_CALIBRATION")
        if not table:
            raise SystemExit(
                "VLLM_TTKV_CALIBRATION is not set, so `customize_spec` cannot size the cache and "
                "the run would fail at cache allocation with a message about a missing table. Point "
                "it at a tools/calibrate_r1.py artifact (calibration/llama-3.2-3b-16384.json ships "
                "for this model).")
        if not pathlib.Path(table).is_file():
            raise SystemExit(f"VLLM_TTKV_CALIBRATION={table!r} does not exist")
        print(f"  allocation: {os.environ.get('VLLM_TTKV_CONFIG', 'safe')} (from VLLM_TTKV_CONFIG)")
        print(f"  calibration: {table}")
        print(f"  codec on path: {os.environ.get('PYTHONPATH', '(unset)')}")
        kw["attention_config"] = AttentionConfig(backend=AttentionBackendEnum.CUSTOM)

    # A CUDA TRACE IS OFF UNLESS ASKED FOR, and it has to be configured here rather than by env var:
    # this vLLM build refuses `start_profile()` with "Profiling is not enabled. Please set
    # --profiler-config", so the config has to exist at construction. `tests/bench_step_kernels.py`
    # is the caller -- it is the only way to attribute a step to KERNELS rather than to phases, and
    # the phase table has been misread twice on this port.
    trace_dir = getattr(args, "trace_dir", None)
    if trace_dir:
        from vllm.config import ProfilerConfig
        kw["profiler_config"] = ProfilerConfig(profiler="torch",
                                              torch_profiler_dir=str(trace_dir))

    return LLM(
        model=MODEL,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        # Host-interactive by construction: the impl splits the batch with `.tolist()`, gathers each
        # request's slots, and looks its decoder up in a Python registry. A CUDA graph captured over
        # that would bake in one batch's shapes and replay the wrong batch, so capture is off and
        # `TTKVAttentionMetadataBuilder` inherits `AttentionCGSupport.NEVER`. `--graphs` on the
        # BASELINE arm is how much that costs, measured rather than asserted.
        enforce_eager=not graphs,
        # CHUNKED PREFILL AND PREFIX CACHING WERE REFUSED HERE UNTIL THE MULTI-QUERY `krt` EXISTED,
        # because both can schedule a CONTINUATION chunk -- a request with keys already in the store
        # and only a few new query tokens. The single-query kernel contracts one query row per (kv
        # head, group) and returns `[HQ, n]`, so it could not express that; a continuation used to
        # take the engine down with a correct refusal. `_prefill` now reads the whole store through
        # `krt_multi_from_slots`, so both are available.
        #
        # THEY STAY OFF BY DEFAULT ANYWAY, and not out of caution: the recorded prefill and
        # decode-vs-context numbers were taken without them, and a split prefill is a DIFFERENT
        # measurement -- the fit lands on the first chunk of a different size, and `_prefill` takes
        # a different branch. `--chunked` turns on chunked prefill; see `prefix` above for why
        # prefix caching is a separate flag.
        enable_chunked_prefill=chunked,
        # DECOUPLED FROM `--chunked` -- see `prefix` above. Turning it on makes the second
        # generation hit the cache, so the harness's decode difference stops measuring decode.
        enable_prefix_caching=prefix,
        # At least `max_model_len`, which pydantic enforces: a smaller batched-token budget would
        # cap the sequence length below what was asked for. See `budget` above for why it is a knob.
        max_num_batched_tokens=budget,
        **kw,
    )


def _run(args) -> dict:
    from vllm import SamplingParams

    t0 = time.time()
    llm = _build(args)
    load_s = time.time() - t0

    prompt = _prompt(args.repeats)
    n_prompt = len(llm.get_tokenizer()(prompt)["input_ids"])
    sp = SamplingParams(temperature=0.0, max_tokens=args.new_tokens)
    print(f"  prompt {n_prompt} tokens, {args.new_tokens} greedy new tokens", flush=True)

    t0 = time.time()
    outs = llm.generate([prompt], sp)
    gen_s = time.time() - t0
    o = outs[0].outputs[0]

    kv = getattr(llm.llm_engine, "cache_config", None)
    if kv is None:
        kv = getattr(getattr(llm.llm_engine, "vllm_config", None), "cache_config", None)
    n_blocks = getattr(kv, "num_gpu_blocks", None)
    rec = {
        "arm": args.arm,
        "model": MODEL,
        "prompt_tokens": n_prompt,
        "new_tokens": args.new_tokens,
        "token_ids": list(o.token_ids),
        "text": o.text,
        "gpu_blocks": n_blocks,
        "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 2),
    }
    return rec


def _compare(a_path: str, b_path: str) -> int:
    a = json.loads(pathlib.Path(a_path).read_text())
    b = json.loads(pathlib.Path(b_path).read_text())
    ta, tb = a["token_ids"], b["token_ids"]
    first_diff = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    agree = sum(1 for x, y in zip(ta, tb) if x == y)

    print(f"{'arm':<10s} {'blocks':>8s} {'load s':>7s} {'gen s':>7s}  text")
    for r in (a, b):
        print(f"{r['arm']:<10s} {str(r['gpu_blocks']):>8s} {r['load_s']:>7} {r['generate_s']:>7}  "
              f"{r['text'][:58]!r}")
    print(f"\ngreedy tokens: {agree}/{len(tb)} identical"
          + (f", first divergence at {first_diff}" if first_diff is not None else ""))
    if a["gpu_blocks"] and b["gpu_blocks"]:
        print(f"KV cache blocks: baseline {b['gpu_blocks']}, ttkv {a['gpu_blocks']} "
              f"({a['gpu_blocks'] / b['gpu_blocks']:.2f}x)")
    print("\nA DIVERGENCE IS NOT BY ITSELF A BUG: TT-KV is lossy, and one flipped token changes the "
          "rest of a greedy rollout.\nA shared first token plus fluent continuation is the "
          "integration signal; both arms failing to load, or the ttkv arm producing '!' repeated, "
          "is not.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ttkv", "baseline"))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("TTKV", "BASELINE"))
    ap.add_argument("--repeats", type=int, default=60,
                    help="prompt repetitions; ~20 tokens each")
    ap.add_argument("--graphs", action="store_true",
                    help="baseline only: let stock vLLM capture CUDA graphs (and run torch.compile), "
                         "which prices what this impl gives up by being host-interactive")
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS,
                    help="greedy tokens to generate. `--new-tokens 1` isolates the PREFILL: the run's "
                         "wall time is then the prompt's forward pass plus one decode step, which is "
                         "the only way to separate a prefill regression from a decode one in this "
                         "harness's single `generate_s` number")
    ap.add_argument("--chunked", action="store_true",
                    help="enable chunked prefill AND prefix caching, both of which schedule "
                         "continuation chunks. Available now that the multi-query `krt` exists; off "
                         "by default because the recorded prefill numbers were taken without them")
    args = ap.parse_args()

    if args.compare:
        return _compare(*args.compare)
    if not args.arm:
        raise SystemExit("--arm or --compare is required")

    rec = _run(args)
    print(f"\n[{rec['arm']}] {rec['text']!r}")
    print(f"  ids: {rec['token_ids']}")
    print(f"  gpu_blocks={rec['gpu_blocks']} load={rec['load_s']}s generate={rec['generate_s']}s")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rec, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
