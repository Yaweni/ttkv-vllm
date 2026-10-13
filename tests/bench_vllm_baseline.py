"""What does vLLM's OWN KV-cache compression buy on this machine?

TT-KV is not involved anywhere in this file. It measures the thing our numbers have to be compared
against, on the same GPU, the same model and the same context length, so that a later TT-KV result
cannot be read against a baseline nobody ran.

WHY BYTES AND SPEED, NOT JUST SPEED
    At EQUAL bytes a compressed cache that dequantizes will beat one that reconstructs a tensor
    train and rotates in-register. The defensible claim for TT-KV is quality-vs-memory, so the two
    axes have to be measured together:
      * `concurrency` -- how many 16k requests fit. This is the BYTES axis, and it is the cleanest
        one available: vLLM sizes the cache at init, so the count is a direct inverse of bytes/token
        and needs no accounting of our own.
      * `ttft_ms` / `decode_tps` -- the SPEED axis, at that compression level.

ONE ARM PER PROCESS, ON PURPOSE
    vLLM does not reliably return GPU memory to the driver after an engine is destroyed, so running
    fp8 and turboquant in one process would make the second arm's `gpu_memory_utilization` budget
    depend on the first arm's residue. Driver script runs one dtype per invocation instead.

    python tests/bench_vllm_baseline.py --dtype fp8
    python tests/bench_vllm_baseline.py --dtype turboquant_k3v4_nc

Runs in the vLLM venv. Prints one `RESULT ...` line per arm for a driver to collect.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time

# Set BEFORE vLLM is imported, because vLLM reads it at sampler construction.
#
# Measured failure without it: vLLM logs "Using FlashInfer for top-p & top-k sampling", and on sm_86
# FlashInfer has no prebuilt cubin for the sampling op, so it tries to JIT-compile and dies with
# `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` -- during
# `warmup_kernels`, before a single token is generated.
#
# Pinned OFF rather than made conditional, because the sampler is NOT the attention path: letting it
# differ between arms would move the timings for a reason unrelated to the thing being compared.
# Set VLLM_USE_FLASHINFER_SAMPLER=1 to override once a CUDA toolkit is installed.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# vLLM logs the KV cache sizing at INFO during engine init. We parse it rather than recomputing from
# `cache_config`, because the recomputation needs the number of GPU blocks, which V1 does not expose
# stably -- and a wrong byte number is worse than a missing one. If the parse fails, the field is
# reported as `NA` with a reason instead of being guessed.
_KV_LINES = (
    re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x"),
    re.compile(r"GPU KV cache size:\s*([\d,]+) tokens"),
    re.compile(r"KV cache size:\s*([\d,]+) tokens"),
)


class _StderrTee:
    """Capture everything written to fd 2, including the engine-core child's logs.

    A `logging.Handler` cannot do this, and the first version of this file proved it: it reported
    `concurrency=NA` on a run whose log plainly contained
    "Maximum concurrency for 16,384 tokens per request: 4.23x". The reason is that the KV-sizing
    line is emitted in the EngineCore CHILD process, and the core client forwards it by writing to
    the inherited stderr DESCRIPTOR -- not through this process's `logging` module.

    So this tears on fd 2 rather than on `sys.stderr`, because a child process can only inherit a
    descriptor. Lines are echoed to the real stderr as well, so a failure is still debuggable.
    """

    def __init__(self) -> None:
        self._r, w = os.pipe()
        self._saved = os.dup(2)
        self.text: list[str] = []
        os.dup2(w, 2)
        os.close(w)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        with os.fdopen(self._r, "r", errors="replace") as fh:
            for line in fh:
                self.text.append(line.rstrip("\n"))
                try:
                    # Write to the SAVED fd, never to `sys.__stderr__`. `sys.__stderr__` is bound
                    # to fd 2, which by now IS the pipe -- so echoing through it feeds the pipe
                    # back into itself and the drain thread spins forever. Measured: the first
                    # version of this class wedged the process and the ssh session hung.
                    os.write(self._saved, line.encode("utf-8", "replace"))
                except Exception:                                         # noqa: BLE001
                    pass

    def close(self) -> None:
        for _ in range(10):                       # let the forwarding thread catch up
            time.sleep(0.05)
        try:
            sys.stderr.flush()
        except Exception:                                                 # noqa: BLE001
            pass
        os.dup2(self._saved, 2)
        os.close(self._saved)


def _parse_kv(text: str):
    concurrency = tokens = None
    for line in text:
        m = _KV_LINES[0].search(line)
        if m:
            concurrency = float(m.group(2))
        for pat in _KV_LINES[1:]:
            m = pat.search(line)
            if m:
                tokens = int(m.group(1).replace(",", ""))
    return concurrency, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-file", default=None,
                    help="Path the caller redirected this process's output to (e.g. `> arm.log "
                         "2>&1`). Used as a fallback source for vLLM's KV-sizing line; see the "
                         "note at the parse site.")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    capture = _StderrTee()

    # A real prompt, `out_tokens` short of the context limit. It must be SHORT of it: a prompt of
    # exactly `max_model_len` leaves no room for a single output token and vLLM rejects the request
    # with VLLMValidationError. `max_model_len` itself stays at `--context` so the cache is still
    # sized for that figure -- which is the number the `concurrency` column reports.
    prompt_len = args.context - args.out_tokens
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(100, 30000, (prompt_len,), generator=g).tolist()

    t_init = time.perf_counter()
    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.dtype,
        max_model_len=args.context,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_seqs,
        enable_prefix_caching=False,
        disable_log_stats=True,
        trust_remote_code=False,
    )
    t_init = time.perf_counter() - t_init

    # Two runs, because `LLM.generate` only returns when the request is finished: a 1-token run
    # bounds prefill (TTFT), and the long run's excess over it is pure decode. Documented as an
    # approximation -- the 1-token run measures prefill + one decode step, so `decode_tps` is very
    # slightly conservative.
    common = dict(temperature=0.0, ignore_eos=True)
    llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, **common))

    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=args.out_tokens, **common))
    t_long = time.perf_counter() - t0

    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, **common))
    t_first = time.perf_counter() - t0

    decode_tps = (args.out_tokens - 1) / (t_long - t_first) if t_long > t_first else float("nan")

    concurrency, tokens = _parse_kv(capture.text)
    capture.close()
    source = "tee"
    if concurrency is None and args.log_file and os.path.exists(args.log_file):
        # The tee is the primary source, but it is not guaranteed: the line is emitted by the
        # EngineCore child and re-emitted by the core client, and it was measured to slip past a
        # capture that a plain `2>&1` redirects catches every time. So fall back to reading the
        # caller's own log file rather than reporting NA for a number that is sitting in it.
        try:
            with open(args.log_file, errors="replace") as fh:
                concurrency, tokens = _parse_kv(fh.read().splitlines())
            source = "log-file"
        except OSError:
            source = "tee"
    why = "" if concurrency is not None else \
        f"  (vLLM's KV-sizing line not found in the tee or {args.log_file!r})"
    cval = "NA" if concurrency is None else f"{concurrency:.2f}"
    print(f"RESULT dtype={args.dtype} ctx={args.context} prompt={prompt_len} init_s={t_init:.1f} "
          f"concurrency={cval}[{source}] kv_tokens={'NA' if tokens is None else tokens} "
          f"ttft_ms={1000 * t_first:.1f} decode_tps={decode_tps:.1f}{why}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
