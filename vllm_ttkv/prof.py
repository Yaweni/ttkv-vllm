"""Phase timing shared by the modules on the decode path.

WHY THIS IS ITS OWN MODULE. The measurement that matters is the one that splits the cost finely
enough to act on, and the work does not live in one file: the write is `impl` -> `encode` -> the
codec, and the read is `impl` -> `decode` -> the Triton kernels. A profiler that only `impl` could
call would measure `encode_frozen` as one 1.4 ms lump, which is not a thing anyone can optimise --
what is actionable is that it is ~80 tiny kernel launches of ~15 us each, and that only a phase per
STEP of the encode can say.

Off unless `VLLM_TTKV_PROFILE=1`. When off, `phase` does nothing at all: no attribute lookups, no
clock reads, no allocations, so the measured path is the shipped path.
"""
from __future__ import annotations

import atexit
import os
import time

ENABLED = bool(os.environ.get("VLLM_TTKV_PROFILE"))
PHASES: dict[str, list[float]] = {}
COUNT = {"syncs": 0, "layers": 0, "steps": 0}
_NOW = time.perf_counter


class phase:
    """Time a phase when profiling is on, and do nothing when it is off."""

    __slots__ = ("name", "t")

    def __init__(self, name: str):
        self.name = name
        self.t = 0.0

    def __enter__(self):
        if ENABLED:
            self.t = _NOW()
        return self

    def __exit__(self, *exc):
        if ENABLED:
            PHASES.setdefault(self.name, []).append(_NOW() - self.t)
        return False


def sync(tag: str) -> None:
    """Record that a DEVICE value is about to be read back on the host.

    Counted rather than timed: the cost shows up in the phases that follow, not in the read itself.
    Each one drains the pipeline, and the decode used to pay for ~6 of them per layer-call.
    """
    if ENABLED:
        COUNT["syncs"] += 1


def layer_call() -> None:
    if ENABLED:
        COUNT["layers"] += 1


def summary(extra: str = "") -> None:
    if not ENABLED or not PHASES:
        return
    steps = max(COUNT["steps"], 1)
    layers = max(COUNT["layers"], 1)
    total = sum(sum(v) for v in PHASES.values())
    print(f"\n[vllm_ttkv profile] {layers} layer-calls, {COUNT['steps']} steps, "
          f"{COUNT['syncs']} device readbacks ({COUNT['syncs'] / layers:.1f} per layer-call)"
          f"{extra}", flush=True)
    print(f"  {'phase':<20s} {'calls':>7s} {'total ms':>9s} {'ms/call':>8s} {'ms/step':>8s}",
          flush=True)
    for name in sorted(PHASES, key=lambda k: -sum(PHASES[k])):
        v = PHASES[name]
        print(f"  {name:<20s} {len(v):>7d} {sum(v) * 1e3:>9.1f} {sum(v) / len(v) * 1e3:>8.3f} "
              f"{sum(v) / steps * 1e3:>8.2f}", flush=True)
    print(f"  {'(sum)':<20s} {'':>7s} {total * 1e3:>9.1f}", flush=True)


atexit.register(summary)
