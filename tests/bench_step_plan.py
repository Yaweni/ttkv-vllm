"""What the PLAN path costs on the host -- and how much of `plan` is work rather than waiting.

`impl._Step` does the whole step's addressing ONCE: the batch split, the block-table readback, every
request's slot list, the angle rows. That is the right shape -- it is why 28 layers share one
readback instead of paying per layer -- but it is also why `plan` shows up in the profile at all, and
the profile cannot say how much of that number is work.

It cannot, because `qsl`, `seq_lens` and `block_table` are read back with `.tolist()`, and a
device-to-host copy on a stream cannot start until every kernel already queued has finished. So the
host time attributed to `plan` is (the op) + (whatever the GPU still owes), and on a deep queue the
second term is the whole number. This instrument separates the two by measuring the same calls at two
queue depths:

    queue empty   `synchronize()` before each call, so the number is the op's own cost
    queue deep    ~`LOAD` matmuls queued first, so the number is what the readback actually pays

The gap is the answer to "optimise this or delete it": if the empty column is microseconds and the
deep column is the queue depth, the readback is a WAIT and the only lever is doing fewer of them.
`block_slots_host` is measured the same way and split into its two halves -- the O(seq_len) Python
list build and the host-to-device upload -- because that part is real work at any queue depth.

The two contexts are the ones the harness uses (`--repeats 60` and `--repeats 240`), since the whole
question is how this scales with context: the batch split is O(1) and the slot list is O(seq_len).

`PER_STEP` is the metadata-object count per step measured in the engine (112 objects over 16 decode
steps), because everything here is charged once per OBJECT, not once per step: 7 objects means 21
readbacks and 7 slot builds per step.

USAGE
    PYTHONPATH=/opt/ttkv/integration python tests/bench_step_plan.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from vllm_ttkv import decode as dc  # noqa: E402

BLOCK = 16
POOL_BLOCKS = 4096
CONTEXTS = (1082, 4322)
REPS = 30
LOAD = 8                       # 2048^2 matmuls queued to give the readbacks a real drain to pay
PER_STEP = 7                   # metadata objects per step in the engine (112 objects / 16 steps)


def _empty(fn, reps=REPS) -> float:
    """ms per call with the queue EMPTY: sync first, so the number is the op's own cost."""
    acc = 0.0
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        acc += time.perf_counter() - t
    return acc / reps * 1e3


def _deep(fn, a, reps=REPS, load=LOAD) -> float:
    """ms per call with the queue DEEP: the op pays the queued work before it can read back."""
    acc = 0.0
    for _ in range(reps):
        for _ in range(load):
            torch.mm(a, a)
        t = time.perf_counter()
        fn()
        acc += time.perf_counter() - t
    return acc / reps * 1e3


def _queue_ms(a, load=LOAD) -> float:
    """How deep that load actually is here, measured rather than assumed."""
    for _ in range(load):
        torch.mm(a, a)
    t = time.perf_counter()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def _list_only(row, seq_len, block_size):
    """The host half of `block_slots_host`: one `range` per block, extended, then truncated."""
    out = []
    for b in row:
        out.extend(range(b * block_size, b * block_size + block_size))
    return out[:seq_len]


def _step_build_host(table, qsl, seq_lens, sm, seq_len, n_slots, device):
    """`impl._Step` as it was: the three readbacks, then the slot list built on the HOST."""
    q = qsl.tolist()
    s = seq_lens.tolist()
    rows = table.tolist()
    sm[:q[-1]]
    dc.block_slots_host(rows[0], s[0], BLOCK, device, n_slots)
    return q, s, rows


def _step_build_device(table, qsl, seq_lens, sm, seq_len, n_slots, device):
    """`impl._Step` now: the same readbacks, then the slots built on the DEVICE from the row tensor.

    The readbacks are identical in both -- they are needed for the decoder key and for the checks --
    so the difference between these two rows is exactly what moving the build bought.
    """
    q = qsl.tolist()
    s = seq_lens.tolist()
    rows = table.tolist()
    sm[:q[-1]]
    dc.check_row_host(rows[0], s[0], BLOCK, n_slots)
    dc.block_slots(table[0], s[0], BLOCK, check=False)
    return q, s, rows


def _case(seq_len: int, dev) -> None:
    n_blocks = -(-seq_len // BLOCK)
    # A row as the allocator hands it over: non-contiguous block ids, then the rest of the row is
    # vLLM's padding sentinel. Non-contiguous matters -- it is what makes the slot list a gather.
    used = torch.randperm(POOL_BLOCKS, device=dev)[:n_blocks].to(torch.int32)
    table = used.unsqueeze(0).contiguous()
    row = [int(b) for b in used]
    qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    sm = torch.tensor([0], dtype=torch.int64, device=dev)
    n_slots = POOL_BLOCKS * BLOCK
    a = torch.empty(2048, 2048, device=dev)

    ref = dc.block_slots_host(row, seq_len, BLOCK, dev, n_slots)
    dev_slots = dc.block_slots(table[0], seq_len, BLOCK, check=False)
    agree = torch.equal(ref, dev_slots)
    if not agree:
        raise AssertionError("the device build and the host build disagree; the swap is not safe")

    print(f"\n=== {seq_len} tokens ({n_blocks} blocks of {BLOCK}) ===")
    print(f"  queue depth for the deep column: {_queue_ms(a):.1f} ms\n")
    print(f"  {'op or phase':46s} {'empty us':>9s} {'deep us':>9s} {'x7/step ms':>11s}")
    rows = [
        ("qsl.tolist()  (2 int32)", lambda: qsl.tolist()),
        ("seq_lens.tolist()  (1 int32)", lambda: seq_lens.tolist()),
        (f"block_table.tolist()  ([1, {n_blocks}])", lambda: table.tolist()),
        (f"slot list, python loop  ({seq_len} int)", lambda: _list_only(row, seq_len, BLOCK)),
        (f"slot list, list -> cuda ({seq_len} int)",
         lambda: torch.tensor(_list_only(row, seq_len, BLOCK), dtype=torch.long, device=dev)),
        ("block_slots_host  (both halves)",
         lambda: dc.block_slots_host(row, seq_len, BLOCK, dev, n_slots)),
        ("block_slots  (device, check=False)",
         lambda: dc.block_slots(table[0], seq_len, BLOCK, check=False)),
        ("one write's slot_mapping slice .tolist()",
         lambda: sm[:1].tolist()),
        ("WHOLE _Step build, host slots",
         lambda: _step_build_host(table, qsl, seq_lens, sm, seq_len, n_slots, dev)),
        ("WHOLE _Step build, device slots",
         lambda: _step_build_device(table, qsl, seq_lens, sm, seq_len, n_slots, dev)),
    ]
    for label, fn in rows:
        e = _empty(fn)
        d = _deep(fn, a)
        print(f"  {label:46s} {e * 1e3:9.1f} {d * 1e3:9.1f} {e * PER_STEP:11.2f}")
    print(f"  {'':46s} {'':9s} {'':9s} {'= ms/step at':>11s} {PER_STEP} objects")
    print(f"  {'':46s} {'':9s} {'':9s} {'':11s} agreement: {agree}")


def main() -> int:
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return 0
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  ({PER_STEP} metadata objects per step)")
    for seq_len in CONTEXTS:
        _case(seq_len, dev)
    print("\n  'empty' is the op's own cost; 'deep' is what it pays on a full queue. A readback whose"
          "\n  empty cost is microseconds cannot be made faster -- it can only be done less often.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
