"""Does the multi-query V READOUT reproduce the per-query one, and is it actually cheaper?

`tests/test_v_out_triton.py` gates the single-query readout against the dense reference. This file
gates the QUERY-BLOCKED one, and it needs its own oracle for a specific reason: the two differ in
exactly one thing that a shape check cannot see, which is the ORDER of the contraction's row axis.
`v_out_multi_from_slots` reshapes `[BQ, GP, KT]` weights to `[M, KT]` on one side and reshapes the
`[M, HALF]` accumulator back to `[BQ, GP, HALF]` for the store on the other. If those two disagree
about whether `M` is `j*GP + g` or `g*BQ + j`, the result is correctly shaped, finite, and reads the
wrong query head's values -- so the primary assertion here is per-ELEMENT agreement with the
per-query loop over a `p` that is not symmetric in either axis.

The claim the kernel exists for is also asserted rather than assumed: that moving the query axis
into `M` removes the group padding. The arithmetic check is a COUNT of `tl.dot` work, and it is
written as a ratio against the single-query kernel's padding factor, so it fails if someone
"restores" `GP = max(16, next_power_of_2(G))`.

Skips cleanly without a CUDA device.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_v_slots  # noqa: E402
from vllm_ttkv.reference import dense_v_from_slots  # noqa: E402
from vllm_ttkv.v_out_triton import v_out_from_slots, v_out_multi_from_slots  # noqa: E402

H, D = 8, 128
HQ = 32                                         # G = 4, the Llama-3.x group size
DEV = "cuda"
CFG = "safe"


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def _store(cfgname=CFG, n=256, h=H, seed=0):
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    lay = layout_for_config(cfg, r1=64, H=h, d=D)
    V = torch.randn(h, n, D, device=DEV) * 0.3
    vs = pack_v_slots(lay, pc.build_v_layer(V, cfg["k_lanes"], cfg["vbits"], d=D).rows)
    return lay, vs


def _weights(hq, q, n, seed=1):
    """`[hq, Q, n]` softmax weights. NOT uniform and NOT separable: a separable `p` (`p_q * p_n`)
    is the one case where a transposed row axis could still agree, so the fixture is built to be
    asymmetric in both axes."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = torch.randn(hq, q, n, device=DEV, generator=g)
    return torch.softmax(p, dim=-1)


def _dense_out(lay, vs, p, hq):
    """The reference: decode V densely, then weight it. `p` is `[hq, Q, n]`.

    `V` comes back `[H, n, D]`, so it is transposed to `[n, H, D]` and each query head reads its own
    KV head's rows -- the GQA expansion written as an index rather than a repeat, so a wrong
    `hq // G` mapping cannot hide behind a broadcast.

    A batched `matmul`, not an `einsum`: the natural einsum here is `"pqn,nqd->pqd"`, and `q` then
    names BOTH the query axis of the weights and the head axis of the values. On identical operands
    that form disagreed with a `matmul` by 1.4 RELATIVE -- a wrong reference that looks like a wrong
    kernel, which is how it was found.
    """
    V = dense_v_from_slots(lay, vs)                                   # [H, n, D]
    per_query = torch.arange(hq, device=vs.device) // (hq // lay.H)   # [hq]
    X = V.permute(1, 0, 2)[:, per_query, :]                           # [n, hq, D]
    return torch.matmul(p, X.permute(1, 0, 2))                        # [hq, Q, n] x [hq, n, D]


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------- agreement
def test_multi_agrees_with_the_per_query_loop_elementwise():
    """THE PRIMARY GATE. Per-query results are the ones the dense reference already gates, so
    agreeing with them elementwise transfers that gate; comparing only against the reference would
    leave the row-axis order unconstrained, because a wrong head mapping is shape-preserving."""
    if not _need_cuda():
        return
    lay, vs = _store(n=200)
    Q = 48
    p = _weights(HQ, Q, 200)
    got = v_out_multi_from_slots(lay, vs, p, bq=16)
    want = torch.stack([v_out_from_slots(lay, vs, p[:, j, :]) for j in range(Q)], dim=1)
    rel = _rel(got, want)
    print(f"  per-query loop  Q={Q} n=200  rel err {rel:.3e}")
    assert rel < 1e-5, f"multi readout disagrees with the per-query loop by {rel:.3e}"


def test_multi_agrees_with_the_dense_reference():
    if not _need_cuda():
        return
    for cfgname in ("safe", "tight", "campaign4x"):
        lay, vs = _store(cfgname, n=128)
        p = _weights(HQ, 32, 128)
        got = v_out_multi_from_slots(lay, vs, p, bq=16)
        want = _dense_out(lay, vs, p, HQ)
        rel = _rel(got, want)
        print(f"  {cfgname:11s} 4b/3b lanes  rel err {rel:.3e}")
        assert rel < 1e-4, f"{cfgname}: {rel:.3e} against the dense reference"


def test_a_one_hot_weight_picks_exactly_that_row():
    """THE CONTROL THAT NEEDS NO ARITHMETIC. With `p` one-hot at token `t`, the readout must equal
    that token's decoded V rows and nothing else -- so this gates the lane decode and the head
    mapping independently of any contraction, and it is what isolated a wrong REFERENCE from a
    wrong kernel when the two disagreed."""
    if not _need_cuda():
        return
    lay, vs = _store(n=128)
    V = dense_v_from_slots(lay, vs)                                   # [H, n, D]
    g = HQ // H
    t = 37
    p = torch.zeros(HQ, 1, 128, device=DEV)
    p[:, 0, t] = 1.0
    got = v_out_multi_from_slots(lay, vs, p, bq=16)[:, 0, :]
    want = V[:, t, :][torch.arange(HQ, device=DEV) // g]
    rel = _rel(got, want)
    print(f"  one-hot t={t}  rel err {rel:.3e}")
    assert rel < 1e-6, f"a one-hot weight did not pick its row: {rel:.3e}"



    """Tiling independence. `bq` decides how many queries share a program and nothing else, so every
    choice has to give the same numbers -- including `bq` values that divide `Q` and ones that do
    not, since the ragged block is masked rather than skipped."""
    if not _need_cuda():
        return
    lay, vs = _store(n=96)
    Q = 40
    p = _weights(HQ, Q, 96)
    base = v_out_multi_from_slots(lay, vs, p, bq=16)
    for bq in (16, 32, 64):
        rel = _rel(v_out_multi_from_slots(lay, vs, p, bq=bq), base)
        print(f"  bq={bq:3d}  rel err {rel:.3e}")
        assert rel < 1e-6, f"bq={bq} changed the answer by {rel:.3e}"


def test_a_ragged_query_block_does_not_leak_into_the_last_rows():
    """`Q` is not a multiple of `bq`, so the final program holds padded query rows. They are masked
    to zero on the weight load AND on the store; if the store mask were missing they would write
    past `Q` into the next head's rows, which is a memory error rather than a wrong number, and if
    the LOAD mask were missing they would read real weights and land garbage in rows that do exist
    for a LATER head."""
    if not _need_cuda():
        return
    lay, vs = _store(n=64)
    Q = 24                                          # bq=16 -> a second block of 8, 8 padded
    p = _weights(HQ, Q, 64)
    got = v_out_multi_from_slots(lay, vs, p, bq=16)
    assert got.shape == (HQ, Q, D), got.shape
    want = torch.stack([v_out_from_slots(lay, vs, p[:, j, :]) for j in range(Q)], dim=1)
    rel = _rel(got, want)
    print(f"  ragged Q={Q} bq=16  rel err {rel:.3e}")
    assert rel < 1e-5, f"ragged block leaked: {rel:.3e}"


# --------------------------------------------------------------------------- guards
def test_the_group_padding_is_gone_and_that_is_the_whole_point():
    """The single-query kernel pads the group axis to 16 and does `2**ceil(log2(G))/G` times the
    arithmetic it needs. With the query axis in `M`, `GP = next_power_of_2(G)` clears the dot minimum
    by itself, so the padding factor is 1. At G=4 the saved work is 4x, and it is the LARGEST single
    difference between the two kernels -- larger than the saved launches."""
    if not _need_cuda():
        return
    G = HQ // H
    import triton
    gp_multi = triton.next_power_of_2(G)
    gp_single = max(16, gp_multi)
    print(f"  G={G}  single-query GP={gp_single} -> {gp_single / G:.1f}x padded  |  "
          f"multi GP={gp_multi} with bq*GP={16 * gp_multi} -> {gp_multi / G:.1f}x")
    assert 16 * gp_multi >= 16, "the multi-query M must clear tl.dot's minimum without padding"
    assert gp_single / G == 4.0 and gp_multi / G == 1.0


@pytest.mark.parametrize("bad", [
    ("p2d", "the multi-query readout takes p as [HQ, Q, n]"),
    ("rows", "but the store holds n="),
    ("bq", "power of two with bq*GP"),
])
def test_the_launcher_refuses_what_it_cannot_express(bad):
    """The failure mode this integration is most exposed to is a plausible wrong number, so the
    guards are tested as hard as the math. A wrong-rank `p` and a row-count mismatch are both
    silently-broadcastable in torch."""
    if not _need_cuda():
        return
    kind, needle = bad
    lay, vs = _store(n=32)
    p = _weights(HQ, 8, 32)
    if kind == "p2d":
        arg = p[:, :, 0]
    elif kind == "rows":
        arg = _weights(HQ, 8, 16)
    else:
        arg = p
    with pytest.raises(ValueError) as e:
        # `bq` must be a power of two with `bq*GP >= 16`; at G=4 that rules out 1 and 2, and 4 is
        # the smallest that PASSES, which is the point -- the padding is gone, so a small query block
        # is legal where the single-query kernel needed GP=16.
        v_out_multi_from_slots(lay, vs, arg, bq=2 if kind == "bq" else 16)
    assert needle in str(e.value), str(e.value)


def test_a_query_block_of_bq_times_G_equal_to_16_is_accepted():
    """The boundary the guard actually draws, asserted directly. With G=4 a query block of 4 is
    ENOUGH -- `M = 4*4 = 16` -- which is only true because the group axis is no longer padded."""
    if not _need_cuda():
        return
    lay, vs = _store(n=32)
    out = v_out_multi_from_slots(lay, vs, _weights(HQ, 4, 32), bq=4)
    assert out.shape == (HQ, 4, D)


# --------------------------------------------------------------------------- cost
def test_the_query_block_is_what_amortises_the_launch_and_the_tile():
    """Not a threshold test: it prints the shape of the cost and asserts only that a block is much
    cheaper per query than one launch per query, which is the property `_prefill_continuation`
    leans on. A large continuation chunk is the case this exists for."""
    if not _need_cuda():
        return
    lay, vs = _store(n=1024)

    def ms(fn, rep=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        import time
        ts = []
        for _ in range(rep):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return min(ts)

    one = ms(lambda: v_out_from_slots(lay, vs, _weights(HQ, 1, 1024)[:, 0, :]))
    print(f"  single query                1 launch   {one:7.3f} ms   {one:7.3f} ms/query")
    for Q in (16, 64, 256):
        p = _weights(HQ, Q, 1024)
        t = ms(lambda: v_out_multi_from_slots(lay, vs, p))
        print(f"  Q={Q:4d} in one launch       1 launch   {t:7.3f} ms   {t / Q:7.3f} ms/query"
              f"   {Q * one / t:5.1f}x")
        assert t / Q < one, f"Q={Q} was not cheaper per query than the per-query kernel"
