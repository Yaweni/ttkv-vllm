"""Does the Triton V READOUT agree with the dense reference, and is the lane decode really per row?

`tests/test_krt_triton.py` gates the score kernel. This gates the other half of the pre-rotary
decode, and it needs its own file for the same reason: the two kernels share the lane decoder's
BYTE conventions but nothing else, so a bug in one cannot be caught by the other's oracle. `krt`
builds keys from coefficients and rotates them; `v_out` has no basis, no rotation, and a different
slot preamble (a V slot has no coefficient block).

The oracle is `reference.dense_v_from_slots`, which reaches the same store through
`layout.unpack_v_slots` -- the exact inverse of the packing under test. So agreement checks the
whole chain: bitmap, MSB-first code stream, per-row fp16 ruler at an odd byte offset, the two-half
split with its rank carry, and the GQA-weighted reduction.

Everything here is a NEGATIVE-CONTROL-shaped assertion, because the values at stake are small
integers: a wrong bit order still produces small plausible numbers, and a readout that used one
global ruler instead of each row's own would be off by a factor, not by a shape.

Skips cleanly without a CUDA device, so it is safe in a CPU-only run.
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_v_slots, packed_row_bytes  # noqa: E402
from vllm_ttkv.reference import dense_v_from_slots  # noqa: E402
from vllm_ttkv.v_out_triton import v_out_from_slots  # noqa: E402

H, D = 8, 128
HQ = 32                                    # G = 4, the Llama-3.1-8B group size
DEV = "cuda"


def _v_store(cfgname, n, h=H, dev=DEV, seed=0):
    """A V layer with real magnitude structure: V is high-rank per token, so unlike K there is no
    low-rank-plus-noise fixture that exercises the honest case."""
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    V = torch.randn(h, n, D, device=dev) * 0.3
    return cfg, pc.build_v_layer(V, cfg["k_lanes"], cfg["vbits"], d=D)


def _dense_out(lay, vs, p, hq):
    """The reference readout: decode V densely, then weight it. `p` is `[hq, n]`.

    `V` comes back as `[H, n, D]`, so it is transposed to `[n, H, D]` and each query head reads its
    own KV head's rows -- the GQA expansion written as an index rather than as a repeat, so a wrong
    `hq // G` mapping cannot hide behind a broadcast.
    """
    V = dense_v_from_slots(lay, vs)                                   # [H, n, D]
    per_query = torch.arange(hq, device=vs.device) // (hq // lay.H)   # [hq]
    return torch.einsum("pn,npd->pd", p, V.permute(1, 0, 2)[:, per_query, :])


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def test_v_out_matches_the_dense_reference_for_every_shipped_config():
    """Every config and several lengths, because they differ in exactly the fields this kernel
    parses: `v_lanes` sets PB and the per-head row stride, `v_bits` sets the code width and whether
    the ruler lands on an odd byte offset. `campaign4x` is 4-bit; `safe`/`tight` are 3-bit, which is
    the width where the row is NOT a whole number of bytes and the codes straddle byte boundaries."""
    if not _need_cuda():
        return
    n = 96
    for cfgname in ("safe", "tight", "campaign4x"):
        cfg, vl = _v_store(cfgname, n)
        lay = layout_for_config(cfg, r1=64, H=H, d=D)
        vs = pack_v_slots(lay, vl.rows)
        torch.manual_seed(1)
        p = torch.softmax(torch.randn(HQ, n, device=DEV), dim=-1)

        got = v_out_from_slots(lay, vs, p)
        want = _dense_out(lay, vs, p, HQ)

        assert tuple(got.shape) == (HQ, D), f"got {tuple(got.shape)}, want {(HQ, D)}"
        rel = ((got - want).abs().max() / want.abs().max()).item()
        pb = packed_row_bytes(lay.v_lanes, lay.v_bits)
        row_stride = D // 8 + pb + 2
        print(f"  {cfgname:<11s} lanes={lay.v_lanes:<3d} bits={lay.v_bits} PB={pb:<3d} "
              f"row_stride={row_stride:<3d} slot={lay.v_slot_bytes:<4d} ruler@{D // 8 + pb:<3d}  "
              f"rel err {rel:.3e}")
        assert rel < 1e-5, (
            f"{cfgname}: the Triton V readout disagrees with the dense reference by {rel:.3e}. "
            f"The lane decode has four places to go wrong silently -- bit order, the rank carry "
            f"across the half split, the odd-offset fp16 ruler, and the group weighting.")
        # The two configs that matter must actually exercise the straddling case, or this test is
        # passing on the easy path. 3 bits over an odd lane count is where codes cross bytes.
        if lay.v_bits == 3:
            assert (lay.v_lanes * lay.v_bits) % 8 != 0, (
                f"3-bit config with {lay.v_lanes} lanes is byte-aligned, so the straddle path is "
                f"not covered by this fixture")


def test_group_size_that_is_not_a_power_of_two():
    """Qwen2.5-0.5B is 14 heads over 2 kv heads, i.e. G=7 -- and `tl.dot` needs M >= 16, so the
    weight rows are padded to 16 with exact zeros and the pad rows are dropped on store. `tl.arange`
    also requires a power-of-two extent, so a G that is neither can only work through this path.

    The pad must be EXACT, not merely small: a pad row that got a real weight would add a phantom
    query head's output into a head that exists.
    """
    if not _need_cuda():
        return
    h, hq, n = 2, 14, 64                       # G = 7
    cfg, vl = _v_store("safe", n, h=h)
    lay = layout_for_config(cfg, r1=64, H=h, d=D)
    vs = pack_v_slots(lay, vl.rows)
    torch.manual_seed(2)
    p = torch.softmax(torch.randn(hq, n, device=DEV), dim=-1)

    got = v_out_from_slots(lay, vs, p)
    want = _dense_out(lay, vs, p, hq)
    assert tuple(got.shape) == (hq, D), f"got {tuple(got.shape)}, want {(hq, D)}"
    rel = ((got - want).abs().max() / want.abs().max()).item()
    print(f"  H={h} HQ={hq} G=7 (padded to 16)  rel err {rel:.3e}")
    assert rel < 1e-5, f"G=7 (non-power-of-two) disagrees with the reference by {rel:.3e}"


def test_each_row_uses_its_own_ruler_and_not_a_global_one():
    """Negative control on the PER-ROW ruler, which is the whole reason a V slot carries one fp16
    value per (head, token) instead of one per tensor. A kernel that read row 0's ruler for every
    row would still return finite, correctly-shaped output -- so this perturbs ONE row's ruler,
    recomputes the oracle from the perturbed store, and requires the kernel to follow.

    It also asserts the perturbation actually MOVES the answer, otherwise the comparison above would
    pass against an unperturbed kernel too.
    """
    if not _need_cuda():
        return
    n = 96
    cfg, vl = _v_store("safe", n)
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    torch.manual_seed(3)
    p = torch.softmax(torch.randn(HQ, n, device=DEV), dim=-1)

    base = pack_v_slots(lay, vl.rows)
    before = _dense_out(lay, base, p, HQ)

    # Row 37 is head 4, token 5. Scaling its ruler scales every kept lane of that row, so the row's
    # contribution to every query head in group 4 changes by 3x.
    victim = 4 * n + 5
    vl.rows.scale[victim] = (vl.rows.scale[victim].float() * 3.0).half()
    after_slots = pack_v_slots(lay, vl.rows)
    after = _dense_out(lay, after_slots, p, HQ)

    moved = ((after - before).abs().max() / before.abs().max()).item()
    assert moved > 1e-2, (
        f"perturbing one row's ruler moved the reference by only {moved:.2e}, so this fixture does "
        f"not exercise the per-row ruler at all and the check below would be vacuous")

    got = v_out_from_slots(lay, after_slots, p)
    rel = ((got - after).abs().max() / after.abs().max()).item()
    stale = ((got - before).abs().max() / before.abs().max()).item()
    print(f"  perturbed rel err {rel:.3e} (vs {stale:.3e} if the ruler were ignored)")
    assert rel < 1e-5, f"the kernel did not follow the perturbed ruler: rel err {rel:.3e}"


def test_the_bitmap_decides_which_lanes_a_rows_codes_land_on():
    """Negative control on the BITMAP.

    The codes for a row do not say WHERE they go -- the bitmap does, and a row's `v_lanes` set bits
    are always exactly `v_lanes`. So the bitmap is PERMUTED here rather than zeroed: reversing a
    row's bitmap bytes keeps the population count (so the row stays representable) while moving every
    code to a different dimension. A kernel that placed codes by rank order without consulting the
    bitmap would put them in the wrong lanes and still return the right shape.

    Zeroing a row's bitmap is deliberately NOT tested, because the format forbids it:
    `PackedRows.build` asserts exactly `k` lanes per row, so a row with none is not a state a V slot
    can hold. Testing it would be testing a codec assertion rather than the kernel.
    """
    if not _need_cuda():
        return
    n = 64
    cfg, vl = _v_store("campaign4x", n)
    lay = layout_for_config(cfg, r1=64, H=H, d=D)
    torch.manual_seed(4)
    p = torch.softmax(torch.randn(HQ, n, device=DEV), dim=-1)

    before = v_out_from_slots(lay, pack_v_slots(lay, vl.rows), p)

    victim_head, victim_tok = 2, 9
    vector = victim_head * n + victim_tok
    popcount_before = sum(bin(int(b)).count("1") for b in vl.rows.bitmap[vector].tolist())
    vl.rows.bitmap[vector] = vl.rows.bitmap[vector].flip(0)
    popcount_after = sum(bin(int(b)).count("1") for b in vl.rows.bitmap[vector].tolist())
    assert popcount_before == popcount_after, (
        f"the bitmap permutation changed the lane count ({popcount_before} -> {popcount_after}), so "
        f"the row is no longer representable and the oracle cannot decode it")

    moved_slots = pack_v_slots(lay, vl.rows)
    got = v_out_from_slots(lay, moved_slots, p)
    want = _dense_out(lay, moved_slots, p, HQ)

    rel = ((got - want).abs().max() / want.abs().max()).item()
    G = HQ // H
    others = [h for h in range(HQ) if not (victim_head * G <= h < (victim_head + 1) * G)]
    untouched = ((got[others] - before[others]).abs().max() / before.abs().max()).item()
    touched = ((got[victim_head * G:(victim_head + 1) * G]
                - before[victim_head * G:(victim_head + 1) * G]).abs().max()
               / before.abs().max()).item()
    print(f"  rel err {rel:.3e}; group {victim_head} moved {touched:.3e}, "
          f"other heads moved {untouched:.3e}")
    assert rel < 1e-5, f"permuted-bitmap readout disagrees with the reference by {rel:.3e}"
    assert untouched == 0.0, (
        f"permuting head {victim_head}'s row moved the OTHER heads by {untouched:.3e}; the bitmap "
        f"is not being applied per (head, token) row")
    assert touched > 1e-3, (
        f"permuting a row's lanes moved its own group by only {touched:.3e}, so the check above is "
        f"vacuous")


if __name__ == "__main__":
    print(f"v_out kernel validation (CUDA available: {torch.cuda.is_available()})")
    for fn in (test_v_out_matches_the_dense_reference_for_every_shipped_config,
               test_group_size_that_is_not_a_power_of_two,
               test_each_row_uses_its_own_ruler_and_not_a_global_one,
               test_the_bitmap_decides_which_lanes_a_rows_codes_land_on):
        print(f"{fn.__name__}:")
        fn()
    print("\nv_out kernel validation complete")
