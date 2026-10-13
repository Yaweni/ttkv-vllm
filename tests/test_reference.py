"""Does the reference pre-rotary decode actually decode?

Three checks, and the third is the one that makes the first two worth anything:

1. the reference and the `[head, group, dim]` oracle agree -- same maths, two index orders, so
   agreement tests the (head, token) ordering and the rotation broadcast;
2. a row rotated by the WRONG position changes the answer -- otherwise the rotation is not being
   applied per row and check 1 is vacuous;
3. the decode still tracks the UNCOMPRESSED attention it approximates -- so the format is doing its
   job and the checks above are not merely self-consistent.

Runs under pytest or as a plain script (`python tests/test_reference.py`).
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
# Support BOTH layouts: <root>/vllm_ttkv_plugin with the codec in <root>/integration (this project),
# and a flat checkout where the codec sits beside the plugin (the GPU box).
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots, pack_v_slots  # noqa: E402
from vllm_ttkv.reference import (attend_raw_store, dense_v_from_slots, oracle_attend,  # noqa: E402
                                 raw_scores, rope_tables)

H, D, HQ, N = 8, 128, 32, 96
SEED = 0


def _fixture(cfg_name="campaign4x", n=N, seed=SEED):
    """A packed store plus a post-rotary query and per-row angle tables."""
    torch.manual_seed(seed)
    cfg = CONFIGS[cfg_name]
    # K is LOW-RANK ALONG TOKENS plus a noise floor, because that is the property the codec
    # exploits and white noise is its documented worst case (`ttkv_codec.self_test` says so
    # itself). A purely random K would make this fixture measure the noise floor rather than the
    # format -- it scored cosine 0.96 on `campaign4x` that way, which says nothing about the codec.
    rank = 16
    K = ((torch.randn(n, rank) * 0.2) @ (torch.randn(rank, D * H) * 0.2)
         ).reshape(n, H, D).permute(1, 0, 2).contiguous()
    K = K + 0.02 * torch.randn(H, n, D)
    # V is high-rank per token by measurement, so random V is the right fixture for it.
    V = torch.randn(H, n, D) * 0.2
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=D)
    vl = pc.build_v_layer(V, cfg["k_lanes"], cfg["vbits"], d=D)
    lay = layout_for_config(cfg, r1=kl.r1)

    K_pre = K.clone()                                     # what actually went in, un-rotated
    cos, sin = rope_tables(torch.arange(n), D)
    q_pre = torch.randn(HQ, D) * 0.1
    pos = torch.arange(n)
    # the query is POST-rotary: it is the token at position n (the next one to be generated)
    q_post = pc.rope_apply(q_pre.unsqueeze(0), cos[:1], sin[:1]).squeeze(0)

    return dict(kl=kl, vl=vl, lay=lay, K_pre=K_pre, V=V, cos=cos, sin=sin,
                q_post=q_post, pos=pos,
                k_slots=pack_k_slots(lay, kl.coeff_q, kl.coeff_scale, kl.resid),
                v_slots=pack_v_slots(lay, vl.rows))


def test_reference_matches_the_dense_oracle():
    """Two index orders, one answer. This is the check that would catch a (head, token) slip."""
    f = _fixture()
    a = attend_raw_store(f["lay"], f["k_slots"], f["v_slots"], f["kl"].decoder_f(),
                         f["q_post"], f["cos"], f["sin"])
    b = oracle_attend(f["lay"], f["k_slots"], f["v_slots"], f["kl"].decoder_f(),
                      f["q_post"], f["cos"], f["sin"])
    rel = ((a - b).abs().max() / b.abs().max()).item()
    assert rel < 1e-5, f"reference and oracle disagree by {rel:.3e} -- suspect the row ordering"
    print(f"  reference == oracle: rel err {rel:.2e}")


def test_wrong_position_rotation_changes_the_answer():
    """Two negative controls, and they are asserted on the SCORES, not on the output.

    (a) Omitting the rotation entirely -- contracting a post-rotary query against position-free
        keys. That is the mistake `ttkv_attn.py::_refuse_pre_storage` exists to prevent, and the one
        a reviewer should worry about, because the result is finite, correctly shaped and plausible.
    (b) Rotating every row by a NEIGHBOUR's position.

    Both are measured on `raw_scores` rather than on the softmax output, because the output is
    genuinely insensitive: with flat weights a score perturbation barely moves a weighted average of
    zero-mean rows. Asserting on the output made this control pass at 1.7e-2 for a mistake as large
    as skipping the rotation, which is a control that does not control for much.
    """
    f = _fixture()
    kw = dict(lay=f["lay"], k_slots=f["k_slots"], decoder_f=f["kl"].decoder_f(), query=f["q_post"])
    good = raw_scores(cos=f["cos"], sin=f["sin"], **kw)

    # (a) no rotation at all: cos=1, sin=0 leaves x unchanged
    unrotated = raw_scores(cos=torch.ones_like(f["cos"]), sin=torch.zeros_like(f["sin"]), **kw)
    rel_a = ((good - unrotated).abs().max() / good.abs().max()).item()
    assert rel_a > 0.5, (
        f"omitting the per-row rotation changed the scores by only {rel_a:.2e}. If a raw store "
        f"could be scored without rotating it, the pre-rotary design would be pointless -- so this "
        f"failing means the fixture is not exercising the rotation")
    print(f"  un-rotated scores (the _refuse_pre_storage mistake): {rel_a:.3f} relative change")

    # (b) every row rotated by its neighbour's position
    rolled = raw_scores(cos=f["cos"].roll(1, 0), sin=f["sin"].roll(1, 0), **kw)
    rel_b = ((good - rolled).abs().max() / good.abs().max()).item()
    assert rel_b > 1e-2, (
        f"rotating by the wrong position changed the scores by only {rel_b:.2e}: the rotation is "
        f"not being applied per row, so the agreement test above is vacuous")
    print(f"  neighbour-position scores: {rel_b:.3f} relative change (fp32 noise is ~1e-7)")


def test_k_decode_tracks_attention_over_the_raw_keys():
    """Fidelity of the K path, with V held IDENTICAL so this isolates K.

    What it establishes is the compression claim, not the decode claim: reconstruct -> residual ->
    rotate -> contract tracks attention over the raw keys, and the residual gap is the K compression
    error (~0.03 mean relative on this fixture). The decode itself is already pinned to fp32 noise
    against the container by `test_reference_matches_the_dense_oracle` -- an earlier version of this
    test used the raw keys on both sides and so was measuring compression while claiming to measure
    decode.

    Deliberately kept separate from the END-TO-END number, and the reason is the finding worth
    recording: end to end (raw K *and* raw V) the cosine is 0.959, and that gap is **entirely the V
    side** -- V keeps 64 of 128 lanes at 4 bits, so its relative error is 0.267 against the K side's
    0.030, nine times larger, and V is what the output is literally made of. On these settings the V
    lane count is the dominant fidelity lever, not the K rank.

    (Fixture caveat: random V is a PESSIMISTIC stand-in. `POST_ROPE_RATIO_PACKET.md` records `safe`
    at 81 of 128 lanes with no measurable loss on real V, so real V is more compressible than random
    V. The direction of the finding holds; the magnitude does not.)
    """
    f = _fixture()
    got = attend_raw_store(f["lay"], f["k_slots"], f["v_slots"], f["kl"].decoder_f(),
                           f["q_post"], f["cos"], f["sin"])

    # V must be byte-identical on both sides or this measures compression. Same dtype on both
    # sides: `materialise` defaults to fp16, which would round one side and not the other.
    V_store = dense_v_from_slots(f["lay"], f["v_slots"])
    assert torch.equal(V_store, f["vl"].materialise(H, dtype=torch.float32)), \
        "the V slot round-trip is not exact, so this test cannot isolate the K path"

    p_all = torch.arange(HQ) // (HQ // H)

    def attend(K_post, Vx):
        s = torch.einsum("npd,pd->np", K_post[:, p_all, :], f["q_post"]) * D ** -0.5
        return torch.einsum("np,npd->pd", torch.softmax(s, dim=0),
                            Vx.permute(1, 0, 2)[:, p_all, :])

    K_raw_post = pc.rope_apply(f["K_pre"].permute(1, 0, 2), f["cos"].unsqueeze(1),
                               f["sin"].unsqueeze(1))                    # per-row position

    def cos_sim(a, b):
        return torch.nn.functional.cosine_similarity(
            a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()

    cos_k = cos_sim(got, attend(K_raw_post, V_store))
    cos_e2e = cos_sim(got, attend(K_raw_post, f["V"]))
    assert cos_k > 0.999, (
        f"the K path only reaches cosine {cos_k:.6f} against attention over the raw keys -- the "
        f"compression is not approximating what it replaces")

    k_relerr = ((f["kl"].materialise(H).float() - f["K_pre"]).abs().mean()
                / f["K_pre"].abs().mean()).item()
    v_relerr = ((V_store - f["V"]).abs().mean() / f["V"].abs().mean()).item()
    print(f"  vs raw keys, V identical (K path only): cosine {cos_k:.6f}")
    print(f"  vs raw keys AND raw V (end to end)    : cosine {cos_e2e:.6f}  (V-limited)")
    print(f"  component cost: K rel err {k_relerr:.4f}, V rel err {v_relerr:.4f}")


if __name__ == "__main__":
    for fn in (test_reference_matches_the_dense_oracle,
               test_wrong_position_rotation_changes_the_answer,
               test_k_decode_tracks_attention_over_the_raw_keys):
        print(f"{fn.__name__}:")
        fn()
    print("\nall reference-decode tests passed")
