"""Step definitions for `features/*.feature` -- the acceptance criteria as executable specs.

WHY THIS LAYER EXISTS SEPARATELY FROM `tests/test_*.py`
    `HANDOFF.md` section 5 states nine acceptance criteria in prose, and prose criteria rot: they
    cannot fail, so nobody notices when they stop being true. These scenarios are the same criteria
    written so that they CAN fail. The step bodies deliberately do not re-implement anything -- they
    call the same helpers the pytest layer calls, so a scenario cannot pass by testing a second
    implementation of the thing it claims to test.

WHY THE FEATURES ARE WORTH READING
    Each one names the mistake it is guarding against, because in this project the mistakes do not
    announce themselves. A mis-rotated key is still orthogonal and position-dependent; a
    carried-across config still produces a number; a store read with the wrong indexing still has the
    right shape. The scenarios exist to turn those into failures.

Needs `pytest-bdd`. Model scenarios additionally need `transformers` and a cached model, and skip
cleanly without them. Kernel scenarios skip without a CUDA device.
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
from vllm_ttkv.krt_triton import krt_from_slots  # noqa: E402
from vllm_ttkv.layout import TTKLayout, layout_for_config, pack_k_slots, packed_row_bytes  # noqa: E402
from vllm_ttkv.reference import raw_scores, rope_tables  # noqa: E402

from pytest_bdd import given, parsers, scenarios, then, when  # noqa: E402

# Step bodies are imported from the pytest layer rather than re-written, so the two cannot drift.
import test_e2e_forward as e2e  # noqa: E402
import test_krt_triton as kt  # noqa: E402

# `scenarios` binds every scenario in the file to this module by name.
scenarios("../features/slot_layout.feature")
scenarios("../features/pre_rotary_decode.feature")
scenarios("../features/end_to_end_forward.feature")

gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def ctx():
    return {}


# --------------------------------------------------------------------------- slot_layout
@given('the shipped config "safe"')
def _cfg_safe(ctx):
    ctx["cfg"] = CONFIGS["safe"]


@given('the shipped config "safe" with its absolute 81 lanes')
def _cfg_safe_lanes(ctx):
    ctx["cfg"] = CONFIGS["safe"]
    assert ctx["cfg"]["k_lanes"] == 81, "this scenario is about the absolute 81-lane allocation"


@when("I account for one token's K slot")
def _account(ctx):
    ctx["lay"] = layout_for_config(ctx["cfg"], r1=128, H=8, d=128)


@then("the slot bytes match the container's own row bytes")
def _match_container(ctx):
    lay = ctx["lay"]
    # Cross-checked against a PackedRows the CONTAINER built, not against the layout's own formula --
    # otherwise this asserts that a number equals itself.
    rows = 8 * 64
    mask = torch.zeros(rows, lay.d, dtype=torch.bool)
    mask.scatter_(1, torch.topk(torch.randn(rows, lay.d).abs(), lay.res_k, dim=1).indices, True)
    pr = pc.PackedRows.build(torch.randn(rows, lay.d), mask, lay.res_bits, lay.d, k=lay.res_k)
    assert pr.nbytes() // rows == lay.k_row_bytes, (
        f"the layout bills {lay.k_row_bytes} B per K row but the container's PackedRows stores "
        f"{pr.nbytes() // rows} B")


@when("I build a layout for head dim 64")
def _build_narrow(ctx):
    try:
        layout_for_config(ctx["cfg"], r1=128, H=8, d=64)
        ctx["raised"] = None
    except Exception as exc:                                                 # noqa: BLE001
        ctx["raised"] = exc


@then("the layout refuses it")
def _refused(ctx):
    assert isinstance(ctx["raised"], ValueError), (
        f"building the 81-lane config at d=64 gave {ctx['raised']!r}; it must be refused, because 81 "
        f"lanes do not exist in a 64-wide row and a silent acceptance would be a nonsense allocation")


@given("a fixed KV width and a fixed lane fraction")
def _fixed_width(ctx):
    ctx["width"] = 1024                    # H * d, the quantity the byte ledger is made of


@when("I build layouts for head dims 64, 128, 256 and 512")
def _sweep_dims(ctx):
    ctx["ratios"] = {}
    for d in (64, 128, 256, 512):
        lay = TTKLayout(H=ctx["width"] // d, d=d, r1=128, qbits=8,
                        res_k=int(d * 0.633), res_bits=3,
                        v_lanes=int(d * 0.633), v_bits=3)
        ctx["ratios"][d] = lay.ratio()


@then("every joint ratio stays between 4.2x and 4.6x")
def _ratios_in_band(ctx):
    bad = {d: r for d, r in ctx["ratios"].items() if not 4.2 <= r <= 4.6}
    print(f"  ratios by head dim: " + ", ".join(f"d={d} {r:.2f}x"
                                                for d, r in ctx["ratios"].items()))
    assert not bad, (
        f"the ratio moved with the head-dim split: {bad}. The invariant is the KV WIDTH (H*d), not "
        f"the split, because the bitmap is H*(d/8) and the residual lanes are H*d*frac")


# --------------------------------------------------------------------------- pre_rotary_decode
@given('a packed K store for the shipped config "campaign4x"')
def _packed_store(ctx):
    n = 96
    cfg, kl = kt._store("campaign4x", n)
    ctx["n"] = n
    ctx["kl"] = kl
    ctx["lay"] = layout_for_config(cfg, r1=kl.r1, H=kt.H, d=kt.D)
    ctx["ks"] = pack_k_slots(ctx["lay"], kl.coeff_q, kl.coeff_scale, kl.resid)


@given("post-rotary queries and per-row angle tables")
def _angles(ctx):
    n = ctx["n"]
    ctx["cos"], ctx["sin"] = rope_tables(torch.arange(n, device=kt.DEV), kt.D)
    ctx["q"] = torch.randn(kt.HQ, kt.D, device=kt.DEV) * 0.1


@gpu_only
@when("the Triton kernel decodes the store")
def _decode(ctx):
    lay, kl = ctx["lay"], ctx["kl"]
    ctx["got"] = krt_from_slots(lay, ctx["ks"], kl.decoder_f(), ctx["cos"], ctx["sin"], ctx["q"])
    ctx["ref"] = raw_scores(lay, ctx["ks"], kl.decoder_f(), ctx["q"], ctx["cos"],
                            ctx["sin"], scale=1.0).T
    ctx["rel"] = ((ctx["got"] - ctx["ref"]).abs().max() / ctx["ref"].abs().max()).item()


@then("its scores agree with the reference to 1e-4 relative")
def _agree(ctx):
    print(f"  Triton vs reference: {ctx['rel']:.3e}")
    assert ctx["rel"] < 1e-4, (
        f"the kernel and the reference disagree by {ctx['rel']:.3e}. Reconstruct, residual, "
        f"per-row rotation and contract are each a place this can go wrong while still returning "
        f"correctly-shaped scores")


@gpu_only
@when("the kernel decodes with the angle table rolled by one position")
def _decode_rolled(ctx):
    lay, kl = ctx["lay"], ctx["kl"]
    good = krt_from_slots(lay, ctx["ks"], kl.decoder_f(), ctx["cos"], ctx["sin"], ctx["q"])
    roll = krt_from_slots(lay, ctx["ks"], kl.decoder_f(),
                          ctx["cos"].roll(1, 0), ctx["sin"].roll(1, 0), ctx["q"])
    ctx["moved"] = ((good - roll).abs().max() / good.abs().max()).item()


@gpu_only
@when("the kernel decodes with an identity rotation")
def _decode_unrotated(ctx):
    lay, kl = ctx["lay"], ctx["kl"]
    n = ctx["n"]
    good = krt_from_slots(lay, ctx["ks"], kl.decoder_f(), ctx["cos"], ctx["sin"], ctx["q"])
    ident = krt_from_slots(lay, ctx["ks"], kl.decoder_f(),
                           torch.ones(n, kt.D, device=kt.DEV),
                           torch.zeros(n, kt.D, device=kt.DEV), ctx["q"])
    ctx["moved"] = ((good - ident).abs().max() / good.abs().max()).item()


@then("the scores move by more than 1e-2 relative")
def _moved(ctx):
    print(f"  scores moved by {ctx['moved']:.3f} (relative)")
    assert ctx["moved"] > 1e-2, (
        f"the scores moved by only {ctx['moved']:.2e}, so the rotation is not being applied per row "
        f"and the agreement scenario above is vacuous")


@given('a layer whose storage mode is "pre"')
def _pre_layer(ctx):
    try:
        from ttkv_attn import _refuse_pre_storage
    except Exception as exc:                                                 # noqa: BLE001
        pytest.skip(f"ttkv_attn unavailable: {exc}")

    class _Layer:
        storage = "pre"

    ctx["layer"], ctx["guard"] = _Layer(), _refuse_pre_storage


@when("the direct-contract path is asked to decode it")
def _ask_direct(ctx):
    try:
        ctx["guard"](ctx["layer"])
        ctx["raised"] = None
    except Exception as exc:                                                 # noqa: BLE001
        ctx["raised"] = exc


@then("it raises rather than returning numbers")
def _raised(ctx):
    assert isinstance(ctx["raised"], NotImplementedError), (
        "a position-free store read by the direct-contract path yields `q_post . k_pre`: finite, "
        "correctly shaped, and wrong. It must refuse")


# --------------------------------------------------------------------------- end_to_end_forward
def _need_model():
    got = e2e._load()
    if got is None:
        pytest.skip(f"{e2e.MODEL} (or transformers) unavailable offline")
    return got


@given("a real model and a nearly lossless config")
def _model_lossless(ctx):
    model, tok, dev = _need_model()
    _, D, _ = e2e._geom(model)
    ctx.update(model=model, tok=tok, dev=dev, cfg=e2e._cfg(D, near_lossless=True))


@given("a real model and a shippable config")
def _model_shippable(ctx):
    model, tok, dev = _need_model()
    _, D, _ = e2e._geom(model)
    ctx.update(model=model, tok=tok, dev=dev, cfg=e2e._cfg(D))


@when(parsers.parse('I prefill through the cache with storage "{storage}"'))
def _prefill(ctx, storage):
    ctx["ids"] = ctx.get("ids") if ctx.get("ids") is not None else e2e._ids(ctx["tok"], ctx["dev"])
    with torch.no_grad():
        ctx["dense"] = ctx["model"](ctx["ids"]).logits
    lg, cache = e2e._forward(ctx["ids"], ctx["cfg"], storage, ctx["model"], ctx["dev"], fold_T=ctx["ids"].shape[1])
    ctx["logits"], ctx["cache"] = lg, cache


@then("the logits track dense to cosine 0.9999")
def _track_dense(ctx):
    c = e2e._cos(ctx["logits"], ctx["dense"])
    print(f"  logit cosine {c:.8f}")
    assert c > 0.9999, (
        f"a nearly lossless store moved the logits to cosine {c:.6f}. If the `post` arm passes and "
        f"`pre` fails, the un-rotate/re-rotate round trip is broken -- and a broken one is still "
        f"orthogonal and position-dependent, so it fails nowhere else")


@when("I prefill through the cache under both storage modes")
def _prefill_both(ctx):
    ctx["ids"] = e2e._ids(ctx["tok"], ctx["dev"])
    with torch.no_grad():
        ctx["dense"] = ctx["model"](ctx["ids"]).logits
    n = ctx["ids"].shape[1]
    ctx["packed"] = {}
    for storage in ("post", "pre"):
        _lg, cache = e2e._forward(ctx["ids"], ctx["cfg"], storage, ctx["model"], ctx["dev"], fold_T=n)
        ctx["packed"][storage] = sum(l.nbytes() for l in cache.layers)
    H, D, NL = e2e._geom(ctx["model"])
    ctx["dense_bytes"] = NL * H * 2 * D * 2 * n
    print(f"  dense {ctx['dense_bytes']} B  post {ctx['packed']['post']} B  "
          f"pre {ctx['packed']['pre']} B")


@then("both stores are smaller than the dense cache")
def _both_smaller(ctx):
    for storage, b in ctx["packed"].items():
        assert b < ctx["dense_bytes"], f"storage={storage!r} stored {b} B against dense {ctx['dense_bytes']} B"


@then("the pre-rotary store is smaller than the post-rotary one")
def _pre_smaller(ctx):
    assert ctx["packed"]["pre"] < ctx["packed"]["post"], (
        f"pre {ctx['packed']['pre']} B vs post {ctx['packed']['post']} B. RoPE disperses the key "
        f"spectrum, so the pre-rotary store is the one that should need less rank")


@given("a fold interval of 2 tokens so the tail folds during decode")
def _fold_interval(ctx):
    ctx["fold_T"] = 2


@when("I prefill and then decode four tokens greedily")
def _greedy(ctx):
    model, ids = ctx["model"], e2e._ids(ctx["tok"], ctx["dev"])
    H, D, NL = e2e._geom(model)

    dense, cur = [], ids
    with torch.no_grad():
        for _ in range(4):
            # The FULL sequence each step and no cache: `past_key_values=None` with
            # `use_cache=True` allocates a fresh cache per call, so a one-token-at-a-time dense arm
            # would have no history and both arms would be wrong together.
            nxt = model(cur, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
            dense.append(int(nxt))
            cur = torch.cat([cur, nxt], dim=1)

    cache = pc.make_cache(ctx["cfg"], H, NL, fold_T=ctx["fold_T"], storage="pre",
                          rope_fn=e2e._rope_fn(model))()
    got, nxt = [], None
    with torch.no_grad():
        for i in range(4):
            nxt = model(ids if i == 0 else nxt, past_key_values=cache, use_cache=True
                        ).logits[:, -1, :].argmax(-1, keepdim=True)
            got.append(int(nxt))
    ctx["dense_toks"], ctx["ttkv_toks"] = dense, got
    ctx["folds"] = sum(l.folds for l in cache.layers)


@then("the greedy tokens match the dense ones exactly")
def _greedy_match(ctx):
    print(f"  dense {ctx['dense_toks']}  ttkv {ctx['ttkv_toks']}")
    assert ctx["ttkv_toks"] == ctx["dense_toks"], (
        f"greedy decode diverged: dense {ctx['dense_toks']} vs ttkv {ctx['ttkv_toks']}")


@then("the store actually folded")
def _folded(ctx):
    assert ctx["folds"] > 0, (
        "the tail never reached fold_T, so `_append` was not exercised and the scenario above says "
        "nothing about the streaming path")
