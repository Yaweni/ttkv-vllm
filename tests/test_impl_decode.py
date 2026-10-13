"""Does `TTKVAttentionImpl` reproduce the reference decode, end to end?

THIS IS THE FIRST TEST THAT DRIVES THE REAL `forward`, which is why it is worth its length. Every
layer below it is gated somewhere else -- `krt` against `reference.raw_scores`, the codec's packing
against its own round trip, the layout's arithmetic against hand-built configurations. What none of
those reach is the ADAPTER: reading a request's slots out of vLLM's cache through its block table,
splitting the shared K||V region, deciding which rows are decodes, and writing into `output`. A bug
there is invisible to every other test in this directory and produces correctly-shaped output.

The oracle is `reference.attend_raw_store`, written independently of the kernels and in a different
index order on purpose (`[token, head, dim]` with the softmax on dim 0, against `oracle_attend`'s
`[head, group, dim]`), so agreement tests the ordering rather than a shared assumption.

AND THE GUARDS ARE TESTED AS HARD AS THE MATH. This integration's failure mode is a plausible wrong
number, so the tests that matter most are the ones that assert it REFUSES: a store that was never
written, a decoder whose rank disagrees with the slot width it was sized for, a continuation chunk
that would silently drop every earlier key, a decode that is not where the batch split thinks it is.

Gated on vLLM being importable -- `impl` is the adapter, so it needs the real base class -- and
skips the kernel-dependent cases without a CUDA device.
"""
from __future__ import annotations

import os
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

pytest.importorskip("vllm.v1.attention.backend")
from vllm_ttkv import decode as dc                                     # noqa: E402
from vllm_ttkv import impl as impl_module                              # noqa: E402
from vllm_ttkv import patch                                            # noqa: E402
from vllm_ttkv.encode import KDecoder                                  # noqa: E402
from vllm_ttkv.impl import TTKVAttentionImpl                           # noqa: E402
from vllm_ttkv.layout import TTKLayout, layout_for_config, pack_k_slots, pack_v_slots  # noqa: E402
from vllm_ttkv.reference import (attend_raw_store, dense_from_slots,  # noqa: E402
                                 dense_v_from_slots, raw_scores_multi)
from vllm_ttkv.rope_store import (TTKVRopeStore, clear_decoders, decoder_for_request,  # noqa: E402
                                  register_decoder, request_key, store_for_layer, CONFIG_ENV)

H, D, HQ = 8, 128, 32                     # G = 4, the Llama-3.1-8B geometry
BLOCK = 16                                # small, so the block table is non-trivial at n~100
CFG = "campaign4x"
G = HQ // H
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _pin_allocation(monkeypatch):
    """Pin `$VLLM_TTKV_CONFIG` to the allocation this module builds its LAYOUTS from.

    Not a test convenience. The layout comes from `CFG` while the ENCODER reads the env var, and
    those two have to name the same allocation -- a store encoded under one and read under the other
    has every field after the coefficient block at the wrong offset, which does not raise. The guard
    in `encode.check_cfg_matches_layout` fired on this fixture the moment the write path landed,
    which is precisely why it exists.
    """
    monkeypatch.setenv(CONFIG_ENV, CFG)


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device (the read path needs the Triton kernels)")
        return False
    return True


def _cos_sin_cache(max_len: int, d: int = D, theta: float = 10000.0):
    """A cache in VLLM's OWN layout: `[max_pos, d]` holding `cat([cos, sin], -1)`, each half `d/2`.

    Built rather than borrowed so this test does not depend on a model, and so the half/straddle
    convention that `decode.cos_sin_from_cache` translates is exercised rather than assumed. In a
    real run the table comes from the model's rotary module (`cos_sin_cache`), because Llama-3.1's
    `inv_freq` carries a ramp that `config.rope_theta` does not reproduce.
    """
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=DEV) / d))
    freqs = torch.outer(torch.arange(max_len, device=DEV).float(), inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class _RotaryTable:
    """Stands in for the model's rotary module. The writer reads `cos_sin_cache` off it -- and only
    the writer, because the store is what carries the table for the read path."""

    def __init__(self, cos_sin_cache):
        self.cos_sin_cache = cos_sin_cache


class _Layer:
    """The minimum a real `Attention` layer has to look like: a name, the layout, the decoder, and
    the model's rotary module."""

    def __init__(self, lay, decoder_f, name="model.layers.0.self_attn", cos_sin_cache=None):
        self.layer_name = name
        self.ttkv_layout = lay
        self.ttkv_decoder_f = decoder_f
        self.rotary_emb = _RotaryTable(cos_sin_cache) if cos_sin_cache is not None else None


class _MD:
    """Stand-in for `TTKVAttentionMetadata`.

    Carries only fields vLLM genuinely produces (`query_start_loc`, `seq_lens`, `block_table`,
    `slot_mapping`). The point of the fixture is that nothing here is invented for the test's
    convenience -- if `_plan` or the write needed a field the engine does not emit, this would not
    build. Note what is NOT here: any "already written" flag. The impl writes unconditionally on
    every call, because a per-STEP flag cannot record that a particular LAYER wrote -- vLLM hands one
    metadata object to every layer of a group (see `metadata.py`).
    """

    def __init__(self, q_lens, seq_lens, block_table, slot_mapping=None):
        self.query_start_loc = torch.tensor([0] + list(q_lens), dtype=torch.int32).cumsum(0).to(
            torch.int32)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.block_table = block_table
        self.slot_mapping = (slot_mapping if slot_mapping is not None
                             else torch.arange(int(sum(seq_lens)), dtype=torch.int64, device=DEV))
        self.num_actual_tokens = int(self.query_start_loc[-1])


R1 = 64
"""The cache has ONE slot width per layer, so every request's fit is made to fit within this."""


def _fit_within_budget(K, cfg, budget, d, H):
    """Fit a request's K, then zero-pad the result up to the cache's rank budget.

    THE PADDING IS THE POINT, and it is the F4 constraint in executable form. vLLM allocates the
    pages before any sequence exists, so the slot width is fixed per layer by the calibration table
    -- but each request's TT-SVD lands at its own natural rank (measured 6..166 across layers). So
    the fit is padded up to the budget, and the padding is exact rather than approximate: a zero
    coefficient times a zero decoder row is zero, so the reconstruction, and therefore every
    residual computed against it, is unchanged. That is the same move
    `ttkv_packed_cache._pad_rank` makes, and it is why `r1` is a BUDGET at serve time rather than a
    measurement.

    Returns `(coeff_q, coeff_scale, decoder_f, resid)`.
    """
    kl = pc.build_k_layer(K, cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=d)
    if kl.r1 > budget:
        raise AssertionError(
            f"the fit needs r1={kl.r1} but the layout budgets {budget}. In a real run the tolerance "
            f"is chosen so the budget binds; here the budget is generous so the fixture stays "
            f"honest about which one is limiting.")
    if kl.r1 == budget:
        return kl.coeff_q, kl.coeff_scale, kl.decoder_f(), kl.resid
    cq = torch.zeros(kl.coeff_q.shape[0], budget, dtype=torch.int8, device=K.device)
    cq[:, :kl.r1] = kl.coeff_q
    dec = torch.zeros(budget, H * d, device=K.device, dtype=torch.float32)
    dec[:kl.r1] = kl.decoder_f()
    return cq, kl.coeff_scale, dec, kl.resid


def _world(seq_lens, q_lens=None, cfgname=CFG, seed=0, write=True):
    """Build `(impl, layer, cache, md, q, per-request payloads, angle cache)`.

    Each request gets its own contiguous, BLOCK-ALIGNED run of slots -- what vLLM's allocator hands
    out -- and its own fitted decoder registered against `block_table[row, 0]`. The blobs are written
    through `slot_mapping`'s own flat indexing, so the read path meets the same convention the write
    path uses.
    """
    torch.manual_seed(seed)
    cfg = CONFIGS[cfgname]
    # Default to DECODE (one query token per request), which is what most of this file exercises.
    # Defaulting to `q_lens = seq_lens` would make every request a prefill instead -- and the tests
    # would still run, against the wrong branch, because `_plan` picks on query length alone.
    q_lens = [1] * len(seq_lens) if q_lens is None else list(q_lens)
    n_reqs = len(seq_lens)
    csc = _cos_sin_cache(max(seq_lens) + 8)
    lay = layout_for_config(cfg, r1=R1, H=H, d=D)          # ONE layout for the whole batch
    C = lay.compressed_bytes_per_token()

    per_req, block_table, next_block = [], [], 0
    for r, n in enumerate(seq_lens):
        # A low-rank-plus-noise K, so the residual path is live; V is high-rank per token.
        rank = 16
        K = ((torch.randn(n, rank, device=DEV) * 0.2)
             @ (torch.randn(rank, H * D, device=DEV) * 0.2)
             ).reshape(n, H, D).permute(1, 0, 2).contiguous()
        K = K + 0.02 * torch.randn(H, n, D, device=DEV)
        V = torch.randn(H, n, D, device=DEV) * 0.3
        cq, cscale, dec, resid = _fit_within_budget(K, cfg, R1, D, H)
        vl = pc.build_v_layer(V, cfg["k_lanes"], cfg["vbits"], d=D)
        ks = pack_k_slots(lay, cq, cscale, resid)
        vs = pack_v_slots(lay, vl.rows)
        nblk = (n + BLOCK - 1) // BLOCK
        bt = torch.arange(next_block, next_block + nblk, dtype=torch.int32, device=DEV)
        slots = dc.block_slots(bt, n, BLOCK)
        per_req.append(dict(n=n, lay=lay, ks=ks, vs=vs, slots=slots, dec=dec,
                            blob=torch.cat([ks, vs], dim=-1)))
        block_table.append(bt)
        next_block += nblk

    cache = torch.zeros(next_block, 1, BLOCK, C, dtype=torch.uint8, device=DEV)
    flat = cache.reshape(-1, C)
    for pr in per_req:
        if write:
            flat[pr["slots"]] = pr["blob"]

    max_blocks = max(len(bt) for bt in block_table)
    btt = torch.zeros(n_reqs, max_blocks, dtype=torch.int32, device=DEV)
    for r, bt in enumerate(block_table):
        btt[r, :len(bt)] = bt

    q = torch.randn(int(sum(q_lens)), HQ, D, device=DEV) * 0.1   # post-rotary, as the model hands it
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    name = f"model.layers.{seed}.self_attn"
    # `decoder_f=None` on the layer and on the store, deliberately: it forces every test through the
    # PER-REQUEST registry, which is the path that is correct for a batch.
    layer = _Layer(lay, None, name=name, cos_sin_cache=csc)
    store = TTKVRopeStore(HQ, H, D, csc, name, lay)        # supplies the angle table
    clear_decoders()
    for r, pr in enumerate(per_req):
        register_decoder(name, request_key(block_table[r]), KDecoder(pr["dec"]))

    md = _MD(q_lens, seq_lens, btt)
    return impl, layer, cache, md, q, per_req, csc


def _prebuilt_store(impl):
    """Make the impl skip its write, for tests that hand it a store built BY HAND.

    That is the point of these tests rather than a convenience: they place a known blob in the
    slots, then check that the READ agrees with `reference.attend_raw_store` reading the same bytes.
    Letting the impl re-encode would overwrite the store with its own fit, and the "independent
    oracle" would be the impl compared against itself through a different door.

    The write has its own tests, and they call `forward` directly:
    `test_forward_writes_the_store_and_then_decodes_from_it` (byte-exact, then read),
    `test_two_layers_of_one_step_each_write_their_own_store`, and the `test_the_write_refuses_*`
    group. In production the write is UNCONDITIONAL -- see `impl.forward` -- because "has this LAYER
    written" is not a per-step fact and the metadata cannot carry it.
    """
    impl._write_store = lambda *a, **k: None
    return impl


def _run(impl, layer, cache, md, q):
    """Drive `forward` against a store this fixture built by hand (see `_prebuilt_store`)."""
    _prebuilt_store(impl)
    out = torch.zeros(q.shape[0], HQ, D, device=DEV)
    ret = impl.forward(layer, q, None, None, cache, md, output=out)
    assert ret is out, "forward must write into `output`, not return a new tensor"
    return out


# --------------------------------------------------------------------------- decode
def test_single_request_decode_matches_the_reference():
    if not _need_cuda():
        return
    impl, layer, cache, md, q, pr, csc = _world([100])
    out = _run(impl, layer, cache, md, q)

    r = pr[0]
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(r["n"], device=DEV), D)
    want = attend_raw_store(r["lay"], r["ks"], r["vs"], r["dec"], q[0], cos, sin,
                            scale=D ** -0.5)
    rel = ((out[0] - want).abs().max() / want.abs().max()).item()
    print(f"  n=100 (7 blocks)  r1={r['lay'].r1}  rel err {rel:.3e}")
    assert rel < 1e-4, (
        f"the impl's decode disagrees with the reference by {rel:.3e}. This path is block table -> "
        f"slot gather -> K/V split -> krt -> softmax -> v_out, so a mismatch is in the adapter "
        f"rather than in any single kernel -- and every one of those steps can be wrong while "
        f"still returning the right shape.")


def test_batched_decode_with_ragged_lengths():
    """Continuous batching, which is the reason the store is per request at all.

    Three requests decode in one step with DIFFERENT context lengths, none of them a multiple of the
    block size, so each one's slot run is ragged and they are not interchangeable. A per-request
    index error here would be completely invisible with a single request.
    """
    if not _need_cuda():
        return
    seq_lens = [64, 100, 129]
    impl, layer, cache, md, q, pr, csc = _world(seq_lens)

    out = _run(impl, layer, cache, md, q)

    worst = 0.0
    for r, payload in enumerate(pr):
        cos, sin = dc.cos_sin_from_cache(csc, torch.arange(payload["n"], device=DEV), D)
        want = attend_raw_store(payload["lay"], payload["ks"], payload["vs"], payload["dec"],
                                q[r], cos, sin, scale=D ** -0.5)
        rel = ((out[r] - want).abs().max() / want.abs().max()).item()
        worst = max(worst, rel)
        print(f"  req {r}: seq_len={payload['n']:<4d} rel err {rel:.3e}")
    assert worst < 1e-4, (
        f"a ragged decode batch disagrees with the reference by {worst:.3e}. With one request this "
        f"would pass; the failure this looks for is one request reading another's slots.")


def test_each_request_gets_its_own_decoder():
    """The decoders differ per request in this fixture, so a batched run that used the first
    request's decoder for all three must fail. Batching is a serving change, not an accuracy one:
    the codec's pooling note records that sharing one fitted basis across a batch moved a slice by
    5.8 in absolute score, because a rank-r1 basis spanning B sequences has to cover B times the key
    diversity."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, pr, csc = _world([64, 100, 129])
    decs = [pr[r]["dec"] for r in range(3)]
    assert not torch.equal(decs[0], decs[1]) or decs[0].shape != decs[1].shape, (
        "the fixture gave every request the same decoder, so this test cannot detect sharing")

    out = _run(impl, layer, cache, md, q)
    # Request 2 must match ITS OWN decoder and not request 0's.
    r = pr[2]
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(r["n"], device=DEV), D)
    mine = attend_raw_store(r["lay"], r["ks"], r["vs"], r["dec"], q[2], cos, sin,
                            scale=D ** -0.5)
    theirs = attend_raw_store(r["lay"], r["ks"], r["vs"], pr[0]["dec"], q[2], cos, sin,
                              scale=D ** -0.5)
    rel_mine = ((out[2] - mine).abs().max() / mine.abs().max()).item()
    spread = ((mine - theirs).abs().max() / mine.abs().max()).item()
    print(f"  own decoder rel err {rel_mine:.3e}; borrowing req 0's would move it {spread:.3e}")
    assert rel_mine < 1e-4, f"request 2 did not use its own decoder: rel err {rel_mine:.3e}"


# --------------------------------------------------------------------------- prefill
def test_prefill_first_chunk_matches_a_manual_causal_grouped_attention():
    """First-chunk prefill, checked against attention written out longhand in the test.

    Deliberately NOT against `dc.attend_prefill`, which is what the impl calls -- that would be a
    tautology. The manual form expands GQA explicitly and masks by hand, so it pins two things the
    stock call could get wrong without changing shape: which query head reads which KV head, and
    which keys a position is allowed to see.
    """
    if not _need_cuda():
        return
    T = 12
    impl, layer, cache, md, q, pr, csc = _world([T], q_lens=[T])
    _prebuilt_store(impl)        # this test is the PREFILL branch; the store is the fixture's
    k = torch.randn(T, H, D, device=DEV) * 0.3
    v = torch.randn(T, H, D, device=DEV) * 0.3
    out = torch.zeros(T, HQ, D, device=DEV)
    impl.forward(layer, q, k.clone(), v.clone(), cache, md, output=out)

    scale = D ** -0.5
    want = torch.zeros_like(out)
    for t in range(T):
        for g in range(H):
            s = (q[t, g * G:(g + 1) * G] @ k[:t + 1, g].T) * scale      # [G, t+1]
            p = torch.softmax(s, dim=-1)
            want[t, g * G:(g + 1) * G] = p @ v[:t + 1, g]
    rel = ((out - want).abs().max() / want.abs().max()).item()
    print(f"  T={T} rel err {rel:.3e}")
    assert rel < 1e-4, (
        f"prefill disagrees with longhand causal GQA by {rel:.3e}; a wrong kv-head mapping or a "
        f"missing causal mask both produce correctly-shaped output")


def test_prefill_is_causal():
    """Negative control on the mask. If prefill attended to future keys, the longhand comparison
    above would still be testing the same function -- so this perturbs the LAST key and requires
    every earlier position's output to be bit-identical."""
    if not _need_cuda():
        return
    T = 12
    impl, layer, cache, md, q, pr, _csc = _world([T], q_lens=[T])
    _prebuilt_store(impl)        # the PREFILL branch again; the store is not what is under test
    k = torch.randn(T, H, D, device=DEV) * 0.3
    v = torch.randn(T, H, D, device=DEV) * 0.3

    base = torch.zeros(T, HQ, D, device=DEV)
    impl.forward(layer, q, k.clone(), v.clone(), cache, md, output=base)

    k2 = k.clone()
    k2[T - 1] += 5.0                                     # a huge change, at the last position only
    moved = torch.zeros(T, HQ, D, device=DEV)
    impl.forward(layer, q, k2, v.clone(), cache, md, output=moved)

    early = (base[:T - 1] - moved[:T - 1]).abs().max().item()
    late = (base[T - 1] - moved[T - 1]).abs().max().item()
    print(f"  perturbing key {T - 1}: earlier rows moved {early:.3e}, the last row moved {late:.3e}")
    assert early == 0.0, f"a change at position {T - 1} moved earlier outputs by {early:.3e}"
    assert late > 1e-2, f"the perturbation barely moved its own row ({late:.3e})"


# --------------------------------------------------------------------------- the whole loop
def test_forward_writes_the_store_and_then_decodes_from_it():
    """THE WHOLE LOOP, with the impl doing both halves: raw K in, attention out.

    Nothing is pre-packed here and the cache starts as zeros, so this is the first test that can fail
    because the WRITE is wrong rather than because the read is. It runs two steps, the minimum that
    exercises both phases: a first-chunk PREFILL (a real TT-SVD fit, registered), then a one-token
    DECODE (a projection onto that frozen decoder).

    Three separable assertions:
      1. the store holds BYTE-EXACTLY what the encoder would have written, at the slots
         `block_slots` names -- so both the slot addressing and the packing are checked;
      2. the decode agrees with the reference reading THE SAME STORE -- so a write and a read that
         disagreed about which slot holds which token could not pass;
      3. the decoder is the SAME OBJECT across the two steps -- so step 2 projected instead of
         refitting.
    """
    if not _need_cuda():
        return
    from vllm_ttkv import encode as enc

    n = 64
    cfg = CONFIGS[CFG]
    lay = layout_for_config(cfg, r1=R1, H=H, d=D)
    C = lay.compressed_bytes_per_token()
    csc = _cos_sin_cache(n + 8)
    name = "model.layers.77.self_attn"
    clear_decoders()
    torch.manual_seed(21)

    # The model's keys and values, RAW. K is low-rank-plus-noise so the residual path is live.
    rank = 16
    K = ((torch.randn(n + 1, rank, device=DEV) * 0.2)
         @ (torch.randn(rank, H * D, device=DEV) * 0.2)
         ).reshape(n + 1, H, D).permute(1, 0, 2).contiguous()
    K = K + 0.02 * torch.randn(H, n + 1, D, device=DEV)
    V = torch.randn(H, n + 1, D, device=DEV) * 0.3
    Kt, Vt = K.permute(1, 0, 2).contiguous(), V.permute(1, 0, 2).contiguous()   # [T, H, d]

    nblk = (n + 1 + BLOCK - 1) // BLOCK
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV)
    slots = dc.block_slots(bt, n + 1, BLOCK)
    cache = torch.zeros(nblk, 1, BLOCK, C, dtype=torch.uint8, device=DEV)
    layer = _Layer(lay, None, name=name, cos_sin_cache=csc)
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    btt = bt.reshape(1, -1)

    # ---- step 1: the first chunk. The impl is the only writer, so it does the write here.
    positions = torch.arange(n, device=DEV)
    cos, sin = dc.cos_sin_from_cache(csc, positions, D)
    patch.stash_raw_k(name, Kt[:n], positions)
    k1 = pc.rope_apply(Kt[:n], cos.unsqueeze(1), sin.unsqueeze(1))
    md1 = _MD([n], [n], btt, slot_mapping=slots[:n])
    q1 = torch.randn(n, HQ, D, device=DEV) * 0.1
    impl.forward(layer, q1, k1, Vt[:n], cache, md1, output=torch.zeros(n, HQ, D, device=DEV))

    assert store_for_layer(name) is not None, "the write did not create this layer's store"
    dec1 = decoder_for_request(name, request_key(btt[0]))
    assert dec1 is not None, "the first chunk did not register a fitted decoder"

    # 1. BYTE-EXACT: the write put the encoder's own bytes at the slots the reader will use.
    blob = dc.gather_slots(cache, slots[:n])
    ks, vs = dc.split_blob(lay, blob)
    fk = enc.fit_into_budget(K[:, :n].contiguous(), lay, cfg)
    assert torch.equal(ks, pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid)), (
        "the store does not hold what the encoder would have written for this chunk. The fit is "
        "deterministic, so this is a write-plumbing failure: wrong slots, wrong blob split, or a "
        "partial write.")
    assert torch.equal(vs, pack_v_slots(lay, enc.encode_v(V[:, :n].contiguous(), lay)))

    # And what the store RECONSTRUCTS is recognisably the model's own keys.
    Kd, Vd = dense_from_slots(lay, ks, vs, dec1.decoder_f)
    k_rel = ((Kd - K[:, :n]).norm() / K[:, :n].norm()).item()
    v_rel = ((Vd - V[:, :n]).norm() / V[:, :n].norm()).item()
    print(f"  step 1 (prefill n={n}): fitted {dec1!r}; store vs model K {k_rel:.4f}, V {v_rel:.4f}")
    assert k_rel < 0.20, f"the stored keys are not the model's keys: rel err {k_rel:.4f}"
    assert v_rel < 0.75, f"the stored values are not the model's values: rel err {v_rel:.4f}"

    # ---- step 2: one decode token. The request has a decoder now, so this must PROJECT.
    n2 = n + 1
    pos2 = torch.arange(n2, device=DEV)
    cos2, sin2 = dc.cos_sin_from_cache(csc, pos2, D)
    patch.stash_raw_k(name, Kt[n:n + 1], pos2[-1:])
    k2 = pc.rope_apply(Kt[n:n + 1], cos2[n:n + 1].unsqueeze(1), sin2[n:n + 1].unsqueeze(1))
    md2 = _MD([1], [n2], btt, slot_mapping=slots[n:n + 1])
    q2 = torch.randn(1, HQ, D, device=DEV) * 0.1
    out2 = torch.zeros(1, HQ, D, device=DEV)
    impl.forward(layer, q2, k2, Vt[n:n + 1], cache, md2, output=out2)

    # 3. NO REFIT. Compared by IDENTITY, not by value: a refit would produce a different object, and
    #    the point is that it did not happen at all.
    dec2 = decoder_for_request(name, request_key(btt[0]))
    assert dec2 is dec1, (
        "the decode step refitted instead of projecting onto the frozen decoder. Refitting is not a "
        "cost here, it is a correctness break: it would invalidate every coefficient already in the "
        "store, so nothing already written could still be read.")

    # 2. ADDRESSING: the read must agree with the write, over the SAME store.
    ks2, vs2 = dc.split_blob(lay, dc.gather_slots(cache, slots[:n2]))
    want = attend_raw_store(lay, ks2, vs2, dec2.decoder_f, q2[0], cos2, sin2, scale=D ** -0.5)
    rel = ((out2[0] - want).abs().max() / want.abs().max()).item()
    print(f"  step 2 (decode): projected onto the frozen decoder, read-vs-write rel err {rel:.3e}")
    assert rel < 1e-4, (
        f"the decode disagrees with the reference reading the same store by {rel:.3e}. Both halves "
        f"have now run through `forward`, so this is the write and the read disagreeing about which "
        f"slot holds which token.")


def test_the_write_refuses_a_request_that_has_no_decoder_and_prior_context():
    """A continuation step for a request whose prefill never ran. Fitting here would give the
    sequence a rank from one token and every later step would inherit it -- so this refuses instead,
    and says which of the two conditions is missing."""
    if not _need_cuda():
        return
    n = 32
    lay = layout_for_config(CONFIGS[CFG], r1=R1, H=H, d=D)
    csc = _cos_sin_cache(n + 8)
    name = "model.layers.78.self_attn"
    clear_decoders()
    torch.manual_seed(22)
    Kt = torch.randn(n, H, D, device=DEV) * 0.2
    Vt = torch.randn(n, H, D, device=DEV) * 0.2
    bt = torch.arange(2, dtype=torch.int32, device=DEV)
    slots = dc.block_slots(bt, n, BLOCK)
    cache = torch.zeros(2, 1, BLOCK, lay.compressed_bytes_per_token(), dtype=torch.uint8, device=DEV)
    layer = _Layer(lay, None, name=name, cos_sin_cache=csc)
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)

    patch.stash_raw_k(name, Kt, torch.arange(n, device=DEV))
    # q_len = 8 but seq_len = 32: 24 keys are supposedly already in the store, and nothing fitted it.
    md = _MD([8], [n], bt.reshape(1, -1), slot_mapping=slots[:8])
    with pytest.raises(RuntimeError, match="CONTINUATION step"):
        impl.forward(layer, torch.randn(8, HQ, D, device=DEV) * 0.1, Kt[:8], Vt[:8], cache, md,
                     output=torch.zeros(8, HQ, D, device=DEV))


# --------------------------------------------------------------------------- the guards
def test_the_write_refuses_when_the_rotary_wrapper_never_ran():
    """The most dangerous state in the whole design, and the one with no natural error.

    The backend declares `forward_includes_kv_cache_update = True`, so vLLM writes nothing and the
    impl is the only writer. If the rotary wrapper did not run there is no raw K to store, and
    decoding anyway would read whatever is in those slots -- zeros on a fresh page, or a finished
    request's keys on a recycled one -- and return finite, well-shaped attention about the wrong
    tokens. `take_raw_k` pops, so this is also where a double-consume is caught.
    """
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64])
    patch._RAW_K_STASH.pop(layer.layer_name, None)
    # NOT via `_run`: that stub suppresses the write, and the write is what must refuse here.
    with pytest.raises(RuntimeError, match="no raw K stashed"):
        impl.forward(layer, q, q, torch.zeros(q.shape[0], H, D, device=DEV), cache, md,
                     output=torch.zeros(q.shape[0], HQ, D, device=DEV))


def test_decode_refuses_a_decoder_whose_rank_disagrees_with_the_layout():
    """`r1` sizes the coefficient block, and every later field is found by adding widths to it. So a
    decoder fitted at a different rank than the cache was allocated at does not fail -- it reads
    residual bitmaps as codes. Real per-layer ranks run 6..166, so a budget that does not bind is the
    DEFAULT outcome of getting the order of operations wrong, not a corner case."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64])
    register_decoder(layer.layer_name, request_key(md.block_table[0]),
                     KDecoder(torch.zeros(R1 + 3, H * D, device=DEV)))
    with pytest.raises(ValueError, match="fitted decoder has r1"):
        _run(impl, layer, cache, md, q)


def test_multi_request_decode_refuses_a_layer_level_decoder():
    """THE GUARD THIS FILE'S OWN FIXTURE FOUND. A layer-level decoder is one request's fit, and using
    it for another request reconstructs that request's keys through a basis fitted for keys it never
    saw. Measured while building this test: request 2 came out 2.2e-02 relative-wrong, and one of the
    three returned NaNs -- with correct shapes and no error anywhere. With one request the two are
    the same object, so the fallback is allowed there and refused as soon as a second request decodes.
    """
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64, 80])
    clear_decoders()                       # as if the encoder had not registered anything yet
    with pytest.raises(RuntimeError, match="decoder belongs to a SEQUENCE"):
        _run(impl, layer, cache, md, q)


def test_a_single_request_may_use_the_layer_level_decoder():
    """The converse of the guard above, so the refusal cannot be satisfied by refusing always."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, pr, csc = _world([64])
    clear_decoders()
    layer.ttkv_decoder_f = pr[0]["dec"]
    out = _run(impl, layer, cache, md, q)
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(64, device=DEV), D)
    want = attend_raw_store(layer.ttkv_layout, pr[0]["ks"], pr[0]["vs"], pr[0]["dec"],
                            q[0], cos, sin, scale=D ** -0.5)
    rel = ((out[0] - want).abs().max() / want.abs().max()).item()
    print(f"  single request via the layer decoder: rel err {rel:.3e}")
    assert rel < 1e-4, f"single-request fallback disagrees with the reference by {rel:.3e}"


def test_refuses_a_kv_cache_that_is_two_tensors():
    """K and V share one region, because `state_content_bytes` describes the whole per-token cell
    (dense K||V is its default). If vLLM ever hands over a tuple, taking element 0 would read K as
    though it were the combined blob -- the V half would be coefficient bytes."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64])
    with pytest.raises(RuntimeError, match="expected ONE cache tensor"):
        impl.forward(layer, q, None, None, (cache, cache), md,
                     output=torch.zeros(1, HQ, D, device=DEV))


def test_refuses_fused_output_quantization():
    """Accepting these and ignoring them would have the caller read unquantized values out of a
    buffer sized for quantized ones. The skeleton accepted them silently; this refuses."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64])
    with pytest.raises(NotImplementedError, match="fused output quantization"):
        impl.forward(layer, q, None, None, cache, md, output=torch.zeros(1, HQ, D, device=DEV),
                     output_scale=torch.ones(1, device=DEV))
    with pytest.raises(NotImplementedError, match="fused output quantization"):
        impl.forward(layer, q, None, None, cache, md, output=torch.zeros(1, HQ, D, device=DEV),
                     output_block_scale=torch.ones(1, device=DEV))


def test_refuses_a_batch_whose_decodes_are_not_at_the_front():
    """vLLM reorders decodes first, and this adapter walks `qsl[r]` for `r < n_decodes`. If the order
    ever changed, the split would still succeed arithmetically -- and the decode branch would read a
    slot mapping written for a prompt population."""
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    # q_lens = [1, 5, 1]: decode, prefill, decode -- so decode rows are [0, 2], not [0, 1].
    md = _MD([1, 5, 1], [40, 40, 40], torch.zeros(3, 1, dtype=torch.int32, device=DEV))
    with pytest.raises(RuntimeError, match="not contiguous at the front"):
        impl._plan(md)


def test_the_step_builds_its_slots_on_the_device_and_agrees_with_the_host_build():
    """The two builds are the same addressing; only one of them is O(seq_len) on the HOST.

    `_Step.slots` checks the row against its host copy -- which the step has anyway, because the
    row's first block is the request's decoder key -- and then builds the ids on the device. That is
    what removes a `range`/`extend` loop and a second host-to-device copy per request per step
    (`tests/bench_step_plan.py` measures the pair), and it is only a swap worth making if the two
    agree on a row the allocator would really hand over: non-contiguous block ids, row padded past
    the used prefix. `check=False` is passed because `check_row_host` ran first, so the rows below
    also fail if that check ever stops covering the bound the device build skips.
    """
    if not _need_cuda():
        return
    # Ragged lengths, so the two requests need different block counts from the same row width.
    impl, _layer, cache, md, _q, per_req, _csc = _world([40, 25], q_lens=[1, 1], seed=11)
    step = impl_module._step_for(md, cache)
    # The allocator's rows, as they really come back: block ids that are not adjacent (the slots of
    # one request are interleaved with every other live request's) and padding past the used prefix.
    plan = [2, 0, 4, 1, 3]
    md.block_table[:, :] = torch.tensor(plan[:md.block_table.shape[1]], dtype=torch.int32,
                                        device=DEV)
    for r, pr in enumerate(per_req):
        n = pr["n"]
        row = [int(b) for b in md.block_table[r]]
        want = dc.block_slots_host(row, n, BLOCK, DEV, step.n_slots)
        got = step.slots(r, DEV)
        assert torch.equal(got, want), (
            f"request {r} ({n} tokens): the device build disagrees with the host build on a row "
            f"with non-contiguous blocks ({row})")


def test_the_step_refuses_a_padded_slot_for_the_whole_step_at_once():
    """The write's slot check is one readback for the step, and it is eager rather than per write.

    It is a device readback, so it is a pipeline drain: doing it per request per layer cost one per
    layer-call. Doing it for the whole step's slice costs one and refuses exactly the same values --
    including a padded (`PAD_SLOT_ID`) token, which is a key that cannot be stored while the read
    still covers `seq_len` rows. Refusing a LATER request's bad slot while the caller is only asking
    about the first is the point: the value is wrong for the step, not for a layer.
    """
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    csc = _cos_sin_cache(64)
    lay = layout_for_config(CONFIGS[CFG], r1=R1, H=H, d=D)
    cache = torch.zeros(4, 1, BLOCK, lay.compressed_bytes_per_token(), dtype=torch.uint8,
                        device=DEV)
    btt = torch.zeros(2, 1, dtype=torch.int32, device=DEV)
    bad = torch.tensor([0, 3, -1, 3], dtype=torch.int64, device=DEV)      # request 1's 2nd token
    md = _MD([2, 2], [2, 2], btt, slot_mapping=bad)
    with pytest.raises(RuntimeError, match="PAD_SLOT_ID"):
        impl_module._step_for(md, cache)
    md.slot_mapping = torch.tensor([0, 3, 999, 3], dtype=torch.int64, device=DEV)
    with pytest.raises(IndexError, match="outside the cache"):
        impl_module._step_for(md, cache)
    md.slot_mapping = torch.tensor([0, 3, 3, 3], dtype=torch.int64, device=DEV)
    step = impl_module._step_for(md, cache)
    assert step.write_slot_ids(1, DEV).tolist() == [3, 3], "the write's own slice moved"


def test_two_layers_of_one_step_each_write_their_own_store():
    """THE BUG THAT COST A RUN, as a test. vLLM hands ONE metadata object to every layer of a group,
    and this impl used to consult a `ttkv_kv_written` flag on it -- copied from HPC's
    `hpc_kv_written` handshake. Layer 0 set the flag and every other layer then skipped its own write
    while still reading its own store, and the real run died at layer 6 with "no TT-KV store for
    layer 'model.layers.6.self_attn.attn'".

    So: two layers, ONE metadata object, each must end up with its own store. A test that built a
    fresh metadata per layer would pass under either design and prove nothing.
    """
    if not _need_cuda():
        return
    cfg = CONFIGS[CFG]
    lay = layout_for_config(cfg, r1=R1, H=H, d=D)
    n, block = 32, BLOCK
    nblk = (n + block - 1) // block
    cache = torch.zeros(nblk, 1, block, lay.compressed_bytes_per_token(), dtype=torch.uint8,
                        device=DEV)
    btt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, -1)
    csc = _cos_sin_cache(n + 8)
    md = _MD([n], [n], btt, slot_mapping=torch.arange(n, dtype=torch.int64, device=DEV))
    q = torch.randn(n, HQ, D, device=DEV) * 0.1

    names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    for name in names:
        impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
        layer = _Layer(lay, None, name=name, cos_sin_cache=csc)
        # A first-chunk prefill, which is the step that writes: q_len == seq_len, so no request is a
        # continuation and no read happens. `key` arrives rotated with HQ heads and `value` with H,
        # as the model hands them over.
        raw_k = torch.randn(n, H, D, device=DEV) * 0.3
        patch.stash_raw_k(name, raw_k, torch.arange(n, device=DEV))
        impl.forward(layer, q, torch.randn(n, H, D, device=DEV) * 0.3,
                     torch.randn(n, H, D, device=DEV) * 0.3, cache, md,
                     output=torch.zeros(n, HQ, D, device=DEV))
        assert store_for_layer(name) is not None, (
            f"{name} did not get a store, so its write was skipped -- with one shared metadata "
            f"object, a per-step 'written' flag does exactly that to every layer after the first")

    assert store_for_layer(names[0]) is not store_for_layer(names[1])
    print("  two layers, one metadata object: both stores created, neither skipped")


def _ref_prefill_multi(lay, ks, vs, dec, q_perm, cos, sin, scale, p0):
    """Reference for a continuation chunk: attention over the WHOLE store, `[q_len, HQ, d]`.

    Written in a different index order from the kernel's call site on purpose -- `[HQ, q_len, n]`
    with the softmax on the last axis, GQA expanded explicitly through `dense_v_from_slots` -- so
    agreement tests the ordering rather than a shared assumption.
    """
    s = raw_scores_multi(lay, ks, dec, q_perm, cos, sin, scale=scale, p0=p0)   # [HQ, Q, n]
    p = torch.softmax(s, dim=-1)
    V = dense_v_from_slots(lay, vs)                                   # [H, n, d]
    p_all = torch.arange(HQ, device=q_perm.device) // G
    return torch.einsum("hqp,phd->qhd", p, V.permute(1, 0, 2)[:, p_all, :])


def test_continuation_prefill_matches_attention_over_the_whole_store():
    """THE CASE THIS BRANCH USED TO REFUSE, now computed -- and it is the guard that matters most
    for serving.

    A chunked-prefill continuation (or a prefix-cache hit) arrives with `seq_len > q_len`: the
    request already has keys in the store. Attending over the current chunk alone would SILENTLY DROP
    all of them, so this used to raise. It now reads the whole store through the multi-query `krt`,
    and this test pins that the answer is the attention over all 40 stored rows and not just the 8
    query tokens' own rows.

    THE STORE IS BUILT BY HAND AND THE WRITE IS SKIPPED (`_run` does that), because letting the impl
    re-encode would place the 8 new tokens at the fixture's `slot_mapping` and overwrite rows the
    continuation is supposed to read -- the test would then be comparing the impl against itself.
    `write=True` on `_world` is therefore REQUIRED, not incidental: it is what puts the known blob in
    the cache in the first place, and `write=False` leaves the cache all zeros so the read returns
    zeros and the reference does not. The first version of this test had that backwards and failed
    with a relative error of exactly 1.0, which is the signature of reading an empty store.
    """
    if not _need_cuda():
        return
    impl, layer, cache, md0, q, per_req, csc = _world([40], q_lens=[8])
    pr = per_req[0]
    lay = pr["lay"]
    # A continuation's new tokens occupy the SLOTS OF THEIR OWN POSITIONS: rows 32..39. The
    # fixture's default `slot_mapping` is `arange(sum(seq_lens))`, which is a first chunk's mapping.
    md = _MD([8], [40], md0.block_table, slot_mapping=pr["slots"][32:40])
    got = _run(impl, layer, cache, md, q)

    n = 40
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(n, device=DEV), D, max_position=n - 1)
    want = _ref_prefill_multi(lay, pr["ks"], pr["vs"], pr["dec"], q.permute(1, 0, 2).contiguous(),
                              cos, sin, D ** -0.5, p0=n - 8)
    rel = ((got - want).abs().max() / want.abs().max()).item()
    print(f"  continuation chunk (8 queries over 40 stored rows): rel err {rel:.3e}")

    # AND THE NEGATIVE CONTROL, because a wrong answer here is the silent kind: attending over the
    # chunk's own rows only is what the refusal existed to prevent, so it has to differ materially.
    only_own = _ref_prefill_multi(lay, pr["ks"][32:], pr["vs"][32:], pr["dec"],
                                  q.permute(1, 0, 2).contiguous(),
                                  cos[32:], sin[32:], D ** -0.5, p0=7)
    dropped = ((want - only_own).abs().max() / want.abs().max()).item()
    print(f"  dropping the 32 earlier keys changes the answer by {dropped:.3f} (relative)")
    assert dropped > 1e-2, (
        f"attending over the chunk's own rows differs from the whole store by only {dropped:.2e}, "
        f"so this test would pass even if the earlier keys were dropped")
    assert rel < 1e-4, (
        f"the continuation disagrees with attention over the whole store by {rel:.3e}. A dropped "
        f"key changes the softmax denominator and every weight in it, and nothing else would say so")


def test_a_mixed_batch_of_decodes_and_prefills_uses_both_paths():
    """A step holding BOTH a decode and a prefill, which is the shape chunked prefill produces.

    THIS PINS A REAL BUG THAT THIS SHAPE EXPOSED. `forward`'s decode loop used to assign
    `key = step.key(r)`, shadowing the `key` PARAMETER -- so the prefill branch below was handed a
    request key (an int) where it expected the key tensor, and died on `key[start:end]`. Every batch
    this port could form before chunked prefill worked was all-decode or all-prefill, so the bug was
    invisible until a mixed batch existed. Both halves are checked here, because a fix that broke
    either path would be worse than the bug: the decode row against the single-query reference, and
    the prefill rows against attention over the whole store.
    """
    if not _need_cuda():
        return
    impl, layer, cache, md, q, per_req, csc = _world([40, 40], q_lens=[1, 5])
    out = _run(impl, layer, cache, md, q)
    lay = per_req[0]["lay"]
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(40, device=DEV), D, max_position=39)

    want_dec = attend_raw_store(lay, per_req[0]["ks"], per_req[0]["vs"], per_req[0]["dec"],
                                q[0], cos, sin, D ** -0.5)
    rel_dec = ((out[0] - want_dec).abs().max() / want_dec.abs().max()).item()

    want_pre = _ref_prefill_multi(lay, per_req[1]["ks"], per_req[1]["vs"], per_req[1]["dec"],
                                  q[1:].permute(1, 0, 2).contiguous(), cos, sin, D ** -0.5, p0=35)
    rel_pre = ((out[1:] - want_pre).abs().max() / want_pre.abs().max()).item()

    print(f"  mixed batch (1 decode + 5-query prefill): decode rel err {rel_dec:.3e}, "
          f"prefill rel err {rel_pre:.3e}")
    assert rel_dec < 1e-4, (
        f"the DECODE half of a mixed batch disagrees with the reference by {rel_dec:.3e}; the "
        f"prefill branch's presence must not disturb the decode path")
    assert rel_pre < 1e-4, (
        f"the PREFILL half of a mixed batch disagrees by {rel_pre:.3e}. A batch with both kinds of "
        f"request goes through both branches in one `forward`, which is where the `key` shadowing "
        f"lived -- an int reached `key[start:end]` and the engine died in EngineCore")


def test_constructor_refuses_the_features_it_cannot_honour():
    """ALiBi, a soft cap and a sliding window all change which keys or weights the answer is made
    of, and none of them is applied by `krt`. KV sharing would make this layer read another layer's
    store, which holds a different fitted decoder."""
    base = dict(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    for kw, match in ((dict(alibi_slopes=[0.1] * HQ), "ALiBi"),
                      (dict(logits_soft_cap=30.0), "logits_soft_cap"),
                      (dict(sliding_window=4096), "sliding window"),
                      (dict(kv_sharing_target_layer_name="model.layers.0.self_attn"), "KV sharing")):
        with pytest.raises(NotImplementedError, match=match):
            TTKVAttentionImpl(**base, **kw)
    print("  refused: alibi, soft cap, sliding window, kv sharing")


def test_layout_head_dim_mismatch_refuses():
    """A rank is a rank for a matrix of a particular shape, so a table measured at one head dim is
    not merely imprecise at another -- it is a rank for a different matrix, and `r1` also sizes the
    slot, so every field after the coefficients is misread."""
    if not _need_cuda():
        return
    impl, layer, cache, md, q, _pr, _csc = _world([64])
    layer.ttkv_layout = TTKLayout(H=H, d=D // 2, r1=64, qbits=8, res_k=32, res_bits=3,
                                  v_lanes=45, v_bits=3)
    with pytest.raises(RuntimeError, match="built for head_dim"):
        _run(impl, layer, cache, md, q)


def test_tail_scores_refuses():
    """The dense tail was deleted, not stubbed: per-token encode measures 0.009 ms against the
    frozen decoder, so there is nothing for a tail to amortise, and it would reintroduce a second
    row population with a different rotation state and a ragged per-request length."""
    impl = TTKVAttentionImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=H)
    with pytest.raises(RuntimeError, match="dense tail does not exist"):
        impl._tail_scores()


def test_row_count_mismatch_refuses():
    """The invariant the tail used to carry. A store one row short drops a key, which changes the
    softmax denominator and therefore every weight in it."""
    with pytest.raises(RuntimeError, match="row counts disagree"):
        dc.check_row_counts(64, 64, 63, 64)
    dc.check_row_counts(64, 64, 64, 64)          # and the agreeing case must be silent


if __name__ == "__main__":
    os.environ[CONFIG_ENV] = CFG          # the pytest path gets this from the autouse fixture
    print(f"impl decode validation (CUDA: {torch.cuda.is_available()})")
    for fn in (test_single_request_decode_matches_the_reference,
               test_batched_decode_with_ragged_lengths,
               test_each_request_gets_its_own_decoder,
               test_prefill_first_chunk_matches_a_manual_causal_grouped_attention,
               test_prefill_is_causal,
               test_forward_writes_the_store_and_then_decodes_from_it,
               test_the_write_refuses_a_request_that_has_no_decoder_and_prior_context,
               test_the_write_refuses_when_the_rotary_wrapper_never_ran,
               test_two_layers_of_one_step_each_write_their_own_store,
               test_decode_refuses_a_decoder_whose_rank_disagrees_with_the_layout,
               test_multi_request_decode_refuses_a_layer_level_decoder,
               test_a_single_request_may_use_the_layer_level_decoder,
               test_refuses_a_kv_cache_that_is_two_tensors,
               test_refuses_fused_output_quantization,
               test_refuses_a_batch_whose_decodes_are_not_at_the_front,
               test_continuation_prefill_refuses_rather_than_dropping_context,
               test_constructor_refuses_the_features_it_cannot_honour,
               test_layout_head_dim_mismatch_refuses,
               test_tail_scores_refuses,
               test_row_count_mismatch_refuses):
        print(f"{fn.__name__}:")
        fn()
    print("\nimpl decode validation complete")
