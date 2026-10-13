"""Does a REAL transformers forward through the TTKV cache reproduce dense?

This is the layer that did not exist. Everything else compares the codec against a torch oracle on
synthetic stores; nothing drove an actual model through `make_cache`, which is the only way to
exercise the parts that a store-level test cannot reach: the `Cache`/`DynamicLayer` contract
transformers calls, the dtype and shape the model hands `update`, the un-rotate/re-rotate round trip
INSIDE a live attention, and whether the logits survive any of it.

THE TWO ARMS ARE NOT INTERCHANGEABLE
    near-lossless   `res_frac=1.0`, all lanes, 8-bit codes -- the store is nearly the tensor it was
                    given, so BOTH storage modes must track dense closely. This is the PORT gate: a
                    broken un-rotate leaves `post` matching dense while `pre` diverges, which is the
                    ONLY way that failure shows up, because a mis-rotation is still orthogonal and
                    position-dependent and therefore still returns well-formed numbers.
    shippable       a real allocation, where the assertion is a floored cosine plus the structural
                    claims (the store is smaller than dense; pre beats post).

WHY `Qwen/Qwen2-0.5B` AND NOT A TINY RANDOM MODEL
    `hf-internal-testing/tiny-random-LlamaForCausalLM` has `head_dim=4`. TT-KV's lane bitmap is
    `d/8` bytes, so that is a ZERO-BYTE bitmap and the format cannot express the row at all. Qwen2-0.5B
    is cached, real, and has usable geometry: d=64, H=2, NL=24.

    It also makes the geometry point concrete. `res_frac` is a fraction of d so it follows the
    geometry; `k_lanes` is an ABSOLUTE lane count in the codec, tuned at d=128, so at d=64 the shipped
    81 lanes do not exist. The config helper below DERIVES the lanes from d rather than carrying them,
    which is the pattern the plugin's calibration table needs.

Runs under pytest or as a plain script. Needs `transformers` and a cached model; skips cleanly
without either.
"""
from __future__ import annotations

import functools
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

MODEL = "Qwen/Qwen2-0.5B"
N = 48
DECODE_STEPS = 4


@functools.lru_cache(maxsize=1)
def _load():
    """`(model, tokenizer, device)` or `None` when the model or transformers is unavailable."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:                                                        # noqa: BLE001
        return None
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, local_files_only=True).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:                                                        # noqa: BLE001
        return None
    return model, tok, dev


@pytest.fixture(scope="module")
def loaded():
    got = _load()
    if got is None:
        pytest.skip(f"{MODEL} (or transformers) unavailable offline; this layer needs a real model")
    return got


def _geom(model):
    c = model.config
    d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_key_value_heads, int(d), c.num_hidden_layers


def _rope_fn(model):
    """The model's OWN frequencies. The llama3 ramp lives in `inv_freq` and is not reconstructible
    from `config.rope_theta`, and a wrong set is still orthogonal and position-dependent, so it fails
    silently -- which is exactly the class of mistake the near-lossless gate is here to catch."""
    rot = next((m for nm, m in model.named_modules() if "rotary" in nm.lower()), None)
    if rot is None:
        raise RuntimeError("no rotary module in the model")
    _, D, _ = _geom(model)

    def fn(positions):
        pos = positions.view(1, -1)
        try:
            dummy = torch.zeros(1, pos.shape[1], D, device=positions.device, dtype=torch.float32)
            cos, sin = rot(dummy, pos)[:2]
            return cos[0].to(torch.float32), sin[0].to(torch.float32)
        except Exception:                                                    # noqa: BLE001
            f = torch.outer(positions.float(), rot.inv_freq.to(positions.device).float())
            e = torch.cat([f, f], dim=-1)
            return e.cos(), e.sin()

    return fn


def _cfg(d, near_lossless=False, lanes_frac=0.5, res_frac=0.5, res_bits=4):
    """A config derived from the geometry, never carried across it. See the module docstring."""
    if near_lossless:
        return dict(tols=0.01, qbits=8, res_frac=1.0, res_bits=8, k_lanes=d, vbits=8, campaign=0.0)
    return dict(tols=0.20, qbits=8, res_frac=res_frac, res_bits=res_bits,
                k_lanes=max(1, int(d * lanes_frac)), vbits=res_bits, campaign=0.0)


def _ids(tok, dev, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, 30000, (1, n), generator=g).to(dev)


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def _forward(ids, cfg, storage, model, dev, fold_T):
    H, D, NL = _geom(model)
    cache = pc.make_cache(cfg, H, NL, fold_T=fold_T, storage=storage,
                          rope_fn=_rope_fn(model) if storage == "pre" else None)()
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
    return out.logits, cache


def test_e2e_prefill_matches_dense_at_a_near_lossless_config(loaded):
    """The port gate. Both storage modes must track dense when the store is nearly lossless."""
    model, tok, dev = loaded
    _, D, _ = _geom(model)
    cfg = _cfg(D, near_lossless=True)
    ids = _ids(tok, dev)
    with torch.no_grad():
        ref = model(ids).logits

    for storage in ("post", "pre"):
        lg, cache = _forward(ids, cfg, storage, model, dev, fold_T=N)
        assert torch.isfinite(lg).all(), f"{storage}: non-finite logits"
        c = _cos(lg, ref)
        print(f"  near-lossless {storage:<4s}: logit cosine {c:.8f}  "
              f"max|d| {(lg - ref).abs().max().item():.3e}")
        assert c > 0.9999, (
            f"a nearly lossless store under storage={storage!r} moved the logits to cosine {c:.6f}. "
            f"If `post` passes and `pre` fails, the un-rotate/re-rotate round trip is broken")


def test_e2e_prefill_tracks_dense_at_a_shippable_config(loaded):
    """A real allocation: the store must be smaller than dense AND still track it."""
    model, tok, dev = loaded
    H, D, NL = _geom(model)
    cfg = _cfg(D)
    ids = _ids(tok, dev)
    with torch.no_grad():
        ref = model(ids).logits

    lgs, packed = {}, {}
    for storage in ("post", "pre"):
        lg, cache = _forward(ids, cfg, storage, model, dev, fold_T=N)
        lgs[storage] = lg
        assert torch.isfinite(lg).all()
        # structural claim: the store really is compressed
        dense = NL * H * 2 * D * 2 * N                    # fp16 K+V, all layers, all tokens
        packed[storage] = sum(l.nbytes() for l in cache.layers)
        print(f"  shippable {storage:<4s}: logit cosine {_cos(lg, ref):.6f}  "
              f"store {packed[storage] / dense:.3f}x dense  ({packed[storage]} B)")
        assert packed[storage] < dense, "the store is not smaller than the dense cache"
        assert _cos(lg, ref) > 0.90, f"storage={storage!r} lost the logits entirely"

    # THE CLAIM THE WHOLE CHANGE RESTS ON, measured rather than narrated. Asserted on BYTES, not on
    # the logit cosine: with a real model both cosines sit near 1.0 and comparing them would pass on
    # saturation alone.
    print(f"  post vs pre logit cosine {_cos(lgs['post'], lgs['pre']):.6f}")
    assert packed["pre"] < packed["post"], (
        f"the pre-rotary store is {packed['pre']} B against post's {packed['post']} B; RoPE "
        f"dispersal should make the pre-rotary store the smaller one")


def test_e2e_compressed_store_is_smaller_per_token(loaded):
    """Criterion 3, but through a live model rather than the layout arithmetic."""
    model, tok, dev = loaded
    H, D, NL = _geom(model)
    ids = _ids(tok, dev)
    _lg, cache = _forward(ids, _cfg(D), "pre", model, dev, fold_T=N)
    dense = NL * H * 2 * D * 2 * N
    packed = sum(l.nbytes() for l in cache.layers)
    print(f"  dense {dense} B vs packed {packed} B -> {dense / packed:.3f}x")
    assert packed < dense


def test_e2e_decode_steps_match_dense_greedy(loaded):
    """Prefill then decode through the SAME cache -- the serving shape, not just a prefill.

    Greedy tokens must match dense for every step. A prefill-only test cannot see a broken append:
    the decode path is the one that calls `update` with q_len == 1, re-materialises the whole block
    and folds the tail, and that is where a folded store's row addressing would go wrong.
    """
    model, tok, dev = loaded
    H, D, NL = _geom(model)
    cfg = _cfg(D, near_lossless=True)
    ids = _ids(tok, dev)

    def greedy_dense():
        """The FULL sequence every step, with no cache.

        Feeding one token at a time while passing `past_key_values=None` would compare two models
        that are both wrong: with `use_cache=True` and no cache object, `past_key_values=None`
        allocates a fresh one per call, so the dense arm would have no history at all -- which is
        exactly how the first version of this test produced a "divergence" that was entirely its
        own.
        """
        cur, toks = ids, []
        with torch.no_grad():
            for _ in range(DECODE_STEPS):
                nxt = model(cur, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
                toks.append(int(nxt))
                cur = torch.cat([cur, nxt], dim=1)
        return toks

    def greedy_ttkv():
        # fold_T=2 against a 48-token prefill: the prefill builds in one go, then the tail reaches
        # fold_T twice during decode, so it is the MODEL's own calls that exercise `_append`.
        cache = pc.make_cache(cfg, H, NL, fold_T=2, storage="pre", rope_fn=_rope_fn(model))()
        nxt, toks = None, []
        with torch.no_grad():
            for i in range(DECODE_STEPS):
                step = ids if i == 0 else nxt
                nxt = model(step, past_key_values=cache, use_cache=True
                            ).logits[:, -1, :].argmax(-1, keepdim=True)
                toks.append(int(nxt))
        return toks, cache

    ref = greedy_dense()
    got, cache = greedy_ttkv()
    folds = sum(l.folds for l in cache.layers)
    print(f"  greedy dense {ref}  ttkv {got}   folds {folds}")
    assert folds > 0, "the tail never folded, so `_append` was not exercised by this test"
    assert got == ref, f"greedy decode diverged: dense {ref} vs ttkv {got}"


def test_store_mode_is_never_inferred():
    """Criterion 9: the direct-contract path must REFUSE a position-free store rather than read it.

    `q_post . k_pre` is finite, correctly shaped and wrong, so refusing is the only safe option where
    the correct path is unavailable.
    """
    sys.path.insert(0, str(_HERE.parents[2] / "integration"))
    try:
        from ttkv_attn import _refuse_pre_storage
    except Exception as exc:                                                 # noqa: BLE001
        pytest.skip(f"ttkv_attn unavailable: {exc}")

    class _Layer:
        def __init__(self, storage):
            self.storage = storage

    with pytest.raises(NotImplementedError):
        _refuse_pre_storage(_Layer("pre"))
    _refuse_pre_storage(_Layer("post"))          # no-op, must not raise
    _refuse_pre_storage(_Layer(None))            # unset means post -- and it is READ, not guessed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--no-header", "-p", "no:cacheprovider"]))
