"""The raw-K capture: what it must do, and the one way it must never quietly fail.

This file exists because the FIRST version of the capture did not survive contact with PyTorch.
It built a delegating wrapper around the rotary module and assigned it over `self.rotary_emb`,
which `nn.Module.__setattr__` refuses -- only a `Module`/`Tensor`/`Parameter` may be a child name.
That version passed its own reasoning (it really did see the raw key) and failed on the model,
which is the expensive place to find out, so the shape it landed on (`forward` rebound on the
instance) is pinned here against a fake module tree rather than against a loaded model.

The properties under test are the ones a load-test would not catch:
  * the rope still RUNS, and runs the model's own implementation -- a capture that replaced the
    rotation with an identity would be silent, orthogonal-looking and wrong;
  * the module tree is BYTE-IDENTICAL afterwards -- same module objects, same `state_dict` keys,
    same `named_modules` -- because a renamed submodule breaks weight loading;
  * the stash key is the name the IMPL is handed, so the reader cannot look in a different place
    than the writer put it;
  * the model's own RoPE table rides along, since the attention layer has none.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_ttkv import patch


class _Rope(nn.Module):
    """Stands in for vLLM's `RotaryEmbedding`: a buffer, and a forward that ROTATES.

    The rotation is `key * 2` rather than a real rotation on purpose -- what is being checked is
    that the inner implementation runs and that the capture sees the pre-image, not that RoPE is
    implemented correctly. The buffer is non-persistent, as vLLM's is (`register_buffer(...,
    persistent=False)`), so `state_dict` is not the thing to watch here.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("cos_sin_cache",
                             torch.arange(16, dtype=torch.float32).reshape(8, 2),
                             persistent=False)
        self.calls = 0

    def forward(self, positions, query, key):
        self.calls += 1
        return query, key * 2


class _Attention(nn.Module):
    """Stands in for vLLM's `Attention`, whose only relevant property is `layer_name = prefix`."""

    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name


class _Block(nn.Module):
    """Stands in for `LlamaAttention`: holds the rope, and holds the `Attention` as `attn`."""

    def __init__(self, prefix, rope=None):
        super().__init__()
        self.rotary_emb = rope if rope is not None else _Rope()
        self.attn = _Attention(f"{prefix}.attn")


@pytest.fixture(autouse=True)
def _clean_stash():
    patch._RAW_K_STASH.clear()
    yield
    patch._RAW_K_STASH.clear()


def test_capture_stashes_raw_k_and_the_rope_still_runs():
    blk = _Block("model.layers.0.self_attn")
    patch._install_capture(blk)
    q = torch.zeros(3, 4)
    k = torch.ones(3, 4)

    _, out = blk.rotary_emb(torch.arange(3), q, k)

    entry = patch.take_raw_k("model.layers.0.self_attn.attn")
    assert entry["raw_k"] is k, "the stash must hold the PRE-rotation key, not a copy of the result"
    assert entry["positions"].tolist() == [0, 1, 2]
    assert torch.equal(out, k * 2), "the model's own rotation must still run"
    assert blk.rotary_emb.calls == 1


def test_capture_is_a_no_op_on_the_module_tree():
    blk = _Block("model.layers.0.self_attn")
    rope = blk.rotary_emb
    names_before = [n for n, _ in blk.named_modules()]

    patch._install_capture(blk)

    assert blk.rotary_emb is not rope, "the capture must wrap the rope, not mutate the shared object"
    assert blk.rotary_emb._inner is rope
    assert [n for n, _ in blk.named_modules()] == names_before, (
        "a registered submodule would insert a name under `rotary_emb`, and `named_modules` feeds "
        "weight loading")
    assert blk.rotary_emb.cos_sin_cache is rope.cos_sin_cache, (
        "the model's own table must still be reachable -- the writer reads it through the wrapper")
    assert blk.rotary_emb.calls == 0


def test_capture_does_not_mutate_a_rope_shared_by_several_layers():
    """THE BUG THAT COST A REAL RUN. `get_rope` memoises into a module-level `_ROPE_DICT`
    (`rotary_embedding/__init__.py:29`), so every layer of a Llama shares ONE rope object. The
    first version bound the capture onto that object and guarded re-installation with a flag on it:
    layer 1 saw the flag, skipped, and its raw K went into the stash under layer 0's name. The run
    then failed at layer 1 with "no raw K stashed for layer 'model.layers.1.self_attn.attn'".

    So the shared object must come out of the install untouched, and each block must name its own
    stash entry.
    """
    shared = _Rope()
    blk0 = _Block("model.layers.0.self_attn", rope=shared)
    blk1 = _Block("model.layers.1.self_attn", rope=shared)

    patch._install_capture(blk0)
    patch._install_capture(blk1)

    assert blk0.rotary_emb._inner is shared and blk1.rotary_emb._inner is shared
    assert blk0.rotary_emb is not blk1.rotary_emb, "each block needs its own wrapper"

    k = torch.ones(2, 4)
    blk1.rotary_emb(torch.arange(2), torch.zeros(2, 4), k)
    assert patch._RAW_K_STASH, "the shared-rope layer stashed nothing"
    assert patch.take_raw_k("model.layers.1.self_attn.attn")["raw_k"] is k, (
        "layer 1's raw K was filed under another layer's name -- which is what a rope-keyed capture "
        "does when the rope is shared")
    assert shared.calls == 1


def test_capture_carries_the_models_own_rope_table():
    blk = _Block("model.layers.0.self_attn")
    patch._install_capture(blk)

    blk.rotary_emb(torch.arange(3), torch.zeros(3, 4), torch.ones(3, 4))

    entry = patch.take_raw_k("model.layers.0.self_attn.attn")
    assert entry["cos_sin_cache"] is blk.rotary_emb.cos_sin_cache


def test_keyword_key_is_accepted():
    blk = _Block("model.layers.0.self_attn")
    patch._install_capture(blk)
    k = torch.ones(2, 4)

    blk.rotary_emb(torch.arange(2), torch.zeros(2, 4), key=k)

    assert patch.take_raw_k("model.layers.0.self_attn.attn")["raw_k"] is k


def test_missing_key_refuses_rather_than_stashing_nothing():
    blk = _Block("model.layers.0.self_attn")
    patch._install_capture(blk)

    with pytest.raises(RuntimeError, match="without a key tensor"):
        blk.rotary_emb(torch.arange(2), torch.zeros(2, 4), key=None)
    assert patch._RAW_K_STASH == {}, "a refused capture must not leave a stale entry behind"


def test_install_refuses_a_block_it_cannot_key():
    class _NoRope(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attention("model.layers.0.self_attn.attn")

    with pytest.raises(RuntimeError, match="no rotary_emb"):
        patch._install_capture(_NoRope())


def test_install_is_idempotent():
    blk = _Block("model.layers.0.self_attn")
    inner = blk.rotary_emb
    patch._install_capture(blk)
    wrapper = blk.rotary_emb
    patch._install_capture(blk)

    assert blk.rotary_emb is wrapper, "a second install must not wrap the wrapper it already made"
    assert wrapper._inner is inner
    blk.rotary_emb(torch.arange(2), torch.zeros(2, 4), torch.ones(2, 4))
    assert inner.calls == 1, "chaining would run the rope more than once"
