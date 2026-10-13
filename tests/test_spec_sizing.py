"""Per-layer cache sizing: the one hook vLLM cannot give us, and the guard that replaces it.

THE BUG THIS FILE PINS. vLLM sizes the cache in two steps:

    spec = attn_module.get_kv_cache_spec(vllm_config)          # the MODULE has the layer name
    spec = attn_module.get_attn_backend().customize_spec(spec) # the BACKEND hook does not

`customize_spec` is a classmethod that receives the spec and nothing else, so a format whose slot
width is per layer -- TT-KV's, because `r1` is a property of each layer's own data -- cannot be
sized through it. Measured on the real run, it was called with the name unrecoverable and cache
allocation stopped with "no calibrated TT-KV layout for layer None". So the sizing moved to
`pack_spec_for_layer`, called from `patch._install_spec_sizing`, which binds `get_kv_cache_spec` on
the attention INSTANCE. These tests hold both ends of that arrangement: the sizing itself, and the
guard that fires if a spec ever reaches `customize_spec` unpacked.

An UNPACKED spec is not a slightly-wrong page. `state_content_bytes = None` means C falls back to
`(head_size + head_size_v) * dtype_size`, i.e. dense K‖V per token; TT-KV writes `k_slot + v_slot`
bytes per token, so every write would land inside the allocation, and the store would read back as
noise with nothing raising. That is why the fallback is a refusal rather than a default.

Gated on vLLM: `backend.py` and `patch._install_hooks` both import it.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

pytest.importorskip("vllm.v1.attention.backend")
from ttkv_codec import CONFIGS                                          # noqa: E402
from vllm.v1.kv_cache_interface import AttentionSpec                    # noqa: E402

from vllm_ttkv import patch                                             # noqa: E402
from vllm_ttkv.backend import (CALIBRATION_ENV, TTKVAttentionBackend,   # noqa: E402
                               pack_spec_for_layer)
from vllm_ttkv.rope_store import (calibration_info, layout_for_layer,   # noqa: E402
                                  load_calibration)

TABLE = pathlib.Path(__file__).resolve().parents[1] / "calibration" / "llama-3.2-3b-16384.json"
BLOCK = 16
KV_HEADS = 8
HEAD_DIM = 128
LAYER = "model.layers.0.self_attn.attn"


@pytest.fixture(autouse=True)
def _load_table(monkeypatch):
    monkeypatch.setenv(CALIBRATION_ENV, str(TABLE))
    monkeypatch.setenv("VLLM_TTKV_CONFIG", "safe")
    load_calibration(TABLE, cfg_name="safe", cfg=CONFIGS["safe"])


def _spec(head_size: int = HEAD_DIM):
    return AttentionSpec(block_size=BLOCK, num_kv_heads=KV_HEADS, head_size=head_size,
                         dtype=torch.bfloat16)


def test_the_table_is_the_one_the_serving_target_was_measured_on():
    info = calibration_info()
    assert info["head_dim"] == HEAD_DIM and info["heads"] == KV_HEADS
    assert len(info["r1"]) == info["layers"] == 28, "Llama-3.2-3B has 28 layers"
    assert min(info["r1"]) < max(info["r1"]), (
        "a flat table would make every per-layer assertion below vacuous")


def test_a_layer_is_sized_from_its_own_rank():
    packed = pack_spec_for_layer(_spec(), LAYER)
    lay = layout_for_layer(LAYER)

    assert packed.num_head_slots == 1, "coefficients are shared across heads: one blob per token"
    assert packed.state_content_bytes == lay.compressed_bytes_per_token()
    assert packed.state_content_size_bytes == lay.compressed_bytes_per_token(), (
        "the property vLLM actually reads must agree with the field we published")
    assert packed.head_size == packed.state_content_bytes, (
        "the packed slot replaces `head_size` so vLLM's own arithmetic agrees with ours")
    # Everything else about the layer must survive the repack untouched.
    assert (packed.block_size, packed.num_kv_heads, packed.dtype) == (BLOCK, KV_HEADS,
                                                                     torch.bfloat16)


def test_two_layers_with_different_ranks_get_different_widths():
    r1 = calibration_info()["r1"]
    thin = next(i for i, v in enumerate(r1) if v == min(r1))
    fat = next(i for i, v in enumerate(r1) if v == max(r1))
    name = "model.layers.{}.self_attn.attn"

    a = pack_spec_for_layer(_spec(), name.format(thin)).state_content_bytes
    b = pack_spec_for_layer(_spec(), name.format(fat)).state_content_bytes

    assert a < b, (
        "the higher-rank layer must get the wider page -- if these were equal the width is coming "
        "from something other than the layer, which is the failure this test exists for")


def test_an_uncalibrated_layer_refuses():
    with pytest.raises(RuntimeError, match="no calibrated TT-KV layout"):
        pack_spec_for_layer(_spec(), "model.layers.99.self_attn.attn")


def test_a_table_from_another_geometry_refuses():
    with pytest.raises(RuntimeError, match="measured at head_dim"):
        pack_spec_for_layer(_spec(head_size=64), LAYER)


def test_customize_spec_passes_a_packed_spec_through():
    packed = pack_spec_for_layer(_spec(), LAYER)
    assert TTKVAttentionBackend.customize_spec(packed) == packed


def test_customize_spec_refuses_an_unpacked_spec():
    with pytest.raises(RuntimeError, match="reached `customize_spec` unpacked"):
        TTKVAttentionBackend.customize_spec(_spec())


class _Rope:
    def __init__(self):
        self.calls = 0

    def forward(self, positions, query, key):
        self.calls += 1
        return query, key


class _Attention:
    """A stand-in for vLLM's `Attention`: a name, a backend, and a spec. Those three are the whole
    contract `_install_hooks` reads."""

    def __init__(self, backend, layer_name=LAYER):
        self._backend = backend
        self.layer_name = layer_name
        self.called = 0

    def get_attn_backend(self):
        return self._backend

    def get_kv_cache_spec(self, vllm_config):
        self.called += 1
        return _spec()


class _Block:
    def __init__(self, backend):
        self.attn = _Attention(backend)
        self.rotary_emb = _Rope()


def test_hooks_are_installed_only_on_ttkv_layers():
    block = _Block(TTKVAttentionBackend)

    patch._install_hooks(block)

    assert block.attn.get_kv_cache_spec(None).state_content_bytes is not None
    assert isinstance(block.rotary_emb, patch._RawKCapturingRotary), (
        "a TT-KV layer must carry the raw-K capture as well as the sizing")


def test_a_foreign_backend_is_left_alone():
    class _Other:
        pass

    block = _Block(_Other)
    # Compare `__func__`, not the bound method: a bound method is built fresh on every attribute
    # access, so `is` on the bound objects fails even when nothing was patched.
    original_spec = block.attn.get_kv_cache_spec.__func__

    patch._install_hooks(block)

    assert block.attn.get_kv_cache_spec.__func__ is original_spec, (
        "the baseline arm must be the engine a user would otherwise get: binding our spec sizing on "
        "a non-TT-KV layer would repack it against the TT-KV table")
    assert not isinstance(block.rotary_emb, patch._RawKCapturingRotary), (
        "and it must not carry the raw-K capture either, or the comparison measures our overhead "
        "against itself")
