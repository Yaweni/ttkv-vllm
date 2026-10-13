"""The model-forward half: capture RAW K and write it into the paged store.

This module is the reason the integration is possible at all, and the reason it cannot be a
backend-only change.

Every standard vLLM model rotates BEFORE the attention layer sees anything:

    # vllm/model_executor/models/llama.py:228-233
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([...], dim=-1)
    q, k = self.rotary_emb(positions, q, k)     # <- rotation happens HERE
    attn_output = self.attn(q, k, v)            # <- cache write happens in here

and `Attention.forward` receives no `positions` and no cos/sin cache, so a backend has no way to
rotate and no way to ever observe raw K. The ordering is owned by the MODEL forward.

The supported way to change it is the pattern `HpcRopeNorm` established
(`vllm/model_executor/layers/hpc/rope_norm.py`, wired into `models/hy_v3.py:310-343`):

  1. a module in the model forward performs the cache write itself;
  2. the backend sets `forward_includes_kv_cache_update = True`, so `Attention.forward` skips its
     generic `unified_kv_cache_update` (`attention.py:541-568`);
  3. a metadata flag coordinates the two (`hpc_kv_written`).

Ours differs in one decisive way: **HPC stores AFTER the rotation, we store BEFORE it.** That
single move is the whole ratio argument -- RoPE disperses the key block's spectrum, so the
same error budget needs ~3.6x more rank post-rotation.

PREVIOUSLY MEASURED for a re-rotation of the whole block; this module does it per stored row
inside the kernel instead, so treat those as an upper bound, not a prediction.
"""
from __future__ import annotations

import json
import pathlib
import re

import torch

from vllm_ttkv.layout import TTKLayout, layout_for_config

# The HPC module keeps a module-global instance registry keyed by `layer_name`
# (`rope_norm.py:28`) because the torch op is a module-level function that routes back by name.
#
# TWO REGISTRIES, AND THEY ARE NOT THE SAME THING. The one above holds layouts that exist because a
# forward ran. `_CALIBRATED` holds layouts that exist because a TABLE said so, and it is the one
# `customize_spec` must read: vLLM sizes the cache at init, before any sequence and therefore before
# any forward, so `_STORES` is empty at exactly the moment the slot width is needed. Deriving the
# layout from a live store works in a test and cannot work in vLLM.
_STORES: dict[str, "TTKVRopeStore"] = {}
_CALIBRATED: dict[int, TTKLayout] = {}
_CALIBRATION: dict = {}


def _layer_index(layer_name: str) -> int | None:
    """Pull the transformer block index out of a vLLM layer name.

    vLLM names are like `model.layers.7.self_attn` or `...layers.7`. The `layers.N` form is matched
    first; the bare last-integer fallback exists because a spec's `layer_name` is not guaranteed to
    carry the full module path, and a table lookup that silently misses would report "uncalibrated"
    for a layer that IS calibrated.
    """
    m = re.search(r"layers\.(\d+)", layer_name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(\d+)", layer_name)
    return int(nums[-1]) if nums else None


def load_calibration(path, cfg_name: str = "safe", cfg: dict | None = None,
                     layer_prefix: str = "model.layers") -> dict:
    """Populate the per-layer layouts from a `tools/calibrate_r1.py` artifact.

    `cfg` is the allocation (`ttkv_codec.CONFIGS[name]`); pass it explicitly when the codec is not
    importable from the runtime path, which is the situation for an installed plugin today -- the
    codec lives in `integration/` and is not yet packaged. Passing it in keeps this function usable
    either way instead of turning a packaging gap into an import error deep inside a lookup.

    Returns the artifact's metadata so a caller can log what it actually loaded. The metadata is not
    decoration: `r1` is meaningless without the corpus and context it was measured on, so anything
    that reports a ratio or a slot size from this table should be able to say where the table came
    from.
    """
    global _CALIBRATION
    if cfg is None:
        from ttkv_codec import CONFIGS
        cfg = CONFIGS[cfg_name]

    data = json.loads(pathlib.Path(path).read_text())
    r1 = data["r1"]
    if len(r1) != data["layers"]:
        raise ValueError(f"{path}: r1 has {len(r1)} entries but layers={data['layers']}")

    _CALIBRATED.clear()
    for idx, rank in enumerate(r1):
        _CALIBRATED[idx] = layout_for_config(cfg, int(rank), H=data["heads"], d=data["head_dim"])
    _CALIBRATION = dict(data)
    return _CALIBRATION


def calibration_info() -> dict:
    """What was loaded, or `{}`. Callers that print a slot size should print this beside it."""
    return dict(_CALIBRATION)


def layout_for_layer(layer_name: str | None) -> TTKLayout | None:
    """The layout for this layer: a live store's if one exists, else the calibrated table's.

    Order matters only for the forward path, where a real store has already fitted a decoder and its
    own rank is the truth. Before any forward -- i.e. inside `customize_spec`, which is the only
    caller that gates cache allocation -- only the calibrated table can answer.

    None is a real answer, not an error to swallow: the slot width decides the cache's allocation,
    so a layer without a calibrated `r1` cannot be served at all. Callers raise.
    """
    if layer_name is None:
        return None
    store = _STORES.get(layer_name)
    if store is not None:
        return store.layout
    idx = _layer_index(layer_name)
    return _CALIBRATED.get(idx) if idx is not None else None


def store_for_layer(layer_name: str | None) -> "TTKVRopeStore | None":
    """The live store for this layer, or None. The store's existence is the evidence that a
    forward has run for this layer, which is why it is separate from the calibrated layout."""
    return _STORES.get(layer_name) if layer_name else None


def decoder_for_layer(layer_name: str | None):
    """The fitted `[r1, H*d]` fp32 decoder for this layer, or None before the first fit.

    PER LAYER, and that is not an optimisation. `r1` is the rank of this layer's own key block --
    measured 6..166 across the layers of a 16k pre-rotary store -- so a decoder belongs to a layer
    exactly as its layout does. Sharing one across layers would mean fitting a basis that has to
    span every layer's key diversity at once, which is the mistake the codec's pooling note
    records as changing the answer (measured 5.8 in absolute score) rather than only the cost.

    None is a real answer: the read path needs this decoder, and nothing else produces it, so a
    caller that gets None is being told the store has not been built rather than being handed a
    default that would reconstruct plausible nonsense.
    """
    store = store_for_layer(layer_name)
    return store.decoder_f if store is not None else None


# ---------------------------------------------------------------- per-REQUEST decoders
# THE FITTED BASIS BELONGS TO A SEQUENCE, NOT TO A LAYER, and this registry is where that lives.
#
# Every request's prefill is its own TT-SVD fit, so two requests in one batch have two different
# decoders of the same SHAPE. Serving them with one shared basis is not a rounding difference: a
# rank-r1 basis that has to span B sequences must cover B times the key diversity, so every
# sequence is approximated more coarsely than it would be alone. The codec measured that directly --
# a batch slice disagreed with that sequence's own single-sequence run by 5.8 in ABSOLUTE score --
# which is why its pooled store gives each sequence its own decoder instead of relabelling tokens.
#
# ADDRESSED BY `block_table[row, 0]`, and that choice is forced rather than convenient. vLLM's
# `CommonAttentionMetadata` carries `query_start_loc`, `seq_lens`, `block_table_tensor` and
# `slot_mapping` -- NO request identifiers -- so `req_id` is not reachable from a kernel. A
# request's first physical block is: it is fixed for the request's lifetime, it is already in the
# batch metadata, and it is the same handle the encoder has when it registers the fit. So a kernel
# could address the decoder the same way it addresses the rows.
#
# LIFETIME: entries are keyed by a block id, and a block is recycled as soon as its request
# finishes, so a stale entry is a leak that also becomes WRONG if the block is reused. The encoder
# that registers these is the thing that must release them (`release_decoder`), on request
# completion.
_DECODERS: dict[tuple[str, int], "KDecoder"] = {}

# The env var that selects the allocation (`ttkv_codec.CONFIGS[...]`). Defined HERE, in the module
# that has no vLLM dependency and that both the backend and the encoder already import, because the
# two readers must agree: the backend sizes the cache from it via the calibration table, and the
# encoder fits the store from it. Two string literals for one env var would drift into a cache sized
# for one allocation and filled at another -- which does not raise, it just misreads every field
# after the coefficients.
CONFIG_ENV = "VLLM_TTKV_CONFIG"


def request_key_host(row) -> int:
    """`request_key`, from a block table row ALREADY on the host.

    `impl._Step` reads the whole block table back once per step (to validate it), and this is the
    same key taken from that copy -- one transfer per step instead of one readback per
    (layer, request), which is 28 times less.
    """
    if not len(row):
        raise ValueError(
            "the block table row is empty, so this request has no blocks and no identity to key a "
            "decoder by. A decode request always owns at least one block (it has at least one "
            "token), so reaching this means the metadata was built for a batch that is not one.")
    return int(row[0])


def request_key(block_row) -> int:
    """A stable per-request identity, taken from the request's own block table.

    `block_row[0]` is the request's first physical block. See the note above for why this rather
    than a request id: the id does not exist in the metadata, and the address has to be one a
    kernel could compute as well as the host can.
    """
    if block_row.numel() == 0:
        raise ValueError(
            "the block table row is empty, so this request has no blocks and no identity to key a "
            "decoder by. A decode request always owns at least one block (it has at least one "
            "token), so reaching this means the metadata was built for a batch that is not one.")
    return int(block_row[0])


def register_decoder(layer_name: str, key: int, decoder: "KDecoder") -> None:
    """Publish a request's fitted decoder for this layer. Called by the encoder after its fit.

    Its presence is ALSO the fold schedule: a request that already has one is past its first chunk,
    so its next step is a frozen-decoder projection rather than another fit. That is why no separate
    per-request counter exists -- the registry is the state, and it is a pure function of which
    requests the batch contains.
    """
    _DECODERS[(layer_name, key)] = decoder


def decoder_for_request(layer_name: str | None, key: int) -> "KDecoder | None":
    """This request's own fitted decoder, or None. None is a real answer, not a fallback:
    `impl.forward` refuses rather than substituting a layer-level basis when a second request is
    decoding."""
    if layer_name is None:
        return None
    return _DECODERS.get((layer_name, key))


def release_decoder(layer_name: str, key: int) -> None:
    """Drop a finished request's decoder. The eviction half of `register_decoder`."""
    _DECODERS.pop((layer_name, key), None)


def clear_decoders() -> None:
    """Drop every registration. For teardown, and for a test that needs a clean slate."""
    _DECODERS.clear()


class TTKVRopeStore:
    """Holds the true RoPE table and writes raw K into the paged store.

    `cos_sin_cache` is passed IN from the model's own `rotary_emb`, exactly as HPC does
    (`cos_sin_cache=self.rotary_emb.cos_sin_cache`, `hy_v3.py:316`). That is not a convenience:
    Llama-3.1's `inv_freq` carries a llama3 frequency ramp that `config.rope_theta` does not
    reproduce, and a wrong frequency set is still an orthogonal, position-dependent transform --
    so a mistake here fails SILENTLY rather than loudly.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 cos_sin_cache: torch.Tensor, layer_name: str, layout: TTKLayout):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cos_sin_cache = cos_sin_cache
        self.layer_name = layer_name
        self.layout = layout
        # `[r1, H*d]` fp32, set once the first chunk's fit has run; None before that. The READ path
        # needs it and nothing else produces it, so `decoder_for_layer` returning None is the
        # honest signal that the store has not been built yet rather than a default to fall back on.
        self.decoder_f: torch.Tensor | None = None
        _STORES[layer_name] = self

    # ------------------------------------------------------------------ no write path here
    # THERE USED TO BE A `forward()` HERE, AND IT WAS THE WRONG PLACE FOR THE WRITE.
    #
    # It took `(qkv, positions, slot_mapping)` and claimed to replace the model's own
    # `self.rotary_emb(positions, q, k)` call outright. But the thing that captures raw K
    # (`patch.py`) intercepts `rotary_emb`, which has NO slot mapping -- vLLM's slot mapping reaches
    # the attention layer and nothing earlier. So a store method that requires one could never have
    # been called from the place it claimed to sit, and the moment the impl was written there would
    # have been two competing writers for one cache.
    #
    # The write lives in `impl._write_store`: the first point in the step holding BOTH the raw K
    # (stashed by the wrapper) and the slot mapping, and the only place that knows the per-request
    # split -- which matters, because the fitted decoder is per request. What stays here is what is
    # genuinely shared: the angle table, the layout, and the registries. `encode.py` holds the slot
    # arithmetic.


def ensure_store(layer_name: str, cos_sin_cache: torch.Tensor, num_heads: int, num_kv_heads: int,
                 head_dim: int, layout: TTKLayout | None) -> "TTKVRopeStore":
    """The store for this layer, created on first use.

    Created by the WRITER rather than at model init, deliberately. The store needs the calibrated
    layout, and at init time that table may not be loaded -- `customize_spec` is what loads it, and
    it runs at cache allocation, whose ordering relative to model construction is not something this
    plugin should be relying on. By the time a prefill is being encoded the cache exists, so the
    layout does too.
    """
    store = _STORES.get(layer_name)
    if store is not None:
        return store
    if layout is None:
        raise RuntimeError(
            f"cannot create a TT-KV store for layer {layer_name!r}: no calibrated layout. The "
            f"store's angle table is needed to rotate stored rows at read time, and the layout "
            f"sizes the slots it is written into. Point VLLM_TTKV_CALIBRATION at a "
            f"tools/calibrate_r1.py artifact.")
    return TTKVRopeStore(num_heads, num_kv_heads, head_dim, cos_sin_cache, layer_name, layout)

