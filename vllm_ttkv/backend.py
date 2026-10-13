"""`AttentionBackend` for TT-KV.

Modelled on the two in-tree precedents for the two things this backend needs:

  * `HpcAttentionBackend` (`vllm/v1/attention/backends/hpc_attn.py`) -- a backend that owns its
    own KV-cache write (`forward_includes_kv_cache_update = True`, `hpc_attn.py:288`), and
    cooperates with a module in the model forward through an attention-metadata flag
    (`hpc_kv_written`).
  * `TurboQuantAttentionBackend` (`turboquant_attn.py:123-161`) -- a packed cache format
    expressed through `customize_spec` by setting `state_content_bytes`.

VERSION NOTE: every symbol below was verified at vllm-project/vllm @ `276fbcff`. vLLM's
attention surface is actively moving -- `customize_spec` is explicitly marked a temporary hook
with issue #42449 as its end state -- so check each signature against the tree you build
against before assuming a failure is yours.
"""
from __future__ import annotations

import os

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ttkv.impl import TTKVAttentionImpl
from vllm_ttkv.metadata import TTKVAttentionMetadataBuilder
from vllm_ttkv.rope_store import calibration_info, layout_for_layer, load_calibration

# Where the calibration artifact comes from. Explicit, not auto-discovered: silently picking a table
# that happens to be on disk could size the cache for a different model, and a wrong `r1` is still a
# valid rank -- the store would reconstruct, just at the wrong width, and nothing would raise.
CALIBRATION_ENV = "VLLM_TTKV_CALIBRATION"
_CALIBRATION_CFG_ENV = "VLLM_TTKV_CONFIG"


def _ensure_calibration() -> None:
    """Load the table once, from `$VLLM_TTKV_CALIBRATION`."""
    if calibration_info():
        return
    path = os.environ.get(CALIBRATION_ENV)
    if not path:
        return
    info = load_calibration(path, cfg_name=os.environ.get(_CALIBRATION_CFG_ENV, "safe"))
    import logging
    logging.getLogger("vllm.ttkv").info(
        "vllm_ttkv: calibration %s -- model=%s context=%d r1 %d..%d",
        path, info["model"], info["context"], min(info["r1"]), max(info["r1"]))

# Head dims with a compiled `krt`/`v_out` instantiation. See `supports_head_size`.
#
# What a supported set has to cover, from the 43 configs surveyed (2026-10): ~75% are 128
# (Llama 2/3/3.1, Qwen2/2.5/3-8B/32B, Mistral, Yi, GLM-4, MiniMax, OLMo-2, DBRX, Command-R);
# 256 is a Gemma signature plus Qwen3-Next's full-attention layers (Gemma-1-7B, Gemma-2-2B/9B,
# Qwen3-Next-80B-A3B); 64 is gpt-oss-120b/20b, Llama-3.2-1B, Qwen2.5-0.5B, Falcon-180B; 96 is
# Phi-3/3.5-mini. So {64, 96, 128, 256} covers everything realistic, and the ORDER to add them
# in is decided by which models you intend to serve -- 128 alone serves Llama-3.1-8B and the
# Qwen2.5/3 dense line, which is the target today.
#
# `head_size` must be READ FROM THE CONFIG, never computed as `hidden_size / num_heads`. Both
# counterexamples are common: Qwen3-32B sets `head_dim: 128` explicitly while
# `hidden/heads = 5120/64 = 80`, and Gemma-2-9B sets 256 while `3584/16 = 224`. vLLM's
# `AttentionSpec.head_size` already comes from the field, so the trap is only for code that
# derives the dim itself.
_SUPPORTED_HEAD_DIMS = frozenset({128})


class TTKVAttentionBackend(AttentionBackend):
    """Pre-rotary TT-KV. Stores raw K and rotates at read time."""

    # The module in the model forward writes the cache, so `Attention.forward` must not also
    # write it -- same contract as HPC (`attention.py:541-568`).
    forward_includes_kv_cache_update = True

    # An EXISTING dtype string, deliberately. A brand-new name would have to be added to the
    # closed `CacheDType` Literal in `vllm/config/cache.py:39-61`, which is an upstream edit;
    # HPC reuses ["auto", "bfloat16", "fp8_e4m3"] and is selected by the platform selector
    # instead. `selector.py:126-131` is what would reject an unknown string.
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        """The `AttentionBackendEnum` MEMBER NAME, which for the third-party slot is `CUSTOM`.

        NOT a free-form identifier, and this is not cosmetic: `Attention.__init__` does
        `self.backend = AttentionBackendEnum[self.attn_backend.get_name()]`
        (`model_executor/layers/attention/attention.py`), i.e. a NAME lookup that raises for a name
        with no member. Returning `"CUSTOM_TTKV"` here failed there, after the engine had already
        started and loaded weights -- and the error it gives ("Unknown attention backend:
        'CUSTOM_TTKV'. Valid options are: ... CUSTOM") is easy to misread as a registration failure
        when the registration actually worked and the backend WAS selected.

        Every in-tree backend follows the same rule and returns its own member name (`FLASH_ATTN`,
        `HPC_ATTN`, `TURBOQUANT`). Nothing is lost by using the generic name: the class that
        implements the layer is reported separately by `Attention.extra_repr` as
        `backend={impl.__class__.__name__}`, i.e. `TTKVAttentionImpl`.
        """
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TTKVAttentionImpl

    @staticmethod
    def get_builder_cls():
        return TTKVAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls):
        """LBNHC, like HPC (`hpc_attn.py:322`) and TurboQuant. The packed content is a flat
        per-token blob, so it does not care about the interleaving; picking an already-supported
        layout keeps `resolve_kv_cache_layout` (`vllm/v1/attention/backends/utils.py`) able to
        satisfy the whole model from the intersection of its backends."""
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """Head dims we have compiled `krt` / `v_out` instantiations for.

        128 only so far, and this is a statement about the KERNELS rather than about the layout.
        `TTKLayout` is parameterised on `d`, and its arithmetic generalises: across
        d in {64, 128, 256, 512} the joint ratio moves only 4.27x..4.57x once the lane FRACTION
        and the kv-hidden are held fixed (measured, `tests/test_layout.py`). What does NOT
        generalise for free is the compiled kernel -- `ttkv_kernels.cu` is specialised on HD, and
        `krt`'s tiling asserts `(KRT_TILE//KRT_MT) * (HD//KRT_MD) == _THREADS`, so each new head
        dim needs its own instantiation AND a re-derived config (`k_lanes` is absolute, see
        `layout.py`). Listing a dim here claims both exist.
        """
        return head_size in _SUPPORTED_HEAD_DIMS

    @classmethod
    def customize_spec(cls, spec):
        """Verify a spec that has ALREADY been packed for its own layer.

        READ THIS BEFORE MOVING THE SIZING BACK IN HERE. vLLM's per-layer cache sizing is
        `attn_utils.get_kv_cache_spec`, which iterates `layer_name -> Attention` and calls
        `attn_module.get_kv_cache_spec(vllm_config)`, then hands the RESULT to this hook:

            spec = attn_module.get_kv_cache_spec(vllm_config)
            spec = attn_module.get_attn_backend().customize_spec(spec)   # <- no layer_name
            kv_cache_spec[layer_name] = spec

        The layer NAME is visible at the first of those two points and not at the second, which is
        fatal for a format whose slot width is per layer: `r1` is a property of each layer's own
        data, so the width has to be chosen per layer. Measured, this hook was reached with the
        name unrecoverable and the run stopped at cache allocation with "no calibrated TT-KV
        layout for layer None".

        So the sizing lives in `pack_spec_for_layer`, called from `patch._install_spec_sizing`,
        which binds `get_kv_cache_spec` on the attention INSTANCE -- the only place holding both
        the spec and the name. This hook stays as the guard for that arrangement rather than as the
        implementation: a spec arriving here still unpacked means the per-layer hook did not run,
        and an unpacked spec is not a slightly-wrong page, it is a page sized `(head_size +
        head_size_v) * dtype_size` -- i.e. dense K‖V -- into which TT-KV writes `k_slot + v_slot`
        bytes per token. Every write would land inside the allocation and read back as noise.
        """
        if spec.state_content_bytes is not None and spec.num_head_slots == 1:
            return spec
        raise RuntimeError(
            "a TT-KV spec reached `customize_spec` unpacked, so the per-layer hook in "
            "`patch._install_spec_sizing` did not run -- most likely `patch.install()` was not "
            "called in this process, or this layer's attention backend is not TT-KV. Refusing: "
            "an unpacked spec is sized for dense K‖V, and the packed writes would stay inside the "
            "allocation while reading back as noise.")


def pack_spec_for_layer(spec, layer_name: str):
    """Size ONE layer's packed page from that layer's calibrated `r1`.

    Lives outside the backend class because it is called from `patch.py` (on the attention
    instance, where the layer name exists) rather than from the `customize_spec` hook (where it
    does not). See `TTKVAttentionBackend.customize_spec` for why that split is forced.

    This is where TT-KV stops being a `k`-bit format and becomes a two-region one: the
    spec's `head_size` is replaced by the slot width, and `num_head_slots` collapses to 1
    because the coefficients are shared across heads (`layout.py`).

    THE HARD PART, and it is a design constraint rather than a coding detail: **the cache is
    allocated before the sequence exists, but the slot width depends on `r1`, which is a
    property of the data.** So `r1` cannot be measured at prefill and used to size the
    cache in the same run. It has to be decided in advance, and there are only two honest
    ways:
      (a) a calibration pass (one forward over a representative corpus at the target
          context, per layer) whose per-layer `r1` table is shipped as an artifact; or
      (b) cap the rank at a budget, which is a format change, not a tuning knob -- measured,
          a cap of 64 under-ranks `tight` by 1.75x and `campaign4x` by ~4.4x.
    (a) is what vLLM itself does for `max_num_batched_tokens`-style limits, and it is what
    this skeleton assumes: `rope_store` holds a per-layer table, populated from a
    calibration artifact. Until that artifact exists, this raises rather than guessing.

    A PER-LAYER width is fine -- it does NOT have to be padded up to a global maximum, and
    no second cache region is needed for it. `UniformTypeKVCacheSpecs`
    (`vllm/v1/kv_cache_interface.py:1225`) holds one spec per layer and its
    `page_size_bytes` is the SUM across layers, so each layer keeps its own page inside the
    group's block; `get_max_layers_per_page_size` even counts repetitions per distinct page
    size, i.e. several widths in one group are expected. That is the path DeepSeek-V4's MLA
    layers take. Two requirements, both satisfiable: every TT layer must be the same
    registered base spec with the same `block_size` (so they bundle into one group), and the
    backend must declare a block-compact layout. If TT layers are mixed with a different
    attention type or block size, the fallback (`unify_kv_cache_spec_page_size`) pads every
    layer to the max, and raises outright for MLA-shaped specs. So: compress all layers,
    do not mix.
    """
    from dataclasses import replace

    _ensure_calibration()
    lay = layout_for_layer(layer_name)
    if lay is None:
        raise RuntimeError(
            f"no calibrated TT-KV layout for layer {layer_name!r}. The slot width must be "
            f"known BEFORE the cache is allocated, so it cannot be derived from a prefill.\n"
            f"  Produce one with:  python tools/calibrate_r1.py --model <id> "
            f"--context <max_model_len> --corpus <parquet dir> --out <table.json>\n"
            f"  Then point vLLM at it with:  {CALIBRATION_ENV}=<table.json>\n"
            f"Shipped allocations: VLLM_TTKV_CONFIG in "
            f"{{safe, tight, campaign4x}} (default safe).")

    # The table is only valid for the geometry it was measured on. `r1` is the rank of the
    # [n, d*H] unfolding, so a table measured at d=128 is not merely imprecise at d=64 -- it
    # is a rank for a different matrix. Cheap to check, and the failure it prevents is silent.
    info = calibration_info()
    table_d = info.get("head_dim")
    spec_d = int(getattr(spec, "head_size", 0) or 0)
    if table_d and spec_d and int(table_d) != spec_d:
        raise RuntimeError(
            f"calibration was measured at head_dim={table_d} but this spec has "
            f"head_size={spec_d}. The table is keyed to the geometry it was fitted on; "
            f"re-run tools/calibrate_r1.py for this model.")

    # `state_content_bytes` is C in `[B, H_slots, N, C]`, and it is the WHOLE per-token cell:
    # `state_content_size_bytes` returns it verbatim when set, and otherwise falls back to
    # `(head_size + head_size_v) * dtype_size` -- i.e. dense K followed by dense V
    # (`v1/kv_cache_interface.py`). So the packed region must be K's slot FOLLOWED BY V's, and
    # one token's blob is `k_slot_bytes + v_slot_bytes`.
    #
    # Publishing `k_slot_bytes` alone was a real bug and a silent one: the region would be
    # sized for K only, so every V write would land in the NEXT token's coefficients while
    # staying inside the allocation. Nothing raises, and the K store reads back as noise.
    #
    # `num_head_slots = 1` makes the logical page `[N, 1, C]`, i.e. one packed blob per token,
    # which is the two-region-avoiding encoding `layout.py` argues for.
    blob = lay.compressed_bytes_per_token()
    if blob != lay.k_slot_bytes + lay.v_slot_bytes:
        raise RuntimeError(
            f"internal: compressed_bytes_per_token() returned {blob} but the two slots sum to "
            f"{lay.k_slot_bytes + lay.v_slot_bytes}")
    return replace(spec, num_head_slots=1, state_content_bytes=blob, head_size=blob)
