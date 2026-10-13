"""Inject the pre-rotation cache write into a model's forward, from a plugin.

WHY THIS FILE EXISTS
--------------------
The ordering lives in the model forward (see `rope_store.py`), so a plugin that wants raw K has
to intervene there. Upstream, HPC did it by editing the model file (`hy_v3.py`). Out-of-tree,
the equivalent is to intercept the model's own RoPE call, and that turns out to be a smaller
intervention than rewriting a model's forward body:

    q, k = self.rotary_emb(positions, q, k)

is called with RAW q and k. So intercepting the model's own RoPE call captures raw k with no
duplication of the model body -- which matters, because a duplicated body drifts from upstream
on every refactor, and this file would then be silently wrong rather than loudly broken.

The capture stashes raw k for the step; `impl._write_store` is immediately next, in the same
layer and the same forward, and it writes the store because it is the first point in the step that
knows the SLOT MAPPING. Splitting it that way is what keeps the interception to one seam, and it
is why nothing here needs to know about cache layout, request identity or the fold schedule: this
file's job is to hand over an un-rotated key that the model would otherwise have rotated in
place.

A SECOND HOOK LIVES HERE, for the same structural reason: `_install_spec_sizing` binds
`get_kv_cache_spec` on the attention instance. vLLM asks the MODULE for the spec and the BACKEND to
customise it, and the layer name is gone by the second call -- but TT-KV's slot width is per layer.
Both hooks are installed only on layers whose resolved backend is TT-KV, so a plain
`--attention-backend FLASH_ATTN` run gets an unmodified engine and is a fair baseline.

`take_raw_k` POPS rather than gets, so a second consumer in the same step is an error instead of a
stale read -- and the write is once per layer per step, because `forward` is called once per layer
per step and the stash is keyed by layer.

WHAT MAKES THIS THE RISKY FILE
------------------------------
It depends on:
  * the model calling `self.rotary_emb(positions, q, k)` with args in that order;
  * the stash being consumed in the same forward pass, by the same layer;
  * the attention layer routing to `TTKVAttentionBackend`.
None of those is enforced by vLLM. So `install()` RAISES when it matches nothing, and the
wrapper raises when its assumptions do not hold. A quiet miss here does not degrade the model --
it serves POST-ROPE keys, which is the exact outcome the whole integration exists to avoid, and
the result is plausible rather than broken.

For a production deployment, prefer the upstream shape: ship a model variant that constructs the
store explicitly, the way `hy_v3.py` does. This patcher is the out-of-tree proof, not the
end state.
"""
from __future__ import annotations

import functools
import logging

from torch import nn

# INSIDE vLLM'S LOGGER HIERARCHY, deliberately. vLLM configures its own logger with
# `propagate: False` and a handler of its own (`vllm/logger.py`), so a sibling logger like
# `vllm_ttkv` has no handler at all and its INFO records are silently dropped -- measured: the
# per-layer trace lines never reached the engine log. As a CHILD of `vllm` they inherit that handler
# and appear with the same formatter, which is what makes a bring-up trace useful.
logger = logging.getLogger("vllm.ttkv")

# Arch names whose attention modules are known to call rotary_emb(positions, q, k). Extend as
# each is verified -- an entry here is a claim, not a wish.
SUPPORTED_ARCHS = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
}

_PATCHED: set[str] = set()
_RAW_K_STASH: dict[str, dict] = {}


def _attention_cls(module: str, name: str):
    """Import a model's attention class, distinguishing ABSENT from BROKEN.

    This distinction is not pedantry, it is the difference between a useful error and a misleading
    one. The first version caught bare `Exception` and recorded a miss, so a missing shared
    library anywhere in the import chain reported as "no supported attention class found" --
    measured: `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so`
    surfaced as "this model family is unsupported", which sends the investigation at the wrong
    thing entirely.

    So: `ModuleNotFoundError` on the vLLM module means this build genuinely does not carry that
    arch, which IS a miss. Anything else propagates with its own cause, because it is an
    environment fault and retrying the other archs will not fix it.
    """
    import importlib

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    cls = getattr(mod, name, None)
    if cls is None:
        logger.warning("vllm_ttkv: %s has no %s; skipping this arch", module, name)
    return cls


def install() -> None:
    """Wrap `rotary_emb` on every supported attention class that the registry knows about.

    Raises rather than warns when nothing matches: with the wrapper absent, the backend still
    runs and still returns correctly-shaped tensors -- over post-RoPE keys. Failing loudly is the
    only safe option.
    """
    matched = 0
    for module, name in (("vllm.model_executor.models.llama", "LlamaAttention"),
                         ("vllm.model_executor.models.qwen2", "Qwen2Attention")):
        cls = _attention_cls(module, name)
        if cls is not None:
            _wrap_rotary(cls)
            matched += 1

    if matched == 0:
        raise RuntimeError(
            "vllm_ttkv: no supported attention class found to wrap (looked for "
            f"{sorted(SUPPORTED_ARCHS)}). Without the wrapper the cache will hold POST-ROPE "
            "keys, which is the failure this integration exists to prevent -- so this refuses "
            "to load rather than serving wrong numbers.")
    logger.info("vllm_ttkv: pre-rotation cache write installed on %d attention class(es)", matched)


def stash_raw_k(layer_name: str, raw_k, positions, cos_sin_cache=None) -> None:
    """Hand this step's un-rotated key -- and the angle table it came from -- to the writer.

    Called by the wrapper below, and by tests that need to drive the write path without standing up
    a model. `raw_k` must be the key BEFORE the rotation -- which is exactly what the wrapper sees,
    since it intercepts the rotation call rather than following it.

    `cos_sin_cache` rides along because this is the ONLY point in the step holding the model's real
    rope module. The attention layer does not (`Attention` has a `layer_name` and no `rotary_emb`;
    the rope lives one level up, on the model's attention block), so a writer reaching for
    `layer.rotary_emb` finds nothing -- and the table it would want is the model's OWN, with the
    llama3 frequency ramp that `config.rope_theta` does not reproduce.
    """
    _RAW_K_STASH[layer_name] = {"raw_k": raw_k, "positions": positions,
                                "cos_sin_cache": cos_sin_cache}


def _wrap_rotary(cls) -> None:
    """Make every instance of `cls` hand over raw k on the way into its rotation."""
    name = cls.__qualname__
    if name in _PATCHED:
        return

    original_init = cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_hooks(self)

    cls.__init__ = patched_init
    _PATCHED.add(name)


def _install_hooks(attn) -> None:
    """Install the two per-layer hooks TT-KV needs, on a TT-KV layer only.

    BOTH HOOKS ARE CONDITIONAL ON THE LAYER'S OWN BACKEND, and that is deliberate rather than an
    optimisation: this plugin is loaded in EVERY vLLM process (that is how entry points work), so
    without the check a plain `--attention-backend FLASH_ATTN` run would also get a rope
    indirection on every layer and a TT-KV-sized spec. The baseline arm of a comparison has to be
    the engine a user would otherwise get, or the comparison is measuring the plugin's overhead
    against itself. `install()` still counts ARCH matches independently, so "we patched nothing"
    still raises.
    """
    inner = getattr(attn, "attn", None)
    if inner is None or not _is_ttkv_layer(inner):
        return
    _install_capture(attn)
    # The SPEC hook goes on `inner`, because that is the module vLLM asks for it:
    # `get_layers_from_vllm_config` yields `Attention` instances, not the model's attention block.
    _install_spec_sizing(inner, getattr(inner, "layer_name", None))


def _is_ttkv_layer(inner) -> bool:
    """True if this attention module's resolved backend is TT-KV."""
    from vllm_ttkv.backend import TTKVAttentionBackend

    try:
        backend = inner.get_attn_backend()
    except Exception:                                     # pragma: no cover - older/newer API
        backend = getattr(inner, "attn_backend", None)
    cls = backend if isinstance(backend, type) else type(backend)
    return isinstance(cls, type) and issubclass(cls, TTKVAttentionBackend)


def _install_spec_sizing(attn, layer_name: str) -> None:
    """Bind a per-layer `get_kv_cache_spec` so the cache is sized from THIS layer's `r1`.

    WHY THE HOOK IS HERE AND NOT IN `customize_spec`: vLLM takes the spec from the module and
    customises it on the backend,

        spec = attn_module.get_kv_cache_spec(vllm_config)
        spec = attn_module.get_attn_backend().customize_spec(spec)      # <- no layer name

    so the layer name exists at the first call and is gone by the second. TT-KV's slot width is
    per layer (`r1` is a property of each layer's data), so `customize_spec` cannot do the job on
    its own -- measured, it was reached with the name unrecoverable and cache allocation stopped
    with "no calibrated TT-KV layout for layer None".

    The bind is an instance attribute, the same technique as the rope capture and for the same
    reason: `attn_module.get_kv_cache_spec(...)` resolves on the instance, so an instance
    attribute shadows the class method without touching any class the engine also uses.
    """
    if layer_name is None:
        raise RuntimeError(
            "vllm_ttkv: a TT-KV attention module has no `layer_name`, so its cache page cannot be "
            "sized -- the slot width comes from that layer's calibrated `r1`. Refusing rather than "
            "sizing every layer alike.")
    original = attn.get_kv_cache_spec

    def get_kv_cache_spec(vllm_config):
        spec = original(vllm_config)
        if spec is None:
            return None
        from vllm.v1.kv_cache_interface import AttentionSpec

        from vllm_ttkv.backend import pack_spec_for_layer

        if not isinstance(spec, AttentionSpec):
            return spec
        return pack_spec_for_layer(spec, layer_name)

    attn.get_kv_cache_spec = get_kv_cache_spec


def _install_capture(attn) -> None:
    """Wrap THIS attention block's rotary module so raw k is caught on the way through.

    TWO THINGS DECIDED THIS SHAPE, AND BOTH WERE MEASURED RATHER THAN ASSUMED.

    1. THE ROTARY MODULE IS SHARED BETWEEN LAYERS. `get_rope` memoises into a module-level
       `_ROPE_DICT` (`rotary_embedding/__init__.py:29`), and every Llama layer asks for the same
       key -- so layer 1's `rotary_emb` IS layer 0's object. The first version of this function
       bound the capture ONTO THAT OBJECT, closing over a name fixed at install time, and guarded
       re-installation with a flag on it. With one shared object the second layer saw the flag,
       skipped, and its raw K was stashed under LAYER 0's name. The real run died at layer 1 with
       "no raw K stashed for layer 'model.layers.1.self_attn.attn'" -- the loud form of that bug.
       So the capture is now per ATTENTION BLOCK, closing over its own `layer_name`, wrapping a rope
       that any number of blocks may share.

    2. THE WRAPPER HAS TO BE AN `nn.Module`. Assigning a plain object over `self.rotary_emb` is
       refused outright: `Module.__setattr__` accepts only a `Module`/`Tensor`/`Parameter` for a
       child name (measured: `TypeError: cannot assign 'vllm_ttkv.patch._RawKCapturingRotary' as
       child module 'rotary_emb'`). So it subclasses `nn.Module` and keeps the real rope with
       `object.__setattr__`, which holds it OUT of the module tree -- nothing below `rotary_emb` is
       renamed and `named_modules()` reports the same names it did before.

    `attn.attn` is where the key comes from: the model's attention block (`LlamaAttention`) owns the
    rope, while `Attention.__init__` sets `self.layer_name = prefix`, and that is the name the IMPL
    is handed as its `layer`. Taking the key from the same object the reader uses makes write and
    read agree by construction rather than by two conventions happening to match.
    """
    rope = getattr(attn, "rotary_emb", None)
    inner = getattr(attn, "attn", None)
    layer_name = getattr(inner, "layer_name", None) or getattr(attn, "layer_name", None)

    why = None
    if rope is None:
        why = "it has no rotary_emb"
    elif layer_name is None:
        why = "its attention submodule carries no layer_name"
    if why is not None:
        raise RuntimeError(
            f"vllm_ttkv: cannot capture raw K on {type(attn).__name__} because {why}. Without the "
            f"capture the store would receive POST-ROPE keys, which is the failure this "
            f"integration exists to prevent, so this refuses rather than installing a no-op.")
    if isinstance(rope, _RawKCapturingRotary):
        return
    attn.rotary_emb = _RawKCapturingRotary(rope, layer_name)


class _RawKCapturingRotary(nn.Module):
    """Delegates the rotation and stashes the key it was handed, BEFORE it is rotated.

    One wrapper per attention block, even when several blocks share one rope (see
    `_install_capture`). The layer name is a closure, not an attribute of the rope, because the rope
    may be a single object serving every layer -- and the first version of this file learned that
    the expensive way.
    """

    def __init__(self, inner, layer_name):
        super().__init__()
        # NOT `self._inner = inner`. Under `nn.Module.__setattr__` that REGISTERS the real rope as a
        # submodule, renaming everything beneath it (`rotary_emb._inner.cos_sin_cache`), and every
        # checkpoint load would then look for a name that does not exist.
        object.__setattr__(self, "_inner", inner)
        self._ttkv_layer_name = layer_name

    def __getattr__(self, item):
        # This shadows `nn.Module.__getattr__` for anything the wrapper does not own, so the guard
        # on `_inner` itself is load-bearing: without it, a probe for an attribute that is not in
        # `__dict__` would recurse until the stack ran out.
        if item == "_inner":
            raise AttributeError(item)
        return getattr(self._inner, item)

    def forward(self, positions, query, key=None, *args, **kwargs):
        raw_k = key if key is not None else kwargs.get("key")
        if raw_k is None:
            raise RuntimeError(
                "vllm_ttkv: rotary_emb was called without a key tensor, so raw K cannot be "
                "captured. The assumption that the model passes k to its rope no longer holds for "
                "this model version.")
        stash_raw_k(self._ttkv_layer_name, raw_k, positions,
                    getattr(self._inner, "cos_sin_cache", None))
        return self._inner(positions, query, key, *args, **kwargs)


def take_raw_k(layer_name: str):
    """Pop the raw-K stash for this layer, or raise if the wrapper never ran.

    Called by the attention impl, which is the first place in the step that knows the slot
    mapping. Popping (not getting) makes a double-consume an error rather than a stale read.
    """
    try:
        return _RAW_K_STASH.pop(layer_name)
    except KeyError:
        raise RuntimeError(
            f"vllm_ttkv: no raw K stashed for layer {layer_name!r}. The rotary wrapper did not "
            f"run before attention, so the store would receive post-RoPE keys. Refusing.")
