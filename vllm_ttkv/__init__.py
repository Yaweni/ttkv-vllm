"""Out-of-tree vLLM integration for TT-KV: a tensor-train KV cache stored PRE-ROPE.

Why a plugin first, and why this shape
--------------------------------------
vLLM's standing recommendation for a new attention backend is to prove it out-of-tree; the
plugin path needs no upstream review and no files in the vLLM tree. Three facts about vLLM
(verified against vllm-project/vllm @ 276fbcff) drive every design choice here.

1. **A backend cannot rotate, and never sees raw K.** `Attention.forward` is
   `(query, key, value, output_shape, output_dtype)` -- no `positions`, no `cos_sin_cache` -- and
   every standard model rotates first (`llama.py`: `q, k = self.rotary_emb(positions, q, k)`
   then `self.attn(q, k, v)`). So the RoPE-versus-cache-write ordering lives in the MODEL
   forward, not in the attention layer, and it cannot be changed from a backend class alone.
   The supported way to change it is the pattern the HPC backend established: a module inside
   the model forward that performs the cache write itself, paired with
   `AttentionBackend.forward_includes_kv_cache_update = True` so `Attention.forward` skips its
   own generic `unified_kv_cache_update`. `rope_store.py` is our version of that module; it is
   INJECTED by `patch.py` rather than edited into each model file.

2. **No new `kv_cache_dtype` string is needed.** `CacheDType` is a closed `Literal` with no
   registration hook, so a brand-new dtype name would need an upstream edit. But
   `HpcAttentionBackend` reuses existing strings (`["auto", "bfloat16", "fp8_e4m3"]`) and is
   selected through the platform's `get_attn_backend_cls` / `AttentionConfig.backend`. Reusing
   `"auto"` keeps this plugin out of `vllm/config/` entirely -- which is the difference between
   a plugin and a fork.

3. **The layout is one region, not two.** Coefficients are per token and shared across heads,
   so the cache slot is per TOKEN with `num_head_slots = 1` and the head axis folded into the
   slot content. See `layout.py` -- this is what dissolves the block-size-divisibility question
   and removes the need for a second `KVCacheGroupSpec`.
"""

__all__ = ["register"]


def register() -> None:
    """Entry point for the `vllm.general_plugins` group. Loaded in EVERY vLLM process.

    Registered against the reserved third-party slot (`AttentionBackendEnum.CUSTOM`); the class
    path is resolved lazily, so importing this package does not require vLLM.

    TWO THINGS THAT WERE VERIFIED THE HARD WAY, both silent failures

    1. **DOTS, not a `module:attr` colon.** vLLM's resolver is `resolve_obj_by_qualname`, which is
       `qualname.rsplit(".", 1)` then `getattr`; a colon leaves the class name glued to the module
       part of the string and raises AttributeError later, at `Backend.get_class()` -- i.e. at
       engine startup rather than at import.
    2. **`AttentionBackendEnum.CUSTOM`, not a `"CUSTOM_TTKV"` string.** The string overload of
       `register_backend` (`backend: AttentionBackendEnum | str`, which dynamically mints a new
       enum member) exists on vLLM main but NOT in release 0.31.0, whose signature is
       `backend: AttentionBackendEnum`. Handed a string there, the function does not raise: it
       skips member creation entirely and returns a no-op decorator. Measured -- `hasattr(enum,
       "CUSTOM_TTKV")` stayed False and the path was never stored. A name that does not exist
       cannot be selected, so the backend would simply never be chosen.
       Because the failure is silent, the registration is ASSERTED below rather than assumed.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    path = "vllm_ttkv.backend.TTKVAttentionBackend"
    register_backend(AttentionBackendEnum.CUSTOM, path)

    stored = AttentionBackendEnum.CUSTOM.get_path()
    if stored != path:
        raise RuntimeError(
            f"vllm_ttkv: registration did not take. {AttentionBackendEnum.CUSTOM!r}.get_path() "
            f"returned {stored!r}, expected {path!r}. `register_backend` is silent when its first "
            f"argument is not an `AttentionBackendEnum` in this vLLM version, so this is checked "
            f"rather than trusted.")

    # The backend alone cannot see raw K (fact 1 above). This is the other half, and it is the
    # half that must fail LOUDLY: a model family whose forward does not match means the module
    # silently never runs, the cache silently holds post-RoPE keys, and every score afterwards
    # is plausible and wrong. `install()` therefore raises rather than skipping.
    from vllm_ttkv.patch import install

    install()
