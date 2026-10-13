"""Does this machine's vLLM plugin surface actually resolve?

This is the FIRST vLLM thing to run, and it is deliberately not a benchmark: it only proves that
the plugin's own imports and its registration path work against the INSTALLED vLLM tree, which is
a different thing from working against the tree the docstrings were written against. Every
`file:line` reference in `backend.py`/`impl.py` was verified against vllm-project/vllm @ 276fbcff
(main, 2026-10-10); a PyPI or wheel-index release can differ, and `file:line` in a docstring
cannot fail a build.

WHY THIS EXISTS AS A SEPARATE STEP
    `register_backend("CUSTOM_TTKV", ...)` stores a CLASS PATH STRING and resolves it lazily, at
    `AttentionBackendEnum.get_class()`. So a bad path registers cleanly and then raises at engine
    startup -- the failure is real but late, and it looks like a missing backend rather than a typo.
    That is not hypothetical: this file was written because the path shipped with a `module:Class`
    colon (entry-point syntax) where vLLM's resolver needs `module.Class`, and
    `resolve_obj_by_qualname` is `qualname.rsplit(".", 1)` + `getattr`.

WHAT IT CHECKS
    1. `import vllm` working at all, and on which CUDA build.
    2. That a CUDA kernel can actually LAUNCH. A CUDA-13 wheel on a CUDA-12 driver imports fine and
       fails only here, so this is the check that tells you the wheel and the driver agree.
    3. Every symbol the plugin imports, resolved from the installed tree.
    4. `register()` -> `get_class()` round trip, which is the late failure described above.

Runs in the vLLM venv, not the kernel venv:
    /opt/ttkv/vllmvenv/bin/python tests/smoke_register_vllm.py
Exits non-zero on the first hard failure so it can gate a benchmark run.
"""
from __future__ import annotations

import pathlib
import sys
import traceback

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[1] / "integration", pathlib.Path("/opt/ttkv/vllm_ttkv_plugin"),
           pathlib.Path("/opt/ttkv/integration")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

FAILURES: list[str] = []


def check(name: str, fn):
    """Run one check, report it, and record a failure instead of aborting.

    Deliberately continues: knowing that ALL FOUR symbols are missing is more useful than knowing
    the first one is.
    """
    try:
        detail = fn()
    except Exception as exc:                                              # noqa: BLE001
        FAILURES.append(name)
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
        if "-v" in sys.argv:
            traceback.print_exc()
        return None
    print(f"  [ ok ] {name}: {detail}")
    return detail


print("=" * 78)
print("1. vLLM itself")
print("=" * 78)


def _vllm_identity():
    import vllm
    return f"vllm {vllm.__version__} at {pathlib.Path(vllm.__file__).parent}"


check("import vllm", _vllm_identity)


def _torch_cuda():
    import torch
    return (f"torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
            f"available={torch.cuda.is_available()}")


check("torch/CUDA build", _torch_cuda)


def _cuda_launch():
    """The check that distinguishes 'imports' from 'works'.

    A wheel built against a newer CUDA major than the driver supports imports perfectly and then
    raises `CUDA driver version is insufficient for CUDA runtime version` on the first real launch.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    a = torch.randn(512, 512, device="cuda")
    b = (a @ a.T).sum().item()
    return f"matmul on GPU ok (sum={b:.3f}), {torch.cuda.get_device_name(0)}"


check("CUDA kernel launch", _cuda_launch)

print()
print("=" * 78)
print("2. Symbols the plugin imports from vLLM")
print("=" * 78)

# Each entry is exactly what the plugin imports, so this list IS the coupling surface. Keeping it
# explicit rather than `import vllm_ttkv` is the point: a failure names the specific symbol that
# moved, instead of surfacing as one opaque ImportError from somewhere inside the package.
SYMBOLS = (
    ("vllm.v1.attention.backends.registry", "register_backend"),
    ("vllm.v1.attention.backend", "AttentionBackend"),
    ("vllm.v1.attention.backend", "AttentionImpl"),
    ("vllm.v1.attention.backend", "AttentionType"),
    ("vllm.v1.kv_cache_layout", "KVCacheLayout"),
    ("vllm.config.cache", "CacheDType"),
)


def _symbol(module: str, attr: str):
    import importlib
    mod = importlib.import_module(module)
    obj = getattr(mod, attr, None)
    if obj is None:
        raise AttributeError(f"{module}.{attr} does not exist in this version")
    return obj


for _mod, _attr in SYMBOLS:
    check(f"{_mod}.{_attr}", lambda m=_mod, a=_attr: f"found {_symbol(m, a)!r}"[:100])


def _cache_dtypes():
    """Which quantized KV-cache strings this version offers.

    This decides what the baseline can even be. The TurboQuant variants are the honest comparison
    for a compressed cache; if they are absent, the baseline degrades to fp8, which caps at 2x and
    is not a like-for-like floor for a 5-7x format.
    """
    import typing
    from vllm.config.cache import CacheDType
    vals = list(typing.get_args(CacheDType))
    quant = [v for v in vals if not v.startswith(("auto", "float", "bfloat"))]
    return f"{len(vals)} dtypes; quantized: {quant}"


check("CacheDType quantized strings", _cache_dtypes)


def _backend_enum():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    return f"CUSTOM={AttentionBackendEnum.CUSTOM!r}, has FLASH_ATTN={'FLASH_ATTN' in AttentionBackendEnum.__members__}"


check("AttentionBackendEnum", _backend_enum)

print()
print("=" * 78)
print("3. Our plugin: import, register, resolve")
print("=" * 78)


def _plugin_import():
    import vllm_ttkv
    return f"vllm_ttkv from {pathlib.Path(vllm_ttkv.__file__).parent}"


check("import vllm_ttkv", _plugin_import)

_registered = None


def _plugin_register():
    global _registered
    import vllm_ttkv
    vllm_ttkv.register()          # entry-point body: register_backend + patch.install()
    _registered = True
    return "register() completed without raising"


check("vllm_ttkv.register()", _plugin_register)


def _plugin_resolve():
    """The late failure. This is what `register_backend`'s stored path is finally resolved by.

    Via `AttentionBackendEnum.CUSTOM`, because release 0.31.0's `register_backend` takes an enum
    member only -- the `str` overload that mints a dynamic `CUSTOM_TTKV` member is on main, and
    handing the release a string silently registers nothing.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    cls = AttentionBackendEnum.CUSTOM.get_class()
    return f"resolved to {cls.__module__}.{cls.__qualname__}"


check("AttentionBackendEnum.CUSTOM.get_class()", _plugin_resolve)

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} failure(s): {FAILURES}")
    sys.exit(1)
print("RESULT: all checks passed -- the plugin's surface resolves against this vLLM.")
