"""Make the vendored codec importable, so `pytest` works from a clean checkout.

`vllm_ttkv` imports `ttkv_codec` and `ttkv_packed_cache` as TOP-LEVEL modules — that is how the
original project laid them out, and renaming them would mean touching every import in the port for no
behavioural gain. They are vendored under `codec/`, and this puts that directory on the path so the
tests do not depend on the caller having set PYTHONPATH.

The same directory is what `[tool.pytest.ini_options] pythonpath` names in `pyproject.toml`; this is
here as well because a `conftest.py` is also loaded when a single test file is run by path, which the
packaging option is not.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "codec"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
