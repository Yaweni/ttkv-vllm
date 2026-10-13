# codec/ — vendored so this repository runs standalone

These five files are the compression codec and the CUDA kernel that this plugin stores into. They are
not the subject of this repository; they are here so that a clean checkout can run the tests and the
benchmarks without a second project on `PYTHONPATH`.

| file | what it is |
| --- | --- |
| `ttkv_codec.py` | the tensor-train SVD and the quantiser: `CONFIGS`, `tt_svd`, `build_k_layer`, `build_v_layer` |
| `ttkv_packed_cache.py` | the reference container: `PackedRows`, `pack_k_slots`, `pack_v_slots`, the row layout |
| `ttkv_cuda.py` | the shipped pre-rotary `krt` kernel, compiled at import time through NVRTC |
| `cuda_driver.py` | the thin NVRTC/PTX driver `ttkv_cuda` compiles through |
| `ttkv_kernels.cu` | the kernel source |

They are imported as **top-level** modules (`import ttkv_codec`), matching the layout of the project
they came from. `conftest.py` at the repository root puts this directory on `sys.path`, and every
script under `tests/` and `tools/` adds it to its own bootstrap loop.

`ttkv_cuda.py` needs the CUDA runtime's NVRTC (`libnvrtc.so.12`, shipped with the `nvidia-cuda-nvrtc`
wheel). Without it, `tests/test_kernel_krt.py` and `tests/bench_krt.py` skip cleanly; everything else
is unaffected, because the Triton `krt` is the default path and needs no toolchain.

Licence and provenance for these files follow the main project rather than this repository.
