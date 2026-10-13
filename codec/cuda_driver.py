"""Minimal CUDA driver-API binding: NVRTC -> PTX -> ctypes -> launch on torch's own stream.

PORTABLE ACROSS WINDOWS AND LINUX. The driver is `nvcuda.dll` on Windows and `libcuda.so.1` on
Linux, and NVRTC is `nvrtc64_*.dll` on one and `libnvrtc.so*` on the other; both are the same API.
It became portable because the project now needs it in both places: this box runs the codec on
Windows natively, while the vLLM plugin's actual deployment target is Linux. `os.name` is the only
branch.

WHY THIS EXISTS INSTEAD OF torch.utils.cpp_extension
    `cpp_extension` needs ninja AND a host compiler to build and link the pybind11 glue. `nvcc` only
    needs a host compiler in order to LINK a shared library -- `nvcc -cubin` emits pure device code
    and needs nothing but nvcc itself -- but on Windows nvcc still refuses to START without `cl.exe`
    on PATH even for `-cubin`, and NVRTC is what avoids that startup requirement entirely.

    The original note here also said the box was offline and ninja therefore could not be installed.
    THAT PREMISE IS NO LONGER TRUE (pypi.org answers 200), so it is recorded rather than relied on:
    ninja is installable, and on Linux neither blocker exists at all. What still holds on the native
    Windows side is the host compiler -- see `compile_cubin` for the version wall that makes MSVC
    unusable with CUDA 12.4 here.

    So the plan is: compile to a cubin, load it through the CUDA driver API via ctypes, and launch
    it against tensors torch allocated. torch already holds the primary context on device 0, so
    `cuDevicePrimaryCtxRetain` returns that same context, the one torch is allocating in. An address
    from `tensor.data_ptr()` is then valid inside the launched kernel -- no copying, no interop, no
    shared-memory handshake. Ordering comes for free because launches go on torch's current stream.

    What this buys over the pybind route: no host compiler, no linker, no ABI to match, and direct
    control of grid and block per launch. The last one is the point. The win being chased here is
    fusion, and fusion means several small kernels per layer rather than one library call per stage;
    a binding that hides the launch config would be the wrong binding.
"""
import ctypes
import os
import subprocess

import torch

__all__ = ["CudaDriver", "Module", "compile_cubin", "compile_ptx", "nvrtc_available",
           "CUDAError"]

_IS_WIN = os.name == "nt"
_NVCC_DEFAULT = (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
                 if _IS_WIN else "/usr/local/cuda/bin/nvcc")


def _load_driver():
    """The CUDA driver library. Same API, two names."""
    return ctypes.WinDLL("nvcuda.dll") if _IS_WIN else ctypes.CDLL("libcuda.so.1")


class CUDAError(RuntimeError):
    pass


class CudaDriver:
    """Just the driver calls a launch needs, bound by hand."""

    def __init__(self):
        self.lib = _load_driver()
        L = self.lib
        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuInit.restype = ctypes.c_int
        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGet.restype = ctypes.c_int
        L.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        L.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        L.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        L.cuCtxSetCurrent.restype = ctypes.c_int
        L.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        L.cuModuleLoad.restype = ctypes.c_int
        L.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        L.cuModuleLoadData.restype = ctypes.c_int
        L.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                          ctypes.c_char_p]
        L.cuModuleGetFunction.restype = ctypes.c_int
        L.cuLaunchKernel.argtypes = [ctypes.c_void_p,
                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                     ctypes.c_uint, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_void_p)]
        L.cuLaunchKernel.restype = ctypes.c_int
        L.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.cuGetErrorString.restype = ctypes.c_int

        self.check(L.cuInit(0), "cuInit")
        dev = ctypes.c_int(0)
        self.check(L.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.dev = dev.value
        ctx = ctypes.c_void_p()
        # The PRIMARY context, which is the one torch is already using. Creating a fresh context
        # instead would put the kernel in a different address space and every pointer would be
        # garbage -- silently, with no error, just wrong numbers.
        self.check(L.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev),
                   "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        self.check(L.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def err_str(self, code):
        p = ctypes.c_char_p()
        self.lib.cuGetErrorString(code, ctypes.byref(p))
        return (p.value or b"?").decode(errors="replace")

    def check(self, code, what):
        if code != 0:
            raise CUDAError(f"{what} failed: {code} {self.err_str(code)}")
        return code

    @classmethod
    def instance(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = cls()
        return cls._inst


class Module:
    """A loaded module -- either a .cubin built by nvcc, or PTX built by NVRTC.

    PTX is the route that actually works on this box. `cuModuleLoadData` on a PTX string JITs it
    against the installed driver at load time, once, and every launch after that is native.
    """

    def __init__(self, cubin_path=None, ptx=None, name="kernels"):
        if (cubin_path is None) == (ptx is None):
            raise ValueError("give exactly one of cubin_path, ptx")
        self.drv = CudaDriver.instance()
        self.mod = ctypes.c_void_p()
        if ptx is not None:
            text = ptx if isinstance(ptx, bytes) else ptx.encode()
            buf = ctypes.create_string_buffer(text)     # NUL-terminated by create_string_buffer
            self.drv.check(self.drv.lib.cuModuleLoadData(ctypes.byref(self.mod), buf),
                           "cuModuleLoadData")
            self.source_desc = f"<nvrtc:{name}, {len(text)} B PTX>"
        else:
            self.cubin = os.path.abspath(cubin_path)
            self.drv.check(self.drv.lib.cuModuleLoad(
                ctypes.byref(self.mod),
                self.cubin.encode("mbcs" if _IS_WIN else "utf-8")), "cuModuleLoad")
            self.source_desc = self.cubin
        self._fns = {}

    def func(self, name):
        if name not in self._fns:
            h = ctypes.c_void_p()
            self.drv.check(self.drv.lib.cuModuleGetFunction(ctypes.byref(h), self.mod,
                                                            name.encode()), "cuModuleGetFunction")
            self._fns[name] = h
        return self._fns[name]

    @staticmethod
    def _arg(v):
        """Wrap one kernel argument. The holder must outlive the launch, hence the caller keeps the
        list -- ctypes passes a POINTER to the holder, so a temporary would be a dangling pointer."""
        if isinstance(v, torch.Tensor):
            return ctypes.c_void_p(v.data_ptr())
        if isinstance(v, bool):
            return ctypes.c_int(int(v))
        if isinstance(v, int):
            return ctypes.c_int(v)
        if isinstance(v, float):
            return ctypes.c_float(v)
        raise TypeError(f"cannot pass {type(v)} as a kernel argument")

    def launch(self, name, grid, block, args, shared=0, stream=None):
        """grid/block are ints or 1-, 2- or 3-tuples. `stream` defaults to torch's current stream."""
        gx, gy, gz = self._dims(grid)
        bx, by, bz = self._dims(block)
        holders = [self._arg(a) for a in args]
        ptrs = (ctypes.c_void_p * len(holders))(*(ctypes.addressof(h) for h in holders))
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        self.drv.check(self.drv.lib.cuLaunchKernel(
            self.func(name), gx, gy, gz, bx, by, bz, shared,
            ctypes.c_void_p(stream), ptrs, None), f"cuLaunchKernel({name})")
        return holders          # keep alive until the function returns, at least

    @staticmethod
    def _dims(d):
        """Pad to 3, accepting 1-, 2- or 3-tuples. A 2-tuple grid is the natural shape for several
        of these kernels and refusing it here would just push the padding into every caller."""
        if isinstance(d, int):
            return d, 1, 1
        d = tuple(d) + (1,) * (3 - len(d))
        return d[0], d[1], d[2]


_CCBIN = {"dir": None, "done": False}


def find_host_compiler():
    """The directory holding cl.exe, if it is not already on PATH.

    nvcc on Windows refuses to start without a host compiler -- `nvcc fatal: Cannot find compiler
    'cl.exe' in PATH` -- even for `-cubin`, which emits no host code and never links. So 'no host
    compiler needed' is true of the link step but NOT of nvcc's startup. Visual Studio is installed
    on this box but its compiler is not on PATH, so point nvcc at it with `-ccbin` rather than
    requiring a vcvars shell.

    The search is recursive on purpose. Visual Studio nests as `<year>\\<edition>\\VC\\Tools\\...`,
    so a single-`*` pattern like `<root>\\*\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64\\cl.exe` silently
    matches nothing on VS 2019/2022/18 -- and 'silently matches nothing' here just looks like the
    confusing original error again.
    """
    if _CCBIN["done"]:
        return _CCBIN["dir"]
    from glob import glob
    from shutil import which
    _CCBIN["done"] = True
    if which("cl"):
        return None                                   # on PATH already, nvcc will find it
    roots = [os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                          "Microsoft Visual Studio"),
             os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                          "Microsoft Visual Studio")]
    hits = []
    for root in roots:
        if os.path.isdir(root):
            hits += glob(os.path.join(root, "**", "Hostx64", "x64", "cl.exe"), recursive=True)
    if not hits:
        return None
    hits.sort()
    _CCBIN["dir"] = os.path.dirname(hits[-1])         # newest toolset wins
    return _CCBIN["dir"]


def compile_cubin(cu_path, cubin_path=None, arch="sm_86", extra=(), nvcc=None, verbose=True):
    """nvcc -cubin. Kept for completeness but NOT the route this box takes -- see compile_ptx.

    Dead on arrival here, for a reason worth recording so nobody rediscovers it: CUDA 12.4 bundles
    cl.exe for its own startup, but this box's only host compiler is MSVC 14.51 (VS 18), and that
    STL refuses CUDA below 13.2:
        yvals_core.h(911): error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer.
    `-allow-unsupported-compiler` gets past nvcc's own version gate and then walks straight into
    that static assert, which is in Microsoft's headers and cannot be flag-flipped away.
    """
    cu_path = os.path.abspath(cu_path)
    if cubin_path is None:
        cubin_path = os.path.splitext(cu_path)[0] + ".cubin"
    nvcc = nvcc or (os.environ.get("NVCC") or _NVCC_DEFAULT)
    if not os.path.exists(nvcc):
        from shutil import which
        nvcc = which("nvcc")
        if not nvcc:
            raise CUDAError("nvcc not found")
    cmd = [nvcc, "-cubin", f"-arch={arch}", "-O3", "-o", cubin_path, cu_path, *extra]
    ccbin = find_host_compiler()
    if ccbin:
        # CUDA 12.4's host_config.h rejects anything newer than VS 2022; this box has VS 18 with
        # MSVC 14.51. The override is safe HERE and only here: `-cubin` emits no host object code,
        # so cl.exe is used to preprocess and nothing else, and none of its output is linked or run.
        # Do not carry this flag to a build that produces a host binary.
        cmd += ["-ccbin", ccbin, "-allow-unsupported-compiler"]
    if verbose:
        print(f"  nvcc -cubin -arch={arch}"
              + (f" -ccbin {ccbin}" if ccbin else " (cl.exe on PATH)"))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise CUDAError(f"nvcc failed ({p.returncode}):\n{' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return cubin_path


# ------------------------------------------------------------------ NVRTC, the route that works
# NVRTC is the CUDA runtime compiler: it takes CUDA C++ source and emits PTX, using its own minimal
# device-side headers and NEVER invoking a host compiler. That is exactly what is needed here, where
# the host compiler on the box is newer than CUDA 12.4 can preprocess.
#
# The trade is that NVRTC has no host standard library -- no <cstdio>, no <vector>, nothing that
# lives on the host side. For these kernels that costs nothing: they are pure device code taking
# raw pointers, and they were always going to be.
_NVRTC = {}


def _nvrtc_lib():
    if "lib" in _NVRTC:
        return _NVRTC["lib"]
    from glob import glob
    import torch
    tlib = os.path.dirname(torch.__file__)
    if _IS_WIN:
        roots = [os.environ.get("CUDA_PATH"),
                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
                 os.path.join(tlib, "lib")]
        pats = ("nvrtc64*.dll", os.path.join("bin", "nvrtc64*.dll"))
        loader = ctypes.WinDLL
    else:
        # torch's own wheel ships NVRTC under `nvidia/cuda_nvrtc/lib` when the toolchain is not
        # installed system-wide, which is exactly this box's situation inside WSL.
        roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                 "/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/lib/x86_64-linux-gnu",
                 os.path.join(tlib, "lib"),
                 os.path.join(tlib, os.pardir, "nvidia", "cuda_nvrtc", "lib")]
        pats = ("libnvrtc.so*", os.path.join("lib64", "libnvrtc.so*"))
        loader = ctypes.CDLL
    found = []
    for r in roots:
        if r and os.path.isdir(r):
            for p in pats:
                found += [f for f in glob(os.path.join(r, p)) if os.path.isfile(f)]
    if not found:
        _NVRTC["lib"] = None
        return None
    # A bare `libnvrtc.so` sorts before the versioned soname, so the last entry is the most
    # specific one present -- a broken dev symlink cannot win over a real library.
    found.sort()
    lib = loader(found[-1])
    lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                       ctypes.c_char_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_char_p),
                                       ctypes.POINTER(ctypes.c_char_p)]
    lib.nvrtcCreateProgram.restype = ctypes.c_int
    lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_char_p)]
    lib.nvrtcCompileProgram.restype = ctypes.c_int
    lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetPTXSize.restype = ctypes.c_int
    lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetPTX.restype = ctypes.c_int
    lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetProgramLog.restype = ctypes.c_int
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.nvrtcDestroyProgram.restype = ctypes.c_int
    _NVRTC["lib"] = lib
    _NVRTC["path"] = found[-1]
    return lib


def nvrtc_available():
    return _nvrtc_lib() is not None


def compile_ptx(src, arch="sm_86", name="kernels.cu", options=("--std=c++17",), verbose=True):
    """CUDA C++ source (str or bytes) -> PTX text. No host compiler in the path at all."""
    lib = _nvrtc_lib()
    if lib is None:
        raise CUDAError("nvrtc64*.dll not found")
    if isinstance(src, str):
        src = src.encode()
    opts = [f"--gpu-architecture={arch}"] + list(options)
    c_opts = [o.encode() for o in opts]
    arr = (ctypes.c_char_p * len(c_opts))(*c_opts)

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, name.encode(), 0, None, None)
    if rc != 0:
        raise CUDAError(f"nvrtcCreateProgram failed: {rc}")
    try:
        rc = lib.nvrtcCompileProgram(prog, len(c_opts), arr)
        if rc != 0:
            n = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(max(n.value, 1))
            lib.nvrtcGetProgramLog(prog, log)
            raise CUDAError(f"nvrtcCompileProgram failed ({rc}):\n{log.value.decode(errors='replace')}")
        n = ctypes.c_size_t()
        if lib.nvrtcGetPTXSize(prog, ctypes.byref(n)) != 0:
            raise CUDAError("nvrtcGetPTXSize failed")
        buf = ctypes.create_string_buffer(n.value)
        if lib.nvrtcGetPTX(prog, buf) != 0:
            raise CUDAError("nvrtcGetPTX failed")
        ptx = buf.value
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
    if verbose:
        print(f"  nvrtc {_NVRTC.get('path')} -> {len(ptx)} B PTX  ({' '.join(opts)})")
    return ptx


def load_source(src, arch="sm_86", name="kernels.cu", verbose=True):
    """Source -> Module, via NVRTC if present and nvcc otherwise."""
    if nvrtc_available():
        return Module(ptx=compile_ptx(src, arch=arch, name=name, verbose=verbose), name=name)
    raise CUDAError("NVRTC unavailable and nvcc unusable on this box; no CUDA compile route")
