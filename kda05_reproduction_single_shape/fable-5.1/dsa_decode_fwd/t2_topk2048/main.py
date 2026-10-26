"""DSA sparse attention (h16, ckv512, kpe64, topk2048, page_size 64) for NVIDIA B200.

Self-contained solution: `kernel.cu` (in this directory) is compiled once with nvcc
for sm_100a into a shared library and launched through ctypes. The only runtime
dependency is torch (for tensors / the current stream). Destination-passing style:

    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernel.cu"

_LIB = None


def _find_nvcc() -> str:
    cands = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(env)
        if v:
            cands.append(Path(v) / "bin" / "nvcc")
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _TORCH_CUDA_HOME  # type: ignore

        if _TORCH_CUDA_HOME:
            cands.append(Path(_TORCH_CUDA_HOME) / "bin" / "nvcc")
    except Exception:
        pass
    w = shutil.which("nvcc")
    if w:
        cands.append(Path(w))
    cands += [Path("/usr/local/cuda/bin/nvcc"), Path("/usr/local/cuda-13/bin/nvcc")]
    for c in cands:
        if c.is_file():
            return str(c)
    raise RuntimeError("nvcc not found; set CUDA_HOME")


def _build() -> Path:
    src = _SRC.read_text()
    arch = os.environ.get("DSA_SM_ARCH", "sm_100a")
    extra = os.environ.get("DSA_NVCC_EXTRA", "").split()
    key = hashlib.sha1((src + arch + " ".join(extra) + "v1").encode()).hexdigest()[:16]
    cache_root = Path(os.environ.get("DSA_BUILD_DIR", Path(tempfile.gettempdir()) / "dsa_sparse_attn_build"))
    out_dir = cache_root / key
    lib = out_dir / "libdsa_sparse_attn.so"
    if lib.is_file():
        return lib
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"lib.{os.getpid()}.tmp.so"
    cmd = [
        _find_nvcc(),
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-lineinfo",
        "--use_fast_math",
        *extra,
        "-o",
        str(tmp),
        str(_SRC),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nvcc failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    os.replace(tmp, lib)
    return lib


def _get_lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(_build()))
        lib.dsa_launch.restype = ctypes.c_int
        lib.dsa_launch.argtypes = [
            ctypes.c_void_p,  # q_nope
            ctypes.c_void_p,  # q_pe
            ctypes.c_void_p,  # ckv
            ctypes.c_void_p,  # kpe
            ctypes.c_void_p,  # indices
            ctypes.c_float,  # sm_scale
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # lse
            ctypes.c_int,  # num_tokens
            ctypes.c_void_p,  # stream
        ]
        _LIB = lib
        _LIB._lib_path = str(_build())
    return _LIB


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    lib = _get_lib()
    num_tokens = q_nope.shape[0]
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    stream = torch.cuda.current_stream(q_nope.device).cuda_stream
    err = lib.dsa_launch(
        q_nope.data_ptr(),
        q_pe.data_ptr(),
        ckv_cache.data_ptr(),
        kpe_cache.data_ptr(),
        sparse_indices.data_ptr(),
        ctypes.c_float(float(sm_scale)),
        output.data_ptr(),
        lse.data_ptr(),
        int(num_tokens),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"dsa_launch failed with CUDA error {err}")
    return output, lse
