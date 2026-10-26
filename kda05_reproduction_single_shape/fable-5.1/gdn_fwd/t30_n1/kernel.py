"""
Fused Gated Delta Net prefill kernel (qk4_v8_d128, k-last state layout) for NVIDIA B200.

The whole operator (gate computation, grouped-value head expansion, the per-sequence
recurrent delta-rule scan, output projection and final-state writeback) runs in ONE
CUDA kernel, `gdn_kernel.cu`, which is compiled from source at import time with nvcc
(-arch=sm_100a). The compiled shared library is driven through ctypes so that no
extra GPU work (allocations, memsets, host syncs) happens around the launch.

Entry point (destination-passing style):
    kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "gdn_kernel.cu")
_LIB = None
_FN = None


def _find_nvcc() -> str:
    cands = []
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            cands.append(os.path.join(CUDA_HOME, "bin", "nvcc"))
    except Exception:
        pass
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    cands += ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"]
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    for c in cands:
        if os.path.isfile(c):
            return c
    raise RuntimeError("nvcc not found; set CUDA_HOME")


def _build() -> str:
    with open(_SRC, "rb") as f:
        src = f.read()
    extra = os.environ.get("GDN_NVCC_EXTRA", "")  # dev only, e.g. "-lineinfo"
    tag = hashlib.sha1(src + extra.encode()).hexdigest()[:16]
    cache_root = os.environ.get("GDN_KERNEL_CACHE") or os.path.join(
        tempfile.gettempdir(), "gdn_prefill_fused_cache"
    )
    os.makedirs(cache_root, exist_ok=True)
    lib_path = os.path.join(cache_root, f"libgdn_prefill_{tag}.so")
    if not os.path.isfile(lib_path):
        tmp_path = lib_path + f".tmp{os.getpid()}"
        cmd = [_find_nvcc(), "-O3", "-std=c++17", "-arch=sm_100a"]
        cmd += extra.split()
        cmd += ["--shared", "-Xcompiler", "-fPIC", "-o", tmp_path, _SRC]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "nvcc failed:\n" + " ".join(cmd) + "\n" + res.stdout + "\n" + res.stderr
            )
        os.replace(tmp_path, lib_path)
    return lib_path


def _get_fn():
    global _LIB, _FN
    if _FN is None:
        _LIB = ctypes.CDLL(_build())
        fn = _LIB.gdn_prefill_launch
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # state (may be NULL)
            ctypes.c_void_p,  # A_log
            ctypes.c_void_p,  # a
            ctypes.c_void_p,  # dt_bias
            ctypes.c_void_p,  # b
            ctypes.c_void_p,  # cu_seqlens (int64)
            ctypes.c_float,  # scale
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # new_state
            ctypes.c_int,  # num_seqs
            ctypes.c_int,  # total_seq_len
            ctypes.c_void_p,  # cuda stream
        ]
        _FN = fn
    return _FN


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    fn = _get_fn()
    if scale is None:
        scale_f = 0.0
    elif isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    if scale_f == 0.0:
        scale_f = 128.0**-0.5
    num_seqs = int(new_state.shape[0])
    total = int(q.shape[0])
    state_ptr = state.data_ptr() if state is not None else 0
    stream = torch.cuda.current_stream(q.device).cuda_stream
    rc = fn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        state_ptr,
        A_log.data_ptr(),
        a.data_ptr(),
        dt_bias.data_ptr(),
        b.data_ptr(),
        cu_seqlens.data_ptr(),
        scale_f,
        output.data_ptr(),
        new_state.data_ptr(),
        num_seqs,
        total,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"gdn_prefill_launch failed with CUDA error {rc}")
