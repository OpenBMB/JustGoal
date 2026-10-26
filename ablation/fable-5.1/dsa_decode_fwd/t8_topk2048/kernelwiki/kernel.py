"""DSA sparse attention (decode) for B200 — entry point `run` per the flashinfer-bench definition
dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.

The CUDA kernel (dsa_kernel.cu) is compiled with nvcc on first use into a shared library next to this
file and launched through ctypes on the current torch stream. One kernel launch per call (one CTA per SM,
64-row work items assigned by status rank, split-KV reduction through a cached global workspace of
(value, epoch) words), no memsets.
"""
import ctypes
import os
import re
import subprocess
import threading

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROF = os.environ.get("DSA_PROF", "0") == "1"
_LIB = None
_LOCK = threading.Lock()


def _find_nvcc():
    for cand in (
        os.environ.get("NVCC"),
        os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
    ):
        if cand and os.path.isfile(cand):
            return cand
    return "nvcc"


def _build():
    src = os.path.join(_HERE, "dsa_kernel.cu")
    _extra = os.environ.get("DSA_EXTRA", "")
    _tag = ("_" + re.sub(r"[^A-Za-z0-9_]", "", _extra.replace("-D", ""))) if _extra else ""
    so = os.path.join(_HERE, f"libdsa_v5{'_prof' if _PROF else ''}{_tag}.so")
    if (not os.path.exists(so)) or os.path.getmtime(so) < os.path.getmtime(src):
        cmd = [
            _find_nvcc(), "-O3", "-std=c++17", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC",
            "-use_fast_math",
            "-o", so, src,
        ] + (["-DPROF"] if _PROF else []) + ([_extra] if _extra else [])
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return so


def _lib():
    global _LIB
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                lib = ctypes.CDLL(_build())
                lib.dsa_launch.restype = ctypes.c_int
                lib.dsa_launch.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ]
                lib.dsa_num_chunks.restype = ctypes.c_int
                _LIB = lib
    return _LIB


_WS = {}


def _workspace(T, device):
    """Per-(T, device) split-KV workspace: partial O and (m, l, epoch-tag) vectors, plus the launch epoch."""
    key = (T, device.index if device.index is not None else torch.cuda.current_device())
    ws = _WS.get(key)
    if ws is None:
        n = _lib().dsa_num_chunks()
        ws = [
            torch.zeros((T, n, 4112), dtype=torch.int64, device=device),  # partial O: (bf16x2, epoch) words
            torch.zeros((T, n, 40), dtype=torch.int64, device=device),    # ((m, epoch), (l, epoch)) per head
            0,                                                                 # epoch of the last launch
        ]
        torch.cuda.synchronize(device)
        _WS[key] = ws
    return ws


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, Hh, Dc = q_nope.shape
    assert Hh == 16 and Dc == 512 and q_pe.shape[-1] == 64
    assert ckv_cache.shape[1] == 64 and sparse_indices.shape[-1] == 2048
    assert q_nope.dtype == torch.bfloat16 and ckv_cache.dtype == torch.bfloat16
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    sparse_indices = sparse_indices.contiguous()
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    device = q_nope.device
    out = torch.empty((T, 16, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((T, 16), dtype=torch.float32, device=device)
    ws = _workspace(T, device)
    ws[2] += 1  # tags are compared against the launch epoch, so no reset between launches is needed
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = _lib().dsa_launch(
        q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(), kpe_cache.data_ptr(),
        sparse_indices.data_ptr(), float(sm_scale), out.data_ptr(), lse.data_ptr(),
        ws[0].data_ptr(), ws[1].data_ptr(), int(ws[2]), int(T), stream,
    )
    if rc != 0:
        raise RuntimeError(f"dsa_launch failed with CUDA error {rc}")
    return out, lse
