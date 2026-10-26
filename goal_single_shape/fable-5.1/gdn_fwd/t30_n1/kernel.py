"""Gated Delta Net prefill forward (varlen, GVA, k-last state) -- custom CUDA kernel for B200.
Entry point: gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale) -> (output, new_state)
"""
import ctypes
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "gdn_kernel.cu")
_SO = os.path.join(_HERE, "libgdn_fwd.so")
_lib = None
_DBG = None


def _dbg_ptr(dev):
    global _DBG
    if _DBG is None:
        _DBG = torch.empty(4096 * 16, dtype=torch.int64, device=dev)   # only written by TIMING builds
    return _DBG.data_ptr()


def _build(so_path):
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "--shared", "-Xcompiler", "-fPIC",
           "-o", so_path, _SRC]
    subprocess.check_call(cmd)


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = None
    if os.path.isfile(_SO):
        try:
            lib = ctypes.CDLL(_SO)
        except OSError:
            lib = None
    if lib is None:
        so = os.path.join(_HERE, "libgdn_fwd_local.so")
        if not os.path.isfile(so):
            _build(so)
        lib = ctypes.CDLL(so)
    vp = ctypes.c_void_p
    lib.gdn_fwd_launch.restype = ctypes.c_int
    lib.gdn_fwd_launch.argtypes = [vp] * 11 + [ctypes.c_int] * 6 + [ctypes.c_float, vp, vp]
    _lib = lib
    return lib


def gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    lib = _load()
    T, HK, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    assert K == 128 and V == 128, "kernel is specialised for head dim 128"
    assert HV % HK == 0 and HK <= 255 and HV <= 255
    N = cu_seqlens.numel() - 1
    # the kernel reads contiguous [T, H, 128] / [T, HV] tensors (no-ops for already-contiguous inputs)
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); a = a.contiguous(); b = b.contiguous()
    if q.dtype != torch.bfloat16: q = q.to(torch.bfloat16)
    if k.dtype != torch.bfloat16: k = k.to(torch.bfloat16)
    if v.dtype != torch.bfloat16: v = v.to(torch.bfloat16)
    if a.dtype != torch.bfloat16: a = a.to(torch.bfloat16)
    if b.dtype != torch.bfloat16: b = b.to(torch.bfloat16)
    state = state.contiguous()
    if state.dtype != torch.float32: state = state.float()
    A_log = A_log.contiguous().float()
    dt_bias = dt_bias.contiguous().float()
    cu = cu_seqlens
    if cu.dtype != torch.int64 or not cu.is_contiguous():
        cu = cu.to(torch.int64).contiguous()
    assert T < 2**31 and state.numel() < 2**31
    out = torch.empty((T, HV, V), dtype=torch.bfloat16, device=q.device)
    if T == 0:
        return out, state.clone()
    new_state = torch.empty_like(state)
    if N > 0:
        rep = HV // HK
        rep_shift = rep.bit_length() - 1 if (rep & (rep - 1)) == 0 else 255
        flags = 1 if (HV == 8 and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0) else 0
        stream = torch.cuda.current_stream(q.device).cuda_stream
        rc = lib.gdn_fwd_launch(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr(), A_log.data_ptr(), a.data_ptr(),
            dt_bias.data_ptr(), b.data_ptr(), cu.data_ptr(), out.data_ptr(), new_state.data_ptr(),
            N, HK, HV, T, rep_shift, flags, float(scale), stream, _dbg_ptr(q.device))
        if rc != 0:
            raise RuntimeError(f"gdn_fwd_launch failed with CUDA error {rc}")
    return out, new_state
