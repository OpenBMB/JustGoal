"""MoE FP8 block-scale (DeepSeek-V3 routing) for B200 — self-contained CUDA solution.

The CUDA source (kernel.cu, shipped with this solution) is compiled with nvcc for sm_100a on
first use and loaded through ctypes. All computation runs in our own kernels.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import torch

HIDDEN = 7168

_LIB = None
_WS = {}

_ARCH_FLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]


def _find_nvcc():
    cands = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    try:
        from torch.utils import cpp_extension as _ce

        if _ce.CUDA_HOME:
            cands.append(os.path.join(_ce.CUDA_HOME, "bin", "nvcc"))
    except Exception:
        pass
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    cands += ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError("nvcc not found")


def _build():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "kernel.cu")
    with open(src, "rb") as f:
        code = f.read()
    extra = os.environ.get("MOE_NVCC_EXTRA", "").split()  # dev-only experiment switches
    key = hashlib.sha256(code + " ".join(_ARCH_FLAGS + extra).encode()).hexdigest()[:16]
    build_dir = os.path.join(tempfile.gettempdir(), f"fib_moe_build_{os.getuid()}", key)
    so = os.path.join(build_dir, "libmoe.so")
    if not os.path.isfile(so):
        os.makedirs(build_dir, exist_ok=True)
        tmp = os.path.join(build_dir, f"libmoe.{os.getpid()}.tmp.so")
        cmd = [_find_nvcc(), "-O3", "-std=c++17", "-lineinfo", *_ARCH_FLAGS, *extra, "-shared",
               "-Xcompiler", "-fPIC", "-o", tmp, src]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.moe_run.restype = ctypes.c_int
    lib.moe_run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.moe_workspace_bytes.restype = ctypes.c_int
    lib.moe_workspace_bytes.argtypes = [ctypes.c_int]
    return lib


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _build()
    return _LIB


def _workspace(T, device):
    key = (T, device)
    ws = _WS.get(key)
    if ws is None:
        ws = torch.zeros(_lib().moe_workspace_bytes(T), dtype=torch.uint8, device=device)  # kernel keeps it zeroed
        _WS[key] = ws
    return ws


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset,
    routed_scaling_factor,
):
    lib = _lib()
    T = routing_logits.shape[0]
    dev = hidden_states.device
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())

    # Inputs are expected contiguous in the documented dtypes; coerce only if needed.
    if routing_logits.dtype != torch.float32 or not routing_logits.is_contiguous():
        routing_logits = routing_logits.contiguous().to(torch.float32)
    if routing_bias.dtype != torch.bfloat16 or not routing_bias.is_contiguous():
        routing_bias = routing_bias.contiguous().to(torch.bfloat16)
    for name, tsr in (("hidden_states", hidden_states), ("gemm1_weights", gemm1_weights),
                      ("gemm2_weights", gemm2_weights)):
        assert tsr.dtype == torch.float8_e4m3fn and tsr.is_contiguous(), name
    if not hidden_states_scale.is_contiguous():
        hidden_states_scale = hidden_states_scale.contiguous()
    if not gemm1_weights_scale.is_contiguous():
        gemm1_weights_scale = gemm1_weights_scale.contiguous()
    if not gemm2_weights_scale.is_contiguous():
        gemm2_weights_scale = gemm2_weights_scale.contiguous()

    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    ws = _workspace(T, dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    rc = lib.moe_run(
        routing_logits.data_ptr(), routing_bias.data_ptr(), hidden_states.data_ptr(),
        hidden_states_scale.data_ptr(), gemm1_weights.data_ptr(), gemm1_weights_scale.data_ptr(),
        gemm2_weights.data_ptr(), gemm2_weights_scale.data_ptr(), int(local_expert_offset),
        float(routed_scaling_factor), out.data_ptr(), ws.data_ptr(), int(T), stream,
    )
    if rc != 0:
        raise RuntimeError(f"moe_run failed with CUDA error {rc}")
    return out
