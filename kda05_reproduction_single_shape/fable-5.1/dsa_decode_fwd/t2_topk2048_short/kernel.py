"""DSA sparse attention (h16 / ckv512 / kpe64 / topk2048 / ps64) for NVIDIA B200.

Python-language solution: the CUDA kernel in ``kernel.cu`` (same directory) is JIT-compiled
with ``torch.utils.cpp_extension.load`` so we control the compile flags (``-arch=sm_100a``).
Destination-passing style: ``run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
output, lse)``.
"""

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_CU = _HERE / "kernel.cu"

# torch appends its own -gencode flags from TORCH_CUDA_ARCH_LIST (or the detected device, which
# would be plain sm_100).  Pin it to 10.0a so the kernel is compiled once, for sm_100a only.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-Xptxas", "-v",
    "-DNDEBUG",
    "-lineinfo",
] + (["-DDSA_TIMING"] if os.environ.get("DSA_TIMING") else []) + os.environ.get("DSA_EXTRA_FLAGS", "").split()


def _build():
    src = _CU.read_bytes()
    digest = hashlib.sha1(src + " ".join(_EXTRA_CUDA_CFLAGS).encode()).hexdigest()[:12]
    name = f"dsa_sparse_attn_{digest}"
    return load(
        name=name,
        sources=[str(_CU)],
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(os.environ.get("DSA_BUILD_VERBOSE")),
    )


_mod = _build()
_kernel = _mod.dsa_sparse_attention
read_timing = _mod.read_timing


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    _kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), output, lse)
