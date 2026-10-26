"""GDN prefill (qk4_v8_d128, k-last state) for NVIDIA B200.

Entry point (destination-passing style):
    run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)

The CUDA kernels (gdn_kernels.cu / gdn_ptx.cuh, raw PTX tcgen05 + TMA, sm_100a) are
compiled once at import time with torch.utils.cpp_extension.
"""
import hashlib
import math
import os
import tempfile

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCES = [os.path.join(_HERE, "gdn_kernels.cu")]
_HEADERS = [os.path.join(_HERE, "gdn_ptx.cuh")]


def _build():
    h = hashlib.sha1()
    for f in _SOURCES + _HEADERS:
        with open(f, "rb") as fh:
            h.update(fh.read())
    tag = h.hexdigest()[:10]
    build_dir = os.path.join(tempfile.gettempdir(), f"gdn_prefill_build_{tag}")
    os.makedirs(build_dir, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    return load(
        name=f"gdn_prefill_{tag}",
        sources=_SOURCES,
        extra_include_paths=[_HERE],
        extra_cuda_cflags=["-gencode", "arch=compute_100a,code=sm_100a", "-O3", "-std=c++17", "--use_fast_math"],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=build_dir,
        verbose=False,
    )


_ext = _build()


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if state is not None and state.dtype != torch.float32:
        state = state.float()
    _ext.gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), output, new_state,
                     int(os.environ.get("GDN_DBG", "0")))
    return output, new_state
