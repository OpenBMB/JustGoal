"""Fused GDN prefill (qk4_v8_d128, k-last) for B200.

Python entry point (destination-passing style). The CUDA kernel lives in gdn_prefill.cu next to this
file and is compiled on first use with torch.utils.cpp_extension.load_inline so that we control the
compile flags (-arch=sm_100a). The whole operator is a single kernel launch; cu_seqlens is read on the
device, so there is no host/device synchronisation in the hot path.
"""
import os
import torch

_MODULE = None

_CPP_DECL = """
#include <torch/extension.h>
void gdn_prefill_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
                     torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
                     torch::Tensor cu_seqlens, double scale, torch::Tensor output,
                     torch::Tensor new_state);
"""


def _build():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load_inline

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "gdn_prefill.cu"), "r", encoding="utf-8") as f:
        cuda_src = f.read()
    extra = os.environ.get("GDN_EXTRA_FLAGS", "").split()
    _MODULE = load_inline(
        name="gdn_prefill_fused_v81" + "".join(extra).replace("-D", "_").replace("=", ""),
        cpp_sources=[_CPP_DECL],
        cuda_sources=[cuda_src],
        functions=["gdn_prefill_fwd"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-Xptxas", "-v",
        ] + extra,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("GDN_VERBOSE_BUILD", "0") == "1",
    )
    return _MODULE


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    mod = _build()
    if scale is None or scale == 0.0:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    mod.gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), output,
                        new_state)
