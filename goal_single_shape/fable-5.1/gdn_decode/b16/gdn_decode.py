"""GDN decode (gated delta rule, single token) for Qwen3-Next spec gdn_decode_qk4_v8_d128_k_last, B200-tuned.

Usage:
    from gdn_decode import gdn_decode_qk4_v8_d128_k_last
    out, new_state = gdn_decode_qk4_v8_d128_k_last(q, k, v, state, A_log, a, dt_bias, b, scale)
`state` is updated in place (same semantics as FlashInfer's gated_delta_rule_decode_pretranspose with fp32 state).
"""
import math, os, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None

def _load():
    global _ext
    if _ext is None:
        bd = os.path.join(_HERE, "build"); os.makedirs(bd, exist_ok=True)
        _ext = load(name="gdn_decode_qk4_v8_d128", sources=[os.path.join(_HERE, "gdn_decode_kernel.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17"],
                    build_directory=bd, verbose=False)
    return _ext

def gdn_decode_qk4_v8_d128_k_last(q, k, v, state, A_log, a, dt_bias, b, scale=None, output=None, prefetch_distance=0, prefetch_small_operands=False):
    """prefetch_distance > 0 (e.g. 640 for B=16) and prefetch_small_operands=True help only when inputs are L2-cold."""
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if output is None:
        output = torch.empty_like(v)
    _load().gdn_decode(q, k, v, state, A_log, a, dt_bias, b, output, float(scale), int(prefetch_distance), 1 if prefetch_small_operands else 0)
    return output, state
