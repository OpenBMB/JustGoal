import torch
from .gdn_kernel import gdn_run


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Gated Delta Net prefill (k-last). Custom fused Triton chunked kernel."""
    return gdn_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
