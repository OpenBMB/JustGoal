"""FIB entry point for gdn_prefill_qk4_v8_d128_k_last."""
from . import gdn_cuda


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    return gdn_cuda.dispatch(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
