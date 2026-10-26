"""Entry point for gdn_prefill_qk4_v8_d128_k_last (destination-passing style).

Pipeline (2 CUDA/tcgen05 kernels, NVRTC-compiled at first use):
  1. gdn_intra_kernel : per chunk x head, everything independent of the start state (W, U, M, B); CTAs in the order the
                        tail needs the chunks; publishes a per-(chunk, head) ready flag
  2. gdn_tail_kernel  : persistent (one CTA per SM) work queue: per-(sequence, head) serial scans of the chunk affine
                        maps (S^T resident in TMEM) first, then per-(chunk, head) output items
                        O = (gamma q) S_c + P (U - W S_c) that wait on per-chunk ready flags. Launched with programmatic
                        dependent launch so it overlaps the intra kernel's partial last wave.
"""
import math

import torch

from .intra_cuda import intra_launch, CHUNK
from .tail_cuda import tail_launch

HEAD = 128
NUM_V_HEADS = 8
TILE = HEAD * HEAD


_WS = {}


def _workspace(T, N, dev):
    """Persistent intermediates (avoid per-call allocation)."""
    key = (T, N, str(dev))
    ws = _WS.get(key)
    if ws is None:
        max_chunks = T // CHUNK + N  # upper bound without a host sync
        ws = dict(
            max_chunks=max_chunks,
            W=torch.empty((T, NUM_V_HEADS, HEAD), dtype=torch.bfloat16, device=dev),
            U=torch.empty((T, NUM_V_HEADS, HEAD), dtype=torch.bfloat16, device=dev),
            m_tiles=torch.empty((max_chunks, NUM_V_HEADS, TILE), dtype=torch.bfloat16, device=dev),
            b_tiles=torch.empty((max_chunks, NUM_V_HEADS, TILE), dtype=torch.bfloat16, device=dev),
            state_chunks=torch.empty((max_chunks, NUM_V_HEADS, TILE), dtype=torch.bfloat16, device=dev),
            # [chunk-state ready flags | intra ready flags | task counter]; zero at allocation, never reset (epoch-valued flags)
            sync=torch.zeros((2 * max_chunks * NUM_V_HEADS + 32,), dtype=torch.int32, device=dev),
            epoch=0,
        )
        _WS[key] = ws
    return ws


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scale = float(scale)
    T = q.shape[0]
    N = state.shape[0]
    dev = q.device
    ws = _workspace(T, N, dev)
    max_chunks = ws["max_chunks"]
    sync = ws["sync"]
    # no per-call reset: ready flags carry the launch epoch, the tail kernel resets its own task counter at exit
    ws["epoch"] = (ws["epoch"] % 0xFFFFFFFF) + 1
    nf = max_chunks * NUM_V_HEADS
    flags, ready, counter = sync[:nf], sync[nf:2 * nf], sync[2 * nf:]
    intra_launch(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ws["W"], ws["U"], ws["m_tiles"], ws["b_tiles"], max_chunks,
                 ready=ready, epoch=ws["epoch"])
    # launched with programmatic dependent launch: starts on the SMs the intra kernel's last wave leaves idle; every read of
    # an intra result is guarded by that chunk's ready flag
    tail_launch(q, k, a, A_log, dt_bias, cu_seqlens, state, ws["m_tiles"], ws["b_tiles"], ws["state_chunks"], ws["W"], ws["U"],
                output, new_state, scale, flags, counter, epoch=ws["epoch"], ready=ready, pdl=True)
    return output, new_state
