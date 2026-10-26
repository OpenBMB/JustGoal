"""Entry points for the FlashInfer-Bench attention backward definitions (B200, CuTe DSL).

  run(q, k, v, out, dout, lse, sm_scale, dq, dk, dv)                 # bshd, MHA / GQA (DPS)
  run_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale) -> (dq, dk, dv)  # thd

The backward kernel lives in fmha_bwd_sm100.py (warp-specialized tcgen05/TMEM/TMA kernel
derived from the BSD-licensed CUTLASS CuTe DSL SM100 FMHA backward example).
"""

import os

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float32, BFloat16

from .fmha_bwd_sm100 import BlackwellFusedMultiHeadAttentionBackward
from .fmha_mask import MaskEnum

MMA_TILER_MN = (128, 128)

_executors = {}
_sm_count = {}
_dbg = {}


def debug_buffer():
    """Pinned host buffer of 64 int32 progress markers (FMHA_BWD_DBGBUF=1), readable from
    the host while the kernel runs."""
    if os.environ.get("FMHA_BWD_DBGBUF") != "1":
        return None
    if "t" not in _dbg:
        _dbg["t"] = torch.zeros(64, dtype=torch.int32).pin_memory()
    return _dbg["t"]
_workspaces = {}
_perms = {}


def _varlen_item_perm(cu_seqlens, h_k, mode, block_k=128):
    """Host-side order of the valid varlen work items (kv block, kv head), cached per
    cu_seqlens. Item id = lin_blk + n_blk_total * head as decoded by the kernel; an item's
    cost is its causal trip count (kv blocks of its sequence after and including it).
    Modes: "lpt" = global heaviest-first (all heads interleaved); "headsort" = head-major with
    the sequences of a head ordered longest-first (blocks heaviest-first inside a sequence);
    "group<G>" = G heads at a time, heaviest-first inside the group."""
    key = (tuple(cu_seqlens.tolist()), h_k, mode)
    perm = _perms.get(key)
    if perm is None:
        cu = key[0]
        num_seqs = len(cu) - 1
        total = cu[-1]
        n_blk_total = (total + block_k - 1) // block_k + num_seqs
        seqs = []  # (nb, base)
        base = 0
        for si in range(num_seqs):
            nb = (cu[si + 1] - cu[si] + block_k - 1) // block_k
            seqs.append((nb, base))
            base += nb
        items = []
        if mode == "lpt" or mode.startswith("group"):
            g = h_k if mode == "lpt" else int(mode[5:])
            for h0 in range(0, h_k, g):
                grp = []
                for nb, b0 in seqs:
                    for j in range(nb):
                        for head in range(h0, min(h0 + g, h_k)):
                            grp.append((-(nb - j), b0 + j + n_blk_total * head))
                grp.sort()
                items.extend(i for _, i in grp)
        elif mode == "headsort":
            order = sorted(range(num_seqs), key=lambda si: -seqs[si][0])
            for head in range(h_k):
                for si in order:
                    nb, b0 = seqs[si]
                    items.extend(b0 + j + n_blk_total * head for j in range(nb))
        else:
            raise ValueError(mode)
        perm = torch.tensor(items, dtype=torch.int32, device=cu_seqlens.device)
        _perms[key] = perm
    return perm


def _as_q_view(t, b, s, h_k, h_r, d):
    # (b, s, h, d) contiguous -> (s, d, h_r, h_k, b) strided view, no copy
    return t.view(b, s, h_k, h_r, d).permute(1, 4, 3, 2, 0)


def _as_kv_view(t, b, s, h_k, d):
    # (b, s, h_k, d) contiguous -> (s, d, 1, h_k, b)
    return t.view(b, s, h_k, 1, d).permute(1, 4, 3, 2, 0)


def _cute_5d(t5):
    ct = from_dlpack(t5, assumed_align=16)
    return ct.mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(
        mode=1, stride_order=(4, 0, 3, 2, 1), divisibility=64
    )


def _cute_lse(lse4):
    return from_dlpack(lse4, assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _get_workspace(key, rows, d, h_q, b, device):
    # rows: packed row space (seq_len per batch entry for dense, total tokens with b=1 for
    # varlen).
    ws = _workspaces.get(key)
    if ws is None:
        nbytes = BlackwellFusedMultiHeadAttentionBackward._get_workspace_size(
            rows, rows, d, h_q, b, Float32
        )
        # zero-initialised: with FMHA_BWD_ACC_ZERO=main (default) every call leaves the dQ
        # accumulator region zeroed again (the main kernel re-zeroes each q tile after it has
        # converted it), so the preprocess never has to zero it
        ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        _workspaces[key] = ws
    return ws


def _get_kernel(varlen):
    return BlackwellFusedMultiHeadAttentionBackward(
        BFloat16, Float32, (MMA_TILER_MN[0], MMA_TILER_MN[1], 128), varlen, MaskEnum.WINDOW_MASK_BWD
    )


def _lpt_group(varlen, h_r, h_k, b):
    # Number of (kv head, batch) pairs whose kv blocks are dispatched heaviest-first as a
    # group. Measured (dev harness, cold L2): MHA best at 8-16 (1.528 ms vs 1.541 at 1),
    # GQA 8:1 best at 4 (1.455 ms vs 1.677 at 1), varlen best at 1 (grouping costs L2
    # locality: 1.945 ms at 8 vs 1.895 at 1).
    env = os.environ.get("FMHA_BWD_LPT_G")
    if env:
        return int(env)
    if varlen:
        return 1
    return 4 if h_r > 1 else 16


def _num_sms():
    n = _sm_count.get("n")
    if n is None:
        n = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        env = os.environ.get("FMHA_BWD_MAX_CTAS")
        if env:
            n = min(n, int(env))
        _sm_count["n"] = n
    return n


def _launch(varlen, problem_shape, Q, K, V, O, dQ, dK, dV, dO, LSE, cu_q, cu_k, scale, ws, lpt_group,
            perm=None):
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    buf = debug_buffer()
    dbg = None
    if buf is not None:
        dbg = cute.runtime.make_ptr(Int32, buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    perm_t = None
    if perm is not None:
        perm_t = from_dlpack(perm, assumed_align=16).mark_layout_dynamic()
    args = (
        problem_shape, Q, K, V, O, dQ, dK, dV, dO, LSE, cu_q, cu_k,
        Float32(scale), None, Int32(0), Int32(lpt_group), Int32(_num_sms()), ws, dbg, perm_t, stream,
    )
    ex = _executors.get(varlen)
    if ex is None:
        ex = cute.compile(_get_kernel(varlen), *args)
        _executors[varlen] = ex
    ex(*args)


def run(q, k, v, out, dout, lse, sm_scale, dq, dk, dv):
    b, s, h, d = q.shape
    h_k = k.shape[2]
    h_r = h // h_k
    problem_shape = (s, s, d, ((h_r, h_k), b))

    Q = _cute_5d(_as_q_view(q, b, s, h_k, h_r, d))
    O = _cute_5d(_as_q_view(out, b, s, h_k, h_r, d))
    dO = _cute_5d(_as_q_view(dout, b, s, h_k, h_r, d))
    dQ = _cute_5d(_as_q_view(dq, b, s, h_k, h_r, d))
    K = _cute_5d(_as_kv_view(k, b, s, h_k, d))
    V = _cute_5d(_as_kv_view(v, b, s, h_k, d))
    dK = _cute_5d(_as_kv_view(dk, b, s, h_k, d))
    dV = _cute_5d(_as_kv_view(dv, b, s, h_k, d))
    LSE = _cute_lse(lse.view(b, h_k, h_r, s).permute(3, 2, 1, 0))

    buf = _get_workspace(("dense", b, s, h, d), s, d, h, b, q.device)
    ws = from_dlpack(buf, assumed_align=16).mark_layout_dynamic()

    _launch(False, problem_shape, Q, K, V, O, dQ, dK, dV, dO, LSE, None, None, float(sm_scale), ws,
            _lpt_group(False, h_r, h_k, b))


def run_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    # Value-returning style: the evaluator cannot size the outputs of this definition from
    # the input shapes (num_seqs is not an input dimension), so DPS is not possible here.
    t, h, d = q.shape
    h_k = k.shape[1]
    h_r = h // h_k
    num_seqs = cu_seqlens.shape[0] - 1
    max_seqlen = int(max_seqlen)
    problem_shape = (max_seqlen, max_seqlen, d, ((h_r, h_k), num_seqs))
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    Q = _cute_5d(_as_q_view(q, 1, t, h_k, h_r, d))
    O = _cute_5d(_as_q_view(out, 1, t, h_k, h_r, d))
    dO = _cute_5d(_as_q_view(dout, 1, t, h_k, h_r, d))
    dQ = _cute_5d(_as_q_view(dq, 1, t, h_k, h_r, d))
    K = _cute_5d(_as_kv_view(k, 1, t, h_k, d))
    V = _cute_5d(_as_kv_view(v, 1, t, h_k, d))
    dK = _cute_5d(_as_kv_view(dk, 1, t, h_k, d))
    dV = _cute_5d(_as_kv_view(dv, 1, t, h_k, d))
    LSE = _cute_lse(lse.view(h_k, h_r, t).permute(2, 1, 0).unsqueeze(-1))
    cu = from_dlpack(cu_seqlens, assumed_align=16).mark_layout_dynamic()

    buf = _get_workspace(("varlen", t, h, d), t, d, h, 1, q.device)
    ws = from_dlpack(buf, assumed_align=16).mark_layout_dynamic()

    perm = None
    mode = os.environ.get("FMHA_BWD_VARLEN_ORDER", "plain")
    if mode != "plain":
        perm = _varlen_item_perm(cu_seqlens, h_k, mode)
    _launch(True, problem_shape, Q, K, V, O, dQ, dK, dV, dO, LSE, cu, cu, float(sm_scale), ws,
            _lpt_group(True, h_r, h_k, num_seqs), perm)
    return dq, dk, dv
