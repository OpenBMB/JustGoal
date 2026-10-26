"""
Blackwell SM100 attention backward wrapper (CuTe DSL / tcgen05 + TMEM + TMA + warp
specialization), built on NVIDIA's CUTLASS FMHA backward example
(BlackwellFusedMultiHeadAttentionBackward, BSD-3-Clause, vendored as fmha_bwd.py).

Two entry points matching the task:
  run_bshd(q,k,v,out,dout,lse,sm_scale)                          -> (dq,dk,dv)  [B,N,H,D]
  run_varlen(q,k,v,out,dout,lse,cu_seqlens,max_seqlen,sm_scale)  -> (dq,dk,dv)  [T,H,D]

Head layout: query head h uses kv head h // (Hq//Hkv). The CUTLASS kernel splits the
H axis as (h_k outer, h_r inner) with h_r the fast index, which is exactly this
convention, so a plain view [.,H,.] -> [.,h_k,h_r,.] reproduces the expected GQA map.
"""
import os
import sys

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float32, BFloat16

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fmha_bwd import BlackwellFusedMultiHeadAttentionBackward  # noqa: E402
from helpers import fmha_helpers as fmha_utils  # noqa: E402

MMA_TILER = (128, 128, 128)
ACC = Float32
ELEM = BFloat16
_D = 128

_compiled = {}      # key -> compiled kernel
_workspace = {}     # key -> uint8 cuda buffer


def _qo_tensor(t5):
    ct = from_dlpack(t5, assumed_align=16)
    ct.element_type = ELEM
    return ct.mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(
        mode=1, stride_order=(4, 0, 3, 2, 1), divisibility=64
    )


def _lse_tensor(t4):
    ct = from_dlpack(t4, assumed_align=16)
    ct.element_type = ACC
    return ct.mark_layout_dynamic(leading_dim=0)


def _get_ws(key, nbytes, device):
    buf = _workspace.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        _workspace[key] = buf
    return buf


def _run(q, k, v, out, dout, lse, sm_scale, varlen, cu_seqlens, max_seqlen):
    Hq = q.shape[-2]
    if varlen:
        T = q.shape[0]
        Hkv = k.shape[1]
        h_r = Hq // Hkv
        orig_b = cu_seqlens.numel() - 1
        s_q_max = int(max_seqlen)
        s_k_max = s_q_max
        b = 1
        s = T
    else:
        B, N, _, _ = q.shape
        Hkv = k.shape[2]
        h_r = Hq // Hkv
        orig_b = B
        s_q_max = N
        s_k_max = N
        b = B
        s = N

    h_k = Hkv
    # The bwd kernel reduce-adds (TMA .add.bf16) each k-block's dQ contribution
    # straight into this bf16 output (no fp32 scratch / convert kernel). dq is zeroed
    # on-GPU inside the sum_OdO pre-pass (fused), so no torch memset is needed here.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # views -> (s, d, h_r, h_k, b)
    def q5(t):
        return t.view(b, s, h_k, h_r, _D).permute(1, 4, 3, 2, 0)

    def k5(t):
        return t.view(b, s, h_k, 1, _D).permute(1, 4, 3, 2, 0)

    q_t = _qo_tensor(q5(q))
    o_t = _qo_tensor(q5(out))
    do_t = _qo_tensor(q5(dout))
    dq_t = _qo_tensor(q5(dq))
    k_t = _qo_tensor(k5(k))
    v_t = _qo_tensor(k5(v))
    dk_t = _qo_tensor(k5(dk))
    dv_t = _qo_tensor(k5(dv))
    lse_t = _lse_tensor(lse.view(b, h_k, h_r, s).permute(3, 2, 1, 0))

    if varlen:
        cu = cu_seqlens.to(torch.int32)
        cu_q = from_dlpack(cu).mark_layout_dynamic()
        cu_k = from_dlpack(cu).mark_layout_dynamic()
    else:
        cu_q = None
        cu_k = None

    problem_shape = (s_q_max, s_k_max, _D, ((h_r, h_k), orig_b))

    # Varlen: pack the fp32 workspace (sum_OdO / scaled_LSE / dQ_acc) to total_tokens
    # instead of padding each of the orig_b sequences to s_q_max. This cuts the zero /
    # sum_OdO / convert cost by (b * s_q_max / total_tokens).
    total_tokens = int(T) if varlen else None
    acc_bytes = ACC.width // 8
    if varlen:
        Hq_ = Hq
        ws_bytes = total_tokens * Hq_ * acc_bytes * 2  # sum_OdO + scaled_LSE
        ws_bytes += total_tokens * Hq_ * _D * acc_bytes  # dQ_acc
    else:
        ws_bytes = BlackwellFusedMultiHeadAttentionBackward._get_workspace_size(
            s_q_max, s_k_max, _D, Hq, orig_b, ACC
        )
    ws_key = (varlen, s_q_max, s_k_max, h_r, h_k, orig_b, total_tokens)
    # Workspace now holds only sum_OdO + scaled_LSE (both fully overwritten each call);
    # the dQ_acc slice is unused in the direct-reduce path but kept in the layout.
    ws_buf = _get_ws(ws_key, ws_bytes, q.device)
    workspace = from_dlpack(ws_buf, assumed_align=16).mark_layout_dynamic()

    stream = cutlass.torch.default_stream()

    key = ws_key
    comp = _compiled.get(key)
    if comp is None:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_BWD
        fmha = BlackwellFusedMultiHeadAttentionBackward(
            ELEM, ACC, MMA_TILER, varlen, mask_type
        )
        fmha.total_tokens = total_tokens
        # Heavy-first CTA dispatch (causal load balance). The bwd grid is
        # (nkblk, H_K, B) CTAs; under causal masking kblock-0 is ~nkblk x heavier
        # than the last block, so when there are few waves the final wave's tail is
        # dominated by a heavy CTA. Reordering to dispatch heavy blocks first cuts
        # that tail — a large win for GQA (few, 8x-heavier CTAs) — but it scatters the
        # per-CTA K/V/Q loads across heads/batch and costs L2 locality, which hurts the
        # many-wave dense-MHA case. Enable only when the wave count is low.
        _NUM_SM_B200 = 148
        nkblk = (s_k_max + MMA_TILER[1] - 1) // MMA_TILER[1]
        est_waves = nkblk * h_k * orig_b / _NUM_SM_B200
        fmha.reorder_grid = (not varlen) and (est_waves <= 8.0)
        comp = cute.compile(
            fmha, problem_shape, q_t, k_t, v_t, o_t, dq_t, dk_t, dv_t, do_t, lse_t,
            cu_q, cu_k, float(sm_scale), None, Int32(0), workspace, stream,
        )
        _compiled[key] = comp

    comp(
        problem_shape, q_t, k_t, v_t, o_t, dq_t, dk_t, dv_t, do_t, lse_t,
        cu_q, cu_k, float(sm_scale), None, Int32(0), workspace, stream,
    )
    return dq, dk, dv


def run_bshd(q, k, v, out, dout, lse, sm_scale):
    return _run(q, k, v, out, dout, lse, sm_scale, False, None, None)


def run_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    return _run(q, k, v, out, dout, lse, sm_scale, True, cu_seqlens, max_seqlen)
