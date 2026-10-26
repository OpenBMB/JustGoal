import torch
import triton
import triton.language as tl

# FlashAttention backward (causal, bf16) - from scratch.
# exp2-based softmax; masked diagonal prologue + unmasked body for causal.

LOG2E = 1.4426950408889634
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _bwd_preprocess(
    O, DO, Delta,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_db, stride_dh, stride_dn,
    N, H,
    BLOCK_M: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N
    base = b * stride_ob + h * stride_oh
    o_ptr = O + base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    do_ptr = DO + base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o = tl.load(o_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + b * stride_db + h * stride_dh + offs_m * stride_dn, delta, mask=mask_m)


@triton.jit
def _dkdv_inner(
    dk, dv, k, v, Q, DO, LSE, Delta,
    scale2, base_q, stride_qn, stride_lh_base, stride_ln,
    offs_n, offs_d, mask_n, N,
    lo, hi,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, MASKED: tl.constexpr,
    BOUND: tl.constexpr,
):
    offs_m0 = tl.arange(0, BLOCK_M)
    for m in range(lo, hi, BLOCK_M):
        offs_m = m + offs_m0
        if BOUND:
            mask_m = offs_m < N
            q = tl.load(Q + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], mask=mask_m[:, None], other=0.0)
            do = tl.load(DO + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], mask=mask_m[:, None], other=0.0)
            lse = tl.load(LSE + stride_lh_base + offs_m * stride_ln, mask=mask_m, other=0.0)
            delta = tl.load(Delta + stride_lh_base + offs_m * stride_ln, mask=mask_m, other=0.0)
        else:
            q = tl.load(Q + base_q + offs_m[:, None] * stride_qn + offs_d[None, :])
            do = tl.load(DO + base_q + offs_m[:, None] * stride_qn + offs_d[None, :])
            lse = tl.load(LSE + stride_lh_base + offs_m * stride_ln)
            delta = tl.load(Delta + stride_lh_base + offs_m * stride_ln)
        # sT[BN,BM] = k . q^T
        sT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(sT * scale2 - lse[None, :] * _LOG2E)  # [BN,BM]
        if MASKED:
            causal = offs_n[:, None] <= offs_m[None, :]
            if BOUND:
                causal = causal & mask_m[None, :] & mask_n[:, None]
            pT = tl.where(causal, pT, 0.0)
        elif BOUND:
            pT = tl.where(mask_m[None, :], pT, 0.0)
        # dv[BN,D] += pT . do
        dv += tl.dot(pT.to(do.dtype), do)
        # dpT[BN,BM] = v . do^T
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    return dk, dv


@triton.jit
def _bwd_dkdv(
    Q, K, V, DO, LSE, Delta, DK, DV,
    stride_qb, stride_qn, stride_qh,
    stride_kb, stride_kn, stride_kh,
    stride_lb, stride_lh, stride_ln,
    stride_dkb, stride_dkn, stride_dkh,
    scale, scale2,
    N, Hq, Hkv, GROUP,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    NEED_BOUND: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // Hq
    hq = pid_bh % Hq
    hkv = hq // GROUP
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < N

    kbase = b * stride_kb + hkv * stride_kh
    if NEED_BOUND:
        k = tl.load(K + kbase + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        v = tl.load(V + kbase + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None], other=0.0)
    else:
        k = tl.load(K + kbase + offs_n[:, None] * stride_kn + offs_d[None, :])
        v = tl.load(V + kbase + offs_n[:, None] * stride_kn + offs_d[None, :])
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    m_start = (start_n // BLOCK_M) * BLOCK_M
    m_mask_end = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    base_q = b * stride_qb + hq * stride_qh
    lbase = b * stride_lb + hq * stride_lh
    dk, dv = _dkdv_inner(dk, dv, k, v, Q, DO, LSE, Delta, scale2, base_q, stride_qn, lbase, stride_ln,
                         offs_n, offs_d, mask_n, N, m_start, m_mask_end,
                         BLOCK_M, BLOCK_N, D, True, NEED_BOUND)
    dk, dv = _dkdv_inner(dk, dv, k, v, Q, DO, LSE, Delta, scale2, base_q, stride_qn, lbase, stride_ln,
                         offs_n, offs_d, mask_n, N, m_mask_end, N,
                         BLOCK_M, BLOCK_N, D, False, NEED_BOUND)
    dk = dk * scale
    dkb = b * stride_dkb + hq * stride_dkh
    if NEED_BOUND:
        tl.store(DK + dkb + offs_n[:, None] * stride_dkn + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(DV + dkb + offs_n[:, None] * stride_dkn + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=mask_n[:, None])
    else:
        tl.store(DK + dkb + offs_n[:, None] * stride_dkn + offs_d[None, :], dk.to(DK.dtype.element_ty))
        tl.store(DV + dkb + offs_n[:, None] * stride_dkn + offs_d[None, :], dv.to(DV.dtype.element_ty))


@triton.jit
def _dq_inner(
    dq, q, do, lse, delta, K, V,
    scale2, base_kv, stride_kn, offs_m, offs_d, mask_m, N,
    lo, hi,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, MASKED: tl.constexpr,
    BOUND: tl.constexpr,
):
    offs_n0 = tl.arange(0, BLOCK_N)
    for n in range(lo, hi, BLOCK_N):
        offs_n = n + offs_n0
        if BOUND:
            mask_n = offs_n < N
            k = tl.load(K + base_kv + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None], other=0.0)
            v = tl.load(V + base_kv + offs_n[:, None] * stride_kn + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(K + base_kv + offs_n[:, None] * stride_kn + offs_d[None, :])
            v = tl.load(V + base_kv + offs_n[:, None] * stride_kn + offs_d[None, :])
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * scale2 - lse[:, None] * _LOG2E)  # [BM,BN]
        if MASKED:
            causal = offs_m[:, None] >= offs_n[None, :]
            if BOUND:
                causal = causal & mask_n[None, :]
            p = tl.where(causal, p, 0.0)
        elif BOUND:
            p = tl.where(mask_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.jit
def _bwd_dq(
    Q, K, V, DO, LSE, Delta, DQ,
    stride_qb, stride_qn, stride_qh,
    stride_kb, stride_kn, stride_kh,
    stride_lb, stride_lh, stride_ln,
    scale, scale2,
    N, Hq, Hkv, GROUP,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    NEED_BOUND: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // Hq
    hq = pid_bh % Hq
    hkv = hq // GROUP
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N

    base_q = b * stride_qb + hq * stride_qh
    if NEED_BOUND:
        q = tl.load(Q + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], mask=mask_m[:, None], other=0.0)
        lbase = b * stride_lb + hq * stride_lh
        lse = tl.load(LSE + lbase + offs_m * stride_ln, mask=mask_m, other=0.0)
        delta = tl.load(Delta + lbase + offs_m * stride_ln, mask=mask_m, other=0.0)
    else:
        q = tl.load(Q + base_q + offs_m[:, None] * stride_qn + offs_d[None, :])
        do = tl.load(DO + base_q + offs_m[:, None] * stride_qn + offs_d[None, :])
        lbase = b * stride_lb + hq * stride_lh
        lse = tl.load(LSE + lbase + offs_m * stride_ln)
        delta = tl.load(Delta + lbase + offs_m * stride_ln)

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    base_kv = b * stride_kb + hkv * stride_kh
    n_body_end = (start_m // BLOCK_N) * BLOCK_N          # fully-valid kv blocks [0, n_body_end)
    n_mask_end = ((start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    dq = _dq_inner(dq, q, do, lse, delta, K, V, scale2, base_kv, stride_kn, offs_m, offs_d, mask_m, N,
                   0, n_body_end, BLOCK_M, BLOCK_N, D, False, NEED_BOUND)
    dq = _dq_inner(dq, q, do, lse, delta, K, V, scale2, base_kv, stride_kn, offs_m, offs_d, mask_m, N,
                   n_body_end, n_mask_end, BLOCK_M, BLOCK_N, D, True, NEED_BOUND)
    dq = dq * scale
    if NEED_BOUND:
        tl.store(DQ + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(DQ + base_q + offs_m[:, None] * stride_qn + offs_d[None, :], dq.to(DQ.dtype.element_ty))


def run(q, k, v, out, dout, lse, sm_scale,
        bm_kv=64, bn_kv=128, w_kv=8, s_kv=3,
        bm_q=128, bn_q=128, w_q=8, s_q=2, bm_pre=64):
    B, N, Hq, D = q.shape
    Hkv = k.shape[2]
    GROUP = Hq // Hkv
    # Shape dispatch for the dq kernel: GQA (grouped) favors a narrower kv tile.
    if GROUP > 1:
        bm_q, bn_q, w_q, s_q = 128, 32, 4, 2
    dq = torch.empty_like(q)
    delta = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)
    scale2 = sm_scale * _LOG2E

    # dkdv produces per-query-head gradients into scratch [B,N,Hq,D]; GQA sums groups after.
    if GROUP == 1:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dks, dvs = dk, dv
    else:
        # per-query-head scratch in bf16: torch.sum accumulates groups in fp32 internally,
        # so this halves scratch bandwidth with no accuracy loss vs an fp32 scratch.
        dks = torch.empty((B, N, Hq, D), device=q.device, dtype=k.dtype)
        dvs = torch.empty((B, N, Hq, D), device=q.device, dtype=v.dtype)

    _bwd_preprocess[(triton.cdiv(N, bm_pre), B * Hq)](
        out, dout, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N, Hq, BLOCK_M=bm_pre, D=D)
    _bwd_dkdv[(triton.cdiv(N, bn_kv), B * Hq)](
        q, k, v, dout, lse, delta, dks, dvs,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1), lse.stride(2),
        dks.stride(0), dks.stride(1), dks.stride(2),
        sm_scale, scale2, N, Hq, Hkv, GROUP,
        BLOCK_M=bm_kv, BLOCK_N=bn_kv, D=D,
        NEED_BOUND=((N % bn_kv != 0) or (N % bm_kv != 0)),
        num_warps=w_kv, num_stages=s_kv)
    if GROUP != 1:
        dk = dks.view(B, N, Hkv, GROUP, D).sum(3).to(k.dtype)
        dv = dvs.view(B, N, Hkv, GROUP, D).sum(3).to(v.dtype)
    _bwd_dq[(triton.cdiv(N, bm_q), B * Hq)](
        q, k, v, dout, lse, delta, dq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1), lse.stride(2),
        sm_scale, scale2, N, Hq, Hkv, GROUP,
        BLOCK_M=bm_q, BLOCK_N=bn_q, D=D,
        NEED_BOUND=((N % bm_q != 0) or (N % bn_q != 0)),
        num_warps=w_q, num_stages=s_q)
    return dq, dk, dv


# =============================== VARLEN ===============================
@triton.jit
def _bwd_preprocess_varlen(
    O, DO, Delta, stride_ot, stride_oh, stride_dh, stride_dt, T, H,
    BLOCK_M: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T
    base = h * stride_oh
    o = tl.load(O + base + offs_m[:, None] * stride_ot + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * stride_ot + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + h * stride_dh + offs_m * stride_dt, tl.sum(o * do, axis=1), mask=mask_m)


@triton.jit
def _bwd_dkdv_varlen(
    Q, K, V, DO, LSE, Delta, DK, DV, SeqStart, SeqLen, LocalStart,
    stride_qt, stride_qh, stride_lh, stride_lt, scale, scale2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    seq_start = tl.load(SeqStart + pid)
    seq_len = tl.load(SeqLen + pid)
    local_start = tl.load(LocalStart + pid)
    offs_n = local_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < seq_len
    base_kv = seq_start * stride_qt + h * stride_qh
    k = tl.load(K + base_kv + offs_n[:, None] * stride_qt + offs_d[None, :], mask=mask_n[:, None], other=0.0)
    v = tl.load(V + base_kv + offs_n[:, None] * stride_qt + offs_d[None, :], mask=mask_n[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    lbase = h * stride_lh + seq_start * stride_lt
    m_start = (local_start // BLOCK_M) * BLOCK_M
    m_mask_end = ((local_start + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    dk, dv = _dkdv_inner(dk, dv, k, v, Q, DO, LSE, Delta, scale2, base_kv, stride_qt, lbase, stride_lt,
                         offs_n, offs_d, mask_n, seq_len, m_start, m_mask_end, BLOCK_M, BLOCK_N, D, True, True)
    dk, dv = _dkdv_inner(dk, dv, k, v, Q, DO, LSE, Delta, scale2, base_kv, stride_qt, lbase, stride_lt,
                         offs_n, offs_d, mask_n, seq_len, m_mask_end, seq_len, BLOCK_M, BLOCK_N, D, False, True)
    dk = dk * scale
    tl.store(DK + base_kv + offs_n[:, None] * stride_qt + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV + base_kv + offs_n[:, None] * stride_qt + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _bwd_dq_varlen(
    Q, K, V, DO, LSE, Delta, DQ, SeqStart, SeqLen, LocalStart,
    stride_qt, stride_qh, stride_lh, stride_lt, scale, scale2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    seq_start = tl.load(SeqStart + pid)
    seq_len = tl.load(SeqLen + pid)
    local_start = tl.load(LocalStart + pid)
    offs_m = local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < seq_len
    base_q = seq_start * stride_qt + h * stride_qh
    q = tl.load(Q + base_q + offs_m[:, None] * stride_qt + offs_d[None, :], mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + base_q + offs_m[:, None] * stride_qt + offs_d[None, :], mask=mask_m[:, None], other=0.0)
    lbase = h * stride_lh + seq_start * stride_lt
    lse = tl.load(LSE + lbase + offs_m * stride_lt, mask=mask_m, other=0.0)
    delta = tl.load(Delta + lbase + offs_m * stride_lt, mask=mask_m, other=0.0)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    n_body_end = (local_start // BLOCK_N) * BLOCK_N
    n_mask_end = ((local_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    dq = _dq_inner(dq, q, do, lse, delta, K, V, scale2, base_q, stride_qt, offs_m, offs_d, mask_m, seq_len,
                   0, n_body_end, BLOCK_M, BLOCK_N, D, False, True)
    dq = _dq_inner(dq, q, do, lse, delta, K, V, scale2, base_q, stride_qt, offs_m, offs_d, mask_m, seq_len,
                   n_body_end, n_mask_end, BLOCK_M, BLOCK_N, D, True, True)
    dq = dq * scale
    tl.store(DQ + base_q + offs_m[:, None] * stride_qt + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


import numpy as _np


def _build_schedule_host(cu_list, blk):
    # compact block schedule from cu_seqlens; packed into one array for a single H2D copy.
    ss, sl, ls = [], [], []
    for s, e in zip(cu_list[:-1], cu_list[1:]):
        n = e - s
        if n <= 0:
            continue
        nb = (n + blk - 1) // blk
        for i in range(nb):
            ss.append(s); sl.append(n); ls.append(i * blk)
    arr = _np.array([ss, sl, ls], dtype=_np.int32)   # [3, P]
    return arr, len(ss)


def run_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
               bm_kv=64, bn_kv=128, w_kv=8, s_kv=3,
               bm_q=128, bn_q=128, w_q=8, s_q=2, bm_pre=64):
    T, H, D = q.shape
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    delta = torch.empty((H, T), device=q.device, dtype=torch.float32)
    scale2 = sm_scale * LOG2E
    cu_list = cu_seqlens.tolist()
    dev = q.device

    _bwd_preprocess_varlen[(triton.cdiv(T, bm_pre), H)](
        out, dout, delta, out.stride(0), out.stride(1),
        delta.stride(0), delta.stride(1), T, H, BLOCK_M=bm_pre, D=D)

    akv, Pkv = _build_schedule_host(cu_list, bn_kv)
    tkv = torch.from_numpy(akv).to(dev, non_blocking=True)        # [3, Pkv], one H2D
    tSSkv, tSLkv, tLSkv = tkv[0], tkv[1], tkv[2]
    if bm_q == bn_kv:
        tSSq, tSLq, tLSq, Pq = tSSkv, tSLkv, tLSkv, Pkv
    else:
        aq, Pq = _build_schedule_host(cu_list, bm_q)
        tq = torch.from_numpy(aq).to(dev, non_blocking=True)
        tSSq, tSLq, tLSq = tq[0], tq[1], tq[2]

    _bwd_dkdv_varlen[(Pkv, H)](
        q, k, v, dout, lse, delta, dk, dv, tSSkv, tSLkv, tLSkv,
        q.stride(0), q.stride(1), lse.stride(0), lse.stride(1), sm_scale, scale2,
        BLOCK_M=bm_kv, BLOCK_N=bn_kv, D=D, num_warps=w_kv, num_stages=s_kv)
    _bwd_dq_varlen[(Pq, H)](
        q, k, v, dout, lse, delta, dq, tSSq, tSLq, tLSq,
        q.stride(0), q.stride(1), lse.stride(0), lse.stride(1), sm_scale, scale2,
        BLOCK_M=bm_q, BLOCK_N=bn_q, D=D, num_warps=w_q, num_stages=s_q)
    return dq, dk, dv


# ----- FIB entry: attention_bwd_causal_h16_d128 -----
_run_impl = run


def run(q, k, v, out, dout, lse, sm_scale):  # noqa: F811
    return _run_impl(q, k, v, out, dout, lse, sm_scale,
                     bm_kv=64, bn_kv=128, w_kv=8, s_kv=3,
                     bm_q=128, bn_q=128, w_q=8, s_q=2)
