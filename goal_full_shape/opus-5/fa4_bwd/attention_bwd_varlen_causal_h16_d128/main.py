"""Causal FlashAttention backward (bf16) for NVIDIA Blackwell, written from scratch in Triton.

Shipped entry points
--------------------
    attn_bwd_bshd(q, k, v, out, dout, lse, sm_scale, ...)              -> dq, dk, dv
    attn_bwd_thd (q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale, ...)

Both run the 5-GEMM schedule: `_preprocess` -> `_bwd_dkdv_tma` (computes dK/dV and
stages dS^T through HBM) -> `_reduce_partials` (GQA only) -> `_bwd_dq_sp`
(a single GEMM dQ = dS K).

The 7-GEMM schedule (`_bwd_dkdv` / `_bwd_dq`, reached through `bwd_dense` /
`bwd_varlen`) and the non-TMA staged variant (`bwd_dense_sp`) are kept because
report/FINAL.md quotes them as the measured alternatives; they are not on the
shipped path.

Math (per batch, head):
    S   = Q K^T * sm_scale            (causal)
    P   = exp(S - lse)
    dP  = dO V^T
    dS  = P * (dP - delta),  delta = rowsum(dO * O)
    dQ  = dS K * sm_scale   dK = dS^T Q * sm_scale   dV = P^T dO
"""
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _preprocess(DO, O, LSE, Delta, LSE2, sm_scale, log2_inv_scale,
                stride_dob, stride_dom, stride_doh,
                stride_lb, stride_lh,
                CU, N_const,
                VARLEN: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        s1 = tl.load(CU + b + 1).to(tl.int32)
        N = s1 - s0
        tok = s0
    else:
        N = N_const
        tok = 0
    offm = pid_m * BM + tl.arange(0, BM)
    if pid_m * BM >= N:
        return
    offd = tl.arange(0, D)
    mask = offm < N
    base = (b if not VARLEN else 0) * stride_dob + h * stride_doh + (offm + tok)[:, None] * stride_dom + offd[None, :]
    do = tl.load(DO + base, mask=mask[:, None], other=0.0).to(tl.float32)
    o = tl.load(O + base, mask=mask[:, None], other=0.0).to(tl.float32)
    loff = (b if not VARLEN else 0) * stride_lb + h * stride_lh + offm + tok
    lse = tl.load(LSE + loff, mask=mask, other=0.0)
    tl.store(Delta + loff, tl.sum(do * o, 1), mask=mask)
    tl.store(LSE2 + loff, lse * LOG2E + log2_inv_scale, mask=mask)


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE2, Delta, DK, DV, CU,
              sm_scale, inv_scale,
              stride_qb, stride_qm, stride_qh,
              stride_kb, stride_kn, stride_kh,
              stride_lb, stride_lh,
              N_const,
              VARLEN: tl.constexpr, GROUP: tl.constexpr,
              D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              WS: tl.constexpr, NS: tl.constexpr):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    hkv = tl.program_id(2)
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
    else:
        N = N_const
        tok = 0
        bo = b

    n0 = pid_n * BN
    if n0 >= N:
        return
    offn = n0 + tl.arange(0, BN)
    offd = tl.arange(0, D)
    nmask = offn < N

    kv_off = bo * stride_kb + hkv * stride_kh + (offn + tok)[:, None] * stride_kn + offd[None, :]
    k = tl.load(K + kv_off, mask=nmask[:, None], other=0.0)
    v = tl.load(V + kv_off, mask=nmask[:, None], other=0.0)

    acc_dk = tl.zeros([BN, D], dtype=tl.float32)
    acc_dv = tl.zeros([BN, D], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E
    m_start = n0 // BM
    nmb = tl.cdiv(N, BM) - m_start
    # tiles that can touch the diagonal
    ndiag: tl.constexpr = (BN + BM - 1) // BM

    qbase = bo * stride_qb + hkv * GROUP * stride_qh
    lbase = bo * stride_lb + hkv * GROUP * stride_lh
    nmbt = tl.cdiv(N, BM) - m_start
    mdiag = m_start + ndiag
    for it in tl.range(0, GROUP * nmbt, warp_specialize=WS, num_stages=NS):
        g = it // nmbt
        mi = (it - g * nmbt) + m_start
        qb = qbase + g * stride_qh
        lb = lbase + g * stride_lh
        m = mi * BM
        offm = m + tl.arange(0, BM)
        mmask = offm < N
        qoff = qb + (offm + tok)[:, None] * stride_qm + offd[None, :]
        q = tl.load(Q + qoff, mask=mmask[:, None], other=0.0)
        do = tl.load(DO + qoff, mask=mmask[:, None], other=0.0)
        loff = lb + offm + tok
        lse2 = tl.load(LSE2 + loff, mask=mmask, other=float("inf"))
        delta = tl.load(Delta + loff, mask=mmask, other=0.0)
        st = tl.dot(k, tl.trans(q))
        pt = tl.exp2(st * qk_scale - lse2[None, :])
        if mi < mdiag:
            pt = tl.where(offn[:, None] <= offm[None, :], pt, 0.0)
        dpt = tl.dot(v, tl.trans(do))
        dst = (pt * (dpt - delta[None, :])).to(q.dtype)
        acc_dv = tl.dot(pt.to(do.dtype), do, acc_dv)
        acc_dk = tl.dot(dst, q, acc_dk)

    tl.store(DK + kv_off, acc_dk.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DV + kv_off, (acc_dv * inv_scale).to(DV.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE2, Delta, DQ, CU,
            sm_scale, inv_scale,
            stride_qb, stride_qm, stride_qh,
            stride_kb, stride_kn, stride_kh,
            stride_lb, stride_lh,
            N_const, NBLK,
            VARLEN: tl.constexpr, GROUP: tl.constexpr,
            D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            WS: tl.constexpr, NS: tl.constexpr):
    pid_m = NBLK - 1 - tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
    else:
        N = N_const
        tok = 0
        bo = b

    m0 = pid_m * BM
    if m0 >= N:
        return
    hkv = h // GROUP
    offm = m0 + tl.arange(0, BM)
    offd = tl.arange(0, D)
    mmask = offm < N

    qoff = bo * stride_qb + h * stride_qh + (offm + tok)[:, None] * stride_qm + offd[None, :]
    q = tl.load(Q + qoff, mask=mmask[:, None], other=0.0)
    do = tl.load(DO + qoff, mask=mmask[:, None], other=0.0)
    loff = bo * stride_lb + h * stride_lh + offm + tok
    lse2 = tl.load(LSE2 + loff, mask=mmask, other=float("inf"))
    delta = tl.load(Delta + loff, mask=mmask, other=0.0)

    acc = tl.zeros([BM, D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E
    kvbase = bo * stride_kb + hkv * stride_kh
    nfull = m0 // BN              # tiles entirely below the diagonal
    nlast = tl.cdiv(min(m0 + BM, N), BN)

    for ni in tl.range(0, nlast, warp_specialize=WS, num_stages=NS):
        n = ni * BN
        offn = n + tl.arange(0, BN)
        nmask = offn < N
        kv_off = kvbase + (offn + tok)[:, None] * stride_kn + offd[None, :]
        k = tl.load(K + kv_off, mask=nmask[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * qk_scale - lse2[:, None])
        if ni >= nfull:
            p = tl.where(offn[None, :] <= offm[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        acc = tl.dot(ds, k, acc)

    tl.store(DQ + qoff, acc.to(DQ.dtype.element_ty), mask=mmask[:, None])


import math

DEF_CFG = dict(BMp=128, pw=4, BM1=64, BN1=128, w1=8, ns1=1, NS1=3, WS1=False,
               BM2=128, BN2=64, w2=4, ns2=1, NS2=3, WS2=False)


def bwd_dense(q, k, v, out, dout, lse, sm_scale, cfg=None, dq=None, dk=None, dv=None):
    c = dict(DEF_CFG); c.update(cfg or {})
    B, N, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(N, BMp), B, HQ)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1), lse, N, False, D, BMp, num_warps=c["pw"])
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv[(triton.cdiv(N, BN1), B, HKV)](
        q, k, v, dout, lse2, delta, dk, dv, lse, sm_scale, 1.0 / sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1), N,
        False, GROUP, D, BM1, BN1, c["WS1"], c["NS1"],
        num_warps=c["w1"], num_stages=c["ns1"])
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(N, BM2)
    _bwd_dq[(nblk, B, HQ)](
        q, k, v, dout, lse2, delta, dq, lse, sm_scale, 1.0 / sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1), N, nblk,
        False, GROUP, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        num_warps=c["w2"], num_stages=c["ns2"], num_ctas=c.get("nc2", 1))
    return dq, dk, dv


def bwd_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale, cfg=None, dq=None, dk=None, dv=None):
    c = dict(DEF_CFG); c.update(cfg or {})
    T, H, D = q.shape
    nseq = int(cu_seqlens.numel()) - 1
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((H, T), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((H, T), dtype=torch.float32, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(max_seqlen, BMp), nseq, H)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        0, dout.stride(0), dout.stride(1),
        0, lse.stride(0), cu_seqlens, 0, True, D, BMp, num_warps=c["pw"])
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv[(triton.cdiv(max_seqlen, BN1), nseq, H)](
        q, k, v, dout, lse2, delta, dk, dv, cu_seqlens, sm_scale, 1.0 / sm_scale,
        0, q.stride(0), q.stride(1),
        0, k.stride(0), k.stride(1),
        0, lse.stride(0), 0,
        True, 1, D, BM1, BN1, c["WS1"], c["NS1"],
        num_warps=c["w1"], num_stages=c["ns1"])
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(max_seqlen, BM2)
    _bwd_dq[(nblk, nseq, H)](
        q, k, v, dout, lse2, delta, dq, cu_seqlens, sm_scale, 1.0 / sm_scale,
        0, q.stride(0), q.stride(1),
        0, k.stride(0), k.stride(1),
        0, lse.stride(0), 0, nblk,
        True, 1, D, BM2, BN2, c["WS2"], c["NS2"],
        num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


# =====================================================================
# Variant B: spill dS from the dK/dV pass so the dQ pass is a single GEMM.
# Total GEMM count drops from 7 to 5 at the cost of streaming dS^T through
# HBM once (bf16 -- exactly the precision the MMA would have consumed).
# =====================================================================

@triton.jit
def _bwd_dkdv_sp(Q, K, V, DO, LSE2, Delta, DK, DV, DS, CU, DKP, DVP, stride_pk,
                 sm_scale, inv_scale,
                 stride_qb, stride_qm, stride_qh,
                 stride_kb, stride_kn, stride_kh,
                 stride_lb, stride_lh,
                 stride_sb, stride_sh, DSN,
                 N_const,
                 VARLEN: tl.constexpr, GROUP: tl.constexpr,
                 D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 WS: tl.constexpr, NS: tl.constexpr, GSPLIT: tl.constexpr,
                 I64: tl.constexpr):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    pz = tl.program_id(2)
    hkv = pz // GSPLIT
    gsub: tl.constexpr = GROUP // GSPLIT
    g0 = (pz % GSPLIT) * gsub
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
        ntok = tok
    else:
        N = N_const
        tok = 0
        bo = b
        ntok = 0
    dsbase = b.to(tl.int64) * stride_sb

    n0 = pid_n * BN
    if n0 >= N:
        return
    offn = n0 + tl.arange(0, BN)
    offd = tl.arange(0, D)
    nmask = offn < N

    kv_off = bo * stride_kb + hkv * stride_kh + (offn + tok)[:, None] * stride_kn + offd[None, :]
    k = tl.load(K + kv_off, mask=nmask[:, None], other=0.0)
    v = tl.load(V + kv_off, mask=nmask[:, None], other=0.0)

    acc_dk = tl.zeros([BN, D], dtype=tl.float32)
    acc_dv = tl.zeros([BN, D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E
    m_start = n0 // BM
    ndiag: tl.constexpr = (BN + BM - 1) // BM
    nmbt = tl.cdiv(N, BM) - m_start
    mdiag = m_start + ndiag

    qbase = bo * stride_qb + (hkv * GROUP + g0) * stride_qh
    lbase = bo * stride_lb + (hkv * GROUP + g0) * stride_lh
    for it in tl.range(0, gsub * nmbt, warp_specialize=WS, num_stages=NS):
        g = it // nmbt
        mi = (it - g * nmbt) + m_start
        qb = qbase + g * stride_qh
        lb = lbase + g * stride_lh
        m = mi * BM
        offm = m + tl.arange(0, BM)
        mmask = offm < N
        qoff = qb + (offm + tok)[:, None] * stride_qm + offd[None, :]
        q = tl.load(Q + qoff, mask=mmask[:, None], other=0.0)
        do = tl.load(DO + qoff, mask=mmask[:, None], other=0.0)
        loff = lb + offm + tok
        lse2 = tl.load(LSE2 + loff, mask=mmask, other=float("inf"))
        delta = tl.load(Delta + loff, mask=mmask, other=0.0)
        st = tl.dot(k, tl.trans(q))
        pt = tl.exp2(st * qk_scale - lse2[None, :])
        if mi < mdiag:
            pt = tl.where(offn[:, None] <= offm[None, :], pt, 0.0)
        dpt = tl.dot(v, tl.trans(do))
        dst = (pt * (dpt - delta[None, :])).to(q.dtype)
        acc_dv = tl.dot(pt.to(do.dtype), do, acc_dv)
        acc_dk = tl.dot(dst, q, acc_dk)
        # stage dS^T  (row = key index, col = query index)
        if I64:
            srow = (offn + ntok)[:, None].to(tl.int64) * DSN
        else:
            srow = (offn + ntok)[:, None] * DSN
        sp = DS + dsbase + (hkv * GROUP + g0 + g).to(tl.int64) * stride_sh \
             + srow + offm[None, :]
        tl.store(sp, dst, mask=nmask[:, None])

    if GSPLIT == 1:
        tl.store(DK + kv_off, acc_dk.to(DK.dtype.element_ty), mask=nmask[:, None])
        tl.store(DV + kv_off, (acc_dv * inv_scale).to(DV.dtype.element_ty), mask=nmask[:, None])
    else:
        po = (pz % GSPLIT) * stride_pk + kv_off
        tl.store(DKP + po, acc_dk, mask=nmask[:, None])
        tl.store(DVP + po, acc_dv * inv_scale, mask=nmask[:, None])


@triton.jit
def _reduce_partials(DKP, DVP, DK, DV, stride_pk, NTOT, GSPLIT: tl.constexpr,
                     BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < NTOT
    a = tl.zeros([BLK], dtype=tl.float32)
    c = tl.zeros([BLK], dtype=tl.float32)
    for i in range(GSPLIT):
        a += tl.load(DKP + i * stride_pk + off, mask=m, other=0.0)
        c += tl.load(DVP + i * stride_pk + off, mask=m, other=0.0)
    tl.store(DK + off, a.to(DK.dtype.element_ty), mask=m)
    tl.store(DV + off, c.to(DV.dtype.element_ty), mask=m)


@triton.jit
def _bwd_dq_sp(K, DS, DQ, CU,
               stride_kb, stride_kn, stride_kh,
               stride_qb, stride_qm, stride_qh,
               stride_sb, stride_sh, DSN,
               N_const, NBLK,
               VARLEN: tl.constexpr, GROUP: tl.constexpr,
               D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               WS: tl.constexpr, NS: tl.constexpr, CMASK: tl.constexpr,
               QX: tl.constexpr, QD: tl.constexpr, KD: tl.constexpr,
               I64: tl.constexpr,
               ):
    # QX: query-tile index fastest (1) or slowest (0); QD: walk high-to-low (1)
    # or low-to-high (0).  Which pair wins is measured, not assumed.
    if QX:
        p0 = tl.program_id(0)
        b = tl.program_id(1)
        h = tl.program_id(2)
    else:
        b = tl.program_id(0)
        h = tl.program_id(1)
        p0 = tl.program_id(2)
    pid_m = (NBLK - 1 - p0) if QD else p0
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
        ntok = tok
    else:
        N = N_const
        tok = 0
        bo = b
        ntok = 0
    m0 = pid_m * BM
    if m0 >= N:
        return
    hkv = h // GROUP
    offm = m0 + tl.arange(0, BM)
    offd = tl.arange(0, D)
    mmask = offm < N
    acc = tl.zeros([BM, D], dtype=tl.float32)
    kvbase = bo * stride_kb + hkv * stride_kh
    sbase = DS + b.to(tl.int64) * stride_sb + h.to(tl.int64) * stride_sh
    nlast = tl.cdiv(min(m0 + BM, N), BN)
    for ii in tl.range(0, nlast, warp_specialize=WS, num_stages=NS):
        # KD=1 walks the key tiles downward: the CTAs resident together are
        # adjacent query tiles, so sweeping upward puts them all on the same K
        # tile at the same instant (contention); downward staggers them.
        ni = (nlast - 1 - ii) if KD else ii
        offn = ni * BN + tl.arange(0, BN)
        nmask = offn < N
        k = tl.load(K + kvbase + (offn + tok)[:, None] * stride_kn + offd[None, :],
                    mask=nmask[:, None], other=0.0)
        # No mask on the contiguous (query) axis: masking it would serialise the
        # load, and the padded row stride keeps it in bounds.  Columns past the
        # sequence end only feed output rows that are never stored.
        # I64 is picked from the shape alone: the dS workspace is indexed as
        # row * DSN + col, which overflows int32 once rows * DSN passes 2^31
        # (a key axis of ~46k).  No dataset workload is near that, so the
        # 32-bit form stays the default and the wide one costs nothing.
        if I64:
            srow = (offn + ntok)[:, None].to(tl.int64) * DSN
        else:
            srow = (offn + ntok)[:, None] * DSN
        dst = tl.load(sbase + srow + offm[None, :],
                      mask=nmask[:, None], other=0.0)
        # The dK/dV pass only staged dS^T at or below the diagonal; when the query
        # tile is wider than that pass' key tile, part of this block was never
        # written.  Everything it could not have written is above the diagonal,
        # where dS is zero anyway.
        if CMASK:
            if ni * BN + BN > m0:
                dst = tl.where(offn[:, None] <= offm[None, :], dst, 0.0)
        acc = tl.dot(tl.trans(dst), k, acc)
    tl.store(DQ + bo * stride_qb + h * stride_qh + (offm + tok)[:, None] * stride_qm + offd[None, :],
             acc.to(DQ.dtype.element_ty), mask=mmask[:, None])


DEF_SP = dict(BMp=32, pw=8, BM1=64, BN1=128, w1=8, ns1=1, NS1=2, WS1=False, GS=0,
              BM2=128, BN2=64, w2=8, ns2=1, NS2=3, WS2=False)


def bwd_dense_sp(q, k, v, out, dout, lse, sm_scale, cfg=None, dq=None, dk=None, dv=None, ws=None):
    c = dict(DEF_SP); c.update(cfg or {})
    B, N, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    # dS^T row stride is padded to 256 so the column-unmasked store (and the dQ
    # pass' read, which may use a wider query tile) always stays inside the row.
    DSN = (N + 255) // 256 * 256
    ds = ws if ws is not None else torch.empty((B, HQ, N, DSN), dtype=torch.bfloat16, device=q.device)
    GS = min(c.get("GS", 4) or 4, GROUP)
    if GS > 1:
        nk = dk.numel()
        part = torch.empty((2 * GS, nk), dtype=torch.float32, device=q.device)
        dkp, dvp = part[:GS], part[GS:]
    else:
        dkp = dvp = dk
        nk = 0
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(N, BMp), B, HQ)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1), lse, N, False, D, BMp, num_warps=c["pw"])
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv_sp[(triton.cdiv(N, BN1), B, HKV * GS)](
        q, k, v, dout, lse2, delta, dk, dv, ds, lse, dkp, dvp, (nk if GS > 1 else 0),
        sm_scale, 1.0 / sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1),
        N * HQ * DSN, N * DSN, DSN, N,
        False, GROUP, D, BM1, BN1, c["WS1"], c["NS1"], GS, _wide(N, DSN),
        num_warps=c["w1"], num_stages=c["ns1"], num_ctas=c.get("nc1", 1))
    if GS > 1:
        _reduce_partials[(triton.cdiv(nk, 4096),)](dkp, dvp, dk, dv, nk, nk, GS, 4096, num_warps=8)
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(N, BM2)
    _bwd_dq_sp[((nblk, B, HQ) if qx else (B, HQ, nblk))](
        k, ds, dq, lse,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        N * HQ * DSN, N * DSN, DSN, N, nblk,
        False, GROUP, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        _wide(N, DSN),
        num_warps=c["w2"], num_stages=c["ns2"], num_ctas=c.get("nc2", 1))
    return dq, dk, dv


@triton.jit
def _ds_offsets(CU, OFF, NSEQ, H):
    """OFF[i] = H * sum_{j<i} (cu[j+1]-cu[j])**2 ; single program, tiny."""
    acc = tl.zeros([], dtype=tl.int64)
    prev = tl.load(CU).to(tl.int64)
    for i in range(0, NSEQ):
        tl.store(OFF + i, acc)
        nxt = tl.load(CU + i + 1).to(tl.int64)
        n = nxt - prev
        acc += n * (((n + 127) // 128) * 128) * H
        prev = nxt
    tl.store(OFF + NSEQ, acc)


def bwd_varlen_sp(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                  cfg=None, dq=None, dk=None, dv=None, ws=None):
    c = dict(DEF_SP); c.update(cfg or {})
    T, H, D = q.shape
    nseq = int(cu_seqlens.numel()) - 1
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((H, T), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((H, T), dtype=torch.float32, device=q.device)
    DSN = (max_seqlen + 127) // 128 * 128
    if ws is None:
        ws = torch.empty((H, T, DSN), dtype=torch.bfloat16, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(max_seqlen, BMp), nseq, H)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        0, dout.stride(0), dout.stride(1),
        0, lse.stride(0), cu_seqlens, 0, True, D, BMp, num_warps=c["pw"])
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv_sp[(triton.cdiv(max_seqlen, BN1), nseq, H)](
        q, k, v, dout, lse2, delta, dk, dv, ws, cu_seqlens, dk, dv, 0,
        sm_scale, 1.0 / sm_scale,
        0, q.stride(0), q.stride(1),
        0, k.stride(0), k.stride(1),
        0, lse.stride(0),
        0, T * DSN, DSN, 0,
        True, 1, D, BM1, BN1, c["WS1"], c["NS1"], 1, _wide(T, DSN),
        num_warps=c["w1"], num_stages=c["ns1"])
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(max_seqlen, BM2)
    _bwd_dq_sp[((nblk, nseq, H) if qx else (nseq, H, nblk))](
        k, ws, dq, cu_seqlens,
        0, k.stride(0), k.stride(1),
        0, q.stride(0), q.stride(1),
        0, T * DSN, DSN, 0, nblk,
        True, 1, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        _wide(T, DSN),
        num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


# ---------------------------------------------------------------------
# Public entry points: pick the cheaper schedule per layout.
#  * bshd (dense): 5-GEMM schedule, dS^T staged through HBM.
#  * thd  (varlen): 7-GEMM schedule -- the ragged dS^T layout makes the
#    staging traffic more expensive than simply recomputing S and dP.
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# dK/dV pass, variant with device-side TMA descriptors for the operand
# tiles.  Same math/schedule as _bwd_dkdv_sp, but Q/dO/K/V come in through
# the TMA engine instead of per-thread global loads.
# ---------------------------------------------------------------------
@triton.jit
def _bwd_dkdv_tma(Q, K, V, DO, LSE2, Delta, DK, DV, DS, CU,
                  DKP, DVP, stride_pk,
                  sm_scale, inv_scale,
                  stride_qb, stride_qm, stride_qh,
                  stride_kb, stride_kn, stride_kh,
                  stride_lb, stride_lh,
                  stride_sb, stride_sh, DSN,
                  N_const, HQD, HKVD,
                  VARLEN: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr,
                  NS: tl.constexpr, GSPLIT: tl.constexpr, SPILL: tl.constexpr,
                  I64: tl.constexpr,
                  ):
    # Grid is (B, HKV*GSPLIT, n_key_tiles): the key-tile index varies slowest so
    # the longest CTAs (lowest key tile, most query tiles) run in the first wave.
    b = tl.program_id(0)
    pz = tl.program_id(1)
    pid_n = tl.program_id(2)
    hkv = pz // GSPLIT
    gsub: tl.constexpr = GROUP // GSPLIT
    g0 = (pz % GSPLIT) * gsub
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
        nrow = tl.load(CU + tl.num_programs(0)).to(tl.int32)
    else:
        N = N_const
        tok = 0
        bo = b
        nrow = N_const
    n0 = pid_n * BN
    if n0 >= N:
        return
    offn = n0 + tl.arange(0, BN)
    offd = tl.arange(0, D)
    nmask = offn < N

    # per-(batch) descriptors: rows clamp at the sequence end, so tails zero-fill
    qd = tl.make_tensor_descriptor(Q + bo * stride_qb, [nrow, HQD], [stride_qm, 1], [BM, D])
    od = tl.make_tensor_descriptor(DO + bo * stride_qb, [nrow, HQD], [stride_qm, 1], [BM, D])
    kd = tl.make_tensor_descriptor(K + bo * stride_kb, [nrow, HKVD], [stride_kn, 1], [BN, D])
    vd = tl.make_tensor_descriptor(V + bo * stride_kb, [nrow, HKVD], [stride_kn, 1], [BN, D])

    k = kd.load([n0 + tok, hkv * D])
    v = vd.load([n0 + tok, hkv * D])

    acc_dk = tl.zeros([BN, D], dtype=tl.float32)
    acc_dv = tl.zeros([BN, D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E
    m_start = n0 // BM
    ndiag: tl.constexpr = (BN + BM - 1) // BM
    nmbt = tl.cdiv(N, BM) - m_start
    mdiag = m_start + ndiag
    dsbase = b.to(tl.int64) * stride_sb
    for it in tl.range(0, gsub * nmbt, num_stages=NS):
        g = it // nmbt
        mi = (it - g * nmbt) + m_start
        h = hkv * GROUP + g0 + g
        m = mi * BM
        offm = m + tl.arange(0, BM)
        mmask = offm < N
        q = qd.load([m + tok, h * D])
        do = od.load([m + tok, h * D])
        loff = bo * stride_lb + h * stride_lh + offm + tok
        lse2 = tl.load(LSE2 + loff, mask=mmask, other=float("inf"))
        delta = tl.load(Delta + loff, mask=mmask, other=0.0)
        st = tl.dot(k, tl.trans(q))
        pt = tl.exp2(st * qk_scale - lse2[None, :])
        if mi < mdiag:
            pt = tl.where(offn[:, None] <= offm[None, :], pt, 0.0)
        dpt = tl.dot(v, tl.trans(do))
        dst = (pt * (dpt - delta[None, :])).to(q.dtype)
        acc_dv = tl.dot(pt.to(do.dtype), do, acc_dv)
        acc_dk = tl.dot(dst, q, acc_dk)
        if SPILL:
            if I64:
                srow = (offn + (tok if VARLEN else 0))[:, None].to(tl.int64) * DSN
            else:
                srow = (offn + (tok if VARLEN else 0))[:, None] * DSN
            sp = DS + dsbase + h.to(tl.int64) * stride_sh + srow + offm[None, :]
            tl.store(sp, dst, mask=nmask[:, None])

    kv_off = bo * stride_kb + hkv * stride_kh + (offn + tok)[:, None] * stride_kn + offd[None, :]
    if GSPLIT == 1:
        tl.store(DK + kv_off, acc_dk.to(DK.dtype.element_ty), mask=nmask[:, None])
        tl.store(DV + kv_off, (acc_dv * inv_scale).to(DV.dtype.element_ty), mask=nmask[:, None])
    else:
        po = (pz % GSPLIT) * stride_pk + kv_off
        tl.store(DKP + po, acc_dk, mask=nmask[:, None])
        tl.store(DVP + po, acc_dv * inv_scale, mask=nmask[:, None])


_ALLOC_SET = [False]
# Scratch for the staged dS^T: one growable buffer per device.  Its size depends
# only on the problem shape, never on the input values, so it is reused across
# calls the same way the JIT-compiled kernels are.
_WS_CACHE = {}


def _wide(rows, dsn):
    """True when the dS workspace needs 64-bit row offsets.

    dS is addressed as row * DSN + col, so the product overflows int32 once
    rows * DSN passes 2^31 (a key axis of roughly 46k).  The margin covers the
    key axis being padded up to a tile.  This depends only on the shape, so the
    kernels select the narrow or wide form at compile time, and every workload
    here keeps the narrow one.
    """
    return 1 if (rows + 256) * dsn >= (1 << 31) else 0


def _get_ws(key, numel, device):
    dev = device if isinstance(device, str) else str(device)
    buf = _WS_CACHE.get(dev)
    if buf is None or buf.numel() < numel:
        buf = None
        _WS_CACHE.pop(dev, None)       # drop the old one before allocating
        buf = torch.empty(numel, dtype=torch.bfloat16, device=device)
        _WS_CACHE[dev] = buf
    return buf


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def bwd_dense_tma(q, k, v, out, dout, lse, sm_scale, cfg=None, dq=None, dk=None, dv=None, ws=None):
    if not _ALLOC_SET[0]:
        triton.set_allocator(_alloc_fn)
        _ALLOC_SET[0] = True
    c = dict(DEF_SP); c.update(cfg or {})
    B, N, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    # dS^T row stride is padded to 256 so the column-unmasked store (and the dQ
    # pass' read, which may use a wider query tile) always stays inside the row.
    DSN = (N + 255) // 256 * 256
    ds = ws if ws is not None else _get_ws(("ds", B, HQ, N, DSN), B * HQ * N * DSN, q.device)
    GS = c.get("GS", 0)
    if GS <= 0:
        # Split the GQA group across CTAs only when the grid is too small to
        # hide the causal imbalance: the longest CTA (key tile 0) walks
        # GROUP/GS * N/BM1 query tiles, and that must not exceed the average
        # per-SM load.  Depends on the shape alone, never on the data.
        need = 296.0 / max(1, B * HKV * triton.cdiv(N, c["BN1"]))
        GS = 1
        while GS < GROUP and GS < need:
            GS *= 2
    GS = min(GS, GROUP)
    if GS > 1:
        nk = dk.numel()
        part = torch.empty((2 * GS, nk), dtype=torch.float32, device=q.device)
        dkp, dvp = part[:GS], part[GS:]
    else:
        dkp = dvp = dk; nk = 0
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(N, BMp), B, HQ)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1), lse, N, False, D, BMp, num_warps=c["pw"])
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv_tma[(B, HKV * GS, triton.cdiv(N, BN1))](
        q, k, v, dout, lse2, delta, dk, dv, ds, lse, dkp, dvp, (nk if GS > 1 else 0),
        sm_scale, 1.0 / sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1),
        N * HQ * DSN, N * DSN, DSN, N, HQ * D, HKV * D,
        False, GROUP, D, BM1, BN1, c["NS1"], GS, True, _wide(N, DSN),
        num_warps=c["w1"], num_stages=c["ns1"])
    if GS > 1:
        _reduce_partials[(triton.cdiv(nk, 4096),)](dkp, dvp, dk, dv, nk, nk, GS, 4096, num_warps=8)
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(N, BM2)
    _bwd_dq_sp[((nblk, B, HQ) if qx else (B, HQ, nblk))](
        k, ds, dq, lse,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        N * HQ * DSN, N * DSN, DSN, N, nblk,
        False, GROUP, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        _wide(N, DSN),
        num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


@triton.jit
def _bwd_dq_tma(Q, K, V, DO, LSE2, Delta, DQ, CU,
                sm_scale,
                stride_qb, stride_qm, stride_qh,
                stride_kb, stride_kn, stride_kh,
                stride_lb, stride_lh,
                N_const, NBLK, HQD, HKVD,
                VARLEN: tl.constexpr, GROUP: tl.constexpr,
                D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr,
                ):
    pid_m = NBLK - 1 - tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    if VARLEN:
        s0 = tl.load(CU + b).to(tl.int32)
        N = tl.load(CU + b + 1).to(tl.int32) - s0
        tok = s0
        bo = 0
        nrow = tl.load(CU + tl.num_programs(1)).to(tl.int32)
    else:
        N = N_const
        tok = 0
        bo = b
        nrow = N_const
    m0 = pid_m * BM
    if m0 >= N:
        return
    hkv = h // GROUP
    offm = m0 + tl.arange(0, BM)
    offd = tl.arange(0, D)
    mmask = offm < N
    qd = tl.make_tensor_descriptor(Q + bo * stride_qb, [nrow, HQD], [stride_qm, 1], [BM, D])
    od = tl.make_tensor_descriptor(DO + bo * stride_qb, [nrow, HQD], [stride_qm, 1], [BM, D])
    kd = tl.make_tensor_descriptor(K + bo * stride_kb, [nrow, HKVD], [stride_kn, 1], [BN, D])
    vd = tl.make_tensor_descriptor(V + bo * stride_kb, [nrow, HKVD], [stride_kn, 1], [BN, D])
    q = qd.load([m0 + tok, h * D])
    do = od.load([m0 + tok, h * D])
    loff = bo * stride_lb + h * stride_lh + offm + tok
    lse2 = tl.load(LSE2 + loff, mask=mmask, other=float("inf"))
    delta = tl.load(Delta + loff, mask=mmask, other=0.0)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E
    nfull = m0 // BN
    nlast = tl.cdiv(min(m0 + BM, N), BN)
    for ni in tl.range(0, nlast, num_stages=NS):
        n = ni * BN
        offn = n + tl.arange(0, BN)
        k = kd.load([n + tok, hkv * D])
        v = vd.load([n + tok, hkv * D])
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * qk_scale - lse2[:, None])
        if ni >= nfull:
            p = tl.where((offn[None, :] <= offm[:, None]) & (offn[None, :] < N), p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        acc = tl.dot(ds, k, acc)
    tl.store(DQ + bo * stride_qb + h * stride_qh + (offm + tok)[:, None] * stride_qm + offd[None, :],
             acc.to(DQ.dtype.element_ty), mask=mmask[:, None])


def bwd_varlen_tma(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                   cfg=None, dq=None, dk=None, dv=None, ws=None, spill=False):
    if not _ALLOC_SET[0]:
        triton.set_allocator(_alloc_fn); _ALLOC_SET[0] = True
    c = dict(DEF_SP); c.update(cfg or {})
    T, H, D = q.shape
    nseq = int(cu_seqlens.numel()) - 1
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((H, T), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((H, T), dtype=torch.float32, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    _preprocess[(triton.cdiv(max_seqlen, BMp), nseq, H)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        0, dout.stride(0), dout.stride(1),
        0, lse.stride(0), cu_seqlens, 0, True, D, BMp, num_warps=c["pw"])
    DSN = (max_seqlen + 255) // 256 * 256
    if spill:
        ds = ws if ws is not None else _get_ws(("dsv", H, T, DSN), H * T * DSN, q.device)
        sh = T * DSN
    else:
        ds = lse2; sh = 0; DSN = 1
    BM1, BN1 = c["BM1"], c["BN1"]
    _bwd_dkdv_tma[(nseq, H, triton.cdiv(max_seqlen, BN1))](
        q, k, v, dout, lse2, delta, dk, dv, ds, cu_seqlens, dk, dv, 0,
        sm_scale, 1.0 / sm_scale,
        0, q.stride(0), q.stride(1),
        0, k.stride(0), k.stride(1),
        0, lse.stride(0),
        0, sh, DSN, 0, H * D, H * D,
        True, 1, D, BM1, BN1, c["NS1"], 1, spill, _wide(T, DSN),
        num_warps=c["w1"], num_stages=c["ns1"])
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(max_seqlen, BM2)
    if spill:
        _bwd_dq_sp[((nblk, nseq, H) if qx else (nseq, H, nblk))](
            k, ds, dq, cu_seqlens,
            0, k.stride(0), k.stride(1),
            0, q.stride(0), q.stride(1),
            0, sh, DSN, 0, nblk,
            True, 1, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
            _wide(T, DSN),
            num_warps=c["w2"], num_stages=c["ns2"])
    else:
        _bwd_dq_tma[(nblk, nseq, H)](
            q, k, v, dout, lse2, delta, dq, cu_seqlens, sm_scale,
            0, q.stride(0), q.stride(1),
            0, k.stride(0), k.stride(1),
            0, lse.stride(0), 0, nblk, H * D, H * D,
            True, 1, D, BM2, BN2, c["NS2"],
            num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Gluon dK/dV pass (Blackwell): warp-specialised, with the softmax backward on
# its own consumer warps (NWC, 8 as shipped) and dedicated TMA / MMA warps.
#
# Why this exists: in the Triton version the dK/dV kernel runs 8 warps on one
# CTA per SM and the softmax backward sits in the dependency chain between the
# S^T/dP^T MMA and the dV/dK MMA of the same tile, so the tensor cores idle
# through it.  Gluon lets the epilogue have a warp group of its own, sized
# independently of the MMA warp (plain Triton drops off the tcgen05 path
# entirely at num_warps=16), and lets dS^T leave as a TMA store out of the
# shared-memory tile the dK MMA already needs, instead of a half-sector store
# from registers.  Measured on MHA: 1995 -> 1735 us.
#
# 8 consumer warps beat 16 by 1.2-1.9% on a settled card (bench/nwab.py): the
# work is unit-limited rather than warp-limited, so halving the warps costs no
# throughput and halves the skew at the two barriers each iteration.
#
# Tensor memory is exactly 512 columns: S^T and dP^T double-buffered (64 each)
# plus the dK and dV accumulators (128 each).
# ---------------------------------------------------------------------------
HAVE_GLUON = True
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia import blackwell as bw
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma, mbarrier, tma,
        fence_async_shared)
except Exception:                                      # pragma: no cover
    HAVE_GLUON = False

if HAVE_GLUON:
    GLOG2E = gl.constexpr(1.4426950408889634)

    @gluon.jit
    def _gl_producer(qd, od, sQ, sO, sL, bFull, bEmpty, LSE2, Delta, loff0, lh,
                     total, nmbt, m_start, hkv, tok, N,
                     GROUP: gl.constexpr, BM: gl.constexpr, D: gl.constexpr,
                     NBUF: gl.constexpr, DESC: gl.constexpr, GX: gl.constexpr,
                     GI: gl.constexpr):
        # one warp: 2 elements per thread covers a BM=64 vector
        lv: gl.constexpr = gl.BlockedLayout([BM // 32], [32], [1], [0])
        # g / mi are carried rather than recomputed: `it // nmbt` is a runtime
        # integer division on the critical path of every iteration.
        g = 0
        mi = (m_start + nmbt - 1) if DESC else m_start
        for it in range(total):
            buf = it % NBUF
            if it >= NBUF:
                mbarrier.wait(bEmpty.index(buf), ((it // NBUF) & 1) ^ 1)
            m = mi * BM + tok
            h = hkv * GROUP + g
            # lse/delta for this tile go through shared memory: read from global
            # inside the consumer they sit on its critical path, and the
            # vector-to-broadcast layout change costs a shuffle per warp.
            offm = m - tok + gl.arange(0, BM, layout=lv)
            lo = loff0 + h * lh + tok
            sL.index(2 * buf).store(
                gl.load(LSE2 + lo + offm, mask=offm < N, other=float("inf")))
            sL.index(2 * buf + 1).store(
                gl.load(Delta + lo + offm, mask=offm < N, other=0.0))
            mbarrier.expect(bFull.index(buf), 2 * BM * D * 2)
            tma.async_copy_global_to_shared(qd, [m, h * D], bFull.index(buf), sQ.index(buf))
            tma.async_copy_global_to_shared(od, [m, h * D], bFull.index(buf), sO.index(buf))
            mbarrier.arrive(bFull.index(buf), count=1)
            if GI:
                g += 1
                if g == GROUP:
                    g = 0
                    mi = (mi - 1) if DESC else (mi + 1)
            else:
                if DESC:
                    mi -= 1
                    if GROUP > 1:
                        if mi < m_start:
                            mi = m_start + nmbt - 1
                            g += 1
                else:
                    mi += 1
                    if GROUP > 1:
                        if mi == m_start + nmbt:
                            mi = m_start
                            g += 1

    @gluon.jit
    def _gl_mma(sK, sV, sQ, sO, sPt, sSt, tSt, tPt, tDK, tDV,
                bFull, bEmpty, bSP, bEP, bFree, bKV, total,
                NBUF: gl.constexpr, NSLOT: gl.constexpr):
        mbarrier.wait(bKV, 0)
        for it in range(total):
            s = it % NSLOT
            buf = it % NBUF
            mbarrier.wait(bFull.index(buf), (it // NBUF) & 1)
            tcgen05_mma(sK, sQ.index(buf).permute([1, 0]), tSt.index(s), use_acc=False)
            tcgen05_mma(sV, sO.index(buf).permute([1, 0]), tPt.index(s), use_acc=False,
                        mbarriers=[bSP.index(s)])
            if it >= 1:
                ip = it - 1
                sp = ip % NSLOT
                bufp = ip % NBUF
                mbarrier.wait(bEP.index(sp), (ip // NSLOT) & 1)
                tcgen05_mma(sPt.index(sp), sO.index(bufp), tDV, use_acc=(ip > 0))
                tcgen05_mma(sSt.index(sp), sQ.index(bufp), tDK, use_acc=(ip > 0),
                            mbarriers=[bFree.index(sp), bEmpty.index(bufp)])
        ip = total - 1
        sp = ip % NSLOT
        bufp = ip % NBUF
        mbarrier.wait(bEP.index(sp), (ip // NSLOT) & 1)
        tcgen05_mma(sPt.index(sp), sO.index(bufp), tDV, use_acc=(ip > 0))
        tcgen05_mma(sSt.index(sp), sQ.index(bufp), tDK, use_acc=(ip > 0),
                    mbarriers=[bFree.index(sp)])
        mbarrier.wait(bFree.index(sp), (ip // NSLOT) & 1)

    @gluon.jit
    def _gl_consumer(sPt, sSt, tSt, tPt, sL, bSP, bEP, bFree, dsd,
                     sm_scale, total, nmbt, m_start, n0, N, tok, NPAD,
                     GROUP: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                     NSLOT: gl.constexpr, NBUF: gl.constexpr, NWC: gl.constexpr,
                     DESC: gl.constexpr, GX: gl.constexpr, GI: gl.constexpr):
        ls: gl.constexpr = bw.get_tmem_reg_layout(
            gl.float32, [BN, BM], TensorMemoryLayout((128, BM), col_stride=1), NWC,
            "32x32b", [])
        offn = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, ls))
        ndiag: gl.constexpr = (BN + BM - 1) // BM
        g = 0
        mi = (m_start + nmbt - 1) if DESC else m_start
        for it in range(total):
            s = it % NSLOT
            buf = it % NBUF
            m = mi * BM
            mbarrier.wait(bSP.index(s), (it // NSLOT) & 1)
            st = tSt.index(s).load(ls)
            dpt = tPt.index(s).load(ls)
            offm = m + gl.arange(0, BM, layout=gl.SliceLayout(0, ls))
            lse2 = sL.index(2 * buf).load(gl.SliceLayout(0, ls))
            delta = sL.index(2 * buf + 1).load(gl.SliceLayout(0, ls))
            pt = gl.exp2(st * (sm_scale * GLOG2E) - lse2[None, :])
            if mi < m_start + ndiag:
                pt = gl.where(offn[:, None] <= offm[None, :], pt, 0.0)
            dst = (pt * (dpt - delta[None, :])).to(gl.bfloat16)
            if it >= NSLOT:
                mbarrier.wait(bFree.index(s), ((it - NSLOT) // NSLOT) & 1)
                tma.store_wait(NSLOT - 1)        # this slot's dS store has drained
                # store_wait is an elected-thread op, so hold the whole partition
                # until it has returned: every warp is about to overwrite sPt/sSt
                gl.barrier()
            sPt.index(s).store(pt.to(gl.bfloat16))
            sSt.index(s).store(dst)
            fence_async_shared()
            # arrive is a single elected-thread arrival, so every warp of this
            # partition has to be past its stores first
            gl.barrier()
            mbarrier.arrive(bEP.index(s), count=1)
            # the varlen descriptor is already based at this sequence's first
            # row, so the coordinate is the key index within the head/sequence
            tma.async_copy_shared_to_global(dsd, [g * NPAD + n0, m], sSt.index(s))
            if GI:
                g += 1
                if g == GROUP:
                    g = 0
                    mi = (mi - 1) if DESC else (mi + 1)
            else:
                if DESC:
                    mi -= 1
                    if GROUP > 1:
                        if mi < m_start:
                            mi = m_start + nmbt - 1
                            g += 1
                else:
                    mi += 1
                    if GROUP > 1:
                        if mi == m_start + nmbt:
                            mi = m_start
                            g += 1

    @gluon.jit
    def _gl_kv(Q, K, V, DO, LSE2, Delta, DK, DV, DS, CU,
               sm_scale, inv_scale,
               stride_qb, stride_qm, stride_qh,
               stride_kb, stride_kn, stride_kh,
               stride_lb, stride_lh,
               stride_sb, stride_sh, DSN, NPAD, NROW,
               N_const, HQD, HKVD,
               VARLEN: gl.constexpr, GROUP: gl.constexpr, D: gl.constexpr,
               BM: gl.constexpr, BN: gl.constexpr,
               NBUF: gl.constexpr, NSLOT: gl.constexpr, NWC: gl.constexpr,
               DESC: gl.constexpr, GX: gl.constexpr, GI: gl.constexpr):
        if GX:
            # key tile varies fastest, so the CTAs resident together are the key
            # tiles of one (batch, kv head); walking the query tiles downward
            # puts all of them on the same Q/dO tile at the same time, which L2
            # can serve once instead of once per CTA
            pid_n = gl.program_id(0)
            b = gl.program_id(1)
            hkv = gl.program_id(2)
        else:
            b = gl.program_id(0)
            hkv = gl.program_id(1)
            pid_n = gl.program_id(2)
        if VARLEN:
            s0 = gl.load(CU + b)
            N = gl.load(CU + b + 1) - s0
            tok = s0
            bo = 0
            nrow = NROW
        else:
            N = N_const
            tok = 0
            bo = b
            nrow = N_const
        n0 = pid_n * BN
        if n0 >= N:
            return

        nvmma: gl.constexpr = gl.NVMMASharedLayout(128, 16, 2)
        qd = tma.make_tensor_descriptor(Q + bo * stride_qb, [nrow, HQD], [stride_qm, 1],
                                        [BM, D], nvmma)
        od = tma.make_tensor_descriptor(DO + bo * stride_qb, [nrow, HQD], [stride_qm, 1],
                                        [BM, D], nvmma)
        kd = tma.make_tensor_descriptor(K + bo * stride_kb, [nrow, HKVD], [stride_kn, 1],
                                        [BN, D], nvmma)
        vd = tma.make_tensor_descriptor(V + bo * stride_kb, [nrow, HKVD], [stride_kn, 1],
                                        [BN, D], nvmma)
        # dS^T descriptor: dense pads the key axis per head so a tail tile's
        # overflow lands in that head's own padding; varlen points at the
        # sequence's own rows so TMA clamping drops the overflow.
        if VARLEN:
            dsd = tma.make_tensor_descriptor(DS + hkv.to(gl.int64) * stride_sh
                                             + tok.to(gl.int64) * DSN,
                                             [N, DSN], [DSN, 1], [BN, BM], nvmma)
        else:
            dsd = tma.make_tensor_descriptor(DS + b.to(gl.int64) * stride_sb
                                             + hkv.to(gl.int64) * GROUP * stride_sh,
                                             [GROUP * NPAD, DSN], [DSN, 1], [BN, BM], nvmma)

        sK = gl.allocate_shared_memory(gl.bfloat16, [BN, D], nvmma)
        sV = gl.allocate_shared_memory(gl.bfloat16, [BN, D], nvmma)
        sQ = gl.allocate_shared_memory(gl.bfloat16, [NBUF, BM, D], nvmma)
        sO = gl.allocate_shared_memory(gl.bfloat16, [NBUF, BM, D], nvmma)
        sPt = gl.allocate_shared_memory(gl.bfloat16, [NSLOT, BN, BM], nvmma)
        sSt = gl.allocate_shared_memory(gl.bfloat16, [NSLOT, BN, BM], nvmma)
        # flat [2*NBUF, BM]: a shared layout may be at most one rank below the
        # allocation's shape, so only one level of .index() is available
        plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        sL = gl.allocate_shared_memory(gl.float32, [2 * NBUF, BM], plain)

        bl: gl.constexpr = mbarrier.MBarrierLayout()
        bKV = gl.allocate_shared_memory(gl.int64, [1], bl)
        bFull = gl.allocate_shared_memory(gl.int64, [NBUF, 1], bl)
        bEmpty = gl.allocate_shared_memory(gl.int64, [NBUF, 1], bl)
        bSP = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], bl)
        bEP = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], bl)
        bFree = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], bl)
        mbarrier.init(bKV, count=1)
        for i in gl.static_range(NBUF):
            mbarrier.init(bFull.index(i), count=2)   # TMA bytes + lse/delta store
            mbarrier.init(bEmpty.index(i), count=1)
        for i in gl.static_range(NSLOT):
            mbarrier.init(bSP.index(i), count=1)
            mbarrier.init(bEP.index(i), count=1)
            mbarrier.init(bFree.index(i), count=1)

        tl_s: gl.constexpr = TensorMemoryLayout((128, BM), col_stride=1)
        tl_o: gl.constexpr = TensorMemoryLayout((128, D), col_stride=1)
        tSt = allocate_tensor_memory(gl.float32, [NSLOT, BN, BM], tl_s)
        tPt = allocate_tensor_memory(gl.float32, [NSLOT, BN, BM], tl_s)
        tDK = allocate_tensor_memory(gl.float32, [BN, D], tl_o)
        tDV = allocate_tensor_memory(gl.float32, [BN, D], tl_o)

        mbarrier.expect(bKV, 2 * BN * D * 2)
        tma.async_copy_global_to_shared(kd, [n0 + tok, hkv * D], bKV, sK)
        tma.async_copy_global_to_shared(vd, [n0 + tok, hkv * D], bKV, sV)

        m_start = n0 // BM
        nmbt = (N + BM - 1) // BM - m_start
        total = GROUP * nmbt
        loff0 = bo * stride_lb

        gl.warp_specialize((
            (_gl_consumer, (sPt, sSt, tSt, tPt, sL, bSP, bEP, bFree, dsd,
                            sm_scale, total, nmbt, m_start, n0, N, tok,
                            NPAD, GROUP, BM, BN, NSLOT, NBUF, NWC, DESC, GX, GI)),
            (_gl_mma, (sK, sV, sQ, sO, sPt, sSt, tSt, tPt, tDK, tDV,
                       bFull, bEmpty, bSP, bEP, bFree, bKV, total, NBUF, NSLOT)),
            (_gl_producer, (qd, od, sQ, sO, sL, bFull, bEmpty, LSE2, Delta, loff0,
                            stride_lh, total, nmbt, m_start, hkv, tok, N,
                            GROUP, BM, D, NBUF, DESC, GX, GI)),
        ), worker_num_warps=[1, 1], worker_num_regs=[32, 32])

        lo: gl.constexpr = bw.get_tmem_reg_layout(
            gl.float32, [BN, D], TensorMemoryLayout((128, D), col_stride=1), NWC,
            "32x32b", [])
        offd = gl.arange(0, D, layout=gl.SliceLayout(0, lo))
        offn2 = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, lo))
        kv_off = bo * stride_kb + hkv * stride_kh + (offn2 + tok)[:, None] * stride_kn \
            + offd[None, :]
        msk = (offn2 < N)[:, None]
        gl.store(DK + kv_off, tDK.load(lo).to(gl.bfloat16), mask=msk)
        gl.store(DV + kv_off, (tDV.load(lo) * inv_scale).to(gl.bfloat16), mask=msk)


DEF_GL = dict(BMp=32, pw=8, BM1=64, BN1=128, NBUF=3, NSLOT=2, NWC=8, DESC=-1, GX=-1,
              QX=1, QD=1, KD=0, GI=0,
              BM2=128, BN2=64, w2=4, ns2=1, NS2=3, WS2=False)


def _order(c, nctas):
    """Pick the CTA order / query-loop direction from the grid size alone.

    With enough waves of CTAs it pays to make the key tile the fastest-varying
    index and walk the query tiles downward: the CTAs resident together are then
    the key tiles of one (batch, kv head), all on the same Q/dO tile at the same
    time, so L2 serves it once (MHA -6%, varlen -4%).  On a small grid the tail
    dominates instead and the longest-CTA-first order wins, which is the other
    way round (GQA +9% if forced).
    """
    d = c.get("DESC", -1)
    g = c.get("GX", -1)
    if d >= 0 and g >= 0:
        return d, g
    if nctas >= 8 * 148:
        return 0, 1          # key tile fastest, query tiles ascending
    return 0, 0              # small grid: longest CTA first


def _gl_ok(q, k):
    """The Gluon pass is specialised for head_dim 128 on Blackwell."""
    return HAVE_GLUON and q.shape[-1] == 128 and torch.cuda.get_device_capability()[0] >= 10


def bwd_dense_gl(q, k, v, out, dout, lse, sm_scale, cfg=None, dq=None, dk=None, dv=None,
                 ws=None):
    if not _ALLOC_SET[0]:
        triton.set_allocator(_alloc_fn)
        _ALLOC_SET[0] = True
    c = dict(DEF_GL)
    c.update(cfg or {})
    B, N, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    nsc = B * HQ * N
    delta = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((B, HQ, N), dtype=torch.float32, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    # Everything the later launches need is derived from the shapes, so it is
    # computed before the first launch: the preprocess kernel only runs for
    # ~43 us, and host work issued after it leaves the card idle until the dK/dV
    # launch catches up.
    BM1, BN1 = c["BM1"], c["BN1"]
    # dS^T: row stride padded to 256 (the dQ pass may read a wider query tile)
    # and the key axis padded to BN1 per head, so a tail tile's TMA store spills
    # into that head's own padding instead of the next head's rows.
    DSN = (N + 255) // 256 * 256
    NPAD = (N + BN1 - 1) // BN1 * BN1
    ds = ws if ws is not None else _get_ws(("dsg", B, HQ, NPAD, DSN),
                                           B * HQ * NPAD * DSN, q.device)
    nt1 = triton.cdiv(N, BN1)
    desc, gx = _order(c, nt1 * B * HKV)
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(N, BM2)
    i64 = _wide(N, DSN)
    grid1 = (nt1, B, HKV) if gx else (B, HKV, nt1)
    grid2 = (nblk, B, HQ) if qx else (B, HQ, nblk)
    _preprocess[(triton.cdiv(N, BMp), B, HQ)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1), lse, N, False, D, BMp, num_warps=c["pw"])
    _gl_kv[grid1](
        q, k, v, dout, lse2, delta, dk, dv, ds, lse,
        sm_scale, 1.0 / sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1),
        HQ * NPAD * DSN, NPAD * DSN, DSN, NPAD, N,
        N, HQ * D, HKV * D,
        False, GROUP, D, BM1, BN1, c["NBUF"], c["NSLOT"], c["NWC"], desc, gx,
        c.get("GI", 0), num_warps=c["NWC"])
    _bwd_dq_sp[grid2](
        k, ds, dq, lse,
        k.stride(0), k.stride(1), k.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        HQ * NPAD * DSN, NPAD * DSN, DSN, N, nblk,
        False, GROUP, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        i64,
        num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


def bwd_varlen_gl(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                  cfg=None, dq=None, dk=None, dv=None, ws=None):
    if not _ALLOC_SET[0]:
        triton.set_allocator(_alloc_fn)
        _ALLOC_SET[0] = True
    c = dict(DEF_GL)
    c.update(cfg or {})
    T, H, D = q.shape
    nseq = int(cu_seqlens.numel()) - 1
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    delta = torch.empty((H, T), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((H, T), dtype=torch.float32, device=q.device)
    l2is = -math.log2(sm_scale)
    BMp = c["BMp"]
    # as in the dense path: all shape-derived host work ahead of the first
    # launch, so the card is not idle while the host catches up
    DSN = (max_seqlen + 255) // 256 * 256
    ds = ws if ws is not None else _get_ws(("dsgv", H, T, DSN), H * T * DSN, q.device)
    BM1, BN1 = c["BM1"], c["BN1"]
    nt1 = triton.cdiv(max_seqlen, BN1)
    desc, gx = _order(c, nt1 * nseq * H)
    BM2, BN2 = c["BM2"], c["BN2"]
    qx, qd, kd = c.get("QX", 1), c.get("QD", 1), c.get("KD", 0)
    nblk = triton.cdiv(max_seqlen, BM2)
    i64 = _wide(T, DSN)
    grid1 = (nt1, nseq, H) if gx else (nseq, H, nt1)
    grid2 = (nblk, nseq, H) if qx else (nseq, H, nblk)
    _preprocess[(triton.cdiv(max_seqlen, BMp), nseq, H)](
        dout, out, lse, delta, lse2, sm_scale, l2is,
        0, dout.stride(0), dout.stride(1),
        0, lse.stride(0), cu_seqlens, 0, True, D, BMp, num_warps=c["pw"])
    _gl_kv[grid1](
        q, k, v, dout, lse2, delta, dk, dv, ds, cu_seqlens,
        sm_scale, 1.0 / sm_scale,
        0, q.stride(0), q.stride(1),
        0, k.stride(0), k.stride(1),
        0, lse.stride(0),
        0, T * DSN, DSN, 0, T,
        0, H * D, H * D,
        True, 1, D, BM1, BN1, c["NBUF"], c["NSLOT"], c["NWC"], desc, gx,
        c.get("GI", 0), num_warps=c["NWC"])
    _bwd_dq_sp[grid2](
        k, ds, dq, cu_seqlens,
        0, k.stride(0), k.stride(1),
        0, q.stride(0), q.stride(1),
        0, T * DSN, DSN, 0, nblk,
        True, 1, D, BM2, BN2, c["WS2"], c["NS2"], BM2 > BN1, qx, qd, kd,
        i64,
        num_warps=c["w2"], num_stages=c["ns2"])
    return dq, dk, dv


def _std(t):
    """True iff the trailing dims are packed the way the kernels address them:
    (..., rows, heads, dim) with dim contiguous, head stride == dim and row
    stride == heads*dim.  The leading (batch) stride is free but must keep the
    16-byte alignment TMA descriptors need."""
    sh, st = t.shape, t.stride()
    if st[-1] != 1 or st[-2] != sh[-1] or st[-3] != sh[-1] * sh[-2]:
        return False
    return all(x % 8 == 0 for x in st[:-1])


def _as_std(t):
    return t if _std(t) else t.contiguous()


def _empty_outs(q, k, v, dq, dk, dv):
    """Degenerate input (no tokens): nothing to compute, and a grid dimension
    would be zero."""
    return (dq if dq is not None else torch.empty_like(q),
            dk if dk is not None else torch.empty_like(k),
            dv if dv is not None else torch.empty_like(v))


def attn_bwd_bshd(q, k, v, out, dout, lse, sm_scale, dq=None, dk=None, dv=None):
    if q.shape[1] == 0 or q.numel() == 0:
        return _empty_outs(q, k, v, dq, dk, dv)
    q, k, v, dout = (_as_std(t) for t in (q, k, v, dout))
    out = out.contiguous() if out.stride() != dout.stride() else out
    lse = lse if lse.stride(-1) == 1 else lse.contiguous()
    outs = (dq, dk, dv)
    stage = [t is not None and not _std(t) for t in outs]
    work = [t.contiguous() if sg else t for t, sg in zip(outs, stage)]
    fn = bwd_dense_gl if _gl_ok(q, k) else bwd_dense_tma
    r = fn(q, k, v, out, dout, lse, sm_scale, dq=work[0], dk=work[1], dv=work[2])
    if any(stage):
        res = []
        for t, w, sg in zip(outs, r, stage):
            if sg:
                t.copy_(w)
                res.append(t)
            else:
                res.append(w)
        return tuple(res)
    return r


DEF_THD = dict(BM2=128, BN2=64, w2=8, NS2=3, WS2=False)


def attn_bwd_thd(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                 dq=None, dk=None, dv=None):
    if int(max_seqlen) == 0 or q.shape[0] == 0 or int(cu_seqlens.numel()) < 2:
        return _empty_outs(q, k, v, dq, dk, dv)
    q, k, v, dout = (_as_std(t) for t in (q, k, v, dout))
    out = out.contiguous() if out.stride() != dout.stride() else out
    lse = lse if lse.stride(-1) == 1 else lse.contiguous()
    outs = (dq, dk, dv)
    stage = [t is not None and not _std(t) for t in outs]
    work = [t.contiguous() if sg else t for t, sg in zip(outs, stage)]
    if _gl_ok(q, k):
        r = bwd_varlen_gl(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                          dq=work[0], dk=work[1], dv=work[2])
    else:
        r = bwd_varlen_tma(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                           cfg=DEF_THD, dq=work[0], dk=work[1], dv=work[2], spill=True)
    if any(stage):
        res = []
        for t, w, sg in zip(outs, r, stage):
            if sg:
                t.copy_(w); res.append(t)
            else:
                res.append(w)
        return tuple(res)
    return r




# ---- FIB entry point ----
def run(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    return attn_bwd_thd(q, k, v, out, dout, lse, cu_seqlens, int(max_seqlen), float(sm_scale))
