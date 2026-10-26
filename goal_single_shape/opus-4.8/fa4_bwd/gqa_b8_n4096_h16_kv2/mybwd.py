import torch, math
import triton
import triton.language as tl

# GQA attention backward, written from scratch (FlashAttention-style recompute).
# Layout (B,N,H,D). causal. scale=1/sqrt(D). bf16 in/out.
# Forward (O, LSE) is computed separately (not timed); backward kernels below are timed.

# ----------------- delta preprocess -----------------
@triton.jit
def _delta_kernel(O, dO, Delta,
                  stride_ob, stride_on, stride_oh, stride_od,
                  stride_db, stride_dn, stride_dh, stride_dd,
                  stride_deltab, stride_deltah, stride_deltan,
                  N, H, D: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    o_ptr = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    do_ptr = dO + b * stride_db + h * stride_dh + offs_m[:, None] * stride_dn + offs_d[None, :] * stride_dd
    mask = offs_m[:, None] < N
    o = tl.load(o_ptr, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    d_ptr = Delta + b * stride_deltab + h * stride_deltah + offs_m * stride_deltan
    tl.store(d_ptr, delta, mask=offs_m < N)


# ----------------- dK/dV kernel (split per query head, transpose-minimal) -----------------
# grid = (num_kv_blocks, B*H). Each program handles one (b, h) at one kv block,
# writing a partial dk/dv for that head. Partials are reduced over the group afterwards.
# Uses pT-layout (keys as rows) to avoid transposing the (M,N) score/ds matrices.
@triton.jit
def _dkdv_split_kernel(Q, K, V, dO, LSE, Delta, dKp, dVp,
                 stride_qb, stride_qn, stride_qh, stride_qd,
                 stride_kb, stride_kn, stride_kh, stride_kd,
                 stride_vb, stride_vn, stride_vh, stride_vd,
                 stride_dob, stride_don, stride_doh, stride_dod,
                 stride_lb, stride_lh, stride_ln,
                 stride_deb, stride_deh, stride_den,
                 stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                 stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                 N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
                 D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kv = h // G
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    kmask = offs_n[:, None] < N
    k_ptr = K + b * stride_kb + kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptr = V + b * stride_vb + kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptr, mask=kmask, other=0.0)   # (BLOCK_N, D)
    v = tl.load(v_ptr, mask=kmask, other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for m in range(start_n, N, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mmask = offs_m < N
        # load qT (D, BLOCK_M) and doT (D, BLOCK_M) directly (transpose via indexing)
        qT_ptr = Q + b * stride_qb + h * stride_qh + offs_d[:, None] * stride_qd + offs_m[None, :] * stride_qn
        doT_ptr = dO + b * stride_dob + h * stride_doh + offs_d[:, None] * stride_dod + offs_m[None, :] * stride_don
        do_ptr = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
        qT = tl.load(qT_ptr, mask=mmask[None, :], other=0.0)     # (D, BLOCK_M)
        doT = tl.load(doT_ptr, mask=mmask[None, :], other=0.0)   # (D, BLOCK_M)
        do = tl.load(do_ptr, mask=mmask[:, None], other=0.0)     # (BLOCK_M, D)
        lse = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_ln, mask=mmask, other=0.0)
        delta = tl.load(Delta + b * stride_deb + h * stride_deh + offs_m * stride_den, mask=mmask, other=0.0)
        # sT (BLOCK_N, BLOCK_M) = k @ qT ; use exp2 (fold log2e into scale)
        sT = tl.dot(k, qT) * (scale * 1.4426950408889634)
        causalT = offs_n[:, None] <= offs_m[None, :]
        sT = tl.where(causalT, sT, float("-inf"))
        pT = tl.exp2(sT - lse[None, :] * 1.4426950408889634)     # (BLOCK_N, BLOCK_M)
        dv += tl.dot(pT.to(do.dtype), do)                        # (BLOCK_N, D)
        dpT = tl.dot(v, doT)                                     # (BLOCK_N, BLOCK_M)
        dsT = (pT * (dpT - delta[None, :]) * scale).to(qT.dtype) # (BLOCK_N, BLOCK_M)
        dk += tl.dot(dsT, tl.trans(qT))                          # (BLOCK_N, D)
    dk_ptr = dKp + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptr = dVp + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptr, dk, mask=kmask)
    tl.store(dv_ptr, dv, mask=kmask)


# ----------------- separate dV-only and dK-only kernels (single accumulator each) -----------------
@triton.jit
def _dv_only_kernel(Q, K, V, dO, LSE, dVp,
        s_qb,s_qn,s_qh,s_qd, s_kb,s_kn,s_kh,s_kd, s_dob,s_don,s_doh,s_dod,
        s_lb,s_lh,s_ln, s_vb_,s_vh_,s_vn_,s_vd_,
        N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0); bh = tl.program_id(1)
    b = bh // H; h = bh % H; kv = h // G
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N); offs_d = tl.arange(0, D)
    nmask = offs_n < N
    k = tl.load(K + b*s_kb + kv*s_kh + offs_n[:,None]*s_kn + offs_d[None,:]*s_kd, mask=nmask[:,None], other=0.0)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    qs = scale * 1.4426950408889634
    for m in range(start_n, N, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M); mmask = offs_m < N
        qT = tl.load(Q + b*s_qb + h*s_qh + offs_d[:,None]*s_qd + offs_m[None,:]*s_qn, mask=mmask[None,:], other=0.0)
        do = tl.load(dO + b*s_dob + h*s_doh + offs_m[:,None]*s_don + offs_d[None,:]*s_dod, mask=mmask[:,None], other=0.0)
        lse = tl.load(LSE + b*s_lb + h*s_lh + offs_m*s_ln, mask=mmask, other=0.0)
        sT = tl.dot(k, qT) * qs
        sT = tl.where(offs_n[:,None] <= offs_m[None,:], sT, float("-inf"))
        pT = tl.exp2(sT - lse[None,:]*1.4426950408889634)
        dv += tl.dot(pT.to(do.dtype), do)
    tl.store(dVp + b*s_vb_ + h*s_vh_ + offs_n[:,None]*s_vn_ + offs_d[None,:]*s_vd_, dv, mask=nmask[:,None])

@triton.jit
def _dk_only_kernel(Q, K, V, dO, LSE, Delta, dKp,
        s_qb,s_qn,s_qh,s_qd, s_kb,s_kn,s_kh,s_kd, s_vb,s_vn,s_vh,s_vd, s_dob,s_don,s_doh,s_dod,
        s_lb,s_lh,s_ln, s_deb,s_deh,s_den, s_kb_,s_kh_,s_kn_,s_kd_,
        N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0); bh = tl.program_id(1)
    b = bh // H; h = bh % H; kv = h // G
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N); offs_d = tl.arange(0, D)
    nmask = offs_n < N
    k = tl.load(K + b*s_kb + kv*s_kh + offs_n[:,None]*s_kn + offs_d[None,:]*s_kd, mask=nmask[:,None], other=0.0)
    v = tl.load(V + b*s_vb + kv*s_vh + offs_n[:,None]*s_vn + offs_d[None,:]*s_vd, mask=nmask[:,None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    qs = scale * 1.4426950408889634
    for m in range(start_n, N, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M); mmask = offs_m < N
        qT = tl.load(Q + b*s_qb + h*s_qh + offs_d[:,None]*s_qd + offs_m[None,:]*s_qn, mask=mmask[None,:], other=0.0)
        doT = tl.load(dO + b*s_dob + h*s_doh + offs_d[:,None]*s_dod + offs_m[None,:]*s_don, mask=mmask[None,:], other=0.0)
        lse = tl.load(LSE + b*s_lb + h*s_lh + offs_m*s_ln, mask=mmask, other=0.0)
        delta = tl.load(Delta + b*s_deb + h*s_deh + offs_m*s_den, mask=mmask, other=0.0)
        sT = tl.dot(k, qT) * qs
        sT = tl.where(offs_n[:,None] <= offs_m[None,:], sT, float("-inf"))
        pT = tl.exp2(sT - lse[None,:]*1.4426950408889634)
        dpT = tl.dot(v, doT)
        dsT = (pT * (dpT - delta[None,:]) * scale).to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
    tl.store(dKp + b*s_kb_ + h*s_kh_ + offs_n[:,None]*s_kn_ + offs_d[None,:]*s_kd_, dk, mask=nmask[:,None])


# ----------------- grouped dK/dV kernel (all G query heads stacked) -----------------
# grid = (num_kv_blocks, B*HKV). One program owns a kv block for (b,kv) and processes
# all G query heads of the group at once by stacking them along the M(query) dimension.
# dk/dv are summed over the group directly -> no partial buffers, no reduce pass.
@triton.jit
def _dkdv_group_kernel(Q, K, V, dO, LSE, Delta, dK, dV,
                 stride_qb, stride_qn, stride_qh, stride_qd,
                 stride_kb, stride_kn, stride_kh, stride_kd,
                 stride_vb, stride_vn, stride_vh, stride_vd,
                 stride_dob, stride_don, stride_doh, stride_dod,
                 stride_lb, stride_lh, stride_ln,
                 stride_deb, stride_deh, stride_den,
                 stride_dkb, stride_dkn, stride_dkh, stride_dkd,
                 stride_dvb, stride_dvn, stride_dvh, stride_dvd,
                 N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
                 D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // HKV
    kv = bkv % HKV
    h0 = kv * G
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < N
    k = tl.load(K + b*stride_kb + kv*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd, mask=nmask[:,None], other=0.0)
    v = tl.load(V + b*stride_vb + kv*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd, mask=nmask[:,None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    GM: tl.constexpr = G * BLOCK_M
    r = tl.arange(0, GM)
    g_idx = r // BLOCK_M           # which head within group
    col_m = r % BLOCK_M            # position within q block
    qs = scale * 1.4426950408889634
    for m in range(start_n, N, BLOCK_M):
        m_idx = m + col_m                          # (GM,) query positions
        vmask = m_idx < N
        hh = h0 + g_idx                            # (GM,) head index
        # load qT_all (D, GM) and doT_all (D, GM) and do_all (GM, D)
        qbase = Q + b*stride_qb + hh[None,:]*stride_qh + m_idx[None,:]*stride_qn + offs_d[:,None]*stride_qd
        dobaseT = dO + b*stride_dob + hh[None,:]*stride_doh + m_idx[None,:]*stride_don + offs_d[:,None]*stride_dod
        dobase = dO + b*stride_dob + hh[:,None]*stride_doh + m_idx[:,None]*stride_don + offs_d[None,:]*stride_dod
        qT = tl.load(qbase, mask=vmask[None,:], other=0.0)      # (D, GM)
        doT = tl.load(dobaseT, mask=vmask[None,:], other=0.0)   # (D, GM)
        do = tl.load(dobase, mask=vmask[:,None], other=0.0)     # (GM, D)
        lse = tl.load(LSE + b*stride_lb + hh*stride_lh + m_idx*stride_ln, mask=vmask, other=0.0)
        delta = tl.load(Delta + b*stride_deb + hh*stride_deh + m_idx*stride_den, mask=vmask, other=0.0)
        sT = tl.dot(k, qT) * qs                                 # (BN, GM)
        causalT = offs_n[:, None] <= m_idx[None, :]
        sT = tl.where(causalT, sT, float("-inf"))
        pT = tl.exp2(sT - lse[None, :] * 1.4426950408889634)    # (BN, GM)
        dv += tl.dot(pT.to(do.dtype), do)                       # (BN, D)
        dpT = tl.dot(v, doT)                                    # (BN, GM)
        dsT = (pT * (dpT - delta[None, :]) * scale).to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))                         # (BN, D)
    tl.store(dK + b*stride_dkb + kv*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd, dk.to(dK.dtype.element_ty), mask=nmask[:,None])
    tl.store(dV + b*stride_dvb + kv*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd, dv.to(dV.dtype.element_ty), mask=nmask[:,None])


# ----------------- dQ kernel -----------------
@triton.jit
def _dq_kernel(Q, K, V, dO, LSE, Delta, dQ,
               stride_qb, stride_qn, stride_qh, stride_qd,
               stride_kb, stride_kn, stride_kh, stride_kd,
               stride_vb, stride_vn, stride_vh, stride_vd,
               stride_dob, stride_don, stride_doh, stride_dod,
               stride_lb, stride_lh, stride_ln,
               stride_deb, stride_deh, stride_den,
               stride_dqb, stride_dqn, stride_dqh, stride_dqd,
               N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
               D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kv = h // G
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptr = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do_ptr = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    qmask = offs_m[:, None] < N
    q = tl.load(q_ptr, mask=qmask, other=0.0)
    do = tl.load(do_ptr, mask=qmask, other=0.0)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_ln, mask=offs_m < N, other=0.0)
    delta = tl.load(Delta + b * stride_deb + h * stride_deh + offs_m * stride_den, mask=offs_m < N, other=0.0)
    lse2 = lse * 1.4426950408889634
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qs = (scale * 1.4426950408889634)
    # loop kv blocks up to (start_m + BLOCK_M)
    end_n = start_m + BLOCK_M
    for n in range(0, end_n, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        kmask = offs_n[:, None] < N
        # load k natural (BN,D) for dq matmul; vT (D,BN) for dp
        k_ptr = K + b * stride_kb + kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        vT_ptr = V + b * stride_vb + kv * stride_vh + offs_d[:, None] * stride_vd + offs_n[None, :] * stride_vn
        k = tl.load(k_ptr, mask=kmask, other=0.0)                # (BN,D)
        vT = tl.load(vT_ptr, mask=(offs_n[None, :] < N), other=0.0)  # (D,BN)
        s = tl.dot(q, tl.trans(k)) * qs                          # (BM,BN)
        causal = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal, s, float("-inf"))
        p = tl.exp2(s - lse2[:, None])
        dp = tl.dot(do, vT)                                      # (BM,BN)
        ds = (p * (dp - delta[:, None]) * scale).to(k.dtype)
        dq += tl.dot(ds, k)
    dq_ptr = dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
    tl.store(dq_ptr, dq.to(dQ.dtype.element_ty), mask=qmask)


# ----------------- grouped dQ kernel (all G query heads stacked) -----------------
# grid = (num_q_blocks, B*HKV). Program processes all G heads of a group for a q block,
# stacking heads along the M dimension -> loads K/V once per (q,kv) tile, bigger matmuls.
@triton.jit
def _dq_group_kernel(Q, K, V, dO, LSE, Delta, dQ,
               stride_qb, stride_qn, stride_qh, stride_qd,
               stride_kb, stride_kn, stride_kh, stride_kd,
               stride_vb, stride_vn, stride_vh, stride_vd,
               stride_dob, stride_don, stride_doh, stride_dod,
               stride_lb, stride_lh, stride_ln,
               stride_deb, stride_deh, stride_den,
               stride_dqb, stride_dqn, stride_dqh, stride_dqd,
               N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
               D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // HKV
    kv = bkv % HKV
    h0 = kv * G
    start_m = pid_m * BLOCK_M
    offs_d = tl.arange(0, D)
    GM: tl.constexpr = G * BLOCK_M
    r = tl.arange(0, GM)
    g_idx = r // BLOCK_M
    col_m = r % BLOCK_M
    m_idx = start_m + col_m                        # (GM,)
    hh = h0 + g_idx                                # (GM,)
    mmask = m_idx < N
    q = tl.load(Q + b*stride_qb + hh[:,None]*stride_qh + m_idx[:,None]*stride_qn + offs_d[None,:]*stride_qd, mask=mmask[:,None], other=0.0)   # (GM,D)
    do = tl.load(dO + b*stride_dob + hh[:,None]*stride_doh + m_idx[:,None]*stride_don + offs_d[None,:]*stride_dod, mask=mmask[:,None], other=0.0)
    lse2 = tl.load(LSE + b*stride_lb + hh*stride_lh + m_idx*stride_ln, mask=mmask, other=0.0) * 1.4426950408889634
    delta = tl.load(Delta + b*stride_deb + hh*stride_deh + m_idx*stride_den, mask=mmask, other=0.0)
    dq = tl.zeros([GM, D], dtype=tl.float32)
    qs = scale * 1.4426950408889634
    end_n = start_m + BLOCK_M
    for n in range(0, end_n, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        k = tl.load(K + b*stride_kb + kv*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd, mask=nmask[:,None], other=0.0)
        vT = tl.load(V + b*stride_vb + kv*stride_vh + offs_d[:,None]*stride_vd + offs_n[None,:]*stride_vn, mask=nmask[None,:], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qs                          # (GM,BN)
        causal = m_idx[:, None] >= offs_n[None, :]
        s = tl.where(causal, s, float("-inf"))
        p = tl.exp2(s - lse2[:, None])
        dp = tl.dot(do, vT)                                      # (GM,BN)
        ds = (p * (dp - delta[:, None]) * scale).to(k.dtype)
        dq += tl.dot(ds, k)                                      # (GM,D)
    tl.store(dQ + b*stride_dqb + hh[:,None]*stride_dqh + m_idx[:,None]*stride_dqn + offs_d[None,:]*stride_dqd,
             dq.to(dQ.dtype.element_ty), mask=mmask[:,None])


# ----------------- FUSED kernel: q-outer, dq local, dk/dv via atomics -----------------
@triton.jit
def _fused_kernel(Q, K, V, dO, LSE, Delta, dQ, dKf, dVf,
                  stride_qb, stride_qn, stride_qh, stride_qd,
                  stride_kb, stride_kn, stride_kh, stride_kd,
                  stride_vb, stride_vn, stride_vh, stride_vd,
                  stride_dob, stride_don, stride_doh, stride_dod,
                  stride_lb, stride_lh, stride_ln,
                  stride_deb, stride_deh, stride_den,
                  stride_dqb, stride_dqn, stride_dqh, stride_dqd,
                  stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                  stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                  N, scale, H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
                  D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kv = h // G
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < N
    q = tl.load(Q + b*stride_qb + h*stride_qh + offs_m[:,None]*stride_qn + offs_d[None,:]*stride_qd, mask=mmask[:,None], other=0.0)
    do = tl.load(dO + b*stride_dob + h*stride_doh + offs_m[:,None]*stride_don + offs_d[None,:]*stride_dod, mask=mmask[:,None], other=0.0)
    lse2 = tl.load(LSE + b*stride_lb + h*stride_lh + offs_m*stride_ln, mask=mmask, other=0.0) * 1.4426950408889634
    delta = tl.load(Delta + b*stride_deb + h*stride_deh + offs_m*stride_den, mask=mmask, other=0.0)
    qs = scale * 1.4426950408889634
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    end_n = start_m + BLOCK_M
    for n in range(0, end_n, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        k = tl.load(K + b*stride_kb + kv*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd, mask=nmask[:,None], other=0.0)
        v = tl.load(V + b*stride_vb + kv*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd, mask=nmask[:,None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qs                          # (BM,BN)
        causal = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal, s, float("-inf"))
        p = tl.exp2(s - lse2[:, None])                           # (BM,BN)
        dp = tl.dot(do, tl.trans(v))                             # (BM,BN)
        ds = p * (dp - delta[:, None]) * scale                   # (BM,BN) fp32
        dsb = ds.to(k.dtype)
        dq += tl.dot(dsb, k)                                     # (BM,D)
        # dk[n] += ds^T @ q ; dv[n] += p^T @ do
        dk_c = tl.dot(tl.trans(dsb), q)                          # (BN,D)
        dv_c = tl.dot(tl.trans(p).to(do.dtype), do)              # (BN,D)
        dk_ptr = dKf + b*stride_dkb + kv*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
        dv_ptr = dVf + b*stride_dvb + kv*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
        tl.atomic_add(dk_ptr, dk_c, mask=nmask[:,None])
        tl.atomic_add(dv_ptr, dv_c, mask=nmask[:,None])
    tl.store(dQ + b*stride_dqb + h*stride_dqh + offs_m[:,None]*stride_dqn + offs_d[None,:]*stride_dqd,
             dq.to(dQ.dtype.element_ty), mask=mmask[:,None])


def my_backward_fused(Q, K, V, O, LSE, dO, causal=True, BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2):
    B, N, H, D = Q.shape
    HKV = K.shape[2]; G = H // HKV
    scale = 1.0 / math.sqrt(D)
    Delta = compute_delta(O, dO)
    dQ = torch.empty_like(Q)
    dKf = torch.zeros((B, HKV, N, D), device=Q.device, dtype=torch.float32)
    dVf = torch.zeros((B, HKV, N, D), device=Q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _fused_kernel[grid](Q, K, V, dO, LSE, Delta, dQ, dKf, dVf,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dKf.stride(0), dKf.stride(1), dKf.stride(2), dKf.stride(3),
        dVf.stride(0), dVf.stride(1), dVf.stride(2), dVf.stride(3),
        N, scale, H, HKV, G, D, BLOCK_M, BLOCK_N, num_warps=num_warps, num_stages=num_stages)
    dK = dKf.permute(0, 2, 1, 3).contiguous().to(K.dtype)
    dV = dVf.permute(0, 2, 1, 3).contiguous().to(V.dtype)
    return dQ, dK, dV


# ----------------- fused group-reduce + cast (dKp (B,H,N,D) fp32 -> dK (B,N,HKV,D) bf16) -----------------
@triton.jit
def _reduce_kernel(dKp, dVp, dK, dV,
                   s_pb, s_ph, s_pn, s_pd,
                   s_ob, s_on, s_oh, s_od,
                   N, G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // HKV
    kv = bkv % HKV
    h0 = kv * G
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m[:, None] < N
    acck = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    accv = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for g in range(G):
        h = h0 + g
        pk = dKp + b*s_pb + h*s_ph + offs_m[:, None]*s_pn + offs_d[None, :]*s_pd
        pv = dVp + b*s_pb + h*s_ph + offs_m[:, None]*s_pn + offs_d[None, :]*s_pd
        acck += tl.load(pk, mask=mmask, other=0.0)
        accv += tl.load(pv, mask=mmask, other=0.0)
    ok = dK + b*s_ob + offs_m[:, None]*s_on + kv*s_oh + offs_d[None, :]*s_od
    ov = dV + b*s_ob + offs_m[:, None]*s_on + kv*s_oh + offs_d[None, :]*s_od
    tl.store(ok, acck.to(dK.dtype.element_ty), mask=mmask)
    tl.store(ov, accv.to(dV.dtype.element_ty), mask=mmask)


def compute_delta(O, dO):
    B, N, H, D = O.shape
    Delta = torch.empty((B, H, N), device=O.device, dtype=torch.float32)
    BLOCK_M = 128
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _delta_kernel[grid](O, dO, Delta,
                        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
                        Delta.stride(0), Delta.stride(1), Delta.stride(2),
                        N, H, D, BLOCK_M)
    return Delta


def my_backward(Q, K, V, O, LSE, dO, causal=True,
                BLOCK_M=64, BLOCK_N=128, num_warps=8, num_stages=2,
                cfg_kv=None, cfg_q=None):
    if cfg_kv is None:
        cfg_kv = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps, num_stages=num_stages)
    if cfg_q is None:
        cfg_q = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps, num_stages=num_stages)
    return _my_backward(Q, K, V, O, LSE, dO, causal, cfg_kv, cfg_q)


def _my_backward(Q, K, V, O, LSE, dO, causal, cfg_kv, cfg_q):
    # Q:(B,N,H,D) K,V:(B,N,HKV,D) O:(B,N,H,D) LSE:(B,H,N) dO:(B,N,H,D)
    B, N, H, D = Q.shape
    HKV = K.shape[2]
    G = H // HKV
    scale = 1.0 / math.sqrt(D)
    Delta = compute_delta(O, dO)
    dQ = torch.empty_like(Q)

    # partial dk/dv buffers in fp32, layout (B,H,N,D)
    dKp = torch.empty((B, H, N, D), device=Q.device, dtype=torch.float32)
    dVp = torch.empty((B, H, N, D), device=Q.device, dtype=torch.float32)
    grid_kv = (triton.cdiv(N, cfg_kv["BLOCK_N"]), B * H)
    _dkdv_split_kernel[grid_kv](Q, K, V, dO, LSE, Delta, dKp, dVp,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dKp.stride(0), dKp.stride(1), dKp.stride(2), dKp.stride(3),
        dVp.stride(0), dVp.stride(1), dVp.stride(2), dVp.stride(3),
        N, scale, H, HKV, G, D, cfg_kv["BLOCK_M"], cfg_kv["BLOCK_N"],
        num_warps=cfg_kv["num_warps"], num_stages=cfg_kv["num_stages"])
    # reduce over group and reshape to (B,N,HKV,D) via fused triton kernel
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    RBM = 128
    grid_r = (triton.cdiv(N, RBM), B * HKV)
    _reduce_kernel[grid_r](dKp, dVp, dK, dV,
        dKp.stride(0), dKp.stride(1), dKp.stride(2), dKp.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        N, G, HKV, D, RBM, num_warps=4)

    grid_q = (triton.cdiv(N, cfg_q["BLOCK_M"]), B * H)
    _dq_kernel[grid_q](Q, K, V, dO, LSE, Delta, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        N, scale, H, HKV, G, D, cfg_q["BLOCK_M"], cfg_q["BLOCK_N"],
        num_warps=cfg_q["num_warps"], num_stages=cfg_q["num_stages"])
    return dQ, dK, dV
