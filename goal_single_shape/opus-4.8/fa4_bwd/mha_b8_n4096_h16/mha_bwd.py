import torch, math
import triton
import triton.language as tl

# ------------------------- Forward (produces o, lse) -------------------------
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE,
                sqb, sqn, sqh, sqd,
                skb, skn, skh, skd,
                svb, svn, svh, svd,
                sob, son, soh, sod,
                slb, slh, sln,
                scale, N, H,
                BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = i * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q_ptr = Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd
    q = tl.load(q_ptr, mask=offs_m[:, None] < N, other=0.0)
    m_i = tl.full([BM], -float('inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    end = (i + 1) * BM  # causal: keys up to this
    for start in range(0, end, BN):
        offs_n = start + tl.arange(0, BN)
        k_ptr = K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd
        k = tl.load(k_ptr, mask=offs_n[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float('inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, 1)
        v_ptr = V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
        v = tl.load(v_ptr, mask=offs_n[:, None] < N, other=0.0)
        acc = acc * corr[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_ptr = O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :] * sod
    tl.store(o_ptr, o.to(O.dtype.element_ty), mask=offs_m[:, None] < N)
    lse_ptr = LSE + b * slb + h * slh + offs_m * sln
    tl.store(lse_ptr, lse, mask=offs_m < N)


def flash_fwd(q, k, v, scale, BM=128, BN=128):
    B, N, H, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BM), B * H)
    _fwd_kernel[grid](q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        scale, N, H, BM=BM, BN=BN, D=D, num_warps=4, num_stages=2)
    return o, lse


# ------------------------- Preprocess: delta = sum(o*do) -------------------------
@triton.jit
def _pre_kernel(O, DO, Delta,
                sob, son, soh, sod,
                sdb, sdn, sdh, sdd,
                sD_b, sD_h, sD_n,
                N, H, BM: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = i * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    o = tl.load(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :] * sod,
                mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :] * sdd,
                 mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, 1)
    tl.store(Delta + b * sD_b + h * sD_h + offs_m * sD_n, delta, mask=offs_m < N)


# ------------------------- dK/dV kernel (parallel over kv blocks) -------------------------
@triton.jit
def _dkdv_kernel(Q, K, V, DO, LSE, Delta, DK, DV,
                 sqb, sqn, sqh, sqd,
                 skb, skn, skh, skd,
                 svb, svn, svh, svd,
                 sdob, sdon, sdoh, sdod,
                 slb, slh, sln,
                 sDb, sDh, sDn,
                 sdkb, sdkn, sdkh, sdkd,
                 sdvb, sdvn, sdvh, sdvd,
                 scale, N, H,
                 BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = j * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                mask=offs_n[:, None] < N, other=0.0)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                mask=offs_n[:, None] < N, other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    start_i = (j * BN) // BM
    for i in range(start_i, tl.cdiv(N, BM)):
        offs_m = i * BM + tl.arange(0, BM)
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd,
                    mask=offs_m[:, None] < N, other=0.0)
        do = tl.load(DO + b * sdob + h * sdoh + offs_m[:, None] * sdon + offs_d[None, :] * sdod,
                     mask=offs_m[:, None] < N, other=0.0)
        lse = tl.load(LSE + b * slb + h * slh + offs_m * sln, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + b * sDb + h * sDh + offs_m * sDn, mask=offs_m < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale       # (BM, BN)
        p = tl.exp(s - lse[:, None])
        p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)   # (BN, D)
        dp = tl.dot(do, tl.trans(v))                  # (BM, BN)
        ds = p * (dp - delta[:, None])                # (BM, BN)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * scale  # (BN, D)
    tl.store(DK + b * sdkb + h * sdkh + offs_n[:, None] * sdkn + offs_d[None, :] * sdkd,
             dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < N)
    tl.store(DV + b * sdvb + h * sdvh + offs_n[:, None] * sdvn + offs_d[None, :] * sdvd,
             dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N)


# ------------------------- dQ kernel (parallel over q blocks) -------------------------
@triton.jit
def _dq_kernel(Q, K, V, DO, LSE, Delta, DQ,
               sqb, sqn, sqh, sqd,
               skb, skn, skh, skd,
               svb, svn, svh, svd,
               sdob, sdon, sdoh, sdod,
               slb, slh, sln,
               sDb, sDh, sDn,
               sdqb, sdqn, sdqh, sdqd,
               scale, N, H,
               BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = i * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd,
                mask=offs_m[:, None] < N, other=0.0)
    do = tl.load(DO + b * sdob + h * sdoh + offs_m[:, None] * sdon + offs_d[None, :] * sdod,
                 mask=offs_m[:, None] < N, other=0.0)
    lse = tl.load(LSE + b * slb + h * slh + offs_m * sln, mask=offs_m < N, other=0.0)
    delta = tl.load(Delta + b * sDb + h * sDh + offs_m * sDn, mask=offs_m < N, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    end = (i + 1) * BM
    for start in range(0, end, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=offs_n[:, None] < N, other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=offs_n[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq *= scale
    tl.store(DQ + b * sdqb + h * sdqh + offs_m[:, None] * sdqn + offs_d[None, :] * sdqd,
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_bwd(q, k, v, o, do, lse, scale,
              kv_cfg=(64, 128, 8, 2), q_cfg=(64, 64, 8, 2)):
    B, N, H, D = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    _pre_kernel[(triton.cdiv(N, 128), B * H)](o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N, H, BM=128, D=D, num_warps=4)
    BM, BN, num_warps, num_stages = kv_cfg
    grid_kv = (triton.cdiv(N, BN), B * H)
    _dkdv_kernel[grid_kv](q, k, v, do, lse, delta, dk, dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        scale, N, H, BM=BM, BN=BN, D=D, num_warps=num_warps, num_stages=num_stages)
    BM, BN, num_warps, num_stages = q_cfg
    grid_q = (triton.cdiv(N, BM), B * H)
    _dq_kernel[grid_q](q, k, v, do, lse, delta, dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        scale, N, H, BM=BM, BN=BN, D=D, num_warps=num_warps, num_stages=num_stages)
    return dq, dk, dv
