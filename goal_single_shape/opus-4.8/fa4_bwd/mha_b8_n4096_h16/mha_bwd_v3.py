import torch, math
import triton
import triton.language as tl
from mha_bwd import _pre_kernel

# v3: single load per tensor + tl.trans on *loaded inputs* only (never on dot-derived
# values, which miscompiles under pipelining on sm_100). Accumulator dots consume
# transposed-score tensors directly (no trans of p/ds).

@triton.jit
def _dkdv_v3(Q, K, V, DO, LSE, Delta, DK, DV,
             sqb, sqn, sqh, sqd, skb, skn, skh, skd,
             svb, svn, svh, svd, sdob, sdon, sdoh, sdod,
             slb, slh, sln, sDb, sDh, sDn,
             sdkb, sdkn, sdkh, sdkd, sdvb, sdvn, sdvh, sdvd,
             scale, N, H, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    j = tl.program_id(0); bh = tl.program_id(1)
    b = bh // H; h = bh % H
    offs_n = j * BN + tl.arange(0, BN); offs_d = tl.arange(0, D)
    k = tl.load(K + b*skb+h*skh + offs_n[:,None]*skn+offs_d[None,:]*skd, mask=offs_n[:,None]<N, other=0.0)
    v = tl.load(V + b*svb+h*svh + offs_n[:,None]*svn+offs_d[None,:]*svd, mask=offs_n[:,None]<N, other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for i in range(j*BN//BM, tl.cdiv(N, BM)):
        offs_m = i*BM + tl.arange(0, BM); mrow = offs_m < N
        q = tl.load(Q + b*sqb+h*sqh + offs_m[:,None]*sqn+offs_d[None,:]*sqd, mask=mrow[:,None], other=0.0)
        do = tl.load(DO + b*sdob+h*sdoh + offs_m[:,None]*sdon+offs_d[None,:]*sdod, mask=mrow[:,None], other=0.0)
        lse = tl.load(LSE + b*slb+h*slh + offs_m*sln, mask=mrow, other=0.0)
        delta = tl.load(Delta + b*sDb+h*sDh + offs_m*sDn, mask=mrow, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale          # (BN,BM)
        pT = tl.exp(sT - lse[None, :])
        if i * BM < (j + 1) * BN:                      # only diagonal-overlapping tiles need mask
            pT = tl.where(offs_n[:, None] <= offs_m[None, :], pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)            # (BN,D)
        dpT = tl.dot(v, tl.trans(do))                # (BN,BM)
        dsT = (pT * (dpT - delta[None, :])).to(q.dtype)
        dk += tl.dot(dsT, q)                         # (BN,D)
    tl.store(DK + b*sdkb+h*sdkh + offs_n[:,None]*sdkn+offs_d[None,:]*sdkd,
             (dk*scale).to(DK.dtype.element_ty), mask=offs_n[:,None]<N)
    tl.store(DV + b*sdvb+h*sdvh + offs_n[:,None]*sdvn+offs_d[None,:]*sdvd,
             dv.to(DV.dtype.element_ty), mask=offs_n[:,None]<N)


@triton.jit
def _dq_v3(Q, K, V, DO, LSE, Delta, DQ,
           sqb, sqn, sqh, sqd, skb, skn, skh, skd,
           svb, svn, svh, svd, sdob, sdon, sdoh, sdod,
           slb, slh, sln, sDb, sDh, sDn, sdqb, sdqn, sdqh, sdqd,
           scale, N, H, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0); bh = tl.program_id(1)
    b = bh // H; h = bh % H
    offs_m = i*BM + tl.arange(0, BM); offs_d = tl.arange(0, D); mrow = offs_m < N
    q = tl.load(Q + b*sqb+h*sqh + offs_m[:,None]*sqn+offs_d[None,:]*sqd, mask=mrow[:,None], other=0.0)
    do = tl.load(DO + b*sdob+h*sdoh + offs_m[:,None]*sdon+offs_d[None,:]*sdod, mask=mrow[:,None], other=0.0)
    lse = tl.load(LSE + b*slb+h*slh + offs_m*sln, mask=mrow, other=0.0)
    delta = tl.load(Delta + b*sDb+h*sDh + offs_m*sDn, mask=mrow, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    for start in range(0, (i+1)*BM, BN):
        offs_n = start + tl.arange(0, BN); mcol = offs_n < N
        k = tl.load(K + b*skb+h*skh + offs_n[:,None]*skn+offs_d[None,:]*skd, mask=mcol[:,None], other=0.0)
        v = tl.load(V + b*svb+h*svh + offs_n[:,None]*svn+offs_d[None,:]*svd, mask=mcol[:,None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale            # (BM,BN)
        p = tl.exp(s - lse[:, None])
        if start + BN > i * BM:                        # only diagonal-overlapping tiles need mask
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))                  # (BM,BN)
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq += tl.dot(ds, k)                           # (BM,D)
    tl.store(DQ + b*sdqb+h*sdqh + offs_m[:,None]*sdqn+offs_d[None,:]*sdqd,
             (dq*scale).to(DQ.dtype.element_ty), mask=mrow[:,None])


def flash_bwd_v3(q, k, v, o, do, lse, scale, kv_cfg=(64,128,8,3), q_cfg=(64,64,8,2)):
    B, N, H, D = q.shape
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    _pre_kernel[(triton.cdiv(N,128), B*H)](o, do, delta, *o.stride(), *do.stride(), *delta.stride(),
        N, H, BM=128, D=D, num_warps=4)
    BM,BN,nw,ns = kv_cfg
    _dkdv_v3[(triton.cdiv(N,BN), B*H)](q,k,v,do,lse,delta,dk,dv, *q.stride(),*k.stride(),*v.stride(),
        *do.stride(),*lse.stride(),*delta.stride(),*dk.stride(),*dv.stride(),
        scale,N,H,BM=BM,BN=BN,D=D,num_warps=nw,num_stages=ns)
    BM,BN,nw,ns = q_cfg
    _dq_v3[(triton.cdiv(N,BM), B*H)](q,k,v,do,lse,delta,dq, *q.stride(),*k.stride(),*v.stride(),
        *do.stride(),*lse.stride(),*delta.stride(),*dq.stride(),
        scale,N,H,BM=BM,BN=BN,D=D,num_warps=nw,num_stages=ns)
    return dq, dk, dv
