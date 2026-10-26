"""Fused single-kernel GDN forward for one short sequence (T <= BT).

Math (derived from the gated delta rule recurrence, k-last state S[v,k]):

    S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

Let A_t = exp(cumsum(g)_t) and w'_t = beta_t (v_t - exp(g_t) S_{t-1} k_t).
Substituting S_t = A_t (S_0 + sum_{i<=t} (w'_i/A_i) k_i^T) turns the scan into

    (I + L) W' = R,   L[t,i] = beta_t (k_t.k_i) A_t/A_i   (i < t)
                      R[t]   = beta_t (v_t - A_t S_0 k_t)
    o_t      = A_t S_0 q_t + sum_{i<=t} (q_t.k_i) (A_t/A_i) W'_i
    S_T      = A_T S_0 + sum_t (A_T/A_t) W'_t k_t^T

so the whole sequence is one chunk: two 32x32 gram matrices, one 32x32 unit
lower-triangular inverse (Neumann doubling, exact since L is nilpotent), and a
few small matmuls. Every v-dimension row is independent, so the grid is
(v_head, v_block) and no block ever needs to talk to another one.

l2norm trick: every matmul consumes the *raw* bf16 q/k (so bf16 tensor cores
are bit-exact on them) and the 1/||q||, 1/||k|| factors are folded into the
32-wide vectors / 32x32 matrices afterwards.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot(a, b, DT: tl.constexpr, IP: tl.constexpr):
    if DT == "bf16":
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)
    if DT == "fp16":
        return tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)
    return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision=IP)


@triton.jit
def _gdn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr, s0_ptr, o_ptr, sn_ptr,
    T, scale,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, NLEV: tl.constexpr,
    DTB: tl.constexpr, DTS: tl.constexpr, IP: tl.constexpr,
):
    hv = tl.program_id(0)
    hk = hv // (HV // HK)
    vv = tl.program_id(1) * BV + tl.arange(0, BV)

    t = tl.arange(0, BT)
    d = tl.arange(0, D)
    mt = t < T

    # ---- raw q, k (BT,D) in their native dtype; norms folded in later
    qb = tl.load(q_ptr + t[:, None] * (HK * D) + hk * D + d[None, :],
                 mask=mt[:, None], other=0.0)
    kb = tl.load(k_ptr + t[:, None] * (HK * D) + hk * D + d[None, :],
                 mask=mt[:, None], other=0.0)
    qf = qb.to(tl.float32)
    kf = kb.to(tl.float32)
    nq = tl.rsqrt(tl.sum(qf * qf, 1) + 1e-6) * scale
    nk = tl.rsqrt(tl.sum(kf * kf, 1) + 1e-6)

    # ---- gates: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
    av = tl.load(a_ptr + t * HV + hv, mask=mt, other=0.0).to(tl.float32)
    bvv = tl.load(b_ptr + t * HV + hv, mask=mt, other=0.0).to(tl.float32)
    x = av + tl.load(dtb_ptr + hv).to(tl.float32)
    sfp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
    g = tl.where(mt, -tl.exp(tl.load(alog_ptr + hv).to(tl.float32)) * sfp, 0.0)
    beta = tl.where(mt, tl.sigmoid(bvv), 0.0)
    gc = tl.cumsum(g, 0)
    gtot = tl.sum(g, 0)

    # ---- pairwise decay A_t/A_i (only the causal half is ever used)
    dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
    kk = _dot(kb, tl.trans(kb), DTB, IP) * (nk[:, None] * nk[None, :])
    qk = _dot(qb, tl.trans(kb), DTB, IP) * (nq[:, None] * nk[None, :])
    neg_l = tl.where(t[:, None] > t[None, :], -beta[:, None] * kk * dec, 0.0)
    pqk = tl.where(t[:, None] >= t[None, :], qk * dec, 0.0)

    # ---- Tinv = (I + L)^-1 = sum_j (-L)^j  via prod_j (I + (-L)^(2^j))
    tinv = tl.where(t[:, None] == t[None, :], 1.0, 0.0) + neg_l
    pw = neg_l
    for _ in tl.static_range(NLEV):
        pw = _dot(pw, pw, DTS, IP)
        tinv += _dot(tinv, pw, DTS, IP)

    # ---- state block (BV,D), k-last
    s0 = tl.load(s0_ptr + hv * (D * D) + vv[:, None] * D + d[None, :])
    s0t = tl.trans(s0)
    ks0 = _dot(kb, s0t, DTB, IP) * nk[:, None]
    qs0 = _dot(qb, s0t, DTB, IP) * nq[:, None]

    va = tl.load(v_ptr + t[:, None] * (HV * D) + hv * D + vv[None, :],
                 mask=mt[:, None], other=0.0).to(tl.float32)
    acc = tl.exp(gc)
    rhs = beta[:, None] * (va - acc[:, None] * ks0)
    w = _dot(tinv, rhs, DTS, IP)

    o = acc[:, None] * qs0 + _dot(pqk, w, DTS, IP)
    tl.store(o_ptr + t[:, None] * (HV * D) + hv * D + vv[None, :],
             o.to(o_ptr.dtype.element_ty), mask=mt[:, None])

    wd = w * (tl.exp(gtot - gc) * nk)[:, None]
    sn = tl.exp(gtot) * s0 + _dot(tl.trans(wd), kb, DTB, IP)
    tl.store(sn_ptr + hv * (D * D) + vv[:, None] * D + d[None, :], sn)


def gdn_fwd(q, k, v, a, b, A_log, dt_bias, state, cu_seqlens=None, scale=None,
            out=None, new_state=None, BV=16, num_warps=4, num_stages=1,
            DTB="bf16", DTS="bf16", IP="tf32", BT=32, NLEV=None):
    """q,k: (T,HK,D) bf16; v: (T,HV,D) bf16; a,b: (T,HV); state: (1,HV,D,D) fp32.

    DTB: dtype of the K=128 matmuls (gram + state), DTS: dtype of the 32x32 ones.
    """
    T, HK, D = q.shape
    HV = v.shape[1]
    if scale is None:
        scale = D ** -0.5
    if out is None:
        out = torch.empty(T, HV, D, device=q.device, dtype=q.dtype)
    if new_state is None:
        new_state = torch.empty_like(state)
    if NLEV is None:
        NLEV = (BT - 1).bit_length()      # exact: covers nilpotency index BT
    _gdn_fwd_kernel[(HV, D // BV)](
        q, k, v, a, b, A_log, dt_bias, state, out, new_state,
        T, scale, HK=HK, HV=HV, D=D, BT=BT, BV=BV, NLEV=NLEV,
        DTB=DTB, DTS=DTS, IP=IP, num_warps=num_warps, num_stages=num_stages,
    )
    return out, new_state
