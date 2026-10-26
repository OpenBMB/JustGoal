"""Fused Kimi Delta Attention forward kernel (Triton, B200 / packed varlen).

Math (per head, per sequence), S in R^{DK x DV}:
    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t
with alpha_t = exp(-exp(A_log_h) * softplus(g_t + dt_bias_h))  (per K channel),
q/k L2-normalised and q scaled by 1/sqrt(DK).

Chunked form.  Inside a BC-token block let L_i = cumsum(log alpha) (inclusive)
and H = state at block entry.  With
    Ka = k * exp(L),  X = beta * (V - Ka H),
    M_ij  = sum_d k_i k_j exp(L_i - L_j)      (j <  i)
    Mq_ij = sum_d q_i k_j exp(L_i - L_j)      (j <= i)
    U     = (I + diag(beta) tril(M,-1))^{-1} X
we get
    O  = (q * exp(L) / sqrt(DK)) H + tril(Mq,0) U
    H' = Diag(exp(L_BC)) H + (k * exp(L_BC - L))^T U

M/Mq are factored as exp(L_i - r) * exp(r - L_j) with a per-channel reference
r = L_BC/2 (the value midpoint, which minimises max|L_i - r|).  Clamping
log(alpha) at -CAP bounds both factors by exp(BC*CAP/2) and costs at most
2*exp(-CAP) absolute error per entry of M.

The (sequence, head) recurrence is one program; DV can be split into BV-wide
slices to raise occupancy at the cost of recomputing the (small) intra-block
terms per slice.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kda_fwd_kernel(
    Q, K, V, G, BETA, ALOG, DTB, ST, CU, ORDER, OUT,
    H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    BC: tl.constexpr, NINV: tl.constexpr, CAP: tl.constexpr, LB: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, ABL: tl.constexpr, SBF16: tl.constexpr,
):
    pid = tl.program_id(0)
    nv: tl.constexpr = DV // BV
    task = tl.load(ORDER + pid // nv)
    vb = pid % nv
    n = task // H
    h = task % H
    t0 = tl.load(CU + n).to(tl.int32)
    t1 = tl.load(CU + n + 1).to(tl.int32)

    rk = tl.arange(0, DK)
    rv = vb * BV + tl.arange(0, BV)
    ri = tl.arange(0, BC)
    lower = ri[:, None] > ri[None, :]
    lower_eq = ri[:, None] >= ri[None, :]
    eye = (ri[:, None] == ri[None, :]).to(tl.float32)

    adec = tl.exp(tl.load(ALOG + h))
    dtb = tl.load(DTB + h * DK + rk)

    sp = ST + (n * H + h) * DK * DV + rk[:, None] * DV + rv[None, :]
    # SBF16 keeps the carried state in bfloat16: halves its register footprint and
    # removes the per-step fp32->bf16 copy needed to feed it to tl.dot.
    hs = tl.load(sp).to(tl.bfloat16) if SBF16 else tl.load(sp)

    for t in range(t0, t1, BC):
        rows = t + ri
        m = rows < t1
        offk = rows[:, None] * (H * DK) + h * DK + rk[None, :]
        offv = rows[:, None] * (H * DV) + h * DV + rv[None, :]
        q = tl.load(Q + offk, mask=m[:, None], other=0.0).to(tl.float32)
        k = tl.load(K + offk, mask=m[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + offv, mask=m[:, None], other=0.0).to(tl.float32)
        bb = tl.load(BETA + rows * H + h, mask=m, other=0.0).to(tl.float32)

        q = q * tl.rsqrt(tl.sum(q * q, 1)[:, None] + EPS)
        k = k * tl.rsqrt(tl.sum(k * k, 1)[:, None] + EPS)

        if ABL == 2:
            la = tl.where(m[:, None], -0.1 + 0.0 * k, 0.0)
        else:
            g = tl.load(G + offk, mask=m[:, None], other=0.0).to(tl.float32)
            x = g + dtb[None, :]
            if LB == 0.0:
                # softplus gate: -exp(A_log) * softplus(x), clamped at -CAP
                spx = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
                la = tl.maximum(-adec * spx, -CAP)
            else:
                # safe gate: LB * sigmoid(exp(A_log) * x), already in (LB, 0)
                la = LB * tl.sigmoid(adec * x)
            la = tl.where(m[:, None], la, 0.0)

        Lb = tl.cumsum(la, 0)
        Lend = tl.sum(la, 0)
        Lref = Lend * 0.5
        er = tl.exp(Lb - Lref[None, :])
        ec = tl.exp(Lref[None, :] - Lb)
        eref = tl.exp(Lref)                                # = exp(Lend - Lref)

        Ka = (k * (er * eref[None, :]))     # k * exp(Lb)
        Qa = (q * (er * (eref[None, :] * SCALE)))
        Ke = (k * (ec * eref[None, :]))     # k * exp(Lend - Lb)

        if ABL == 3:
            Tm = eye
            mq = eye
        else:
            Kr = k * er
            Kc = tl.trans(k * ec)
            mm = tl.dot(Kr, Kc, input_precision="tf32x3")
            mq = tl.dot(q * (er * SCALE), Kc, input_precision="tf32x3")
            A = tl.where(lower, -bb[:, None] * mm, 0.0)
            if ABL == 1:
                Tm = eye + A
            else:
                S = eye + A
                P = A
                for _ in tl.static_range(NINV):
                    P = tl.dot(P, P, input_precision="tf32x3")
                    S = S + tl.dot(P, S, input_precision="tf32x3")
                Tm = S

        hb = hs.to(tl.float32)
        X = bb[:, None] * (v - tl.dot(Ka, hb, input_precision="tf32x3"))
        Ub = tl.dot(Tm, X, input_precision="tf32x3")
        o = tl.dot(tl.where(lower_eq, mq, 0.0), Ub, tl.dot(Qa, hb, input_precision="tf32x3"), input_precision="tf32x3")
        tl.store(OUT + offv, o.to(tl.bfloat16), mask=m[:, None])

        hn = tl.dot(tl.trans(Ke).to(tl.float32), Ub.to(tl.float32), tl.exp(Lend)[:, None] * hb, input_precision="tf32x3")
        hs = hn.to(tl.bfloat16) if SBF16 else hn

    tl.store(sp, hs.to(tl.float32))


_ORDER_CACHE = {}


def _order(cu, H):
    key = (cu.data_ptr(), int(cu[-1]), H)
    o = _ORDER_CACHE.get(key)
    if o is None:
        lens = (cu[1:] - cu[:-1]).tolist()
        idx = sorted(range(len(lens)), key=lambda i: -lens[i])
        o = torch.tensor([n * H + h for n in idx for h in range(H)],
                         dtype=torch.int32, device=cu.device)
        _ORDER_CACHE[key] = o
    return o


def kda_fwd(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens, out=None,
            BC=32, CAP=5.0, BV=None, lower_bound=-5.0, scale=None, eps=1e-6,
            num_warps=8, num_stages=2, abl=0, sbf16=False):
    """In-place state update; returns the output tensor (bf16, shape of v)."""
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    assert B == 1 and state.is_contiguous() and q.is_contiguous()
    if scale is None:
        scale = DK ** -0.5
    if BV is None:
        BV = DV
    if out is None:
        out = torch.empty_like(v)
    order = _order(cu_seqlens, H)
    kk = kda_fwd_kernel[(order.numel() * (DV // BV),)](
        q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens, order, out,
        H=H, DK=DK, DV=DV, BV=BV, BC=BC, NINV=max(BC.bit_length() - 2, 0),
        CAP=CAP, LB=(0.0 if lower_bound is None else float(lower_bound)),
        SCALE=scale, EPS=eps, ABL=abl, SBF16=sbf16,
        num_warps=num_warps, num_stages=num_stages,
    )
    kda_fwd.last = kk
    return out
