"""Triton GDN forward for packed variable-length sequences (3 kernel launches).

Chunked gated-delta-rule with the UT (WY) transform.  Per v-head, per chunk of C
tokens, with gc_i = cumsum of log-decay inside the chunk and Gamma_i = exp(gc_i):

    u_i = beta_i (v_i - S^T (Gamma_i k_i)) - beta_i sum_{j<i} (Gamma_i/Gamma_j)(k_i.k_j) u_j
  =>  U   = u0 - w S            u0 = Tm V,  w = Tm K_Gamma,
                                Tm  = (I + beta*Mtilde)^{-1} diag(beta)
      o_i = S^T (Gamma_i q_i) + sum_{j<=i} (Gamma_i/Gamma_j)(q_i.k_j) u_j
      S'  = gam S + KZ^T U      KZ_i = exp(gc_last - gc_i) k_i,  gam = exp(gc_last)

Kernel 1 (prep) is parallel over all (block, head) and emits u0, w, KZ, gc, gam.
Kernel 2 (scan) runs the state recurrence; every column of S (the v dimension)
is independent, so it is split over program_id(2).  It only checkpoints S every
SEG blocks.  Kernel 3 (out) replays SEG blocks from a checkpoint and emits the
outputs -- fully parallel, so the state traffic stays small.

CHAIN MERGING (M>1).  The UT transform costs O(C^3) per chunk, so C stays 64,
but the serial chain of the scan is one step per *block*.  M adjacent chunks are
therefore merged inside prep into one exactly equivalent M*C-row transition: for
two chunks, U_1 = u0_1 - w_1 S_1 with S_1 = (gam_0 I - KZ_0^T w_0) S_0 + KZ_0^T u0_0
rewrites to U_1 = (u0_1 - G u0_0) - (gam_0 w_1 - G w_0) S_0 with G = w_1 KZ_0^T.
Stacking both row blocks gives a 2C-row transition in the very same form, with
*identical* total scan FLOPs (2*M*C*K*BV either way) but half the chain depth.
The merge itself costs 3 extra C-sized matmuls and runs in the parallel prep.

Block slots are padded per sequence to a multiple of SEG and the intermediates
are block-aligned, so both hot loops have a static trip count and need no
in-loop conditionals (padding blocks get beta=0/k=0 => u0=w=KZ=0, gam=1, i.e.
an exact identity transition).  Every exp() argument is <= 0 (decay ratios are
applied after the matmul), so nothing can overflow.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64
NS_STEPS = 6  # doubling steps; exact for a strictly-lower CxC matrix when 2**NS >= C


@triton.jit
def _l2norm_rows(P_ptr, t0, clen, hk, NH: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    r = tl.arange(0, C)
    d = tl.arange(0, D)
    x = tl.load(P_ptr + (t0 + r[:, None]) * (NH * D) + hk * D + d[None, :],
                mask=(r < clen)[:, None], other=0.0).to(tl.float32)
    return x  # The public contract supplies Q/K; do not renormalize them.


@triton.jit
def _chunk_ut(K_ptr, V_ptr, A_ptr, B_ptr, ALOG_ptr, DTB_ptr, t0, clen, h, hk,
              NHQ: tl.constexpr, NHV: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
              NS: tl.constexpr):
    """UT transform of one C-token chunk -> (u0, w, kz, gc, gl), all fp32."""
    r = tl.arange(0, C)
    d = tl.arange(0, D)
    m = r < clen

    av = tl.load(A_ptr + (t0 + r) * NHV + h, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(B_ptr + (t0 + r) * NHV + h, mask=m, other=0.0).to(tl.float32)
    x = av + tl.load(DTB_ptr + h).to(tl.float32)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    gv = tl.where(m, -tl.exp(tl.load(ALOG_ptr + h).to(tl.float32)) * sp, 0.0)
    gc = tl.cumsum(gv, 0)
    gl = tl.min(gc, 0)
    beta = tl.where(m, 1.0 / (1.0 + tl.exp(-bv)), 0.0)

    kn = _l2norm_rows(K_ptr, t0, clen, hk, NHQ, C, D)
    knb = kn.to(tl.bfloat16)
    mk = tl.dot(knb, tl.trans(knb))
    dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
    p = tl.where(r[:, None] > r[None, :], -mk * dec * beta[:, None], 0.0)

    # X = (I - P)^{-1} = prod_{j<NS} (I + P^(2^j))   (P strictly lower -> exact)
    x_inv = (r[:, None] == r[None, :]).to(tl.float32) + p
    pj = p
    for _ in tl.static_range(NS - 1):
        pjb = pj
        pj = tl.dot(pjb, pjb, input_precision="tf32x3")
        x_inv = x_inv + tl.dot(x_inv, pj, input_precision="tf32x3")

    tm = (x_inv * beta[None, :]).to(tl.bfloat16)
    vv = tl.load(V_ptr + (t0 + r[:, None]) * (NHV * D) + h * D + d[None, :],
                 mask=m[:, None], other=0.0)
    u0 = tl.dot(tm, vv)
    w = tl.dot(tm, (kn * tl.exp(gc)[:, None]).to(tl.bfloat16))
    kz = kn * tl.exp(gl - gc)[:, None]
    return u0, w, kz, gc, gl, kn


@triton.jit
def prep_kernel(
    K_ptr, V_ptr, A_ptr, B_ptr, ALOG_ptr, DTB_ptr, Q_ptr,
    U0_ptr, W_ptr, KZ_ptr, GC_ptr, GAM_ptr, AQ_ptr, QG_ptr, BT0_ptr, BLEN_ptr, SCALE,
    NHQ: tl.constexpr, NHV: tl.constexpr, GRP: tl.constexpr,
    C: tl.constexpr, D: tl.constexpr, NS: tl.constexpr, M: tl.constexpr,
    FUSE_ATTN: tl.constexpr,
):
    pb = tl.program_id(0)
    h = tl.program_id(1)
    t0 = tl.load(BT0_ptr + pb)
    blen = tl.load(BLEN_ptr + pb)
    hk = h // GRP
    r = tl.arange(0, C)
    d = tl.arange(0, D)
    base = pb * (C * M)

    u00, w0, kz0, gc0, gl0, kn0 = _chunk_ut(K_ptr, V_ptr, A_ptr, B_ptr, ALOG_ptr, DTB_ptr,
                                       t0, blen, h, hk, NHQ, NHV, C, D, NS)
    if M == 1:
        oo = (base + r[:, None]) * (NHV * D) + h * D + d[None, :]
        tl.store(U0_ptr + oo, u00.to(tl.bfloat16))
        tl.store(W_ptr + oo, w0.to(tl.bfloat16))
        tl.store(KZ_ptr + oo, kz0.to(tl.bfloat16))
        tl.store(GC_ptr + (base + r) * NHV + h, gc0)
        tl.store(GAM_ptr + pb * NHV + h, tl.exp(gl0))
        if FUSE_ATTN:
            qn = _l2norm_rows(Q_ptr, t0, blen, hk, NHQ, C, D) * SCALE
            aq = tl.dot(qn.to(tl.bfloat16), tl.trans(kn0.to(tl.bfloat16)))
            dec0 = tl.exp(tl.minimum(gc0[:, None] - gc0[None, :], 0.0))
            aq = tl.where(r[:, None] >= r[None, :], aq * dec0, 0.0)
            tl.store(AQ_ptr + pb * (NHV * C * C) + h * (C * C)
                     + r[:, None] * C + r[None, :], aq.to(tl.bfloat16))
            tl.store(QG_ptr + oo, (qn * tl.exp(gc0)[:, None]).to(tl.bfloat16))
    else:
        u01, w1, kz1, gc1, gl1, _kn1 = _chunk_ut(K_ptr, V_ptr, A_ptr, B_ptr, ALOG_ptr, DTB_ptr,
                                           t0 + C, blen - C, h, hk, NHQ, NHV, C, D, NS)
        g0 = tl.exp(gl0)
        g1 = tl.exp(gl1)
        gmat = tl.dot(w1.to(tl.bfloat16), tl.trans(kz0.to(tl.bfloat16))).to(tl.bfloat16)
        w1m = g0 * w1 - tl.dot(gmat, w0.to(tl.bfloat16))
        u01m = u01 - tl.dot(gmat, u00.to(tl.bfloat16))

        oo = (base + r[:, None]) * (NHV * D) + h * D + d[None, :]
        tl.store(U0_ptr + oo, u00.to(tl.bfloat16))
        tl.store(W_ptr + oo, w0.to(tl.bfloat16))
        tl.store(KZ_ptr + oo, (g1 * kz0).to(tl.bfloat16))
        tl.store(GC_ptr + (base + r) * NHV + h, gc0)
        oo1 = oo + C * (NHV * D)
        tl.store(U0_ptr + oo1, u01m.to(tl.bfloat16))
        tl.store(W_ptr + oo1, w1m.to(tl.bfloat16))
        tl.store(KZ_ptr + oo1, kz1.to(tl.bfloat16))
        tl.store(GC_ptr + (base + C + r) * NHV + h, gc1 + gl0)
        tl.store(GAM_ptr + pb * NHV + h, g0 * g1)


@triton.jit
def scan_kernel(
    U0_ptr, W_ptr, KZ_ptr, GAM_ptr, S0_ptr, CKPT_ptr, SNEW_ptr, SCB_ptr, SNSEG_ptr, SBASE_ptr,
    NHV: tl.constexpr, MC: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    SEG: tl.constexpr, STAGES: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    cb = tl.load(SCB_ptr + n)
    nseg = tl.load(SNSEG_ptr + n)
    sbase = tl.load(SBASE_ptr + n)
    r = tl.arange(0, MC)
    dk = tl.arange(0, D)
    dv = vb * BV + tl.arange(0, BV)

    so = n * (NHV * D * D) + h * (D * D) + dv[None, :] * D + dk[:, None]
    s = tl.load(S0_ptr + so)  # (D, BV) slice of S_kv

    for g in range(0, nseg):
        tl.store(CKPT_ptr + (sbase + g) * (NHV * D * D) + h * (D * D)
                 + dk[:, None] * D + dv[None, :], s)
        for j in tl.range(0, SEG, num_stages=STAGES):
            pb = cb + g * SEG + j
            row = (pb * MC + r[:, None]) * (NHV * D) + h * D
            wv = tl.load(W_ptr + row + dk[None, :])
            u0 = tl.load(U0_ptr + row + dv[None, :])
            kz = tl.load(KZ_ptr + row + dk[None, :])
            gam = tl.load(GAM_ptr + pb * NHV + h)
            u = u0.to(tl.float32) - tl.dot(wv, s.to(tl.bfloat16))
            s = gam * s + tl.dot(tl.trans(kz), u.to(tl.bfloat16))

    tl.store(SNEW_ptr + so, s)


@triton.jit
def out_kernel(
    Q_ptr, K_ptr, U0_ptr, W_ptr, KZ_ptr, GC_ptr, GAM_ptr, CKPT_ptr, O_ptr,
    GT0_ptr, GEND_ptr, GCB_ptr, SCALE,
    NHQ: tl.constexpr, NHV: tl.constexpr, GRP: tl.constexpr,
    MC: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    SEG: tl.constexpr, STAGES: tl.constexpr,
):
    ps = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    t_beg = tl.load(GT0_ptr + ps)
    t_end = tl.load(GEND_ptr + ps)
    cb = tl.load(GCB_ptr + ps)
    hk = h // GRP
    r = tl.arange(0, MC)
    dk = tl.arange(0, D)
    dv = vb * BV + tl.arange(0, BV)

    s = tl.load(CKPT_ptr + ps * (NHV * D * D) + h * (D * D) + dk[:, None] * D + dv[None, :])

    for c in tl.range(0, SEG, num_stages=STAGES):
        pb = cb + c
        t0 = t_beg + c * MC
        clen = tl.minimum(MC, t_end - t0)
        row = (pb * MC + r[:, None]) * (NHV * D) + h * D
        wv = tl.load(W_ptr + row + dk[None, :])
        u0 = tl.load(U0_ptr + row + dv[None, :])
        kz = tl.load(KZ_ptr + row + dk[None, :])
        gam = tl.load(GAM_ptr + pb * NHV + h)
        gc = tl.load(GC_ptr + (pb * MC + r) * NHV + h)
        kn = _l2norm_rows(K_ptr, t0, clen, hk, NHQ, MC, D).to(tl.bfloat16)
        qn = _l2norm_rows(Q_ptr, t0, clen, hk, NHQ, MC, D) * SCALE

        sb = s.to(tl.bfloat16)
        u = u0.to(tl.float32) - tl.dot(wv, sb)
        aq = tl.dot(qn.to(tl.bfloat16), tl.trans(kn))
        dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
        aq = tl.where(r[:, None] >= r[None, :], aq * dec, 0.0)
        o = tl.dot((qn * tl.exp(gc)[:, None]).to(tl.bfloat16), sb)
        o += tl.dot(aq.to(tl.bfloat16), u.to(tl.bfloat16))
        tl.store(O_ptr + (t0 + r[:, None]) * (NHV * D) + h * D + dv[None, :],
                 o.to(tl.bfloat16), mask=(r < clen)[:, None])
        s = gam * s + tl.dot(tl.trans(kz), u.to(tl.bfloat16))



@triton.jit
def seg_scan_kernel(
    U0_ptr, W_ptr, KZ_ptr, GAM_ptr, DMAT_ptr, PMAT_ptr, GCB_ptr,
    NHV: tl.constexpr, MC: tl.constexpr, D: tl.constexpr,
    BW: tl.constexpr, SEG: tl.constexpr, STAGES: tl.constexpr,
):
    """Level 1 of the parallel scan: per segment, solve the augmented state
    [Delta | P] (Delta starts at 0, P at I) so that one segment collapses into
    S_out = P S_in + Delta.  Fully parallel over (segment, head, column block)."""
    ps = tl.program_id(0)
    h = tl.program_id(1)
    cbk = tl.program_id(2)
    cb = tl.load(GCB_ptr + ps)
    r = tl.arange(0, MC)
    dk = tl.arange(0, D)
    col = cbk * BW + tl.arange(0, BW)
    is_d = col < D
    s = tl.where((col[None, :] - D) == dk[:, None], 1.0, 0.0)

    for c in tl.range(0, SEG, num_stages=STAGES):
        pb = cb + c
        row = (pb * MC + r[:, None]) * (NHV * D) + h * D
        wv = tl.load(W_ptr + row + dk[None, :])
        kz = tl.load(KZ_ptr + row + dk[None, :])
        u0 = tl.load(U0_ptr + row + (col % D)[None, :], mask=is_d[None, :], other=0.0)
        gam = tl.load(GAM_ptr + pb * NHV + h)
        u = u0.to(tl.float32) - tl.dot(wv, s.to(tl.bfloat16))
        s = gam * s + tl.dot(tl.trans(kz), u.to(tl.bfloat16))

    hbase = ps * (NHV * D * D) + h * (D * D) + dk[:, None] * D
    tl.store(DMAT_ptr + hbase + col[None, :], s, mask=is_d[None, :])
    tl.store(PMAT_ptr + hbase + (col - D)[None, :], s.to(tl.bfloat16),
             mask=(col >= D)[None, :])


@triton.jit
def top_scan_kernel(
    DMAT_ptr, PMAT_ptr, S0_ptr, CKPT_ptr, SNEW_ptr, SBASE_ptr, SNSEG_ptr,
    NHV: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
):
    """Level 2: short serial scan over segment transitions, S <- P S + Delta."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    sbase = tl.load(SBASE_ptr + n)
    nseg = tl.load(SNSEG_ptr + n)
    dk = tl.arange(0, D)
    dj = tl.arange(0, D)
    dv = vb * BV + tl.arange(0, BV)
    so = n * (NHV * D * D) + h * (D * D) + dv[None, :] * D + dk[:, None]
    s = tl.load(S0_ptr + so)

    for g in range(0, nseg):
        tl.store(CKPT_ptr + (sbase + g) * (NHV * D * D) + h * (D * D)
                 + dk[:, None] * D + dv[None, :], s)
        base = (sbase + g) * (NHV * D * D) + h * (D * D) + dk[:, None] * D
        dl = tl.load(DMAT_ptr + base + dv[None, :])
        pm = tl.load(PMAT_ptr + base + dj[None, :])
        s = dl + tl.dot(pm, s.to(tl.bfloat16))

    tl.store(SNEW_ptr + so, s)



@triton.jit
def attn_kernel(
    Q_ptr, K_ptr, GC_ptr, AQ_ptr, QG_ptr, BT0_ptr, BLEN_ptr, SCALE,
    NHQ: tl.constexpr, NHV: tl.constexpr, GRP: tl.constexpr,
    MC: tl.constexpr, D: tl.constexpr,
):
    """Precompute the intra-block attention matrix and the decayed queries.

    aq[i,j] = scale*(qn_i.kn_j)*exp(gc_i-gc_j) for j<=i,  qg_i = scale*qn_i*exp(gc_i).
    Pulling this out of the output kernel removes one matmul, two row-reductions
    and an MCxMC exp() from that kernel's serial replay loop, and shrinks its
    shared-memory footprint so more blocks fit per SM.
    """
    pb = tl.program_id(0)
    h = tl.program_id(1)
    t0 = tl.load(BT0_ptr + pb)
    blen = tl.load(BLEN_ptr + pb)
    hk = h // GRP
    r = tl.arange(0, MC)
    d = tl.arange(0, D)
    gc = tl.load(GC_ptr + (pb * MC + r) * NHV + h)
    kn = _l2norm_rows(K_ptr, t0, blen, hk, NHQ, MC, D).to(tl.bfloat16)
    qn = _l2norm_rows(Q_ptr, t0, blen, hk, NHQ, MC, D) * SCALE
    aq = tl.dot(qn.to(tl.bfloat16), tl.trans(kn))
    dec = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
    aq = tl.where(r[:, None] >= r[None, :], aq * dec, 0.0)
    tl.store(AQ_ptr + pb * (NHV * MC * MC) + h * (MC * MC) + r[:, None] * MC + r[None, :],
             aq.to(tl.bfloat16))
    tl.store(QG_ptr + (pb * MC + r[:, None]) * (NHV * D) + h * D + d[None, :],
             (qn * tl.exp(gc)[:, None]).to(tl.bfloat16))


@triton.jit
def out2_kernel(
    U0_ptr, W_ptr, KZ_ptr, GAM_ptr, AQ_ptr, QG_ptr, CKPT_ptr, O_ptr,
    GT0_ptr, GEND_ptr, GCB_ptr,
    NHV: tl.constexpr, MC: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    SEG: tl.constexpr, STAGES: tl.constexpr,
):
    """Output kernel using the precomputed aq/qg: 4 matmuls, no reductions."""
    ps = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    t_beg = tl.load(GT0_ptr + ps)
    t_end = tl.load(GEND_ptr + ps)
    cb = tl.load(GCB_ptr + ps)
    r = tl.arange(0, MC)
    dk = tl.arange(0, D)
    dv = vb * BV + tl.arange(0, BV)

    s = tl.load(CKPT_ptr + ps * (NHV * D * D) + h * (D * D) + dk[:, None] * D + dv[None, :])

    for c in tl.range(0, SEG, num_stages=STAGES):
        pb = cb + c
        t0 = t_beg + c * MC
        row = (pb * MC + r[:, None]) * (NHV * D) + h * D
        wv = tl.load(W_ptr + row + dk[None, :])
        u0 = tl.load(U0_ptr + row + dv[None, :])
        kz = tl.load(KZ_ptr + row + dk[None, :])
        qg = tl.load(QG_ptr + row + dk[None, :])
        aq = tl.load(AQ_ptr + pb * (NHV * MC * MC) + h * (MC * MC)
                     + r[:, None] * MC + r[None, :])
        gam = tl.load(GAM_ptr + pb * NHV + h)
        sb = s.to(tl.bfloat16)
        u = u0.to(tl.float32) - tl.dot(wv, sb)
        o = tl.dot(qg, sb) + tl.dot(aq, u.to(tl.bfloat16))
        tl.store(O_ptr + (t0 + r[:, None]) * (NHV * D) + h * D + dv[None, :],
                 o.to(tl.bfloat16), mask=(r < tl.minimum(MC, t_end - t0))[:, None])
        s = gam * s + tl.dot(tl.trans(kz), u.to(tl.bfloat16))


def build_plan(cu_seqlens, C=CHUNK, m=1, seg=4):
    """Block = m chunks of C tokens.  Block slots are padded to a multiple of seg."""
    mc = C * m
    cu = cu_seqlens.tolist()
    dev = cu_seqlens.device
    bt0, blen = [], []
    scb, snseg, sbase = [], [], []
    gt0, gend, gcb = [], [], []
    for n in range(len(cu) - 1):
        s, e = cu[n], cu[n + 1]
        nb = (e - s + mc - 1) // mc
        ns = (nb + seg - 1) // seg
        scb.append(len(bt0))
        snseg.append(ns)
        sbase.append(len(gt0))
        for g0 in range(0, ns * seg, seg):
            gt0.append(s + g0 * mc)
            gend.append(e)
            gcb.append(len(bt0) + g0)
        for b in range(ns * seg):
            bt0.append(min(s + b * mc, e))
            blen.append(max(0, min(mc, e - s - b * mc)))

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=dev)

    return dict(bt0=i32(bt0), blen=i32(blen), nblk=len(bt0), nseq=len(cu) - 1,
                scb=i32(scb), snseg=i32(snseg), sbase=i32(sbase),
                gt0=i32(gt0), gend=i32(gend), gcb=i32(gcb),
                nseg=len(gt0), seg=seg, chunk=C, m=m, mc=mc)


class Runner:
    """Preallocated 3-launch GDN forward."""

    def __init__(self, q, k, v, a, b, A_log, dt_bias, state, plan, scale=None,
                 bv_scan=32, bv_out=64, ns=NS_STEPS, mode="two", bw=128, fused_attn=True, fuse_prep=False,
                 warps_prep=8, warps_scan=8, warps_out=8, warps_seg=8, warps_top=8,
                 stages_scan=1, stages_out=1, stages_seg=1):
        T, NHQ, D = q.shape
        NHV = v.shape[1]
        self.args = (q, k, v, a, b, A_log, dt_bias, state)
        self.plan = plan
        self.cfg = dict(NHQ=NHQ, NHV=NHV, GRP=NHV // NHQ, D=D)
        self.scale = D ** -0.5 if scale is None else scale
        self.bv_scan, self.bv_out = bv_scan, bv_out
        # Every triangular power through C-1 is required for stable state transitions.
        self.ns = max(ns, plan["chunk"].bit_length() - 1)
        self.mode, self.bw, self.fused_attn = mode, bw, fused_attn
        self.fuse_prep_attn = fuse_prep and fused_attn and plan["m"] == 1
        self.warps = (warps_prep, warps_scan, warps_out, warps_seg, warps_top)
        self.stages = (stages_scan, stages_out, stages_seg)
        nrow = plan["nblk"] * plan["mc"]
        dev = q.device
        self.u0 = torch.empty(nrow, NHV, D, dtype=torch.bfloat16, device=dev)
        self.w = torch.empty(nrow, NHV, D, dtype=torch.bfloat16, device=dev)
        self.kz = torch.empty(nrow, NHV, D, dtype=torch.bfloat16, device=dev)
        self.gc = torch.empty(nrow, NHV, dtype=torch.float32, device=dev)
        self.gam = torch.empty(plan["nblk"], NHV, dtype=torch.float32, device=dev)
        self.ckpt = torch.empty(plan["nseg"], NHV, D, D, dtype=torch.float32, device=dev)
        self.o = torch.empty(T, NHV, D, dtype=torch.bfloat16, device=dev)
        self.snew = torch.empty_like(state)
        two = mode == "two"
        self.dmat = (torch.empty(plan["nseg"], NHV, D, D, dtype=torch.float32, device=dev)
                     if two else None)
        self.pmat = (torch.empty(plan["nseg"], NHV, D, D, dtype=torch.bfloat16, device=dev)
                     if two else None)
        self.aq = (torch.empty(plan["nblk"], NHV, plan["mc"], plan["mc"],
                               dtype=torch.bfloat16, device=dev) if fused_attn else None)
        self.qg = (torch.empty(nrow, NHV, D, dtype=torch.bfloat16, device=dev)
                   if fused_attn else None)

    def prep(self):
        q, k, v, a, b, A_log, dt_bias, state = self.args
        p = self.plan
        prep_kernel[(p["nblk"], self.cfg["NHV"])](
            k, v, a, b, A_log, dt_bias, q, self.u0, self.w, self.kz, self.gc, self.gam,
            self.aq, self.qg, p["bt0"], p["blen"], self.scale,
            C=p["chunk"], NS=self.ns, M=p["m"], FUSE_ATTN=self.fuse_prep_attn,
            num_warps=self.warps[0], **self.cfg)

    def scan(self):
        state = self.args[7]
        p = self.plan
        c = self.cfg
        scan_kernel[(p["nseq"], c["NHV"], c["D"] // self.bv_scan)](
            self.u0, self.w, self.kz, self.gam, state, self.ckpt, self.snew,
            p["scb"], p["snseg"], p["sbase"],
            NHV=c["NHV"], MC=p["mc"], D=c["D"], BV=self.bv_scan, SEG=p["seg"],
            STAGES=self.stages[0], num_warps=self.warps[1])

    def attn(self):
        q, k = self.args[0], self.args[1]
        p = self.plan
        c = self.cfg
        attn_kernel[(p["nblk"], c["NHV"])](
            q, k, self.gc, self.aq, self.qg, p["bt0"], p["blen"], self.scale,
            MC=p["mc"], num_warps=self.warps[2], **c)

    def out2(self):
        p = self.plan
        c = self.cfg
        out2_kernel[(p["nseg"], c["NHV"], c["D"] // self.bv_out)](
            self.u0, self.w, self.kz, self.gam, self.aq, self.qg, self.ckpt, self.o,
            p["gt0"], p["gend"], p["gcb"],
            NHV=c["NHV"], MC=p["mc"], D=c["D"], BV=self.bv_out, SEG=p["seg"],
            STAGES=self.stages[1], num_warps=self.warps[2])

    def out(self):
        q, k = self.args[0], self.args[1]
        p = self.plan
        c = self.cfg
        out_kernel[(p["nseg"], c["NHV"], c["D"] // self.bv_out)](
            q, k, self.u0, self.w, self.kz, self.gc, self.gam, self.ckpt, self.o,
            p["gt0"], p["gend"], p["gcb"], self.scale,
            MC=p["mc"], BV=self.bv_out, SEG=p["seg"], STAGES=self.stages[1],
            num_warps=self.warps[2], **c)

    def seg(self):
        p = self.plan
        c = self.cfg
        seg_scan_kernel[(p["nseg"], c["NHV"], 2 * c["D"] // self.bw)](
            self.u0, self.w, self.kz, self.gam, self.dmat, self.pmat, p["gcb"],
            NHV=c["NHV"], MC=p["mc"], D=c["D"], BW=self.bw, SEG=p["seg"],
            STAGES=self.stages[2], num_warps=self.warps[3])

    def top(self):
        state = self.args[7]
        p = self.plan
        c = self.cfg
        top_scan_kernel[(p["nseq"], c["NHV"], c["D"] // self.bv_scan)](
            self.dmat, self.pmat, state, self.ckpt, self.snew, p["sbase"], p["snseg"],
            NHV=c["NHV"], D=c["D"], BV=self.bv_scan, num_warps=self.warps[4])

    def parts(self):
        mid = ([("seg", self.seg), ("top", self.top)] if self.mode == "two"
               else [("scan", self.scan)])
        if not self.fused_attn:
            tail = [("out", self.out)]
        elif self.fuse_prep_attn:
            tail = [("out2", self.out2)]
        else:
            tail = [("attn", self.attn), ("out2", self.out2)]
        return [("prep", self.prep)] + mid + tail

    def __call__(self):
        for _, fn in self.parts():
            fn()
        return self.o, self.snew
