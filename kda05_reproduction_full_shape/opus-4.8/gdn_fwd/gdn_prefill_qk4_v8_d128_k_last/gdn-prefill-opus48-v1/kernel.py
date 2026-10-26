"""
Gated Delta Net prefill (k-last state layout) for NVIDIA B200.

Two Triton kernels with automatic shape dispatch in ``run`` (value-returning):
    run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    -> (output, new_state)

GVA mapping: v-head h uses q/k head h // (HV // HQ). State S is [K, V] in fp32.
Gates computed in-kernel: g_t = -exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b).

1) Fused recurrent kernel (overhead-optimal; used when avg seqlen T/N < 20):
     S = exp(g_t)*S ; old_v = k_t@S ; u = beta_t*(v_t-old_v)
     S = S + outer(k_t,u) ; o_t = scale*(q_t@S)
   One program per (seq, v-head, V-block); K not tiled, V tiled (BV=16, 8 warps).

2) Fused chunked delta-rule kernel (tensor cores; used otherwise):
   Numerically-stable chunked form (all exp args <= 0). Per chunk of C tokens:
     gc=cumsum(logg); D[i,j]=exp(gc_i-gc_j); L=tril(beta*(K@K^T)*D,-1)
     W=(I+L)^-1 @ (beta*(V - egc*(K@S0)))      # (I+L)^-1 via nilpotent doubling
     O=scale*(egc*(Q@S0) + tril(Q@K^T*D,0)@W)
     S_end = egc[-1]*S0 + (exp(gc[-1]-gc)*K)^T @ W
   All within-chunk products are tl.dot (tcgen05). Precision is right-sized to
   what the fp32 new_state / bf16 output actually need:
   - OUTPUT-path MMAs Q@S and QK@W: tf32 (GDN_CHK_OUT_PREC) — Q is bf16 and both
     feed only the bf16 `output`, so fp32-accurate MMA is wasted (-13..-24%).
   - INVERSE-path MMAs (nilpotent-doubling Pp@Pp, acc@(I+Pp), and W=Tinv@rhs):
     **must stay tf32x3** (GDN_CHK_INV_PREC, default tf32x3). These feed the fp32
     new_state. tf32 was tried (P9) and REJECTED: although it is -18% and passes
     the deepest cases with 2.5e-3 margin, it FAILS 5/100 on specific medium
     workloads where (I+L) is ill-conditioned / L has large entries — the
     nilpotent doubling relies on L^C == 0 exactly, which tf32 rounding breaks,
     and the residual can blow up (f5619793 -> 2.84e16). So the inverse is the
     one state-path MMA that genuinely needs the extra passes.
   - K@K^T, Q@K^T, K@S, Kd@W are tf32 (bf16-exact operands / shallow accum).

3) 3-kernel chunk-parallel path (chunk3; deepest serial scans): prepare (UT
   inverse + u,w, parallel over all chunks) + light serial state-scan + fully
   parallel output. Intermediates (U/W/S_c/W_c) are stored in bf16, so the
   scan's w@S and the output's Q@S_c / QK@W_c read bf16-precision operands:
   tf32x3 there was pure waste (triple MMA passes + register pressure for
   precision the bf16 operand cannot carry). Using tf32 instead cuts the chunk3
   scan+out register pressure -> higher occupancy on these latency-bound kernels
   (NCU: scan occ 10%, SM 11%), for -14.5% on the 18 xl workloads / -10% full-100
   (50.05 -> 45.07ms, locked harness) with byte-identical correctness (output is
   bf16; state max_abs 5.5e-3 << atol 0.01). See docs/draft.md P5.
"""

import math
import os

import torch
import triton
import triton.language as tl

_BV = int(os.environ.get("GDN_BV", "16"))
_NW = int(os.environ.get("GDN_NW", "8"))
_NS = int(os.environ.get("GDN_NS", "2"))


@triton.jit
def _fused_recurrent_gdn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    h0_ptr, o_ptr, ht_ptr, cu_ptr,
    scale,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
):
    i_nh = tl.program_id(0)     # seq * HV + v-head
    i_v = tl.program_id(1)      # v-block
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_hq = i_h // (HV // HQ)    # GVA: v-head -> q/k head

    seq_start = tl.load(cu_ptr + i_n).to(tl.int32)
    seq_end = tl.load(cu_ptr + i_n + 1).to(tl.int32)

    offs_k = tl.arange(0, K)                 # [K]
    offs_v = i_v * BV + tl.arange(0, BV)     # [BV]

    # initial state h0[i_n, i_h] is [V, K]; S[k, v] = h0[v, k]
    h0_base = h0_ptr + (i_n * HV + i_h) * V * K
    p_h0 = h0_base + offs_v[None, :] * K + offs_k[:, None]   # [K, BV]
    b_h = tl.load(p_h0).to(tl.float32)

    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)

    for t in range(seq_start, seq_end):
        off_q = (t * HQ + i_hq) * K
        off_v = (t * HV + i_h) * V
        b_q = tl.load(q_ptr + off_q + offs_k).to(tl.float32) * scale    # [K]
        b_k = tl.load(k_ptr + off_q + offs_k).to(tl.float32)            # [K]
        b_v = tl.load(v_ptr + off_v + offs_v).to(tl.float32)           # [BV]
        a_t = tl.load(a_ptr + t * HV + i_h).to(tl.float32)
        b_t = tl.load(b_ptr + t * HV + i_h).to(tl.float32)

        x = a_t + dtb
        softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        decay = tl.exp(neg_exp_alog * softplus)           # exp(g_t)
        beta = 1.0 / (1.0 + tl.exp(-b_t))

        b_h = b_h * decay
        old_v = tl.sum(b_h * b_k[:, None], axis=0)         # [BV]
        u = (b_v - old_v) * beta                           # [BV]
        b_h = b_h + b_k[:, None] * u[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)           # [BV]
        tl.store(o_ptr + off_v + offs_v, b_o.to(tl.bfloat16))

    ht_base = ht_ptr + (i_n * HV + i_h) * V * K
    p_ht = ht_base + offs_v[None, :] * K + offs_k[:, None]
    tl.store(p_ht, b_h.to(tl.float32))


@triton.jit
def _chunk_gdn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    h0_ptr, o_ptr, ht_ptr, cu_ptr,
    scale,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, C: tl.constexpr, NINV: tl.constexpr,
    OUT_PREC: tl.constexpr = "tf32x3",
    INV_PREC: tl.constexpr = "tf32x3",
):
    i_nh = tl.program_id(0)     # seq * HV + v-head
    i_v = tl.program_id(1)      # v-block
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_hq = i_h // (HV // HQ)

    seq_start = tl.load(cu_ptr + i_n).to(tl.int32)
    seq_end = tl.load(cu_ptr + i_n + 1).to(tl.int32)

    offs_k = tl.arange(0, K)                 # [K]
    offs_v = i_v * BV + tl.arange(0, BV)     # [BV]
    offs_c = tl.arange(0, C)                 # [C]

    h0_base = h0_ptr + (i_n * HV + i_h) * V * K
    S = tl.load(h0_base + offs_v[None, :] * K + offs_k[:, None]).to(tl.float32)  # [K,BV]

    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)

    lower_strict = offs_c[:, None] > offs_c[None, :]     # [C,C]
    lower_incl = offs_c[:, None] >= offs_c[None, :]
    eyeC = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)

    for cs in range(seq_start, seq_end, C):
        clen = seq_end - cs
        mask_c = offs_c < clen                            # [C]
        # loads (padded rows -> 0)
        q_rows = cs + offs_c
        Q = tl.load(q_ptr + (q_rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)     # [C,K]
        Kk = tl.load(k_ptr + (q_rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                     mask=mask_c[:, None], other=0.0).to(tl.float32)    # [C,K]
        Vv = tl.load(v_ptr + (q_rows[:, None] * HV + i_h) * V + offs_v[None, :],
                     mask=mask_c[:, None], other=0.0).to(tl.float32)    # [C,BV]
        at = tl.load(a_ptr + q_rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
        bt = tl.load(b_ptr + q_rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)

        x = at + dtb
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        logg = tl.where(mask_c, neg_exp_alog * sp, 0.0)   # [C], padded -> 0
        beta = tl.where(mask_c, 1.0 / (1.0 + tl.exp(-bt)), 0.0)
        gc = tl.cumsum(logg, axis=0)                      # [C] <=0
        egc = tl.exp(gc)                                  # <=1
        D = tl.exp(gc[:, None] - gc[None, :])             # [C,C], use where lower

        KK = tl.dot(Kk, tl.trans(Kk), input_precision="tf32")      # [C,C], bf16-exact in tf32
        L = tl.where(lower_strict, beta[:, None] * KK * D, 0.0)   # strictly lower
        # T = (I+L)^-1 via nilpotent doubling: prod (I + P^{2^j}), P=-L
        P = -L
        acc = eyeC + P
        Pp = P
        for _ in tl.static_range(NINV - 1):
            Pp = tl.dot(Pp, Pp, input_precision=INV_PREC)
            acc = tl.dot(acc, eyeC + Pp, input_precision=INV_PREC)
        Tinv = acc                                        # [C,C]

        KS = tl.dot(Kk, S, input_precision="tf32")        # [C,BV]
        rhs = beta[:, None] * (Vv - egc[:, None] * KS)    # [C,BV]
        W = tl.dot(Tinv, rhs, input_precision=INV_PREC)   # [C,BV]

        QS = tl.dot(Q, S, input_precision=OUT_PREC)       # [C,BV]
        QK = tl.where(lower_incl, tl.dot(Q, tl.trans(Kk), input_precision="tf32") * D, 0.0)  # [C,C]
        Oc = scale * (egc[:, None] * QS + tl.dot(QK, W, input_precision=OUT_PREC))  # [C,BV]
        tl.store(o_ptr + (q_rows[:, None] * HV + i_h) * V + offs_v[None, :],
                 Oc.to(tl.bfloat16), mask=mask_c[:, None])

        # state update: S = egc_last*S + (exp(gc_last-gc)*Kk)^T @ W
        gc_last = tl.sum(logg, axis=0)                    # = gc[clen-1] (padded logg 0)
        egc_last = tl.exp(gc_last)
        Kd = tl.exp(gc_last - gc)[:, None] * Kk           # [C,K]
        S = egc_last * S + tl.dot(tl.trans(Kd), W, input_precision="tf32")  # [K,BV]

    ht_base = ht_ptr + (i_n * HV + i_h) * V * K
    tl.store(ht_base + offs_v[None, :] * K + offs_k[:, None], S.to(tl.float32))


@triton.jit
def _chunk2_prepare_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    U_ptr, W_ptr, gtok0_ptr, gclen_ptr,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NINV: tl.constexpr, UW_PREC: tl.constexpr = "tf32x3",
    INV_PREC: tl.constexpr = "tf32x3",
):
    # One program per (global-chunk, v-head). Computes the expensive, state-
    # INDEPENDENT per-chunk quantities (parallel across ALL chunks -> fills the
    # device even for low-N):
    #   u = Tinv @ (beta * V)         [C, V]
    #   w = Tinv @ (beta * egc * K)   [C, K]
    # where Tinv = (I + tril(beta*K@K^T*D,-1))^-1 (UT transform, nilpotent
    # doubling). Everything else (D, QK, Kd, egc_last) is cheap and recomputed in
    # the scan kernel from reloaded Q/K/gates.
    i_gc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_hq = i_h // (HV // HQ)

    tok0 = tl.load(gtok0_ptr + i_gc).to(tl.int32)
    clen = tl.load(gclen_ptr + i_gc).to(tl.int32)

    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_c = tl.arange(0, C)
    mask_c = offs_c < clen
    rows = tok0 + offs_c

    Q_unused = 0  # keep symmetry; prepare does not need Q
    Kk = tl.load(k_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                 mask=mask_c[:, None], other=0.0).to(tl.float32)   # [C,K]
    Vv = tl.load(v_ptr + (rows[:, None] * HV + i_h) * V + offs_v[None, :],
                 mask=mask_c[:, None], other=0.0).to(tl.float32)   # [C,V]
    at = tl.load(a_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
    bt = tl.load(b_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)

    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)

    x = at + dtb
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    logg = tl.where(mask_c, neg_exp_alog * sp, 0.0)
    beta = tl.where(mask_c, 1.0 / (1.0 + tl.exp(-bt)), 0.0)
    gc = tl.cumsum(logg, axis=0)
    egc = tl.exp(gc)
    D = tl.exp(gc[:, None] - gc[None, :])

    lower_strict = offs_c[:, None] > offs_c[None, :]
    eyeC = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)

    KK = tl.dot(Kk, tl.trans(Kk), input_precision="tf32")
    L = tl.where(lower_strict, beta[:, None] * KK * D, 0.0)
    P = -L
    acc = eyeC + P
    Pp = P
    for _ in tl.static_range(NINV - 1):
        Pp = tl.dot(Pp, Pp, input_precision=INV_PREC)
        acc = tl.dot(acc, eyeC + Pp, input_precision=INV_PREC)
    Tinv = acc

    u = tl.dot(Tinv, beta[:, None] * Vv, input_precision=UW_PREC)           # [C,V]
    w = tl.dot(Tinv, (beta * egc)[:, None] * Kk, input_precision=UW_PREC)   # [C,K]

    ubase = U_ptr + (i_gc * HV + i_h) * C * V
    tl.store(ubase + offs_c[:, None] * V + offs_v[None, :], u, mask=mask_c[:, None])
    wbase = W_ptr + (i_gc * HV + i_h) * C * K
    tl.store(wbase + offs_c[:, None] * K + offs_k[None, :], w, mask=mask_c[:, None])


@triton.jit
def _chunk2_scan_out_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    U_ptr, W_ptr, h0_ptr, o_ptr, ht_ptr,
    cbase_ptr, ccnt_ptr, gtok0_ptr, gclen_ptr,
    scale,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, C: tl.constexpr,
):
    # One program per (seq, v-head, v-block). Sequential over this seq's chunks,
    # but each step is LIGHT (no inverse): W = u - w@S ; O = scale(egc*Q@S +
    # tril(Q@K^T*D)@W) ; S = egc_last*S + Kd^T@W.
    i_nh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_hq = i_h // (HV // HQ)

    base = tl.load(cbase_ptr + i_n).to(tl.int32)
    cnt = tl.load(ccnt_ptr + i_n).to(tl.int32)

    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, C)

    h0_base = h0_ptr + (i_n * HV + i_h) * V * K
    S = tl.load(h0_base + offs_v[None, :] * K + offs_k[:, None]).to(tl.float32)  # [K,BV]

    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)

    lower_incl = offs_c[:, None] >= offs_c[None, :]

    for lc in range(cnt):
        i_gc = base + lc
        tok0 = tl.load(gtok0_ptr + i_gc).to(tl.int32)
        clen = tl.load(gclen_ptr + i_gc).to(tl.int32)
        mask_c = offs_c < clen
        rows = tok0 + offs_c

        Q = tl.load(q_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)      # [C,K]
        Kk = tl.load(k_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                     mask=mask_c[:, None], other=0.0).to(tl.float32)     # [C,K]
        at = tl.load(a_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
        bt = tl.load(b_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)

        x = at + dtb
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        logg = tl.where(mask_c, neg_exp_alog * sp, 0.0)
        gc = tl.cumsum(logg, axis=0)
        egc = tl.exp(gc)
        D = tl.exp(gc[:, None] - gc[None, :])

        u = tl.load(U_ptr + (i_gc * HV + i_h) * C * V + offs_c[:, None] * V + offs_v[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)      # [C,BV]
        w = tl.load(W_ptr + (i_gc * HV + i_h) * C * K + offs_c[:, None] * K + offs_k[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)      # [C,K]

        Wc = u - tl.dot(w, S, input_precision="tf32x3")                  # [C,BV]
        QS = tl.dot(Q, S, input_precision="tf32x3")                      # [C,BV]
        QK = tl.where(lower_incl, tl.dot(Q, tl.trans(Kk), input_precision="tf32") * D, 0.0)
        Oc = scale * (egc[:, None] * QS + tl.dot(QK, Wc, input_precision="tf32x3"))
        tl.store(o_ptr + (rows[:, None] * HV + i_h) * V + offs_v[None, :],
                 Oc.to(tl.bfloat16), mask=mask_c[:, None])

        gc_last = tl.sum(logg, axis=0)
        egc_last = tl.exp(gc_last)
        Kd = tl.exp(gc_last - gc)[:, None] * Kk                          # [C,K]
        S = egc_last * S + tl.dot(tl.trans(Kd), Wc, input_precision="tf32")  # [K,BV]

    ht_base = ht_ptr + (i_n * HV + i_h) * V * K
    tl.store(ht_base + offs_v[None, :] * K + offs_k[:, None], S.to(tl.float32))


@triton.jit
def _chunk3_scan_kernel(
    k_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    U_ptr, W_ptr, h0_ptr, ht_ptr, Sc_ptr, Wc_ptr,
    cbase_ptr, ccnt_ptr, gtok0_ptr, gclen_ptr,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, C: tl.constexpr, WS_PREC: tl.constexpr = "tf32x3",
):
    # Pure sequential state scan (LIGHT: 2 small matmuls/chunk, no output, no
    # inverse). Stores per-chunk start-state S_c and per-chunk W_c for the
    # parallel output kernel. One program per (seq, v-head, v-block).
    i_nh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_hq = i_h // (HV // HQ)

    base = tl.load(cbase_ptr + i_n).to(tl.int32)
    cnt = tl.load(ccnt_ptr + i_n).to(tl.int32)

    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, C)

    h0_base = h0_ptr + (i_n * HV + i_h) * V * K
    S = tl.load(h0_base + offs_v[None, :] * K + offs_k[:, None]).to(tl.float32)  # [K,BV]

    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)

    for lc in range(cnt):
        i_gc = base + lc
        tok0 = tl.load(gtok0_ptr + i_gc).to(tl.int32)
        clen = tl.load(gclen_ptr + i_gc).to(tl.int32)
        mask_c = offs_c < clen
        rows = tok0 + offs_c

        Kk = tl.load(k_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                     mask=mask_c[:, None], other=0.0).to(tl.float32)     # [C,K]
        at = tl.load(a_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
        bt = tl.load(b_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
        x = at + dtb
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        logg = tl.where(mask_c, neg_exp_alog * sp, 0.0)
        gc = tl.cumsum(logg, axis=0)

        u = tl.load(U_ptr + (i_gc * HV + i_h) * C * V + offs_c[:, None] * V + offs_v[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)      # [C,BV]
        w = tl.load(W_ptr + (i_gc * HV + i_h) * C * K + offs_c[:, None] * K + offs_k[None, :],
                    mask=mask_c[:, None], other=0.0).to(tl.float32)      # [C,K]
        Wc = u - tl.dot(w, S, input_precision=WS_PREC)                   # [C,BV]

        # store chunk-start state S_c (this v-block slice) and W_c for output pass
        tl.store(Sc_ptr + (i_gc * HV + i_h) * K * V + offs_k[:, None] * V + offs_v[None, :], S)
        tl.store(Wc_ptr + (i_gc * HV + i_h) * C * V + offs_c[:, None] * V + offs_v[None, :],
                 Wc, mask=mask_c[:, None])

        gc_last = tl.sum(logg, axis=0)
        egc_last = tl.exp(gc_last)
        Kd = tl.exp(gc_last - gc)[:, None] * Kk                          # [C,K]
        S = egc_last * S + tl.dot(tl.trans(Kd), Wc, input_precision="tf32")  # [K,BV]

    ht_base = ht_ptr + (i_n * HV + i_h) * V * K
    tl.store(ht_base + offs_v[None, :] * K + offs_k[:, None], S.to(tl.float32))


@triton.jit
def _chunk3_out_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    Sc_ptr, Wc_ptr, o_ptr, gtok0_ptr, gclen_ptr,
    scale,
    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, C: tl.constexpr, OUT_PREC: tl.constexpr = "tf32x3",
):
    # Fully parallel output: one program per (global-chunk, v-head, v-block).
    #   O = scale*(egc*(Q@S_c) + tril(Q@K^T*D,0) @ W_c)
    i_gc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_v = tl.program_id(2)
    i_hq = i_h // (HV // HQ)

    tok0 = tl.load(gtok0_ptr + i_gc).to(tl.int32)
    clen = tl.load(gclen_ptr + i_gc).to(tl.int32)

    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, C)
    mask_c = offs_c < clen
    rows = tok0 + offs_c

    Q = tl.load(q_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                mask=mask_c[:, None], other=0.0).to(tl.float32)
    Kk = tl.load(k_ptr + (rows[:, None] * HQ + i_hq) * K + offs_k[None, :],
                 mask=mask_c[:, None], other=0.0).to(tl.float32)
    at = tl.load(a_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
    bt = tl.load(b_ptr + rows * HV + i_h, mask=mask_c, other=0.0).to(tl.float32)
    neg_exp_alog = -tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
    dtb = tl.load(dtb_ptr + i_h).to(tl.float32)
    x = at + dtb
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    logg = tl.where(mask_c, neg_exp_alog * sp, 0.0)
    gc = tl.cumsum(logg, axis=0)
    egc = tl.exp(gc)
    D = tl.exp(gc[:, None] - gc[None, :])
    lower_incl = offs_c[:, None] >= offs_c[None, :]

    S = tl.load(Sc_ptr + (i_gc * HV + i_h) * K * V + offs_k[:, None] * V
                + offs_v[None, :]).to(tl.float32)                        # [K,BV]
    Wc = tl.load(Wc_ptr + (i_gc * HV + i_h) * C * V + offs_c[:, None] * V + offs_v[None, :],
                 mask=mask_c[:, None], other=0.0).to(tl.float32)         # [C,BV]

    QS = tl.dot(Q, S, input_precision=OUT_PREC)
    QK = tl.where(lower_incl, tl.dot(Q, tl.trans(Kk), input_precision="tf32") * D, 0.0)
    Oc = scale * (egc[:, None] * QS + tl.dot(QK, Wc, input_precision=OUT_PREC))
    tl.store(o_ptr + (rows[:, None] * HV + i_h) * V + offs_v[None, :],
             Oc.to(tl.bfloat16), mask=mask_c[:, None])


def _log2_int(n):
    r = 0
    while (1 << r) < n:
        r += 1
    return r


def _build_chunk_schedule(cu, C, device):
    """Global-chunk schedule for the chunk-parallel paths. Built on CPU (one
    H2D copy at the end, no per-call GPU sync — a GPU ``.item()`` sync adds a
    large fixed latency that dominates small workloads). ``cu`` is cu_seqlens.
    Returns (chunk_base[N], chunk_cnt[N], g_tok0[GC], g_clen[GC], GC)."""
    cu_list = cu.tolist()
    N = len(cu_list) - 1
    cbase = []
    ccnt = []
    gtok0 = []
    gclen = []
    gc = 0
    for n in range(N):
        s0 = int(cu_list[n])
        s1 = int(cu_list[n + 1])
        L = s1 - s0
        nc = (L + C - 1) // C if L > 0 else 0
        cbase.append(gc)
        ccnt.append(nc)
        for lc in range(nc):
            t0 = s0 + lc * C
            gtok0.append(t0)
            gclen.append(min(C, s1 - t0))
        gc += nc
    i32 = torch.int32
    return (
        torch.tensor(cbase, dtype=i32, device=device),
        torch.tensor(ccnt, dtype=i32, device=device),
        torch.tensor(gtok0 if gtok0 else [0], dtype=i32, device=device),
        torch.tensor(gclen if gclen else [0], dtype=i32, device=device),
        gc,
    )


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, HQ, K = q.shape
    HV = v.shape[1]
    V = v.shape[2]
    N = cu_seqlens.numel() - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    scale = float(scale)

    if state is None:
        h0 = torch.zeros((N, HV, V, K), dtype=torch.float32, device=device)
    else:
        h0 = state

    o = torch.empty((T, HV, V), dtype=torch.bfloat16, device=device)
    ht = torch.empty((N, HV, V, K), dtype=torch.float32, device=device)

    # Shape dispatch calibrated on the OFFICIAL metric: the flashinfer-bench
    # harness times wall-per-iteration (cold_l2_cache=True) INCLUDING host launch
    # gaps. Two consequences:
    #  (1) The dispatch must be SYNC-FREE. A host read of cu_seqlens (e.g. for a
    #      per-seq max depth) adds a device->host sync the harness counts as
    #      latency on every workload (~0.1ms each -> ~10ms over 100). So route
    #      using only T and N (shapes / cu_seqlens.numel(); no copy).
    #  (2) chunk3 launches 3 kernels + streams large bf16 intermediates through
    #      HBM; the harness per-launch overhead + cold-L2 traffic make it pay off
    #      ONLY for the deepest serial scans: full T=8192 batches with few seqs
    #      (one very long seq -> deep) and long low-N medium batches. N is the
    #      sync-free proxy for depth (fewer seqs over fixed T -> longer max seq).
    #   avg<16                                  -> recurrent (tiniest only)
    #   (T>=7000 & N<=44) or (T>=3000 & N<=13)  -> chunk3    (deepest scans)
    #   else                                    -> chunk     (fused, on-chip)
    # avg_thresh 25->16 (P11): the fused chunk kernel is a FLAT ~0.016ms (one
    # padded chunk, tensor cores) for any short seq, while both the recurrent
    # kernel AND the flashinfer baseline grow with max_seqlen. So for max_seqlen
    # >~16 chunk beats the baseline (speedup 1.1-1.5x) whereas recurrent only
    # ties it (~1.0x). Recurrent wins only for the very tiniest (<=~15 tokens),
    # where chunk's fixed overhead dominates. Measured crossover at avg~16.
    # After P7 made chunk3 ~20% faster its break-even T dropped: 3271N2 and
    # 3028N5 both flip to chunk3 by ~10% (t_lo 3500->3000), while 2857N3 stays
    # chunk (prefers it). No chunk-routed workload in T[3000,3500) regresses.
    mode = os.environ.get("GDN_MODE", "auto")
    if mode == "auto":
        avg_seqlen = T / max(N, 1)
        avg_thresh = float(os.environ.get("GDN_AVG_THRESH", "16"))
        t_xl = int(os.environ.get("GDN_T_XL", "7000"))
        n_xl = int(os.environ.get("GDN_N_XL", "44"))
        t_lo = int(os.environ.get("GDN_T_LO", "3000"))
        n_lo = int(os.environ.get("GDN_N_LO", "13"))
        if avg_seqlen < avg_thresh:
            mode = "recurrent"
        elif (T >= t_xl and N <= n_xl) or (T >= t_lo and N <= n_lo):
            mode = "chunk3"
        else:
            mode = "chunk"
    if mode == "chunk3":
        C = int(os.environ.get("GDN_C3", "32"))
        BVc = int(os.environ.get("GDN_CBV3", "16"))
        # scan and out can use DIFFERENT v-block sizes: Sc/Wc are stored full-V,
        # so the occupancy-starved serial scan (NCU: occ 10%, 233 regs) wants a
        # SMALL BV (more CTAs, fewer regs -> higher occupancy) while the
        # tensor-core-bound parallel out kernel wants a LARGER BV (better MMA
        # efficiency). Defaults from the xl sweep: scan 8, out 64 (both beat the
        # old shared BV=16 by ~20% on the chunk3-15 bucket).
        BVscan = int(os.environ.get("GDN_C3_SCAN_BV", "8"))
        BVout = int(os.environ.get("GDN_C3_OUT_BV", "64"))
        NVc = triton.cdiv(V, BVc)
        NVscan = triton.cdiv(V, BVscan)
        NVout = triton.cdiv(V, BVout)
        cbase, ccnt, gtok0, gclen, GC = _build_chunk_schedule(cu_seqlens, C, device)
        # Intermediates default to bf16: chunk3 is HBM-bound under the official
        # cold-L2 cupti metric, so halving intermediate traffic speeds it up.
        # bf16 round-trip (~2^-8 rel) is within the atol/rtol 0.01 gate.
        _idt = torch.bfloat16 if os.environ.get("GDN_C3_IDT", "bf16") == "bf16" else torch.float32
        U = torch.empty((GC, HV, C, V), dtype=_idt, device=device)
        Wb = torch.empty((GC, HV, C, K), dtype=_idt, device=device)
        Sc = torch.empty((GC, HV, K, V), dtype=_idt, device=device)
        Wc = torch.empty((GC, HV, C, V), dtype=_idt, device=device)
        _chunk2_prepare_kernel[(GC, HV)](
            q, k, v, a, b, A_log, dt_bias, U, Wb, gtok0, gclen,
            HQ, HV, K, V, C, _log2_int(C),
            UW_PREC=os.environ.get("GDN_PREP_UW_PREC", "tf32x3"),
            INV_PREC=os.environ.get("GDN_PREP_INV_PREC", "tf32x3"),
            num_warps=int(os.environ.get("GDN_NWP", "4")),
            num_stages=int(os.environ.get("GDN_NSP", "1")),
        )
        _chunk3_scan_kernel[(N * HV, NVscan)](
            k, a, b, A_log, dt_bias, U, Wb, h0, ht, Sc, Wc,
            cbase, ccnt, gtok0, gclen,
            HQ, HV, K, V, BVscan, C,
            WS_PREC=os.environ.get("GDN_SCAN_WS_PREC", "tf32"),
            num_warps=int(os.environ.get("GDN_NWS", "4")),
            num_stages=int(os.environ.get("GDN_NSS", "1")),
        )
        _chunk3_out_kernel[(GC, HV, NVout)](
            q, k, a, b, A_log, dt_bias, Sc, Wc, o, gtok0, gclen, scale,
            HQ, HV, K, V, BVout, C,
            OUT_PREC=os.environ.get("GDN_OUT_PREC", "tf32"),
            num_warps=int(os.environ.get("GDN_NWO", "4")),
            num_stages=int(os.environ.get("GDN_NSO", "1")),
        )
        return o, ht
    if mode == "chunk2":
        C = int(os.environ.get("GDN_C2", "32"))
        BVc = int(os.environ.get("GDN_CBV2", "16"))
        NVc = triton.cdiv(V, BVc)
        cbase, ccnt, gtok0, gclen, GC = _build_chunk_schedule(cu_seqlens, C, device)
        U = torch.empty((GC, HV, C, V), dtype=torch.float32, device=device)
        Wb = torch.empty((GC, HV, C, K), dtype=torch.float32, device=device)
        _chunk2_prepare_kernel[(GC, HV)](
            q, k, v, a, b, A_log, dt_bias, U, Wb, gtok0, gclen,
            HQ, HV, K, V, C, _log2_int(C),
            num_warps=int(os.environ.get("GDN_NWP", "4")),
            num_stages=int(os.environ.get("GDN_NSP", "1")),
        )
        _chunk2_scan_out_kernel[(N * HV, NVc)](
            q, k, a, b, A_log, dt_bias, U, Wb, h0, o, ht,
            cbase, ccnt, gtok0, gclen, scale,
            HQ, HV, K, V, BVc, C,
            num_warps=int(os.environ.get("GDN_NWS", "4")),
            num_stages=int(os.environ.get("GDN_NSS", "1")),
        )
        return o, ht
    if mode == "chunk":
        C = int(os.environ.get("GDN_C", "32"))
        # BV sets tensor-core tile size vs CTA count (grid = N*HV*(V/BV)).
        # Larger BV = bigger MMA tiles (more TC-efficient) but fewer CTAs, so it
        # only pays off when there are enough sequences to still fill the device:
        #   N<=2            -> 16 (very-low-N underfills; most programs)
        #   high-N xl       -> 64 (T>=7000 & N>=45: lots of work + enough CTAs;
        #                          -12..-18% vs BV=32 and beats chunk3 for them)
        #   else            -> 32
        # (overridable via GDN_CBV). BV must divide V=128 (16/32/64/128).
        _cbv_env = os.environ.get("GDN_CBV")
        if _cbv_env:
            BVc = int(_cbv_env)
        elif N <= 2:
            BVc = 16
        elif T >= 7000 and N >= 45:
            BVc = 64
        else:
            BVc = 32
        NVc = triton.cdiv(V, BVc)
        grid = (N * HV, NVc)
        _chunk_gdn_fwd_kernel[grid](
            q, k, v, a, b, A_log, dt_bias, h0, o, ht, cu_seqlens,
            scale, HQ, HV, K, V, BVc, C, _log2_int(C),
            OUT_PREC=os.environ.get("GDN_CHK_OUT_PREC", "tf32"),
            INV_PREC=os.environ.get("GDN_CHK_INV_PREC", "tf32x3"),
            num_warps=int(os.environ.get("GDN_NW", "4")),
            num_stages=int(os.environ.get("GDN_NS", "1")),
        )
        return o, ht

    BV = _BV
    NV = triton.cdiv(V, BV)
    grid = (N * HV, NV)
    _fused_recurrent_gdn_fwd_kernel[grid](
        q, k, v, a, b, A_log, dt_bias, h0, o, ht, cu_seqlens,
        scale,
        HQ, HV, K, V, BV,
        num_warps=_NW,
        num_stages=_NS,
    )
    return o, ht
