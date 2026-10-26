import torch
import triton
import triton.language as tl

BT = 64
DK = 128
DV = 128


@triton.jit
def _inv_tril(Lm, rt, eye, same):
    # (I+Lm)^{-1}, Lm strictly lower [BT,BT]; stable blocked inverse, bf16 dots
    Ld = tl.where(same, Lm, 0.0)
    Lo = tl.where(same, 0.0, Lm)
    P = -Ld
    Dinv = eye + P
    Ppow = P
    for _ in range(3):
        Ppow = tl.dot(Ppow.to(tl.bfloat16), Ppow.to(tl.bfloat16))
        Dinv = Dinv + tl.dot(Ppow.to(tl.bfloat16), Dinv.to(tl.bfloat16))
    M = -tl.dot(Dinv.to(tl.bfloat16), Lo.to(tl.bfloat16))
    Tinv = Dinv
    term = Dinv
    for _ in range(3):
        term = tl.dot(M.to(tl.bfloat16), term.to(tl.bfloat16))
        Tinv = Tinv + term
    return Tinv


@triton.jit
def _k1_kernel(
    k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    ctok0_ptr, clen_ptr,
    Wv_ptr, Rk_ptr, Ks_ptr, gcl_ptr,
    Hv: tl.constexpr, rep: tl.constexpr,
    s_k_t, s_k_h, s_v_t, s_v_h, s_ab_t, s_ab_h,
    s_wv_c, s_wv_h, s_rk_c, s_rk_h, s_ks_c, s_ks_h,
    BT: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
):
    pid = tl.program_id(0)
    gc = pid // Hv
    h = pid % Hv
    hqk = h // rep
    tok0 = tl.load(ctok0_ptr + gc)
    clen = tl.load(clen_ptr + gc)

    if clen <= 0:
        return
    rk = tl.arange(0, DK)
    rv = tl.arange(0, DV)
    rt = tl.arange(0, BT)
    tok = tok0 + rt
    mask_t = rt < clen

    kc = tl.load(k_ptr + tok[:, None] * s_k_t + hqk * s_k_h + rk[None, :],
                 mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
    alog = tl.load(alog_ptr + h).to(tl.float32)
    dtb = tl.load(dtb_ptr + h).to(tl.float32)
    a_exp = tl.exp(alog)
    av = tl.load(a_ptr + tok * s_ab_t + h * s_ab_h, mask=mask_t, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + tok * s_ab_t + h * s_ab_h, mask=mask_t, other=0.0).to(tl.float32)
    sp = tl.where(av + dtb > 20.0, av + dtb, tl.log(1.0 + tl.exp(av + dtb)))
    lg = tl.where(mask_t, -a_exp * sp, 0.0)
    beta = tl.where(mask_t, 1.0 / (1.0 + tl.exp(-bv)), 0.0)
    clog = tl.cumsum(lg, axis=0)
    gamma = tl.exp(clog)

    causal = rt[:, None] >= rt[None, :]
    strict = rt[:, None] > rt[None, :]
    eye = tl.where(rt[:, None] == rt[None, :], 1.0, 0.0)
    same = (rt // 16)[:, None] == (rt // 16)[None, :]
    D = tl.where(causal, tl.exp(clog[:, None] - clog[None, :]), 0.0)
    KKt = tl.dot(kc, tl.trans(kc))
    Lm = tl.where(strict, D * KKt, 0.0) * beta[:, None]
    Tinv = _inv_tril(Lm, rt, eye, same)                        # fp32

    clast = tl.min(clog, axis=0)
    gl = tl.exp(clast - clog)
    gclast = tl.exp(clast)
    # store each WY factor as soon as it's computed -> fewer live [BT,*] tiles at peak
    Ks = (gl[:, None] * kc.to(tl.float32)).to(tl.bfloat16)     # [BT,K]
    tl.store(Ks_ptr + gc * s_ks_c + h * s_ks_h + rt[:, None] * DK + rk[None, :], Ks)
    tl.store(gcl_ptr + gc * Hv + h, gclast)

    Tinv_b = Tinv.to(tl.bfloat16)
    bgk = ((beta * gamma)[:, None] * kc.to(tl.float32)).to(tl.bfloat16)  # [BT,K]
    Rk = tl.dot(Tinv_b, bgk).to(tl.bfloat16)                    # [BT,K]
    tl.store(Rk_ptr + gc * s_rk_c + h * s_rk_h + rt[:, None] * DK + rk[None, :], Rk)

    vc = tl.load(v_ptr + tok[:, None] * s_v_t + h * s_v_h + rv[None, :],
                 mask=mask_t[:, None], other=0.0)              # [BT,V]
    bvv = (beta[:, None] * vc).to(tl.bfloat16)                  # [BT,V]
    Wv = tl.dot(Tinv_b, bvv).to(tl.bfloat16)                    # [BT,V]
    tl.store(Wv_ptr + gc * s_wv_c + h * s_wv_h + rt[:, None] * DV + rv[None, :], Wv)


@triton.jit
def _scan_kernel(
    Wv_ptr, Rk_ptr, Ks_ptr, gcl_ptr, state_ptr, h_ptr, nstate_ptr, Wout_ptr,
    seqc0_ptr, seqnc_ptr,
    Hv: tl.constexpr, NVB: tl.constexpr,
    s_wv_c, s_wv_h, s_rk_c, s_rk_h, s_ks_c, s_ks_h,
    s_st_n, s_st_h, s_st_v, s_st_k,
    s_h_c, s_h_h,
    DK: tl.constexpr, DV: tl.constexpr, VS: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % Hv
    n = tmp // Hv
    c0 = tl.load(seqc0_ptr + n)
    nc = tl.load(seqnc_ptr + n)

    rk = tl.arange(0, DK)
    rt = tl.arange(0, BT)
    rv = vb * VS + tl.arange(0, VS)
    st_base = state_ptr + n * s_st_n + h * s_st_h
    S = tl.load(st_base + rv[None, :] * s_st_v + rk[:, None] * s_st_k).to(tl.float32)  # [K,VS]

    for c in range(0, nc):
        gc = c0 + c
        tl.store(h_ptr + gc * s_h_c + h * s_h_h + rk[:, None] * DV + rv[None, :],
                 S.to(tl.bfloat16))
        Rk = tl.load(Rk_ptr + gc * s_rk_c + h * s_rk_h + rt[:, None] * DK + rk[None, :])
        Ks = tl.load(Ks_ptr + gc * s_ks_c + h * s_ks_h + rt[:, None] * DK + rk[None, :])
        Wv = tl.load(Wv_ptr + gc * s_wv_c + h * s_wv_h + rt[:, None] * DV + rv[None, :])
        gcl = tl.load(gcl_ptr + gc * Hv + h)
        WW = (Wv - tl.dot(Rk, S.to(tl.bfloat16))).to(tl.bfloat16)   # [BT,VS] == W for kout
        tl.store(Wout_ptr + gc * s_wv_c + h * s_wv_h + rt[:, None] * DV + rv[None, :], WW)
        S = gcl * S + tl.dot(tl.trans(Ks), WW)                       # [K,VS]
    nst_base = nstate_ptr + n * s_st_n + h * s_st_h
    tl.store(nst_base + rv[None, :] * s_st_v + rk[:, None] * s_st_k, S)


@triton.jit
def _kout_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
    h_ptr, Wout_ptr, out_ptr,
    ctok0_ptr, clen_ptr,
    scale,
    Hv: tl.constexpr, rep: tl.constexpr,
    s_q_t, s_q_h, s_k_t, s_k_h, s_ab_t, s_ab_h,
    s_h_c, s_h_h, s_wv_c, s_wv_h, s_o_t, s_o_h,
    BT: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
):
    pid = tl.program_id(0)
    gc = pid // Hv
    h = pid % Hv
    hqk = h // rep
    tok0 = tl.load(ctok0_ptr + gc)
    clen = tl.load(clen_ptr + gc)

    if clen <= 0:
        return
    rk = tl.arange(0, DK)
    rv = tl.arange(0, DV)
    rt = tl.arange(0, BT)
    tok = tok0 + rt
    mask_t = rt < clen

    qc = tl.load(q_ptr + tok[:, None] * s_q_t + hqk * s_q_h + rk[None, :],
                 mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
    kc = tl.load(k_ptr + tok[:, None] * s_k_t + hqk * s_k_h + rk[None, :],
                 mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
    alog = tl.load(alog_ptr + h).to(tl.float32)
    dtb = tl.load(dtb_ptr + h).to(tl.float32)
    a_exp = tl.exp(alog)
    av = tl.load(a_ptr + tok * s_ab_t + h * s_ab_h, mask=mask_t, other=0.0).to(tl.float32)
    sp = tl.where(av + dtb > 20.0, av + dtb, tl.log(1.0 + tl.exp(av + dtb)))
    lg = tl.where(mask_t, -a_exp * sp, 0.0)
    clog = tl.cumsum(lg, axis=0)
    gamma = tl.exp(clog)

    causal = rt[:, None] >= rt[None, :]
    D = tl.where(causal, tl.exp(clog[:, None] - clog[None, :]), 0.0)
    QKt = tl.dot(qc, tl.trans(kc))
    Pintra = (tl.where(causal, D * QKt, 0.0)).to(tl.bfloat16)   # [BT,BT] shared across V halves

    HV2: tl.constexpr = DV // 2
    for vh in tl.static_range(2):
        rvh = vh * HV2 + tl.arange(0, HV2)
        S = tl.load(h_ptr + gc * s_h_c + h * s_h_h + rk[:, None] * DV + rvh[None, :]).to(tl.bfloat16)  # [K,HV2]
        W = tl.load(Wout_ptr + gc * s_wv_c + h * s_wv_h + rt[:, None] * DV + rvh[None, :]).to(tl.bfloat16)  # [BT,HV2]
        O = scale * (gamma[:, None] * tl.dot(qc, S) + tl.dot(Pintra, W))
        tl.store(out_ptr + tok[:, None] * s_o_t + h * s_o_h + rvh[None, :],
                 O.to(tl.bfloat16), mask=mask_t[:, None])


@triton.jit
def _meta_chunk(cu_ptr, ctok0_ptr, clen_ptr, N, BT: tl.constexpr):
    gc = tl.program_id(0)
    c0 = 0
    tok0 = 0
    ln = 0
    found = 0
    for n in range(N):
        s0 = tl.load(cu_ptr + n).to(tl.int32)
        s1 = tl.load(cu_ptr + n + 1).to(tl.int32)
        nch = (s1 - s0 + BT - 1) // BT
        here = (found == 0) and (gc >= c0) and (gc < c0 + nch)
        local = gc - c0
        rem = s1 - s0 - local * BT
        tok0 = tl.where(here, s0 + local * BT, tok0)
        ln = tl.where(here, tl.minimum(BT, rem), ln)
        found = tl.where(here, 1, found)
        c0 = c0 + nch
    tl.store(ctok0_ptr + gc, tok0)
    tl.store(clen_ptr + gc, tl.where(found == 1, ln, 0))


@triton.jit
def _meta_seq(cu_ptr, seqc0_ptr, seqnc_ptr, N, BT: tl.constexpr):
    n = tl.program_id(0)
    c0 = 0
    my = 0
    for m in range(N):
        s0 = tl.load(cu_ptr + m).to(tl.int32)
        s1 = tl.load(cu_ptr + m + 1).to(tl.int32)
        nch = (s1 - s0 + BT - 1) // BT
        c0 = tl.where(m < n, c0 + nch, c0)
        my = tl.where(m == n, nch, my)
    tl.store(seqc0_ptr + n, c0)
    tl.store(seqnc_ptr + n, my)


def gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, Hqk, K = q.shape
    Hv, V = v.shape[1], v.shape[2]
    N = state.shape[0]
    rep = Hv // Hqk
    dev = q.device

    # chunk metadata computed on-GPU via tiny kernels (graph-capturable, no CPU sync)
    NC_max = (T + BT - 1) // BT + N
    cu = cu_seqlens.to(torch.int64)
    c_tok0 = torch.empty(NC_max, dtype=torch.int32, device=dev)
    c_len = torch.empty(NC_max, dtype=torch.int32, device=dev)
    seq_c0 = torch.empty(N, dtype=torch.int32, device=dev)
    seq_nc = torch.empty(N, dtype=torch.int32, device=dev)
    _meta_chunk[(NC_max,)](cu, c_tok0, c_len, N, BT)
    _meta_seq[(N,)](cu, seq_c0, seq_nc, N, BT)
    NC = NC_max

    hstate = torch.empty(NC, Hv, K, V, dtype=torch.bfloat16, device=dev)
    Wv = torch.empty(NC, Hv, BT, V, dtype=torch.bfloat16, device=dev)
    Rk = torch.empty(NC, Hv, BT, K, dtype=torch.bfloat16, device=dev)
    Ks = torch.empty(NC, Hv, BT, K, dtype=torch.bfloat16, device=dev)
    Wout = torch.empty(NC, Hv, BT, V, dtype=torch.bfloat16, device=dev)
    gcl = torch.empty(NC, Hv, dtype=torch.float32, device=dev)
    NVB = 8
    VS = V // NVB
    out = torch.empty(T, Hv, V, dtype=torch.bfloat16, device=dev)
    new_state = torch.empty_like(state)

    _k1_kernel[(NC * Hv,)](
        k, v, a, b, A_log, dt_bias, c_tok0, c_len, Wv, Rk, Ks, gcl,
        Hv, rep,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), a.stride(0), a.stride(1),
        Wv.stride(0), Wv.stride(1), Rk.stride(0), Rk.stride(1), Ks.stride(0), Ks.stride(1),
        BT, K, V, num_warps=4, num_stages=1,
    )
    _scan_kernel[(N * Hv * NVB,)](
        Wv, Rk, Ks, gcl, state, hstate, new_state, Wout, seq_c0, seq_nc,
        Hv, NVB,
        Wv.stride(0), Wv.stride(1), Rk.stride(0), Rk.stride(1), Ks.stride(0), Ks.stride(1),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        hstate.stride(0), hstate.stride(1),
        K, V, VS, BT, num_warps=4, num_stages=3,
    )
    _kout_kernel[(NC * Hv,)](
        q, k, a, b, A_log, dt_bias, hstate, Wout, out, c_tok0, c_len,
        scale, Hv, rep,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        a.stride(0), a.stride(1),
        hstate.stride(0), hstate.stride(1), Wout.stride(0), Wout.stride(1),
        out.stride(0), out.stride(1),
        BT, K, V, num_warps=8, num_stages=1,
    )
    return out, new_state
