import torch
import triton
import triton.language as tl

C = 32          # chunk size; stable via per-channel ref r=0.5*G_last (|G-r|<=80)
D = 128
_NW_PREP = 4
_NW_SCAN = 4
_BV = 64
_NS = 2


@triton.jit
def _prepare(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, Alog_ptr, dtb_ptr,
    u0_ptr, W_ptr, Kdn_ptr, Kd2_ptr, Qp_ptr, Qpr_ptr, eg_ptr, ctok_ptr, clen_ptr,
    stride_kt, stride_kh, stride_ut, stride_uh, stride_ec, stride_eh,
    H: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok0 = tl.load(ctok_ptr + pid_c)
    clen = tl.load(clen_ptr + pid_c)

    row = tl.arange(0, BT)
    col = tl.arange(0, BK)
    rmask = row < clen
    off = (tok0 + row)[:, None] * stride_kt + pid_h * stride_kh + col[None, :]
    m2 = rmask[:, None]

    q = tl.load(q_ptr + off, mask=m2, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=m2, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + off, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=m2, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + (tok0 + row) * H + pid_h, mask=rmask, other=0.0).to(tl.float32)
    beta = tl.sigmoid(beta)
    alog = tl.load(Alog_ptr + pid_h).to(tl.float32)
    dtb = tl.load(dtb_ptr + pid_h * BK + col).to(tl.float32)

    q = q / tl.maximum(tl.sqrt(tl.sum(q * q, 1)), 1e-12)[:, None]
    k = k / tl.maximum(tl.sqrt(tl.sum(k * k, 1)), 1e-12)[:, None]
    logdec = -5.0 * tl.sigmoid(tl.exp(alog) * (g + dtb[None, :]))
    logdec = tl.where(rmask[:, None], logdec, 0.0)
    Gc = tl.cumsum(logdec, axis=0)
    glast = tl.sum(tl.where(row[:, None] == (clen - 1), Gc, 0.0), 0)
    r = 0.5 * glast
    eGmr = tl.exp2((Gc - r[None, :]) * 1.4426950408889634)
    emGr = tl.exp(r[None, :] - Gc)
    eG = tl.exp2(Gc * 1.4426950408889634)
    Kup_r = k * eGmr
    Kdown_r = k * emGr

    A = tl.dot(Kup_r.to(tl.bfloat16), tl.trans(Kdown_r).to(tl.bfloat16))
    A = tl.where(row[:, None] > row[None, :], A, 0.0)
    # store decayed keys early to free registers before the inverse
    oo = (tok0 + row)[:, None] * stride_ut + pid_h * stride_uh + col[None, :]
    tl.store(Kdn_ptr + oo, Kdown_r.to(tl.bfloat16), mask=m2)
    tl.store(Kd2_ptr + oo, (k * tl.exp2((glast[None, :] - Gc) * 1.4426950408889634)).to(tl.bfloat16), mask=m2)
    tl.store(Qp_ptr + oo, (q * eG).to(tl.bfloat16), mask=m2)
    tl.store(Qpr_ptr + oo, (q * eGmr).to(tl.bfloat16), mask=m2)
    tl.store(eg_ptr + pid_c * stride_ec + pid_h * stride_eh + col, tl.exp2(glast * 1.4426950408889634))
    Y = -(beta[:, None] * A)
    X = Y + (row[:, None] == row[None, :]).to(tl.float32)
    for _ in tl.static_range(4):
        Y = tl.dot(Y.to(tl.bfloat16), Y.to(tl.bfloat16))
        X = X + tl.dot(Y.to(tl.bfloat16), X.to(tl.bfloat16))
    Tm = X.to(tl.bfloat16)
    u0 = tl.dot(Tm, (beta[:, None] * v).to(tl.bfloat16))
    W = tl.dot(Tm, (beta[:, None] * (k * eG)).to(tl.bfloat16))
    tl.store(u0_ptr + oo, u0.to(tl.bfloat16), mask=m2)
    tl.store(W_ptr + oo, W, mask=m2)


@triton.jit
def _pmat(
    Kdn_ptr, Qpr_ptr, P_ptr, ctok_ptr, clen_ptr,
    stride_ut, stride_uh, stride_pc, stride_ph,
    H: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok0 = tl.load(ctok_ptr + pid_c)
    clen = tl.load(clen_ptr + pid_c)
    row = tl.arange(0, BT)
    kcol = tl.arange(0, BK)
    km = (row < clen)[:, None]
    ko = (tok0 + row)[:, None] * stride_ut + pid_h * stride_uh + kcol[None, :]
    Kdn = tl.load(Kdn_ptr + ko, mask=km, other=0.0)
    Qpr = tl.load(Qpr_ptr + ko, mask=km, other=0.0)
    P = tl.dot(Qpr, tl.trans(Kdn))
    P = tl.where(row[:, None] >= row[None, :], P, 0.0)
    po = pid_c * stride_pc + pid_h * stride_ph + row[:, None] * BT + row[None, :]
    tl.store(P_ptr + po, P.to(tl.bfloat16))


@triton.jit
def _scan_output(
    u0_ptr, W_ptr, Kd2_ptr, Qp_ptr, P_ptr, eg_ptr,
    state_ptr, out_ptr, cu_ptr, coff_ptr,
    stride_ut, stride_uh, stride_ec, stride_eh, stride_pc, stride_ph,
    stride_sn, stride_sh, stride_sv,
    H: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    scale: tl.constexpr, NS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    s = tl.load(cu_ptr + pid_n)
    e = tl.load(cu_ptr + pid_n + 1)
    gc = tl.load(coff_ptr + pid_n)

    row = tl.arange(0, BT)
    kcol = tl.arange(0, BK)
    vcol = tl.arange(0, BV)
    v0 = pid_v * BV
    soff = pid_n * stride_sn + pid_h * stride_sh + (v0 + vcol)[:, None] * stride_sv + kcol[None, :]
    S = tl.load(state_ptr + soff).to(tl.float32)

    nch = tl.cdiv(e - s, BT)
    for i in tl.range(0, nch, num_stages=NS):
        pos = s + i * BT
        clen = tl.minimum(BT, e - pos)
        km = (row < clen)[:, None]
        koff = (pos + row)[:, None] * stride_ut + pid_h * stride_uh + kcol[None, :]
        voff = (pos + row)[:, None] * stride_ut + pid_h * stride_uh + (v0 + vcol)[None, :]
        gci = gc + i
        u0 = tl.load(u0_ptr + voff, mask=km, other=0.0)
        W = tl.load(W_ptr + koff, mask=km, other=0.0)
        Kd2 = tl.load(Kd2_ptr + koff, mask=km, other=0.0)
        Qp = tl.load(Qp_ptr + koff, mask=km, other=0.0)
        po = gci * stride_pc + pid_h * stride_ph + row[:, None] * BT + row[None, :]
        P = tl.load(P_ptr + po)
        eg = tl.load(eg_ptr + gci * stride_ec + pid_h * stride_eh + kcol)

        Sb = S.to(tl.bfloat16)
        UT = tl.trans(u0).to(tl.float32) - tl.dot(Sb, tl.trans(W))
        UTb = UT.to(tl.bfloat16)
        oT = tl.dot(UTb, tl.trans(P)) + tl.dot(Sb, tl.trans(Qp))
        tl.store(out_ptr + voff, (tl.trans(oT) * scale).to(out_ptr.dtype.element_ty), mask=km)
        S = S * eg[None, :] + tl.dot(UTb, Kd2)

    tl.store(state_ptr + soff, S.to(state_ptr.dtype.element_ty))


def kda_fwd(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens):
    assert q.shape[0] == 1
    T = q.shape[1]; H = q.shape[2]
    N = cu_seqlens.numel() - 1
    dev = q.device
    q2 = q[0]; k2 = k[0]; v2 = v[0]; g2 = g[0]
    beta2 = beta[0]
    A_log = A_log.float().contiguous()
    dt_bias = dt_bias.float().contiguous()
    cu = cu_seqlens.to(torch.int64)

    cul = cu_seqlens.tolist()
    ctok = []; clen = []; coff = []
    for n in range(N):
        coff.append(len(ctok))
        s = cul[n]; e = cul[n + 1]; p = s
        while p < e:
            ctok.append(p); clen.append(min(C, e - p)); p += C
    ctok_t = torch.tensor(ctok, dtype=torch.int64, device=dev)
    clen_t = torch.tensor(clen, dtype=torch.int64, device=dev)
    coff_t = torch.tensor(coff, dtype=torch.int64, device=dev)
    n_chunks = len(ctok)

    u0 = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    W = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    Kdn = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    Kd2 = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    Qp = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    Qpr = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    eg = torch.empty(n_chunks, H, D, dtype=torch.float32, device=dev)
    P = torch.empty(n_chunks, H, C, C, dtype=torch.bfloat16, device=dev)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)

    _prepare[(n_chunks, H)](
        q2, k2, v2, g2, beta2, A_log, dt_bias, u0, W, Kdn, Kd2, Qp, Qpr, eg, ctok_t, clen_t,
        k2.stride(0), k2.stride(1), u0.stride(0), u0.stride(1), eg.stride(0), eg.stride(1),
        H=H, BT=C, BK=D, num_warps=_NW_PREP,
    )

    _pmat[(n_chunks, H)](
        Kdn, Qpr, P, ctok_t, clen_t,
        u0.stride(0), u0.stride(1), P.stride(0), P.stride(1),
        H=H, BT=C, BK=D, num_warps=4,
    )
    BV = _BV
    scale = 1.0 / (D ** 0.5)
    _scan_output[(D // BV, H, N)](
        u0, W, Kd2, Qp, P, eg, state, out[0], cu, coff_t,
        u0.stride(0), u0.stride(1), eg.stride(0), eg.stride(1), P.stride(0), P.stride(1),
        state.stride(0), state.stride(1), state.stride(2),
        H=H, BT=C, BK=D, BV=BV, scale=scale, NS=_NS, num_warps=_NW_SCAN,
    )
    return out, state
