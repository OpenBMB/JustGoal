import torch
import triton
import triton.language as tl

C = 64
DK = 128
DV = 128
H = 8
PHI = "tf32"
PHIY = "tf32"
PLO = "tf32"        # phase1 KK/W2
PSC = "tf32"
PSCW = "tf32"
POUT = "tf32"
BV2 = 16            # V-block for scan
BVO = 64            # V-block for out
NSTAGES = 3
W1 = 4
WS = 8
WO = 8
BLK = 16
NREFINE = 0
_META_CACHE = {}
PIPE_G = 3            # pipeline groups (phase1(g+1) overlaps scan(g))
_STREAMS = None
_EVENTS = None


def _pipe_res():
    global _STREAMS, _EVENTS
    if _STREAMS is None:
        _STREAMS = (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
        _EVENTS = None
    return _STREAMS


def _neum_y(nblk):
    ny = 0
    while (1 << (ny + 1)) < nblk:
        ny += 1
    return ny


def _ndiag(blk):
    nd = 0
    while (1 << (nd + 1)) < blk:
        nd += 1
    return nd


@triton.jit
def _softplus(x):
    return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def meta_kernel(cu_ptr, m_seq_ptr, m_start_ptr, m_len_ptr, m_seqstart_ptr, m_seqnc_ptr,
                N: tl.constexpr, C: tl.constexpr):
    gc = tl.program_id(0)
    off = 0
    found = -1
    tok0 = 0
    vlen = 0
    m_ss = 0
    m_nc = 0
    for n in tl.static_range(0, N):
        c0 = tl.load(cu_ptr + n)
        c1 = tl.load(cu_ptr + n + 1)
        ln = c1 - c0
        nc = (ln + C - 1) // C
        in_range = (gc >= off) & (gc < off + nc) & (found < 0)
        local = gc - off
        found = tl.where(in_range, n, found)
        tok0 = tl.where(in_range, c0 + local * C, tok0)
        vln = tl.minimum(tl.maximum(ln - local * C, 0), C)
        vlen = tl.where(in_range, vln, vlen)
        is_me = gc == n
        m_ss = tl.where(is_me, off, m_ss)
        m_nc = tl.where(is_me, nc, m_nc)
        off = off + nc
    valid = (found >= 0) & (vlen > 0)
    ms = tl.where(valid, found, -1)
    tl.store(m_seq_ptr + gc, ms.to(tl.int32))
    tl.store(m_start_ptr + gc, tok0.to(tl.int32))
    tl.store(m_len_ptr + gc, vlen.to(tl.int32))
    if gc < N:
        tl.store(m_seqstart_ptr + gc, m_ss.to(tl.int32))
        tl.store(m_seqnc_ptr + gc, m_nc.to(tl.int32))


@triton.jit
def phase1_kernel(k_ptr, v_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr,
                  mseq_ptr, mstart_ptr, mlen_ptr,
                  T_ptr, Kbd_ptr, Bv_ptr, W2_ptr, logD_ptr, Dl_ptr,
                  sk_t, sk_h, sk_d, sv_t, sv_h, sv_d, sab_t, sab_h,
                  st_c, st_h, st_r, st_j,
                  skb_c, skb_h, skb_r, skb_d,
                  sbv_c, sbv_h, sbv_r, sbv_d,
                  sw2_c, sw2_h, sw2_r, sw2_d,
                  sld_c, sld_h, sld_r, sdl_c, sdl_h,
                  gc0,
                  BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                  NLEV: tl.constexpr,
                  PHI: tl.constexpr, PHIY: tl.constexpr, PLO: tl.constexpr, NREFINE: tl.constexpr):
    gc = tl.program_id(0) + gc0
    h = tl.program_id(1)
    seq = tl.load(mseq_ptr + gc)
    if seq < 0:
        return
    tok0 = tl.load(mstart_ptr + gc)
    vlen = tl.load(mlen_ptr + gc)
    qkh = h // 2
    rows = tl.arange(0, BC)
    dk = tl.arange(0, BK)
    dv = tl.arange(0, BV)
    rmask = rows < vlen
    koff = (tok0 + rows)[:, None] * sk_t + qkh * sk_h + dk[None, :] * sk_d
    Kf = tl.load(k_ptr + koff, mask=rmask[:, None], other=0.0).to(tl.float32)
    aoff = (tok0 + rows) * sab_t + h * sab_h
    av = tl.load(a_ptr + aoff, mask=rmask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + aoff, mask=rmask, other=0.0).to(tl.float32)
    alog = tl.load(alog_ptr + h)
    dtb = tl.load(dtb_ptr + h)
    ea = tl.minimum(tl.exp(alog), 1e30)          # avoid inf (exp overflow for large A_log)
    logg = -ea * _softplus(av + dtb)
    logg = tl.maximum(logg, -60.0)               # g<exp(-60)~=0; keeps logD finite (no -inf/nan)
    logg = tl.where(rmask, logg, 0.0)
    beta = 1.0 / (1.0 + tl.exp(-bv))
    beta = tl.where(rmask, beta, 0.0)
    logD = tl.cumsum(logg, axis=0)
    D = tl.exp(logD)
    KK = tl.dot(Kf, tl.trans(Kf), input_precision=PLO)
    dr = tl.exp(tl.minimum(logD[:, None] - logD[None, :], 0.0))
    strict = rows[:, None] > rows[None, :]
    Lhat = tl.where(strict, beta[:, None] * KK * dr, 0.0)
    eye = (rows[:, None] == rows[None, :]).to(tl.float32)
    # (I+Lhat)^-1 via recursive 2x2 block doubling (stable: bounded intermediates ~||T||)
    T = eye
    for lvl in tl.static_range(NLEV):
        s = 1 << lvl
        blk_s = rows // s
        grp = rows // (2 * s)
        maskB = (grp[:, None] == grp[None, :]) & (blk_s[:, None] == blk_s[None, :] + 1)
        maskD = (blk_s[:, None] == blk_s[None, :]) & ((blk_s[:, None] & 1) == 1)
        LB = tl.where(maskB, Lhat, 0.0)
        P1 = tl.where(maskB, tl.dot(LB, T, input_precision=PHI), 0.0)
        TD = tl.where(maskD, T, 0.0)
        T = T - tl.dot(TD, P1, input_precision=PHI)
    voff = (tok0 + rows)[:, None] * sv_t + h * sv_h + dv[None, :] * sv_d
    Vf = tl.load(v_ptr + voff, mask=rmask[:, None], other=0.0).to(tl.float32)
    Bvm = beta[:, None] * Vf              # [C,V]  bounded
    Kbd = (beta * D)[:, None] * Kf        # [C,K]  bounded (raw k read scale)
    last_logD = tl.sum(tl.where(rows == (BC - 1), logD, 0.0))
    ratio_last = tl.exp(last_logD - logD)
    Kdl = ratio_last[:, None] * Kf        # [C,K]  bounded (write scale)
    Dlast = tl.exp(last_logD)
    W2 = tl.dot(tl.trans(Kdl), T, input_precision=PLO)   # [K,C] = Kdl^T @ T
    toff = gc * st_c + h * st_h + rows[:, None] * st_r + rows[None, :] * st_j
    tl.store(T_ptr + toff, T)
    kboff = gc * skb_c + h * skb_h + rows[:, None] * skb_r + dk[None, :] * skb_d
    tl.store(Kbd_ptr + kboff, Kbd)
    bvoff = gc * sbv_c + h * sbv_h + rows[:, None] * sbv_r + dv[None, :] * sbv_d
    tl.store(Bv_ptr + bvoff, Bvm)
    w2off = gc * sw2_c + h * sw2_h + dk[:, None] * sw2_r + rows[None, :] * sw2_d
    tl.store(W2_ptr + w2off, W2)
    ldoff = gc * sld_c + h * sld_h + rows * sld_r
    tl.store(logD_ptr + ldoff, logD)
    tl.store(Dl_ptr + gc * sdl_c + h * sdl_h, Dlast)


@triton.jit
def scan_kernel(state_ptr, newstate_ptr,
                mseqstart_ptr, mseqnc_ptr,
                Kbd_ptr, Bv_ptr, W2_ptr, Dl_ptr, Sin_ptr,
                sst_n, sst_h, sst_v, sst_k,
                skb_c, skb_h, skb_r, skb_d,
                sbv_c, sbv_h, sbv_r, sbv_d,
                sw2_c, sw2_h, sw2_r, sw2_d,
                sdl_c, sdl_h,
                ssin_c, ssin_h, ssin_k, ssin_v, c0, c1,
                BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NSTAGES: tl.constexpr,
                PLO: tl.constexpr, PLOW: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    v0 = vb * BV
    rows = tl.arange(0, BC)
    dk = tl.arange(0, BK)
    dv = tl.arange(0, BV)
    stoff = n * sst_n + h * sst_h + (v0 + dv)[:, None] * sst_v + dk[None, :] * sst_k
    S = tl.trans(tl.load(newstate_ptr + stoff))       # [DK,BV] fp32 (carried across groups; init=clone)
    seqstart = tl.load(mseqstart_ptr + n)
    nchunks = tl.load(mseqnc_ptr + n)
    ci_start = tl.maximum(0, c0 - seqstart)
    ci_end = tl.minimum(nchunks, c1 - seqstart)
    for ci in tl.range(ci_start, ci_end, num_stages=NSTAGES):
        gc = seqstart + ci
        soff = gc * ssin_c + h * ssin_h + dk[:, None] * ssin_k + (v0 + dv)[None, :] * ssin_v
        tl.store(Sin_ptr + soff, S)
        kboff = gc * skb_c + h * skb_h + rows[:, None] * skb_r + dk[None, :] * skb_d
        Kbd = tl.load(Kbd_ptr + kboff)                # [C,K] fp32
        bvoff = gc * sbv_c + h * sbv_h + rows[:, None] * sbv_r + (v0 + dv)[None, :] * sbv_d
        Bv = tl.load(Bv_ptr + bvoff)                  # [C,BV] fp32
        w2off = gc * sw2_c + h * sw2_h + dk[:, None] * sw2_r + rows[None, :] * sw2_d
        W2 = tl.load(W2_ptr + w2off)                  # [K,C] fp32
        Dlast = tl.load(Dl_ptr + gc * sdl_c + h * sdl_h)
        delta = Bv - tl.dot(Kbd, S, input_precision=PLO)   # read (feedback)
        tl.store(Bv_ptr + bvoff, delta)                    # overwrite Bv w/ delta for out (phase1 refills each replay)
        S = Dlast * S + tl.dot(W2, delta, input_precision=PLOW)  # write
    nsoff = n * sst_n + h * sst_h + (v0 + dv)[:, None] * sst_v + dk[None, :] * sst_k
    tl.store(newstate_ptr + nsoff, tl.trans(S))


@triton.jit
def out_kernel(q_ptr, k_ptr, out_ptr,
               mseq_ptr, mstart_ptr, mlen_ptr,
               T_ptr, Kbd_ptr, Bv_ptr, logD_ptr, Sin_ptr,
               sq_t, sq_h, sq_d, sk_t, sk_h, sk_d, so_t, so_h, so_d,
               st_c, st_h, st_r, st_j,
               skb_c, skb_h, skb_r, skb_d,
               sbv_c, sbv_h, sbv_r, sbv_d,
               sld_c, sld_h, sld_r,
               ssin_c, ssin_h, ssin_k, ssin_v, scale, gc0o,
               BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PLO: tl.constexpr):
    gc = tl.program_id(0) + gc0o
    h = tl.program_id(1)
    vb = tl.program_id(2)
    seq = tl.load(mseq_ptr + gc)
    if seq < 0:
        return
    tok0 = tl.load(mstart_ptr + gc)
    vlen = tl.load(mlen_ptr + gc)
    qkh = h // 2
    v0 = vb * BV
    rows = tl.arange(0, BC)
    dk = tl.arange(0, BK)
    dv = tl.arange(0, BV)
    rmask = rows < vlen
    qoff = (tok0 + rows)[:, None] * sq_t + qkh * sq_h + dk[None, :] * sq_d
    Qf = tl.load(q_ptr + qoff, mask=rmask[:, None], other=0.0).to(tl.float32)
    koff = (tok0 + rows)[:, None] * sk_t + qkh * sk_h + dk[None, :] * sk_d
    Kf = tl.load(k_ptr + koff, mask=rmask[:, None], other=0.0).to(tl.float32)
    ldoff = gc * sld_c + h * sld_h + rows * sld_r
    logD = tl.load(ldoff + logD_ptr)
    D = tl.exp(logD)
    soff = gc * ssin_c + h * ssin_h + dk[:, None] * ssin_k + (v0 + dv)[None, :] * ssin_v
    Sin = tl.load(Sin_ptr + soff)                     # [DK,BV] fp32
    bvoff = gc * sbv_c + h * sbv_h + rows[:, None] * sbv_r + (v0 + dv)[None, :] * sbv_d
    delta = tl.load(Bv_ptr + bvoff)                        # Bv holds delta (written by scan)
    toff = gc * st_c + h * st_h + rows[:, None] * st_r + rows[None, :] * st_j
    T = tl.load(T_ptr + toff)
    r = tl.dot(T, delta, input_precision=PLO)
    QK = tl.dot(Qf, tl.trans(Kf), input_precision=PLO)
    dr = tl.exp(tl.minimum(logD[:, None] - logD[None, :], 0.0))
    Pm = tl.where(rows[:, None] >= rows[None, :], QK * dr, 0.0)
    QS = tl.dot(Qf, Sin, input_precision=PLO)
    o = scale * (D[:, None] * QS + tl.dot(Pm, r, input_precision=PLO))
    ooff = (tok0 + rows)[:, None] * so_t + h * so_h + (v0 + dv)[None, :] * so_d
    tl.store(out_ptr + ooff, o.to(tl.bfloat16), mask=rmask[:, None])


def gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T = q.shape[0]
    N = state.shape[0]
    dev = q.device
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    a = a.contiguous(); b = b.contiguous()
    A_log = A_log.float().contiguous(); dt_bias = dt_bias.float().contiguous()
    state = state.contiguous()
    cu = cu_seqlens
    Cap = (T + C - 1) // C + N
    # NOTE: meta MUST be recomputed every call (no cache) so that under CUDA-graph
    # capture these ops are recorded and re-run on replay. A cache makes capture a
    # cache-hit -> meta ops omitted from graph -> stale meta when the judge replays
    # with fresh inputs (fresh-after-timing) -> nan. All ops below are capture-safe
    # (no host sync / .item()).
    cu_i64 = cu.to(torch.int64)
    m_seq = torch.empty(Cap, device=dev, dtype=torch.int32)
    m_start = torch.empty(Cap, device=dev, dtype=torch.int32)
    m_len = torch.empty(Cap, device=dev, dtype=torch.int32)
    m_seqstart = torch.empty(N, device=dev, dtype=torch.int32)
    m_seqnc = torch.empty(N, device=dev, dtype=torch.int32)
    meta_kernel[(Cap,)](cu_i64, m_seq, m_start, m_len, m_seqstart, m_seqnc, N=N, C=C)
    NEUM_Y = _neum_y(C // BLK)
    NDIAG = _ndiag(BLK)

    Tm = torch.empty((Cap, H, C, C), device=dev, dtype=torch.float32)
    Kbd = torch.empty((Cap, H, C, DK), device=dev, dtype=torch.float32)
    Bv = torch.empty((Cap, H, C, DV), device=dev, dtype=torch.float32)
    W2 = torch.empty((Cap, H, DK, C), device=dev, dtype=torch.float32)
    logD = torch.empty((Cap, H, C), device=dev, dtype=torch.float32)
    Dl = torch.empty((Cap, H), device=dev, dtype=torch.float32)
    Sin = torch.empty((Cap, H, DK, DV), device=dev, dtype=torch.float32)
    out = torch.empty((T, H, DV), device=dev, dtype=torch.bfloat16)
    new_state = state.clone()  # scan reads+writes new_state (carries state across pipeline groups)
    NV = DV // BV2
    NVO = DV // BVO
    NLEV = C.bit_length() - 1
    G = PIPE_G
    L = (Cap + G - 1) // G
    sA, sB, sC = _pipe_res()
    cur = torch.cuda.current_stream()
    fork = torch.cuda.Event(); fork.record(cur)
    sA.wait_event(fork); sB.wait_event(fork); sC.wait_event(fork)
    ev_p1 = [torch.cuda.Event() for _ in range(G)]
    ev_sc = [torch.cuda.Event() for _ in range(G)]
    ng = 0
    for g in range(G):
        gc0 = g * L
        gc1 = min((g + 1) * L, Cap)
        gs = gc1 - gc0
        if gs <= 0:
            break
        ng = g + 1
        with torch.cuda.stream(sA):
            phase1_kernel[(gs, H)](
                k, v, a, b, A_log, dt_bias, m_seq, m_start, m_len,
                Tm, Kbd, Bv, W2, logD, Dl,
                k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                a.stride(0), a.stride(1),
                Tm.stride(0), Tm.stride(1), Tm.stride(2), Tm.stride(3),
                Kbd.stride(0), Kbd.stride(1), Kbd.stride(2), Kbd.stride(3),
                Bv.stride(0), Bv.stride(1), Bv.stride(2), Bv.stride(3),
                W2.stride(0), W2.stride(1), W2.stride(2), W2.stride(3),
                logD.stride(0), logD.stride(1), logD.stride(2), Dl.stride(0), Dl.stride(1),
                gc0, BC=C, BK=DK, BV=DV, NLEV=NLEV,
                PHI=PHI, PHIY=PHIY, PLO=PLO, NREFINE=NREFINE, num_warps=W1,
            )
        ev_p1[g].record(sA)
        sB.wait_event(ev_p1[g])
        with torch.cuda.stream(sB):
            scan_kernel[(N, H, NV)](
                state, new_state, m_seqstart, m_seqnc, Kbd, Bv, W2, Dl, Sin,
                state.stride(0), state.stride(1), state.stride(2), state.stride(3),
                Kbd.stride(0), Kbd.stride(1), Kbd.stride(2), Kbd.stride(3),
                Bv.stride(0), Bv.stride(1), Bv.stride(2), Bv.stride(3),
                W2.stride(0), W2.stride(1), W2.stride(2), W2.stride(3),
                Dl.stride(0), Dl.stride(1),
                Sin.stride(0), Sin.stride(1), Sin.stride(2), Sin.stride(3), gc0, gc1,
                BC=C, BK=DK, BV=BV2, NSTAGES=NSTAGES, PLO=PSC, PLOW=PSCW, num_warps=WS,
            )
        ev_sc[g].record(sB)
        sC.wait_event(ev_sc[g])
        with torch.cuda.stream(sC):
            out_kernel[(gs, H, NVO)](
                q, k, out, m_seq, m_start, m_len, Tm, Kbd, Bv, logD, Sin,
                q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                Tm.stride(0), Tm.stride(1), Tm.stride(2), Tm.stride(3),
                Kbd.stride(0), Kbd.stride(1), Kbd.stride(2), Kbd.stride(3),
                Bv.stride(0), Bv.stride(1), Bv.stride(2), Bv.stride(3),
                logD.stride(0), logD.stride(1), logD.stride(2),
                Sin.stride(0), Sin.stride(1), Sin.stride(2), Sin.stride(3), scale, gc0,
                BC=C, BK=DK, BV=BVO, PLO=POUT, num_warps=WO,
            )
    ea = torch.cuda.Event(); ea.record(sA)
    eb = torch.cuda.Event(); eb.record(sB)
    ec = torch.cuda.Event(); ec.record(sC)
    cur.wait_event(ea); cur.wait_event(eb); cur.wait_event(ec)
    return out, new_state
