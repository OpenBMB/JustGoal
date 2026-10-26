"""GDN varlen prefill v58 (= v57 + first-segment end state written by the chain straight into Sseg[e0]; branch-free segment chain with one store fewer) (v57: = v56 + prep no longer computes/writes unused tables) (v56: = v55 + branch-free token-indexed chain outputs OZ=[ZP;out]: no first-chunk pass; KC adds S_seg@ZP into out) (v55: = v52 + output scale folded into K1's Gam*Q and Aq; output passes do no scaling / fewer casts) (v52: = v51 + K1 gate masks via load fill values, first-segment end states of multi-segment sequences kept in bf16 for the segment chain) (v51: = v49 + packed index records: one tiny load per program instead of 2-4 in K1 / first-chunk pass / KC) (v49: = v48 + K1 small-op trims: negation folded into Tg, no Tb cast round-trip, min() for Llast, single-where D) (v48: = v46 + cheaper [d;gam] packing, chunk table packed as [tok0,nv] pairs) (v46: = v43 + packed small K1 outputs ([Aq;Tb] one store, [d;gam] one store), prep at num_warps=1) (v43: = v41 + later-segment N-part end states stored in bf16 (XNL)) (v41: = v39 + chain order / mseq tables computed inside prep: one table launch) (v39: = v38 + no per-step table loads in the chain: token range derived from the segment start) (v38: = v37 + Kt dropped from K1: chain uses raw K and the decay vector d: S <- gam S + (Yt*d) K) (v37: = v34b + Uv moved into the chain: uvt = Vt Tb^T from raw V and the small Tb) (v34: v34b: Gam*Q stored by K1, output dot accumulates in place) (= v23 + lean output path: K1 no longer builds Zq/Oloc; the chain computes Zt = (S Q^T)*Gam + Yt Aq^T from raw Q and the small Aq) (v23: = v18 + order/mseq tables in one launch) (v18: = v15 with later-segment N-part end states kept in fp32 (separate buffer), P-part bf16) (v15: = v13 + compacted program lists for the output passes (real work first, no-ops last)) (v13: = v12 + optional 2D per-head K1 (HB=1) so that C=64 works) (v12: = v11 + K1 tweaks (fused Gam*Q / d*K without broadcast, contiguous Uv/Oloc stores) + KB merged with the
first-segment output pass (KBC) + KC only for later chunks) (= v9 + chain programs sorted longest-first via a rank table) (= v6 + packed bf16 [P|N] for KB + later-segment P/N parts merged in one program + KB only for multi-segment seqs) (v5 + multi-program prep + K1 batched over heads) (Triton). S-form (state kept as [V,K], always the left/A operand of dots; B operands are loaded).
compact chunk indexing; fat bulk K1; one launch for both affine parts; KB chain over segments; KC output pass."""
import torch, triton, triton.language as tl


@triton.jit
def _prep(CT, FCR, LCR, cu, ch_tok0, ch_nv, ch_e, seg_c0, seg_nch, seq_e0, seq_ne, fc, lc, cnt, perm, mseq, NMS, N, NE, NCH,
          C: tl.constexpr, S: tl.constexpr, NB: tl.constexpr):
    """multi-program table builder: program p handles chunk p, later-segment p and sequence p."""
    p = tl.program_id(0)
    i = tl.arange(0, NB)
    valid = i < N
    c0 = tl.load(cu + i, mask=i < N + 1, other=0)
    c1 = tl.load(cu + i + 1, mask=valid, other=0)
    ln = tl.where(valid, c1 - c0, 0)
    nc = (ln + C - 1) // C
    ne = tl.maximum((nc + S - 1) // S - 1, 0)
    ccum = tl.cumsum(nc, 0); ccum_ex = ccum - nc
    ecum = tl.cumsum(ne, 0); ecum_ex = ecum - ne
    if p < N:
        selp = i == p
        tl.store(seq_e0 + p, tl.sum(tl.where(selp, ecum_ex, 0), 0).to(tl.int32))
        tl.store(seq_ne + p, tl.sum(tl.where(selp, ne, 0), 0).to(tl.int32))
        tl.store(seg_c0 + p, tl.sum(tl.where(selp, ccum_ex, 0), 0).to(tl.int32))
        tl.store(seg_nch + p, tl.minimum(tl.sum(tl.where(selp, nc, 0), 0), S).to(tl.int32))
    if p < NE:
        n = tl.sum(((ecum <= p) & valid).to(tl.int64), 0)
        sel = (i == n) & valid
        cc0 = tl.sum(tl.where(sel, ccum_ex, 0), 0)
        ncn = tl.sum(tl.where(sel, nc, 0), 0)
        k = p - tl.sum(tl.where(sel, ecum_ex, 0), 0) + 1
        ok = n < N
        tl.store(seg_c0 + N + p, tl.where(ok, cc0 + k * S, 0).to(tl.int32))
        tl.store(seg_nch + N + p, tl.where(ok, tl.minimum(ncn - k * S, S), 0).to(tl.int32))
    if p < NCH:
        nn = tl.sum(((ccum <= p) & valid).to(tl.int64), 0)
        sel2 = (i == nn) & valid
        t0n = tl.sum(tl.where(sel2, c0, 0), 0)
        t1n = tl.sum(tl.where(sel2, c1, 0), 0)
        j = p - tl.sum(tl.where(sel2, ccum_ex, 0), 0)
        een = tl.sum(tl.where(sel2, ecum_ex, 0), 0)
        okc = nn < N
        tok0 = t0n + j * C
        tk = tl.where(okc, tok0, 0).to(tl.int32)
        nvv = tl.where(okc, tl.minimum(t1n - tok0, C), 0).to(tl.int32)
        tl.store(CT + p * 2 + tl.arange(0, 2), tl.join(tk, nvv))
        before = valid & (i < nn)
        if okc:
            pi = p.to(tl.int32)
            if j >= S:
                lr = tl.sum(tl.where(before, tl.maximum(nc - S, 0), 0), 0) + (j - S)
                ee = (een + j // S - 1).to(tl.int32)
                tl.store(LCR + lr * 4 + tl.arange(0, 4), tl.reshape(tl.join(tl.join(pi, ee), tl.join(nvv, tk)), (4,)))
    # ---- chain order (entries sorted by length, longest first) and multi-segment sequence list, all from the per-sequence vectors
    fl = tl.minimum(nc, S)                                   # first-segment length per seq
    ll = nc - ne * S                                         # last later-segment length per seq (valid if ne >= 1)
    etot = tl.sum(ne, 0)
    if p < NE + N + NE:
        isP = p < NE
        isF = (p >= NE) & (p < NE + N)
        e_q = tl.where(isP, p, p - NE - N)                   # later-segment index for P / later-N entries
        n_q = tl.where(isF, p - NE, tl.sum(((ecum <= e_q) & valid).to(tl.int64), 0))
        selq = (i == n_q) & valid
        nc_q = tl.sum(tl.where(selq, nc, 0), 0)
        k_q = e_q - tl.sum(tl.where(selq, ecum_ex, 0), 0) + 1
        x = tl.where(isF, tl.minimum(nc_q, S), tl.where(e_q < etot, tl.minimum(nc_q - k_q * S, S), 0))
        has = ne >= 1
        gt_later = tl.maximum(ne - 1, 0) * tl.where(S > x, 1, 0) + tl.where(has & (ll > x), 1, 0)
        cnt_gt = tl.sum(tl.where(valid & (fl > x), 1, 0), 0) + 2 * tl.sum(tl.where(valid, gt_later, 0), 0)
        # equal-length later-segment entries with index < EP (valid ones + the zero-length padding slots)
        EPq = tl.where(isF, NE, e_q)
        full_eq = tl.where(S == x, 1, 0) * tl.minimum(tl.maximum(EPq - ecum_ex, 0), tl.maximum(ne - 1, 0))
        last_eq = tl.where(has & (ll == x) & ((ecum - 1) < EPq), 1, 0)
        peq_q = tl.sum(tl.where(valid, full_eq + last_eq, 0), 0) + tl.where(x == 0, tl.maximum(EPq - etot, 0), 0)
        full_all = tl.where(S == x, 1, 0) * tl.maximum(ne - 1, 0)
        last_all = tl.where(has & (ll == x), 1, 0)
        peq_all = tl.sum(tl.where(valid, full_all + last_all, 0), 0) + tl.where(x == 0, NE - etot, 0)
        NFq = tl.where(isP, 0, tl.where(isF, p - NE, N))
        feq = tl.sum(tl.where(valid & (i < NFq) & (fl == x), 1, 0), 0)
        ties = tl.where(isP, peq_q, tl.where(isF, peq_all + feq, peq_all + feq + peq_q))
        tl.store(perm + cnt_gt + ties, p)
    if p < N:
        mine = tl.sum(tl.where(i == p, ne, 0), 0)
        rk = tl.sum((valid & ((ne > mine) | ((ne == mine) & (i < p)))).to(tl.int32), 0)
        tl.store(mseq + rk, p, mask=rk < NMS)
    if p == 0:
        tl.store(cnt + 1, tl.sum(tl.maximum(nc - S, 0), 0).to(tl.int32))


@triton.jit
def _k1(Qp, Kp, Vp, Ap, Bp, A_log, dt_bias, ch_tok0, ch_nv, REC, KT, UVT, GAM, OL,
        C: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, LOGC: tl.constexpr, DT: tl.constexpr,
        HB: tl.constexpr):
    """bulk chunk-local work, batched over HB value heads (HB/2 key heads) per program."""
    c = tl.program_id(0); hg = tl.program_id(1)
    t0 = tl.load(ch_tok0 + c)
    nv = tl.load(ch_nv + c)
    if nv > 0:
        ic = tl.arange(0, C); ik = tl.arange(0, DK)
        ih = hg * HB + tl.arange(0, HB)                     # value heads
        ihk = hg * (HB // 2) + tl.arange(0, HB // 2)        # key heads
        tok = t0 + ic
        m = ic < nv
        kptr = (tok[None, :, None] * HK + ihk[:, None, None]) * DK + ik[None, None, :]
        vptr = (tok[None, :, None] * HV + ih[:, None, None]) * DK + ik[None, None, :]
        K4 = tl.load(Kp + kptr, mask=m[None, :, None], other=0.0)          # [HB/2, C, DK]
        Q4 = tl.load(Qp + kptr, mask=m[None, :, None], other=0.0)
        V = tl.load(Vp + vptr, mask=m[None, :, None], other=0.0)           # [HB, C, DK]
        gptr = tok[None, :] * HV + ih[:, None]
        a = tl.load(Ap + gptr, mask=m[None, :], other=0.0).to(tl.float32)  # [HB, C]
        b = tl.load(Bp + gptr, mask=m[None, :], other=0.0).to(tl.float32)
        Al = tl.load(A_log + ih); db = tl.load(dt_bias + ih)
        x = a + db[:, None]
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        lg = tl.where(m[None, :], -tl.exp(Al)[:, None] * sp, 0.0)
        beta = tl.where(m[None, :], tl.sigmoid(b), 0.0)
        L = tl.cumsum(lg, 1)
        Llast = tl.sum(tl.where(ic[None, :] == C - 1, L, 0.0), 1)          # [HB]
        Gam = tl.exp(L)
        K4t = tl.permute(K4, (0, 2, 1))
        G4 = tl.dot(K4, K4t)                                               # [HB/2, C, C]
        QK4 = tl.dot(Q4, K4t)
        G = tl.reshape(tl.broadcast_to(G4[:, None, :, :], (HB // 2, 2, C, C)), (HB, C, C))
        QK = tl.reshape(tl.broadcast_to(QK4[:, None, :, :], (HB // 2, 2, C, C)), (HB, C, C))
        K = tl.reshape(tl.broadcast_to(K4[:, None, :, :], (HB // 2, 2, C, DK)), (HB, C, DK))
        low = (ic[:, None] > ic[None, :])[None, :, :]
        D = tl.where(low, tl.exp(tl.where(low, L[:, :, None] - L[:, None, :], 0.0)), 0.0)
        Lm = beta[:, :, None] * D * G
        eye = tl.where(ic[:, None] == ic[None, :], 1.0, 0.0)[None, :, :]
        Tm = eye - tl.where(((ic[:, None] // 2) == (ic[None, :] // 2))[None, :, :], Lm, 0.0)
        for lv in tl.static_range(1, LOGC):
            bs = 2 ** lv
            off = ((ic[:, None] // (2 * bs)) == (ic[None, :] // (2 * bs))) & ((ic[:, None] // bs) != (ic[None, :] // bs))
            Aoff = tl.where(off[None, :, :], Lm, 0.0)
            Th = Tm.to(DT)
            X1 = tl.dot(Aoff.to(DT), Th).to(DT)
            Tm = Tm - tl.dot(Th, X1)
        Tg = (Tm * (beta * Gam)[:, None, :]).to(DT)
        Tb = (Tm * beta[:, None, :]).to(DT)
        Aqh = ((D + eye) * QK).to(DT)
        A2 = tl.dot(Aqh, Tg).to(DT)
        A3 = tl.dot(Aqh, Tb).to(DT)
        base = ((c * HV + ih) * C * DK)[:, None, None]
        rowp = (ic[:, None] * DK + ik[None, :])[None, :, :]
        tl.store(GAM + c * HV + ih, tl.exp(Llast))
        tl.store(REC + 2 * base + rowp, -tl.dot(Tg, K))
        GQ = tl.reshape(tl.reshape(Gam, (HB // 2, 2, C, 1)) * Q4[:, None, :, :], (HB, C, DK))
        tl.store(REC + 2 * base + C * DK + rowp, tl.dot(-A2, K, acc=GQ))
        tl.store(KT + base + rowp, tl.reshape(tl.reshape(tl.exp(Llast[:, None] - L), (HB // 2, 2, C, 1)) * K4[:, None, :, :], (HB, C, DK)))
        tl.store(UVT + base + rowp, tl.dot(Tb, V))
        tl.store(OL + base + rowp, tl.dot(A3, V))


@triton.jit
def _k1_2d(scale, CT, Qp, Kp, Vp, Ap, Bp, A_log, dt_bias, ch_tok0, ch_nv, REC, KT, TBB, GAM, AQB, GVB,
           C: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, LOGC: tl.constexpr, DT: tl.constexpr):
    """per (chunk, value head) 2D version (needed for C=64 where batched TMEM-path dots are pathological)."""
    c = tl.program_id(0); h = tl.program_id(1)
    t0, nv = tl.split(tl.load(CT + c * 2 + tl.arange(0, 2)))
    if nv > 0:
        ic = tl.arange(0, C); ik = tl.arange(0, DK)
        tok = t0 + ic
        m = ic < nv
        hk = h // (HV // HK)
        kptr = (tok[:, None] * HK + hk) * DK + ik[None, :]
        K = tl.load(Kp + kptr, mask=m[:, None], other=0.0)
        Q = tl.load(Qp + kptr, mask=m[:, None], other=0.0)
        a = tl.load(Ap + tok * HV + h, mask=m, other=-10000.0).to(tl.float32)
        b = tl.load(Bp + tok * HV + h, mask=m, other=-10000.0).to(tl.float32)
        Al = tl.load(A_log + h); db = tl.load(dt_bias + h)
        x = a + db
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        lg = -tl.exp(Al) * sp                      # padded tokens: a=-1e4 -> softplus=0 -> log g = 0
        beta = tl.sigmoid(b)                       # padded tokens: b=-1e4 -> beta = 0
        L = tl.cumsum(lg, 0)
        Llast = tl.min(L, 0)
        Gam = tl.exp(L)
        Kt_ = tl.trans(K)
        G = tl.dot(K, Kt_)
        QK = tl.dot(Q, Kt_)
        low = ic[:, None] > ic[None, :]
        D = tl.where(low, tl.exp(L[:, None] - L[None, :]), 0.0)
        Lm = beta[:, None] * D * G
        eye = tl.where(ic[:, None] == ic[None, :], 1.0, 0.0)
        Tm = eye - tl.where((ic[:, None] // 2) == (ic[None, :] // 2), Lm, 0.0)
        for lv in tl.static_range(1, LOGC):
            bs = 2 ** lv
            off = ((ic[:, None] // (2 * bs)) == (ic[None, :] // (2 * bs))) & ((ic[:, None] // bs) != (ic[None, :] // bs))
            Aoff = tl.where(off, Lm, 0.0)
            Th = Tm.to(DT)
            X1 = tl.dot(Aoff.to(DT), Th).to(DT)
            Tm = Tm - tl.dot(Th, X1)
        nTg = (Tm * (-beta * Gam)[None, :]).to(DT)
        base = (c * HV + h) * C * DK
        rowp = ic[:, None] * DK + ik[None, :]
        tl.store(REC + 2 * base + rowp, tl.dot(nTg, K))
        tl.store(REC + 2 * base + C * DK + rowp, (scale * Gam)[:, None] * Q)
        dvv = tl.exp(Llast - L)
        dg = tl.reshape(tl.permute(tl.join(dvv, dvv * 0.0 + tl.exp(Llast)), (1, 0)), (2 * C,))
        tl.store(GVB + (c * HV + h) * 2 * C + tl.arange(0, 2 * C), dg)
        AT = tl.reshape(tl.permute(tl.join((D + eye) * (QK * scale), Tm * beta[None, :]), (2, 0, 1)), (2 * C, C))
        tl.store(AQB + (c * HV + h) * 2 * C * C + tl.arange(0, 2 * C)[:, None] * C + ic[None, :], AT)


@triton.jit
def _order(seg_nch, perm, seq_ne, mseq, N, NE, NMS, QB: tl.constexpr, NB: tl.constexpr):
    """perm[rank] = chain entry sorted by length (longest first); entry q < NE: homogeneous part of later segment q, q >= NE: particular
    part of segment q-NE.  mseq[rank] = sequences sorted by number of later segments (descending), first NMS kept."""
    q = tl.program_id(0)
    if q < NE + N + NE:
        j = tl.arange(0, QB)
        ok = j < NE + N + NE
        gj = tl.where(j < NE, j + N, j - NE)
        lens = tl.load(seg_nch + gj, mask=ok, other=-1)
        mine = tl.sum(tl.where(j == q, lens, 0), 0)
        rank = tl.sum((ok & ((lens > mine) | ((lens == mine) & (j < q)))).to(tl.int32), 0)
        tl.store(perm + rank, q)
    if q < N:
        jn = tl.arange(0, NB)
        ne = tl.load(seq_ne + jn, mask=jn < N, other=-1)
        minen = tl.sum(tl.where(jn == q, ne, 0), 0)
        rk = tl.sum(((jn < N) & ((ne > minen) | ((ne == minen) & (jn < q)))).to(tl.int32), 0)
        tl.store(mseq + rk, q, mask=rk < NMS)


@triton.jit
def _ka(CT, Kp, Vp, ch_tok0, ch_nv, AQB, TBB, GVB, HK: tl.constexpr, State, NewState, REC, KT, UVT, GAM, ZT, ZSTRIDE, XP, Sseg, seq_e0, XNL, seg_c0, seg_nch, seq_ne, perm, N, NE,
        C: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DT: tl.constexpr):
    """thin S-form chains. pid0 < NE: homogeneous part of later segment e (S0 = I);
    pid0 >= NE: particular part of segment g = pid0-NE (S0 = true state if g < N else 0)."""
    pp = tl.program_id(0)
    pid = tl.load(perm + pp // HV); h = pp % HV
    ik = tl.arange(0, DK); ic = tl.arange(0, C)
    sq = ik[:, None] * DK + ik[None, :]
    isN = pid >= NE
    g = tl.where(isN, pid - NE, pid + N)
    if isN:
        if g < N:
            Sm = tl.load(State + (g * HV + h) * DK * DK + sq)
        else:
            Sm = tl.zeros((DK, DK), dtype=tl.float32)
    else:
        Sm = tl.where(ik[:, None] == ik[None, :], 1.0, 0.0)
    fN = tl.where(isN, 1.0, 0.0)
    c0 = tl.load(seg_c0 + g)
    nch = tl.load(seg_nch + g)
    zoff = tl.where(isN, ZSTRIDE, 0)               # ZT = [ZPT ; ZNT] in one buffer -> branch-free loop body
    rowp = ic[:, None] * DK + ik[None, :]
    colp = ik[:, None] * C + ic[None, :]
    hk = h // (HV // HK)
    hk = h // (HV // HK)
    cc = ic[:, None] * C + ic[None, :]
    clast = c0 + tl.maximum(nch - 1, 0)
    seg_t0 = tl.load(CT + c0 * 2)
    seg_te = tl.load(CT + clast * 2) + tl.load(CT + clast * 2 + 1)
    for j in range(nch):
        c = c0 + j
        base = (c * HV + h) * C * DK
        nW = tl.load(REC + 2 * base + rowp)
        GQ = tl.load(REC + 2 * base + C * DK + rowp)
        t0 = seg_t0 + j * C
        nv = seg_te - t0
        Vt = tl.load(Vp + ((t0 + ic)[None, :] * HV + h) * DK + ik[:, None], mask=(ic < nv)[None, :], other=0.0)
        Kb = tl.load(Kp + ((t0 + ic)[:, None] * HK + hk) * DK + ik[None, :], mask=(ic < nv)[:, None], other=0.0)
        dv = tl.load(GVB + (c * HV + h) * 2 * C + ic)
        Tb = tl.load(AQB + (c * HV + h) * 2 * C * C + C * C + cc)
        uvt = tl.dot(Vt, tl.trans(Tb))
        gam = tl.load(GVB + (c * HV + h) * 2 * C + C)
        Aq = tl.load(AQB + (c * HV + h) * 2 * C * C + cc)
        Sh = Sm.to(DT)
        Yt = tl.dot(Sh, tl.trans(nW)) + fN * uvt
        Yth = Yt.to(DT)
        tl.store(ZT + zoff + ((t0 + ic)[None, :] * HV + h) * DK + ik[:, None], tl.dot(Yth, tl.trans(Aq), acc=tl.dot(Sh, tl.trans(GQ))), mask=(ic < nv)[None, :])
        Sm = gam * Sm + tl.dot((Yt * dv[None, :]).to(DT), Kb)
    if isN:
        if g < N:
            ne = tl.load(seq_ne + g)
            if ne == 0:
                if nch == 0:
                    Sm = tl.zeros((DK, DK), dtype=tl.float32)
                tl.store(NewState + (g * HV + h) * DK * DK + sq, Sm)
            else:
                tl.store(Sseg + (tl.load(seq_e0 + g) * HV + h) * DK * DK + sq, Sm)
        else:
            tl.store(XNL + (((g - N) * HV + h) * DK * DK + sq), Sm)
    else:
        tl.store(XP + (pid * HV + h) * DK * DK + sq, Sm)


@triton.jit
def _kbc(FCR, XP, XN, XNL, Sseg, NewState, seq_e0, seq_ne, mseq, ZNT, OL, Out, ch_tok0, ch_nv, fc, cnt, scale, NMS, NSEQ,
         C: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DT: tl.constexpr):
    """programs [0, NMS*HV): segment chain of the NMS multi-segment sequences (longest latency first);
    programs after that: final outputs of first-segment chunks: out = scale * (zN + Oloc)."""
    pp = tl.program_id(0)
    ik = tl.arange(0, DK)
    if pp < NMS * HV:
        n = tl.load(mseq + pp // HV); h = pp % HV
        ne = tl.load(seq_ne + n)
        sq = ik[:, None] * DK + ik[None, :]
        e0 = tl.load(seq_e0 + n)
        Sm = tl.load(Sseg + (e0 * HV + h) * DK * DK + sq).to(tl.float32)
        for s in range(ne - 1):
            e = e0 + s
            P = tl.load(XP + (e * HV + h) * DK * DK + sq)
            Nn = tl.load(XNL + (e * HV + h) * DK * DK + sq)
            Sm = tl.dot(Sm.to(DT), P) + Nn.to(tl.float32)
            tl.store(Sseg + ((e + 1) * HV + h) * DK * DK + sq, Sm)
        if ne > 0:
            el = e0 + ne - 1
            P = tl.load(XP + (el * HV + h) * DK * DK + sq)
            Nn = tl.load(XNL + (el * HV + h) * DK * DK + sq)
            tl.store(NewState + (n * HV + h) * DK * DK + sq, tl.dot(Sm.to(DT), P) + Nn.to(tl.float32))


@triton.jit
def _kc(LCR, ZPT, ZNT, OL, Sseg, Out, ch_tok0, ch_nv, ch_e, lc, cnt, scale,
        C: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DT: tl.constexpr):
    pp = tl.program_id(0)
    i = pp // HV; h = pp % HV
    if i < tl.load(cnt + 1):
        r0, r1 = tl.split(tl.reshape(tl.load(LCR + i * 4 + tl.arange(0, 4)), (2, 2)))
        c, e = tl.split(r0)
        nv, t0 = tl.split(r1)
        ic = tl.arange(0, C); ik = tl.arange(0, DK)
        tok = t0 + ic
        m = ic < nv
        base = (c * HV + h) * C * DK
        colp = ik[:, None] * C + ic[None, :]
        optr = (tok[None, :] * HV + h) * DK + ik[:, None]
        z = tl.load(ZNT + optr, mask=m[None, :], other=0.0).to(tl.float32)
        zp = tl.load(ZPT + optr, mask=m[None, :], other=0.0)
        Ss = tl.load(Sseg + (e * HV + h) * DK * DK + ik[:, None] * DK + ik[None, :])
        tl.store(ZNT + optr, tl.dot(Ss, zp, acc=z), mask=m[None, :])


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, C=32, S=27, HB=1, nw1=1, nwa=8, nwb=8, nwc=2, NS=3, nwp=1, NSB=1):
    T, HK, DK = q.shape; HV = v.shape[1]; N = cu_seqlens.shape[0] - 1
    dev = q.device
    NCH = T // C + N
    NE = T // (C * S)
    NG = N + NE
    NB = triton.next_power_of_2(N + 1)
    i32 = dict(dtype=torch.int32, device=dev); f32 = dict(dtype=torch.float32, device=dev); h16 = dict(dtype=torch.bfloat16, device=dev)
    ch_tok0 = torch.empty(NCH, **i32); ch_nv = torch.empty(NCH, **i32); ch_e = torch.empty(NCH, **i32)
    seg_c0 = torch.empty(NG, **i32); seg_nch = torch.empty(NG, **i32)
    seq_e0 = torch.empty(N, **i32); seq_ne = torch.empty(N, **i32)
    fc = torch.empty(NCH, **i32); lc = torch.empty(NCH, **i32); cnt = torch.empty(2, **i32)
    REC = torch.empty(NCH, HV, 2 * C, DK, **h16); AQB = torch.empty(NCH, HV, 2 * C, C, **h16); GVB = torch.empty(NCH, HV, 2 * C, **f32)
    KT = torch.empty(NCH, HV, C, DK, **h16); UVT = torch.empty(1, **h16); TBB = torch.empty(NCH, HV, C, C, **h16); OL = torch.empty(NCH, HV, C, DK, **h16)
    ZT = torch.empty(2, T, HV, DK, **h16); ZPT = ZT[0]; ZNT = ZT[1]
    GAM = torch.empty(NCH, HV, **f32)
    XP = torch.empty(max(NE, 1), HV, DK, DK, **h16); XN = torch.empty(N, HV, DK, DK, **h16); XNL = torch.empty(max(NE, 1), HV, DK, DK, **h16)
    Sseg = torch.empty(max(NE, 1), HV, DK, DK, **h16)
    out = torch.empty(T, HV, DK, **h16)
    new_state = torch.empty(N, HV, DK, DK, **f32)
    DT = tl.bfloat16
    CT = torch.empty(NCH, 2, **i32); FCR = torch.empty(NCH, 4, **i32); LCR = torch.empty(NCH, 4, **i32)
    perm = torch.empty(NE + NG, **i32)
    NMS = min(N, NE)
    mseq = torch.empty(max(NMS, 1), **i32)
    _prep[(max(NCH, NE + NG),)](CT, FCR, LCR, cu_seqlens, ch_tok0, ch_nv, ch_e, seg_c0, seg_nch, seq_e0, seq_ne, fc, lc, cnt, perm, mseq, NMS, N, NE, NCH,
                                C=C, S=S, NB=NB, num_stages=1, num_warps=nwp)
    if HB >= 2:
        _k1[(NCH, HV // HB)](q, k, v, a, b, A_log, dt_bias, ch_tok0, ch_nv, REC, KT, UVT, GAM, OL,
                            C=C, HK=HK, HV=HV, DK=DK, LOGC=C.bit_length() - 1, DT=DT, HB=HB, num_stages=1, num_warps=nw1)
    else:
        _k1_2d[(NCH, HV)](scale, CT, q, k, v, a, b, A_log, dt_bias, ch_tok0, ch_nv, REC, KT, TBB, GAM, AQB, GVB,
                          C=C, HK=HK, HV=HV, DK=DK, LOGC=C.bit_length() - 1, DT=DT, num_stages=1, num_warps=nw1)
    _ka[((NE + NG) * HV,)](CT, k, v, ch_tok0, ch_nv, AQB, TBB, GVB, HK, state, new_state, REC, KT, UVT, GAM, ZT, T * HV * DK, XP, Sseg, seq_e0, XNL, seg_c0, seg_nch, seq_ne, perm, N, NE, C=C, HV=HV, DK=DK, DT=DT,
                       num_stages=NS, num_warps=nwa)
    _kbc[(max(NMS, 1) * HV,)](FCR, XP, XN, XNL, Sseg, new_state, seq_e0, seq_ne, mseq, ZNT, OL, out, ch_tok0, ch_nv, fc, cnt, scale, NMS, N,
                                 C=C, HV=HV, DK=DK, DT=DT, num_stages=NSB, num_warps=nwb)
    if NE > 0:
        _kc[(max(NCH - S, 1) * HV,)](LCR, ZPT, ZNT, OL, Sseg, out, ch_tok0, ch_nv, ch_e, lc, cnt, scale, C=C, HV=HV, DK=DK, DT=DT, num_stages=1, num_warps=nwc)
    return ZNT, new_state
