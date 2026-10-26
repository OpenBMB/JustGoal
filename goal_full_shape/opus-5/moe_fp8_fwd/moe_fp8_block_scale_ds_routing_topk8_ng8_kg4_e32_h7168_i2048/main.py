"""FP8 block-scale MoE with DeepSeek-V3 no-aux routing, hand-written in Triton.

Pipeline (3 kernels per call):
  1. routing_kernel : sigmoid / group-topk / topk8 / weights, scatters the
                      (token, local expert) pairs into per-expert buckets.
  2. gemm1_kernel   : A[tok] @ W13[e].T -> SwiGLU -> fp8 block-quantised C.
                      Also zeroes the output tensor (it runs before gemm2).
  3. gemm2_kernel   : C @ W2[e].T, scaled by the routing weight, accumulated
                      into the bf16 output with atomics.

Only the routing is data dependent; tile shapes are picked from seq_len, so
correctness never depends on the chosen configuration.
"""
import importlib.util
import os
import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDA = None


def _cuda_mod():
    """Load the CUDA helper module by path (robust to sys.path layout)."""
    global _CUDA
    if _CUDA is None:
        spec = importlib.util.spec_from_file_location(
            "moe_main_cuda", os.path.join(_HERE, "main_cuda.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CUDA = mod
    return _CUDA

H = 7168
I = 2048
E_GLOBAL = 256
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GSZ = E_GLOBAL // N_GROUP

NEG = tl.constexpr(-1.0e30)
KH = tl.constexpr(H)
KI = tl.constexpr(I)
KBLK = tl.constexpr(BLOCK)
EGLOB = tl.constexpr(E_GLOBAL)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
@triton.jit
def routing_kernel(logits_ptr, bias_ptr, cnt_ptr, tokids_ptr, tokw_ptr,
                   T, local_off, scale_f, stride_tok,
                   BT: tl.constexpr, E: tl.constexpr, NG: tl.constexpr,
                   GS: tl.constexpr, TOPK: tl.constexpr, KG: tl.constexpr,
                   EL: tl.constexpr):
    pid = tl.program_id(0)
    toks = pid * BT + tl.arange(0, BT)
    tm = toks < T
    oe = tl.arange(0, E)

    lg = tl.load(logits_ptr + toks[:, None] * E + oe[None, :],
                 mask=tm[:, None], other=0.0).to(tl.float32)
    s = tl.sigmoid(lg)
    bias = tl.load(bias_ptr + oe).to(tl.float32)
    swb = s + bias[None, :]

    # ---- per-group top-2 sum -------------------------------------------- #
    g = tl.reshape(swb, (BT, NG, GS))
    ar_gs = tl.arange(0, GS)
    m1 = tl.max(g, axis=2)
    i1 = tl.argmax(g, axis=2)
    m2 = tl.max(tl.where(ar_gs[None, None, :] == i1[:, :, None], NEG, g), axis=2)
    gscore = m1 + m2

    # ---- keep top-KG groups --------------------------------------------- #
    ar_ng = tl.arange(0, NG)
    keepg = tl.zeros((BT, NG), dtype=tl.float32)
    for _ in tl.static_range(KG):
        gi = tl.argmax(gscore, axis=1)
        sel = ar_ng[None, :] == gi[:, None]
        keepg = tl.where(sel, 1.0, keepg)
        gscore = tl.where(sel, NEG, gscore)
    keep = tl.reshape(tl.broadcast_to(keepg[:, :, None], (BT, NG, GS)), (BT, E))

    # ---- global top-k over kept groups ---------------------------------- #
    pruned = tl.where(keep > 0.0, swb, NEG)
    selm = tl.zeros((BT, E), dtype=tl.float32)
    for _ in tl.static_range(TOPK):
        ei = tl.argmax(pruned, axis=1)
        sel = oe[None, :] == ei[:, None]
        selm = tl.where(sel, 1.0, selm)
        pruned = tl.where(sel, NEG, pruned)

    w = s * selm
    wsum = tl.sum(w, axis=1) + 1e-20
    wn = w * (scale_f / wsum)[:, None]

    # ---- scatter local pairs -------------------------------------------- #
    lidx = oe - local_off
    ok = (lidx >= 0) & (lidx < EL) & (selm > 0.0) & tm[:, None]
    lsafe = tl.where(ok, lidx, 0)
    pos = tl.atomic_add(cnt_ptr + lsafe, 1, mask=ok)
    dst = lsafe * stride_tok + pos
    tl.store(tokids_ptr + dst, tl.broadcast_to(toks[:, None], (BT, E)), mask=ok)
    tl.store(tokw_ptr + dst, wn, mask=ok)


# --------------------------------------------------------------------------- #
# gemm1 + swiglu + fp8 quantise  (+ zero the output buffer)
# --------------------------------------------------------------------------- #
@triton.jit
def _route(logits, bias, T, scale_f, off, TB: tl.constexpr, EL: tl.constexpr):
    """Recompute the whole routing from the logits.

    For short sequences the entire [T, 256] score matrix fits in one small tile,
    so gemm1 can do the routing itself and the separate routing kernel - whose
    ~5 us is almost all fixed latency, plus a kernel launch - disappears.
    Returns the per-(token, local expert) selection, its combination weight and
    the per-expert token counts.
    """
    tar = tl.arange(0, TB)
    a256 = tl.arange(0, EGLOB)
    a32 = tl.arange(0, 32)
    ael = tl.arange(0, EL)
    tmask = tar < T

    lg = tl.load(logits + tar[:, None].to(tl.int64) * EGLOB + a256[None, :],
                 mask=tmask[:, None], other=0.0)
    sg = tl.sigmoid(lg)
    b = tl.load(bias + a256).to(tl.float32)
    v = tl.where(tmask[:, None], sg + b[None, :], NEG)

    # per group: sum of its top two biased scores
    v3 = tl.reshape(v, (TB, 8, 32))
    m1 = tl.max(v3, 2)
    am = tl.argmax(v3, 2)
    m2 = tl.max(tl.where(a32[None, None, :] == am[:, :, None], NEG, v3), 2)
    gs = m1 + m2

    keep = tl.zeros((TB, 8), dtype=tl.int1)
    for _ in range(4):
        gi = tl.argmax(gs, 1)
        hitg = tl.arange(0, 8)[None, :] == gi[:, None]
        keep = keep | hitg
        gs = tl.where(hitg, NEG, gs)

    km = tl.reshape(tl.broadcast_to(keep[:, :, None], (TB, 8, 32)), (TB, EGLOB))
    vk = tl.where(km, v, NEG)

    selm = tl.zeros((TB, EGLOB), dtype=tl.int1)
    wsum = tl.zeros((TB,), dtype=tl.float32)
    for _ in range(8):
        ii = tl.argmax(vk, 1)
        hit = a256[None, :] == ii[:, None]
        selm = selm | hit
        wsum += tl.sum(tl.where(hit, sg, 0.0), 1)
        vk = tl.where(hit, NEG, vk)

    wfull = tl.where(selm, sg * (scale_f / (wsum + 1e-20))[:, None], 0.0)

    # the local experts are one whole group of 32 (the host only takes this
    # path when the offset is a multiple of 32)
    g0 = off // 32
    gm = tl.arange(0, 8)[None, :, None] == g0
    sel_le = tl.sum(tl.where(gm, tl.reshape(selm.to(tl.int32), (TB, 8, 32)), 0), 1)
    w_le = tl.sum(tl.where(gm, tl.reshape(wfull, (TB, 8, 32)), 0.0), 1)
    sel_le = tl.where(tmask[:, None] & (ael[None, :] < EL), sel_le, 0)
    cnt = tl.sum(sel_le, 0)
    pos = tl.cumsum(sel_le, 0) - 1
    return sel_le, w_le, pos, cnt


@triton.jit
def gemm1r_kernel(A, Ascale, W13, S13, Cq, Cs, cnt_ptr, tokids, tokw, out_ptr,
                  logits, bias, scale_f, off,
                  T, TPS, PAIRS, out_numel,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NCTA: tl.constexpr, EL: tl.constexpr, ZB: tl.constexpr,
                  TB: tl.constexpr, PF: tl.constexpr = 1):
    """Fused gate+up gemm1 that also does the routing, for short sequences.

    Measured slower than keeping the separate routing kernel (170 us against
    145 us at seq_len 16): every block redoes the twelve cross-lane argmax
    reductions the selection needs, and that costs more than the ~9 us the
    extra kernel and its launch were worth.  Left in, selectable with RIN, but
    off.
    """
    pid = tl.program_id(0)
    NT: tl.constexpr = KI // BN
    NKB: tl.constexpr = KH // BK
    NSB: tl.constexpr = (BN + KBLK - 1) // KBLK

    zoff = tl.arange(0, ZB)
    zz = tl.zeros((ZB,), dtype=out_ptr.dtype.element_ty)
    nz = (out_numel + ZB - 1) // ZB
    for z in range(pid, nz, NCTA):
        idx = z * ZB + zoff
        tl.store(out_ptr + idx, zz, mask=idx < out_numel)

    are = tl.arange(0, EL)
    tar = tl.arange(0, TB)
    sel_le, w_le, pos, c = _route(logits, bias, T, scale_f, off, TB, EL)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    # hand the routing to gemm2 through the same buffers the routing kernel
    # would have filled; one block per expert, so there is no race
    if pid < EL:
        selc = tl.sum(tl.where(are[None, :] == pid, sel_le, 0), 1)
        posc = tl.sum(tl.where(are[None, :] == pid, pos, 0), 1)
        wc = tl.sum(tl.where(are[None, :] == pid, w_le, 0.0), 1)
        mb = (posc[:, None] == tar[None, :]) & (selc[:, None] > 0)
        tl.store(cnt_ptr + pid, tl.sum(tl.where(are == pid, c, 0)))
        cnt_p = tl.sum(tl.where(are == pid, c, 0))
        tl.store(tokids + pid * TPS + tar,
                 tl.sum(tl.where(mb, tar[:, None], 0), 0), mask=tar < cnt_p)
        tl.store(tokw + pid * TPS + tar,
                 tl.sum(tl.where(mb, wc[:, None], 0.0), 0), mask=tar < cnt_p)

    ar_m = tl.arange(0, BM)
    ar_n = tl.arange(0, BN)
    ar_k = tl.arange(0, BK)

    for wi in range(pid, total * NT, NCTA):
        q = wi // NT
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NT
        nt = loc // mt_e
        ml = loc % mt_e

        rows = ml * BM + ar_m
        rmask = rows < cnt_e
        selc = tl.sum(tl.where(are[None, :] == e, sel_le, 0), 1)
        posc = tl.sum(tl.where(are[None, :] == e, pos, 0), 1)
        mb = (posc[:, None] == rows[None, :]) & (selc[:, None] > 0)
        tok = tl.sum(tl.where(mb, tar[:, None], 0), 0)

        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        # a tile spans NSB 128-wide weight scale blocks (host asserts NSB<=2)
        nb_g = (nt * BN) // KBLK
        nb_u = (KI + nt * BN) // KBLK
        wptr = (W13 + e.to(tl.int64) * (2 * KI * KH)
                + (nt * BN + ar_n)[None, :].to(tl.int64) * KH)
        aptr = A + tok[:, None].to(tl.int64) * KH
        sptr = S13 + e * (32 * 56)
        # the block scales are single small loads per k block; carrying them
        # one iteration ahead keeps them off the critical path
        if PF:
            # the block scales are single small loads per k block that the
            # pipeliner does not prefetch; carrying them one iteration ahead
            # keeps them off the critical path
            sa_n = tl.load(Ascale + tok, mask=rmask, other=0.0)
            g0_n = tl.load(sptr + nb_g * 56)
            u0_n = tl.load(sptr + nb_u * 56)
            if NSB > 1:
                g1_n = tl.load(sptr + (nb_g + 1) * 56)
                u1_n = tl.load(sptr + (nb_u + 1) * 56)
            else:
                g1_n = g0_n
                u1_n = u0_n
        for kb in range(NKB):
            a = tl.load(aptr + kb * BK + ar_k[None, :])
            wg = tl.load(wptr + (kb * BK + ar_k)[:, None])
            wu = tl.load(wptr + KI * KH + (kb * BK + ar_k)[:, None])
            if PF:
                sa = sa_n
                sg0 = g0_n
                su0 = u0_n
                sg1 = g1_n
                su1 = u1_n
                kn = min(kb + 1, NKB - 1)
                sa_n = tl.load(Ascale + kn * T + tok, mask=rmask, other=0.0)
                g0_n = tl.load(sptr + nb_g * 56 + kn)
                u0_n = tl.load(sptr + nb_u * 56 + kn)
                if NSB > 1:
                    g1_n = tl.load(sptr + (nb_g + 1) * 56 + kn)
                    u1_n = tl.load(sptr + (nb_u + 1) * 56 + kn)
            else:
                sa = tl.load(Ascale + kb * T + tok, mask=rmask, other=0.0)
                sg0 = tl.load(sptr + nb_g * 56 + kb)
                su0 = tl.load(sptr + nb_u * 56 + kb)
                if NSB > 1:
                    sg1 = tl.load(sptr + (nb_g + 1) * 56 + kb)
                    su1 = tl.load(sptr + (nb_u + 1) * 56 + kb)
                else:
                    sg1 = sg0
                    su1 = su0
            if NSB == 1:
                acc_g += tl.dot(a, wg) * (sa[:, None] * sg0)
                acc_u += tl.dot(a, wu) * (sa[:, None] * su0)
            else:
                sg = tl.where(ar_n < KBLK, sg0, sg1)
                su = tl.where(ar_n < KBLK, su0, su1)
                acc_g += tl.dot(a, wg) * (sa[:, None] * sg[None, :])
                acc_u += tl.dot(a, wu) * (sa[:, None] * su[None, :])

        cv = (acc_u * tl.sigmoid(acc_u)) * acc_g
        amax = tl.max(tl.abs(cv), axis=1)
        sc = amax * (1.0 / 448.0)
        inv = tl.where(sc > 0.0, 1.0 / tl.where(sc > 0.0, sc, 1.0), 0.0)
        q8 = (cv * inv[:, None]).to(tl.float8e4nv)

        slots = slot_e + rows
        tl.store(Cq + slots[:, None].to(tl.int64) * KI + nt * BN + ar_n[None, :], q8,
                 mask=rmask[:, None])
        tl.store(Cs + nt.to(tl.int64) * PAIRS + slots, sc, mask=rmask)




@triton.jit
def gemm1_kernel(A, Ascale, W13, S13, Cq, Cs, cnt_ptr, tokids, out_ptr, nloc,
                 T, TPS, PAIRS, out_numel,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 NCTA: tl.constexpr, EL: tl.constexpr, ZB: tl.constexpr,
                 PF: tl.constexpr = 1, NLZ: tl.constexpr = 0,
                 G2R: tl.constexpr = 0):
    pid = tl.program_id(0)
    NT: tl.constexpr = KI // BN
    NKB: tl.constexpr = KH // BK
    NSB: tl.constexpr = (BN + KBLK - 1) // KBLK

    # zero the output tensor (same kernel => ordered before gemm2)
    zoff = tl.arange(0, ZB)
    zz = tl.zeros((ZB,), dtype=out_ptr.dtype.element_ty)
    if G2R:
        pass          # the reduce pass writes every output row
    elif NLZ:
        # only rows that gemm2 will not overwrite outright need zeroing
        NZC: tl.constexpr = KH // ZB
        for z in range(pid, T * NZC, NCTA):
            tk = z // NZC
            if tl.load(nloc + tk) != 1:
                tl.store(out_ptr + tk.to(tl.int64) * KH + (z - tk * NZC) * ZB + zoff,
                         zz)
    else:
        nz = (out_numel + ZB - 1) // ZB
        for z in range(pid, nz, NCTA):
            idx = z * ZB + zoff
            tl.store(out_ptr + idx, zz, mask=idx < out_numel)

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM)
    ar_n = tl.arange(0, BN)
    ar_k = tl.arange(0, BK)

    for wi in range(pid, total * NT, NCTA):
        q = wi // NT
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NT
        nt = loc // mt_e
        ml = loc % mt_e

        rows = ml * BM + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)

        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        # a tile spans NSB 128-wide weight scale blocks (host asserts NSB<=2)
        nb_g = (nt * BN) // KBLK
        nb_u = (KI + nt * BN) // KBLK
        wptr = (W13 + e.to(tl.int64) * (2 * KI * KH)
                + (nt * BN + ar_n)[None, :].to(tl.int64) * KH)
        aptr = A + tok[:, None].to(tl.int64) * KH
        sptr = S13 + e * (32 * 56)
        # the block scales are single small loads per k block; carrying them
        # one iteration ahead keeps them off the critical path
        if PF:
            # the block scales are single small loads per k block that the
            # pipeliner does not prefetch; carrying them one iteration ahead
            # keeps them off the critical path
            sa_n = tl.load(Ascale + tok, mask=rmask, other=0.0)
            g0_n = tl.load(sptr + nb_g * 56)
            u0_n = tl.load(sptr + nb_u * 56)
            if NSB > 1:
                g1_n = tl.load(sptr + (nb_g + 1) * 56)
                u1_n = tl.load(sptr + (nb_u + 1) * 56)
            else:
                g1_n = g0_n
                u1_n = u0_n
        for kb in range(NKB):
            a = tl.load(aptr + kb * BK + ar_k[None, :])
            wg = tl.load(wptr + (kb * BK + ar_k)[:, None])
            wu = tl.load(wptr + KI * KH + (kb * BK + ar_k)[:, None])
            if PF:
                sa = sa_n
                sg0 = g0_n
                su0 = u0_n
                sg1 = g1_n
                su1 = u1_n
                kn = min(kb + 1, NKB - 1)
                sa_n = tl.load(Ascale + kn * T + tok, mask=rmask, other=0.0)
                g0_n = tl.load(sptr + nb_g * 56 + kn)
                u0_n = tl.load(sptr + nb_u * 56 + kn)
                if NSB > 1:
                    g1_n = tl.load(sptr + (nb_g + 1) * 56 + kn)
                    u1_n = tl.load(sptr + (nb_u + 1) * 56 + kn)
            else:
                sa = tl.load(Ascale + kb * T + tok, mask=rmask, other=0.0)
                sg0 = tl.load(sptr + nb_g * 56 + kb)
                su0 = tl.load(sptr + nb_u * 56 + kb)
                if NSB > 1:
                    sg1 = tl.load(sptr + (nb_g + 1) * 56 + kb)
                    su1 = tl.load(sptr + (nb_u + 1) * 56 + kb)
                else:
                    sg1 = sg0
                    su1 = su0
            if NSB == 1:
                acc_g += tl.dot(a, wg) * (sa[:, None] * sg0)
                acc_u += tl.dot(a, wu) * (sa[:, None] * su0)
            else:
                sg = tl.where(ar_n < KBLK, sg0, sg1)
                su = tl.where(ar_n < KBLK, su0, su1)
                acc_g += tl.dot(a, wg) * (sa[:, None] * sg[None, :])
                acc_u += tl.dot(a, wu) * (sa[:, None] * su[None, :])

        cv = (acc_u * tl.sigmoid(acc_u)) * acc_g
        amax = tl.max(tl.abs(cv), axis=1)
        sc = amax * (1.0 / 448.0)
        inv = tl.where(sc > 0.0, 1.0 / tl.where(sc > 0.0, sc, 1.0), 0.0)
        q8 = (cv * inv[:, None]).to(tl.float8e4nv)

        slots = slot_e + rows
        tl.store(Cq + slots[:, None].to(tl.int64) * KI + nt * BN + ar_n[None, :], q8,
                 mask=rmask[:, None])
        tl.store(Cs + nt.to(tl.int64) * PAIRS + slots, sc, mask=rmask)


@triton.jit
def gemm1ft_kernel(A, Ascale, W13, S13, Cq, Cs, cnt_ptr, tokids, out_ptr, nloc,
                   T, TPS, PAIRS, out_numel, PCAP,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   NCTA: tl.constexpr, EL: tl.constexpr, ZB: tl.constexpr,
                   PF: tl.constexpr = 1, NLZ: tl.constexpr = 0,
                   G2R: tl.constexpr = 0):
    """Fused gate+up gemm1 with the weights fetched through TMA."""
    pid = tl.program_id(0)
    NT: tl.constexpr = KI // BN
    NKB: tl.constexpr = KH // BK
    NSB: tl.constexpr = (BN + KBLK - 1) // KBLK

    wdesc = tl.make_tensor_descriptor(W13, shape=[EL * 2 * KI, KH],
                                      strides=[KH, 1], block_shape=[BN, BK])
    zoff = tl.arange(0, ZB)
    zz = tl.zeros((ZB,), dtype=out_ptr.dtype.element_ty)
    if G2R:
        pass          # the reduce pass writes every output row
    elif NLZ:
        # only rows that gemm2 will not overwrite outright need zeroing
        NZC: tl.constexpr = KH // ZB
        for z in range(pid, T * NZC, NCTA):
            tk = z // NZC
            if tl.load(nloc + tk) != 1:
                tl.store(out_ptr + tk.to(tl.int64) * KH + (z - tk * NZC) * ZB + zoff,
                         zz)
    else:
        nz = (out_numel + ZB - 1) // ZB
        for z in range(pid, nz, NCTA):
            idx = z * ZB + zoff
            tl.store(out_ptr + idx, zz, mask=idx < out_numel)

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM)
    ar_n = tl.arange(0, BN)
    ar_k = tl.arange(0, BK)

    for wi in range(pid, total * NT, NCTA):
        q = wi // NT
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NT
        nt = loc // mt_e
        ml = loc % mt_e

        rows = ml * BM + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)

        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        nb_g = (nt * BN) // KBLK
        nb_u = (KI + nt * BN) // KBLK
        rg = e * (2 * KI) + nt * BN
        ru = rg + KI
        aptr = A + tok[:, None].to(tl.int64) * KH
        sptr = S13 + e * (32 * 56)
        if PF:
            sa_n = tl.load(Ascale + tok, mask=rmask, other=0.0)
            g0_n = tl.load(sptr + nb_g * 56)
            u0_n = tl.load(sptr + nb_u * 56)
            if NSB > 1:
                g1_n = tl.load(sptr + (nb_g + 1) * 56)
                u1_n = tl.load(sptr + (nb_u + 1) * 56)
            else:
                g1_n = g0_n
                u1_n = u0_n
        for kb in range(NKB):
            a = tl.load(aptr + kb * BK + ar_k[None, :])
            wg = wdesc.load([rg, kb * BK])
            wu = wdesc.load([ru, kb * BK])
            if PF:
                sa = sa_n
                sg0 = g0_n
                su0 = u0_n
                sg1 = g1_n
                su1 = u1_n
                kn = min(kb + 1, NKB - 1)
                sa_n = tl.load(Ascale + kn * T + tok, mask=rmask, other=0.0)
                g0_n = tl.load(sptr + nb_g * 56 + kn)
                u0_n = tl.load(sptr + nb_u * 56 + kn)
                if NSB > 1:
                    g1_n = tl.load(sptr + (nb_g + 1) * 56 + kn)
                    u1_n = tl.load(sptr + (nb_u + 1) * 56 + kn)
            else:
                sa = tl.load(Ascale + kb * T + tok, mask=rmask, other=0.0)
                sg0 = tl.load(sptr + nb_g * 56 + kb)
                su0 = tl.load(sptr + nb_u * 56 + kb)
                if NSB > 1:
                    sg1 = tl.load(sptr + (nb_g + 1) * 56 + kb)
                    su1 = tl.load(sptr + (nb_u + 1) * 56 + kb)
                else:
                    sg1 = sg0
                    su1 = su0
            if NSB == 1:
                acc_g += tl.dot(a, tl.trans(wg)) * (sa[:, None] * sg0)
                acc_u += tl.dot(a, tl.trans(wu)) * (sa[:, None] * su0)
            else:
                sg = tl.where(ar_n < KBLK, sg0, sg1)
                su = tl.where(ar_n < KBLK, su0, su1)
                acc_g += tl.dot(a, tl.trans(wg)) * (sa[:, None] * sg[None, :])
                acc_u += tl.dot(a, tl.trans(wu)) * (sa[:, None] * su[None, :])

        cv = (acc_u * tl.sigmoid(acc_u)) * acc_g
        amax = tl.max(tl.abs(cv), axis=1)
        sc = amax * (1.0 / 448.0)
        inv = tl.where(sc > 0.0, 1.0 / tl.where(sc > 0.0, sc, 1.0), 0.0)
        q8 = (cv * inv[:, None]).to(tl.float8e4nv)

        slots = slot_e + rows
        tl.store(Cq + slots[:, None].to(tl.int64) * KI + nt * BN + ar_n[None, :],
                 q8, mask=rmask[:, None])
        tl.store(Cs + nt.to(tl.int64) * PAIRS + slots, sc, mask=rmask)


# --------------------------------------------------------------------------- #
# gemm1 split: one accumulator per work item (gate and up are separate n-tiles)
# --------------------------------------------------------------------------- #
@triton.jit
def gemm1s_kernel(A, Ascale, W13, S13, G1, cnt_ptr, tokids, out_ptr, nloc,
                  T, TPS, out_numel,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NCTA: tl.constexpr, EL: tl.constexpr, ZB: tl.constexpr,
                  WSP: tl.constexpr = False, UNR: tl.constexpr = 1,
                  NLZ: tl.constexpr = 0, G2R: tl.constexpr = 0):
    pid = tl.program_id(0)
    NT: tl.constexpr = (2 * KI) // BN
    NKB: tl.constexpr = KH // BK
    NSB: tl.constexpr = (BN + KBLK - 1) // KBLK

    zoff = tl.arange(0, ZB)
    zz = tl.zeros((ZB,), dtype=out_ptr.dtype.element_ty)
    if G2R:
        pass          # the reduce pass writes every output row
    elif NLZ:
        # only rows that gemm2 will not overwrite outright need zeroing
        NZC: tl.constexpr = KH // ZB
        for z in range(pid, T * NZC, NCTA):
            tk = z // NZC
            if tl.load(nloc + tk) != 1:
                tl.store(out_ptr + tk.to(tl.int64) * KH + (z - tk * NZC) * ZB + zoff,
                         zz)
    else:
        nz = (out_numel + ZB - 1) // ZB
        for z in range(pid, nz, NCTA):
            idx = z * ZB + zoff
            tl.store(out_ptr + idx, zz, mask=idx < out_numel)

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM)
    ar_n = tl.arange(0, BN)
    ar_k = tl.arange(0, BK)

    for wi in range(pid, total * NT, NCTA):
        q = wi // NT
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NT
        nt = loc // mt_e
        ml = loc - nt * mt_e

        rows = ml * BM + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        wptr = W13 + e.to(tl.int64) * (2 * KI * KH) + (nt * BN + ar_n)[None, :].to(tl.int64) * KH
        aptr = A + tok[:, None].to(tl.int64) * KH
        # a tile spans NSB 128-wide weight scale blocks (host asserts NSB<=2)
        sptr = S13 + e * (32 * 56) + ((nt * BN) // KBLK) * 56
        for kb in tl.range(NKB, warp_specialize=WSP, loop_unroll_factor=UNR):
            a = tl.load(aptr + kb * BK + ar_k[None, :])
            sa = tl.load(Ascale + kb * T + tok, mask=rmask, other=0.0)
            w = tl.load(wptr + (kb * BK + ar_k)[:, None])
            if NSB == 1:
                acc += tl.dot(a, w) * (sa[:, None] * tl.load(sptr + kb))
            else:
                sw = tl.where(ar_n < KBLK, tl.load(sptr + kb),
                              tl.load(sptr + 56 + kb))
                acc += tl.dot(a, w) * (sa[:, None] * sw[None, :])

        slots = slot_e + rows
        tl.store(G1 + slots[:, None].to(tl.int64) * (2 * KI) + nt * BN + ar_n[None, :],
                 acc.to(G1.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def gather_kernel(A, Ascale, Ag, Asg, cnt_ptr, tokids, T, TPS, PCAP,
                  BM: tl.constexpr, BKG: tl.constexpr, NCTA: tl.constexpr,
                  EL: tl.constexpr):
    """Compact the activation rows (and their block scales) into slot order.

    gemm1 then reads both of its operands as dense tiles, which is what lets
    the activation side go through TMA as well.
    """
    pid = tl.program_id(0)
    NCP: tl.constexpr = KH // BKG
    NKB: tl.constexpr = KH // KBLK

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM)
    ar_g = tl.arange(0, BKG)

    for wi in range(pid, total * NCP, NCTA):
        q = wi // NCP
        cp = wi % NCP
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))

        rows = (q - pre_e) * BM + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)
        slots = slot_e + rows

        v = tl.load(A + tok[:, None].to(tl.int64) * KH + cp * BKG + ar_g[None, :])
        tl.store(Ag + slots[:, None].to(tl.int64) * KH + cp * BKG + ar_g[None, :], v)

        if cp == 0:
            for kb in range(NKB):
                sa = tl.load(Ascale + kb * T + tok, mask=rmask, other=0.0)
                tl.store(Asg + kb * PCAP + slots, sa, mask=rmask)


@triton.jit
def gemm1t_kernel(A, Ascale, W13, S13, G1, cnt_ptr, tokids, out_ptr, nloc,
                  T, TPS, out_numel, PCAP,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NCTA: tl.constexpr, EL: tl.constexpr, ZB: tl.constexpr,
                  WSP: tl.constexpr = False, NAC: tl.constexpr = 1,
                  GA: tl.constexpr = 0, RS: tl.constexpr = None,
                  FLT: tl.constexpr = False, KS: tl.constexpr = 1,
                  NLZ: tl.constexpr = 0, G2R: tl.constexpr = 0):
    """gemm1 (split) with the weight operand fetched through TMA.

    The activation rows are a gather over token ids, so they stay on the plain
    load path; the weights are a dense [EL*2I, H] matrix and are the operand
    that dominates the traffic.
    """
    pid = tl.program_id(0)
    NT: tl.constexpr = (2 * KI) // BN
    NKB: tl.constexpr = (KH // BK) // KS
    NSB: tl.constexpr = (BN + KBLK - 1) // KBLK

    wdesc = tl.make_tensor_descriptor(W13, shape=[EL * 2 * KI, KH],
                                      strides=[KH, 1], block_shape=[BN, BK])
    if GA:
        adesc = tl.make_tensor_descriptor(A, shape=[PCAP, KH], strides=[KH, 1],
                                          block_shape=[BM, BK])

    zoff = tl.arange(0, ZB)
    zz = tl.zeros((ZB,), dtype=out_ptr.dtype.element_ty)
    if G2R:
        pass          # the reduce pass writes every output row
    elif NLZ:
        # only rows that gemm2 will not overwrite outright need zeroing
        NZC: tl.constexpr = KH // ZB
        for z in range(pid, T * NZC, NCTA):
            tk = z // NZC
            if tl.load(nloc + tk) != 1:
                tl.store(out_ptr + tk.to(tl.int64) * KH + (z - tk * NZC) * ZB + zoff,
                         zz)
    else:
        nz = (out_numel + ZB - 1) // ZB
        for z in range(pid, nz, NCTA):
            idx = z * ZB + zoff
            tl.store(out_ptr + idx, zz, mask=idx < out_numel)

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    cs = tl.cumsum(mt)
    pre = cs - mt
    pslot = pre * BM
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM)
    ar_n = tl.arange(0, BN)
    ar_k = tl.arange(0, BK)

    for wj in tl.range(pid, total * NT * KS, NCTA, flatten=FLT):
        ks = wj // (total * NT)
        wi = wj - ks * total * NT
        kof = ks * NKB
        q = wi // NT
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NT
        nt = loc // mt_e
        ml = loc - nt * mt_e

        rows = ml * BM + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        aptr = A + tok[:, None].to(tl.int64) * KH
        arow = slot_e + ml * BM
        slots = slot_e + rows
        wrow = e * (2 * KI) + nt * BN
        sptr = S13 + e * (32 * 56) + ((nt * BN) // KBLK) * 56
        if NAC == 1:
            # the block scales are tiny loads that the TMA pipeline does not
            # prefetch, so carry them one iteration ahead by hand; without this
            # every k block stalls on an L1TEX scoreboard dependency
            if GA:
                sa_n = tl.load(Ascale + kof * PCAP + slots, mask=rmask, other=0.0)
            else:
                sa_n = tl.load(Ascale + kof * T + tok, mask=rmask, other=0.0)
            sw_n = tl.load(sptr + kof)
            sv_n = tl.load(sptr + 56 + kof)
            for kb in tl.range(kof, kof + NKB, warp_specialize=WSP, num_stages=RS):
                sa = sa_n
                sw0 = sw_n
                sw1 = sv_n
                kn = min(kb + 1, kof + NKB - 1)
                if GA:
                    a = adesc.load([arow, kb * BK])
                    sa_n = tl.load(Ascale + kn * PCAP + slots, mask=rmask, other=0.0)
                else:
                    a = tl.load(aptr + kb * BK + ar_k[None, :])
                    sa_n = tl.load(Ascale + kn * T + tok, mask=rmask, other=0.0)
                sw_n = tl.load(sptr + kn)
                sv_n = tl.load(sptr + 56 + kn)
                w = wdesc.load([wrow, kb * BK])
                if NSB == 1:
                    acc += tl.dot(a, tl.trans(w)) * (sa[:, None] * sw0)
                else:
                    sw = tl.where(ar_n < KBLK, sw0, sw1)
                    acc += tl.dot(a, tl.trans(w)) * (sa[:, None] * sw[None, :])
        else:
            # two independent accumulator chains so that the MMA of one k block
            # does not have to wait for the fp32 promotion of its predecessor
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
            for kp in tl.range(NKB // 2, warp_specialize=WSP):
                k0 = 2 * kp
                a = tl.load(aptr + k0 * BK + ar_k[None, :])
                b = tl.load(aptr + (k0 + 1) * BK + ar_k[None, :])
                sa = tl.load(Ascale + k0 * T + tok, mask=rmask, other=0.0)
                sb = tl.load(Ascale + (k0 + 1) * T + tok, mask=rmask, other=0.0)
                w = wdesc.load([wrow, k0 * BK])
                x = wdesc.load([wrow, (k0 + 1) * BK])
                if NSB == 1:
                    acc += tl.dot(a, tl.trans(w)) * (sa[:, None] * tl.load(sptr + k0))
                    acc2 += tl.dot(b, tl.trans(x)) * (sb[:, None] * tl.load(sptr + k0 + 1))
                else:
                    sw = tl.where(ar_n < KBLK, tl.load(sptr + k0),
                                  tl.load(sptr + 56 + k0))
                    sx = tl.where(ar_n < KBLK, tl.load(sptr + k0 + 1),
                                  tl.load(sptr + 56 + k0 + 1))
                    acc += tl.dot(a, tl.trans(w)) * (sa[:, None] * sw[None, :])
                    acc2 += tl.dot(b, tl.trans(x)) * (sb[:, None] * sx[None, :])
            acc += acc2

        tl.store(G1 + (ks * PCAP + slots)[:, None].to(tl.int64) * (2 * KI)
                 + nt * BN + ar_n[None, :],
                 acc.to(G1.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def swiglu_kernel(G1, Cq, Cs, cnt_ptr, PAIRS, PCAP,
                  BM: tl.constexpr, BLK: tl.constexpr, NB: tl.constexpr,
                  NCTA: tl.constexpr, EL: tl.constexpr, KS: tl.constexpr = 1):
    pid = tl.program_id(0)
    NIB: tl.constexpr = KI // BLK
    NG: tl.constexpr = NIB // NB
    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    nslot = tl.sum(mt) * BM
    ar = tl.arange(0, NB * BLK)
    arb = tl.arange(0, NB)
    for wi in range(pid, nslot * NG, NCTA):
        slot = wi // NG
        ibg = wi - slot * NG
        base = G1 + slot.to(tl.int64) * (2 * KI) + ibg * (NB * BLK) + ar
        g = tl.load(base).to(tl.float32)
        u = tl.load(base + KI).to(tl.float32)
        for ks in tl.static_range(1, KS):
            off = ks * PCAP.to(tl.int64) * (2 * KI)
            g += tl.load(base + off).to(tl.float32)
            u += tl.load(base + off + KI).to(tl.float32)
        cv = tl.reshape((u * tl.sigmoid(u)) * g, (NB, BLK))
        m = tl.max(tl.abs(cv), axis=1)
        sc = m * (1.0 / 448.0)
        inv = tl.where(sc > 0.0, 1.0 / tl.where(sc > 0.0, sc, 1.0), 0.0)
        q = tl.reshape(cv * inv[:, None], (NB * BLK,))
        tl.store(Cq + slot.to(tl.int64) * KI + ibg * (NB * BLK) + ar,
                 q.to(tl.float8e4nv))
        tl.store(Cs + (ibg * NB + arb).to(tl.int64) * PAIRS + slot, sc)


# --------------------------------------------------------------------------- #
# gemm2 + weighted accumulate
# --------------------------------------------------------------------------- #
@triton.jit
def gemm2_kernel(Cq, Cs, W2, S2, out_ptr, cnt_ptr, cnt_clear_ptr, tokids, tokw,
                 nloc, G2, T, TPS, PAIRS,
                 BM: tl.constexpr, BM2: tl.constexpr, BH: tl.constexpr,
                 BKC: tl.constexpr, NCTA: tl.constexpr, EL: tl.constexpr,
                 WSP: tl.constexpr = False, UNR: tl.constexpr = 1,
                 PF: tl.constexpr = 1, NLZ: tl.constexpr = 0,
                 G2R: tl.constexpr = 0):
    pid = tl.program_id(0)
    NH: tl.constexpr = KH // BH
    NIB: tl.constexpr = KI // BKC
    NSB: tl.constexpr = (BH + KBLK - 1) // KBLK

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    # the slot layout is fixed by gemm1's BM; gemm2 is free to tile the rows of
    # each expert's slot range with its own height BM2, which lets the two
    # GEMMs use the m-tile each one actually wants
    pslot = (tl.cumsum((c + BM - 1) // BM) - (c + BM - 1) // BM) * BM
    mt = (c + BM2 - 1) // BM2
    cs = tl.cumsum(mt)
    pre = cs - mt
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM2)
    ar_h = tl.arange(0, BH)
    ar_kc = tl.arange(0, BKC)

    for wi in range(pid, total * NH, NCTA):
        q = wi // NH
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NH
        ht = loc // mt_e
        ml = loc % mt_e

        rows = ml * BM2 + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)
        slots = slot_e + rows

        acc = tl.zeros((BM2, BH), dtype=tl.float32)
        wptr = W2 + e.to(tl.int64) * (KH * KI) + (ht * BH + ar_h)[None, :].to(tl.int64) * KI
        cptr = Cq + slots[:, None].to(tl.int64) * KI
        s2p = S2 + e * (56 * 16) + ((ht * BH) // KBLK) * 16
        if PF:
            sc_n = tl.load(Cs + slots, mask=rmask, other=0.0)
            s0_n = tl.load(s2p)
            s1_n = tl.load(s2p + 16) if NSB > 1 else s0_n
        for ib in tl.range(NIB, warp_specialize=WSP, loop_unroll_factor=UNR):
            cq = tl.load(cptr + ib * BKC + ar_kc[None, :])
            w2 = tl.load(wptr + (ib * BKC + ar_kc)[:, None])
            if PF:
                sc = sc_n
                s0 = s0_n
                s1 = s1_n
                inx = min(ib + 1, NIB - 1)
                sc_n = tl.load(Cs + inx.to(tl.int64) * PAIRS + slots, mask=rmask,
                               other=0.0)
                s0_n = tl.load(s2p + (inx * BKC) // KBLK)
                if NSB > 1:
                    s1_n = tl.load(s2p + 16 + (inx * BKC) // KBLK)
            else:
                sc = tl.load(Cs + ib.to(tl.int64) * PAIRS + slots, mask=rmask,
                             other=0.0)
                s0 = tl.load(s2p + (ib * BKC) // KBLK)
                s1 = tl.load(s2p + 16 + (ib * BKC) // KBLK) if NSB > 1 else s0
            if NSB == 1:
                acc += tl.dot(cq, w2) * (sc[:, None] * s0)
            else:
                s2v = tl.where(ar_h < KBLK, s0, s1)
                acc += tl.dot(cq, w2) * (sc[:, None] * s2v[None, :])

        wt = tl.load(tokw + e * TPS + rows, mask=rmask, other=0.0)
        vals = acc * wt[:, None]
        optr = out_ptr + tok[:, None].to(tl.int64) * KH + ht * BH + ar_h[None, :]
        if G2R:
            # slots are contiguous within the tile, so this is a plain store
            tl.store(G2 + slots[:, None].to(tl.int64) * KH + ht * BH
                     + ar_h[None, :],
                     vals.to(out_ptr.dtype.element_ty), mask=rmask[:, None])
        elif NLZ:
            # a token with exactly one local expert gets a plain store; the
            # others still accumulate into rows that gemm1 zeroed
            one = tl.load(nloc + tok, mask=rmask, other=0) == 1
            tl.store(optr, vals.to(out_ptr.dtype.element_ty),
                     mask=rmask[:, None] & one[:, None])
            tl.atomic_add(optr, vals.to(out_ptr.dtype.element_ty),
                          mask=rmask[:, None] & (one == 0)[:, None])
        else:
            tl.atomic_add(optr, vals.to(out_ptr.dtype.element_ty),
                          mask=rmask[:, None])

    # reset the counter buffer that the *next* call will use
    if pid < EL:
        tl.store(cnt_clear_ptr + pid, 0)


# --------------------------------------------------------------------------- #
# gemm2 variant whose two operands are fetched with TMA (both are contiguous:
# Cq rows are the compacted (expert, slot) order and W2 is a plain [E*H, I]
# matrix), which is what lets the Blackwell pipeliner warp-specialise the loop
# --------------------------------------------------------------------------- #
@triton.jit
def gemm2t_kernel(Cq, Cs, W2, S2, out_ptr, cnt_ptr, cnt_clear_ptr, tokids, tokw,
                  nloc, G2, T, TPS, PAIRS, PCAP,
                  BM: tl.constexpr, BM2: tl.constexpr, BH: tl.constexpr,
                  BKC: tl.constexpr, NCTA: tl.constexpr, EL: tl.constexpr,
                  WSP: tl.constexpr = False, NAC: tl.constexpr = 1,
                  RS: tl.constexpr = None, FLT: tl.constexpr = False,
                  NLZ: tl.constexpr = 0, UNR: tl.constexpr = 1,
                  G2R: tl.constexpr = 0):
    pid = tl.program_id(0)
    NH: tl.constexpr = KH // BH
    NIB: tl.constexpr = KI // BKC
    NSB: tl.constexpr = (BH + KBLK - 1) // KBLK

    cdesc = tl.make_tensor_descriptor(Cq, shape=[PCAP, KI], strides=[KI, 1],
                                      block_shape=[BM2, BKC])
    wdesc = tl.make_tensor_descriptor(W2, shape=[EL * KH, KI], strides=[KI, 1],
                                      block_shape=[BH, BKC])

    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    # the slot layout is fixed by gemm1's BM; gemm2 is free to tile the rows of
    # each expert's slot range with its own height BM2, which lets the two
    # GEMMs use the m-tile each one actually wants
    pslot = (tl.cumsum((c + BM - 1) // BM) - (c + BM - 1) // BM) * BM
    mt = (c + BM2 - 1) // BM2
    cs = tl.cumsum(mt)
    pre = cs - mt
    total = tl.sum(mt)

    ar_m = tl.arange(0, BM2)
    ar_h = tl.arange(0, BH)

    for wi in tl.range(pid, total * NH, NCTA, flatten=FLT):
        q = wi // NH
        valid = (pre <= q) & (mt > 0)
        e = tl.max(tl.where(valid, are, 0))
        sel_e = are == e
        pre_e = tl.sum(tl.where(sel_e, pre, 0))
        cnt_e = tl.sum(tl.where(sel_e, c, 0))
        slot_e = tl.sum(tl.where(sel_e, pslot, 0))
        mt_e = tl.sum(tl.where(sel_e, mt, 0))
        loc = wi - pre_e * NH
        ht = loc // mt_e
        ml = loc % mt_e

        rows = ml * BM2 + ar_m
        rmask = rows < cnt_e
        tok = tl.load(tokids + e * TPS + rows, mask=rmask, other=0)
        slots = slot_e + rows

        acc = tl.zeros((BM2, BH), dtype=tl.float32)
        s2p = S2 + e * (56 * 16) + ((ht * BH) // KBLK) * 16
        crow = slot_e + ml * BM2
        wrow = e * KH + ht * BH
        if NAC == 1:
            sc_n = tl.load(Cs + slots, mask=rmask, other=0.0)
            sa_n = tl.load(s2p)
            sb_n = tl.load(s2p + 16)
            for ib in tl.range(NIB, warp_specialize=WSP, num_stages=RS,
                               loop_unroll_factor=UNR):
                sc = sc_n
                s0 = sa_n
                s1 = sb_n
                inx = min(ib + 1, NIB - 1)
                cq = cdesc.load([crow, ib * BKC])
                w2 = wdesc.load([wrow, ib * BKC])
                sc_n = tl.load(Cs + inx.to(tl.int64) * PAIRS + slots, mask=rmask,
                               other=0.0)
                sa_n = tl.load(s2p + (inx * BKC) // KBLK)
                sb_n = tl.load(s2p + 16 + (inx * BKC) // KBLK)
                if NSB == 1:
                    acc += tl.dot(cq, tl.trans(w2)) * (sc[:, None] * s0)
                else:
                    s2v = tl.where(ar_h < KBLK, s0, s1)
                    acc += tl.dot(cq, tl.trans(w2)) * (sc[:, None] * s2v[None, :])
        else:
            acc2 = tl.zeros((BM2, BH), dtype=tl.float32)
            for ip in tl.range(NIB // 2, warp_specialize=WSP):
                i0 = 2 * ip
                cq = cdesc.load([crow, i0 * BKC])
                cr = cdesc.load([crow, (i0 + 1) * BKC])
                w2 = wdesc.load([wrow, i0 * BKC])
                w3 = wdesc.load([wrow, (i0 + 1) * BKC])
                sc = tl.load(Cs + i0 * PAIRS + slots, mask=rmask, other=0.0)
                sd = tl.load(Cs + (i0 + 1) * PAIRS + slots, mask=rmask, other=0.0)
                if NSB == 1:
                    acc += tl.dot(cq, tl.trans(w2)) * (
                        sc[:, None] * tl.load(s2p + (i0 * BKC) // KBLK))
                    acc2 += tl.dot(cr, tl.trans(w3)) * (
                        sd[:, None] * tl.load(s2p + ((i0 + 1) * BKC) // KBLK))
                else:
                    s2v = tl.where(ar_h < KBLK, tl.load(s2p + (i0 * BKC) // KBLK),
                                   tl.load(s2p + 16 + (i0 * BKC) // KBLK))
                    s2w = tl.where(ar_h < KBLK,
                                   tl.load(s2p + ((i0 + 1) * BKC) // KBLK),
                                   tl.load(s2p + 16 + ((i0 + 1) * BKC) // KBLK))
                    acc += tl.dot(cq, tl.trans(w2)) * (sc[:, None] * s2v[None, :])
                    acc2 += tl.dot(cr, tl.trans(w3)) * (sd[:, None] * s2w[None, :])
            acc += acc2

        wt = tl.load(tokw + e * TPS + rows, mask=rmask, other=0.0)
        vals = acc * wt[:, None]
        optr = out_ptr + tok[:, None].to(tl.int64) * KH + ht * BH + ar_h[None, :]
        if G2R:
            # slots are contiguous within the tile, so this is a plain store
            tl.store(G2 + slots[:, None].to(tl.int64) * KH + ht * BH
                     + ar_h[None, :],
                     vals.to(out_ptr.dtype.element_ty), mask=rmask[:, None])
        elif NLZ:
            # a token with exactly one local expert gets a plain store; the
            # others still accumulate into rows that gemm1 zeroed
            one = tl.load(nloc + tok, mask=rmask, other=0) == 1
            tl.store(optr, vals.to(out_ptr.dtype.element_ty),
                     mask=rmask[:, None] & one[:, None])
            tl.atomic_add(optr, vals.to(out_ptr.dtype.element_ty),
                          mask=rmask[:, None] & (one == 0)[:, None])
        else:
            tl.atomic_add(optr, vals.to(out_ptr.dtype.element_ty),
                          mask=rmask[:, None])

    if pid < EL:
        tl.store(cnt_clear_ptr + pid, 0)


@triton.jit
def reduce_kernel(G2, out_ptr, nloc, tslot, cnt_ptr, T, TP, BM: tl.constexpr,
                  BR: tl.constexpr, NCTA: tl.constexpr, EL: tl.constexpr):
    """Sum each token's per-expert partial outputs.

    gemm2 can then store its tile instead of accumulating it with bf16 atomics,
    which on this shape costs about 140 us at seq_len 11948 - more than this
    extra pass, because an atomic read-modify-write moves the same bytes at a
    fraction of the streaming rate.

    One work item is one token's whole output row, held as an (NR, BR) tile so
    that the per-token slot lookup is done once rather than once per chunk.
    """
    pid = tl.program_id(0)
    NR: tl.constexpr = triton.next_power_of_2(KH // BR)
    are = tl.arange(0, EL)
    c = tl.load(cnt_ptr + are)
    mt = (c + BM - 1) // BM
    pslot = (tl.cumsum(mt) - mt) * BM
    off = tl.arange(0, NR)[:, None] * BR + tl.arange(0, BR)[None, :]
    msk = off < KH

    for t in range(pid, T, NCTA):
        nl = tl.load(nloc + t)
        sel = tl.load(tslot + t * EL + are)
        hit = sel > 0
        rank = tl.cumsum(hit.to(tl.int32)) - 1
        acc = tl.zeros((NR, BR), dtype=tl.float32)
        for j in range(nl):
            pick = hit & (rank == j)
            base = tl.sum(tl.where(pick, pslot + sel - 1, 0))
            acc += tl.load(G2 + base.to(tl.int64) * KH + off, mask=msk,
                           other=0.0).to(tl.float32)
        tl.store(out_ptr + t.to(tl.int64) * KH + off,
                 acc.to(out_ptr.dtype.element_ty), mask=msk)


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #
_WS = {}
_ALLOC = [None]


def _tma_scratch(size, align, stream):
    """Scratch for device-side TMA descriptors (shape independent, one-off)."""
    buf = _ALLOC[0]
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 20), dtype=torch.int8, device="cuda")
        _ALLOC[0] = buf
    return buf


triton.set_allocator(_tma_scratch)

DEFAULT = dict(BM=16, BN=128, BK=128, BH=128, W1=4, S1=3, WG2=4, S2=3,
               NC1=592, NC2=592, ZB=1024, SPLIT=0, BNS=128, WS=8, SS=4, NCS=592,
               SWNB=8, SWW=4, WSP=0, UNR=1, WSP2=0, UNR2=1,
               TMA1=1, NAC=0, NAC2=0, TMA2=0, TMA1F=0,
               CT1=1, CT2=1, GATH=0, BKG=512, NCG=592, GW=4, RS1=0, RS2=0,
               PF1=0, PF2=0, FLT1=0, FLT2=0, RK=1, KS=1, NLZ=0, BM2=0, G2R=0, BR=1024, NCR=592, RW=4,
               RIN=0)

# Per-seq_len tuned configuration (shape only, never data dependent).
_TUNE = [
    (4, dict(BM=16, BN=64, BH=128, W1=4, WG2=4)),
    (128, dict(BM=16, BN=128, BH=128, W1=4, WG2=8)),
    (2048, dict(BM=64, BN=128, BH=128, W1=8, S1=4, WG2=8, S2=4,
                SPLIT=1, BNS=128, WS=8, SS=4, TMA2=1)),
    (1 << 30, dict(BM=64, BN=128, BH=128, W1=8, S1=4, WG2=8, S2=4,
                   SPLIT=1, BNS=128, WS=8, SS=4, TMA2=1, G2R=1,
                   NCS=296, NC2=296, NCG=888, NCR=888)),
]


# Per-shape overrides measured on this card (shape only, never data dependent);
# any sequence length not listed falls back to the range table above.
_EXACT = {
    1: dict(BH=64, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=2, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=0, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=5, SPLIT=1, SS=5, SWNB=4, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    7: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=64, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=1, PF2=0, RK=1, RS1=0, RS2=0, RW=4, S1=4, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=8, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    14: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=512, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=1, PF2=0, RK=1, RS1=0, RS2=0, RW=8, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=8, WS=8, WSP=0, WSP2=0, ZB=1024),
    15: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=2, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=4, SPLIT=1, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    16: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=512, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=1, PF2=0, RK=1, RS1=0, RS2=0, RW=8, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=8, WS=8, WSP=0, WSP2=0, ZB=1024),
    32: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=1, SS=5, SWNB=4, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    52: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=1, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=8, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    53: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=1, SS=5, SWNB=16, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    54: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=1, SS=5, SWNB=4, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    55: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=1, SS=5, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    56: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=1, SS=5, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    57: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    58: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=8, S1=3, S2=3, SPLIT=1, SS=5, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    59: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=8, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    62: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=1, PF1=1, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=4, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    80: dict(BH=128, BK=128, BKG=512, BM=16, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=0, GATH=0, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=592, NCG=592, NCR=592, NCS=592, NLZ=0, PF1=1, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=3, S2=3, SPLIT=0, SS=4, SWNB=8, SWW=4, TMA1=1, TMA1F=0, TMA2=0, UNR=1, UNR2=1, W1=4, WG2=4, WS=8, WSP=0, WSP2=0, ZB=1024),
    901: dict(BH=128, BK=128, BKG=512, BM=64, BM2=0, BN=128, BNS=128, BR=512, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=1, GATH=1, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=296, NCG=888, NCR=888, NCS=296, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=8, S1=4, S2=4, SPLIT=1, SS=4, SWNB=16, SWW=8, TMA1=1, TMA1F=1, TMA2=1, UNR=1, UNR2=1, W1=8, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    11948: dict(BH=128, BK=128, BKG=512, BM=64, BM2=0, BN=128, BNS=128, BR=512, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=1, GATH=1, GW=8, KS=1, NAC=0, NAC2=0, NC1=592, NC2=296, NCG=888, NCR=888, NCS=296, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=8, S1=4, S2=4, SPLIT=1, SS=4, SWNB=16, SWW=8, TMA1=1, TMA1F=1, TMA2=1, UNR=1, UNR2=1, W1=8, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
    14107: dict(BH=128, BK=128, BKG=512, BM=64, BM2=0, BN=128, BNS=128, BR=1024, CT1=1, CT2=1, FLT1=0, FLT2=0, G2R=1, GATH=1, GW=4, KS=1, NAC=0, NAC2=0, NC1=592, NC2=296, NCG=888, NCR=888, NCS=296, NLZ=0, PF1=0, PF2=1, RK=1, RS1=0, RS2=0, RW=4, S1=4, S2=4, SPLIT=1, SS=4, SWNB=16, SWW=4, TMA1=1, TMA1F=1, TMA2=1, UNR=1, UNR2=1, W1=8, WG2=4, WS=4, WSP=0, WSP2=0, ZB=1024),
}


def _cfg(T):
    c = dict(DEFAULT)
    for tmax, upd in _TUNE:
        if T <= tmax:
            c.update(upd)
            break
    if T in _EXACT:
        c.update(_EXACT[T])
    env = os.environ.get("MOE_CFG")
    if env:
        for kv in env.split(","):
            k, v = kv.split("=")
            c[k] = int(v)
    # the kernels read one scalar weight scale per n/h tile, which is only
    # valid when a tile cannot straddle two 128-wide scale blocks
    for k in ("BN", "BH", "BNS"):
        v = c[k]
        assert (BLOCK % v == 0) or (v % BLOCK == 0 and v <= 2 * BLOCK), \
            "tile width %s=%d must divide %d or be 2x it" % (k, v, BLOCK)
    # gemm2's k tile (BKC) reads one W2 scale per 128-wide k block, so it may
    # not straddle two of them; BKC is 128 on the split path and BN otherwise
    assert c["SPLIT"] or BLOCK % c["BN"] == 0
    # the k split needs the split path, a plain accumulator and an exact
    # division of the 56 quantisation blocks along the hidden dimension
    assert c["KS"] == 1 or (c["SPLIT"] and c["NAC"] == 0
                            and (H // c["BK"]) % c["KS"] == 0)
    # the "skip zeroing / store instead of accumulate" path needs the per-token
    # local expert count, which only the redux routing kernel produces, and a
    # zeroing chunk that divides the hidden size
    assert not c["NLZ"] or (c["RK"] == 1 and H % c["ZB"] == 0)
    # every tiling that is turned into a constexpr trip count has to divide the
    # dimension it walks, otherwise the kernel silently covers only part of it
    assert H % c["BH"] == 0, "BH must divide H"
    assert I % c["BN"] == 0 and (2 * I) % c["BNS"] == 0
    assert H % c["BK"] == 0, "BK must divide H"
    assert H % c["BKG"] == 0, "the activation gather chunk must divide H"
    assert H % c["BR"] == 0, "the reduce chunk must divide H"
    assert not c["G2R"] or c["RK"] == 1, "the reduce path needs the inverse map"
    assert (I // BLOCK) % c["SWNB"] == 0, "SWNB must divide I/128"
    assert I % (c["BN"] if not c["SPLIT"] else BLOCK) == 0
    return c


def _ws(T, device):
    w = _WS.get(T)
    if w is None:
        c = _cfg(T)
        BM, BN = c["BM"], c["BN"]
        BM2 = c["BM2"] or BM
        # gemm2's taller tiles may read a little past the last expert's slots;
        # those rows are masked out of the store, but they must stay in bounds
        pairs_cap = min(8 * T, 32 * T) + 32 * BM + 2 * BM2
        tp = (T + 3) & ~3
        w = dict(
            cfg=c, tp=tp, par=0,
            cnt=torch.zeros((2, 32), dtype=torch.int32, device=device),
            tokids=torch.zeros((33, tp), dtype=torch.int32, device=device),
            tokw=torch.zeros((33, tp), dtype=torch.float32, device=device),
            ascg=torch.zeros(1, dtype=torch.float32, device=device),
            nloc=(torch.zeros(tp, dtype=torch.int32, device=device)
                  if (c["NLZ"] or c["G2R"]) else
                  torch.zeros(1, dtype=torch.int32, device=device)),
            tslot=(torch.zeros(tp * 32, dtype=torch.int32, device=device)
                   if c["G2R"] else torch.zeros(1, dtype=torch.int32,
                                                device=device)),
            G2=(torch.zeros((pairs_cap, H), dtype=torch.bfloat16, device=device)
                if c["G2R"] else torch.zeros(1, dtype=torch.bfloat16,
                                             device=device)),
            Cq=torch.zeros((pairs_cap, I), dtype=torch.float8_e4m3fn, device=device),
            Cs=torch.zeros((64, pairs_cap), dtype=torch.float32, device=device),
            G1=(torch.zeros((c["KS"] * pairs_cap, 2 * I),
                            dtype=(torch.float32 if c["KS"] > 1 else torch.bfloat16),
                            device=device)
                if c["SPLIT"] else torch.zeros(1, dtype=torch.bfloat16, device=device)),
            Ag=(torch.zeros((pairs_cap, H), dtype=torch.float8_e4m3fn, device=device)
                if c["GATH"] else torch.zeros(1, dtype=torch.float8_e4m3fn,
                                              device=device)),
            Asg=(torch.zeros((56, pairs_cap), dtype=torch.float32, device=device)
                 if c["GATH"] else torch.zeros(1, dtype=torch.float32, device=device)),
            pairs_cap=pairs_cap,
        )
        _WS[T] = w
    return w


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    T = routing_logits.shape[0]
    EL = gemm1_weights.shape[0]
    dev = hidden_states.device
    w = _ws(T, dev)
    c = w["cfg"]
    BM, BN, BH = c["BM"], c["BN"], c["BH"]
    pairs_cap = w["pairs_cap"]
    tp = w["tp"]

    # The expert counters ping-pong: gemm2 clears the buffer the *next* call
    # will use, so no per-call memset kernel is needed.  The parity lives in the
    # workspace, not in a global, because that invariant only holds per
    # workspace - a caller that alternates sequence lengths would otherwise get
    # counters that were never cleared.
    par = w["par"]
    w["par"] = 1 - par
    cnt = w["cnt"][par]
    cnt_clear = w["cnt"][1 - par]

    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)

    # Everything the host has to do is done before the first kernel is issued:
    # the launch arguments and the cached CompiledKernel lookups.  Otherwise the
    # GPU sits idle between the short routing kernel and gemm1 while the host
    # prepares the next launch.
    ub_mt = min(32 * ((T + BM - 1) // BM), 32 + (8 * T) // BM)
    # the inline-routing gemm1 holds the whole [T, 256] score matrix in one
    # tile, and its local-expert slice has to be one aligned group of 32
    inline_route = (c["RIN"] and not c["SPLIT"]
                    and int(local_expert_offset) % 32 == 0
                    and EL == 32 and T <= c["RIN"])
    plan = []

    if c["SPLIT"]:
        BNS = c["BNS"]
        ncs = max(min(ub_mt * ((2 * I) // BNS) * c["KS"], c["NCS"]), 148)
        a1 = (hidden_states, hidden_states_scale, gemm1_weights,
              gemm1_weights_scale, w["G1"], cnt, w["tokids"], out, w["nloc"],
              T, tp, T * H)
        if c["GATH"]:
            ncg = max(min(ub_mt * (H // c["BKG"]), c["NCG"]), 148)
            plan.append((gather_kernel,
                         ("gath", BM, c["BKG"], ncg, EL, c["GW"]), ncg,
                         (hidden_states, hidden_states_scale, w["Ag"], w["Asg"],
                          cnt, w["tokids"], T, tp, pairs_cap),
                         dict(BM=BM, BKG=c["BKG"], NCTA=ncg, EL=EL,
                              num_warps=c["GW"])))
            a1 = (w["Ag"], w["Asg"]) + a1[2:]
        if c["TMA1"]:
            plan.append((gemm1t_kernel,
                         ("g1t", BM, BNS, ncs, EL, c["BK"], c["ZB"], c["WS"],
                          c["SS"], c["WSP"], c["NAC"], c["CT1"], c["GATH"],
                          c["RS1"], c["FLT1"], c["KS"], c["NLZ"], c["G2R"]),
                         ncs, a1 + (pairs_cap,),
                         dict(BM=BM, BN=BNS, BK=c["BK"], NCTA=ncs, EL=EL,
                              ZB=c["ZB"], WSP=bool(c["WSP"]), NAC=1 + c["NAC"],
                              num_ctas=c["CT1"], GA=c["GATH"],
                              RS=(c["RS1"] or None), FLT=bool(c["FLT1"]),
                              KS=c["KS"], NLZ=c["NLZ"], G2R=c["G2R"],
                              num_warps=c["WS"], num_stages=c["SS"])))
        else:
            plan.append((gemm1s_kernel,
                         ("g1s", BM, BNS, ncs, EL, c["BK"], c["ZB"], c["WS"],
                          c["SS"], c["WSP"], c["UNR"], c["NLZ"], c["G2R"]),
                         ncs, a1,
                         dict(BM=BM, BN=BNS, BK=c["BK"], NCTA=ncs, EL=EL,
                              ZB=c["ZB"], WSP=bool(c["WSP"]), UNR=c["UNR"],
                              NLZ=c["NLZ"], G2R=c["G2R"],
                              num_warps=c["WS"], num_stages=c["SS"])))
        plan.append((swiglu_kernel,
                     ("sw", BM, EL, c["SWNB"], c["SWW"], c["KS"]), 592,
                     (w["G1"], w["Cq"], w["Cs"], cnt, pairs_cap, pairs_cap),
                     dict(BM=BM, BLK=128, NB=c["SWNB"], NCTA=592, EL=EL,
                          KS=c["KS"], num_warps=c["SWW"])))
        bkc = 128
    else:
        ncta1 = max(min(ub_mt * (I // BN), c["NC1"]), 148)
        if inline_route:
            tb = max(1 << (T - 1).bit_length(), 1)
            plan.append((gemm1r_kernel,
                         ("g1r", BM, BN, ncta1, EL, c["BK"], c["ZB"], c["W1"],
                          c["S1"], c["PF1"], tb),
                         ncta1,
                         (hidden_states, hidden_states_scale, gemm1_weights,
                          gemm1_weights_scale, w["Cq"], w["Cs"], cnt,
                          w["tokids"], w["tokw"], out, routing_logits,
                          routing_bias, float(routed_scaling_factor),
                          int(local_expert_offset), T, tp, pairs_cap, T * H),
                         dict(BM=BM, BN=BN, BK=c["BK"], NCTA=ncta1, EL=EL,
                              ZB=c["ZB"], TB=tb, PF=c["PF1"],
                              num_warps=c["W1"], num_stages=c["S1"])))
            bkc = BN
            a1 = None
        else:
            a1 = (hidden_states, hidden_states_scale, gemm1_weights,
                  gemm1_weights_scale, w["Cq"], w["Cs"], cnt, w["tokids"], out,
                  w["nloc"], T, tp, pairs_cap, T * H)
        if inline_route:
            pass
        elif c["TMA1F"]:
            plan.append((gemm1ft_kernel,
                         ("g1ft", BM, BN, ncta1, EL, c["BK"], c["ZB"], c["W1"],
                          c["S1"], c["PF1"], c["NLZ"], c["G2R"]),
                         ncta1, a1 + (pairs_cap,),
                         dict(BM=BM, BN=BN, BK=c["BK"], NCTA=ncta1, EL=EL,
                              ZB=c["ZB"], PF=c["PF1"], NLZ=c["NLZ"],
                              G2R=c["G2R"],
                              num_warps=c["W1"], num_stages=c["S1"])))
        else:
            plan.append((gemm1_kernel,
                         ("g1", BM, BN, ncta1, EL, c["BK"], c["ZB"], c["W1"],
                          c["S1"], c["PF1"], c["NLZ"], c["G2R"]),
                         ncta1, a1,
                         dict(BM=BM, BN=BN, BK=c["BK"], NCTA=ncta1, EL=EL,
                              ZB=c["ZB"], PF=c["PF1"], NLZ=c["NLZ"],
                              G2R=c["G2R"],
                              num_warps=c["W1"], num_stages=c["S1"])))
        bkc = BN

    BM2 = c["BM2"] or BM
    ub_mt2 = min(32 * ((T + BM2 - 1) // BM2), 32 + (8 * T) // BM2)
    ncta2 = max(min(ub_mt2 * (H // BH), c["NC2"]), 148)
    a2 = (w["Cq"], w["Cs"], gemm2_weights, gemm2_weights_scale, out, cnt,
          cnt_clear, w["tokids"], w["tokw"], w["nloc"], w["G2"], T, tp,
          pairs_cap)
    if c["TMA2"]:
        plan.append((gemm2t_kernel,
                     ("g2t", BM, BM2, BH, bkc, ncta2, EL, c["WG2"], c["S2"],
                      c["WSP2"], c["NAC2"], c["CT2"], c["RS2"], c["FLT2"],
                      c["NLZ"], c["UNR2"], c["G2R"]),
                     ncta2, a2 + (pairs_cap,),
                     dict(BM=BM, BM2=BM2, BH=BH, BKC=bkc, NCTA=ncta2, EL=EL,
                          WSP=bool(c["WSP2"]), NAC=1 + c["NAC2"],
                          num_ctas=c["CT2"], RS=(c["RS2"] or None),
                          FLT=bool(c["FLT2"]), NLZ=c["NLZ"], UNR=c["UNR2"],
                          G2R=c["G2R"],
                          num_warps=c["WG2"], num_stages=c["S2"])))
    else:
        plan.append((gemm2_kernel,
                     ("g2", BM, BM2, BH, bkc, ncta2, EL, c["WG2"], c["S2"],
                      c["WSP2"], c["UNR2"], c["PF2"], c["NLZ"], c["G2R"]),
                     ncta2, a2,
                     dict(BM=BM, BM2=BM2, BH=BH, BKC=bkc, NCTA=ncta2, EL=EL,
                          WSP=bool(c["WSP2"]), UNR=c["UNR2"], PF=c["PF2"],
                          NLZ=c["NLZ"], G2R=c["G2R"],
                          num_warps=c["WG2"], num_stages=c["S2"])))

    # bind the grid now as well, so that after the routing kernel is issued the
    # host only has to make the launch calls themselves
    if c["G2R"]:
        ncr = max(min(T, c["NCR"]), 148)
        plan.append((reduce_kernel,
                     ("red", BM, c["BR"], ncr, EL, c["RW"]), ncr,
                     (w["G2"], out, w["nloc"], w["tslot"], cnt, T, tp),
                     dict(BM=BM, BR=c["BR"], NCTA=ncr, EL=EL,
                          num_warps=c["RW"])))

    if not inline_route:
        _cuda_mod().routing_only(routing_logits, routing_bias,
                                 hidden_states_scale, cnt, w["tokids"],
                                 w["tokw"], w["ascg"], w["nloc"], w["tslot"],
                                 int(local_expert_offset),
                                 float(routed_scaling_factor), EL, 0, c["RK"])

    # Use Triton's public launcher; it owns constexpr ABI and specialization caching.
    for fn, key, grid, args, cx in plan:
        fn[(grid,)](*args, **cx)
    return out
