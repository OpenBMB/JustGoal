# MoE FP8 block-scale (DeepSeek-V3 routing) for B200 / sm_100.
# Pipeline: routing -> expert bucket offsets -> permutation -> activation gather
#           -> grouped GEMM1 + SwiGLU + fp8 requant -> grouped GEMM2 + weighted scatter.
# All matmuls are hand-written Triton kernels (tcgen05 via tl.dot); torch is glue only.
import os
import torch
import triton
import triton.language as tl

_SCRATCH = {}

# Hand-written routing kernel (warp-per-token, all reductions via shuffles).  Compilation
# is shape- and hardware-independent one-time work; if it fails we fall back to the Triton
# routing kernel, which computes the same thing.
_CU = None
try:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    from torch.utils.cpp_extension import load as _load
    _CU = _load(name="moe_route",
                sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
                         for f in ("route_host.cu", "route.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
except Exception:
    _CU = None


def _alloc(size, align, stream):
    buf = _SCRATCH.get("tma")
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 20), dtype=torch.int8, device="cuda")
        _SCRATCH["tma"] = buf
    return buf


triton.set_allocator(_alloc)

BM = 128
NSM = 148
CFG = dict(g1_st=4, g1_w=8, g1_nbk=1, g1_ep=1, g2_bn=256, g2_st=4, g2_w=8, rt_bt=1,
           rt_w=1, rt_zb=256, rt_zob=1024, flat=0, g1_gather=0, g1_nzc=8, gr_prog=64, gr_w=8, gr_seq=0, gs_bp=32, gs_w=4, g1_prog=1, g2_prog=1, atomic=1, zout=1, g1_ord=0, g2_ord=1, g1_gm=8, g2_gm=8, graph=1, g1_os=1, g2_os=1, rt_cu=1)


# ---------------------------------------------------------------- routing ----
@triton.jit
def k_route(logits_ptr, bias_ptr, pt_ptr, p2, p3, p4, p5, p6, p7, tokw_ptr, tokr_ptr, cnt_ptr, ptok_ptr, pw_ptr, out_ptr,
            flag_ptr, base_ptr, mt_ptr, NOUT, T,
            E: tl.constexpr, EL: tl.constexpr, ELP: tl.constexpr, NGRP: tl.constexpr,
            KGRP: tl.constexpr, TOPK: tl.constexpr, OFF, RSF, PMAX, NPROG,
            BT: tl.constexpr, GS: tl.constexpr, ZB: tl.constexpr, BMC: tl.constexpr,
            ZPT: tl.constexpr, ZOB: tl.constexpr, HH: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:        # publish the addresses the graph-replayed kernels read
        tl.store(pt_ptr + 2, p2)
        tl.store(pt_ptr + 3, p3)
        tl.store(pt_ptr + 4, p4)
        tl.store(pt_ptr + 5, p5)
        tl.store(pt_ptr + 6, p6)
        tl.store(pt_ptr + 7, p7)
    offt = pid * BT + tl.arange(0, BT)
    mt = offt < T
    offe = tl.arange(0, E)
    x = tl.load(logits_ptr + offt[:, None] * E + offe[None, :], mask=mt[:, None], other=0.0)
    s = tl.sigmoid(x)
    b = tl.load(bias_ptr + offe).to(tl.float32)
    swb = s + b[None, :]
    g = tl.reshape(swb, [BT, NGRP, GS])
    # group score = sum of the top-2 values in the group (multiplicity-aware)
    m1 = tl.max(g, axis=2)
    c1 = tl.sum((g == m1[:, :, None]).to(tl.int32), axis=2)
    m2b = tl.max(tl.where(g < m1[:, :, None], g, float("-inf")), axis=2)
    gs = m1 + tl.where(c1 >= 2, m1, m2b)
    # top-KGRP groups; ties resolved by lowest index (matches torch.topk)
    ng = tl.arange(0, NGRP)
    gsel = tl.zeros([BT, NGRP], dtype=tl.int1)
    cur = gs
    for _ in tl.static_range(KGRP):
        j = tl.argmax(cur, axis=1)
        hit = ng[None, :] == j[:, None]
        gsel = gsel | hit
        cur = tl.where(hit, float("-inf"), cur)
    gmask = tl.reshape(tl.broadcast_to(gsel[:, :, None], [BT, NGRP, GS]), [BT, E])
    pr = tl.where(gmask, swb, float("-inf"))
    # top-TOPK experts within the kept groups
    sel = tl.zeros([BT, E], dtype=tl.int1)
    for _ in tl.static_range(TOPK):
        j = tl.argmax(pr, axis=1)
        hit = offe[None, :] == j[:, None]
        sel = sel | hit
        pr = tl.where(hit, float("-inf"), pr)
    w = tl.where(sel, s, 0.0)
    w = w / (tl.sum(w, axis=1) + 1e-20)[:, None] * RSF
    isloc = (offe >= OFF) & (offe < OFF + EL)
    le = offe - OFF
    i = tl.arange(0, ELP)
    # tokens with exactly one local expert get a negative weight: gemm2 then writes their
    # output row with a plain store instead of an atomic add, and it needs no pre-zeroing.
    nl = tl.sum(tl.where(isloc[None, :] & sel, 1, 0), axis=1)
    tl.store(tokw_ptr + offt[:, None] * ELP + le[None, :],
             tl.where((nl == 1)[:, None], -w, w), mask=mt[:, None] & isloc[None, :])
    # the counter atomic's return value gives each slot its rank inside its expert bucket,
    # exactly as the hand-written kernel does, so k_perm needs no atomics
    seli = tl.where(isloc[None, :] & sel & mt[:, None], 1, 0)
    cc = tl.sum(seli, axis=0)
    rsv = tl.atomic_add(cnt_ptr + le, cc, mask=isloc & (cc > 0))
    tl.store(tokr_ptr + offt[:, None] * ELP + le[None, :],
             rsv[None, :] + tl.cumsum(seli, axis=0) - seli,
             mask=mt[:, None] & isloc[None, :])
    # clear the permutation arrays (padding slots must read as token 0 / weight 0)
    zi = tl.zeros([ZB], dtype=tl.int32)
    zf = tl.zeros([ZB], dtype=tl.float32)
    for jj in tl.range(0, (PMAX + NPROG * ZB - 1) // (NPROG * ZB)):
        oo = pid * ZB + jj * (NPROG * ZB) + tl.arange(0, ZB)
        tl.store(ptok_ptr + oo, zi, mask=oo < PMAX)
        tl.store(pw_ptr + oo, zf, mask=oo < PMAX)
    # clear the atomic-accumulation target for gemm2 (only rows that need accumulation)
    tl.static_assert(HH % ZOB == 0)      # the per-row clear must cover the whole row
    zb = tl.zeros([ZOB], dtype=tl.bfloat16)
    if ZPT:                        # one row per program (BT == 1); skip single-expert rows
        if tl.sum(nl) != 1:
            for jj in tl.range(0, HH // ZOB):
                tl.store(out_ptr + pid * HH + jj * ZOB + tl.arange(0, ZOB), zb)
    else:
        for jj in tl.range(0, (NOUT + NPROG * ZOB - 1) // (NPROG * ZOB)):
            oo = pid * ZOB + jj * (NPROG * ZOB) + tl.arange(0, ZOB)
            tl.store(out_ptr + oo, zb, mask=oo < NOUT)
    # the last program to arrive builds the padded expert-bucket prefix sum
    done = tl.atomic_add(flag_ptr, 1, sem="acq_rel", scope="gpu")
    if done == NPROG - 1:
        c = tl.load(cnt_ptr + i, mask=i < EL, other=0)
        pad = ((c + BMC - 1) // BMC) * BMC
        cum = tl.cumsum(pad, axis=0) - pad
        tl.store(base_ptr + i, cum, mask=i < EL)
        tot = tl.sum(pad, axis=0)
        tl.store(base_ptr + EL, tot)
        tl.store(mt_ptr, tot // BMC)
        tl.store(mt_ptr + 1, tot)
        tl.store(cnt_ptr + i, tl.zeros([ELP], dtype=tl.int32), mask=i < EL)
        tl.store(flag_ptr, 0)


# --------------------------------------------------- bucket offsets (1 CTA) ---
@triton.jit
def k_offsets(cnt_ptr, base_ptr, mt_ptr, ELP: tl.constexpr, EL, BMC: tl.constexpr):
    i = tl.arange(0, ELP)
    c = tl.load(cnt_ptr + i, mask=i < EL, other=0)
    pad = ((c + BMC - 1) // BMC) * BMC
    cum = tl.cumsum(pad, axis=0) - pad
    tl.store(base_ptr + i, cum, mask=i < EL)
    tot = tl.sum(pad, axis=0)
    tl.store(base_ptr + EL, tot)
    tl.store(mt_ptr, tot // BMC)
    tl.store(mt_ptr + 1, tot)


# ------------------------------------------------------------- permutation ---
@triton.jit
def k_perm(tokw_ptr, tokr_ptr, base_ptr, ptok_ptr, pw_ptr, tsl_ptr, tcnt_ptr, T,
           ELP: tl.constexpr, EL, MAXL: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    offt = pid * BT + tl.arange(0, BT)
    mt = offt < T
    i = tl.arange(0, ELP)
    act = (i < EL)[None, :] & mt[:, None]
    w = tl.load(tokw_ptr + offt[:, None] * ELP + i[None, :], mask=act, other=0.0)
    r = tl.load(tokr_ptr + offt[:, None] * ELP + i[None, :], mask=act, other=0)
    sel = act & (w != 0)
    seli = sel.to(tl.int32)
    base = tl.load(base_ptr + i, mask=i < EL, other=0)
    slot = base[None, :] + r
    tl.store(ptok_ptr + slot, tl.broadcast_to(offt[:, None], [BT, ELP]), mask=sel)
    tl.store(pw_ptr + slot, w, mask=sel)
    jj = tl.cumsum(seli, axis=1) - seli            # rank within this token
    tl.store(tsl_ptr + offt[:, None] * MAXL + jj, slot, mask=sel)
    tl.store(tcnt_ptr + offt, tl.sum(seli, axis=1), mask=mt)


# -------------------------------------------------------- activation gather --
@triton.jit
def k_grows(pt_ptr, tsl_ptr, tcnt_ptr, a_ptr, sa_ptr, ctr_ptr, T, P,
            NKB: tl.constexpr, NKBP: tl.constexpr, MAXL: tl.constexpr, H, HP: tl.constexpr,
            EL: tl.constexpr, ELP: tl.constexpr, NSMC: tl.constexpr):
    # One program per token: read hidden_states[t] once, fan it out to that token's slots.
    hs_ptr = tl.multiple_of(tl.load(pt_ptr + 2), 16).to(tl.pointer_type(tl.int64))
    hss_ptr = tl.multiple_of(tl.load(pt_ptr + 3), 16).to(tl.pointer_type(tl.float32))
    if tl.program_id(0) == 0:      # reset the slot-reservation counters for the next call
        ci = tl.arange(0, ELP)
        tl.store(ctr_ptr + ci, tl.zeros([ELP], dtype=tl.int32), mask=ci < EL)
    # the rows move as 8-byte words: an indirect base pointer carries no alignment
    # information, so byte-wide accesses would cost 8x the instructions
    H8 = H // 8
    off = tl.arange(0, HP)
    m = off < H8
    kb = tl.arange(0, NKBP)
    mk = kb < NKB
    for t in tl.range(tl.program_id(0), T, NSMC, num_stages=2):
        n = tl.load(tcnt_ptr + t)
        if n > 0:
            v = tl.load(hs_ptr + t * H8 + off, mask=m, other=0)
            sv = tl.load(hss_ptr + kb * T + t, mask=mk, other=0.0)
            for j in tl.range(0, n):
                sl = tl.load(tsl_ptr + t * MAXL + j)
                tl.store(a_ptr + sl * H8 + off, v, mask=m)
                tl.store(sa_ptr + kb * P + sl, sv, mask=mk)


# ------------------------------------------- grouped GEMM1 + SwiGLU + requant --
@triton.jit
def k_gemm1(a_ptr, sa_ptr, pt_ptr, c_ptr, sc_ptr, base_ptr, mt_ptr, tok_ptr, out_ptr, NOUT, T,
            P, EL: tl.constexpr, ELP: tl.constexpr, H: tl.constexpr,
            I: tl.constexpr, BMC: tl.constexpr, NBK: tl.constexpr, NSTAGE: tl.constexpr,
            ORD: tl.constexpr, GM: tl.constexpr, ZOUT: tl.constexpr, OSTG: tl.constexpr,
            NSMC: tl.constexpr, EP: tl.constexpr, GATHER: tl.constexpr):
    w_ptr = tl.multiple_of(tl.load(pt_ptr + 4), 16).to(tl.pointer_type(tl.float8e4nv))
    ws_ptr = tl.multiple_of(tl.load(pt_ptr + 5), 16).to(tl.pointer_type(tl.float32))
    BC: tl.constexpr = 128 * NBK
    BN2: tl.constexpr = 256 * NBK
    NN: tl.constexpr = I // BC
    NKB: tl.constexpr = H // 128
    NIB: tl.constexpr = I // 128
    NWB: tl.constexpr = (2 * I) // 128
    if GATHER:
        a_desc = tl.make_tensor_descriptor(a_ptr, [T, H], [H, 1], [1, 128])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, [P, H], [H, 1], [BMC, 128])
    w_desc = tl.make_tensor_descriptor(w_ptr, [EL, I, 2, H], [2 * I * H, H, I * H, 1],
                                       [1, BC, 2, 128])
    c_desc = tl.make_tensor_descriptor(c_ptr, [P, I], [I, 1], [BMC, BC])
    nmt = tl.load(mt_ptr)
    i = tl.arange(0, ELP)
    tb = tl.load(base_ptr + i, mask=i <= EL, other=2147483647) // BMC
    nbv = tl.arange(0, NBK)
    zb = tl.zeros([2048], dtype=tl.bfloat16)
    zo = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    zleft = (NOUT + NSMC * 2048 - 1) // (NSMC * 2048)
    ntl = nmt * NN if ORD < 2 else ((nmt + GM - 1) // GM) * GM * NN
    for t in tl.range(tl.program_id(0), ntl, NSMC, num_stages=OSTG):
        if ORD == 2:               # global (GM m-tiles x all n-tiles) groups for L2 reuse
            gi = t % (GM * NN)
            tm = tl.minimum((t // (GM * NN)) * GM + gi % GM, nmt - 1)
            pn = gi // GM
            e = tl.sum((tb <= tm).to(tl.int32), axis=0) - 1
        else:
            e = tl.sum((tb * NN <= t).to(tl.int32), axis=0) - 1
            tbe = tl.sum(tl.where(i == e, tb, 0), axis=0)
            nte = tl.sum(tl.where(i == e + 1, tb, 0), axis=0) - tbe
            l = t - tbe * NN
            pn = l // nte if ORD == 0 else l % NN
            tm = tbe + (l % nte if ORD == 0 else l // NN)
        om = tm * BMC
        on = pn * BC
        wb = e * NWB * NKB + ((pn * NBK + nbv)[:, None]
                              + NIB * tl.arange(0, 2)[None, :]) * NKB
        acc = tl.zeros([BMC, BN2], dtype=tl.float32)
        offm = om + tl.arange(0, BMC)
        tks = tl.load(tok_ptr + offm) if GATHER else offm
        for k in tl.range(0, H, 128, num_stages=NSTAGE, warp_specialize=True):
            if GATHER:
                a = a_desc.gather(tks, k)
            else:
                a = a_desc.load([om, k])
            b = tl.reshape(w_desc.load([e, on, 0, k]), [BN2, 128])
            kb = k // 128
            if GATHER:
                sa = tl.load(sa_ptr + kb * T + tks)
            else:
                sa = tl.load(sa_ptr + kb * P + offm)
            sb2 = tl.load(ws_ptr + wb + kb)
            sv = sa[:, None, None] * sb2[None, :, :]
            sc2 = tl.reshape(tl.broadcast_to(sv[:, :, None, :], [BMC, NBK, 128, 2]),
                             [BMC, BN2])
            acc += tl.dot(a, b.T) * sc2
        gg, uu = tl.split(tl.reshape(acc, [BMC, BC, 2]))
        if EP:   # SwiGLU as x/2*(1+tanh(x/2)): one SFU op instead of exp + reciprocal
            hh = uu * 0.5
            th = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [hh],
                                           dtype=tl.float32, is_pure=True, pack=1)
            c = gg * (hh * (1.0 + th))
        else:
            c = gg + uu
        c3 = tl.reshape(c, [BMC, NBK, 128])
        am = tl.max(tl.abs(c3), axis=2)
        sc = tl.where(am > 0, am * (1.0 / 448.0), 1.0)
        inv = tl.where(am > 0, tl.fdiv(448.0, am, ieee_rounding=False), 1.0)
        cq = tl.reshape(c3 * inv[:, :, None], [BMC, BC]).to(tl.float8e4nv)
        c_desc.store([om, on], cq)
        tl.store(sc_ptr + (pn * NBK + nbv)[None, :] * P + offm[:, None], sc)
    if ZOUT:                       # whatever the tile loop did not get to
        while zleft > 0:
            tl.store(out_ptr + zo, zb, mask=zo < NOUT)
            zo += NSMC * 2048
            zleft -= 1


# -------------------------------------- grouped GEMM2 + weighted scatter-add --
@triton.jit
def k_gemm2(c_ptr, sc_ptr, pt_ptr, out_ptr, tok_ptr, wt_ptr, base_ptr, mt_ptr, P,
            EL: tl.constexpr, ELP: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
            BMC: tl.constexpr, BN: tl.constexpr, NSTAGE: tl.constexpr, ORD: tl.constexpr,
            GM: tl.constexpr, OSTG: tl.constexpr, NSMC: tl.constexpr,
            ATOMIC: tl.constexpr):
    w_ptr = tl.multiple_of(tl.load(pt_ptr + 6), 16).to(tl.pointer_type(tl.float8e4nv))
    ws_ptr = tl.multiple_of(tl.load(pt_ptr + 7), 16).to(tl.pointer_type(tl.float32))
    NN: tl.constexpr = H // BN
    NKB: tl.constexpr = I // 128
    NHB: tl.constexpr = H // 128
    NB: tl.constexpr = BN // 128
    c_desc = tl.make_tensor_descriptor(c_ptr, [P, I], [I, 1], [BMC, 128])
    w_desc = tl.make_tensor_descriptor(w_ptr, [EL, H, I], [H * I, I, 1], [1, BN, 128])
    nmt = tl.load(mt_ptr)
    i = tl.arange(0, ELP)
    tb = tl.load(base_ptr + i, mask=i <= EL, other=2147483647) // BMC
    ntl = nmt * NN if ORD < 2 else ((nmt + GM - 1) // GM) * GM * NN
    for t in tl.range(tl.program_id(0), ntl, NSMC, num_stages=OSTG):
        live = True
        if ORD == 2:
            gi = t % (GM * NN)
            tmr = (t // (GM * NN)) * GM + gi % GM
            live = tmr < nmt
            tm = tl.minimum(tmr, nmt - 1)
            pn = gi // GM
            e = tl.sum((tb <= tm).to(tl.int32), axis=0) - 1
        else:
            e = tl.sum((tb * NN <= t).to(tl.int32), axis=0) - 1
            tbe = tl.sum(tl.where(i == e, tb, 0), axis=0)
            nte = tl.sum(tl.where(i == e + 1, tb, 0), axis=0) - tbe
            l = t - tbe * NN
            pn = l // nte if ORD == 0 else l % NN
            tm = tbe + (l % nte if ORD == 0 else l // NN)
        om = tm * BMC
        on = pn * BN
        acc = tl.zeros([BMC, BN], dtype=tl.float32)
        offm = om + tl.arange(0, BMC)
        for k in tl.range(0, I, 128, num_stages=NSTAGE, warp_specialize=True):
            a = c_desc.load([om, k])
            b = tl.reshape(w_desc.load([e, on, k]), [BN, 128])
            kb = k // 128
            sa = tl.load(sc_ptr + kb * P + offm)
            sb = tl.load(ws_ptr + e * NHB * NKB + (pn * NB + tl.arange(0, NB)) * NKB + kb)
            sv = sa[:, None] * sb[None, :]                      # [BMC, NB]
            sc2 = tl.reshape(tl.broadcast_to(sv[:, :, None], [BMC, NB, 128]), [BMC, BN])
            acc += tl.dot(a, b.T) * sc2
        wt = tl.load(wt_ptr + offm)
        if ORD == 2:
            wt = tl.where(live, wt, 0.0)       # phantom tiles of the last group add nothing
        tok = tl.load(tok_ptr + offm)
        acc = acc * tl.abs(wt)[:, None]
        offn = on + tl.arange(0, BN)
        ob = acc.to(tl.bfloat16)
        oad = out_ptr + tok[:, None] * H + offn[None, :]
        if ATOMIC:
            tl.atomic_add(oad, ob, mask=(wt > 0)[:, None], sem="relaxed")
            tl.store(oad, ob, mask=(wt < 0)[:, None])
        else:
            tl.store(oad, ob, mask=(wt != 0)[:, None])


def _ws(T, H, I, E, EL, dev):
    key = (T, H, I, E, EL)
    w = _SCRATCH.get(key)
    if w is None:
        ELP = max(16, triton.next_power_of_2(EL + 1))
        Pmax = ((T * 8 + EL * BM + BM - 1) // BM) * BM
        w = dict(
            ELP=ELP, Pmax=Pmax,
            tokw=torch.empty((T, ELP), dtype=torch.float32, device=dev),
            tokr=torch.empty((T, ELP), dtype=torch.int32, device=dev),
            ctrs=torch.zeros(2 * ELP + 1, dtype=torch.int32, device=dev),
            tsl=torch.empty(T * 8, dtype=torch.int32, device=dev),
            tcnt=torch.empty(T, dtype=torch.int32, device=dev),
            base=torch.empty(ELP, dtype=torch.int32, device=dev),
            nmt=torch.empty(2, dtype=torch.int32, device=dev),
            ptok=torch.empty(Pmax, dtype=torch.int32, device=dev),
            pw=torch.empty(Pmax, dtype=torch.float32, device=dev),
            tok=torch.empty(Pmax, dtype=torch.int32, device=dev),
            wt=torch.empty(Pmax, dtype=torch.float32, device=dev),
            a=torch.empty((Pmax, H), dtype=torch.float8_e4m3fn, device=dev),
            sa=torch.empty((H // 128, Pmax), dtype=torch.float32, device=dev),
            c=torch.empty((Pmax, I), dtype=torch.float8_e4m3fn, device=dev),
            sc=torch.empty((I // 128, Pmax), dtype=torch.float32, device=dev),
            out=torch.empty((T, H), dtype=torch.bfloat16, device=dev),
            pdev=torch.zeros(8, dtype=torch.int64, device=dev),
            graphs={},
        )
        _SCRATCH[key] = w
    return w


@torch.no_grad()
def _route(w, T, H, E, EL, off, rsf, rl, rb, hs, hss, w1, w1s, w2, w2s):
    ELP = w["ELP"]
    cnt = w["ctrs"][:ELP]
    flag = w["ctrs"][2 * ELP:]
    # the hand-written kernel bakes in this definition's shape (256 experts in 8 groups,
    # 4 groups and 8 experts kept, a local range that is a whole multiple of 8 experts);
    # anything else falls through to the fully parameterised Triton kernel
    if (_CU is not None and CFG["rt_cu"] and E == 256 and EL <= 32 and off % 8 == 0
            and EL % 8 == 0 and H % 8 == 0):
        _CU.route(rl, rb, w["tokw"], cnt, w["ptok"], w["pw"], w["out"], w["tokr"], flag,
                  w["base"],
                  w["nmt"], w["pdev"], hs.data_ptr(), hss.data_ptr(), w1.data_ptr(),
                  w1s.data_ptr(), w2.data_ptr(), w2s.data_ptr(), EL, ELP, off, rsf,
                  w["Pmax"], CFG["zout"])
        return
    BT = CFG["rt_bt"]
    nprog = triton.cdiv(T, BT)
    k_route[(nprog,)](rl, rb, w["pdev"], hs.data_ptr(), hss.data_ptr(), w1.data_ptr(),
                      w1s.data_ptr(), w2.data_ptr(), w2s.data_ptr(), w["tokw"], w["tokr"], cnt,
                      w["ptok"], w["pw"], w["out"], flag, w["base"], w["nmt"],
                      T * H * CFG["zout"], T, E, EL, ELP, 8, 4, 8, off, rsf, w["Pmax"],
                      nprog, BT, E // 8, CFG["rt_zb"], BM,
                      1 if (BT == 1 and CFG["zout"]) else 0, CFG["rt_zob"], H,
                      num_warps=CFG["rt_w"])


def _launch(w, T, H, I, E, EL, off, rsf):
    ELP = w["ELP"]
    P = w["Pmax"]
    pd = w["pdev"]
    ctr = w["ctrs"][ELP:2 * ELP]
    bp = CFG["gs_bp"]
    k_perm[(triton.cdiv(T, bp),)](w["tokw"], w["tokr"], w["base"], w["ptok"], w["pw"],
                                  w["tsl"], w["tcnt"], T, ELP, EL, 8, bp,
                                  num_warps=CFG["gs_w"])
    ng = NSM * CFG["gr_prog"]
    k_grows[(ng,)](pd, w["tsl"], w["tcnt"], w["a"].view(torch.int64), w["sa"], ctr, T, P,
                   H // 128, triton.next_power_of_2(H // 128), 8, H,
                   triton.next_power_of_2(H // 8), EL, ELP, ng, num_warps=CFG["gr_w"])
    n1 = NSM * CFG["g1_prog"]
    k_gemm1[(n1,)](w["a"], w["sa"], pd, w["c"], w["sc"], w["base"], w["nmt"], w["ptok"],
                   w["out"], T * H, T, P, EL, ELP, H, I, BM, CFG["g1_nbk"], CFG["g1_st"],
                   CFG["g1_ord"], CFG["g1_gm"], 1 - CFG["zout"], CFG["g1_os"], n1,
                   CFG["g1_ep"], 0, num_warps=CFG["g1_w"])
    n2 = NSM * CFG["g2_prog"]
    k_gemm2[(n2,)](w["c"], w["sc"], pd, w["out"], w["ptok"], w["pw"], w["base"], w["nmt"],
                   P, EL, ELP, H, I, BM, CFG["g2_bn"], CFG["g2_st"], CFG["g2_ord"],
                   CFG["g2_gm"], CFG["g2_os"], n2, CFG["atomic"], num_warps=CFG["g2_w"])


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset,
        routed_scaling_factor):
    T, E = routing_logits.shape
    EL, I2, H = gemm1_weights.shape
    I = I2 // 2
    w = _ws(T, H, I, E, EL, hidden_states.device)
    off = int(local_expert_offset)
    rsf = float(routed_scaling_factor)
    # Routing runs eagerly: its launch arguments carry this call's input addresses, which it
    # publishes into the device pointer table that the replayed kernels read.  The launch
    # sequence after it depends only on shapes and the two scalars, so it is replayed from a
    # CUDA graph and nothing input-derived survives across calls.
    _route(w, T, H, E, EL, off, rsf, routing_logits, routing_bias, hidden_states,
           hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights,
           gemm2_weights_scale)
    key = (off, rsf)
    g = w["graphs"].get(key)
    if g is not None:
        g.replay()
        return w["out"]
    _launch(w, T, H, I, E, EL, off, rsf)
    if CFG["graph"]:
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                # the warm-up must redo the routing too: gemm2 accumulates into the output,
                # so replaying only the tail would double every multi-expert row
                _route(w, T, H, E, EL, off, rsf, routing_logits, routing_bias, hidden_states,
                       hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights,
                       gemm2_weights_scale)
                _launch(w, T, H, I, E, EL, off, rsf)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                _launch(w, T, H, I, E, EL, off, rsf)
            w["graphs"][key] = gr
        except Exception:
            CFG["graph"] = 0
    return w["out"]
