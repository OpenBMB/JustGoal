import torch
import triton
import triton.language as tl

H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = E_GLOBAL // N_GROUP   # 32
NHB = H // BLOCK                   # 56
NIB = I // BLOCK                   # 16
NG1 = (2 * I) // BLOCK             # 32
F8 = torch.float8_e4m3fn
F8_MAX = 448.0

_EL = tl.constexpr(32)


# ----------------------- routing (fused) -----------------------
@triton.jit
def route_kernel(logits, bias, le_tok, w_tok, counts, T, off, rsf, s_lt,
                 slk_t, slk_k, swk_t, swk_k):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, 256)
    lg = tl.load(logits + t * s_lt + offs).to(tl.float32)
    b = tl.load(bias + offs).to(tl.float32)
    s = tl.sigmoid(lg)
    swb = s + b
    g2 = tl.reshape(swb, (8, 32))
    m1 = tl.max(g2, axis=1)
    am1 = tl.argmax(g2, axis=1)
    a32 = tl.arange(0, 32)
    g2m = tl.where(a32[None, :] == am1[:, None], -float('inf'), g2)
    m2 = tl.max(g2m, axis=1)
    gscore = m1 + m2
    gtmp = gscore
    thr = -float('inf')
    for _ in range(4):
        thr = tl.max(gtmp)
        gtmp = tl.where(gtmp == thr, -float('inf'), gtmp)
    gmask = gscore >= thr
    egrp = offs // 32
    allowed = tl.sum(tl.where((tl.arange(0, 8)[:, None] == egrp[None, :]) & gmask[:, None], 1, 0), axis=0) > 0
    sp = tl.where(allowed, swb, -float('inf'))
    karange = tl.arange(0, 8)
    sel_s = tl.zeros((8,), dtype=tl.float32)
    sel_le = tl.zeros((8,), dtype=tl.int32)
    sel_in = tl.zeros((8,), dtype=tl.int32)
    for k in range(8):
        idx = tl.argmax(sp, axis=0).to(tl.int32)
        sval = tl.sum(tl.where(offs == idx, s, 0.))
        sel_s = tl.where(karange == k, sval, sel_s)
        le = idx - off
        inr = (le >= 0) & (le < 32)
        sel_le = tl.where(karange == k, tl.where(inr, le, 32), sel_le)
        sel_in = tl.where(karange == k, tl.where(inr, 1, 0), sel_in)
        sp = tl.where(offs == idx, -float('inf'), sp)
    wsum = tl.sum(sel_s) + 1e-20
    w = sel_s / wsum * rsf
    inr = sel_in > 0
    w_s = tl.where(inr, w, 0.)
    tl.store(le_tok + t * slk_t + karange * slk_k, sel_le)
    tl.store(w_tok + t * swk_t + karange * swk_k, w_s)
    tl.atomic_add(counts + sel_le, tl.where(inr, 1, 0), mask=inr)


@triton.jit
def offsets_kernel(counts, pad_off, block_expert, fill, MAXB, BM: tl.constexpr):
    b = tl.program_id(0)
    e32 = tl.arange(0, 32)
    c = tl.load(counts + e32).to(tl.int32)
    padded = ((c + BM - 1) // BM) * BM
    padcum = tl.cumsum(padded, axis=0)          # inclusive prefix sum
    if b < MAXB:
        be = tl.sum((padcum <= b * BM).to(tl.int32))
        tl.store(block_expert + b, be)
    if b < 32:
        po = padcum - padded                    # exclusive prefix (pad_off)
        tl.store(pad_off + b, tl.sum(tl.where(e32 == b, po, 0)))
        tl.store(fill + b, 0)


@triton.jit
def place_kernel(le_tok, w_tok, pad_off, fill, sorted_tok, sorted_w, T,
                 slk_t, slk_k, swk_t, swk_k):
    t = tl.program_id(0)
    if t >= T:
        return
    for k in range(8):
        le = tl.load(le_tok + t * slk_t + k * slk_k)
        if le < 32:
            pos = tl.atomic_add(fill + le, 1)
            dest = tl.load(pad_off + le) + pos
            tl.store(sorted_tok + dest, t)
            tl.store(sorted_w + dest, tl.load(w_tok + t * swk_t + k * swk_k))


# ----------------------- GEMM1 (fp8, single accumulator) -> G[Np,2I] bf16 -----------------------
@triton.jit
def gemm1_kernel(A, Ascale, W, Wscale, sorted_tok, G, T,
                 sa_t, sa_h, sas_hb, sas_t,
                 sw_e, sw_n, sw_h, sws_e, sws_ob, sws_hb,
                 sg_p, sg_n, block_expert,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NHB: tl.constexpr, RATIO: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert + pid_m // RATIO)
    if e >= _EL:
        return
    offs_m = tl.arange(0, BM)
    tok = tl.load(sorted_tok + pid_m * BM + offs_m)
    valid = tok < T
    toks = tl.where(valid, tok, 0)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    n0 = pid_n * BN
    ob = n0 // 128
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(NHB):
        k0 = kb * BK
        a = tl.load(A + toks[:, None] * sa_t + (k0 + offs_k)[None, :] * sa_h,
                    mask=valid[:, None], other=0.).to(tl.float8e4nv)
        w = tl.load(W + e * sw_e + (n0 + offs_n)[:, None] * sw_n + (k0 + offs_k)[None, :] * sw_h).to(tl.float8e4nv)
        a_s = tl.load(Ascale + kb * sas_hb + toks * sas_t, mask=valid, other=0.)
        w_s = tl.load(Wscale + e * sws_e + ob * sws_ob + kb * sws_hb)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32) * (a_s[:, None] * w_s)
    p = pid_m * BM + offs_m
    tl.store(G + p[:, None] * sg_p + (n0 + offs_n)[None, :] * sg_n, acc.to(tl.bfloat16))


# ----------------------- SwiGLU + quant: G[Np,2I] -> Cfp8[Np,I], Cscale[Np,NIB] -----------------------
@triton.jit
def swiglu_kernel(G, Cout, Cscale, sg_p, sg_n, sc_p, sc_i, scs_p, scs_ib, Np,
                  BMs: tl.constexpr, I: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.arange(0, BMs)
    p = pid * BMs + offs_m
    m = p < Np
    offs = tl.arange(0, 128)
    n0 = pid_n * 128
    gate = tl.load(G + p[:, None] * sg_p + (n0 + offs)[None, :] * sg_n, mask=m[:, None], other=0.).to(tl.float32)
    up = tl.load(G + p[:, None] * sg_p + (I + n0 + offs)[None, :] * sg_n, mask=m[:, None], other=0.).to(tl.float32)
    cc = (up * tl.sigmoid(up)) * gate
    amax = tl.max(tl.abs(cc), axis=1)
    cs = amax / 448.0
    css = tl.where(cs > 0, cs, 1.0)
    cq = tl.clamp(cc / css[:, None], -448., 448.).to(tl.float8e4nv)
    tl.store(Cout + p[:, None] * sc_p + (n0 + offs)[None, :] * sc_i, cq, mask=m[:, None])
    tl.store(Cscale + p * scs_p + pid_n * scs_ib, cs, mask=m)


# ----------------------- GEMM2 (fp8) + weighted scatter-add -----------------------
@triton.jit
def gemm2_kernel(C, Cscale, W, Wscale, sorted_tok, sorted_w, Out, T,
                 sc_p, sc_i, scs_p, scs_ib,
                 sw_e, sw_h, sw_i, sws_e, sws_hb, sws_ib, so_t, so_h, block_expert,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, I: tl.constexpr, OUTDT: tl.constexpr, RATIO: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert + pid_m // RATIO)
    if e >= _EL:
        return
    offs_m = tl.arange(0, BM)
    p = pid_m * BM + offs_m
    tok = tl.load(sorted_tok + pid_m * BM + offs_m)
    valid = tok < T
    w_tok = tl.load(sorted_w + pid_m * BM + offs_m, mask=valid, other=0.)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    n0 = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    hb0 = n0 // 128
    nkb = I // BK
    for kb in range(nkb):
        k0 = kb * BK
        c = tl.load(C + p[:, None] * sc_p + (k0 + offs_k)[None, :] * sc_i).to(tl.float8e4nv)
        cs = tl.load(Cscale + p * scs_p + kb * scs_ib)
        w = tl.load(W + e * sw_e + (n0 + offs_n)[:, None] * sw_h + (k0 + offs_k)[None, :] * sw_i).to(tl.float8e4nv)
        wsv = tl.load(Wscale + e * sws_e + hb0 * sws_hb + kb * sws_ib)
        acc += tl.dot(c, tl.trans(w), out_dtype=tl.float32) * (cs[:, None] * wsv)
    acc = acc * w_tok[:, None]
    tl.atomic_add(Out + tok[:, None] * so_t + (n0 + offs_n)[None, :] * so_h, acc.to(OUTDT), mask=valid[:, None])


# ----------------------- orchestration -----------------------
import os as _os


def _cfg(T):
    # returns (BM, BN1, NW1, NS1, BN2, NW2, NS2, BMS)
    ov = _os.environ.get("FIBCFG")
    if ov:
        # format: "T_thresh:cfgA|cfgB"  where cfg = BM,BN1,NW1,NS1,BN2,NW2,NS2,BMS
        # small T (< thresh) uses cfgA, else cfgB
        thresh, rest = ov.split(":")
        a, b = rest.split("|")
        pa = tuple(int(x) for x in a.split(","))
        pb = tuple(int(x) for x in b.split(","))
        return pa if T < int(thresh) else pb
    # per-shape configs from cold-L2 autotune
    # (BM1,BN1,NW1,NS1, BM2,BN2,NW2,NS2, BMS)
    if T <= 1:
        return (64, 64, 4, 4, 64, 128, 4, 4, 64)
    if T <= 128:
        return (64, 128, 4, 3, 64, 128, 4, 4, 64)   # gemm2 NW=4 (autotune: best for T<=80)
    if T < 2048:
        return (64, 128, 4, 3, 64, 128, 8, 4, 64)   # gemm2 NW=8 (T=901)
    # large T: gemm1 BM=64, gemm2 BM=128 BN=64 (shared 128-padded layout)
    return (64, 128, 4, 3, 128, 64, 4, 3, 64)


def _forward(inp, MAXB, cfg):
    BM1, BN1, NW1, NS1, BM2, BN2, NW2, NS2, BMS = cfg
    BM_PAD = BM1 if BM1 >= BM2 else BM2
    R1 = BM_PAD // BM1
    R2 = BM_PAD // BM2
    dev = inp['hidden_states'].device
    T = inp['routing_logits'].shape[0]
    off = int(inp['local_expert_offset'])
    rsf = float(inp['routed_scaling_factor'])
    hs = inp['hidden_states']
    hss = inp['hidden_states_scale'].to(torch.float32)
    g1 = inp['gemm1_weights']; g1s = inp['gemm1_weights_scale'].to(torch.float32)
    g2 = inp['gemm2_weights']; g2s = inp['gemm2_weights_scale'].to(torch.float32)
    logits = inp['routing_logits']; bias = inp['routing_bias']

    Np = MAXB * BM_PAD
    le_tok = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    w_tok = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    counts = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
    fill = torch.empty(E_LOCAL, dtype=torch.int32, device=dev)
    pad_off = torch.empty(E_LOCAL, dtype=torch.int32, device=dev)
    block_expert = torch.empty(MAXB, dtype=torch.int32, device=dev)
    sorted_tok = torch.full((Np,), T, dtype=torch.int32, device=dev)
    sorted_w = torch.zeros((Np,), dtype=torch.float32, device=dev)

    route_kernel[(T,)](logits, bias, le_tok, w_tok, counts, T, off, rsf,
                       logits.stride(0), le_tok.stride(0), le_tok.stride(1),
                       w_tok.stride(0), w_tok.stride(1))
    offsets_kernel[(max(MAXB, E_LOCAL),)](counts, pad_off, block_expert, fill, MAXB, BM=BM_PAD)
    place_kernel[(T,)](le_tok, w_tok, pad_off, fill, sorted_tok, sorted_w, T,
                       le_tok.stride(0), le_tok.stride(1), w_tok.stride(0), w_tok.stride(1))

    G = torch.empty((Np, 2 * I), dtype=torch.bfloat16, device=dev)
    Cf = torch.empty((Np, I), dtype=F8, device=dev)
    Csc = torch.empty((Np, NIB), dtype=torch.float32, device=dev)
    out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)

    gemm1_kernel[(MAXB * R1, (2 * I) // BN1)](
        hs, hss, g1, g1s, sorted_tok, G, T,
        hs.stride(0), hs.stride(1), hss.stride(0), hss.stride(1),
        g1.stride(0), g1.stride(1), g1.stride(2), g1s.stride(0), g1s.stride(1), g1s.stride(2),
        G.stride(0), G.stride(1), block_expert,
        BM=BM1, BN=BN1, BK=BLOCK, NHB=NHB, RATIO=R1, num_warps=NW1, num_stages=NS1)
    swiglu_kernel[((Np + BMS - 1) // BMS, NIB)](
        G, Cf, Csc, G.stride(0), G.stride(1), Cf.stride(0), Cf.stride(1),
        Csc.stride(0), Csc.stride(1), Np, BMs=BMS, I=I, num_warps=8, num_stages=2)
    gemm2_kernel[(MAXB * R2, H // BN2)](
        Cf, Csc, g2, g2s, sorted_tok, sorted_w, out, T,
        Cf.stride(0), Cf.stride(1), Csc.stride(0), Csc.stride(1),
        g2.stride(0), g2.stride(1), g2.stride(2), g2s.stride(0), g2s.stride(1), g2s.stride(2),
        out.stride(0), out.stride(1), block_expert,
        BM=BM2, BN=BN2, BK=BLOCK, I=I, OUTDT=tl.bfloat16, RATIO=R2, num_warps=NW2, num_stages=NS2)
    return out


_PLANS = {}
_CAPTURES = {}
_MAX_CAPTURES = 2


def _ptrkey(inp):
    ks = ('routing_logits', 'routing_bias', 'hidden_states', 'hidden_states_scale',
          'gemm1_weights', 'gemm1_weights_scale', 'gemm2_weights', 'gemm2_weights_scale')
    return tuple(inp[k].data_ptr() for k in ks)


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    inp = dict(routing_logits=routing_logits, routing_bias=routing_bias,
               hidden_states=hidden_states, hidden_states_scale=hidden_states_scale,
               gemm1_weights=gemm1_weights, gemm1_weights_scale=gemm1_weights_scale,
               gemm2_weights=gemm2_weights, gemm2_weights_scale=gemm2_weights_scale,
               local_expert_offset=local_expert_offset, routed_scaling_factor=routed_scaling_factor)
    T = routing_logits.shape[0]
    off = int(local_expert_offset)
    rsf = float(routed_scaling_factor)
    cfg = _cfg(T)
    BM_PAD = max(cfg[0], cfg[4])
    P_max = T * TOP_K
    MAXB = min(P_max, E_LOCAL) + P_max // BM_PAD + 1
    # If the caller is already capturing a CUDA graph (e.g. the benchmark harness),
    # run eagerly to avoid a nested/illegal capture.
    try:
        if torch.cuda.is_current_stream_capturing():
            return _forward(inp, MAXB, cfg)
    except Exception:
        pass
    key = (T, off, rsf, _ptrkey(inp))
    plan = _PLANS.get(key)
    if plan is not None:
        g, out = plan
        g.replay()
        return out
    # Decide whether to capture a CUDA graph. Capturing is worth it only when the
    # same input tensors are reused across calls (the timed case). If we have already
    # captured several distinct graphs for this (T, off, rsf) — i.e. the caller keeps
    # passing fresh tensors — stop capturing and run eagerly (still fully recomputes
    # from the inputs) to avoid unbounded graph allocation.
    sig = (T, off, rsf)
    n_cap = _CAPTURES.get(sig, 0)
    if n_cap >= _MAX_CAPTURES:
        return _forward(inp, MAXB, cfg)
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _forward(inp, MAXB, cfg)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = _forward(inp, MAXB, cfg)
    except Exception:
        return _forward(inp, MAXB, cfg)
    _CAPTURES[sig] = n_cap + 1
    _PLANS[key] = (g, out)
    g.replay()
    return out
