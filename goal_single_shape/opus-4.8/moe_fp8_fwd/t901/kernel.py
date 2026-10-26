"""From-scratch MoE FP8 block-scale kernel (DeepSeek-V3 routing) for B200.

run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor) -> output bf16 [T, 7168]

No external GEMM/MoE libraries; the two grouped FP8 block-scale GEMMs, SwiGLU,
activation requantization and weighted scatter are all custom Triton.
"""
import torch
import triton
import triton.language as tl

H = 7168
I = 2048
N2I = 2 * I          # 4096
BLOCK = 128
NKH = H // BLOCK     # 56
NII = I // BLOCK     # 16
NG1 = N2I // BLOCK   # 32
E_LOCAL = 32
E_GLOBAL = 256
N_GROUP = 8
GRP = E_GLOBAL // N_GROUP
TOPK_GROUP = 4
TOP_K = 8

BM = 64
BN1 = 256
BN2 = 128
MAXB = 160
NCTA = 296
SENT = 1 << 30
FP8_MAX = 448.0

_BLK = tl.constexpr(BLOCK)
_H = tl.constexpr(H)
_I = tl.constexpr(I)
_N2I = tl.constexpr(N2I)
_NKH = tl.constexpr(NKH)
_NII = tl.constexpr(NII)
_FP8MAX = tl.constexpr(FP8_MAX)

_ws = {}


def _workspace(T, device):
    key = (T, device)
    w = _ws.get(key)
    if w is None:
        slots = MAXB * BM
        w = dict(
            sorted_tok=torch.empty(slots + 1, dtype=torch.int32, device=device),
            sorted_w=torch.empty(slots + 1, dtype=torch.float32, device=device),
            expert_ids=torch.empty(MAXB, dtype=torch.int32, device=device),
            gate=torch.empty((T, E_LOCAL), dtype=torch.float32, device=device),
            nblocks=torch.empty(1, dtype=torch.int32, device=device),
            cnt=torch.empty(E_LOCAL, dtype=torch.int32, device=device),
            cursor=torch.empty(E_LOCAL, dtype=torch.int32, device=device),
            A_g=torch.empty((slots, H), dtype=torch.float8_e4m3fn, device=device),
            asc_g=torch.empty((slots, NKH), dtype=torch.float32, device=device),
            G1=torch.empty((slots, N2I), dtype=torch.bfloat16, device=device),
            Cq=torch.empty((slots, I), dtype=torch.float8_e4m3fn, device=device),
            cscale=torch.empty((NII, slots), dtype=torch.float32, device=device),
            out_fp32=torch.empty((T, H), dtype=torch.float32, device=device),
            out_bf16=torch.empty((T, H), dtype=torch.bfloat16, device=device),
        )
        _ws[key] = w
    return w


@triton.jit
def _gemm1_kernel(
    A_g, asc_g, W13, s13, G1,
    SLOTS, stride_s13e,
    BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr, expert_ids,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(expert_ids + pid_m)
    if e < 0:
        return
    NSUB: tl.constexpr = BN // _BLK
    colsub = tl.arange(0, BN) // _BLK
    offs_m = pid_m * BM + tl.arange(0, BM)
    we = e.to(tl.int64) * (_N2I * _H)
    bw = tl.make_block_ptr(W13 + we, shape=(_H, _N2I), strides=(1, _H),
                           offsets=(0, pid_n * BN), block_shape=(_BLK, BN), order=(0, 1))
    ap = tl.make_block_ptr(A_g, shape=(SLOTS, _H), strides=(_H, 1),
                           offsets=(pid_m * BM, 0), block_shape=(BM, _BLK), order=(1, 0))
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    s13e = s13 + e * stride_s13e + pid_n * NSUB * _NKH
    for kb in range(0, NK):
        a = tl.load(ap)
        a_s = tl.load(asc_g + offs_m * _NKH + kb)
        w = tl.load(bw)
        svv = tl.load(s13e + colsub * _NKH + kb)
        acc += tl.dot(a, w) * (a_s[:, None] * svv[None, :])
        bw = tl.advance(bw, (_BLK, 0))
        ap = tl.advance(ap, (0, _BLK))
    gp = tl.make_block_ptr(G1, shape=(SLOTS, _N2I), strides=(_N2I, 1),
                           offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(gp, acc.to(tl.bfloat16))


@triton.jit
def _swiglu_kernel(
    G1, Cq, cscale, SLOTS,
    BM: tl.constexpr, expert_ids,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)        # intermediate block 0..15
    e = tl.load(expert_ids + pid_m)
    if e < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, _BLK)
    g_ptr = G1 + offs_m[:, None] * _N2I + (pid_i * _BLK + cols)[None, :]
    u_ptr = G1 + offs_m[:, None] * _N2I + (_I + pid_i * _BLK + cols)[None, :]
    x1 = tl.load(g_ptr).to(tl.float32)
    x2 = tl.load(u_ptr).to(tl.float32)
    c = (x2 * tl.sigmoid(x2)) * x1
    amax = tl.max(tl.abs(c), axis=1)
    cs = tl.maximum(amax * (1.0 / _FP8MAX), 1e-12)
    cq = (c / cs[:, None]).to(tl.float8e4nv)
    tl.store(Cq + offs_m[:, None] * _I + (pid_i * _BLK + cols)[None, :], cq)
    tl.store(cscale + pid_i * SLOTS + offs_m, cs)


@triton.jit
def _gemm2_kernel(
    Cq, cscale, W2, s2, sorted_tok, sorted_w, out,
    T, SLOTS, stride_s2e,
    BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr, expert_ids,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    e = tl.load(expert_ids + pid_m)
    if e < 0:
        return
    NSUB: tl.constexpr = BN // _BLK
    hsub = tl.arange(0, BN) // _BLK
    offs_m = pid_m * BM + tl.arange(0, BM)
    tok = tl.load(sorted_tok + offs_m)
    mask_m = tok < T
    tok_safe = tl.where(mask_m, tok, 0)
    wcomb = tl.load(sorted_w + offs_m, mask=mask_m, other=0.0)
    offs_h = pid_h * BN + tl.arange(0, BN)
    we = e.to(tl.int64) * (_H * _I)
    bw = tl.make_block_ptr(W2 + we, shape=(_I, _H), strides=(1, _I),
                           offsets=(0, pid_h * BN), block_shape=(_BLK, BN), order=(0, 1))
    cqp = tl.make_block_ptr(Cq, shape=(SLOTS, _I), strides=(_I, 1),
                            offsets=(pid_m * BM, 0), block_shape=(BM, _BLK), order=(1, 0))
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    s2e = s2 + e * stride_s2e + pid_h * NSUB * _NII
    for kb in range(0, NK):
        cq = tl.load(cqp)
        cs = tl.load(cscale + kb * SLOTS + offs_m)
        wb = tl.load(bw)
        svv = tl.load(s2e + hsub * _NII + kb)
        acc += tl.dot(cq, wb) * (cs[:, None] * svv[None, :])
        bw = tl.advance(bw, (_BLK, 0))
        cqp = tl.advance(cqp, (0, _BLK))
    o = acc * wcomb[:, None]
    tl.atomic_add(out + tok_safe[:, None] * _H + offs_h[None, :], o, mask=mask_m[:, None])


@triton.jit
def _route_kernel(s_in, rb, gate_out, cnt, T, loff, rsf,
                  NG: tl.constexpr, GS: tl.constexpr, EG: tl.constexpr,
                  TKG: tl.constexpr, TK: tl.constexpr, EL: tl.constexpr):
    t = tl.program_id(0)
    if t >= T:
        return
    g = tl.arange(0, NG)
    c = tl.arange(0, GS)
    idx = g[:, None] * GS + c[None, :]            # [NG,GS]
    en = tl.load(s_in + t * EG + idx).to(tl.float32)   # exp(-logits)
    s = 1.0 / (1.0 + en)                               # exact f32 sigmoid (matches reference)
    bias = tl.load(rb + idx).to(tl.float32)
    sb = s + bias                                 # [NG,GS]
    NEG = -1e30
    m1 = tl.max(sb, axis=1)                        # [NG]
    am = tl.argmax(sb, axis=1)                     # [NG] remove exactly one max
    sb_no1 = tl.where(tl.arange(0, GS)[None, :] == am[:, None], NEG, sb)
    m2 = tl.max(sb_no1, axis=1)
    gscore = m1 + m2                              # [NG]
    keep = tl.zeros([NG], dtype=tl.int1)
    gw = gscore
    for _i in range(TKG):
        j = tl.argmax(gw, 0)
        keep = keep | (tl.arange(0, NG) == j)
        gw = tl.where(tl.arange(0, NG) == j, NEG, gw)
    sb_pruned = tl.where(keep[:, None], sb, NEG).reshape(EG)
    s_flat = s.reshape(EG)
    sw = sb_pruned
    wsum = 0.0
    sel = tl.zeros([EG], dtype=tl.float32)
    rng = tl.arange(0, EG)
    for _i in range(TK):
        j = tl.argmax(sw, 0)
        sj = tl.sum(tl.where(rng == j, s_flat, 0.0))
        wsum += sj
        sel = tl.where(rng == j, s_flat, sel)
        sw = tl.where(rng == j, NEG, sw)
    w = sel / (wsum + 1e-20) * rsf                # [EG]
    le = tl.arange(0, EL)
    glocal = tl.sum(tl.where(rng[None, :] == (loff + le[:, None]), w[None, :], 0.0), axis=1)  # [EL]
    tl.store(gate_out + t * EL + le, glocal)
    tl.atomic_add(cnt + le, (glocal > 0).to(tl.int32))


@triton.jit
def _offsets_kernel(cnt, cursor, expert_ids, nblocks,
                    BM: tl.constexpr, EL: tl.constexpr, MAXB: tl.constexpr, MP2: tl.constexpr):
    e = tl.arange(0, EL)
    c = tl.load(cnt + e).to(tl.int32)
    bpe = (c + BM - 1) // BM
    cumb = tl.cumsum(bpe, 0)                 # [EL] cumulative blocks
    nb = tl.sum(bpe)
    po = (cumb - bpe) * BM                    # first slot per expert
    tl.store(cursor + e, po)
    tl.store(nblocks + tl.zeros([1], tl.int32), nb)
    b = tl.arange(0, MP2)
    eid = tl.sum((cumb[None, :] <= b[:, None]).to(tl.int32), axis=1)   # expert for block b
    eid = tl.where(b < nb, eid, -1)
    tl.store(expert_ids + b, eid, mask=b < MAXB)


@triton.jit
def _scatter_kernel(gate, cursor, sorted_tok, sorted_w, T, EL: tl.constexpr):
    t = tl.program_id(0)
    if t >= T:
        return
    for le in range(EL):
        w = tl.load(gate + t * EL + le)
        m = w > 0.0
        slot = tl.atomic_add(cursor + le, 1, mask=m)
        tl.store(sorted_tok + slot, t, mask=m)
        tl.store(sorted_w + slot, w, mask=m)


@triton.jit
def _gather_kernel(A, asc, sorted_tok, A_g, asc_g, T, SLOTS,
                   BM: tl.constexpr, BLK: tl.constexpr, expert_ids):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    e = tl.load(expert_ids + pid_m)
    if e < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    tok = tl.load(sorted_tok + offs_m)
    tok_safe = tl.where(tok < T, tok, 0)
    cols = pid_h * BLK + tl.arange(0, BLK)
    a = tl.load(A + tok_safe[:, None] * _H + cols[None, :])
    tl.store(A_g + offs_m[:, None] * _H + cols[None, :], a)
    if pid_h == 0:
        kb = tl.arange(0, 64)
        km = kb < _NKH
        sc = tl.load(asc + kb[None, :] * T + tok_safe[:, None], mask=km[None, :], other=0.0)
        tl.store(asc_g + offs_m[:, None] * _NKH + kb[None, :], sc, mask=km[None, :])


@triton.jit
def _cast_kernel(src, dst, n, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    x = tl.load(src + offs, mask=m)
    tl.store(dst + offs, x.to(tl.bfloat16), mask=m)


@torch.no_grad()
def _routing(routing_logits, routing_bias, local_expert_offset, routed_scaling_factor, ws):
    T = routing_logits.shape[0]
    gate = ws["gate"]
    cnt = ws["cnt"]
    cnt.zero_()
    en = torch.exp(-routing_logits.float())                # exp(-logits); sigmoid done in-kernel
    _route_kernel[(T,)](
        en, routing_bias, gate, cnt, T,
        int(local_expert_offset), float(routed_scaling_factor),
        N_GROUP, GRP, E_GLOBAL, TOPK_GROUP, TOP_K, E_LOCAL,
        num_warps=1,
    )
    return gate, cnt


@torch.no_grad()
def _build_layout(gate, cnt_i, hs, hss, ws, T):
    """Grouped-by-expert layout (fused kernels) + A gather. Minimal torch on 32-elem arrays."""
    SLOTS = MAXB * BM
    _offsets_kernel[(1,)](cnt_i, ws["cursor"], ws["expert_ids"], ws["nblocks"],
                          BM, E_LOCAL, MAXB, 256, num_warps=1)

    sorted_tok = ws["sorted_tok"]
    sorted_tok.fill_(SENT)
    _scatter_kernel[(T,)](gate, ws["cursor"], sorted_tok, ws["sorted_w"], T, E_LOCAL, num_warps=1)
    _gather_kernel[(MAXB, H // 512)](
        hs, hss, sorted_tok, ws["A_g"], ws["asc_g"], T, SLOTS, BM, 512, ws["expert_ids"],
        num_warps=4, num_stages=2,
    )


@torch.no_grad()
def _pre_impl(rl, rb, hs, hss, loff, rsf, ws, T):
    gate, cnt = _routing(rl, rb, loff, rsf, ws)
    _build_layout(gate, cnt, hs, hss, ws, T)


@torch.no_grad()
def _gemm_impl(g1w, g1s, g2w, g2s, ws, T, GM):
    SLOTS = MAXB * BM
    A_g = ws["A_g"]; asc_g = ws["asc_g"]; G1 = ws["G1"]; Cq = ws["Cq"]
    cscale = ws["cscale"]; out_fp32 = ws["out_fp32"]
    sorted_tok = ws["sorted_tok"]; sorted_w = ws["sorted_w"]; expert_ids = ws["expert_ids"]
    out_fp32.zero_()
    _gemm1_kernel[(GM, N2I // BN1)](
        A_g, asc_g, g1w, g1s, G1, SLOTS, g1s.stride(0),
        BM, BN1, NKH, expert_ids, num_warps=8, num_stages=4,
    )
    _swiglu_kernel[(GM, NII)](
        G1, Cq, cscale, SLOTS, BM, expert_ids, num_warps=4, num_stages=2,
    )
    _gemm2_kernel[(GM, H // BN2)](
        Cq, cscale, g2w, g2s, sorted_tok, sorted_w, out_fp32, T, SLOTS, g2s.stride(0),
        BM, BN2, NII, expert_ids, num_warps=8, num_stages=4,
    )
    out = ws["out_bf16"]
    n = T * H
    _cast_kernel[(triton.cdiv(n, 1024),)](out_fp32.view(-1), out.view(-1), n, 1024)
    return out


@torch.no_grad()
def _impl(rl, rb, hs, hss, g1w, g1s, g2w, g2s, loff, rsf, ws, T):
    _pre_impl(rl, rb, hs, hss, loff, rsf, ws, T)
    return _gemm_impl(g1w, g1s, g2w, g2s, ws, T, MAXB)


_BUCKETS = (64, 128, MAXB)
_gstate = {}
USE_GRAPH = True


def _bucket(nb):
    for b in _BUCKETS:
        if nb <= b:
            return b
    return MAXB


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    dev = hidden_states.device
    T = routing_logits.shape[0]
    ws = _workspace(T, dev)
    loff = int(local_expert_offset); rsf = float(routed_scaling_factor)

    if not USE_GRAPH:
        return _impl(routing_logits, routing_bias, hidden_states, hidden_states_scale,
                     gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                     loff, rsf, ws, T)

    key = (T, dev)
    sig = (gemm1_weights.data_ptr(), gemm1_weights_scale.data_ptr(),
           gemm2_weights.data_ptr(), gemm2_weights_scale.data_ptr(), loff, rsf)
    gs = _gstate.get(key)
    if gs is None or gs["sig"] != sig:
        rl_b = routing_logits.clone(); rb_b = routing_bias.clone()
        hs_b = hidden_states.clone(); hss_b = hidden_states_scale.clone()
        # warmup (compile all kernels for every bucket)
        for _ in range(2):
            _pre_impl(rl_b, rb_b, hs_b, hss_b, loff, rsf, ws, T)
            for b in _BUCKETS:
                _gemm_impl(gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, ws, T, b)
        torch.cuda.synchronize()
        # pre-graph (routing + layout)
        gpre = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gpre):
            _pre_impl(rl_b, rb_b, hs_b, hss_b, loff, rsf, ws, T)
        # one gemm-graph per bucket
        ggemm = {}
        for b in _BUCKETS:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outb = _gemm_impl(gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, ws, T, b)
            ggemm[b] = g
        gs = dict(gpre=gpre, ggemm=ggemm, rl=rl_b, rb=rb_b, hs=hs_b, hss=hss_b,
                  out=ws["out_bf16"], sig=sig)
        _gstate[key] = gs

    gs["rl"].copy_(routing_logits); gs["rb"].copy_(routing_bias)
    gs["hs"].copy_(hidden_states); gs["hss"].copy_(hidden_states_scale)
    gs["gpre"].replay()
    nb = int(ws["nblocks"].item())
    gs["ggemm"][_bucket(nb)].replay()
    return gs["out"].clone()
