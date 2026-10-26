import torch
import triton
import triton.language as tl

H = 7168
I = 2048
TWOI = 4096
E_LOCAL = 32
NKB = 56
NIB = 16
E_GLOBAL = 256
N_GROUP = 8
GROUP_SIZE = 32
TOP_K = 8
TOPK_GROUP = 4


@triton.jit
def route_kernel(logits_ptr, bias_ptr, lw_ptr, emap_ptr, offset, scaling,
                 E: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                 TOPK: tl.constexpr, TOPG: tl.constexpr, EL: tl.constexpr):
    if tl.program_id(0) != 0:
        return
    NEG = -1e30
    off = tl.arange(0, E)
    s = tl.sigmoid(tl.load(logits_ptr + off).to(tl.float32))
    swb = s + tl.load(bias_ptr + off).to(tl.float32)
    swb_g = tl.reshape(swb, (G, GS))
    m1 = tl.max(swb_g, axis=1)
    m2 = tl.max(tl.where(swb_g == m1[:, None], NEG, swb_g), axis=1)
    gscore = m1 + m2
    gs_sorted = tl.sort(gscore, descending=True)
    gthr = tl.sum(tl.where(tl.arange(0, G) == TOPG - 1, gs_sorted, 0.0))
    gsel = gscore >= gthr
    esel = tl.reshape(tl.broadcast_to(gsel[:, None], (G, GS)), (E,))
    msw = tl.where(esel, swb, NEG)
    msw_sorted = tl.sort(msw, descending=True)
    ethr = tl.sum(tl.where(off == TOPK - 1, msw_sorted, 0.0))
    topsel = msw >= ethr
    w = s * topsel.to(tl.float32)
    weights = (w / (tl.sum(w) + 1e-20)) * scaling
    loc = tl.arange(0, EL)
    gidx = offset + loc
    lw = tl.where((gidx >= 0) & (gidx < E),
                  tl.sum(tl.where(off[None, :] == gidx[:, None], weights[None, :], 0.0), axis=1), 0.0)
    tl.store(lw_ptr + loc, lw)
    sel = (lw > 0.0).to(tl.int32)
    pos = tl.cumsum(sel, axis=0) - sel
    match = (sel[:, None] == 1) & (pos[:, None] == loc[None, :])
    emap = tl.where(tl.sum(match.to(tl.int32), axis=0) > 0, tl.sum(tl.where(match, loc[:, None], 0), axis=0), -1)
    tl.store(emap_ptr + loc, emap)


@triton.jit
def gemm1_kernel(a_ptr, as_ptr, w_ptr, ws_ptr, emap_ptr, g1_ptr,
                 BN: tl.constexpr, CK: tl.constexpr, NKB: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(emap_ptr + slot)
    if e < 0:
        return
    row = pid * BN + tl.arange(0, BN)      # g1 row in [0, 4096)
    rb = row // 128
    wb = e.to(tl.int64) * 4096 * 7168
    sb = e * 32 * NKB
    SUB: tl.constexpr = CK // 128
    NIT: tl.constexpr = 7168 // CK
    offk = tl.arange(0, CK)
    sub_id = offk // 128
    ar = tl.arange(0, SUB)
    acc = tl.zeros([BN], tl.float32)
    for it in range(NIT):
        kb0 = it * SUB
        col = it * CK + offk
        adeq = tl.load(as_ptr + kb0 + sub_id) * tl.load(a_ptr + col).to(tl.float32)
        wm = tl.load(w_ptr + wb + row[:, None].to(tl.int64) * 7168 + col[None, :]).to(tl.float32)
        p = tl.sum(tl.reshape(wm * adeq[None, :], (BN, SUB, 128)), axis=2)
        sc = tl.load(ws_ptr + sb + rb[:, None] * NKB + (kb0 + ar)[None, :])
        acc += tl.sum(p * sc, axis=1)
    tl.store(g1_ptr + e * 4096 + row, acc.to(tl.bfloat16))


@triton.jit
def gemm2_kernel(g1_ptr, w_ptr, ws_ptr, emap_ptr, lw_ptr, out_ptr,
                 BN: tl.constexpr, CK: tl.constexpr, NIB: tl.constexpr, MAXSLOT: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BN + tl.arange(0, BN)      # h in [0, H)
    rb = row // 128
    SUB: tl.constexpr = CK // 128
    NIT: tl.constexpr = 2048 // CK
    offk = tl.arange(0, CK)
    sub_id = offk // 128
    ar = tl.arange(0, SUB)
    acc = tl.zeros([BN], tl.float32)
    for slot in range(MAXSLOT):
        e = tl.load(emap_ptr + slot)
        if e >= 0:
            wgt = tl.load(lw_ptr + e)
            wb = e.to(tl.int64) * 7168 * 2048
            sb = e * 56 * NIB
            eacc = tl.zeros([BN], tl.float32)
            for it in range(NIT):
                ib0 = it * SUB
                col = it * CK + offk
                x1 = tl.load(g1_ptr + e * 4096 + col).to(tl.float32)
                x2 = tl.load(g1_ptr + e * 4096 + 2048 + col).to(tl.float32)
                cc = (x2 * tl.sigmoid(x2)) * x1            # inline SwiGLU [CK]
                wm = tl.load(w_ptr + wb + row[:, None].to(tl.int64) * 2048 + col[None, :]).to(tl.float32)
                p = tl.sum(tl.reshape(wm * cc[None, :], (BN, SUB, 128)), axis=2)
                sc = tl.load(ws_ptr + sb + rb[:, None] * NIB + (ib0 + ar)[None, :])
                eacc += tl.sum(p * sc, axis=1)
            acc += wgt * eacc
    tl.store(out_ptr + row, acc.to(tl.bfloat16))


_ws = {}


def _get_ws(device):
    if device not in _ws:
        _ws[device] = {
            "lw": torch.empty(E_LOCAL, dtype=torch.float32, device=device),
            "emap": torch.empty(E_LOCAL, dtype=torch.int32, device=device),
            "g1": torch.empty(E_LOCAL * TWOI, dtype=torch.bfloat16, device=device),
            "out_bf16": torch.empty(H, dtype=torch.bfloat16, device=device),
        }
    return _ws[device]


BN1 = 8
CK1 = 1024
NW1 = 2
BN2 = 16
CK2 = 2048
NW2 = 4
MAXSLOT = 8


def _launch(ws, routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            offset, scaling):
    lw = ws["lw"]; emap = ws["emap"]; g1 = ws["g1"]; out = ws["out_bf16"]
    route_kernel[(1,)](
        routing_logits, routing_bias, lw, emap, offset, scaling,
        E=E_GLOBAL, G=N_GROUP, GS=GROUP_SIZE, TOPK=TOP_K, TOPG=TOPK_GROUP, EL=E_LOCAL,
        num_warps=1,
    )
    gemm1_kernel[(TWOI // BN1, E_LOCAL)](
        hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, emap, g1,
        BN=BN1, CK=CK1, NKB=NKB, num_warps=NW1,
    )
    gemm2_kernel[(H // BN2,)](
        g1, gemm2_weights, gemm2_weights_scale, emap, lw, out,
        BN=BN2, CK=CK2, NIB=NIB, MAXSLOT=MAXSLOT, num_warps=NW2,
    )


_pool = {}
_warmed = {}


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    device = hidden_states.device
    ws = _get_ws(device)
    offset = int(local_expert_offset)
    scaling = float(routed_scaling_factor)
    args = (routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            offset, scaling)

    if device not in _warmed:
        for _ in range(3):
            _launch(ws, *args)
        torch.cuda.synchronize()
        _pool[device] = torch.cuda.graph_pool_handle()
        _warmed[device] = True

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_pool[device]):
        _launch(ws, *args)
    g.replay()
    return ws["out_bf16"].view(1, H)
