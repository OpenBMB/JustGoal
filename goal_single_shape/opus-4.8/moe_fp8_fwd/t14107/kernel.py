import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

_alloc_set = False
def _ensure_alloc():
    global _alloc_set
    if not _alloc_set:
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))
        _alloc_set = True


# ---------------- Routing (fused Triton, one program per token) ----------------
@triton.jit
def route_kernel(Logits, Bias, TopkIdx, TopkW, Counts, T, rsf, LO: tl.constexpr, HI: tl.constexpr,
                 E: tl.constexpr, NG: tl.constexpr, GSZ: tl.constexpr,
                 TKG: tl.constexpr, TOPK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= T:
        return
    off = tl.arange(0, E)
    l = tl.load(Logits + pid * E + off).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-l))
    b = tl.load(Bias + off).to(tl.float32)
    swb = s + b
    NEG = -1e30
    # group top-2 sum  (reshape [NG,GSZ]); remove exactly one max (handle ties)
    g = tl.reshape(swb, (NG, GSZ))
    m1 = tl.max(g, axis=1)
    idx1 = tl.argmax(g, axis=1)           # [NG]
    colr = tl.arange(0, GSZ)
    g2 = tl.where(colr[None, :] == idx1[:, None], NEG, g)
    m2 = tl.max(g2, axis=1)
    gscore = m1 + m2                      # [NG]
    # pick top-TKG groups
    gmask = tl.zeros((NG,), dtype=tl.int1)
    gs = gscore
    gidx_arange = tl.arange(0, NG)
    for _ in range(TKG):
        gm = tl.max(gs, axis=0)
        sel = (gs >= gm) & (~gmask)
        # pick first matching
        selidx = tl.min(tl.where(sel, gidx_arange, NG))
        gmask = gmask | (gidx_arange == selidx)
        gs = tl.where(gidx_arange == selidx, NEG, gs)
    # expand group mask to experts
    emask = tl.reshape(gmask[:, None] & tl.full((NG, GSZ), True, tl.int1), (E,))
    scores = tl.where(emask, swb, NEG)
    # top-TOPK experts
    ar = tl.arange(0, E)
    wsum = 0.0
    # first pass: find indices, accumulate s
    for j in range(TOPK):
        mx = tl.max(scores, axis=0)
        sel = (scores >= mx)
        selidx = tl.min(tl.where(sel, ar, E))
        sval = tl.sum(tl.where(ar == selidx, s, 0.0))
        tl.store(TopkIdx + pid * TOPK + j, selidx.to(tl.int32))
        tl.store(TopkW + pid * TOPK + j, sval)    # temporarily store s value
        if (selidx >= LO) & (selidx < HI):
            tl.atomic_add(Counts + (selidx - LO), 1)
        wsum += sval
        scores = tl.where(ar == selidx, NEG, scores)
    inv = rsf / (wsum + 1e-20)
    for j in range(TOPK):
        v = tl.load(TopkW + pid * TOPK + j)
        tl.store(TopkW + pid * TOPK + j, v * inv)


@torch.no_grad()
def route(routing_logits, routing_bias, rsf, local_offset, E_local):
    T, E = routing_logits.shape
    dev = routing_logits.device
    topk_idx = torch.empty((T, 8), dtype=torch.int32, device=dev)
    topk_w = torch.empty((T, 8), dtype=torch.float32, device=dev)
    counts = torch.zeros(E_local, dtype=torch.int32, device=dev)
    route_kernel[(T,)](routing_logits, routing_bias, topk_idx, topk_w, counts, T, float(rsf),
                       local_offset, local_offset + E_local, E, 8, E // 8, 4, 8, num_warps=1)
    return topk_idx, topk_w, counts


@triton.jit
def count_experts_kernel(TopkIdx, Counts, T, lo, hi, TK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < T:
        for s in range(TK):
            idx = tl.load(TopkIdx + pid * TK + s)
            if (idx >= lo) & (idx < hi):
                tl.atomic_add(Counts + (idx - lo), 1)


@triton.jit
def scatter_experts_kernel(TopkIdx, TopkW, Wptr, PermToken, PermW, T, lo, hi, TK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < T:
        for s in range(TK):
            idx = tl.load(TopkIdx + pid * TK + s)
            if (idx >= lo) & (idx < hi):
                pos = tl.atomic_add(Wptr + (idx - lo), 1)
                tl.store(PermToken + pos, pid)
                tl.store(PermW + pos, tl.load(TopkW + pid * TK + s))


@triton.jit
def count_tokens_kernel(PermToken, Tcount, N):
    pid = tl.program_id(0)
    if pid < N:
        tl.atomic_add(Tcount + tl.load(PermToken + pid), 1)


@triton.jit
def scatter_tokens_kernel(PermToken, Twptr, RowSorted, N):
    pid = tl.program_id(0)
    if pid < N:
        pos = tl.atomic_add(Twptr + tl.load(PermToken + pid), 1)
        tl.store(RowSorted + pos, pid)


@torch.no_grad()
def build_perm(topk_idx, topk_w, counts, local_offset, E_local, BM):
    dev = topk_idx.device
    T = topk_idx.shape[0]
    lo, hi = local_offset, local_offset + E_local
    NMAX = T * 8
    real_start = torch.zeros(E_local + 1, dtype=torch.int32, device=dev)
    torch.cumsum(counts, 0, out=real_start[1:])
    perm_token = torch.empty(NMAX, dtype=torch.int32, device=dev)
    perm_w = torch.empty(NMAX, dtype=torch.float32, device=dev)
    wptr = real_start[:E_local].clone()
    scatter_experts_kernel[(T,)](topk_idx, topk_w, wptr, perm_token, perm_w, T, lo, hi, 8, num_warps=1)
    tiles_per_e = (counts + BM - 1) // BM
    tile_off = torch.zeros(E_local + 1, dtype=torch.int32, device=dev)
    torch.cumsum(tiles_per_e, 0, out=tile_off[1:])
    both = torch.stack([real_start[E_local], tile_off[E_local]]).tolist()  # one sync
    N, total_tiles = both[0], both[1]
    # token-grouped layout for no-atomic reduction
    tcount = torch.zeros(T, dtype=torch.int32, device=dev)
    count_tokens_kernel[(N,)](perm_token, tcount, N)
    tstart = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    torch.cumsum(tcount, 0, out=tstart[1:])
    twptr = tstart[:T].clone()
    row_sorted = torch.empty(N, dtype=torch.int32, device=dev)
    scatter_tokens_kernel[(N,)](perm_token, twptr, row_sorted, N)
    return perm_token, perm_w, real_start, tile_off, total_tiles, N, row_sorted, tstart


# ---------------- GEMM1: block-scaled, TMA ----------------
@triton.jit
def gemm1_kernel(a_desc, w_desc, Ascale, S13, G1, tile_off, real_start,
                 N, H: tl.constexpr, Nout: tl.constexpr, KB: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, NT: tl.constexpr, E_local: tl.constexpr):
    t = tl.program_id(0)
    e = 0
    for ee in tl.static_range(1, E_local + 1):
        e += (t >= tl.load(tile_off + ee) * NT).to(tl.int32)
    base = tl.load(tile_off + e)
    mt = tl.load(tile_off + e + 1) - base
    local = t - base * NT
    n_tile = local // mt
    m_within = local % mt
    rstart = tl.load(real_start + e)
    rend = tl.load(real_start + e + 1)
    row0 = rstart + m_within * BM
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    n0 = n_tile * BN
    wbase = e * (32 * KB) + (n0 // 128) * KB
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(KB):
        a = a_desc.load([row0, kb * 128])
        w = w_desc.load([e * Nout + n0, kb * 128])
        asc = tl.load(Ascale + kb * N + rows, mask=rmask, other=0.0)
        wsc = tl.load(S13 + wbase + kb)
        acc += tl.dot(a, w.T) * (asc * wsc)[:, None]
    offs_n = n0 + tl.arange(0, BN)
    tl.store(G1 + rows[:, None] * Nout + offs_n[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


# ---------------- SwiGLU + per-128-block fp8 quant ----------------
@triton.jit
def swiglu_quant_kernel(G1, Cfp8, Cscale, N, I: tl.constexpr, Nout: tl.constexpr,
                        NB: tl.constexpr, ROWS: tl.constexpr):
    # one program handles ROWS rows, all NB blocks of 128 along I
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    rmask = r < N
    bj = tl.arange(0, NB)[:, None] * 128 + tl.arange(0, 128)[None, :]   # [NB,128] col idx
    for rr in tl.static_range(ROWS):
        row = pid * ROWS + rr
        if row < N:
            x1 = tl.load(G1 + row * Nout + bj).to(tl.float32)            # [NB,128]
            x2 = tl.load(G1 + row * Nout + I + bj).to(tl.float32)
            c = (x2 * (1.0 / (1.0 + tl.exp(-x2)))) * x1
            amax = tl.max(tl.abs(c), axis=1)                             # [NB]
            scale = amax / 448.0
            inv = tl.where(scale > 0, 1.0 / scale, 0.0)
            q = tl.minimum(tl.maximum(c * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
            tl.store(Cfp8 + row * I + bj, q)
            tl.store(Cscale + tl.arange(0, NB) * N + row, scale)


# ---------------- GEMM2: block-scaled, TMA, weighted scatter ----------------
@triton.jit
def gemm2_kernel(c_desc, w_desc, Cscale, S2, Wtok, Ob,
                 tile_off, real_start,
                 N, H: tl.constexpr, I: tl.constexpr, KB: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, NT: tl.constexpr, E_local: tl.constexpr):
    t = tl.program_id(0)
    e = 0
    for ee in tl.static_range(1, E_local + 1):
        e += (t >= tl.load(tile_off + ee) * NT).to(tl.int32)
    base = tl.load(tile_off + e)
    mt = tl.load(tile_off + e + 1) - base
    local = t - base * NT
    n_tile = local // mt
    m_within = local % mt
    rstart = tl.load(real_start + e)
    rend = tl.load(real_start + e + 1)
    row0 = rstart + m_within * BM
    rows = row0 + tl.arange(0, BM)
    rmask = rows < rend
    n0 = n_tile * BN
    wbase = e * ((H // 128) * KB) + (n0 // 128) * KB
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(KB):
        c = c_desc.load([row0, kb * 128])
        w = w_desc.load([e * H + n0, kb * 128])
        csc = tl.load(Cscale + kb * N + rows, mask=rmask, other=0.0)
        wsc = tl.load(S2 + wbase + kb)
        acc += tl.dot(c, w.T) * (csc * wsc)[:, None]
    wt = tl.load(Wtok + rows, mask=rmask, other=0.0)
    acc = acc * wt[:, None]
    offs_n = n0 + tl.arange(0, BN)
    tl.store(Ob + rows[:, None] * H + offs_n[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def reduce_kernel(Ob, RowSorted, Tstart, Out, H: tl.constexpr, BLK: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1) * BLK + tl.arange(0, BLK)
    s = tl.load(Tstart + t)
    e = tl.load(Tstart + t + 1)
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for i in range(0, e - s):
        r = tl.load(RowSorted + s + i)
        acc += tl.load(Ob + r * H + cb).to(tl.float32)
    tl.store(Out + t * H + cb, acc.to(tl.bfloat16))


# ---------------- Orchestration ----------------
@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    _ensure_alloc()
    H, I = 7168, 2048
    Nout = 2 * I
    E_local = gemm1_weights.shape[0]
    T = routing_logits.shape[0]
    KB1, KB2 = H // 128, I // 128
    dev = hidden_states.device
    BM = 64
    BN1, BN2 = 128, 128

    topk_idx, topk_w, counts = route(routing_logits, routing_bias, float(routed_scaling_factor),
                                     int(local_expert_offset), E_local)
    perm_token, perm_w, real_start, tile_off, total_tiles, N, row_sorted, tstart = \
        build_perm(topk_idx, topk_w, counts, int(local_expert_offset), E_local, BM)
    ptl = perm_token[:N].long()
    Aperm = hidden_states.index_select(0, ptl).contiguous()
    Ascale = hidden_states_scale.index_select(1, ptl).contiguous()

    G1 = torch.empty((N, Nout), dtype=torch.bfloat16, device=dev)
    ad = TensorDescriptor.from_tensor(Aperm, [BM, 128])
    wd1 = TensorDescriptor.from_tensor(gemm1_weights.view(E_local * Nout, H), [BN1, 128])
    NT1 = Nout // BN1
    gemm1_kernel[(total_tiles * NT1,)](ad, wd1, Ascale, gemm1_weights_scale, G1,
                                       tile_off, real_start,
                                       N, H, Nout, KB1, BM, BN1, NT1, E_local, num_warps=4, num_stages=3)

    Cfp8 = torch.empty((N, I), dtype=torch.float8_e4m3fn, device=dev)
    Cscale = torch.empty((KB2, N), dtype=torch.float32, device=dev)
    ROWS = 4
    swiglu_quant_kernel[((N + ROWS - 1) // ROWS,)](G1, Cfp8, Cscale, N, I, Nout, KB2, ROWS, num_warps=4)

    Ob = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
    cd = TensorDescriptor.from_tensor(Cfp8, [BM, 128])
    wd2 = TensorDescriptor.from_tensor(gemm2_weights.view(E_local * H, I), [BN2, 128])
    NT2 = H // BN2
    gemm2_kernel[(total_tiles * NT2,)](cd, wd2, Cscale, gemm2_weights_scale,
                                       perm_w, Ob,
                                       tile_off, real_start,
                                       N, H, I, KB2, BM, BN2, NT2, E_local, num_warps=4, num_stages=3)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    reduce_kernel[(T, H // 1024)](Ob, row_sorted, tstart, out, H, 1024, num_warps=4)
    return out
