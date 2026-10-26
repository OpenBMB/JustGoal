"""From-scratch varlen MHA causal attention backward (Triton).
FA2-style: two kernels (dK/dV parallel over KV blocks; dQ parallel over Q blocks),
avoiding atomics. Derived from the FlashAttention math, not from FA4 source.

Layout: q,k,v,o,do : (T,H,D). lse,delta : (T,H). grads (T,H,D).

dkdv kernel: parallel over KV blocks. Works in transposed (BN,BM) orientation so
  the value-direction matmuls (dv, dk) have no in-loop transpose of computed tensors.
dq kernel: parallel over Q blocks.
"""
import torch
import triton
import triton.language as tl


# ---------------- preprocessing (NOT timed): O, LSE ----------------

def forward_ref_state(q, k, v, cu, scale):
    T, H, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(T, H, device=q.device, dtype=torch.float32)
    cul = cu.tolist()
    for i in range(len(cul) - 1):
        s, e = cul[i], cul[i + 1]
        L = e - s
        qs = q[s:e].transpose(0, 1).float()
        ks = k[s:e].transpose(0, 1).float()
        vs = v[s:e].transpose(0, 1).float()
        sc = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        sc.masked_fill_(mask, float("-inf"))
        m = sc.max(dim=-1, keepdim=True).values
        p = torch.exp(sc - m)
        denom = p.sum(dim=-1, keepdim=True)
        lse_s = (m + torch.log(denom)).squeeze(-1)
        os = torch.matmul(p / denom, vs)
        o[s:e] = os.transpose(0, 1).to(q.dtype)
        lse[s:e] = lse_s.transpose(0, 1)
    return o, lse


@triton.jit
def _delta_kernel(O, DO, Delta, T, H, D: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs = pid * BLK + tl.arange(0, BLK)
    d = tl.arange(0, D)
    mask = offs < T
    o = tl.load(O + (offs[:, None] * H + h) * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + (offs[:, None] * H + h) * D + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs * H + h, delta, mask=mask)


def compute_delta(o, do):
    T, H, D = o.shape
    delta = torch.empty(T, H, device=o.device, dtype=torch.float32)
    BLK = 128
    _delta_kernel[(triton.cdiv(T, BLK), H)](o, do, delta, T, H, D, BLK)
    return delta


# ---------------- block schedule for varlen ----------------

def build_schedule(cu, BLOCK, kind="kv", sort=True):
    """kind='kv': dkdv workload for a kv block at b ~ (L - b*BLOCK) (downstream queries).
       kind='q' : dq workload for a q block at b ~ (b+1)*BLOCK (upstream kvs).
       sort=True orders blocks by descending workload for better load balance."""
    cul = cu.tolist()
    entries = []  # (seq0, L, blk, work)
    for i in range(len(cul) - 1):
        s, e = cul[i], cul[i + 1]
        L = e - s
        nb = (L + BLOCK - 1) // BLOCK
        for b in range(nb):
            if kind == "kv":
                work = L - b * BLOCK
            else:
                work = (b + 1) * BLOCK
            entries.append((s, L, b, work))
    if sort:
        entries.sort(key=lambda x: -x[3])
    dev = cu.device
    seq0 = [e[0] for e in entries]; seqlen = [e[1] for e in entries]; blk = [e[2] for e in entries]
    return (torch.tensor(seq0, dtype=torch.int32, device=dev),
            torch.tensor(seqlen, dtype=torch.int32, device=dev),
            torch.tensor(blk, dtype=torch.int32, device=dev))


# ---------------- dK/dV kernel (transposed orientation) ----------------

@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV,
              Seq0, Seqlen, Blk, scale,
              H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid)
    L = tl.load(Seqlen + pid)
    nblk = tl.load(Blk + pid)
    n0 = nblk * BN
    offs_n = n0 + tl.arange(0, BN)
    d = tl.arange(0, D)
    n_mask = offs_n < L

    kp = K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :]
    vp = V + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :]
    k = tl.load(kp, mask=n_mask[:, None], other=0.0)   # (BN,D)
    v = tl.load(vp, mask=n_mask[:, None], other=0.0)   # (BN,D)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)

    # diagonal band: query rows overlap kv col range -> needs causal mask
    diag_end = n0 + BN
    m_start = (n0 // BM) * BM
    for m in range(m_start, diag_end, BM):
        offs_m = m + tl.arange(0, BM)
        m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q))            # q is pre-scaled by scale*log2e
        valid = (offs_m[None, :] >= offs_n[:, None]) & m_mask[None, :] & n_mask[:, None]
        pT = tl.where(valid, tl.exp2(sT - lse[None, :]), 0.0)   # lse is lse2 (log2 units)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)

    # bulk: query rows strictly below kv range -> no causal mask (only boundary)
    for m in range(diag_end, L, BM):
        offs_m = m + tl.arange(0, BM)
        m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(sT - lse[None, :])
        pT = tl.where(m_mask[None, :] & n_mask[:, None], pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)

    dk *= scale    # scale slot carries inv_log2e (ln2) for dkdv
    tl.store(DK + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


# ---------------- TMA-based dK/dV kernel ----------------

@triton.jit
def _bwd_dkdv_tma(Q, K, V, DO, LSE, Delta, DK, DV,
                  Seq0, Seqlen, Blk, scale, T,
                  H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid); L = tl.load(Seqlen + pid); nblk = tl.load(Blk + pid)
    n0 = nblk * BN
    offs_n = n0 + tl.arange(0, BN); d = tl.arange(0, D)
    n_mask = offs_n < L

    dq_k = tl.make_tensor_descriptor(Q + h * D, [T, D], [H * D, 1], [BM, D])
    dk_k = tl.make_tensor_descriptor(K + h * D, [T, D], [H * D, 1], [BN, D])
    dv_k = tl.make_tensor_descriptor(V + h * D, [T, D], [H * D, 1], [BN, D])
    ddo_k = tl.make_tensor_descriptor(DO + h * D, [T, D], [H * D, 1], [BM, D])

    k = dk_k.load([seq0 + n0, 0])
    v = dv_k.load([seq0 + n0, 0])
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)

    diag_end = n0 + BN
    m_start = (n0 // BM) * BM
    for m in range(m_start, diag_end, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = dq_k.load([seq0 + m, 0])
        do = ddo_k.load([seq0 + m, 0])
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q))
        valid = (offs_m[None, :] >= offs_n[:, None]) & m_mask[None, :] & n_mask[:, None]
        pT = tl.where(valid, tl.exp2(sT - lse[None, :]), 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    for m in range(diag_end, L, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = dq_k.load([seq0 + m, 0])
        do = ddo_k.load([seq0 + m, 0])
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(sT - lse[None, :])
        pT = tl.where(m_mask[None, :] & n_mask[:, None], pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)

    dk *= scale
    tl.store(DK + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


# ---------------- separate dV kernel (single accumulator, large tiles) ----------------

@triton.jit
def _bwd_dv_k(Q, K, V, DO, LSE, Delta, DV,
             Seq0, Seqlen, Blk, scale,
             H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid); L = tl.load(Seqlen + pid); nblk = tl.load(Blk + pid)
    n0 = nblk * BN
    offs_n = n0 + tl.arange(0, BN); d = tl.arange(0, D)
    n_mask = offs_n < L
    k = tl.load(K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    diag_end = n0 + BN
    m_start = (n0 // BM) * BM
    for m in range(m_start, diag_end, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale
        valid = (offs_m[None, :] >= offs_n[:, None]) & m_mask[None, :] & n_mask[:, None]
        pT = tl.where(valid, tl.exp2((sT - lse[None, :]) * 1.4426950408889634), 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
    for m in range(diag_end, L, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale
        pT = tl.where(m_mask[None, :] & n_mask[:, None], tl.exp2((sT - lse[None, :]) * 1.4426950408889634), 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
    tl.store(DV + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


@triton.jit
def _bwd_dk_k(Q, K, V, DO, LSE, Delta, DK,
             Seq0, Seqlen, Blk, scale,
             H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid); L = tl.load(Seqlen + pid); nblk = tl.load(Blk + pid)
    n0 = nblk * BN
    offs_n = n0 + tl.arange(0, BN); d = tl.arange(0, D)
    n_mask = offs_n < L
    k = tl.load(K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
    v = tl.load(V + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    diag_end = n0 + BN
    m_start = (n0 // BM) * BM
    for m in range(m_start, diag_end, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale
        valid = (offs_m[None, :] >= offs_n[:, None]) & m_mask[None, :] & n_mask[:, None]
        pT = tl.where(valid, tl.exp2((sT - lse[None, :]) * 1.4426950408889634), 0.0)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    for m in range(diag_end, L, BM):
        offs_m = m + tl.arange(0, BM); m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        sT = tl.dot(k, tl.trans(q)) * scale
        pT = tl.where(m_mask[None, :] & n_mask[:, None], tl.exp2((sT - lse[None, :]) * 1.4426950408889634), 0.0)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    dk *= scale
    tl.store(DK + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask[:, None])


# ---------------- fused kernel: dk,dv + atomic dq (p computed once) ----------------

@triton.jit
def _bwd_fused(Q, K, V, DO, LSE, Delta, DK, DV, DQ,
               Seq0, Seqlen, Blk, scale,
               H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid)
    L = tl.load(Seqlen + pid)
    nblk = tl.load(Blk + pid)
    n0 = nblk * BN
    offs_n = n0 + tl.arange(0, BN)
    d = tl.arange(0, D)
    n_mask = offs_n < L

    k = tl.load(K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
    v = tl.load(V + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)

    m_start = (n0 // BM) * BM
    for m in range(m_start, L, BM):
        offs_m = m + tl.arange(0, BM)
        m_mask = offs_m < L
        q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)

        sT = tl.dot(k, tl.trans(q)) * scale          # (BN,BM)
        valid = (offs_m[None, :] >= offs_n[:, None]) & m_mask[None, :] & n_mask[:, None]
        pT = tl.where(valid, tl.exp2((sT - lse[None, :]) * 1.4426950408889634), 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))                # (BN,BM)
        dsT = pT * (dpT - delta[None, :])            # (BN,BM)
        dk += tl.dot(dsT.to(q.dtype), q)
        # dq contribution for this query block: (BM,D) = scale * ds @ k = scale * dsT^T @ k
        dq_blk = tl.dot(tl.trans(dsT).to(k.dtype), k) * scale
        dqp = DQ + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :]
        tl.atomic_add(dqp, dq_blk, mask=m_mask[:, None])

    dk *= scale
    tl.store(DK + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


# ---------------- TMA dQ kernel ----------------

@triton.jit
def _bwd_dq_tma(Q, K, V, DO, LSE, Delta, DQ,
                Seq0, Seqlen, Blk, scale, T,
                H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid); L = tl.load(Seqlen + pid); mblk = tl.load(Blk + pid)
    m0 = mblk * BM
    offs_m = m0 + tl.arange(0, BM); d = tl.arange(0, D)
    m_mask = offs_m < L

    dq_k = tl.make_tensor_descriptor(Q + h * D, [T, D], [H * D, 1], [BM, D])
    dk_k = tl.make_tensor_descriptor(K + h * D, [T, D], [H * D, 1], [BN, D])
    dv_k = tl.make_tensor_descriptor(V + h * D, [T, D], [H * D, 1], [BN, D])
    ddo_k = tl.make_tensor_descriptor(DO + h * D, [T, D], [H * D, 1], [BM, D])

    q = dq_k.load([seq0 + m0, 0])
    do = ddo_k.load([seq0 + m0, 0])
    lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BM, D], dtype=tl.float32)

    for n in range(0, m0, BN):
        k = dk_k.load([seq0 + n, 0])
        v = dv_k.load([seq0 + n, 0])
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq += tl.dot(ds, k)
    for n in range(m0, m0 + BM, BN):
        offs_n = n + tl.arange(0, BN); n_mask = offs_n < L
        k = dk_k.load([seq0 + n, 0])
        v = dv_k.load([seq0 + n, 0])
        s = tl.dot(q, tl.trans(k))
        p = tl.where(offs_m[:, None] >= offs_n[None, :], tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq += tl.dot(ds, k)

    dq *= scale
    tl.store(DQ + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


# ---------------- dQ kernel ----------------

@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, Delta, DQ,
           Seq0, Seqlen, Blk, scale,
           H, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    seq0 = tl.load(Seq0 + pid)
    L = tl.load(Seqlen + pid)
    mblk = tl.load(Blk + pid)
    m0 = mblk * BM
    offs_m = m0 + tl.arange(0, BM)
    d = tl.arange(0, D)
    m_mask = offs_m < L

    q = tl.load(Q + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(LSE + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + (seq0 + offs_m) * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BM, D], dtype=tl.float32)

    # bulk: kv cols strictly left of query rows -> no causal mask
    for n in range(0, m0, BN):
        offs_n = n + tl.arange(0, BN)
        k = tl.load(K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :])
        v = tl.load(V + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :])
        s = tl.dot(q, tl.trans(k))           # q pre-scaled by scale*log2e
        p = tl.exp2(s - lse[:, None])        # lse is lse2
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq += tl.dot(ds, k)

    # diagonal band: needs causal mask
    for n in range(m0, m0 + BM, BN):
        offs_n = n + tl.arange(0, BN)
        n_mask = offs_n < L
        k = tl.load(K + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(V + ((seq0 + offs_n)[:, None] * H + h) * D + d[None, :], mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        p = tl.where(offs_m[:, None] >= offs_n[None, :], tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq += tl.dot(ds, k)

    dq *= scale
    tl.store(DQ + ((seq0 + offs_m)[:, None] * H + h) * D + d[None, :], dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


# ---------------- driver ----------------

DEFAULT = dict(kv_BM=64, kv_BN=128, kv_w=8, kv_s=2, kv_tma=True,
               q_BM=128, q_BN=128, q_w=8, q_s=2, q_tma=False)

_alloc_set = [False]


def _ensure_alloc():
    if not _alloc_set[0]:
        def alloc_fn(size, alignment, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        _alloc_set[0] = True


class Bwd:
    def __init__(self, q, k, v, cu, scale, cfg=None):
        self.scale = scale
        self.cu = cu
        self.T = q.shape[0]; self.H = q.shape[1]; self.D = q.shape[2]
        self.cfg = dict(DEFAULT)
        if cfg:
            self.cfg.update(cfg)
        c = self.cfg
        self.sched_kv = build_schedule(cu, c["kv_BN"], kind="kv", sort=c.get("sort", True))
        self.sched_q = build_schedule(cu, c["q_BM"], kind="q", sort=c.get("sort", True))
        if c.get("kv_tma", True) or c.get("q_tma", False):
            _ensure_alloc()

    def run(self, q2, k, v, do, lse2, delta):
        """q2 = q * (scale*log2e), lse2 = lse*log2e. dkdv uses inv_log2e (ln2)
        as final multiplier; dq uses scale."""
        H, D, T = self.H, self.D, self.T
        c = self.cfg
        LN2 = 0.6931471805599453
        dq = torch.empty_like(q2); dk = torch.empty_like(k); dv = torch.empty_like(v)
        s0n, sln, bln = self.sched_kv
        s0q, slq, blq = self.sched_q
        if c.get("kv_tma", True):
            _bwd_dkdv_tma[(s0n.numel(), H)](q2, k, v, do, lse2, delta, dk, dv,
                                            s0n, sln, bln, LN2, T, H, D,
                                            c["kv_BM"], c["kv_BN"], num_warps=c["kv_w"], num_stages=c["kv_s"])
        else:
            _bwd_dkdv[(s0n.numel(), H)](q2, k, v, do, lse2, delta, dk, dv,
                                        s0n, sln, bln, LN2, H, D,
                                        c["kv_BM"], c["kv_BN"], num_warps=c["kv_w"], num_stages=c["kv_s"])
        if c.get("q_tma", False):
            _bwd_dq_tma[(s0q.numel(), H)](q2, k, v, do, lse2, delta, dq,
                                          s0q, slq, blq, self.scale, T, H, D,
                                          c["q_BM"], c["q_BN"], num_warps=c["q_w"], num_stages=c["q_s"])
        else:
            _bwd_dq[(s0q.numel(), H)](q2, k, v, do, lse2, delta, dq,
                                      s0q, slq, blq, self.scale, H, D,
                                      c["q_BM"], c["q_BN"], num_warps=c["q_w"], num_stages=c["q_s"])
        return dq, dk, dv


LOG2E = 1.4426950408889634


def make_state(q, k, v, cu, scale):
    """Untimed forward state: O, LSE. Also returns pre-scaled q2 and lse2."""
    o, lse = forward_ref_state(q, k, v, cu, scale)
    q2 = (q.float() * (scale * LOG2E)).to(q.dtype)
    lse2 = lse * LOG2E
    return o, lse, q2, lse2


if __name__ == "__main__":
    import problem as P
    dev = "cuda"
    q, k, v = P.make_inputs(dev, requires_grad=False)
    cu = P.cu_seqlens(dev)
    do = P.make_dout(dev)
    scale = P.SOFTMAX_SCALE
    o, lse, q2, lse2 = make_state(q, k, v, cu, scale)
    delta = compute_delta(o, do)
    bwd = Bwd(q, k, v, cu, scale)
    dq, dk, dv = bwd.run(q2, k, v, do, lse2, delta)
    from naive import naive_backward
    dqn, dkn, dvn = naive_backward(q, k, v, do, cu)
    for name, a, b in [("dq", dq, dqn), ("dk", dk, dkn), ("dv", dv, dvn)]:
        err = (a.float() - b).abs().max().item()
        print(f"{name}: max abs err vs naive = {err:.6e}  shape={tuple(a.shape)} dtype={a.dtype}")
