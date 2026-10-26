"""
GQA paged decode, page_size = 1, bf16, written from scratch for NVIDIA B200.

Shapes handled:
    q         (B, 32, 128)              bf16
    k_cache   (n_pages, 1, 8, 128)      bf16
    v_cache   (n_pages, 1, 8, 128)      bf16
    kv_indptr (B+1,)                    int32   ragged row pointers
    kv_indices(total_kv,)               int32   page id of every kv token
    sm_scale  float
  ->  out     (B, 32, 128)              bf16
      lse     (B, 32)                   float32   (natural log)

Design
------
* One CTA owns (batch, kv_head, kv_chunk).  The 4 q heads that share a kv head
  are packed into the M dimension of the MMA tile; M is padded to 16 because
  that is the smallest legal tl.dot tile.  The padded rows cost tensor-core
  flops only, and this kernel is bandwidth bound, so they are free.
* Flash-decoding style split over the kv axis with a fixed chunk length gives
  ~8x more CTAs than there are SMs, which hides the long-tail of the 496..3159
  token length distribution.
* The split reduction is fused into the same kernel: every CTA publishes its
  partial (acc, m, l), then an acq_rel atomic elects the last CTA of each
  (batch, kv_head) group to combine them.  That removes a second kernel launch
  (~4 us here) from the critical path.  The counter is reset by the winner, so
  the kernel is safe to replay inside a CUDA graph.
* Sequences short enough to fit in a single chunk bypass the partial buffers
  and write their result directly.
"""
import torch
import triton
import triton.language as tl

NEG = tl.constexpr(-3.0e38)


@triton.jit
def _gqa_decode_kernel(Q, KC, VC, INDPTR, INDICES, PACC, PML, COUNT, OUT, LSE,
                       sm_scale,
                       MAXS: tl.constexpr, CHUNK: tl.constexpr,
                       BN: tl.constexpr, PAD_M: tl.constexpr):
    s = tl.program_id(0)          # kv chunk index
    bh = tl.program_id(1)         # batch * 8 + kv head
    b = bh // 8
    h = bh % 8

    ks = tl.load(INDPTR + b).to(tl.int32)
    L = tl.load(INDPTR + b + 1).to(tl.int32) - ks
    lo = s * CHUNK
    if lo >= L:
        return
    hi = tl.minimum(lo + CHUNK, L)
    nsp = tl.cdiv(L, CHUNK)

    rm = tl.arange(0, PAD_M)
    rd = tl.arange(0, 128)
    qm = rm < 4                    # 4 real q heads per kv head, rest is padding
    qh = h * 4 + rm
    q = tl.load(Q + b * 4096 + qh[:, None] * 128 + rd[None, :],
                mask=qm[:, None], other=0.0)
    rdh = h * 128 + rd             # column offset of this kv head inside a page

    m_i = tl.full([PAD_M], NEG, tl.float32)
    l_i = tl.zeros([PAD_M], tl.float32)
    acc = tl.zeros([PAD_M, 128], tl.float32)

    for n0 in tl.range(lo, hi, BN):
        o = n0 + tl.arange(0, BN)
        msk = o < hi
        page = tl.load(INDICES + ks + o, mask=msk, other=0).to(tl.int64)
        base = page[:, None] * 1024 + rdh[None, :]
        k = tl.load(KC + base, mask=msk[:, None], other=0.0)
        v = tl.load(VC + base, mask=msk[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(msk[None, :], qk, NEG)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    if nsp == 1:
        o_ = acc / l_i[:, None]
        tl.store(OUT + b * 4096 + qh[:, None] * 128 + rd[None, :],
                 o_.to(OUT.dtype.element_ty), mask=qm[:, None])
        tl.store(LSE + b * 32 + qh, m_i + tl.log(l_i), mask=qm)
        return

    pb = (b * MAXS + s) * 4096 + qh[:, None] * 128 + rd[None, :]
    tl.store(PACC + pb, acc, mask=qm[:, None])
    mb = (b * MAXS + s) * 64 + qh
    tl.store(PML + mb, m_i, mask=qm)
    tl.store(PML + mb + 32, l_i, mask=qm)

    tl.debug_barrier()
    prev = tl.atomic_add(COUNT + bh, 1, sem="acq_rel", scope="gpu")
    if prev == nsp - 1:
        tl.atomic_xchg(COUNT + bh, 0, sem="release", scope="gpu")
        m = tl.full([PAD_M], NEG, tl.float32)
        l = tl.zeros([PAD_M], tl.float32)
        a = tl.zeros([PAD_M, 128], tl.float32)
        for t in range(0, nsp):
            mlb = (b * MAXS + t) * 64 + qh
            ms = tl.load(PML + mlb, mask=qm, other=NEG)
            ls = tl.load(PML + mlb + 32, mask=qm, other=0.0)
            at = tl.load(PACC + (b * MAXS + t) * 4096 + qh[:, None] * 128 + rd[None, :],
                         mask=qm[:, None], other=0.0)
            m2 = tl.maximum(m, ms)
            ca = tl.exp(m - m2)
            cb = tl.exp(ms - m2)
            l = l * ca + ls * cb
            a = a * ca[:, None] + at * cb[:, None]
            m = m2
        o_ = a / l[:, None]
        tl.store(OUT + b * 4096 + qh[:, None] * 128 + rd[None, :],
                 o_.to(OUT.dtype.element_ty), mask=qm[:, None])
        tl.store(LSE + b * 32 + qh, m + tl.log(l), mask=qm)


class GQAPagedDecode:
    """Pre-allocates the scratch buffers once; call it like a function."""

    def __init__(self, batch, max_seqlen, device="cuda",
                 chunk=640, bn=128, num_warps=4, num_stages=4, pad_m=16):
        self.B = batch
        self.chunk, self.bn, self.pad_m = chunk, bn, pad_m
        self.num_warps, self.num_stages = num_warps, num_stages
        self.maxs = max(1, (max_seqlen + chunk - 1) // chunk)
        self.pacc = torch.empty(batch * self.maxs * 4096, device=device, dtype=torch.float32)
        self.pml = torch.empty(batch * self.maxs * 64, device=device, dtype=torch.float32)
        self.count = torch.zeros(batch * 8, device=device, dtype=torch.int32)
        self.out = torch.empty(batch, 32, 128, device=device, dtype=torch.bfloat16)
        self.lse = torch.empty(batch, 32, device=device, dtype=torch.float32)

    def __call__(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        _gqa_decode_kernel[(self.maxs, self.B * 8)](
            q, k_cache, v_cache, kv_indptr, kv_indices,
            self.pacc, self.pml, self.count, self.out, self.lse, sm_scale,
            MAXS=self.maxs, CHUNK=self.chunk, BN=self.bn, PAD_M=self.pad_m,
            num_warps=self.num_warps, num_stages=self.num_stages)
        return self.out, self.lse
