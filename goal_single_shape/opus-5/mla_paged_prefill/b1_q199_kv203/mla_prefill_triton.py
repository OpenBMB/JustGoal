"""MLA paged prefill, causal -- clean-room Triton kernel for
mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 on NVIDIA B200 (sm_100).

Operator
--------
  q_nope   (Lq, H, 512) bf16      ckv_cache (P, 1, 512) bf16
  q_pe     (Lq, H,  64) bf16      kpe_cache (P, 1,  64) bf16
  qo_indptr (B+1,) i32            kv_indptr (B+1,) i32
  kv_indices (Lkv,) i32           sm_scale  scalar
  ->  out (Lq, H, 512) bf16 , lse (Lq, H) f32   (natural log)

  Q[i,h] = concat(q_nope[i,h], q_pe[i,h])                    (576 dims)
  K[j]   = concat(ckv[idx[j],0], kpe[idx[j],0])              (576, shared by heads)
  V[j]   = ckv[idx[j],0]                                     (512, shared by heads)
  causal, bottom-right aligned: query i attends keys j <= i + (Lkv - Lq)

Design notes
------------
* Rows are flattened to r = (query, head).  Because the latent KV is shared by
  all H heads, the H rows of one query form a tile that reuses a single K/V load
  and shares one causal mask -- K/V bandwidth per row drops by 16x.
* One block = one query (H=16 rows).  That keeps the fp32 accumulator this block
  must hold across the kv loop at 16x512 instead of 32x512 or 64x512; the wider
  accumulators pin occupancy to one block per SM and double the critical path of
  the last (longest) query.
* Blocks are issued heaviest-first.  With causal masking block q scans q+5 keys,
  so a forward-ordered grid puts the longest blocks in the ragged second wave;
  reversing it leaves only short blocks there.
* exp2 with log2(e) folded into sm_scale; lse converted back to natural log.
"""
import torch, triton, triton.language as tl

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453

# tuned on B200 / Triton 3.6 for Lq=199, Lkv=203, H=16
CFG = dict(BN=128, num_warps=8, num_stages=2)


@triton.jit
def _mla_paged_prefill_causal(Qn, Qp, CKV, KPE, KVI, Out, LSE,
                              scale_log2, qlen, kvlen, coff, nblk,
                              BN: tl.constexpr, NH: tl.constexpr,
                              DC: tl.constexpr, DP: tl.constexpr):
    # heaviest query first
    q0 = nblk - 1 - tl.program_id(0)
    rm = tl.arange(0, NH)
    rows = q0 * NH + rm
    dc = tl.arange(0, DC)
    dp = tl.arange(0, DP)

    qn = tl.load(Qn + rows[:, None] * DC + dc[None, :])
    qp = tl.load(Qp + rows[:, None] * DP + dp[None, :])

    acc = tl.zeros((NH, DC), dtype=tl.float32)
    m_i = tl.full((NH,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((NH,), dtype=tl.float32)

    hi = tl.minimum(kvlen, q0 + 1 + coff)
    for n0 in range(0, hi, BN):
        kn = n0 + tl.arange(0, BN)
        kmask = kn < kvlen
        idx = tl.load(KVI + kn, mask=kmask, other=0)
        kc = tl.load(CKV + idx[:, None] * DC + dc[None, :], mask=kmask[:, None], other=0.0)
        kp = tl.load(KPE + idx[:, None] * DP + dp[None, :], mask=kmask[:, None], other=0.0)
        s = tl.dot(qn, tl.trans(kc)) + tl.dot(qp, tl.trans(kp))
        s = s * scale_log2
        s = tl.where(kmask[None, :] & (kn[None, :] <= q0 + coff), s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(CKV.dtype.element_ty), kc, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + rows[:, None] * DC + dc[None, :], acc.to(Out.dtype.element_ty))
    tl.store(LSE + rows, m_i * 0.6931471805599453 + tl.log(l_i))


def mla_paged_prefill_causal(q_nope, q_pe, ckv_cache, kpe_cache,
                             qo_indptr, kv_indptr, kv_indices, sm_scale,
                             out=None, lse=None, **cfg):
    """Single-request (B=1) MLA paged prefill with causal masking."""
    c = dict(CFG); c.update(cfg)
    assert qo_indptr.numel() == 2 and kv_indptr.numel() == 2, \
        "this kernel is specialised for a single request (B=1)"
    qlen, NH, DC = q_nope.shape
    DP = q_pe.shape[-1]
    kvlen = kv_indices.numel()
    if out is None:
        out = torch.empty_like(q_nope)
    if lse is None:
        lse = torch.empty(qlen, NH, device=q_nope.device, dtype=torch.float32)
    _mla_paged_prefill_causal[(qlen,)](
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, out, lse,
        sm_scale * _LOG2E, qlen, kvlen, kvlen - qlen, qlen,
        BN=c["BN"], NH=NH, DC=DC, DP=DP,
        num_warps=c["num_warps"], num_stages=c["num_stages"])
    return out, lse


def mla(inp, out=None, lse=None, **cfg):
    """Adapter for the benchmark harness."""
    return mla_paged_prefill_causal(
        inp["q_nope"], inp["q_pe"], inp["ckv"], inp["kpe"],
        inp["qo_indptr"], inp["kv_indptr"], inp["kv_indices"], inp["sm_scale"],
        out=out, lse=lse, **cfg)
