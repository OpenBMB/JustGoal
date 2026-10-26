"""Causal grouped-query attention with FP32 online softmax accumulation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, K, V, O, S: tl.constexpr, HQ: tl.constexpr,
               HKV: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    block = tl.program_id(0)
    head_batch = tl.program_id(1)
    batch = head_batch // HQ
    head = head_batch % HQ
    kv_head = head // (HQ // HKV)
    rows = block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, D)
    q = tl.load(Q + ((batch * S + rows[:, None]) * HQ + head) * D + dims[None, :],
                mask=rows[:, None] < S, other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    for start in range(0, tl.minimum((block + 1) * BM, S), BN):
        tokens = start + cols
        k = tl.load(K + ((batch * S + tokens[None, :]) * HKV + kv_head) * D + dims[:, None],
                    mask=tokens[None, :] < S, other=0)
        logits = tl.dot(q, k).to(tl.float32) * SCALE
        logits = tl.where((tokens[None, :] <= rows[:, None]) & (tokens[None, :] < S), logits, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        rescale = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(logits - new_maximum[:, None])
        v = tl.load(V + ((batch * S + tokens[:, None]) * HKV + kv_head) * D + dims[None, :],
                    mask=tokens[:, None] < S, other=0)
        numerator = numerator * rescale[:, None] + tl.dot(probabilities.to(v.dtype), v)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        maximum = new_maximum
    result = numerator / denominator[:, None]
    tl.store(O + ((batch * S + rows[:, None]) * HQ + head) * D + dims[None, :],
             result, mask=rows[:, None] < S)


def attn(q, k, v):
    """Return causal GQA output for contiguous B,S,H,D BF16 tensors."""
    batch, sequence, heads, dimension = q.shape
    kv_heads = k.shape[2]
    if (q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or dimension != 128 or heads % kv_heads
            or k.shape != (batch, sequence, kv_heads, dimension) or v.shape != k.shape
            or not all(t.is_cuda and t.is_contiguous() for t in (q, k, v))):
        raise ValueError('Expected contiguous CUDA BF16 causal GQA tensors with D=128')
    out = torch.empty_like(q)
    _attention[(triton.cdiv(sequence, 64), batch * heads)](
        q, k, v, out, sequence, heads, kv_heads, dimension,
        dimension ** -0.5 * 1.4426950408889634, 64, 64, num_warps=4, num_stages=2)
    return out
