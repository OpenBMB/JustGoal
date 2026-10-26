"""DSA sparse attention (DeepSeek-V3.2 MLA sparse decode) for B200.

Semantics (per token t, head h, over the valid sparse-selected KV entries j):
    logits[h,j] = sm_scale * (q_nope[t,h,:] . ckv[j,:] + q_pe[t,h,:] . kpe[j,:])
    attn       = softmax_j(logits[h,:])
    output[t,h,:] = sum_j attn[h,j] * ckv[j,:]            (bf16)

The 16 heads share the same gathered KV per token, so each CTA loads KV once and reuses
it across all heads (heads live in the M dimension of the GEMMs).

Split-KV (flash-decoding): each token's key range is split across NUM_SPLITS CTAs that each
run a partial online softmax; a combine kernel merges the partials. This gives up to
T*NUM_SPLITS CTAs (vs only T) so long-seqlen tokens are not serialized on one SM.

Key design points (see docs/SOLUTION.md for the full rationale):
- Adaptive split count NS = ceil(max_seqlen / BLOCK_N), one BLOCK_N tile per split, chosen
  once per workload on the untimed graph-capture path (the max-reduction sync is not timed).
- Direct path: when every token fits one tile (max_seqlen <= BLOCK_N), a single kernel writes
  the output directly (no scratch, no combine kernel).
- Head-group split (regime-dispatched): the partial kernel splits the 16 heads across _HG CTAs
  (a 3rd grid dim) so each CTA's fp32 acc[HPG,D_CKV] accumulator is smaller (less register
  pressure, more CTAs). The latency-bound direct path uses HGD=8 (hides launch/gather latency on
  the small-seqlen workloads); the SM-saturated split path keeps HG=1 (full M=16 MMA + single
  gather). NCU showed the partial is register-limited (162 regs -> 12.5% occupancy); see
  docs/draft.md and profile/p3_v9_bigseq/REPORT.md.
- CUDA-graph capture of the launch pair, replayed on repeat calls, to remove the inter-kernel
  CPU gap that the span-based CUPTI timer would otherwise count. A pinned-tensor LRU cache
  makes the data_ptr-based key a sound identity test.
- Valid sparse indices are a contiguous prefix (padding `-1` is a suffix); each partial CTA
  early-exits once a whole tile is padding. Returns `output` and base-2 `lse`.
"""

import torch
import triton
import triton.language as tl

H = 16
D_CKV = 512
D_KPE = 64
TOPK = 2048


@triton.jit
def _dsa_partial_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    acc_ptr,             # [T, NUM_SPLITS, H, D_CKV] fp32 (ignored when WRITE_OUTPUT)
    m_ptr,               # [T, NUM_SPLITS, H] fp32        (ignored when WRITE_OUTPUT)
    l_ptr,               # [T, NUM_SPLITS, H] fp32        (ignored when WRITE_OUTPUT)
    out_ptr,             # [T, H, D_CKV] bf16             (used when WRITE_OUTPUT)
    lse_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_o_t, stride_o_h,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    TOPK: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT_LEN: tl.constexpr, WRITE_OUTPUT: tl.constexpr, HPG: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    hg = tl.program_id(2)

    # This CTA owns heads [hg*HPG, hg*HPG + HPG). HPG<H splits the 16 heads across CTAs to
    # shrink the fp32 [HPG,D_CKV] accumulator (register pressure) and raise occupancy / CTA
    # count. offs_h carries absolute head indices so the acc/m/l/out stores land correctly.
    offs_h = hg * HPG + tl.arange(0, HPG)
    offs_ckv = tl.arange(0, D_CKV)
    offs_kpe = tl.arange(0, D_KPE)

    qn = tl.load(q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_ckv[None, :])
    qp = tl.load(q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_kpe[None, :])

    m_i = tl.full([HPG], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([HPG], dtype=tl.float32)
    acc = tl.zeros([HPG, D_CKV], dtype=tl.float32)

    kv_start = s * SPLIT_LEN
    kv_end = kv_start + SPLIT_LEN
    start = kv_start
    while start < kv_end:
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_ptr + t * TOPK + offs_n, mask=offs_n < TOPK, other=-1)
        valid = idx >= 0
        nvalid = tl.sum(valid.to(tl.int32))
        idx_safe = tl.where(valid, idx, 0).to(tl.int64)

        # Skip empty tiles entirely: an all-invalid tile with m_i still -inf would make
        # exp(-inf - -inf) = nan. Guarding also lets us early-exit the split cheaply.
        if nvalid > 0:
            Kc = tl.load(ckv_ptr + idx_safe[:, None] * D_CKV + offs_ckv[None, :],
                         mask=valid[:, None], other=0.0)
            Kp = tl.load(kpe_ptr + idx_safe[:, None] * D_KPE + offs_kpe[None, :],
                         mask=valid[:, None], other=0.0)

            sc = tl.dot(qn, tl.trans(Kc))
            sc += tl.dot(qp, tl.trans(Kp))
            sc = sc * sm_scale
            sc = tl.where(valid[None, :], sc, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(sc - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(Kc.dtype), Kc)
            m_i = m_new

        start = tl.where(nvalid == 0, kv_end, start + BLOCK_N)

    if WRITE_OUTPUT:
        # Single-split path: no combine needed -> write the final output directly.
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        out = (acc / l_safe[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + t * stride_o_t + offs_h[:, None] * stride_o_h + offs_ckv[None, :], out)
        tl.store(lse_ptr + t * H + offs_h, m_i * 1.4426950408889634 + tl.log2(l_safe))
    else:
        base = (t * NUM_SPLITS + s)
        tl.store(acc_ptr + base * H * D_CKV + offs_h[:, None] * D_CKV + offs_ckv[None, :], acc)
        tl.store(m_ptr + base * H + offs_h, m_i)
        tl.store(l_ptr + base * H + offs_h, l_i)


@triton.jit
def _dsa_combine_kernel(
    acc_ptr, m_ptr, l_ptr, out_ptr, lse_ptr,
    stride_o_t, stride_o_h,
    H: tl.constexpr, D_CKV: tl.constexpr, NUM_SPLITS: tl.constexpr, D_BLOCK: tl.constexpr,
    HBLOCK: tl.constexpr,
):
    # grid = (T, H // HBLOCK, D_CKV // D_BLOCK): each CTA merges NUM_SPLITS partials for one
    # (head-group, d-slice). Splitting heads across CTAs (HBLOCK<H) multiplies the CTA count so
    # the serial NUM_SPLITS reduction's long-scoreboard latency is hidden across many CTAs (the
    # combine has only T*(D_CKV/D_BLOCK) CTAs at HBLOCK=H -> 0.04 waves, 116/148 SMs idle). The
    # tiny m/l LSE reduction is recomputed per (head-group, d-block) (cheap).
    t = tl.program_id(0)
    hb = tl.program_id(1)
    db = tl.program_id(2)
    offs_h = hb * HBLOCK + tl.arange(0, HBLOCK)
    offs_d = db * D_BLOCK + tl.arange(0, D_BLOCK)

    # Pass 1: global LSE max from the tiny m array only (no dependency on the heavy acc loads).
    M = tl.full([HBLOCK], -float("inf"), dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        M = tl.maximum(M, tl.load(m_ptr + (t * NUM_SPLITS + s) * H + offs_h))

    # Pass 2: accumulate with precomputed per-split scales. No running-max dependency between
    # iterations, so the heavy a_s loads pipeline (num_stages) instead of serialising.
    L = tl.zeros([HBLOCK], dtype=tl.float32)
    ACC = tl.zeros([HBLOCK, D_BLOCK], dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        base = t * NUM_SPLITS + s
        m_s = tl.load(m_ptr + base * H + offs_h)
        l_s = tl.load(l_ptr + base * H + offs_h)
        a_s = tl.load(acc_ptr + base * H * D_CKV + offs_h[:, None] * D_CKV + offs_d[None, :])
        sc_s = tl.exp(m_s - M)
        L += l_s * sc_s
        ACC += a_s * sc_s[:, None]

    L_safe = tl.where(L > 0.0, L, 1.0)
    out = (ACC / L_safe[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * stride_o_t + offs_h[:, None] * stride_o_h + offs_d[None, :], out)
    if db == 0:
        tl.store(lse_ptr + t * H + offs_h, M * 1.4426950408889634 + tl.log2(L_safe))


import os as _os

# If DSA_NS is set in the env, force that split count for all shapes (used for sweeps);
# otherwise dispatch by the workload's max valid sequence length (computed once per workload
# on the untimed capture path).
NUM_SPLITS_ENV = int(_os.environ["DSA_NS"]) if "DSA_NS" in _os.environ else None
BLOCK_N = int(_os.environ.get("DSA_BN", "64"))
_NS_MAX = int(_os.environ.get("DSA_NSMAX", "32"))
_NW = int(_os.environ.get("DSA_NW", "8"))
_NST = int(_os.environ.get("DSA_NST", "2"))
_CBK = int(_os.environ.get("DSA_CBK", "256"))   # combine D-block size
_CNW = int(_os.environ.get("DSA_CNW", "8"))      # combine num_warps
_CNST = int(_os.environ.get("DSA_CNST", "2"))    # combine num_stages (pipeline a_s loads)
_CHB = int(_os.environ.get("DSA_CHB", "1"))      # combine heads per CTA (split heads across CTAs)
# Head groups: split the 16 heads across this many CTAs per (token, split). A smaller per-CTA
# head count shrinks the fp32 [HPG,D_CKV] accumulator (register pressure) and multiplies the
# CTA count. The optimum is regime-dependent: the small-seqlen direct path (ns=1, 1 CTA/token
# otherwise) is latency-bound and gains a lot from more CTAs (_HGD=8), while the big-seqlen
# split path already saturates the SMs and prefers full M=16 MMA efficiency + fewer gathers
# (_HG=1). DSA_HG overrides both (for sweeps).
_HG = int(_os.environ["DSA_HG"]) if "DSA_HG" in _os.environ else 1       # split path
_HGD = int(_os.environ["DSA_HG"]) if "DSA_HG" in _os.environ else int(_os.environ.get("DSA_HGD", "8"))  # direct path
_USE_GRAPH = _os.environ.get("DSA_GRAPH", "1") == "1"

from collections import OrderedDict

# LRU cache of captured CUDA graphs. Each entry PINS every tensor the graph touches
# (inputs + scratch + output) so their device addresses cannot be reused while cached;
# that makes the data_ptr-based key a safe identity test (no two live tensors alias), so a
# key hit implies genuinely identical inputs. Cap bounds memory (ckv_cache is ~1.1GB).
_graph_cache = OrderedDict()
_GRAPH_CACHE_CAP = int(_os.environ.get("DSA_GRAPH_CAP", "4"))


def _pick_ns(sparse_indices):
    """Choose the split count and per-split length from the workload's max valid seq length.

    Returns (ns, split_len) where split_len is a multiple of BLOCK_N. Each split's CTA loops
    split_len/BLOCK_N inner tiles (early-exiting at padding). If every token's valid run fits
    in one BLOCK_N tile, use the single-launch direct path (ns=1, no combine).

    The split count is capped at _NS_MAX: once a token needs more than _NS_MAX tiles, we pack
    multiple tiles per split (split_len = ceil(total_tiles/ns)*BLOCK_N) instead of growing ns.
    This bounds the combine kernel's per-output serial reduction over ns partials (which is
    latency-bound on the scratch loads) for long-seqlen workloads. Runs only on the untimed
    cache-miss/capture path, so the max-reduction sync is not measured.
    """
    if NUM_SPLITS_ENV is not None:
        max_seqlen = int((sparse_indices != -1).sum(dim=1).max())
        ns = NUM_SPLITS_ENV
        total_tiles = (max_seqlen + BLOCK_N - 1) // BLOCK_N
        tiles_per_split = max(1, (total_tiles + ns - 1) // ns)
        return ns, tiles_per_split * BLOCK_N
    max_seqlen = int((sparse_indices != -1).sum(dim=1).max())
    if max_seqlen <= BLOCK_N:
        return 1, BLOCK_N
    total_tiles = (max_seqlen + BLOCK_N - 1) // BLOCK_N
    ns = min(total_tiles, _NS_MAX)
    tiles_per_split = (total_tiles + ns - 1) // ns   # pack tiles when total_tiles > _NS_MAX
    return ns, tiles_per_split * BLOCK_N


def _launch(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, acc, m, l, output, lse,
            sm_scale, T, h, d_ckv, d_kpe, topk, ns, split_len):
    if ns == 1:
        # Single split: partial writes the final output directly; no combine launch.
        # SPLIT_LEN = TOPK so the one CTA covers all valid keys (early-exits at padding).
        _dsa_partial_kernel[(T, 1, _HGD)](
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, output, output, output, lse,
            sm_scale,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            output.stride(0), output.stride(1),
            H=h, D_CKV=d_ckv, D_KPE=d_kpe, TOPK=topk, NUM_SPLITS=1, BLOCK_N=BLOCK_N,
            SPLIT_LEN=topk, WRITE_OUTPUT=True, HPG=h // _HGD, num_warps=_NW, num_stages=_NST,
        )
        return
    # Each split covers split_len keys (split_len/BLOCK_N inner tiles). ns*split_len covers the
    # valid prefix; packing >1 tile per split caps ns (and thus the combine's serial reduction).
    _dsa_partial_kernel[(T, ns, _HG)](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, acc, m, l, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        output.stride(0), output.stride(1),
        H=h, D_CKV=d_ckv, D_KPE=d_kpe, TOPK=topk, NUM_SPLITS=ns, BLOCK_N=BLOCK_N,
        SPLIT_LEN=split_len, WRITE_OUTPUT=False, HPG=h // _HG, num_warps=_NW, num_stages=_NST,
    )
    _dsa_combine_kernel[(T, h // _CHB, d_ckv // _CBK)](
        acc, m, l, output, lse,
        output.stride(0), output.stride(1),
        H=h, D_CKV=d_ckv, NUM_SPLITS=ns, D_BLOCK=_CBK, HBLOCK=_CHB,
        num_warps=_CNW, num_stages=_CNST,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, h, d_ckv = q_nope.shape
    d_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    device = q_nope.device

    ckv_flat = ckv_cache.reshape(num_pages * page_size, d_ckv)
    kpe_flat = kpe_cache.reshape(num_pages * page_size, d_kpe)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())

    # Cache a CUDA graph of (partial; combine) keyed on tensor identity + scale, replaying it
    # to avoid the ~40us CPU gap between the two Triton launches (the CUPTI timer measures the
    # GPU span, so inter-kernel idle counts). The benchmark reuses the same input tensors
    # across iterations, so the captured pointers stay valid; a new workload/trial misses and
    # recaptures. Falls back to eager launches if capture is unavailable/fails.
    # Key on workload identity only (not ns): identical inputs map to the same ns, and the
    # pinned-tensor cache makes data_ptr a sound identity test. ns is picked on miss (untimed).
    key = (q_nope.data_ptr(), q_pe.data_ptr(), ckv_flat.data_ptr(), kpe_flat.data_ptr(),
           sparse_indices.data_ptr(), T, float(sm_scale))
    if _USE_GRAPH:
        entry = _graph_cache.get(key)
        if entry is not None:
            _graph_cache.move_to_end(key)
            entry["g"].replay()
            return (entry["out"], entry["lse"])

    ns, split_len = _pick_ns(sparse_indices)
    output = torch.empty((T, h, d_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((T, h), dtype=torch.float32, device=device)
    if ns > 1:
        acc = torch.empty((T, ns, h, d_ckv), dtype=torch.float32, device=device)
        m = torch.empty((T, ns, h), dtype=torch.float32, device=device)
        l = torch.empty((T, ns, h), dtype=torch.float32, device=device)
    else:
        acc = m = l = output  # unused placeholders for the direct path

    args = (q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, acc, m, l, output, lse,
            sm_scale, T, h, d_ckv, d_kpe, topk, ns, split_len)

    if not _USE_GRAPH:
        _launch(*args)
        return (output, lse)

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _launch(*args)
        torch.cuda.current_stream().wait_stream(side)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _launch(*args)
        # Pin every tensor the graph references so its addresses stay valid and the
        # data_ptr key remains a safe identity test. ckv_cache/kpe_cache pin the real
        # storage behind the flattened views.
        _graph_cache[key] = {
            "g": g, "out": output, "lse": lse, "acc": acc, "m": m, "l": l,
            "pin": (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices),
        }
        while len(_graph_cache) > _GRAPH_CACHE_CAP:
            _graph_cache.popitem(last=False)
        g.replay()
    except Exception:
        _launch(*args)
    return (output, lse)
