"""Host-side Triton prologue kernels for the CuTe varlen MHA backward.

Two tiny passes run before the fused tcgen05 kernel:
  _pre_kernel   -- Delta = rowsum(O*dO) and zero the dQ output buffer, which
                   the main kernel accumulates into with TMA store-reduce.
  _sched_kernel -- flatten (batch, n-tile) pairs into a work list ordered by
                   descending tile cost, so the longest tiles start first.
"""

import triton
import triton.language as tl


@triton.jit
def _pre_kernel(O, DO, LSE, LSED, DQACC, T, sm_scale, stride_t: tl.constexpr,
                stride_h: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """Pack (lse*log2e, rowsum(O*dO)*sm_scale) into LSED, and zero dQ.

    LSED is laid out as (H, T, 4) fp32: slot (h, t) holds head h's scaled lse
    and scaled delta for token t, with two floats spare.  The padding makes
    element 4*(start + m0) 16B-aligned for every cu_seqlens, which is what lets
    the main kernel stage a tile with a single cp.async.bulk instead of two
    synchronous global loads.  Both scale factors are folded in here so the
    softmax needs neither.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = pid * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    m = offs_t < T
    off = offs_t[:, None] * stride_t + h * stride_h + offs_d[None, :]
    o = tl.load(O + off, mask=m[:, None], other=0.0).to(tl.float32)
    g = tl.load(DO + off, mask=m[:, None], other=0.0).to(tl.float32)
    l = tl.load(LSE + h * T + offs_t, mask=m, other=0.0)
    slot = h * T * 4 + offs_t * 4
    tl.store(LSED + slot, l * 1.4426950408889634, mask=m)
    tl.store(LSED + slot + 1, tl.sum(o * g, 1) * sm_scale, mask=m)
    tl.store(DQACC + off, tl.zeros([BT, D], tl.float32), mask=m[:, None])


@triton.jit
def _sched_kernel(CU, SCHED, nbatch, npair_max, BN: tl.constexpr, BM: tl.constexpr,
                  BLK: tl.constexpr, MOUT: tl.constexpr = 0):
    """Build the (start, slen, n_idx, cost) work list, longest tile first.

    Tile (b, j) covers n-rows [j*BN, (j+1)*BN) of document b and, because the
    mask is causal, only m-tiles i >= (j*BN)//BM contribute, so its cost is
    cost(b, j) = cdiv(slen_b, BM) - (j*BN)//BM, which is non-increasing in j.
    The rank of a tile is therefore counted analytically instead of sorted:
    within a document, #{j' : cost(b, j') > c} = clamp(cdiv((nm_b - c)*BM, BN),
    0, nb_b).  Ties break by document order, then by j.

    Runs as a single CTA; slots past the end keep the zeros the caller wrote,
    so those CTAs read slen == 0 and exit immediately.
    """
    idx = tl.arange(0, BLK)
    j_of = tl.zeros([BLK], tl.int32)
    # Index in falling-cost order.  Equal to j_of except under MOUT,
    # where j_of is reversed so the analytic rank below still sees a
    # cost that falls with the index.
    ord_of = tl.zeros([BLK], tl.int32)
    start_of = tl.zeros([BLK], tl.int32)
    slen_of = tl.zeros([BLK], tl.int32)
    # -1 marks a slot that no document claims.
    cost_of = tl.full([BLK], -1, tl.int32)
    b_of = tl.full([BLK], nbatch, tl.int32)

    off = 0
    for b in range(nbatch):
        s0 = tl.load(CU + b)
        sl = tl.load(CU + b + 1) - s0
        nb = (sl + BN - 1) // BN
        nm = (sl + BM - 1) // BM
        # Number of tiles this document contributes, and the per-tile cost.
        nt = nm if MOUT else nb
        j = idx - off
        hit = (j >= 0) & (j < nt)
        start_of = tl.where(hit, s0, start_of)
        slen_of = tl.where(hit, sl, slen_of)
        j_of = tl.where(hit, j, j_of)
        ord_of = tl.where(hit, j, ord_of)
        if MOUT:
            # m-tile j covers rows [j*BM, (j+1)*BM); causality lets it reach
            # n-tiles 0 .. (j*BM)//BN, so cost rises with j.  Reverse the
            # index so the shared ranking code still sees a falling cost.
            cost_of = tl.where(hit, ((nt - 1 - j) * BM) // BN + 1, cost_of)
            j_of = tl.where(hit, nt - 1 - j, j_of)
        else:
            cost_of = tl.where(hit, nm - (j * BN) // BM, cost_of)
        b_of = tl.where(hit, b, b_of)
        off += nt

    rank = tl.zeros([BLK], tl.int32)
    gt_own = tl.zeros([BLK], tl.int32)
    for b in range(nbatch):
        s0 = tl.load(CU + b)
        sl = tl.load(CU + b + 1) - s0
        nb = (sl + BN - 1) // BN
        nm = (sl + BM - 1) // BM
        nt = nm if MOUT else nb
        # #{cost > c} and #{cost >= c} inside document b.
        if MOUT:
            # cost(rev) = (rev*BM)//BN + 1 with rev = nt-1-j, so cost > c
            # iff rev*BM >= c*BN, i.e. rev >= cdiv(c*BN, BM).
            gt = tl.minimum(tl.maximum(nt - (cost_of * BN + BM - 1) // BM, 0), nt)
            ge = tl.minimum(
                tl.maximum(nt - ((cost_of - 1) * BN + BM - 1) // BM, 0), nt)
        else:
            gt = tl.minimum(tl.maximum(((nm - cost_of) * BM + BN - 1) // BN, 0), nb)
            ge = tl.minimum(tl.maximum(((nm - cost_of + 1) * BM + BN - 1) // BN, 0), nb)
        rank += tl.where(b < b_of, ge, gt)
        gt_own = tl.where(b == b_of, gt, gt_own)
    # Position inside this document's run of equal-cost tiles.
    rank += ord_of - gt_own

    valid = cost_of >= 0
    base = SCHED + rank * 4
    tl.store(base + 0, start_of, mask=valid)
    tl.store(base + 1, slen_of, mask=valid)
    tl.store(base + 2, j_of, mask=valid)
    tl.store(base + 3, cost_of, mask=valid)
