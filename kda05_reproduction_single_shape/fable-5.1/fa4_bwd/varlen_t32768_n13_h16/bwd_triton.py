"""Triton (Blackwell) FA2-style varlen causal attention backward. Phase 1 baseline.

Kernels:
  _bwd_preprocess : delta[h,t] = sum_d dout*out  (fp32) and zero dq_accum
  _bwd_kernel     : kv-outer loop; per (kv block, head) program; dK/dV accumulated locally,
                    dQ accumulated with fp32 atomics into dq_accum
  _bwd_convert_dq : dq = (dq_accum * sm_scale).to(bf16)
"""

import math

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _bwd_preprocess(O, DO, Delta, DQacc, T, H: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
                    cu_seqlens, num_seqs, table, max_tiles, BLOCK_N: tl.constexpr, MAX_NBLK: tl.constexpr, NSEQ_PAD: tl.constexpr,
                    DQ_T: tl.constexpr = False, T_pad=0, T_grid=0,
                    LSE=None, XQ=None, XDO=None, inv_scale=1.0, XBLK: tl.constexpr = False,
                    SEQ_ACC=None, COUNTER=None, SEQ_MAJOR: tl.constexpr = 0, ALL_HEADS: tl.constexpr = False,
                    ZERO_ACC: tl.constexpr = True, BLK_CNT=None, NBLK=0, HFAST: tl.constexpr = False, HG: tl.constexpr = 1):
    """delta[h, t] = sum_d dout*out (fp32); zero dq_accum rows; the extra last program builds the LPT tile table
    (and, for the persistent kernel, the per-sequence padded accumulator starts and the zeroed work-item counter)."""
    if HFAST:
        h = tl.program_id(0)                # heads fastest: the 16 programs of a token block run together (whole 4 KB rows)
        pid_t = tl.program_id(1)
    else:
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
    n_tblk = tl.cdiv(T_grid, BLOCK_T)      # the extra (last) program builds the tile table
    if pid_t == n_tblk:
        if SEQ_MAJOR == 2:
            # full (tile, head) item table, sequence-major (length desc), head-major within a sequence, kv order within a head:
            # program h writes the entries of head h; entry = head << 24 | seq << 16 | kv_block
            offs_s = tl.arange(0, NSEQ_PAD)
            cu0 = tl.load(cu_seqlens + offs_s, mask=offs_s < num_seqs, other=0)
            cu1 = tl.load(cu_seqlens + offs_s + 1, mask=offs_s < num_seqs, other=0)
            lens = tl.where(offs_s < num_seqs, cu1 - cu0, 0)
            nblk = (lens + BLOCK_N - 1) // BLOCK_N
            nb_r = nblk[None, :]
            nb_c = nblk[:, None]
            before = (nb_r > nb_c) | ((nb_r == nb_c) & (offs_s[None, :] < offs_s[:, None]))
            # head groups of HG: items of a sequence are ordered (group, tile, head in group) so HG heads of a tile run together
            hg = h // HG
            hi = h % HG
            base_s = tl.sum(tl.where(before, nb_r, 0), axis=1) * H + hg * (HG * nblk)
            for i in range(0, MAX_NBLK):
                m = nblk > i
                tl.store(table + base_s + i * HG + hi, h * 16777216 + offs_s * 65536 + i, mask=m)
            total = tl.sum(nblk, axis=0) * H
            offs_tt = tl.arange(0, 1024)
            for t0 in range(0, max_tiles * H, 1024):
                idx = t0 + offs_tt
                tl.store(table + idx, -1, mask=(idx >= total) & (idx < max_tiles * H) & (idx % H == h))
            if BLK_CNT is not None:
                for t0 in range(0, NBLK, 1024):
                    idx = t0 + offs_tt
                    tl.store(BLK_CNT + h * NBLK + idx, 0, mask=idx < NBLK)
            if h == 0:
                if SEQ_ACC is not None:
                    pads = ((lens + 63) // 64) * 64
                    starts = tl.cumsum(pads, axis=0) - pads
                    tl.store(SEQ_ACC + offs_s, starts, mask=offs_s < num_seqs)
                    tl.store(COUNTER, 0)
        elif h == 0:
            offs_s = tl.arange(0, NSEQ_PAD)
            cu0 = tl.load(cu_seqlens + offs_s, mask=offs_s < num_seqs, other=0)
            cu1 = tl.load(cu_seqlens + offs_s + 1, mask=offs_s < num_seqs, other=0)
            lens = tl.where(offs_s < num_seqs, cu1 - cu0, 0)
            nblk = (lens + BLOCK_N - 1) // BLOCK_N
            base = 0
            if SEQ_MAJOR == 1:
                # sequences ordered by length (desc, ties by index); tiles of a sequence in kv order
                nb_r = nblk[None, :]
                nb_c = nblk[:, None]
                before = (nb_r > nb_c) | ((nb_r == nb_c) & (offs_s[None, :] < offs_s[:, None]))   # [s, s']: s' precedes s
                base_s = tl.sum(tl.where(before, nb_r, 0), axis=1)
                for i in range(0, MAX_NBLK):
                    m = nblk > i
                    tl.store(table + base_s + i, offs_s * 65536 + i, mask=m)
                base = tl.sum(nblk, axis=0)
            else:
                for i in range(0, MAX_NBLK):
                    w = MAX_NBLK - i
                    m = nblk >= w
                    pos = base + tl.cumsum(m.to(tl.int32), axis=0) - 1
                    tl.store(table + pos, offs_s * 65536 + (nblk - w), mask=m)
                    base += tl.sum(m.to(tl.int32), axis=0)
            offs_tt = tl.arange(0, 1024)
            for t0 in range(0, max_tiles, 1024):
                idx = t0 + offs_tt
                tl.store(table + idx, -1, mask=(idx >= base) & (idx < max_tiles))
            if SEQ_ACC is not None:
                pads = ((lens + 63) // 64) * 64
                starts = tl.cumsum(pads, axis=0) - pads
                tl.store(SEQ_ACC + offs_s, starts, mask=offs_s < num_seqs)
                tl.store(COUNTER, 0)
    elif ALL_HEADS:
        # program (pid_t, h == 0): delta / x rows for BLOCK_T tokens of all heads (contiguous H*D rows); other h: zeroing only
        if h == 0:
            offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
            offs_hd = tl.arange(0, H * D)
            mask = offs_t < T
            ptr = offs_t[:, None] * (H * D) + offs_hd[None, :]
            o = tl.load(O + ptr, mask=mask[:, None], other=0.0)
            do = tl.load(DO + ptr, mask=mask[:, None], other=0.0)
            prod = tl.reshape(o.to(tl.float32) * do.to(tl.float32), [BLOCK_T, H, D])
            delta = tl.sum(prod, axis=2)                                   # [BLOCK_T, H]
            offs_h = tl.arange(0, H)
            tl.store(Delta + offs_h[None, :] * T + offs_t[:, None], delta, mask=mask[:, None])
            if XBLK:
                offs8 = tl.arange(0, 8)
                lse = tl.load(LSE + offs_h[None, :] * T + offs_t[:, None], mask=mask[:, None], other=0.0)
                vq = -lse * inv_scale
                hq = vq.to(tl.bfloat16)
                lq = (vq - hq.to(tl.float32)).to(tl.bfloat16)
                k8 = offs8[None, None, :]
                rowq = tl.where(k8 == 0, hq[:, :, None], tl.where(k8 == 1, lq[:, :, None], tl.zeros([1, 1, 1], dtype=tl.bfloat16)))
                vd = -delta
                hd = vd.to(tl.bfloat16)
                ld = (vd - hd.to(tl.float32)).to(tl.bfloat16)
                rowd = tl.where(k8 == 0, hd[:, :, None], tl.where(k8 == 1, ld[:, :, None], tl.zeros([1, 1, 1], dtype=tl.bfloat16)))
                xptr = (offs_h[None, :, None] * T + offs_t[:, None, None]) * 8 + k8
                tl.store(XQ + xptr, rowq, mask=mask[:, None, None])
                tl.store(XDO + xptr, rowd, mask=mask[:, None, None])
        if DQ_T and ZERO_ACC:
            flat = (h * n_tblk + pid_t) * (BLOCK_T * D) + tl.arange(0, BLOCK_T * D)
            tl.store(DQacc + flat, tl.zeros([BLOCK_T * D], dtype=tl.float32).to(DQacc.dtype.element_ty), mask=flat < H * D * T_pad)
    else:
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, D)
        mask = offs_t < T
        ptr = (offs_t[:, None] * H + h) * D + offs_d[None, :]
        o = tl.load(O + ptr, mask=mask[:, None], other=0.0)
        do = tl.load(DO + ptr, mask=mask[:, None], other=0.0)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        tl.store(Delta + h * T + offs_t, delta, mask=mask)
        if XBLK:
            # 9th-k-step operand rows (bf16 x 8 per token): [hi, lo, 0...] of -lse/scale (XQ) and of -delta (XDO)
            offs8 = tl.arange(0, 8)
            lse = tl.load(LSE + h * T + offs_t, mask=mask, other=0.0)
            vq = -lse * inv_scale
            hq = vq.to(tl.bfloat16)
            lq = (vq - hq.to(tl.float32)).to(tl.bfloat16)
            rowq = tl.where(offs8[None, :] == 0, hq[:, None], tl.where(offs8[None, :] == 1, lq[:, None], tl.zeros([1, 1], dtype=tl.bfloat16)))
            vd = -delta
            hd = vd.to(tl.bfloat16)
            ld = (vd - hd.to(tl.float32)).to(tl.bfloat16)
            rowd = tl.where(offs8[None, :] == 0, hd[:, None], tl.where(offs8[None, :] == 1, ld[:, None], tl.zeros([1, 1], dtype=tl.bfloat16)))
            xptr = (h * T + offs_t)[:, None] * 8 + offs8[None, :]
            tl.store(XQ + xptr, rowq, mask=mask[:, None])
            tl.store(XDO + xptr, rowd, mask=mask[:, None])
        if DQ_T and ZERO_ACC:
            # accumulator stored as (H, D, T_pad): zero it as a flat array, one contiguous BLOCK_T*D chunk per program
            flat = (h * n_tblk + pid_t) * (BLOCK_T * D) + tl.arange(0, BLOCK_T * D)
            tl.store(DQacc + flat, tl.zeros([BLOCK_T * D], dtype=tl.float32).to(DQacc.dtype.element_ty), mask=flat < H * D * T_pad)
        else:
            tl.store(DQacc + ptr, tl.zeros([BLOCK_T, D], dtype=tl.float32).to(DQacc.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _bwd_convert_dq_t(DQacc, DQ, T, T_pad, cu_seqlens, num_seqs, sm_scale, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
                      NSEQ_PAD: tl.constexpr, MODE: tl.constexpr = 0, BD: tl.constexpr = 128):
    """dq[t, h, :] = sm_scale * dq_accum_t[h, :, p] where the accumulator is (H, D, T_pad) fp16 and sequence s occupies
    padded rows [pad_start(s), pad_start(s) + roundup(len_s, 64)). Program = one BT-row block (BT | 64) of one head."""
    pid_b = tl.program_id(0)
    h = tl.program_id(1)
    p0 = pid_b * BT
    # locate the sequence containing padded row p0 (vectorized over the padded sequence list)
    offs_s = tl.arange(0, NSEQ_PAD)
    smask = offs_s < num_seqs
    cu0 = tl.load(cu_seqlens + offs_s, mask=smask, other=0)
    cu1 = tl.load(cu_seqlens + offs_s + 1, mask=smask, other=0)
    lens = tl.where(smask, cu1 - cu0, 0)
    plen = (lens + 63) // 64 * 64
    pstart = tl.cumsum(plen, axis=0) - plen
    hit = smask & (p0 >= pstart) & (p0 < pstart + plen)
    any_hit = tl.sum(hit.to(tl.int32), axis=0) > 0
    t_start = tl.sum(tl.where(hit, cu0 + (p0 - pstart), 0), axis=0)
    len_found = tl.sum(tl.where(hit, lens - (p0 - pstart), 0), axis=0)
    offs_i = tl.arange(0, BT)
    offs_d = tl.program_id(2) * BD + tl.arange(0, BD)
    valid = (offs_i < len_found) & any_hit
    src = DQacc + (h * D + offs_d[:, None]) * T_pad + (p0 + offs_i)[None, :]
    dst = DQ + ((t_start + offs_i)[:, None] * H + h) * D + offs_d[None, :]
    if MODE == 1:      # experiment: loads only (reduce to one value per column)
        x = tl.load(src, mask=valid[None, :], other=0.0).to(tl.float32)
        y = tl.sum(x, axis=0)
        tl.store(DQ + h * T + offs_i, y.to(tl.bfloat16), mask=valid & (y == 12345.0))
    elif MODE == 2:    # experiment: stores only
        y = tl.zeros([BT, BD], dtype=tl.float32) + sm_scale
        tl.store(dst, y.to(tl.bfloat16), mask=valid[:, None])
    else:
        x = tl.load(src, mask=valid[None, :], other=0.0).to(tl.float32)
        y = tl.trans(x) * sm_scale
        tl.store(dst, y.to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _bwd_convert_dq(DQacc, DQ, n_elem, sm_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(DQacc + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(DQ + offs, (x * sm_scale).to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_kernel(
    Q, K, V, DO, LSE, Delta, DQacc, DK, DV, cu_seqlens,
    num_seqs, T, sm_scale,
    MAX_NBLK: tl.constexpr,
    NSEQ_PAD: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)

    # ---- map pid -> (seq, kv block j), ordered by j ascending across sequences (LPT-like) ----
    offs_s = tl.arange(0, NSEQ_PAD)
    cu0 = tl.load(cu_seqlens + offs_s, mask=offs_s < num_seqs, other=0)
    cu1 = tl.load(cu_seqlens + offs_s + 1, mask=offs_s < num_seqs, other=0)
    lens = tl.where(offs_s < num_seqs, cu1 - cu0, 0)
    nblk = (lens + BLOCK_N - 1) // BLOCK_N
    total = tl.sum(nblk, axis=0)
    if pid >= total:
        return
    # find j such that cum(j) <= pid < cum(j+1), cum(j) = sum_{j'<j} count(j'), count(j') = #{s: nblk_s > j'}
    cum = 0
    base = 0
    j = -1
    for jj in range(0, MAX_NBLK):
        cnt = tl.sum((nblk > jj).to(tl.int32), axis=0)
        hit = (j < 0) & (pid < cum + cnt)
        j = tl.where(hit, jj, j)
        base = tl.where(hit, cum, base)
        cum = cum + cnt
    r = pid - base  # rank within group j
    flags = (nblk > j).to(tl.int32)
    pref = tl.cumsum(flags, axis=0)  # inclusive
    s = tl.sum(tl.where(flags * (pref == r + 1) == 1, offs_s, 0), axis=0)
    seq_start = tl.sum(tl.where(offs_s == s, cu0, 0), axis=0)
    seqlen = tl.sum(tl.where(offs_s == s, lens, 0), axis=0)

    kv0 = j * BLOCK_N
    offs_n = kv0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < seqlen
    tok_n = seq_start + offs_n
    kv_ptr = (tok_n[:, None] * H + h) * D + offs_d[None, :]
    k = tl.load(K + kv_ptr, mask=mask_n[:, None], other=0.0)
    v = tl.load(V + kv_ptr, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E
    m_start = kv0 // BLOCK_M
    m_end = (seqlen + BLOCK_M - 1) // BLOCK_M
    for m in range(m_start, m_end):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen
        tok_m = seq_start + offs_m
        q_ptr = (tok_m[:, None] * H + h) * D + offs_d[None, :]
        q = tl.load(Q + q_ptr, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + q_ptr, mask=mask_m[:, None], other=0.0)
        lse = tl.load(LSE + h * T + tok_m, mask=mask_m, other=0.0)
        delta = tl.load(Delta + h * T + tok_m, mask=mask_m, other=0.0)

        qkT = tl.dot(k, tl.trans(q))  # [N, M]
        pT = tl.math.exp2(qkT * qk_scale - (lse * LOG2E)[None, :])
        causal = (offs_n[:, None] <= offs_m[None, :]) & mask_m[None, :] & mask_n[:, None]
        pT = tl.where(causal, pT, 0.0)
        dpT = tl.dot(v, tl.trans(do))  # [N, M]
        dsT = pT * (dpT - delta[None, :])
        dv += tl.dot(pT.to(tl.bfloat16), do)
        dk += tl.dot(dsT.to(tl.bfloat16), q)
        dq = tl.dot(tl.trans(dsT.to(tl.bfloat16)), k)  # [M, D]
        tl.atomic_add(DQacc + q_ptr, dq, mask=mask_m[:, None], sem="relaxed")

    tl.store(DK + kv_ptr, (dk * sm_scale).to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(DV + kv_ptr, dv.to(tl.bfloat16), mask=mask_n[:, None])


@triton.jit
def _build_tile_table(cu_seqlens, num_seqs, table, max_tiles, BLOCK_N: tl.constexpr, MAX_NBLK: tl.constexpr, NSEQ_PAD: tl.constexpr):
    """One program: emit (seq, kv_block) tiles ordered by causal work (number of q iterations) descending.
    table[t] = seq << 16 | kv_block, or -1 beyond the last tile."""
    offs_s = tl.arange(0, NSEQ_PAD)
    cu0 = tl.load(cu_seqlens + offs_s, mask=offs_s < num_seqs, other=0)
    cu1 = tl.load(cu_seqlens + offs_s + 1, mask=offs_s < num_seqs, other=0)
    lens = tl.where(offs_s < num_seqs, cu1 - cu0, 0)
    nblk = (lens + BLOCK_N - 1) // BLOCK_N
    base = 0
    for i in range(0, MAX_NBLK):
        w = MAX_NBLK - i
        m = nblk >= w
        pos = base + tl.cumsum(m.to(tl.int32), axis=0) - 1
        tl.store(table + pos, offs_s * 65536 + (nblk - w), mask=m)
        base += tl.sum(m.to(tl.int32), axis=0)
    offs_t = tl.arange(0, 1024)
    for t0 in range(0, max_tiles, 1024):
        idx = t0 + offs_t
        tl.store(table + idx, -1, mask=(idx >= base) & (idx < max_tiles))


_DQ_ACC_CACHE = {}


_XBLK_CACHE = {}


def _get_xblk(T, H, device):
    """Per-token extra-operand rows (H, T, 8) bf16 for lse and delta, and the constant 4 KB ones/zero block."""
    key = (T, H, str(device))
    r = _XBLK_CACHE.get(key)
    if r is None:
        xq = torch.empty(H, T, 8, dtype=torch.bfloat16, device=device)
        xdo = torch.empty(H, T, 8, dtype=torch.bfloat16, device=device)
        cb = torch.zeros(1536, dtype=torch.bfloat16, device=device)
        idx = torch.arange(1024, device=device)
        cb[:1024] = (idx % 8 < 2).to(torch.bfloat16)        # ones block: columns 0,1 of every 8-element core-matrix row
        r = (xq, xdo, cb)
        _XBLK_CACHE[key] = r
    return r


def _get_dq_accum(T, H, D, device, dtype=torch.float32, transposed=False, num_seqs=0, tiled=False):
    """(T, H, D) accumulator, or transposed (H, D, T_cap) where every sequence starts at a 64-aligned offset
    (T_cap = T + 64*num_seqs rounded to 8: TMA needs 16-B aligned row strides and box starts), or tiled
    (H, T_cap/64, D, 64): one contiguous (D, 64 q) block per 64-token block (T_cap a multiple of 64)."""
    T_cap = (T + 64 * num_seqs + 63) // 64 * 64 if tiled else (T + 64 * num_seqs + 7) // 8 * 8
    key = (T, H, D, str(device), dtype, transposed, num_seqs, tiled)
    buf = _DQ_ACC_CACHE.get(key)
    if buf is None:
        shape = (H, T_cap // 64, D, 64) if tiled else ((H, D, T_cap) if transposed else (T, H, D))
        buf = torch.empty(shape, dtype=dtype, device=device)
        _DQ_ACC_CACHE[key] = buf
    return buf


@triton.jit
def _bwd_convert_dq_tiled(DQacc, DQ, T, cu_seqlens, num_seqs, sm_scale, H: tl.constexpr, D: tl.constexpr, NBLK,
                          NSEQ_PAD: tl.constexpr, EVICT: tl.constexpr = 0, HFAST: tl.constexpr = False):
    """dq[t, h, :] = sm_scale * acc[h, blk, :, i] for the tiled (H, NBLK, D, 64) fp16 accumulator; sequence s occupies
    padded 64-row blocks starting at pad_start(s). Program = (64-row block, head): one contiguous 16 KB read."""
    if HFAST:
        h = tl.program_id(0)
        blk = tl.program_id(1)
    else:
        blk = tl.program_id(0)
        h = tl.program_id(1)
    p0 = blk * 64
    offs_s = tl.arange(0, NSEQ_PAD)
    smask = offs_s < num_seqs
    cu0 = tl.load(cu_seqlens + offs_s, mask=smask, other=0)
    cu1 = tl.load(cu_seqlens + offs_s + 1, mask=smask, other=0)
    lens = tl.where(smask, cu1 - cu0, 0)
    plen = (lens + 63) // 64 * 64
    pstart = tl.cumsum(plen, axis=0) - plen
    hit = smask & (p0 >= pstart) & (p0 < pstart + plen)
    any_hit = tl.sum(hit.to(tl.int32), axis=0) > 0
    t_start = tl.sum(tl.where(hit, cu0 + (p0 - pstart), 0), axis=0)
    len_found = tl.sum(tl.where(hit, lens - (p0 - pstart), 0), axis=0)
    offs_i = tl.arange(0, 64)
    offs_d = tl.arange(0, D)
    valid = (offs_i < len_found) & any_hit
    src = DQacc + (h * NBLK + blk) * (D * 64) + offs_d[:, None] * 64 + offs_i[None, :]
    dst = DQ + ((t_start + offs_i)[:, None] * H + h) * D + offs_d[None, :]
    if EVICT == 1:
        x = tl.load(src, mask=valid[None, :], other=0.0, eviction_policy="evict_first").to(tl.float32)
        y = tl.trans(x) * sm_scale
        tl.store(dst, y.to(tl.bfloat16), mask=valid[:, None], eviction_policy="evict_first")
    elif EVICT == 2:
        x = tl.load(src, mask=valid[None, :], other=0.0, eviction_policy="evict_first").to(tl.float32)
        y = tl.trans(x) * sm_scale
        tl.store(dst, y.to(tl.bfloat16), mask=valid[:, None])
    else:
        x = tl.load(src, mask=valid[None, :], other=0.0).to(tl.float32)
        y = tl.trans(x) * sm_scale
        tl.store(dst, y.to(tl.bfloat16), mask=valid[:, None])


def attention_bwd_triton(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                         BLOCK_M=64, BLOCK_N=128, num_warps=8, num_stages=2):
    T, H, D = q.shape
    num_seqs = cu_seqlens.numel() - 1
    device = q.device
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dq_accum = _get_dq_accum(T, H, D, device)
    delta = torch.empty(H, T, dtype=torch.float32, device=device)

    max_nblk = triton.cdiv(int(max_seqlen), BLOCK_N)
    nseq_pad = max(16, triton.next_power_of_2(num_seqs + 1))
    BLOCK_T = 64
    dummy_table = torch.empty(16, dtype=torch.int32, device=device)
    _bwd_preprocess[(triton.cdiv(T, BLOCK_T) + 1, H)](out, dout, delta, dq_accum, T, H=H, D=D, BLOCK_T=BLOCK_T,
                                                     cu_seqlens=cu_seqlens, num_seqs=num_seqs, table=dummy_table, max_tiles=16,
                                                     BLOCK_N=128, MAX_NBLK=max_nblk, NSEQ_PAD=nseq_pad, num_warps=4, T_grid=T)
    grid0 = triton.cdiv(T, BLOCK_N) + num_seqs  # upper bound on sum of per-seq block counts
    _bwd_kernel[(grid0, H)](
        q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens,
        num_seqs, T, float(sm_scale),
        MAX_NBLK=max_nblk, NSEQ_PAD=nseq_pad, H=H, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    n_elem = T * H * D
    BLOCK = 4096
    _bwd_convert_dq[(triton.cdiv(n_elem, BLOCK),)](dq_accum, dq, n_elem, float(sm_scale), BLOCK=BLOCK, num_warps=8)
    return dq, dk, dv
