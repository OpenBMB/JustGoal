"""Causal FlashAttention backward for Blackwell (B200), written from scratch in Gluon (Triton's
explicit-control DSL): TMA loads, tcgen05 MMAs with TMEM accumulators and TMEM-resident A operands,
warp specialization with two ping-pong softmax warpgroups, PERSISTENT CTAs (one per SM) that walk a
static snake-ordered list of work items so the next item's loads overlap the current item's tail.

Two-pass design (no atomics):
  pass A (dK/dV): work item = (kv block of BN=128 rows, sequence, kv head). K lives in tensor memory
      (MMA A operand), V in shared memory (double-buffered across items); dK/dV accumulate in TMEM.
      For each causal q tile (BM=64 rows) of every q head of the group:
        S^T = K Q^T (TMEM A)       -> softmax warps: P^T = exp(S^T*scale - lse) stored in place (bf16)
        dP^T = V dO^T (smem A)     -> softmax warps: dS^T = P^T (dP^T - delta) stored in place (bf16)
        dV += P^T dO (TMEM A),  dK += dS^T Q (TMEM A)
  pass B (dQ): work item = (q tile of BMQ=128 rows, sequence, q head). Q and dO live in tensor memory;
      dQ accumulates in TMEM. For each causal kv tile (BNQ=64 rows):
        S = Q K^T, dP = dO V^T (TMEM A) -> softmax warps: dS = P (dP - delta) stored in place (bf16)
        dQ += dS K (TMEM A).  dQ is written directly as bf16.
A small preprocess kernel computes delta = rowsum(dO * O) and lse*log2(e) in a [H, T] layout.
"""
import os
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tma, mbarrier, tcgen05_mma, tcgen05_commit,
    get_tmem_reg_layout, fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

LOG2E = gl.constexpr(1.4426950408889634)
SM_WARPS = gl.constexpr(4)  # warps per softmax partition
SMP = gl.constexpr(int(os.environ.get("GL_SMP", "2")))  # softmax partitions per CTA
MMAONLY = gl.constexpr(int(os.environ.get("GL_MMAONLY", "0")))  # timing probe: MMA warp never waits for the softmax
ORDER = gl.constexpr(int(os.environ.get("GL_ORDER", "0")))
LSEFAKE = gl.constexpr(int(os.environ.get("GL_LSEFAKE", "0")))  # timing probe: loader skips the lse/delta global loads
LSETMA = gl.constexpr(int(os.environ.get("GL_LSETMA", "0")))
LSEPF = gl.constexpr(int(os.environ.get("GL_LSEPF", "1")))
VB = gl.constexpr(int(os.environ.get("GL_VB", "2")))
LSEPART = gl.constexpr(int(os.environ.get("GL_LSEPART", "0")))  # pass A: softmax partitions load lse/delta themselves (no smem ring)  # pass A: V staging buffers (1 frees 32KB smem for a 5th Q/dO stage)  # loader prefetches the next tile's lse/delta before waiting on the ring  # pass A: lse2/delta per tile via 1-D TMA (async) instead of warp loads  # 1: (batch, head)-major item order (concurrent CTAs share operand tiles in L2)


@triton.jit
def _preprocess(O, DO, LSE, Delta, LSE2, stride_ot, stride_oh, T, N, VARLEN: tl.constexpr,
                H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    # Delta[h, t] = sum_d O[t,h,d] * DO[t,h,d];  LSE2[h, t] = lse * log2(e)  (both in [H, T] layout)
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = pid * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    mask = offs_t < T
    ptr = offs_t[:, None] * stride_ot + h * stride_oh + offs_d[None, :]
    o = tl.load(O + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + h * T + offs_t, tl.sum(o * do, axis=1), mask=mask)
    if VARLEN:
        lse_ptr = LSE + h * T + offs_t
    else:
        b = offs_t // N
        lse_ptr = LSE + b * (H * N) + h * N + (offs_t - b * N)
    lse = tl.load(lse_ptr, mask=mask, other=float("inf"))
    tl.store(LSE2 + h * T + offs_t, lse * 1.4426950408889634, mask=mask)


@gluon.jit
def _clock():
    # 64-bit SM clock (single-element tensor), 1-warp partitions
    one = gl.full([1], 0, gl.int32, layout=gl.BlockedLayout([1], [32], [1], [0]))
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [one], dtype=gl.int64, is_pure=False, pack=1)


@gluon.jit
def _clock4():
    # 64-bit SM clock (single-element tensor), 4-warp partitions
    one = gl.full([1], 0, gl.int32, layout=gl.BlockedLayout([1], [32], [4], [0]))
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [one], dtype=gl.int64, is_pure=False, pack=1)


@gluon.jit
def _exp2_poly(x):
    # 2^x on the FMA/ALU pipes only (no MUFU / XU): round-to-nearest integer via the 1.5*2^23 trick,
    # degree-5 polynomial for 2^f on f in [-0.5, 0.5], exponent assembled with an integer add. rel err ~2.6e-7
    x = gl.maximum(x, -126.0)
    t = x + 12582912.0
    nf = t - 12582912.0
    ni = t.to(gl.int32, bitcast=True) - 1262485504   # bits of 12582912.0f
    f = x - nf
    pf = 1.0 + f * (0.6931469381 + f * (0.2402231824 + f * (0.0555083375 + f * (0.0096667497 + f * 0.0013216959))))
    return (pf.to(gl.int32, bitcast=True) + (ni << 23)).to(gl.float32, bitcast=True)


@gluon.jit
def _consume_store(SINK, x):
    # make a later mbarrier.arrive (release) depend on the shared-memory loads that produced x: fold every register
    # into a per-row integer and store it to a dummy global sink before the arrive. The release orders the store, and
    # the store depends on all loaded data, so the staging buffer cannot be re-filled (TMA) while loads are in flight.
    xi = x.to(gl.int16, bitcast=True).to(gl.int32)
    v = gl.sum(xi, axis=1)
    rows = gl.arange(0, x.shape[0], layout=gl.SliceLayout(1, x.type.layout))
    gl.store(SINK + rows, v)


@gluon.jit
def _consume(x):
    # pass every register through a non-pure inline-asm move: forces the shared-memory loads that produced them
    # to complete before any later side effect (e.g. an mbarrier arrive that frees the staging buffer)
    xi = x.to(gl.int16, bitcast=True)
    yi = gl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [xi], dtype=gl.int16, is_pure=False, pack=1)
    return yi.to(gl.bfloat16, bitcast=True)


@gluon.jit
def _tmem_store_fence():
    # make preceding tcgen05.st visible before signalling an mbarrier (4-warp partitions)
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    d = gl.full([1], 0, gl.int32, layout=lay)
    gl.inline_asm_elementwise("tcgen05.wait::st.sync.aligned;\ntcgen05.fence::before_thread_sync;\nmov.u32 $0, 0;", "=r,r", [d], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _tmem_after_sync():
    # order tcgen05 ops after an mbarrier wait (1-warp MMA partition)
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    d = gl.full([1], 0, gl.int32, layout=lay)
    gl.inline_asm_elementwise("tcgen05.fence::after_thread_sync;\nmov.u32 $0, 0;", "=r,r", [d], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _tmem_after_sync4():
    # order tcgen05 loads after an mbarrier wait (4-warp softmax partitions)
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    d = gl.full([1], 0, gl.int32, layout=lay)
    gl.inline_asm_elementwise("tcgen05.fence::after_thread_sync;\nmov.u32 $0, 0;", "=r,r", [d], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _item_index(r, c, G, W):
    # snake order: even rounds ascending, odd rounds descending
    if r % 2 == 0:
        w = r * G + c
    else:
        w = r * G + (G - 1 - c)
    return w


@gluon.jit
def _b_item(w, c, G, W, nseq, num_q_tiles, H: gl.constexpr):
    # item w -> (qt descending = heaviest first, b, h)
    if ORDER == 1:
        qt = num_q_tiles - 1 - (w % num_q_tiles)
        rem = w // num_q_tiles
    else:
        qt = num_q_tiles - 1 - w // (nseq * H)
        rem = w - (num_q_tiles - 1 - qt) * (nseq * H)
    b = rem // H
    h = rem - b * H
    return qt, b, h


@gluon.jit
def _a_has_next(r, c, G, W, num_rounds, nseq, N, CU, VARLEN: gl.constexpr, HKV: gl.constexpr, BN: gl.constexpr):
    if VARLEN:
        found = 0
        rr = r + 1
        while (found == 0) and (rr < num_rounds):
            w = _item_index(rr, c, G, W)
            if w < W:
                if ORDER == 1:
                    nkvb = (N + BN - 1) // BN
                    kvb = w % nkvb
                    rem = w // nkvb
                else:
                    kvb = w // (nseq * HKV)
                    rem = w - kvb * (nseq * HKV)
                b = rem // HKV
                seq_start, n = _seq_info(CU, b, N, VARLEN)
                if kvb * BN < n:
                    found = 1
            rr += 1
        return found
    else:
        if r + 1 < num_rounds:
            w = _item_index(r + 1, c, G, W)
            return (w < W).to(gl.int32)
        else:
            return 0


@gluon.jit
def _b_has_next(r, c, G, W, num_rounds, nseq, num_q_tiles, N, CU, VARLEN: gl.constexpr, H: gl.constexpr, BMQ: gl.constexpr):
    if VARLEN:
        found = 0
        rr = r + 1
        while (found == 0) and (rr < num_rounds):
            w = _item_index(rr, c, G, W)
            if w < W:
                qt, b, h = _b_item(w, c, G, W, nseq, num_q_tiles, H)
                seq_start, n = _seq_info(CU, b, N, VARLEN)
                if qt * BMQ < n:
                    found = 1
            rr += 1
        return found
    else:
        if r + 1 < num_rounds:
            w = _item_index(r + 1, c, G, W)
            return (w < W).to(gl.int32)
        else:
            return 0


@gluon.jit
def _seq_info(CU, b, N, VARLEN: gl.constexpr):
    if VARLEN:
        seq_start = gl.load(CU + b)
        n = gl.load(CU + b + 1) - seq_start
    else:
        seq_start = b * N
        n = N
    return seq_start, n


# =============================================================================== pass A: dK / dV
@gluon.jit
def _a_load(q_desc, do_desc, k_desc, v_desc, lse_desc, del_desc, LSE2, Delta, CU, q_smem3, do_smem3, k_smem3, v_smem3, lse_smem, del_smem,
            k_ready_s, k_copied, kv_ready, kv_free, q_ready, q_empty,
            c, G, W, num_rounds, nseq, N, T,
            VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
            BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr):
    vec_layout: gl.constexpr = gl.BlockedLayout([2], [32], [1], [0])
    offs_v = gl.arange(0, BM, layout=vec_layout)
    t = 0
    jj = 0
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            if ORDER == 1:
                nkvb = (N + BN - 1) // BN
                kvb = w % nkvb
                rem = w // nkvb
            else:
                kvb = w // (nseq * HKV)
                rem = w - kvb * (nseq * HKV)
            b = rem // HKV
            hkv = rem - b * HKV
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            kv_start = kvb * BN
            if kv_start < n:
                p = jj % VB
                m = jj // VB
                q_lo = (kv_start // BM) * BM
                tiles_per_head = (n - q_lo + BM - 1) // BM
                n_tiles = tiles_per_head * GROUP
                if jj >= 1:
                    mbarrier.wait(k_copied, (jj - 1) & 1)
                mbarrier.expect(k_ready_s, BN * D * 2)
                tma.async_copy_global_to_shared(k_desc, [seq_start + kv_start, hkv, 0], k_ready_s, k_smem3)
                if m >= 1:
                    mbarrier.wait(kv_free.index(p), (m - 1) & 1)
                mbarrier.expect(kv_ready.index(p), BN * D * 2)
                tma.async_copy_global_to_shared(v_desc, [seq_start + kv_start, hkv, 0], kv_ready.index(p), v_smem3.index(p))
                if LSEPF and not LSEPART:
                    # prefetch tile 0's lse/delta (loads in flight while waiting for the ring)
                    m0 = q_lo
                    h0 = hkv * GROUP
                    qv0 = m0 + offs_v
                    pf_lse = gl.load(LSE2 + h0 * T + seq_start + m0 + offs_v, mask=qv0 < n, other=float("inf"))
                    pf_del = gl.load(Delta + h0 * T + seq_start + m0 + offs_v, mask=qv0 < n, other=0.0)
                for i in range(n_tiles):
                    tt = t + i
                    s = tt % STAGES
                    g = i // tiles_per_head
                    m_start = q_lo + (i - g * tiles_per_head) * BM
                    h = hkv * GROUP + g
                    if tt >= STAGES:
                        mbarrier.wait(q_empty.index(s), ((tt - STAGES) // STAGES) & 1)
                    if LSETMA:
                        mbarrier.expect(q_ready.index(s), 2 * BM * D * 2 + 2 * BM * 4)
                    else:
                        mbarrier.expect(q_ready.index(s), 2 * BM * D * 2)
                    if LSEPART:
                        mbarrier.arrive(q_ready.index(s), count=1)
                    tma.async_copy_global_to_shared(q_desc, [seq_start + m_start, h, 0], q_ready.index(s), q_smem3.index(s))
                    tma.async_copy_global_to_shared(do_desc, [seq_start + m_start, h, 0], q_ready.index(s), do_smem3.index(s))
                    qv = m_start + offs_v
                    base = h * T + seq_start + m_start
                    if LSEPART:
                        pass
                    elif LSETMA:
                        # rows beyond the sequence get the neighbour's values (or zeros past T): the softmax masks q >= n
                        tma.async_copy_global_to_shared(lse_desc, [base], q_ready.index(s), lse_smem.index(s))
                        tma.async_copy_global_to_shared(del_desc, [base], q_ready.index(s), del_smem.index(s))
                        mbarrier.arrive(q_ready.index(s), count=1)
                    elif LSEFAKE:
                        lse_v = gl.zeros([BM], gl.float32, layout=vec_layout)
                        del_v = gl.zeros([BM], gl.float32, layout=vec_layout)
                    elif LSEPF:
                        lse_v = pf_lse
                        del_v = pf_del
                    else:
                        lse_v = gl.load(LSE2 + base + offs_v, mask=qv < n, other=float("inf"))
                        del_v = gl.load(Delta + base + offs_v, mask=qv < n, other=0.0)
                    if not LSETMA and not LSEPART:
                        lse_smem.index(s).store(lse_v)
                        del_smem.index(s).store(del_v)
                        fence_async_shared()
                        mbarrier.arrive(q_ready.index(s), count=1)
                    if LSEPF and not LSEPART and (i + 1 < n_tiles):
                        g1 = (i + 1) // tiles_per_head
                        m1 = q_lo + (i + 1 - g1 * tiles_per_head) * BM
                        h1 = hkv * GROUP + g1
                        qv1 = m1 + offs_v
                        pf_lse = gl.load(LSE2 + h1 * T + seq_start + m1 + offs_v, mask=qv1 < n, other=float("inf"))
                        pf_del = gl.load(Delta + h1 * T + seq_start + m1 + offs_v, mask=qv1 < n, other=0.0)
                t += n_tiles
                jj += 1


@gluon.jit
def _a_mma(CU, q_smem3, do_smem3, v_smem3, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
           k_ready, k_free, kv_ready, kv_free, acc_ready, acc_free, q_ready, q_empty, s_ready, dp_ready, p_ready, ds_ready, sink, dv_done,
           c, G, W, num_rounds, nseq, N, DBG,
           VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
           BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr,
           HAZ: gl.constexpr):
    p_layout: gl.constexpr = TensorMemoryLayout(block=[BN, BM], col_stride=1)
    t = 0
    jj = 0
    if TIMING:
        c_start = _clock()
        w_item = c_start - c_start
        w_q = c_start - c_start
        w_p = c_start - c_start
        w_ds = c_start - c_start
        w_acc = c_start - c_start
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            if ORDER == 1:
                nkvb = (N + BN - 1) // BN
                kvb = w % nkvb
                rem = w // nkvb
            else:
                kvb = w // (nseq * HKV)
                rem = w - kvb * (nseq * HKV)
            b = rem // HKV
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            kv_start = kvb * BN
            if kv_start < n:
                p = jj % VB
                m = jj // VB
                q_lo = (kv_start // BM) * BM
                tiles_per_head = (n - q_lo + BM - 1) // BM
                n_tiles = tiles_per_head * GROUP
                v_smem = v_smem3.index(p).reshape([BN, D])
                if TIMING:
                    c0 = _clock()
                mbarrier.wait(kv_ready.index(p), m & 1)
                mbarrier.wait(k_ready, jj & 1)
                _tmem_after_sync()
                if TIMING:
                    w_item += _clock() - c0
                # prologue: S(0), dP(0), S(1)
                st0 = t % STAGES
                if TIMING:
                    c0 = _clock()
                mbarrier.wait(q_ready.index(st0), (t // STAGES) & 1)
                if TIMING:
                    w_q += _clock() - c0
                q0 = q_smem3.index(st0).reshape([BM, D])
                do0 = do_smem3.index(st0).reshape([BM, D])
                tcgen05_mma(k_tmem, q0.permute((1, 0)), s_tmem.index(t % 2), use_acc=False,
                            mbarriers=[s_ready.index(t % 2), k_free], mbarrier_preds=[True, n_tiles == 1])
                tcgen05_mma(v_smem, do0.permute((1, 0)), dp_tmem, use_acc=False,
                            mbarriers=[dp_ready.index(t % 2), kv_free.index(p)], mbarrier_preds=[True, n_tiles == 1])
                if n_tiles > 1:
                    st1 = (t + 1) % STAGES
                    mbarrier.wait(q_ready.index(st1), ((t + 1) // STAGES) & 1)
                    q1 = q_smem3.index(st1).reshape([BM, D])
                    tcgen05_mma(k_tmem, q1.permute((1, 0)), s_tmem.index((t + 1) % 2), use_acc=False,
                                mbarriers=[s_ready.index((t + 1) % 2), k_free], mbarrier_preds=[True, n_tiles == 2])
                for i in range(n_tiles):
                    tt = t + i
                    slot = tt % 2
                    stage = tt % STAGES
                    q_c = q_smem3.index(stage).reshape([BM, D])
                    do_c = do_smem3.index(stage).reshape([BM, D])
                    p_t = s_tmem.index(slot)._reinterpret(gl.bfloat16, [BN, BM], p_layout)
                    ds_t = dp_tmem._reinterpret(gl.bfloat16, [BN, BM], p_layout)
                    if TIMING:
                        c0 = _clock()
                    if not MMAONLY:
                        mbarrier.wait(p_ready.index(slot), (tt // 2) & 1)
                    _tmem_after_sync()
                    if TIMING:
                        c1 = _clock()
                        w_p += c1 - c0
                    if (i == 0) and (jj >= 1):
                        mbarrier.wait(acc_free, (jj - 1) & 1)
                    if TIMING:
                        w_acc += _clock() - c1
                    tcgen05_mma(p_t, do_c, dv_tmem, use_acc=(i > 0), mbarriers=[dv_done])
                    if i + 2 < n_tiles:
                        # S(i+2) into slot i%2 (P(i) there was consumed by dV(i), issued just above)
                        nnstage = (tt + 2) % STAGES
                        if TIMING:
                            c0 = _clock()
                        mbarrier.wait(q_ready.index(nnstage), ((tt + 2) // STAGES) & 1)
                        if TIMING:
                            w_q += _clock() - c0
                        q_nn = q_smem3.index(nnstage).reshape([BM, D])
                        # WAR hazard: dV(i) reads P^T(i) from this slot as its A operand and the tensor core does
                        # not interlock that read against a later accumulator write -> wait for dV(i) to complete
                        if HAZ:
                            mbarrier.wait(dv_done, tt & 1)
                        tcgen05_mma(k_tmem, q_nn.permute((1, 0)), s_tmem.index(slot), use_acc=False,
                                    mbarriers=[s_ready.index(slot), k_free], mbarrier_preds=[True, i + 3 == n_tiles])
                    if TIMING:
                        c0 = _clock()
                    if not MMAONLY:
                        mbarrier.wait(ds_ready, tt & 1)
                    _tmem_after_sync()
                    if TIMING:
                        w_ds += _clock() - c0
                    tcgen05_mma(ds_t, q_c, dk_tmem, use_acc=(i > 0), mbarriers=[q_empty.index(stage)])
                    if i + 1 < n_tiles:
                        nstage = (tt + 1) % STAGES
                        do_n = do_smem3.index(nstage).reshape([BM, D])
                        # same hazard: dK(i) reads dS^T(i) from the dP slot -> wait for dK(i) (q_empty commit)
                        if HAZ:
                            mbarrier.wait(q_empty.index(stage), (tt // STAGES) & 1)
                        tcgen05_mma(v_smem, do_n.permute((1, 0)), dp_tmem, use_acc=False,
                                    mbarriers=[dp_ready.index((tt + 1) % 2), kv_free.index(p)], mbarrier_preds=[True, i + 2 == n_tiles])
                tcgen05_commit(acc_ready)
                t += n_tiles
                jj += 1
    if TIMING:
        total = _clock() - c_start
        offs = gl.arange(0, 1, layout=gl.BlockedLayout([1], [32], [1], [0]))
        gl.atomic_add(DBG + 0 + offs, total)
        gl.atomic_add(DBG + 1 + offs, w_item)
        gl.atomic_add(DBG + 2 + offs, w_q)
        gl.atomic_add(DBG + 3 + offs, w_p)
        gl.atomic_add(DBG + 4 + offs, w_ds)
        gl.atomic_add(DBG + 5 + offs, w_acc)


@gluon.jit
def _a_softmax(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
               k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
               PARITY: gl.constexpr, VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
               BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG,
               EARLY: gl.constexpr):
    if TIMING:
        c_start = _clock4()
        w_s = c_start - c_start
        w_p = c_start - c_start
        w_dp = c_start - c_start
        w_ds = c_start - c_start
    HB: gl.constexpr = BM // 2
    sh_layout: gl.constexpr = TensorMemoryLayout(block=[BN, HB], col_stride=1)   # fp32 half slot [BN, 32]
    ph_layout: gl.constexpr = TensorMemoryLayout(block=[BN, HB], col_stride=1)   # bf16 half [BN, 32] packed (16 cols)
    acc_layout: gl.constexpr = TensorMemoryLayout(block=[BN, D], col_stride=1)
    sh_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, [BN, HB], sh_layout, SM_WARPS)
    ph_reg: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BN, HB], ph_layout, SM_WARPS)
    acc_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, [BN, D], acc_layout, SM_WARPS)
    kt_layout: gl.constexpr = TensorMemoryLayout(block=[BN, D], col_stride=1)
    k_reg: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BN, D], kt_layout, SM_WARPS)
    col_layout: gl.constexpr = gl.SliceLayout(0, sh_reg)
    row_layout: gl.constexpr = gl.SliceLayout(1, sh_reg)
    offs_h = gl.arange(0, HB, layout=col_layout)
    offs_n = gl.arange(0, BN, layout=row_layout)
    acc_rows = gl.arange(0, BN, layout=gl.SliceLayout(1, acc_reg))
    acc_cols = gl.arange(0, D, layout=gl.SliceLayout(0, acc_reg))
    dp_lo = dp_tmem.slice(0, HB)
    dp_hi = dp_tmem.slice(HB, HB)
    ds_lo_t = dp_tmem.slice(0, HB // 2)._reinterpret(gl.bfloat16, [BN, HB], ph_layout)
    ds_hi_t = dp_tmem.slice(HB // 2, HB // 2)._reinterpret(gl.bfloat16, [BN, HB], ph_layout)
    t = 0
    jj = 0
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            if ORDER == 1:
                nkvb = (N + BN - 1) // BN
                kvb = w % nkvb
                rem = w // nkvb
            else:
                kvb = w // (nseq * HKV)
                rem = w - kvb * (nseq * HKV)
            b = rem // HKV
            hkv = rem - b * HKV
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            kv_start = kvb * BN
            if kv_start < n:
                p = jj % VB
                m = jj // VB
                q_lo = (kv_start // BM) * BM
                tiles_per_head = (n - q_lo + BM - 1) // BM
                n_tiles = tiles_per_head * GROUP
                if (PARITY == 0) and (jj == 0):
                    # first item: copy K (shared staging) -> tensor memory as the resident MMA A operand
                    mbarrier.wait(k_ready_s, 0)
                    kvals = k_smem3.reshape([BN, D]).load(k_reg)
                    if EARLY == 1:
                        kvals = _consume(kvals)
                        mbarrier.arrive(k_copied, count=1)
                    if EARLY == 2:
                        _consume_store(DBG, kvals)
                        mbarrier.arrive(k_copied, count=1)
                    k_tmem.store(kvals)
                    _tmem_store_fence()
                    if EARLY == 0:
                        mbarrier.arrive(k_copied, count=1)   # staging free only after the registers were consumed
                    mbarrier.arrive(k_ready, count=1)
                kv_idx = gl.expand_dims(kv_start + offs_n, 1)
                i0 = (PARITY - t % SMP + SMP) % SMP
                if MMAONLY:
                    i0 = n_tiles
                if LSEPART:
                    # prefetch this partition's first tile lse2/delta (64 q rows) straight from global memory
                    g0 = i0 // tiles_per_head
                    m0 = q_lo + (i0 - g0 * tiles_per_head) * BM
                    h0 = hkv * GROUP + g0
                    base0 = h0 * T + seq_start + m0
                    qv0 = m0 + offs_h
                    pf_lse_lo = gl.load(LSE2 + base0 + offs_h, mask=qv0 < n, other=float("inf"))
                    pf_lse_hi = gl.load(LSE2 + base0 + HB + offs_h, mask=qv0 + HB < n, other=float("inf"))
                    pf_del_lo = gl.load(Delta + base0 + offs_h, mask=qv0 < n, other=0.0)
                    pf_del_hi = gl.load(Delta + base0 + HB + offs_h, mask=qv0 + HB < n, other=0.0)
                for i in range(i0, n_tiles, SMP):
                    tt = t + i
                    slot = tt % 2
                    phase = (tt // 2) & 1
                    stage = tt % STAGES
                    s_lo = s_tmem.index(slot).slice(0, HB)
                    s_hi = s_tmem.index(slot).slice(HB, HB)
                    p_lo_t = s_tmem.index(slot).slice(0, HB // 2)._reinterpret(gl.bfloat16, [BN, HB], ph_layout)
                    p_hi_t = s_tmem.index(slot).slice(HB // 2, HB // 2)._reinterpret(gl.bfloat16, [BN, HB], ph_layout)
                    g = i // tiles_per_head
                    m_start = q_lo + (i - g * tiles_per_head) * BM
                    need_mask = (m_start < kv_start + BN) or (m_start + BM > n)
                    if TIMING:
                        c0 = _clock4()
                    mbarrier.wait(s_ready.index(slot), phase)
                    _tmem_after_sync4()
                    if TIMING:
                        c1 = _clock4()
                        w_s += c1 - c0
                    if LSEPART:
                        lse_lo = pf_lse_lo
                        lse_hi = pf_lse_hi
                        del_lo = pf_del_lo
                        del_hi = pf_del_hi
                        if i + SMP < n_tiles:
                            g2 = (i + SMP) // tiles_per_head
                            m2 = q_lo + (i + SMP - g2 * tiles_per_head) * BM
                            h2 = hkv * GROUP + g2
                            base2 = h2 * T + seq_start + m2
                            qv2 = m2 + offs_h
                            pf_lse_lo = gl.load(LSE2 + base2 + offs_h, mask=qv2 < n, other=float("inf"))
                            pf_lse_hi = gl.load(LSE2 + base2 + HB + offs_h, mask=qv2 + HB < n, other=float("inf"))
                            pf_del_lo = gl.load(Delta + base2 + offs_h, mask=qv2 < n, other=0.0)
                            pf_del_hi = gl.load(Delta + base2 + HB + offs_h, mask=qv2 + HB < n, other=0.0)
                    else:
                        lse_lo = lse_smem.index(stage).slice(0, HB).load(col_layout)
                        lse_hi = lse_smem.index(stage).slice(HB, HB).load(col_layout)
                    p0 = gl.exp2(s_lo.load(sh_reg) * sm_scale_log2e - gl.expand_dims(lse_lo, 0))
                    if need_mask:
                        q0 = gl.expand_dims(m_start + offs_h, 0)
                        p0 = gl.where((q0 >= kv_idx) & (q0 < n), p0, 0.0)
                    p_lo_t.store(gl.convert_layout(p0.to(gl.bfloat16), ph_reg))
                    p1 = gl.exp2(s_hi.load(sh_reg) * sm_scale_log2e - gl.expand_dims(lse_hi, 0))
                    if need_mask:
                        q1 = gl.expand_dims(m_start + HB + offs_h, 0)
                        p1 = gl.where((q1 >= kv_idx) & (q1 < n), p1, 0.0)
                    p_hi_t.store(gl.convert_layout(p1.to(gl.bfloat16), ph_reg))
                    _tmem_store_fence()
                    mbarrier.arrive(p_ready.index(slot), count=1)
                    if TIMING:
                        c2 = _clock4()
                        w_p += c2 - c1
                    mbarrier.wait(dp_ready.index(slot), phase)
                    _tmem_after_sync4()
                    if TIMING:
                        c3 = _clock4()
                        w_dp += c3 - c2
                    if not LSEPART:
                        del_lo = del_smem.index(stage).slice(0, HB).load(col_layout)
                        del_hi = del_smem.index(stage).slice(HB, HB).load(col_layout)
                    ds0 = p0 * (dp_lo.load(sh_reg) - gl.expand_dims(del_lo, 0))
                    ds_lo_t.store(gl.convert_layout(ds0.to(gl.bfloat16), ph_reg))
                    ds1 = p1 * (dp_hi.load(sh_reg) - gl.expand_dims(del_hi, 0))
                    ds_hi_t.store(gl.convert_layout(ds1.to(gl.bfloat16), ph_reg))
                    _tmem_store_fence()
                    mbarrier.arrive(ds_ready, count=1)
                    if TIMING:
                        w_ds += _clock4() - c3
                if PARITY == 0:
                    # lookahead: copy the NEXT item's K into tensor memory before this item's epilogue
                    if _a_has_next(r, c, G, W, num_rounds, nseq, N, CU, VARLEN, HKV, BN) == 1:
                        mbarrier.wait(k_ready_s, (jj + 1) & 1)
                        kvals = k_smem3.reshape([BN, D]).load(k_reg)
                        if EARLY == 1:
                            kvals = _consume(kvals)
                            mbarrier.arrive(k_copied, count=1)
                        if EARLY == 2:
                            _consume_store(DBG, kvals)
                            mbarrier.arrive(k_copied, count=1)
                        mbarrier.wait(k_free, jj & 1)
                        k_tmem.store(kvals)
                        _tmem_store_fence()
                        if EARLY == 0:
                            mbarrier.arrive(k_copied, count=1)
                        mbarrier.arrive(k_ready, count=1)
                # epilogue: parity 0 writes dK, parity 1 writes dV
                mbarrier.wait(acc_ready, jj & 1)
                _tmem_after_sync4()
                kv_rows = kv_start + acc_rows
                row_mask = gl.expand_dims(kv_rows < n, 1)
                optr = gl.expand_dims((seq_start + kv_rows) * stride_dkt + hkv * stride_dkh, 1) + gl.expand_dims(acc_cols, 0)
                if PARITY == 0:
                    dk = dk_tmem.load(acc_reg)
                    mbarrier.arrive(acc_free, count=1)
                    gl.store(DK + optr, (dk * sm_scale).to(gl.bfloat16), mask=row_mask)
                elif PARITY == 1:
                    dv = dv_tmem.load(acc_reg)
                    mbarrier.arrive(acc_free, count=1)
                    gl.store(DV + optr, dv.to(gl.bfloat16), mask=row_mask)
                t += n_tiles
                jj += 1
    if TIMING:
        total = _clock4() - c_start
        offs = gl.arange(0, 1, layout=gl.BlockedLayout([1], [32], [4], [0]))
        gl.atomic_add(DBG + 8 * (4 + PARITY) + 0 + offs, total)
        gl.atomic_add(DBG + 8 * (4 + PARITY) + 1 + offs, w_s)
        gl.atomic_add(DBG + 8 * (4 + PARITY) + 2 + offs, w_p)
        gl.atomic_add(DBG + 8 * (4 + PARITY) + 3 + offs, w_dp)
        gl.atomic_add(DBG + 8 * (4 + PARITY) + 4 + offs, w_ds)


@gluon.jit
def _a_softmax0(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
                BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG,
               EARLY: gl.constexpr):
    _a_softmax(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
               k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
               0, VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)


@gluon.jit
def _a_softmax1(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
                BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG,
               EARLY: gl.constexpr):
    _a_softmax(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
               k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
               1, VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)


@gluon.jit
def _a_softmax2(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
                BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG,
               EARLY: gl.constexpr):
    _a_softmax(CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
               k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
               2, VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)


@gluon.jit
def _a_kernel(q_desc, do_desc, k_desc, v_desc, lse_desc, del_desc, LSE2, Delta, DK, DV, CU, DBG,
              sm_scale, sm_scale_log2e, N, T, stride_dkt, stride_dkh, W, num_rounds, nseq,
              VARLEN: gl.constexpr, GROUP: gl.constexpr, HKV: gl.constexpr,
              BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr,
              HAZ: gl.constexpr, EARLY: gl.constexpr, num_warps: gl.constexpr):
    c = gl.program_id(0)
    G = gl.num_programs(0)

    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, 1, D], gl.bfloat16)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, 1, D], gl.bfloat16)
    q_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, 1, D], q_layout)
    do_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, 1, D], q_layout)
    k_smem3 = gl.allocate_shared_memory(gl.bfloat16, [BN, 1, D], kv_layout)
    v_smem3 = gl.allocate_shared_memory(gl.bfloat16, [VB, BN, 1, D], kv_layout)
    vec_smem_layout: gl.constexpr = gl.NVMMASharedLayout(0, 32, 1) if LSETMA else gl.SwizzledSharedLayout(1, 1, 1, [0])
    RING: gl.constexpr = 1 if LSEPART else STAGES
    lse_smem = gl.allocate_shared_memory(gl.float32, [RING, BM], vec_smem_layout)
    del_smem = gl.allocate_shared_memory(gl.float32, [RING, BM], vec_smem_layout)

    bar_layout: gl.constexpr = mbarrier.MBarrierLayout()
    k_ready = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    k_free = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    k_ready_s = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    k_copied = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    acc_ready = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    acc_free = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    sink = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    dv_done = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    dp_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    ds_ready = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    kv_ready = gl.allocate_shared_memory(gl.int64, [VB, 1], bar_layout)
    kv_free = gl.allocate_shared_memory(gl.int64, [VB, 1], bar_layout)
    q_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    q_empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    mbarrier.init(k_ready, count=1)
    mbarrier.init(k_free, count=1)
    mbarrier.init(k_ready_s, count=1)
    mbarrier.init(k_copied, count=1)
    mbarrier.init(acc_ready, count=1)
    mbarrier.init(acc_free, count=2)   # both softmax partitions
    mbarrier.init(sink, count=1)
    mbarrier.init(dv_done, count=1)
    mbarrier.init(ds_ready, count=1)
    for j in gl.static_range(VB):
        mbarrier.init(kv_ready.index(j), count=1)
        mbarrier.init(kv_free.index(j), count=1)   # MMA commit after the item's last dP (V no longer read)
    for j in gl.static_range(2):
        mbarrier.init(s_ready.index(j), count=1)
        mbarrier.init(p_ready.index(j), count=1)
        mbarrier.init(dp_ready.index(j), count=1)
    for j in gl.static_range(STAGES):
        mbarrier.init(q_ready.index(j), count=2)  # TMA (expect) + loader's lse/delta arrive
        mbarrier.init(q_empty.index(j), count=1)
    fence_async_shared()

    s_layout: gl.constexpr = TensorMemoryLayout(block=[BN, BM], col_stride=1)
    acc_layout: gl.constexpr = TensorMemoryLayout(block=[BN, D], col_stride=1)
    kt_layout: gl.constexpr = TensorMemoryLayout(block=[BN, D], col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [2, BN, BM], s_layout)
    dp_tmem = allocate_tensor_memory(gl.float32, [BN, BM], s_layout)
    dk_tmem = allocate_tensor_memory(gl.float32, [BN, D], acc_layout)
    dv_tmem = allocate_tensor_memory(gl.float32, [BN, D], acc_layout)
    k_tmem = allocate_tensor_memory(gl.bfloat16, [BN, D], kt_layout)

    if SMP == 3:
        gl.warp_specialize([
            (_a_softmax0, (CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                       k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                       VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)),
            (_a_softmax1, (CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                       k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                       VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)),
            (_a_softmax2, (CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                       k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                       VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)),
            (_a_load, (q_desc, do_desc, k_desc, v_desc, lse_desc, del_desc, LSE2, Delta, CU, q_smem3, do_smem3, k_smem3, v_smem3, lse_smem, del_smem,
                   k_ready_s, k_copied, kv_ready, kv_free, q_ready, q_empty, c, G, W, num_rounds, nseq, N, T,
                   VARLEN, GROUP, HKV, BM, BN, D, STAGES)),
            (_a_mma, (CU, q_smem3, do_smem3, v_smem3, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                  k_ready, k_free, kv_ready, kv_free, acc_ready, acc_free, q_ready, q_empty, s_ready, dp_ready, p_ready, ds_ready, sink, dv_done,
                  c, G, W, num_rounds, nseq, N, DBG, VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, HAZ)),
        ], [4, 4, 1, 1], [160, 160, 40, 40])
    else:
        gl.warp_specialize([
            (_a_softmax0, (CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                       k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                       VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)),
            (_a_softmax1, (CU, DK, DV, LSE2, Delta, T, k_smem3, lse_smem, del_smem, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                       k_ready_s, k_copied, k_ready, k_free, s_ready, dp_ready, p_ready, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, N, sm_scale, sm_scale_log2e, stride_dkt, stride_dkh,
                       VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, DBG, EARLY)),
            (_a_load, (q_desc, do_desc, k_desc, v_desc, lse_desc, del_desc, LSE2, Delta, CU, q_smem3, do_smem3, k_smem3, v_smem3, lse_smem, del_smem,
                   k_ready_s, k_copied, kv_ready, kv_free, q_ready, q_empty, c, G, W, num_rounds, nseq, N, T,
                   VARLEN, GROUP, HKV, BM, BN, D, STAGES)),
            (_a_mma, (CU, q_smem3, do_smem3, v_smem3, k_tmem, s_tmem, dp_tmem, dk_tmem, dv_tmem,
                  k_ready, k_free, kv_ready, kv_free, acc_ready, acc_free, q_ready, q_empty, s_ready, dp_ready, p_ready, ds_ready, sink, dv_done,
                  c, G, W, num_rounds, nseq, N, DBG, VARLEN, GROUP, HKV, BM, BN, D, STAGES, TIMING, HAZ)),
        ], [4, 1, 1], [192, 40, 40])

    mbarrier.invalidate(k_ready)
    mbarrier.invalidate(k_free)
    mbarrier.invalidate(k_ready_s)
    mbarrier.invalidate(k_copied)
    mbarrier.invalidate(acc_ready)
    mbarrier.invalidate(acc_free)
    mbarrier.invalidate(sink)
    mbarrier.invalidate(dv_done)
    mbarrier.invalidate(ds_ready)
    for j in gl.static_range(VB):
        mbarrier.invalidate(kv_ready.index(j))
        mbarrier.invalidate(kv_free.index(j))
    for j in gl.static_range(2):
        mbarrier.invalidate(s_ready.index(j))
        mbarrier.invalidate(p_ready.index(j))
        mbarrier.invalidate(dp_ready.index(j))
    for j in gl.static_range(STAGES):
        mbarrier.invalidate(q_ready.index(j))
        mbarrier.invalidate(q_empty.index(j))




@gluon.jit
def _qchk_readback(DBG, qv, tm, c, num_rounds, j, PARITY: gl.constexpr, BMQ: gl.constexpr, D: gl.constexpr):
    # debug: re-read the tensor-memory operand just stored and record per-row |diff| sums
    back = tm.load(qv.type.layout)
    dd = gl.sum(gl.abs(back.to(gl.float32) - qv.to(gl.float32)), axis=1)
    rows = gl.arange(0, BMQ, layout=gl.SliceLayout(1, qv.type.layout))
    gl.store(DBG + ((c * num_rounds + j) * 2 + PARITY) * BMQ + rows, dd)


@gluon.jit
def _qchk_store(DBG, qv, c, num_rounds, j, PARITY: gl.constexpr, BMQ: gl.constexpr, D: gl.constexpr):
    # debug: per-row checksum (fp32 row sum) of the operand tile that was just copied out of shared memory
    rs = gl.sum(qv.to(gl.float32), axis=1)
    rows = gl.arange(0, BMQ, layout=gl.SliceLayout(1, qv.type.layout))
    gl.store(DBG + ((c * num_rounds + j) * 2 + PARITY) * BMQ + rows, rs)



@gluon.jit
def _spin4(cycles):
    t0 = gl.max(_clock4(), axis=0)
    tn = t0
    while tn - t0 < cycles:
        tn = gl.max(_clock4(), axis=0)


@gluon.jit
def _spin1(cycles):
    t0 = gl.max(_clock(), axis=0)
    tn = t0
    while tn - t0 < cycles:
        tn = gl.max(_clock(), axis=0)


# =============================================================================== pass B: dQ
@gluon.jit
def _b_load(q_desc, do_desc, k_desc, v_desc, CU, q_smem3, do_smem3, k_smem3, v_smem3,
            qdo_ready, qdo_free, kv_ready, kv_empty,
            c, G, W, num_rounds, nseq, num_q_tiles, N,
            VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
            BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, SPIN: gl.constexpr):
    t = 0
    jj = 0
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            qt, b, h = _b_item(w, c, G, W, nseq, num_q_tiles, H)
            hkv = h // GROUP
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            m_start = qt * BMQ
            if m_start < n:
                kv_end = gl.minimum(m_start + BMQ, n)
                n_kv = (kv_end + BNQ - 1) // BNQ
                if jj >= 1:
                    mbarrier.wait(qdo_free, (jj - 1) & 1)
                if SPIN == 2:
                    _spin1(20000)
                mbarrier.expect(qdo_ready, 2 * BMQ * D * 2)
                tma.async_copy_global_to_shared(q_desc, [seq_start + m_start, h, 0], qdo_ready, q_smem3)
                tma.async_copy_global_to_shared(do_desc, [seq_start + m_start, h, 0], qdo_ready, do_smem3)
                for i in range(n_kv):
                    tt = t + i
                    s = tt % STAGES
                    if tt >= STAGES:
                        mbarrier.wait(kv_empty.index(s), ((tt - STAGES) // STAGES) & 1)
                    mbarrier.expect(kv_ready.index(s), 2 * BNQ * D * 2)
                    tma.async_copy_global_to_shared(k_desc, [seq_start + i * BNQ, hkv, 0], kv_ready.index(s), k_smem3.index(s))
                    tma.async_copy_global_to_shared(v_desc, [seq_start + i * BNQ, hkv, 0], kv_ready.index(s), v_smem3.index(s))
                t += n_kv
                jj += 1


@gluon.jit
def _b_mma(CU, k_smem3, v_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
           qt_ready, qt_free, kv_ready, kv_empty, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free, sink,
           c, G, W, num_rounds, nseq, num_q_tiles, N, DBG,
           VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
           BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr,
           SERIAL: gl.constexpr, SPIN: gl.constexpr, HAZ: gl.constexpr):
    # Schedule (per q tile item): S is issued TWO kv tiles ahead, dP one ahead. The S slot is free as soon as the
    # softmax partition has loaded S (s_consumed), because dS is written into the dP slot (dP was loaded before).
    ds_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, BNQ], col_stride=1)
    t = 0
    jj = 0
    if TIMING:
        c_start = _clock()
        w_item = c_start - c_start
        w_q = c_start - c_start
        w_p = c_start - c_start
        w_ds = c_start - c_start
        w_acc = c_start - c_start
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            qt, b, h = _b_item(w, c, G, W, nseq, num_q_tiles, H)
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            m_start = qt * BMQ
            if m_start < n:
                kv_end = gl.minimum(m_start + BMQ, n)
                n_kv = (kv_end + BNQ - 1) // BNQ
                if TIMING:
                    c0 = _clock()
                mbarrier.wait(qt_ready.index(0), jj & 1)
                mbarrier.wait(qt_ready.index(1), jj & 1)
                _tmem_after_sync()
                if TIMING:
                    c1 = _clock()
                    w_item += c1 - c0
                # prologue: S(t), dP(t), S(t+1)
                st0 = t % STAGES
                mbarrier.wait(kv_ready.index(st0), (t // STAGES) & 1)
                if TIMING:
                    w_q += _clock() - c1
                if (not MMAONLY) and (t >= 2):
                    mbarrier.wait(s_consumed.index(t % 2), ((t - 2) // 2) & 1)
                k0 = k_smem3.index(st0).reshape([BNQ, D])
                v0 = v_smem3.index(st0).reshape([BNQ, D])
                tcgen05_mma(q_tmem, k0.permute((1, 0)), s_tmem.index(t % 2), use_acc=False, mbarriers=[s_ready.index(t % 2)])
                tcgen05_mma(do_tmem, v0.permute((1, 0)), dp_tmem.index(t % 2), use_acc=False,
                            mbarriers=[dp_ready.index(t % 2), qt_free], mbarrier_preds=[True, n_kv == 1])
                if n_kv > 1:
                    st1 = (t + 1) % STAGES
                    mbarrier.wait(kv_ready.index(st1), ((t + 1) // STAGES) & 1)
                    if (not MMAONLY) and (t + 1 >= 2):
                        mbarrier.wait(s_consumed.index((t + 1) % 2), ((t - 1) // 2) & 1)
                    k1 = k_smem3.index(st1).reshape([BNQ, D])
                    tcgen05_mma(q_tmem, k1.permute((1, 0)), s_tmem.index((t + 1) % 2), use_acc=False,
                                mbarriers=[s_ready.index((t + 1) % 2)])
                for i in range(n_kv):
                    tt = t + i
                    slot = tt % 2
                    stage = tt % STAGES
                    if i + 1 < n_kv:
                        # dP(tt+1): its slot held dS(tt-1), read by dQ(tt-1) issued last iteration (in-order)
                        nslot = (tt + 1) % 2
                        nstage = (tt + 1) % STAGES
                        if TIMING:
                            c0 = _clock()
                        mbarrier.wait(kv_ready.index(nstage), ((tt + 1) // STAGES) & 1)
                        if TIMING:
                            w_q += _clock() - c0
                        v_n = v_smem3.index(nstage).reshape([BNQ, D])
                        tcgen05_mma(do_tmem, v_n.permute((1, 0)), dp_tmem.index(nslot), use_acc=False,
                                    mbarriers=[dp_ready.index(nslot), qt_free], mbarrier_preds=[True, i + 2 == n_kv])
                    if i + 2 < n_kv:
                        # S(tt+2) into slot tt%2 once the softmax has loaded S(tt)
                        nnstage = (tt + 2) % STAGES
                        if TIMING:
                            c0 = _clock()
                        mbarrier.wait(kv_ready.index(nnstage), ((tt + 2) // STAGES) & 1)
                        if not MMAONLY:
                            mbarrier.wait(s_consumed.index(slot), (tt // 2) & 1)
                        if TIMING:
                            w_p += _clock() - c0
                        k_nn = k_smem3.index(nnstage).reshape([BNQ, D])
                        tcgen05_mma(q_tmem, k_nn.permute((1, 0)), s_tmem.index(slot), use_acc=False,
                                    mbarriers=[s_ready.index(slot)])
                    if TIMING:
                        c0 = _clock()
                    if not MMAONLY:
                        mbarrier.wait(ds_ready.index(slot), (tt // 2) & 1)
                    _tmem_after_sync()
                    if TIMING:
                        c1 = _clock()
                        w_ds += c1 - c0
                    if (i == 0) and (jj >= 1):
                        mbarrier.wait(acc_free, (jj - 1) & 1)
                    if TIMING:
                        w_acc += _clock() - c1
                    k_c = k_smem3.index(stage).reshape([BNQ, D])
                    ds_t = dp_tmem.index(slot)._reinterpret(gl.bfloat16, [BMQ, BNQ], ds_layout)
                    tcgen05_mma(ds_t, k_c, dq_tmem, use_acc=(i > 0), mbarriers=[kv_empty.index(stage)])
                tcgen05_commit(acc_ready)
                t += n_kv
                jj += 1
    if TIMING:
        total = _clock() - c_start
        offs = gl.arange(0, 1, layout=gl.BlockedLayout([1], [32], [1], [0]))
        gl.atomic_add(DBG + 8 + 0 + offs, total)
        gl.atomic_add(DBG + 8 + 1 + offs, w_item)
        gl.atomic_add(DBG + 8 + 2 + offs, w_q)
        gl.atomic_add(DBG + 8 + 3 + offs, w_p)
        gl.atomic_add(DBG + 8 + 4 + offs, w_ds)
        gl.atomic_add(DBG + 8 + 5 + offs, w_acc)


@gluon.jit
def _b_softmax(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
               qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
               PARITY: gl.constexpr, VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
               BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG,
               QCHK: gl.constexpr, SPIN: gl.constexpr, EARLY: gl.constexpr):
    if TIMING:
        c_start = _clock4()
        w_s = c_start - c_start
        w_p = c_start - c_start
        w_dp = c_start - c_start
        w_ds = c_start - c_start
    HB: gl.constexpr = BNQ // 2
    sh_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, HB], col_stride=1)
    ph_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, HB], col_stride=1)
    acc_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, D], col_stride=1)
    qt_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, D], col_stride=1)
    sh_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, [BMQ, HB], sh_layout, SM_WARPS)
    ph_reg: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BMQ, HB], ph_layout, SM_WARPS)
    acc_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, [BMQ, D], acc_layout, SM_WARPS)
    q_reg: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BMQ, D], qt_layout, SM_WARPS)
    col_layout: gl.constexpr = gl.SliceLayout(0, sh_reg)
    row_layout: gl.constexpr = gl.SliceLayout(1, sh_reg)
    offs_h = gl.arange(0, HB, layout=col_layout)
    offs_m = gl.arange(0, BMQ, layout=row_layout)
    acc_rows = gl.arange(0, BMQ, layout=gl.SliceLayout(1, acc_reg))
    acc_cols = gl.arange(0, D, layout=gl.SliceLayout(0, acc_reg))
    t = 0
    jj = 0
    for r in range(num_rounds):
        w = _item_index(r, c, G, W)
        if w < W:
            qt, b, h = _b_item(w, c, G, W, nseq, num_q_tiles, H)
            seq_start, n = _seq_info(CU, b, N, VARLEN)
            m_start = qt * BMQ
            if m_start < n:
                kv_end = gl.minimum(m_start + BMQ, n)
                n_kv = (kv_end + BNQ - 1) // BNQ
                if ((jj == 0) or (SPIN == 5)) and (PARITY < 2):
                    # first item: copy Q (parity 0) / dO (parity 1) from shared to tensor memory
                    mbarrier.wait(qdo_ready, jj & 1)
                    if jj >= 1:
                        mbarrier.wait(qt_free, (jj - 1) & 1)
                    if PARITY == 0:
                        qv = q_smem3.reshape([BMQ, D]).load(q_reg)
                    else:
                        qv = do_smem3.reshape([BMQ, D]).load(q_reg)
                    if QCHK == 1:
                        _qchk_store(DBG, qv, c, num_rounds, 0, PARITY, BMQ, D)
                    if EARLY == 1:
                        qv = _consume(qv)
                        mbarrier.arrive(qdo_free, count=1)
                    if EARLY == 2:
                        _consume_store(DBG, qv)
                        mbarrier.arrive(qdo_free, count=1)
                    if PARITY == 0:
                        q_tmem.store(qv)
                    else:
                        do_tmem.store(qv)
                    _tmem_store_fence()
                    if EARLY == 0:
                        # free the staging buffer only after the registers were consumed by the TMEM store
                        # (otherwise the shared-memory loads can still be in flight when the loader's next TMA lands)
                        mbarrier.arrive(qdo_free, count=1)
                    if QCHK == 3:
                        if PARITY == 0:
                            _qchk_readback(DBG, qv, q_tmem, c, num_rounds, jj, PARITY, BMQ, D)
                        else:
                            _qchk_readback(DBG, qv, do_tmem, c, num_rounds, jj, PARITY, BMQ, D)
                    mbarrier.arrive(qt_ready.index(PARITY), count=1)
                q_idx = m_start + offs_m
                q_mask = q_idx < n
                lse2 = gl.load(LSE2 + h * T + seq_start + q_idx, mask=q_mask, other=float("inf"))
                delta = gl.load(Delta + h * T + seq_start + q_idx, mask=q_mask, other=0.0)
                lse2e = gl.expand_dims(lse2, 1)
                deltae = gl.expand_dims(delta, 1)
                q_idxe = gl.expand_dims(q_idx, 1)
                i0 = (PARITY - t % SMP + SMP) % SMP
                if MMAONLY:
                    i0 = n_kv
                for i in range(i0, n_kv, SMP):
                    tt = t + i
                    slot = tt % 2
                    phase = (tt // 2) & 1
                    s_lo = s_tmem.index(slot).slice(0, HB)
                    s_hi = s_tmem.index(slot).slice(HB, HB)
                    dp_lo = dp_tmem.index(slot).slice(0, HB)
                    dp_hi = dp_tmem.index(slot).slice(HB, HB)
                    ds_lo_t = dp_tmem.index(slot).slice(0, HB // 2)._reinterpret(gl.bfloat16, [BMQ, HB], ph_layout)
                    ds_hi_t = dp_tmem.index(slot).slice(HB // 2, HB // 2)._reinterpret(gl.bfloat16, [BMQ, HB], ph_layout)
                    kv_start = i * BNQ
                    if TIMING:
                        c0 = _clock4()
                    mbarrier.wait(s_ready.index(slot), phase)
                    _tmem_after_sync4()
                    if TIMING:
                        c1 = _clock4()
                        w_s += c1 - c0
                    need_mask = (kv_start + BNQ > m_start) or (kv_start + BNQ > n)
                    sv0 = s_lo.load(sh_reg)
                    sv1 = s_hi.load(sh_reg)
                    _tmem_store_fence()
                    mbarrier.arrive(s_consumed.index(slot), count=1)   # S slot may be overwritten by S(tt+2) now
                    p0 = gl.exp2(sv0 * sm_scale_log2e - lse2e)
                    if need_mask:
                        kv0 = gl.expand_dims(kv_start + offs_h, 0)
                        p0 = gl.where((q_idxe >= kv0) & (kv0 < n), p0, 0.0)
                    p1 = gl.exp2(sv1 * sm_scale_log2e - lse2e)
                    if need_mask:
                        kv1 = gl.expand_dims(kv_start + HB + offs_h, 0)
                        p1 = gl.where((q_idxe >= kv1) & (kv1 < n), p1, 0.0)
                    if TIMING:
                        c2 = _clock4()
                        w_p += c2 - c1
                    mbarrier.wait(dp_ready.index(slot), phase)
                    _tmem_after_sync4()
                    if TIMING:
                        c3 = _clock4()
                        w_dp += c3 - c2
                    dpv0 = dp_lo.load(sh_reg)
                    dpv1 = dp_hi.load(sh_reg)
                    ds0 = p0 * (dpv0 - deltae)
                    ds_lo_t.store(gl.convert_layout(ds0.to(gl.bfloat16), ph_reg))
                    ds1 = p1 * (dpv1 - deltae)
                    ds_hi_t.store(gl.convert_layout(ds1.to(gl.bfloat16), ph_reg))
                    _tmem_store_fence()
                    mbarrier.arrive(ds_ready.index(slot), count=1)
                    if TIMING:
                        w_ds += _clock4() - c3
                # lookahead: copy the NEXT item's Q/dO into tensor memory before this item's epilogue
                if (PARITY < 2) and (SPIN != 5) and (_b_has_next(r, c, G, W, num_rounds, nseq, num_q_tiles, N, CU, VARLEN, H, BMQ) == 1):
                    mbarrier.wait(qdo_ready, (jj + 1) & 1)
                    if PARITY == 0:
                        qv = q_smem3.reshape([BMQ, D]).load(q_reg)
                    else:
                        qv = do_smem3.reshape([BMQ, D]).load(q_reg)
                    if QCHK == 1:
                        _qchk_store(DBG, qv, c, num_rounds, jj + 1, PARITY, BMQ, D)
                    if EARLY == 1:
                        qv = _consume(qv)
                        mbarrier.arrive(qdo_free, count=1)
                    if EARLY == 2:
                        _consume_store(DBG, qv)
                        mbarrier.arrive(qdo_free, count=1)
                    mbarrier.wait(qt_free, jj & 1)
                    if PARITY == 0:
                        q_tmem.store(qv)
                    else:
                        do_tmem.store(qv)
                    _tmem_store_fence()
                    if EARLY == 0:
                        mbarrier.arrive(qdo_free, count=1)
                    if QCHK == 3:
                        if PARITY == 0:
                            _qchk_readback(DBG, qv, q_tmem, c, num_rounds, jj + 1, PARITY, BMQ, D)
                        else:
                            _qchk_readback(DBG, qv, do_tmem, c, num_rounds, jj + 1, PARITY, BMQ, D)
                    mbarrier.arrive(qt_ready.index(PARITY), count=1)
                mbarrier.wait(acc_ready, jj & 1)
                _tmem_after_sync4()
                if PARITY == 0:
                    dq = dq_tmem.load(acc_reg)
                    if QCHK == 2:
                        # debug: re-read the accumulator after a delay and record per-row differences
                        t0 = gl.max(_clock4(), axis=0)
                        tn = t0
                        while tn - t0 < 20000:
                            tn = gl.max(_clock4(), axis=0)
                        dq2 = dq_tmem.load(acc_reg)
                        dd = gl.sum(gl.abs(dq2 - dq), axis=1)
                        gl.store(DBG + (c * num_rounds + jj) * BMQ + acc_rows, dd)
                    mbarrier.arrive(acc_free, count=1)
                    q_rows = m_start + acc_rows
                    row_mask = gl.expand_dims(q_rows < n, 1)
                    optr = gl.expand_dims((seq_start + q_rows) * stride_dqt + h * stride_dqh, 1) + gl.expand_dims(acc_cols, 0)
                    gl.store(DQ + optr, (dq * sm_scale).to(gl.bfloat16), mask=row_mask)
                t += n_kv
                jj += 1
    if TIMING:
        total = _clock4() - c_start
        offs = gl.arange(0, 1, layout=gl.BlockedLayout([1], [32], [4], [0]))
        gl.atomic_add(DBG + 8 * (2 + PARITY) + 0 + offs, total)
        gl.atomic_add(DBG + 8 * (2 + PARITY) + 1 + offs, w_s)
        gl.atomic_add(DBG + 8 * (2 + PARITY) + 2 + offs, w_p)
        gl.atomic_add(DBG + 8 * (2 + PARITY) + 3 + offs, w_dp)
        gl.atomic_add(DBG + 8 * (2 + PARITY) + 4 + offs, w_ds)


@gluon.jit
def _b_softmax0(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
                BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG, QCHK: gl.constexpr, SPIN: gl.constexpr, EARLY: gl.constexpr):
    _b_softmax(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
               qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
               0, VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)


@gluon.jit
def _b_softmax1(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
                BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG, QCHK: gl.constexpr, SPIN: gl.constexpr, EARLY: gl.constexpr):
    _b_softmax(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
               qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
               1, VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)


@gluon.jit
def _b_softmax2(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
                BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr, DBG, QCHK: gl.constexpr, SPIN: gl.constexpr, EARLY: gl.constexpr):
    _b_softmax(CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
               qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
               c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
               2, VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)


@gluon.jit
def _b_kernel(q_desc, do_desc, k_desc, v_desc, LSE2, Delta, DQ, CU, DBG,
              sm_scale, sm_scale_log2e, N, T, stride_dqt, stride_dqh, W, num_rounds, nseq, num_q_tiles,
              VARLEN: gl.constexpr, GROUP: gl.constexpr, H: gl.constexpr,
              BMQ: gl.constexpr, BNQ: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, TIMING: gl.constexpr,
              SERIAL: gl.constexpr, QCHK: gl.constexpr, SPIN: gl.constexpr, HAZ: gl.constexpr, EARLY: gl.constexpr, num_warps: gl.constexpr):
    c = gl.program_id(0)
    G = gl.num_programs(0)

    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BMQ, 1, D], gl.bfloat16)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BNQ, 1, D], gl.bfloat16)
    q_smem3 = gl.allocate_shared_memory(gl.bfloat16, [BMQ, 1, D], q_layout)
    do_smem3 = gl.allocate_shared_memory(gl.bfloat16, [BMQ, 1, D], q_layout)
    k_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BNQ, 1, D], kv_layout)
    v_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BNQ, 1, D], kv_layout)

    bar_layout: gl.constexpr = mbarrier.MBarrierLayout()
    acc_ready = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    acc_free = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    qt_free = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    sink = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    qt_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    qdo_ready = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    qdo_free = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    kv_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    kv_empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    dp_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    s_consumed = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    ds_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    mbarrier.init(acc_ready, count=1)
    mbarrier.init(acc_free, count=1)
    mbarrier.init(qt_free, count=1)
    mbarrier.init(sink, count=1)
    mbarrier.init(qdo_ready, count=1)
    mbarrier.init(qdo_free, count=2)   # both softmax partitions read their operand from staging
    for j in gl.static_range(STAGES):
        mbarrier.init(kv_ready.index(j), count=1)
        mbarrier.init(kv_empty.index(j), count=1)
    for j in gl.static_range(2):
        mbarrier.init(qt_ready.index(j), count=1)
        mbarrier.init(s_ready.index(j), count=1)
        mbarrier.init(dp_ready.index(j), count=1)
        mbarrier.init(ds_ready.index(j), count=1)
        mbarrier.init(s_consumed.index(j), count=1)
    fence_async_shared()

    s_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, BNQ], col_stride=1)
    acc_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, D], col_stride=1)
    qt_layout: gl.constexpr = TensorMemoryLayout(block=[BMQ, D], col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [2, BMQ, BNQ], s_layout)
    dp_tmem = allocate_tensor_memory(gl.float32, [2, BMQ, BNQ], s_layout)
    dq_tmem = allocate_tensor_memory(gl.float32, [BMQ, D], acc_layout)
    q_tmem = allocate_tensor_memory(gl.bfloat16, [BMQ, D], qt_layout)
    do_tmem = allocate_tensor_memory(gl.bfloat16, [BMQ, D], qt_layout)

    if SMP == 3:
        gl.warp_specialize([
            (_b_softmax0, (CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                       qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                       VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)),
            (_b_softmax1, (CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                       qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                       VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)),
            (_b_softmax2, (CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                       qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                       VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)),
            (_b_load, (q_desc, do_desc, k_desc, v_desc, CU, q_smem3, do_smem3, k_smem3, v_smem3,
                   qdo_ready, qdo_free, kv_ready, kv_empty, c, G, W, num_rounds, nseq, num_q_tiles, N,
                   VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, SPIN)),
            (_b_mma, (CU, k_smem3, v_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                  qt_ready, qt_free, kv_ready, kv_empty, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free, sink,
                  c, G, W, num_rounds, nseq, num_q_tiles, N, DBG, VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, SERIAL, SPIN, HAZ)),
        ], [4, 4, 1, 1], [160, 160, 40, 40])
    else:
        gl.warp_specialize([
            (_b_softmax0, (CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                       qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                       VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)),
            (_b_softmax1, (CU, LSE2, Delta, DQ, q_smem3, do_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                       qdo_ready, qdo_free, qt_ready, qt_free, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free,
                       c, G, W, num_rounds, nseq, num_q_tiles, N, T, sm_scale, sm_scale_log2e, stride_dqt, stride_dqh,
                       VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, DBG, QCHK, SPIN, EARLY)),
            (_b_load, (q_desc, do_desc, k_desc, v_desc, CU, q_smem3, do_smem3, k_smem3, v_smem3,
                   qdo_ready, qdo_free, kv_ready, kv_empty, c, G, W, num_rounds, nseq, num_q_tiles, N,
                   VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, SPIN)),
            (_b_mma, (CU, k_smem3, v_smem3, q_tmem, do_tmem, s_tmem, dp_tmem, dq_tmem,
                  qt_ready, qt_free, kv_ready, kv_empty, s_ready, dp_ready, s_consumed, ds_ready, acc_ready, acc_free, sink,
                  c, G, W, num_rounds, nseq, num_q_tiles, N, DBG, VARLEN, GROUP, H, BMQ, BNQ, D, STAGES, TIMING, SERIAL, SPIN, HAZ)),
        ], [4, 1, 1], [192, 40, 40])

    mbarrier.invalidate(acc_ready)
    mbarrier.invalidate(acc_free)
    mbarrier.invalidate(qt_free)
    mbarrier.invalidate(sink)
    for j in gl.static_range(STAGES):
        mbarrier.invalidate(kv_ready.index(j))
        mbarrier.invalidate(kv_empty.index(j))
    mbarrier.invalidate(qdo_ready)
    mbarrier.invalidate(qdo_free)
    for j in gl.static_range(2):
        mbarrier.invalidate(qt_ready.index(j))
        mbarrier.invalidate(s_ready.index(j))
        mbarrier.invalidate(dp_ready.index(j))
        mbarrier.invalidate(ds_ready.index(j))
        mbarrier.invalidate(s_consumed.index(j))


# =============================================================================== host
_BM = int(os.environ.get("GL_BM", "64"))
_BN = int(os.environ.get("GL_BN", "128"))
_BMQ = int(os.environ.get("GL_BMQ", "128"))
_BNQ = int(os.environ.get("GL_BNQ", "64"))
_STA = int(os.environ.get("GL_STA", "4"))
_STB = int(os.environ.get("GL_STB", "5"))
_DEBUG = int(os.environ.get("GL_DEBUG", "0"))
_ONLY = os.environ.get("GL_ONLY", "")  # "A" or "B" to time one pass
_NUM_SMS = None
_SYNC = int(os.environ.get("GL_SYNC", "0"))
_TIMING = int(os.environ.get("GL_TIMING", "0"))
_SERIAL = int(os.environ.get("GL_SERIAL", "0"))
_QCHK = int(os.environ.get("GL_QCHK", "0"))
_SPIN = int(os.environ.get("GL_SPIN", "0"))
_PRE_TORCH = int(os.environ.get("GL_PRE_TORCH", "0"))
_HAZ = int(os.environ.get("GL_HAZ", "0"))
_STREAMS = int(os.environ.get("GL_STREAMS", "0"))
_GCAP_A = int(os.environ.get("GL_GCAP_A", "0"))  # per-pass overrides (0 = use G_max)
_GCAP_B = int(os.environ.get("GL_GCAP_B", "0"))  # 1: run pass B on a side stream concurrently with pass A
_SIDE = None
_EARLY = int(os.environ.get("GL_EARLY", "0"))
DBG_BUF = None
_GCAP = int(os.environ.get("GL_GCAP", "-1"))  # -1: choose by shape


_SINK = None


def _sink_buffer(device):
    # data-independent one-time workspace: dummy sink for the staging-release ordering stores
    global _SINK
    if _SINK is None or _SINK.device != device:
        _SINK = torch.zeros(1024, dtype=torch.int32, device=device)
    return _SINK


def _num_sms(device):
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def attention_bwd(q, k, v, out, dout, lse, sm_scale, cu_seqlens=None, max_seqlen=None,
                  BM=_BM, BN=_BN, BMQ=_BMQ, BNQ=_BNQ):
    varlen = cu_seqlens is not None
    if varlen:
        T, H, D = q.shape
        Hkv = k.shape[1]
        nseq = cu_seqlens.numel() - 1
        N = int(max_seqlen)
        q2, k2, v2, o2, do2 = [t.contiguous() for t in (q, k, v, out, dout)]
    else:
        B, N, H, D = q.shape
        Hkv = k.shape[2]
        nseq = B
        T = B * N
        q2, k2, v2, o2, do2 = [t.contiguous().view(B * N, t.shape[2], D) for t in (q, k, v, out, dout)]
    assert D == 128 and H == 16 and H % Hkv == 0
    lse = lse.contiguous()
    if varlen and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    if varlen:
        cu_seqlens = cu_seqlens.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(H, T, dtype=torch.float32, device=q.device)
    lse2 = torch.empty(H, T, dtype=torch.float32, device=q.device)
    PRE_BLOCK = 128
    if _PRE_TORCH:
        delta = (o2.float() * do2.float()).sum(-1).t().contiguous()
        lse2 = (lse.float().reshape(H, T) if varlen else lse.float().permute(1, 0, 2).reshape(H, T)) * 1.4426950408889634
        lse2 = lse2.contiguous()
    else:
        _preprocess[(triton.cdiv(T, PRE_BLOCK), H)](o2, do2, lse, delta, lse2, o2.stride(0), o2.stride(1), T, N,
                                                    VARLEN=varlen, H=H, D=D, BLOCK=PRE_BLOCK)
    dk2 = dk.view(T, Hkv, D)
    dv2 = dv.view(T, Hkv, D)
    dq2 = dq.view(T, H, D)
    sm_scale_log2e = sm_scale * 1.4426950408889634
    cu = cu_seqlens if varlen else delta
    if _GCAP == 0:
        G_max = _num_sms(q.device)
    elif _GCAP > 0:
        G_max = _GCAP
    else:
        # shape-based choice (measured on B200): dense MHA prefers 2 CTAs/SM, GQA and varlen prefer 4 CTAs/SM
        G_max = 2 * _num_sms(q.device) if (not varlen and Hkv == H) else 4 * _num_sms(q.device)
    global DBG_BUF
    if _TIMING:
        DBG_BUF = torch.zeros(6, 8, dtype=torch.int64, device=q.device)
    dbg = DBG_BUF if _TIMING else delta
    if _EARLY == 2 and not _TIMING and not _QCHK:
        dbg = _sink_buffer(q.device)
    if _QCHK:
        DBG_BUF = torch.zeros(max(_GCAP, 148) * 128 * 2 * BMQ, dtype=torch.float32, device=q.device)
        dbg = DBG_BUF
    global _SIDE
    main_stream = torch.cuda.current_stream(q.device)
    if _STREAMS and _SIDE is None:
        _SIDE = torch.cuda.Stream(device=q.device)
    if _STREAMS:
        _SIDE.wait_stream(main_stream)   # pass B needs lse2/delta from the preprocess
    if _ONLY != "B":
        q_layout = gl.NVMMASharedLayout.get_default_for([BM, 1, D], gl.bfloat16)
        kv_layout = gl.NVMMASharedLayout.get_default_for([BN, 1, D], gl.bfloat16)
        q_desc = TensorDescriptor.from_tensor(q2, [BM, 1, D], q_layout)
        do_desc = TensorDescriptor.from_tensor(do2, [BM, 1, D], q_layout)
        k_desc = TensorDescriptor.from_tensor(k2, [BN, 1, D], kv_layout)
        v_desc = TensorDescriptor.from_tensor(v2, [BN, 1, D], kv_layout)
        W = triton.cdiv(N, BN) * nseq * Hkv
        G = min(W, _GCAP_A if _GCAP_A > 0 else G_max)
        num_rounds = triton.cdiv(W, G)
        vec_layout = gl.NVMMASharedLayout(0, 32, 1)   # no swizzle, fp32, rank 1
        lse_desc = TensorDescriptor.from_tensor(lse2.view(-1), [BM], vec_layout)
        del_desc = TensorDescriptor.from_tensor(delta.view(-1), [BM], vec_layout)
        kern = _a_kernel[(G,)](q_desc, do_desc, k_desc, v_desc, lse_desc, del_desc, lse2, delta, dk2, dv2, cu, dbg if _TIMING else dbg,
                               sm_scale, sm_scale_log2e, N, T, dk2.stride(0), dk2.stride(1), W, num_rounds, nseq,
                               VARLEN=varlen, GROUP=H // Hkv, HKV=Hkv, BM=BM, BN=BN, D=D, STAGES=_STA, TIMING=_TIMING, HAZ=_HAZ, EARLY=_EARLY, num_warps=4)
        if _DEBUG:
            print("passA regs", kern.n_regs, "spills", kern.n_spills, "smem", kern.metadata.shared, "tmem", kern.metadata.tmem_size)
        if _SYNC:
            torch.cuda.synchronize()
    if _ONLY != "A":
      with torch.cuda.stream(_SIDE if _STREAMS else main_stream):
          q_layout = gl.NVMMASharedLayout.get_default_for([BMQ, 1, D], gl.bfloat16)
          kv_layout = gl.NVMMASharedLayout.get_default_for([BNQ, 1, D], gl.bfloat16)
          q_desc = TensorDescriptor.from_tensor(q2, [BMQ, 1, D], q_layout)
          do_desc = TensorDescriptor.from_tensor(do2, [BMQ, 1, D], q_layout)
          k_desc = TensorDescriptor.from_tensor(k2, [BNQ, 1, D], kv_layout)
          v_desc = TensorDescriptor.from_tensor(v2, [BNQ, 1, D], kv_layout)
          num_q_tiles = triton.cdiv(N, BMQ)
          W = num_q_tiles * nseq * H
          G = min(W, _GCAP_B if _GCAP_B > 0 else G_max)
          num_rounds = triton.cdiv(W, G)
          kern = _b_kernel[(G,)](q_desc, do_desc, k_desc, v_desc, lse2, delta, dq2, cu, dbg if _TIMING else dbg,
                                 sm_scale, sm_scale_log2e, N, T, dq2.stride(0), dq2.stride(1), W, num_rounds, nseq, num_q_tiles,
                                 VARLEN=varlen, GROUP=H // Hkv, H=H, BMQ=BMQ, BNQ=BNQ, D=D, STAGES=_STB, TIMING=_TIMING, SERIAL=_SERIAL, QCHK=_QCHK, SPIN=_SPIN, HAZ=_HAZ, EARLY=_EARLY, num_warps=4)
          if _DEBUG:
              print("passB regs", kern.n_regs, "spills", kern.n_spills, "smem", kern.metadata.shared, "tmem", kern.metadata.tmem_size)
    if _STREAMS:
        main_stream.wait_stream(_SIDE)
        for t_ in (dq, lse2, delta):
            t_.record_stream(_SIDE)
    return dq, dk, dv
