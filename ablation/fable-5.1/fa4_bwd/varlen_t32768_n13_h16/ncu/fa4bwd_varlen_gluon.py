import os
# From-scratch FlashAttention-style varlen causal MHA backward for Blackwell (sm_100), written in Gluon.
# v6: persistent + dynamic scheduling, 3-stage Q/dO TMA ring, double-buffered S/dP in TMEM,
#     P^T kept in TMEM (bf16) as the A operand of the dV MMA, dS^T double-buffered in smem,
#     two MMA-issuing warps (S/dP issuer and dV/dK/dQ issuer), dQ accumulated in fp16 via TMA reduce-add.
import math
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared, get_tmem_reg_layout)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language._core import builtin
from triton._C.libtriton import ir

LOG2E = 1.4426950408889634


@builtin
def tma_reduce_add(desc, coord, src, _semantic=None):
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(ir.DESCRIPTOR_REDUCE_KIND.ADD, desc.handle, coord, src.handle)


@gluon.jit
def _fetch_unit(counter_ptr):
    return gl.inline_asm_elementwise(
        "{\n.reg .pred p;\n.reg .u32 r;\nmov.u32 r, 0;\nelect.sync _|p, 0xffffffff;\n"
        "@p atom.global.add.u32 r, [$1], 1;\nredux.sync.max.u32 $0, r, 0xffffffff;\n}",
        "=r,l", [counter_ptr], dtype=gl.int32, is_pure=False, pack=1)


# ----------------------------------------------------------------------------------------------
# pre / post processing (plain Triton).  Padded layouts: each document starts at a multiple of BM.
# ----------------------------------------------------------------------------------------------
@triton.jit
def _preprocess_kernel(o_ptr, do_ptr, lse_ptr, aux_ptr, padrow_ptr, dqacc_ptr, counter_ptr, counter_init, n_rows, nz, T, TP,
                       H, log2e, LD: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr, ZBLOCK: tl.constexpr):
    """delta = rowsum(o * do) and lse2 = lse * log2e for ROWS consecutive (t, h) rows (fully contiguous reads of the
    (T, H, D) tensors), scattered into the per-document 64-aligned padded layout; plus a flat chunk of dq_acc zeroing."""
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(counter_ptr, counter_init)      # reset the persistent kernel's work counter (CTA i starts on unit i)
    r = pid * ROWS + tl.arange(0, ROWS)          # flat row index = t * H + h
    rmask = r < n_rows
    d = tl.arange(0, D)
    offs = r[:, None] * D + d[None, :]
    o = tl.load(o_ptr + offs, mask=rmask[:, None], other=0.0, cache_modifier=LD)
    do = tl.load(do_ptr + offs, mask=rmask[:, None], other=0.0, cache_modifier=LD)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    t = r // H
    h = r % H
    prow = tl.load(padrow_ptr + t, mask=rmask, other=0)
    tl.store(aux_ptr + h * (2 * TP) + TP + prow, delta, mask=rmask)
    lse = tl.load(lse_ptr + h * T + t, mask=rmask, other=0.0)
    tl.store(aux_ptr + h * (2 * TP) + prow, lse * log2e, mask=rmask)
    z = pid * ZBLOCK + tl.arange(0, ZBLOCK)
    tl.store(dqacc_ptr + z, tl.zeros([ZBLOCK], dtype=tl.float16), mask=z < nz, cache_modifier=".cs")


@triton.jit
def _postprocess_kernel(dqacc_ptr, dq_ptr, blk_t0_ptr, blk_n_ptr, T, TP, H, scale, D: tl.constexpr, BLOCK: tl.constexpr,
                        SUB: tl.constexpr):
    pid = tl.program_id(0)
    pb = pid // (BLOCK // SUB)
    sb = pid % (BLOCK // SUB)
    h = tl.program_id(1)
    t0 = tl.load(blk_t0_ptr + pb) + sb * SUB
    nv = tl.load(blk_n_ptr + pb) - sb * SUB
    r = tl.arange(0, SUB)
    d = tl.arange(0, D)
    t = t0 + r
    mask = r < nv
    prow = pb * BLOCK + sb * SUB + r
    NPB = tl.num_programs(0) // (BLOCK // SUB)
    acc = tl.load(dqacc_ptr + (h * NPB + pb) * (D * BLOCK) + d[:, None] * BLOCK + (sb * SUB + r)[None, :]).to(tl.float32)
    accT = tl.trans(acc)
    tl.store(dq_ptr + ((t[:, None] * H + h) * D + d[None, :]), (accT * scale).to(tl.bfloat16), mask=mask[:, None])


# ----------------------------------------------------------------------------------------------
# work-unit decode: work table rows are (doc b, n-block j, head h, cost)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _decode_unit(u, work_ptr, cu_ptr, pad_ptr, BM: gl.constexpr, BN: gl.constexpr):
    b = gl.load(work_ptr + u * 4)
    j = gl.load(work_ptr + u * 4 + 1)
    h = gl.load(work_ptr + u * 4 + 2)
    q_start = gl.load(cu_ptr + b)
    q_end = gl.load(cu_ptr + b + 1)
    p_start = gl.load(pad_ptr + b)
    L = q_end - q_start
    kv_row = q_start + j * BN
    i_begin = j * (BN // BM)
    i_end = (L + BM - 1) // BM
    return h, q_start, p_start, L, kv_row, i_begin, i_end



@gluon.jit
def _lane(vec, idx, i: gl.constexpr):
    return gl.max(gl.where(idx == i, vec, -2147483647), axis=0)


@gluon.jit
def _unit_from_smem(unit_smem, lay: gl.constexpr):
    vec = unit_smem.load(lay)
    idx = gl.arange(0, 32, layout=lay)
    return (_lane(vec, idx, 0), _lane(vec, idx, 1), _lane(vec, idx, 2), _lane(vec, idx, 3), _lane(vec, idx, 4),
            _lane(vec, idx, 5), _lane(vec, idx, 6), _lane(vec, idx, 7))


@gluon.jit
def _pack_unit(idx, u, h, q_start, p_start, L, kv_row, i_begin, i_end):
    return gl.where(idx == 0, u, gl.where(idx == 1, h, gl.where(idx == 2, q_start, gl.where(idx == 3, p_start,
           gl.where(idx == 4, L, gl.where(idx == 5, kv_row, gl.where(idx == 6, i_begin,
           gl.where(idx == 7, i_end, 0))))))))


@gluon.jit
def _tc_fence_before():
    """tcgen05.fence::before_thread_sync: order prior tcgen05 ops before a following thread sync (arrive)."""
    gl.inline_asm_elementwise("mov.u32 $0, 0;\ntcgen05.fence::before_thread_sync;", "=r", [], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _tc_fence_after():
    """tcgen05.fence::after_thread_sync: order tcgen05 ops after a preceding thread sync (wait)."""
    gl.inline_asm_elementwise("mov.u32 $0, 0;\ntcgen05.fence::after_thread_sync;", "=r", [], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _clk64():
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=gl.int64, is_pure=False, pack=1)


@gluon.jit
def _spin(cycles):
    t0 = _clk64()
    t = t0
    while t - t0 < cycles:
        t = _clk64()


@gluon.jit
def _p_view(s_tmem, b, half, BM: gl.constexpr, BN: gl.constexpr):
    """bf16 (packed) [BN, BM/2] view of P^T[:, half*BM/2 : (half+1)*BM/2], aliasing the S slot of buffer b.
    Softmax partition `half` loads S columns [half*BM/2, (half+1)*BM/2) (32-bit columns) and stores its P^T half
    (packed, BM/4 32-bit columns) at the start of that same region, so the two partitions never touch each other's
    unread S columns."""
    wide_tl: gl.constexpr = TensorMemoryLayout((BN, 2 * BM), col_stride=1)
    return s_tmem.index(b)._reinterpret(gl.bfloat16, [BN, 2 * BM], wide_tl).slice(half * BM, BM // 2)


# ----------------------------------------------------------------------------------------------
# partitions
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _load_partition(q_desc, k_desc, v_desc, do_desc, aux_desc,
                    kv_smem4, q_smem3, do_smem3, lse_smem, unit_smem,
                    unit_ready, unit_free, k_ready, v_ready, k_free, v_free, qdo_ready, qdo_free,
                    counter_ptr, work_ptr, cu_ptr, pad_ptr, num_units, TP,
                    BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr):
    g = 0
    n = 0
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    idx = gl.arange(0, 32, layout=lay)
    u = gl.program_id(0)                       # CTA i starts on unit i; the rest is dynamic
    h, q_start, p_start, L, kv_row, i_begin, i_end = _decode_unit(gl.minimum(u, num_units - 1), work_ptr, cu_ptr,
                                                                  pad_ptr, BM, BN)
    unit_smem.store(_pack_unit(idx, u, h, q_start, p_start, L, kv_row, i_begin, i_end))
    mbarrier.arrive(unit_ready, count=1)
    while u < num_units:
        # K(n) goes into the buffer that held V(n-1): free as soon as the previous unit's last dP^T MMA has
        # completed (committed by warp A), i.e. well before that unit's last dQ^T MMA.
        kb = n % 2
        if n > 0:
            mbarrier.wait(v_free, (n - 1) & 1)
        mbarrier.expect(k_ready, BN * D * 2)
        tma.async_copy_global_to_shared(k_desc, [kv_row, h, 0], k_ready, kv_smem4.index(kb))
        for i in range(i_begin, i_end):
            s = g % STAGES
            k = g // STAGES
            mbarrier.wait(qdo_free.index(s), (k & 1) ^ 1)
            mbarrier.expect(qdo_ready.index(s), 2 * BM * D * 2 + 2 * BM * 4)
            row = q_start + i * BM
            tma.async_copy_global_to_shared(q_desc, [row, h, 0], qdo_ready.index(s), q_smem3.index(s))
            tma.async_copy_global_to_shared(do_desc, [row, h, 0], qdo_ready.index(s), do_smem3.index(s))
            tma.async_copy_global_to_shared(aux_desc, [2 * h, p_start + i * BM], qdo_ready.index(s), lse_smem.index(s))
            g += 1
            if i == i_begin:
                # V(n) goes into the buffer that held K(n-1): free once the previous unit's last dQ^T MMA has
                # completed (committed by warp B)
                if n > 0:
                    mbarrier.wait(k_free, (n - 1) & 1)
                mbarrier.expect(v_ready, BN * D * 2)
                tma.async_copy_global_to_shared(v_desc, [kv_row, h, 0], v_ready, kv_smem4.index(1 - kb))
        n += 1
        u = _fetch_unit(counter_ptr)
        h, q_start, p_start, L, kv_row, i_begin, i_end = _decode_unit(gl.minimum(u, num_units - 1), work_ptr,
                                                                      cu_ptr, pad_ptr, BM, BN)
        mbarrier.wait(unit_free, (n & 1) ^ 1)
        unit_smem.store(_pack_unit(idx, u, h, q_start, p_start, L, kv_row, i_begin, i_end))
        mbarrier.arrive(unit_ready, count=1)


@gluon.jit
def _sdp_partition(kv_smem4, q_smem3, do_smem3, unit_smem, s_tmem, dp_tmem,
                   unit_ready, unit_free, k_ready, v_ready, v_free, qdo_ready, sdp_ready, sdp_free, p_free, dq_free,
                   work_ptr, cu_ptr, pad_ptr, num_units,
                   BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr):
    """Warp A: issues S^T = K Q^T and dP^T = V dO^T for every tile, as far ahead as the buffers allow."""
    g = 0
    n = 0
    lay: gl.constexpr = gl.BlockedLayout([32], [32], [1], [0])
    mbarrier.wait(unit_ready, 0)
    u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
    mbarrier.arrive(unit_free, count=1)
    while u < num_units:
        n_it = i_end - i_begin
        kb = n % 2
        k_s = kv_smem4.index(kb).reshape([BN, D])
        v_s = kv_smem4.index(1 - kb).reshape([BN, D])
        mbarrier.wait(k_ready, n & 1)
        for lt in range(0, n_it):
            gg = g + lt
            s = gg % STAGES
            k = gg // STAGES
            b = gg % 2
            k2 = gg // 2
            # last dQ tile written into the S slot of buffer b before S(gg).  dQ(t) lives in buffer (t+1)%2;
            # within a unit dQ(gg-1) is only issued after the softmax loaded S(gg), so it is dQ(gg-3); for the
            # first tile of a unit it is the previous unit's last dQ (a real wait now that K/V are reloaded early).
            last_dq = gg - 3
            if lt == 0:
                last_dq = gg - 1
            mbarrier.wait(qdo_ready.index(s), k & 1)
            mbarrier.wait(sdp_free.index(b), (k2 & 1) ^ 1)   # softmax loaded S/dP(gg-2)
            mbarrier.wait(p_free.index(b), (k2 & 1) ^ 1)     # dV(gg-2) done reading P^T(gg-2) (S slot)
            _tc_fence_after()
            tcgen05_mma(k_s, q_smem3.index(s).reshape([BM, D]).permute((1, 0)), s_tmem.index(b), use_acc=False)
            if lt == 0:
                mbarrier.wait(v_ready, n & 1)
            if last_dq >= 0:
                mbarrier.wait(dq_free.index(b), (last_dq // 2) & 1)   # dQ(last_dq) drained from the dP slot
            _tc_fence_after()
            tcgen05_mma(v_s, do_smem3.index(s).reshape([BM, D]).permute((1, 0)), dp_tmem.index(b), use_acc=False)
            tcgen05_commit(sdp_ready.index(b))
            if lt + 1 == n_it:
                tcgen05_commit(v_free)      # V is no longer read by this unit
        g += n_it
        n += 1
        mbarrier.wait(unit_ready, n & 1)
        u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
        mbarrier.arrive(unit_free, count=1)


@gluon.jit
def _dkv_partition(q_smem3, do_smem3, ds_smem, unit_smem, s_tmem, dv_tmem, dk_tmem,
                   unit_ready, unit_free, qdo_free, p_ready, p_free, ds_ready, ds_free,
                   dv_ready, dkdv_ready, dkdv_free,
                   work_ptr, cu_ptr, pad_ptr, num_units,
                   BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr):
    """Warp B1: issues dV += P^T dO and dK += dS^T Q (dQ is issued by warp B2 so that this stream never waits
    for the softmax of the next tile)."""
    g = 0
    n = 0
    lay: gl.constexpr = gl.BlockedLayout([32], [32], [1], [0])
    mbarrier.wait(unit_ready, 0)
    u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
    mbarrier.arrive(unit_free, count=1)
    while u < num_units:
        n_it = i_end - i_begin
        mbarrier.wait(dkdv_free, (n & 1) ^ 1)
        _tc_fence_after()
        for lt in range(0, n_it):
            gg = g + lt
            s = gg % STAGES
            b = gg % 2
            k2 = gg // 2
            q_s = q_smem3.index(s).reshape([BM, D])
            do_s = do_smem3.index(s).reshape([BM, D])
            use_acc = lt > 0
            # dV += P^T(gg) dO(gg): P^T lives (bf16, packed) in the S slot of buffer b
            mbarrier.wait(p_ready.index(b), k2 & 1)
            _tc_fence_after()
            tcgen05_mma(_p_view(s_tmem, b, 0, BM, BN), do_s.slice(0, BM // 2, dim=0), dv_tmem, use_acc=use_acc)
            tcgen05_mma(_p_view(s_tmem, b, 1, BM, BN), do_s.slice(BM // 2, BM // 2, dim=0), dv_tmem, use_acc=True)
            tcgen05_commit(p_free.index(b))
            if lt + 1 == n_it:
                tcgen05_commit(dv_ready)        # dV of this unit is final
            # dK += dS^T(gg) Q(gg)
            mbarrier.wait(ds_ready.index(b), k2 & 1)
            _tc_fence_after()
            tcgen05_mma(ds_smem.index(b), q_s, dk_tmem, use_acc=use_acc)
            tcgen05_commit(qdo_free.index(s))
            tcgen05_commit(ds_free.index(b))    # (1 of 2 arrivals) dK is done reading dS^T(gg)
            if lt + 1 == n_it:
                tcgen05_commit(dkdv_ready)      # dK of this unit is final
        g += n_it
        n += 1
        mbarrier.wait(unit_ready, n & 1)
        u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
        mbarrier.arrive(unit_free, count=1)


@gluon.jit
def _dq_partition(kv_smem4, ds_smem, unit_smem, dp_tmem,
                  unit_ready, unit_free, sdp_free, ds_ready, ds_free, dq_ready, dq_free, k_free,
                  work_ptr, cu_ptr, pad_ptr, num_units,
                  BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr):
    """Warp B2: issues dQ^T(gg) = K^T dS^T(gg) into the dP slot of buffer (gg+1)%2."""
    g = 0
    n = 0
    lay: gl.constexpr = gl.BlockedLayout([32], [32], [1], [0])
    mbarrier.wait(unit_ready, 0)
    u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
    mbarrier.arrive(unit_free, count=1)
    while u < num_units:
        n_it = i_end - i_begin
        kT = kv_smem4.index(n % 2).reshape([BN, D]).permute((1, 0))  # [D, BN]
        for lt in range(0, n_it):
            gg = g + lt
            b = gg % 2
            k2 = gg // 2
            nb = (gg + 1) % 2
            mbarrier.wait(ds_ready.index(b), k2 & 1)
            # the dP slot of buffer nb is free once the softmax loaded S/dP(gg+1); for the last tile of the unit
            # it is free once the epilogue drained dQ(gg-2)
            if lt + 1 < n_it:
                mbarrier.wait(sdp_free.index(nb), ((gg + 1) // 2) & 1)
            elif gg >= 2:
                mbarrier.wait(dq_free.index(nb), ((gg - 2) // 2) & 1)
            _tc_fence_after()
            tcgen05_mma(kT, ds_smem.index(b), dp_tmem.index(nb), use_acc=False)
            tcgen05_commit(ds_free.index(b))    # (2 of 2 arrivals) dQ is done reading dS^T(gg)
            tcgen05_commit(dq_ready.index(nb))
        tcgen05_commit(k_free)          # K is no longer read by this unit (A's S MMAs all preceded the last dQ)
        g += n_it
        n += 1
        mbarrier.wait(unit_ready, n & 1)
        u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
        mbarrier.arrive(unit_free, count=1)


@gluon.jit
def _softmax_iter(g, i, s_tmem, dp_tmem, ds_smem, lse_smem,
                  qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free,
                  key_idx, cols, L, scale_log2e, MODE: gl.constexpr, PFIRST: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, PCOL: gl.constexpr, P: gl.constexpr,
                  reg_l: gl.constexpr, col_l: gl.constexpr, p_reg_l: gl.constexpr):
    s = g % STAGES
    k = g // STAGES
    b = g % 2
    k2 = g // 2
    mbarrier.wait(qdo_ready.index(s), k & 1)
    flat_l: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    aux = lse_smem.index(s)._reinterpret(gl.float32, [2 * BM], flat_l)
    lse = aux.slice(P * PCOL, PCOL).load(col_l)
    delta = aux.slice(BM + P * PCOL, PCOL).load(col_l)
    mbarrier.wait(sdp_ready.index(b), k2 & 1)
    _tc_fence_after()
    sv = s_tmem.index(b).slice(P * PCOL, PCOL).load(reg_l)
    dpv = dp_tmem.index(b).slice(P * PCOL, PCOL).load(reg_l)
    _tc_fence_before()
    mbarrier.arrive(sdp_free.index(b), count=1)
    p = gl.exp2(sv * scale_log2e - gl.expand_dims(lse, 0))
    if MODE == 1:
        # causal mask only (the two diagonal tiles of a unit that is not also its last tile)
        q_idx = cols + i * BM
        p = gl.where(gl.expand_dims(q_idx, 0) >= gl.expand_dims(key_idx, 1), p, 0.0)
    if MODE == 2:
        q_idx = cols + i * BM
        valid = (gl.expand_dims(q_idx, 0) >= gl.expand_dims(key_idx, 1)) & gl.expand_dims(q_idx < L, 0)
        p = gl.where(valid, p, 0.0)
    ds = p * (dpv - gl.expand_dims(delta, 0))
    dsb = ds.to(gl.bfloat16)
    if PFIRST:
        # first tiles of a unit: P^T first, the dV MMA of the new unit is what the boundary chain waits for
        pb = gl.convert_layout(p.to(gl.bfloat16), p_reg_l)
        _p_view(s_tmem, b, P, BM, BN).store(pb)
        _tc_fence_before()
        mbarrier.arrive(p_ready.index(b), count=1)
    # dS^T: it feeds dK and dQ, and the dQ -> drain -> next S/dP chain is the steady-state critical loop
    mbarrier.wait(ds_free.index(b), (k2 & 1) ^ 1)
    ds_smem.index(b).slice(P * PCOL, PCOL, dim=1).store(dsb)
    fence_async_shared()
    mbarrier.arrive(ds_ready.index(b), count=1)
    if not PFIRST:
        # P^T (bf16, packed) goes back into the S slot of buffer b: it is the A operand of the dV MMA
        pb = gl.convert_layout(p.to(gl.bfloat16), p_reg_l)
        _p_view(s_tmem, b, P, BM, BN).store(pb)
        _tc_fence_before()
        mbarrier.arrive(p_ready.index(b), count=1)


@gluon.jit
def _softmax_partition(s_tmem, dp_tmem, ds_smem, lse_smem, unit_smem,
                       unit_ready, unit_free, qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free,
                       work_ptr, cu_ptr, pad_ptr, num_units, scale_log2e,
                       dk_desc, dk_tmem, dk_stage, dkdv_ready, dkdv_free, dk_ptr, scale, H,
                       P: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr,
                       num_warps: gl.constexpr):
    PCOL: gl.constexpr = BM // 2
    SL: gl.constexpr = 32
    sl_tl: gl.constexpr = TensorMemoryLayout((BN, SL), col_stride=1)
    sl_reg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, SL), sl_tl, num_warps)
    sl_rows = gl.arange(0, BN, layout=gl.SliceLayout(1, sl_reg_l))
    sl_cols = gl.arange(0, SL, layout=gl.SliceLayout(0, sl_reg_l))
    s_tl: gl.constexpr = TensorMemoryLayout((BN, PCOL), col_stride=1)
    reg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, PCOL), s_tl, num_warps)
    row_l: gl.constexpr = gl.SliceLayout(1, reg_l)
    col_l: gl.constexpr = gl.SliceLayout(0, reg_l)
    p_tl: gl.constexpr = TensorMemoryLayout((BN, PCOL), col_stride=1)
    p_reg_l: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, (BN, PCOL), p_tl, num_warps)
    rows = gl.arange(0, BN, layout=row_l)
    cols = gl.arange(0, PCOL, layout=col_l) + P * PCOL
    lay: gl.constexpr = gl.BlockedLayout([32], [32], [num_warps], [0])
    g = 0
    n = 0
    mbarrier.wait(unit_ready, 0)
    u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
    mbarrier.arrive(unit_free, count=1)
    while u < num_units:
        n_it = i_end - i_begin
        key_idx = rows + (kv_row - q_start)
        n_head = gl.minimum(2, n_it)
        tail = 0
        if L % BM != 0:
            tail = 1
        mid_end = gl.maximum(n_head, n_it - tail)
        if n_it > 2:
            # diagonal tiles that are not the unit's last tile: causal mask only
            for lt in range(0, n_head):
                _softmax_iter(g + lt, i_begin + lt, s_tmem, dp_tmem, ds_smem, lse_smem,
                              qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free, key_idx, cols, L,
                              scale_log2e, 1, True, BM, BN, STAGES, PCOL, P, reg_l, col_l, p_reg_l)
        else:
            for lt in range(0, n_head):
                _softmax_iter(g + lt, i_begin + lt, s_tmem, dp_tmem, ds_smem, lse_smem,
                              qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free, key_idx, cols, L,
                              scale_log2e, 2, True, BM, BN, STAGES, PCOL, P, reg_l, col_l, p_reg_l)
        for lt in range(n_head, mid_end):
            _softmax_iter(g + lt, i_begin + lt, s_tmem, dp_tmem, ds_smem, lse_smem,
                          qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free, key_idx, cols, L, scale_log2e,
                          0, False, BM, BN, STAGES, PCOL, P, reg_l, col_l, p_reg_l)
        for lt in range(mid_end, n_it):
            _softmax_iter(g + lt, i_begin + lt, s_tmem, dp_tmem, ds_smem, lse_smem,
                          qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free, key_idx, cols, L, scale_log2e,
                          2, False, BM, BN, STAGES, PCOL, P, reg_l, col_l, p_reg_l)
        if P == 1:
            # this partition is idle until the next unit's S arrives: drain dK (the epilogue drains dV)
            mbarrier.wait(dkdv_ready, n & 1)
            _tc_fence_after()
            if L - (kv_row - q_start) >= BN:
                for c in gl.static_range(D // SL // 2):
                    dk0 = dk_tmem.slice((2 * c) * SL, SL).load(sl_reg_l)
                    dk1 = dk_tmem.slice((2 * c + 1) * SL, SL).load(sl_reg_l)
                    tma.store_wait(1)
                    dk_stage.index(0).store((dk0 * scale).to(gl.bfloat16))
                    fence_async_shared()
                    tma.async_copy_shared_to_global(dk_desc, [kv_row, h * D + (2 * c) * SL], dk_stage.index(0))
                    tma.store_wait(1)
                    dk_stage.index(1).store((dk1 * scale).to(gl.bfloat16))
                    fence_async_shared()
                    tma.async_copy_shared_to_global(dk_desc, [kv_row, h * D + (2 * c + 1) * SL], dk_stage.index(1))
            else:
                rmask = (sl_rows + (kv_row - q_start)) < L
                base = ((kv_row + sl_rows) * H + h) * D
                for c in gl.static_range(D // SL):
                    dk = dk_tmem.slice(c * SL, SL).load(sl_reg_l)
                    gl.store(dk_ptr + gl.expand_dims(base, 1) + c * SL + gl.expand_dims(sl_cols, 0),
                             (dk * scale).to(gl.bfloat16), mask=gl.expand_dims(rmask, 1))
            _tc_fence_before()
            mbarrier.arrive(dkdv_free, count=1)
        g += n_it
        n += 1
        mbarrier.wait(unit_ready, n & 1)
        u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
        mbarrier.arrive(unit_free, count=1)
    if P == 1:
        tma.store_wait(0)


@gluon.jit
def _epilogue_partition(dq_desc, dk_desc, dv_desc, dp_tmem, dv_tmem, dk_tmem, dq_smem, unit_smem,
                        unit_ready, unit_free, dq_ready, dq_free, dv_ready, dkdv_free,
                        dk_ptr, dv_ptr, work_ptr, cu_ptr, pad_ptr, num_units, H, NPB, scale,
                        BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, num_warps: gl.constexpr):
    HB: gl.constexpr = BM // 2
    dqh_tl: gl.constexpr = TensorMemoryLayout((D, HB), col_stride=1)
    dqh_reg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (D, HB), dqh_tl, num_warps)
    SL: gl.constexpr = 32
    sl_tl: gl.constexpr = TensorMemoryLayout((BN, SL), col_stride=1)
    sl_reg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, SL), sl_tl, num_warps)
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, sl_reg_l))
    cols = gl.arange(0, SL, layout=gl.SliceLayout(0, sl_reg_l))
    lay: gl.constexpr = gl.BlockedLayout([32], [32], [num_warps], [0])
    # dK/dV staging for TMA stores: the two dq_smem halves reinterpreted as bf16 [BN, SL] tiles (same swizzle)
    st_smem = dq_smem._reinterpret(gl.bfloat16, [2, BN, SL], dk_desc.layout)
    g = 0
    n = 0
    mbarrier.wait(unit_ready, 0)
    u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
    mbarrier.arrive(unit_free, count=1)
    while u < num_units:
        n_it = i_end - i_begin
        for lt in range(0, n_it):
            gg = g + lt
            nb = (gg + 1) % 2
            k2 = gg // 2
            mbarrier.wait(dq_ready.index(nb), k2 & 1)
            _tc_fence_after()
            dq_lo = dp_tmem.index(nb).slice(0, HB).load(dqh_reg_l).to(gl.float16)  # [D, HB]
            dq_hi = dp_tmem.index(nb).slice(HB, HB).load(dqh_reg_l).to(gl.float16)
            _tc_fence_before()
            mbarrier.arrive(dq_free.index(nb), count=1)
            if lt + 1 == n_it:
                # last tile of the unit (its dQ slot is already released): dV is final -> drain it now (dK is
                # drained by softmax partition 1) so that warp B can start the next unit's dV MMAs early
                mbarrier.wait(dv_ready, n & 1)
                _tc_fence_after()
                if L - (kv_row - q_start) >= BN:
                    for c in gl.static_range(D // SL // 2):
                        dv0 = dv_tmem.slice((2 * c) * SL, SL).load(sl_reg_l)
                        dv1 = dv_tmem.slice((2 * c + 1) * SL, SL).load(sl_reg_l)
                        tma.store_wait(1)
                        st_smem.index(0).store(dv0.to(gl.bfloat16))
                        fence_async_shared()
                        tma.async_copy_shared_to_global(dv_desc, [kv_row, h * D + (2 * c) * SL], st_smem.index(0))
                        tma.store_wait(1)
                        st_smem.index(1).store(dv1.to(gl.bfloat16))
                        fence_async_shared()
                        tma.async_copy_shared_to_global(dv_desc, [kv_row, h * D + (2 * c + 1) * SL], st_smem.index(1))
                else:
                    key_idx = rows + (kv_row - q_start)
                    rmask = key_idx < L
                    base = ((kv_row + rows) * H + h) * D
                    for c in gl.static_range(D // SL):
                        dv = dv_tmem.slice(c * SL, SL).load(sl_reg_l)
                        gl.store(dv_ptr + gl.expand_dims(base, 1) + c * SL + gl.expand_dims(cols, 0),
                                 dv.to(gl.bfloat16), mask=gl.expand_dims(rmask, 1))
                _tc_fence_before()
                mbarrier.arrive(dkdv_free, count=1)
            blk = h * NPB + (p_start + (i_begin + lt) * BM) // BM      # block-major dq_acc block index
            tma.store_wait(1)
            dq_smem.index(0).reshape([D, HB]).store(dq_lo)
            fence_async_shared()
            tma_reduce_add(dq_desc, [blk, 0, 0], dq_smem.index(0))
            tma.store_wait(1)
            dq_smem.index(1).reshape([D, HB]).store(dq_hi)
            fence_async_shared()
            tma_reduce_add(dq_desc, [blk, 0, HB], dq_smem.index(1))
        g += n_it
        n += 1
        mbarrier.wait(unit_ready, n & 1)
        u, h, q_start, p_start, L, kv_row, i_begin, i_end = _unit_from_smem(unit_smem, lay)
        mbarrier.arrive(unit_free, count=1)
    tma.store_wait(0)


@gluon.jit
def _bwd_kernel(q_desc, k_desc, v_desc, do_desc, aux_desc, dq_desc, dk_desc, dv_desc,
                counter_ptr, work_ptr, cu_ptr, pad_ptr, dk_ptr, dv_ptr, num_units, TP, H, NPB, scale_log2e, scale,
                BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, STAGES: gl.constexpr, num_warps: gl.constexpr):
    # ---- shared memory ----
    kv_l: gl.constexpr = k_desc.layout
    q_l: gl.constexpr = q_desc.layout
    kv_smem4 = gl.allocate_shared_memory(gl.bfloat16, [2, BN, 1, D], kv_l)   # K/V, roles alternate per unit
    q_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, 1, D], q_l)
    do_smem3 = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BM, 1, D], q_l)
    ds_l: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BM], gl.bfloat16)
    ds_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BN, BM], ds_l)
    dq_smem = gl.allocate_shared_memory(gl.float16, [2, 1, D, BM // 2], dq_desc.layout)
    dk_stage = gl.allocate_shared_memory(gl.bfloat16, [2, BN, 32], dk_desc.layout)
    lse_smem = gl.allocate_shared_memory(gl.float32, [STAGES, 2, BM], aux_desc.layout)   # [lse2; delta] per stage
    unit_smem = gl.allocate_shared_memory(gl.int32, [32], gl.SwizzledSharedLayout(1, 1, 1, [0]))

    # ---- barriers ----
    unit_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    unit_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    k_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    v_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    k_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    v_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    qdo_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    qdo_free = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    sdp_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    sdp_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    p_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    ds_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    ds_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    dq_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    dq_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    dkdv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dkdv_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(unit_ready, count=1)
    mbarrier.init(unit_free, count=6)
    mbarrier.init(k_ready, count=1)
    mbarrier.init(v_ready, count=1)
    mbarrier.init(k_free, count=1)
    mbarrier.init(v_free, count=1)
    for s in gl.static_range(STAGES):
        mbarrier.init(qdo_ready.index(s), count=1)
        mbarrier.init(qdo_free.index(s), count=1)
    for b in gl.static_range(2):
        mbarrier.init(sdp_ready.index(b), count=1)
        mbarrier.init(sdp_free.index(b), count=2)
        mbarrier.init(p_ready.index(b), count=2)
        mbarrier.init(p_free.index(b), count=1)
        mbarrier.init(ds_ready.index(b), count=2)
        mbarrier.init(ds_free.index(b), count=2)
        mbarrier.init(dq_ready.index(b), count=1)
        mbarrier.init(dq_free.index(b), count=1)
    mbarrier.init(dkdv_ready, count=1)
    mbarrier.init(dv_ready, count=1)
    mbarrier.init(dkdv_free, count=2)
    fence_async_shared()

    # ---- tensor memory: S[2] dP[2] (4 x 64 cols) + dV + dK (2 x 128) = 512 columns ----
    s_tl: gl.constexpr = TensorMemoryLayout((BN, BM), col_stride=1)
    acc_tl: gl.constexpr = TensorMemoryLayout((BN, D), col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [2, BN, BM], s_tl)
    dp_tmem = allocate_tensor_memory(gl.float32, [2, BN, BM], s_tl)
    dv_tmem = allocate_tensor_memory(gl.float32, [BN, D], acc_tl)
    dk_tmem = allocate_tensor_memory(gl.float32, [BN, D], acc_tl)

    gl.warp_specialize([
        (_softmax_partition, (s_tmem, dp_tmem, ds_smem, lse_smem, unit_smem,
                              unit_ready, unit_free, qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free,
                              work_ptr, cu_ptr, pad_ptr, num_units, scale_log2e,
                              dk_desc, dk_tmem, dk_stage, dkdv_ready, dkdv_free, dk_ptr, scale, H,
                              0, BM, BN, D, STAGES, num_warps)),
        (_softmax_partition, (s_tmem, dp_tmem, ds_smem, lse_smem, unit_smem,
                              unit_ready, unit_free, qdo_ready, sdp_ready, sdp_free, p_ready, ds_ready, ds_free,
                              work_ptr, cu_ptr, pad_ptr, num_units, scale_log2e,
                              dk_desc, dk_tmem, dk_stage, dkdv_ready, dkdv_free, dk_ptr, scale, H,
                              1, BM, BN, D, STAGES, 4)),
        (_epilogue_partition, (dq_desc, dk_desc, dv_desc, dp_tmem, dv_tmem, dk_tmem, dq_smem, unit_smem,
                               unit_ready, unit_free, dq_ready, dq_free, dv_ready, dkdv_free,
                               dk_ptr, dv_ptr, work_ptr, cu_ptr, pad_ptr, num_units, H, NPB, scale, BM, BN, D, 4)),
        (_load_partition, (q_desc, k_desc, v_desc, do_desc, aux_desc,
                           kv_smem4, q_smem3, do_smem3, lse_smem, unit_smem,
                           unit_ready, unit_free, k_ready, v_ready, k_free, v_free, qdo_ready, qdo_free,
                           counter_ptr, work_ptr, cu_ptr, pad_ptr, num_units, TP, BM, BN, D, STAGES)),
        (_sdp_partition, (kv_smem4, q_smem3, do_smem3, unit_smem, s_tmem, dp_tmem,
                          unit_ready, unit_free, k_ready, v_ready, v_free, qdo_ready, sdp_ready, sdp_free, p_free, dq_free,
                          work_ptr, cu_ptr, pad_ptr, num_units, BM, BN, D, STAGES)),
        (_dkv_partition, (q_smem3, do_smem3, ds_smem, unit_smem, s_tmem, dv_tmem, dk_tmem,
                          unit_ready, unit_free, qdo_free, p_ready, p_free, ds_ready, ds_free,
                          dv_ready, dkdv_ready, dkdv_free,
                          work_ptr, cu_ptr, pad_ptr, num_units, BM, BN, D, STAGES)),
        (_dq_partition, (kv_smem4, ds_smem, unit_smem, dp_tmem,
                         unit_ready, unit_free, sdp_free, ds_ready, ds_free, dq_ready, dq_free, k_free,
                         work_ptr, cu_ptr, pad_ptr, num_units, BM, BN, D, STAGES)),
    ], [4, 4, 1, 1, 1, 1], [128, 128, 64, 64, 64, 64])


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------
class VarlenBwd:
    BM, BN, D, STAGES = 64, 128, 128, 3

    def __init__(self, cu_seqlens_cpu, H, T, device, num_ctas=None):
        BM, BN = self.BM, self.BN
        self.T, self.H = T, H
        cu = list(cu_seqlens_cpu)
        units = []
        pad_starts = []
        blk_t0, blk_n = [], []
        p = 0
        for b in range(len(cu) - 1):
            L = cu[b + 1] - cu[b]
            n_blocks = (L + BN - 1) // BN
            i_end = (L + BM - 1) // BM
            pad_starts.append(p)
            for i in range(i_end):
                blk_t0.append(cu[b] + i * BM)
                blk_n.append(min(BM, L - i * BM))
            p += i_end * BM
            for h in range(H):
                for j in range(n_blocks):
                    units.append((L, h, j, b, i_end - j * (BN // BM)))
        self.TP = p
        self.num_pblk = p // BM
        # locality-friendly order: longest docs first, then head, then n-block ascending (cost descending);
        # the short documents at the end of the schedule are ordered by cost (LPT) to shrink the tail
        self.PRE_CFG = tuple(int(v) for v in os.environ.get("PRE_CFG", "8,1024,1").split(","))
        self.POST_CFG = tuple(int(v) for v in os.environ.get("POST_CFG", "64,8").split(","))
        self.PRE_LD = os.environ.get("PRE_LD", "")
        lpt = int(os.environ.get("TAIL_LPT", "0"))
        units.sort(key=lambda x: (0, -x[0], x[1], x[2]) if x[0] >= lpt else (1, -x[4], x[1], x[2]))
        self.work = torch.tensor([[b, j, h, c] for (L, h, j, b, c) in units], dtype=torch.int32, device=device)
        self.num_units = len(units)
        self.cu = torch.tensor(cu, dtype=torch.int32, device=device)
        self.pad = torch.tensor(pad_starts, dtype=torch.int32, device=device)
        self.blk_t0 = torch.tensor(blk_t0, dtype=torch.int32, device=device)
        self.blk_n = torch.tensor(blk_n, dtype=torch.int32, device=device)
        # aux[h, 0, :] = lse * log2e, aux[h, 1, :] = delta (padded per-document layout; padding rows stay 0)
        self.aux = torch.zeros(H, 2, self.TP, dtype=torch.float32, device=device)
        padrow = torch.empty(T, dtype=torch.int32)
        for b in range(len(cu) - 1):
            padrow[cu[b]:cu[b + 1]] = torch.arange(cu[b + 1] - cu[b], dtype=torch.int32) + pad_starts[b]
        self.padrow = padrow.to(device)
        self.dq_acc = torch.empty(H * self.num_pblk, self.D, BM, dtype=torch.float16, device=device)  # block-major
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_ctas = num_ctas or torch.cuda.get_device_properties(device).multi_processor_count

    def __call__(self, q, k, v, o, do, lse, scale):
        T, H, D = q.shape
        BM, BN, STAGES, TP = self.BM, self.BN, self.STAGES, self.TP
        assert T == self.T and H == self.H and D == self.D
        dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
        n_rows = T * H
        nz = self.dq_acc.numel()
        R_, Z_, W_ = self.PRE_CFG
        grid_pre = max(triton.cdiv(n_rows, R_), triton.cdiv(nz, Z_))
        _preprocess_kernel[(grid_pre,)](o, do, lse, self.aux, self.padrow, self.dq_acc, self.counter, self.num_ctas,
                                        n_rows, nz, T, TP, H, LOG2E, LD=self.PRE_LD, D=D, ROWS=R_, ZBLOCK=Z_, num_warps=W_)
        kv_l = gl.NVMMASharedLayout.get_default_for([BN, 1, D], gl.bfloat16)
        q_l = gl.NVMMASharedLayout.get_default_for([BM, 1, D], gl.bfloat16)
        q_desc = TensorDescriptor.from_tensor(q, [BM, 1, D], q_l)
        do_desc = TensorDescriptor.from_tensor(do, [BM, 1, D], q_l)
        k_desc = TensorDescriptor.from_tensor(k, [BN, 1, D], kv_l)
        v_desc = TensorDescriptor.from_tensor(v, [BN, 1, D], kv_l)
        vec_l = gl.NVMMASharedLayout(0, 32, 2)
        aux_desc = TensorDescriptor.from_tensor(self.aux.view(H * 2, TP), [2, BM], vec_l)
        dq_l = gl.NVMMASharedLayout.get_default_for([1, D, BM // 2], gl.float16)
        dq_desc = TensorDescriptor.from_tensor(self.dq_acc, [1, D, BM // 2], dq_l)
        dkv_l = gl.NVMMASharedLayout.get_default_for([BN, 32], gl.bfloat16)
        dk_desc = TensorDescriptor.from_tensor(dk.view(T, H * D), [BN, 32], dkv_l)
        dv_desc = TensorDescriptor.from_tensor(dv.view(T, H * D), [BN, 32], dkv_l)
        _bwd_kernel[(self.num_ctas,)](q_desc, k_desc, v_desc, do_desc, aux_desc, dq_desc, dk_desc, dv_desc,
                                      self.counter, self.work, self.cu, self.pad, dk, dv, self.num_units, TP, H, self.num_pblk,
                                      scale * LOG2E, scale, BM=BM, BN=BN, D=D, STAGES=STAGES, num_warps=4)
        SUB_, PW_ = self.POST_CFG
        _postprocess_kernel[(self.num_pblk * (BM // SUB_), H)](self.dq_acc, dq, self.blk_t0, self.blk_n, T, TP, H, scale,
                                                                D=D, BLOCK=BM, SUB=SUB_, num_warps=PW_)
        return dq, dk, dv
