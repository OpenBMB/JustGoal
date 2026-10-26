"""Varlen causal MHA backward on Blackwell (sm_100a), written from scratch in Gluon (Triton 3.7).

Design (per CTA, persistent, warp-specialized, one CTA per SM):
  unit = (doc, kv tile of BN=128 rows, head). For each unit the CTA keeps K (TMEM + smem) and V (smem) resident,
  accumulates dK, dV in TMEM (fp32) and loops over the q tiles (BM=64) from the causal diagonal to the doc end.
  Per q tile, tensor core order (two MMA-issuing warps, all hazards through mbarriers):
      warp M1:  S^T = K Q^T (+ bias MMA adding -lse/scale),  dP^T = V dO^T (+ bias MMA adding -delta)
      softmax warps (2 x 4 warps, 32 q columns each):  P^T = exp2(S'^T * scale*log2e),  dS^T = P^T * dP'^T
                                                       -> bf16 P^T | dS^T into TMEM, dS^T into smem
      warp M2:  dV += P^T dO,  dK += dS^T Q (A from TMEM),  dQ^T = K^T dS^T -> TMEM (S region)
      epilogue warps: dQ^T -> fp16 smem staging -> TMA reduce-add into the (H*D, Tpad) accumulator;
                      at unit end dV / dK -> bf16 -> smem staging -> TMA stores; also stage the next unit's K
                      global -> TMEM (the last KS tiles of a unit use K from smem so TMEM K is free early).
  A preprocess kernel computes delta = rowsum(O * dO); a postprocess kernel converts the dQ accumulator to bf16.
"""
import os
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, TensorMemoryLayout, get_tmem_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton._C.libtriton import ir

LOG2E = 1.4426950408889634
BM, BN, D_ = 64, 128, 128
NQS = 3            # Q / dO smem stages
KA = 16            # K dimension of the bias MMAs (hi/lo bf16 split of -lse/scale and -delta)
KS = 5             # last KS tiles of a unit take K from smem, freeing TMEM K for the next unit's copy
UC = 3             # schedule cost model: per-unit overhead in tile units
_KS = gl.constexpr(KS)
# register budgets of the worker partitions: softmax B, epilogue, loader, MMA warp 1, MMA warp 2
_R_SB, _R_EP, _R_LD, _R_M1, _R_M2 = gl.constexpr(128), gl.constexpr(168), gl.constexpr(64), gl.constexpr(64), gl.constexpr(64)

# ---- debug instrumentation (compiled out by default; used by the ts_*.py diagnostics scripts)
DBGT = gl.constexpr(int(os.environ.get("DBGT", "0")))      # 1: per-tile / per-unit clock64 timestamps of CTA DBGPID
DBGW = gl.constexpr(int(os.environ.get("DBGW", "0")))      # 1: also record the last mbarrier wait site per partition
DBGD = gl.constexpr(int(os.environ.get("DBGD", "0")))      # 1: per-partition done flags
DBGPID = gl.constexpr(int(os.environ.get("DBGPID", "0")))  # CTA whose timestamps are recorded
TSLO = gl.constexpr(int(os.environ.get("TSLO", "0")))      # first tile index recorded (64 tiles)


@gluon.jit
def _ts(ts_ptr, it, slot, pid):
    if DBGT:
        if pid == DBGPID:
            if (it >= TSLO) & (it < TSLO + 64):
                t = gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,l", [it.to(gl.int64)], dtype=gl.int64, is_pure=False, pack=1)
                gl.store(ts_ptr + (it - TSLO) * 16 + slot, t)


@gluon.jit
def _uts(ts_ptr, uc, slot, pid):
    if DBGT:
        if pid == DBGPID:
            if uc < 32:
                t = gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,l", [uc.to(gl.int64)], dtype=gl.int64, is_pure=False, pack=1)
                gl.store(ts_ptr + 64 * 16 + uc * 16 + slot, t)


@gluon.jit
def _wt(ts_ptr, wslot, code, pid):
    if DBGW:
        if pid == DBGPID:
            gl.store(ts_ptr + 64 * 16 + 32 * 16 + wslot * 2, code)
            gl.store(ts_ptr + 64 * 16 + 32 * 16 + wslot * 2 + 1, gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,l", [pid.to(gl.int64)], dtype=gl.int64, is_pure=False, pack=1))


@gluon.jit
def _ev(ts_ptr, base, uc, pid):
    if DBGT:
        if pid == DBGPID:
            if uc < 32:
                gl.store(ts_ptr + 64 * 16 + 32 * 16 + 16 + base * 32 + uc, gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,l", [uc.to(gl.int64)], dtype=gl.int64, is_pure=False, pack=1))


@gluon.jit
def _done(ts_ptr, pid, part):
    if DBGD:
        gl.store(ts_ptr + 2048 + pid * 8 + part, 1)


@gluon.jit
def _gtimer(dbg_ptr, pid, slot):
    if DBGT:
        t = gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l", [pid.to(gl.int64)], dtype=gl.int64, is_pure=False, pack=1)
        gl.store(dbg_ptr + pid * 4 + slot, (t & 0x7FFFFFFF).to(gl.int32))


@builtin
def tma_reduce_add(tensor_desc, coord, src, _semantic=None):
    """TMA reduce-add (bulk async group) of an smem tile into global memory."""
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(ir.DESCRIPTOR_REDUCE_KIND.ADD, tensor_desc.handle, coord, src.handle)


# ----------------------------------------------------------------------------- pre / post processing (plain Triton)
@triton.jit
def _preprocess_kernel(o_ptr, do_ptr, delta_ptr, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """delta[h, t] = sum_d o[t, h, d] * do[t, h, d]."""
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    offs = t[:, None] * (H * D) + h * D + d[None, :]
    mask = t[:, None] < T
    o = tl.load(o_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(delta_ptr + h * T + t, tl.sum(o * do, axis=1), mask=t < T)


@triton.jit
def _zero_rows_kernel(ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = ptr + row * ncols
    for c0 in range(0, ncols, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        tl.store(base + offs, tl.zeros([BLOCK], dtype=ptr.dtype.element_ty), mask=offs < ncols)


@triton.jit
def _postprocess_kernel(dq_acc_ptr, dq_ptr, qtile_tab, Tpad, scale, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    """dq[t0 + j, h, :] = scale * dq_acc[h*D + d, c0 + j]^T for one q tile."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    t0 = tl.load(qtile_tab + pid * 3 + 0)      # first global token of the tile
    nvalid = tl.load(qtile_tab + pid * 3 + 1)  # valid rows in tile
    c0 = tl.load(qtile_tab + pid * 3 + 2)      # padded column offset
    d = tl.arange(0, D)
    j = tl.arange(0, BM)
    acc = tl.load(dq_acc_ptr + (h * D + d)[:, None].to(tl.int64) * Tpad + c0 + j[None, :]).to(tl.float32)  # [D, BM]
    dq = (tl.trans(acc) * scale).to(tl.bfloat16)  # [BM, D]
    offs = (t0 + j)[:, None] * (H * D) + h * D + d[None, :]
    tl.store(dq_ptr + offs, dq, mask=(j < nvalid)[:, None])


# ----------------------------------------------------------------------------- main kernel: loader warp
@gluon.jit
def _load_qdo(q_desc, do_desc, lse_ptr, delta_ptr, T, inv_scale, q_smem, do_smem, lse_smem, delta_smem,
              q_ready, q_free, it, h, cu_b, j, idx, is0, is1, ts_ptr, pid,
              BM: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr, KA: gl.constexpr, b_layout: gl.constexpr):
    """Stage s = it % NQS: TMA-load the Q and dO tiles and write the bias B tiles (-lse/scale, -delta as hi/lo bf16
    columns 0/1 of a [BM, KA] tile; the ones-matrix MMA adds hi + lo). q_ready[s] counts 2 arrivals: the TMA
    transaction and the bias tile writes."""
    s = it % NQS
    ph = ((it // NQS) & 1) ^ 1
    _wt(ts_ptr, 0, 1, pid)
    mbarrier.wait(q_free.index(s), ph)
    _ts(ts_ptr, it, 11, pid)
    mbarrier.expect(q_ready.index(s), q_desc.block_type.nbytes + do_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [cu_b + j * BM, h * D], q_ready.index(s), q_smem.index(s))
    tma.async_copy_global_to_shared(do_desc, [cu_b + j * BM, h * D], q_ready.index(s), do_smem.index(s))
    gq = cu_b + j * BM + idx
    gmask = gq < T
    vl = -gl.load(lse_ptr + h * T + gq, mask=gmask, other=0.0) * inv_scale   # S' = S - lse/scale
    vd = -gl.load(delta_ptr + h * T + gq, mask=gmask, other=0.0)            # dP' = dP - delta
    vl_hi = vl.to(gl.bfloat16)
    vl_lo = (vl - vl_hi.to(gl.float32)).to(gl.bfloat16)
    vd_hi = vd.to(gl.bfloat16)
    vd_lo = (vd - vd_hi.to(gl.float32)).to(gl.bfloat16)
    zero_b = gl.zeros([BM, KA], gl.bfloat16, b_layout)
    lse_smem.index(s).store(gl.where(is0, gl.expand_dims(vl_hi, 1), gl.where(is1, gl.expand_dims(vl_lo, 1), zero_b)))
    delta_smem.index(s).store(gl.where(is0, gl.expand_dims(vd_hi, 1), gl.where(is1, gl.expand_dims(vd_lo, 1), zero_b)))
    fence_async_shared()
    gl.barrier()
    mbarrier.arrive(q_ready.index(s))
    _ts(ts_ptr, it, 12, pid)


@gluon.jit
def _load_part(q_desc, k_desc, v_desc, do_desc, unit_tab, u_begin, u_end, ts_ptr, pid, lse_ptr, delta_ptr, T, inv_scale,
               q_smem, do_smem, kv_a, kv_b, lse_smem, delta_smem, q_ready, q_free, kv_ready, kv_free, v_ready, v_free,
               BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr, KA: gl.constexpr):
    b_layout: gl.constexpr = gl.BlockedLayout([2, KA], [32, 1], [1, 1], [1, 0])
    idx = gl.arange(0, BM, layout=gl.SliceLayout(1, b_layout))
    kk = gl.arange(0, KA, layout=gl.SliceLayout(0, b_layout))
    is0 = gl.expand_dims(kk == 0, 0)
    is1 = gl.expand_dims(kk == 1, 0)
    it = 0
    uc = 0
    for u in range(u_begin, u_end):
        h = gl.load(unit_tab + u * 8 + 0)
        cu_b = gl.load(unit_tab + u * 8 + 1)
        m = gl.load(unit_tab + u * 8 + 3)
        nq = gl.load(unit_tab + u * 8 + 5)
        j0 = gl.load(unit_tab + u * 8 + 6)
        # 1. V into the V buffer once the previous unit's last dP MMA is done
        _uts(ts_ptr, uc, 8, pid)
        _wt(ts_ptr, 0, 3, pid)
        mbarrier.wait(v_free, (uc & 1) ^ 1)
        _uts(ts_ptr, uc, 9, pid)
        mbarrier.expect(v_ready, v_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(v_desc, [cu_b + m * BN, h * D], v_ready, kv_b)
        # 2. prefetch the first NQS q tiles while the previous unit is still finishing
        jpre = gl.minimum(nq, j0 + NQS)
        for j in range(j0, jpre):
            _load_qdo(q_desc, do_desc, lse_ptr, delta_ptr, T, inv_scale, q_smem, do_smem, lse_smem, delta_smem,
                      q_ready, q_free, it, h, cu_b, j, idx, is0, is1, ts_ptr, pid, BM, D, NQS, KA, b_layout)
            it += 1
        # 3. K into the K buffer (dQ^T MMAs and the last KS S MMAs) once the previous unit's last dQ^T is done
        _wt(ts_ptr, 0, 5, pid)
        mbarrier.wait(kv_free, (uc & 1) ^ 1)
        _uts(ts_ptr, uc, 12, pid)
        mbarrier.expect(kv_ready, k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [cu_b + m * BN, h * D], kv_ready, kv_a)
        uc += 1
        for j in range(jpre, nq):
            _load_qdo(q_desc, do_desc, lse_ptr, delta_ptr, T, inv_scale, q_smem, do_smem, lse_smem, delta_smem,
                      q_ready, q_free, it, h, cu_b, j, idx, is0, is1, ts_ptr, pid, BM, D, NQS, KA, b_layout)
            it += 1
        _uts(ts_ptr, uc - 1, 13, pid)
    _done(ts_ptr, pid, 0)


# ----------------------------------------------------------------------------- main kernel: MMA warps
@gluon.jit
def _mma_sdp_part(unit_tab, u_begin, u_end, ts_ptr, pid, q_smem, do_smem, v_smem, k_smem, ones_smem, lse_smem, delta_smem,
                  s_tmem, dp_tmem, k_tmem, q_ready, v_ready, v_free, k_ready, kv_ready, s_ready, s_free, dp_ready, dp_free,
                  dq_free, dql_done, BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr):
    """MMA warp 1: S_i = K Q_i^T (K from TMEM, or from smem for the last KS tiles of a unit) and dP_i = V dO_i^T, each
    followed by the K=16 bias MMA (-lse/scale resp. -delta). Runs decoupled from the dV/dK/dQ warp."""
    it = 0
    uc = 0
    for u in range(u_begin, u_end):
        nq = gl.load(unit_tab + u * 8 + 5)
        j0 = gl.load(unit_tab + u * 8 + 6)
        n = nq - j0
        _wt(ts_ptr, 1, 43, pid)
        mbarrier.wait(k_ready, uc & 1)
        _uts(ts_ptr, uc, 1, pid)
        for i in range(n):
            s = it % NQS
            ph = (it // NQS) & 1
            _wt(ts_ptr, 1, 44, pid)
            mbarrier.wait(q_ready.index(s), ph)
            # s_tmem hazard: inside a unit dQ^T_{it-1} is issued by the other MMA warp only after S_it was loaded, so
            # S_it needs dQ^T_{it-2} drained (dq_free parity it&1; the barrier is then exactly 0 or 1 phase short).
            # The last dQ^T of a unit does not depend on the next unit's S_0 (dq_free may run 2 phases ahead there),
            # so the first tile of a unit waits on the per-unit dql_done barrier instead.
            if i == 0:
                _wt(ts_ptr, 1, 45, pid)
                mbarrier.wait(s_free, (it & 1) ^ 1)        # softmax loaded S_{it-1}
                _wt(ts_ptr, 1, 46, pid)
                mbarrier.wait(dql_done, (uc & 1) ^ 1)      # last dQ^T of the previous unit drained
                _ts(ts_ptr, it, 1, pid)
                if i >= n - _KS:
                    _wt(ts_ptr, 1, 56, pid)
                    mbarrier.wait(kv_ready, uc & 1)        # K smem copy of this unit present
                    tcgen05_mma(k_smem, q_smem.index(s).permute((1, 0)), s_tmem, use_acc=False)
                else:
                    tcgen05_mma(k_tmem, q_smem.index(s).permute((1, 0)), s_tmem, use_acc=False)
                tcgen05_mma(ones_smem, lse_smem.index(s).permute((1, 0)), s_tmem, use_acc=True)
                tcgen05_commit(s_ready)
                _wt(ts_ptr, 1, 47, pid)
                mbarrier.wait(v_ready, uc & 1)             # (V is loaded after the previous unit's last dP)
                _uts(ts_ptr, uc, 2, pid)
            _wt(ts_ptr, 1, 48, pid)
            mbarrier.wait(dp_free, (it & 1) ^ 1)           # softmax loaded dP_{it-1}
            _ts(ts_ptr, it, 0, pid)
            tcgen05_mma(v_smem, do_smem.index(s).permute((1, 0)), dp_tmem, use_acc=False)
            tcgen05_mma(ones_smem, delta_smem.index(s).permute((1, 0)), dp_tmem, use_acc=True)
            tcgen05_commit(dp_ready)
            if i == n - 1:
                tcgen05_commit(v_free)
            if i > 0:
                _wt(ts_ptr, 1, 49, pid)
                mbarrier.wait(s_free, (it & 1) ^ 1)
                _wt(ts_ptr, 1, 50, pid)
                mbarrier.wait(dq_free, it & 1)
                _ts(ts_ptr, it, 1, pid)
                if i >= n - _KS:
                    # last KS tiles: K from smem (SS MMA) -> k_tmem is free for the next unit's K
                    if i == n - _KS:
                        _wt(ts_ptr, 1, 56, pid)
                        mbarrier.wait(kv_ready, uc & 1)
                    tcgen05_mma(k_smem, q_smem.index(s).permute((1, 0)), s_tmem, use_acc=False)
                else:
                    tcgen05_mma(k_tmem, q_smem.index(s).permute((1, 0)), s_tmem, use_acc=False)
                tcgen05_mma(ones_smem, lse_smem.index(s).permute((1, 0)), s_tmem, use_acc=True)
                tcgen05_commit(s_ready)
            it += 1
        uc += 1
    _done(ts_ptr, pid, 1)


@gluon.jit
def _mma_dkvq_part(unit_tab, u_begin, u_end, ts_ptr, pid, q_smem, do_smem, k_smem, ds_smem, s_tmem, pd_tmem, dv_tmem, dk_tmem,
                   q_free, kv_ready, kv_free, s_free, pds_ready, pk_done, dq_ready, dq_free, ds_free, dkv_ready, dkv_free,
                   BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr):
    """MMA warp 2: dV += P^T dO, dK += dS^T Q (P^T / dS^T bf16 in TMEM), then dQ^T_i = K^T dS^T_i into s_tmem once
    the softmax warps have loaded S_{i+1}."""
    tl_p: gl.constexpr = TensorMemoryLayout(block=(BN, BM), col_stride=1)
    p_tmem = pd_tmem._reinterpret(gl.bfloat16, [BN, BM], tl_p)
    dst_tmem = pd_tmem.slice(BM // 2, BM // 2)._reinterpret(gl.bfloat16, [BN, BM], tl_p)
    it = 0
    uc = 0
    for u in range(u_begin, u_end):
        nq = gl.load(unit_tab + u * 8 + 5)
        j0 = gl.load(unit_tab + u * 8 + 6)
        n = nq - j0
        _uts(ts_ptr, uc, 0, pid)
        _wt(ts_ptr, 5, 51, pid)
        mbarrier.wait(dkv_free, (uc & 1) ^ 1)             # dV/dK of the previous unit left TMEM
        for i in range(n):
            s = it % NQS
            sb = it & 1
            _wt(ts_ptr, 5, 52, pid)
            mbarrier.wait(pds_ready, it & 1)
            _ts(ts_ptr, it, 2, pid)
            if i == 0:
                _uts(ts_ptr, uc, 3, pid)
            if i == n - 1:
                _uts(ts_ptr, uc, 4, pid)
            tcgen05_mma(p_tmem, do_smem.index(s), dv_tmem, use_acc=(i > 0))
            tcgen05_mma(dst_tmem, q_smem.index(s), dk_tmem, use_acc=(i > 0))
            tcgen05_commit(pk_done)
            tcgen05_commit(q_free.index(s))
            if i == n - 1:
                tcgen05_commit(dkv_ready)
            if i == 0:
                _wt(ts_ptr, 5, 53, pid)
                mbarrier.wait(kv_ready, uc & 1)            # K smem copy present
            if i + 1 < n:
                _wt(ts_ptr, 5, 54, pid)
                mbarrier.wait(s_free, (it + 1) & 1)        # softmax loaded S_{i+1} -> s_tmem reusable
            else:
                # last tile of the unit: nothing else orders this dQ^T after the epilogue's consumption of
                # dQ^T_{it-1}; without this wait dq_ready could complete twice before the epilogue's parity wait
                _wt(ts_ptr, 5, 55, pid)
                mbarrier.wait(dq_free, (it & 1) ^ 1)       # dQ^T_{it-1} drained
            _ts(ts_ptr, it, 3, pid)
            tcgen05_mma(k_smem.permute((1, 0)), ds_smem.index(sb), s_tmem, use_acc=False)
            tcgen05_commit(dq_ready)
            tcgen05_commit(ds_free.index(sb))
            if i == n - 1:
                tcgen05_commit(kv_free)
            it += 1
        uc += 1
    _done(ts_ptr, pid, 5)


# ----------------------------------------------------------------------------- main kernel: softmax warps
@gluon.jit
def _p_chunk(s_c, c, scale_log2e, need_mask, kv_rows, q0, L, BN: gl.constexpr, CH: gl.constexpr, reg_c: gl.constexpr):
    p_c = gl.exp2(s_c * scale_log2e)
    if need_mask:
        qc = q0 + c + gl.arange(0, CH, layout=gl.SliceLayout(0, reg_c))
        valid = (gl.expand_dims(kv_rows, 1) <= gl.expand_dims(qc, 0)) & (gl.expand_dims(kv_rows, 1) < L) & (gl.expand_dims(qc, 0) < L)
        p_c = gl.where(valid, p_c, 0.0)
    return p_c


@gluon.jit
def _softmax_part(unit_tab, u_begin, u_end, ts_ptr, pid, ds_smem, s_tmem, dp_tmem, pd_tmem,
                  s_ready, s_free, dp_ready, dp_free, pds_ready, ds_free, pk_done, scale_log2e,
                  PART: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, NQS: gl.constexpr, num_warps: gl.constexpr):
    """Softmax warps: partition PART (0/1) handles q columns [32*PART, 32*PART+32) of every tile:
    S^T and dP^T already contain -lse/scale and -delta (bias MMAs). P^T (bf16) goes to PD columns [0,32), dS^T (bf16)
    to PD columns [32,64) and to the smem dS^T buffer of the tile."""
    CH: gl.constexpr = 16
    COL0: gl.constexpr = PART * (BM // 2)
    wslot: gl.constexpr = 2 + PART
    tl_s: gl.constexpr = TensorMemoryLayout(block=(BN, BM), col_stride=1)
    tl_c: gl.constexpr = TensorMemoryLayout(block=(BN, CH), col_stride=1)
    reg_c: gl.constexpr = get_tmem_reg_layout(gl.float32, [BN, CH], tl_c, num_warps)
    reg_pc: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BN, CH], tl_c, num_warps)
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, reg_c))
    p_tmem = pd_tmem._reinterpret(gl.bfloat16, [BN, BM], tl_s)
    dst_tmem = pd_tmem.slice(BM // 2, BM // 2)._reinterpret(gl.bfloat16, [BN, BM], tl_s)
    it = 0
    for u in range(u_begin, u_end):
        L = gl.load(unit_tab + u * 8 + 2)
        m = gl.load(unit_tab + u * 8 + 3)
        nq = gl.load(unit_tab + u * 8 + 5)
        j0 = gl.load(unit_tab + u * 8 + 6)
        kv0 = m * BN
        kv_rows = kv0 + rows
        kv_last_partial = (kv0 + BN) > L
        for j in range(j0, nq):
            sb = it & 1
            q0 = j * BM
            need_mask = (q0 < kv0 + BN) | ((q0 + BM) > L) | kv_last_partial
            _wt(ts_ptr, wslot, 34, pid)
            mbarrier.wait(s_ready, it & 1)
            if PART == 0:
                _ts(ts_ptr, it, 4, pid)
            else:
                _ts(ts_ptr, it, 13, pid)
            s0 = s_tmem.slice(COL0 + 0 * CH, CH).load(reg_c)
            s1 = s_tmem.slice(COL0 + 1 * CH, CH).load(reg_c)
            gl.barrier()
            mbarrier.arrive(s_free)
            p0 = _p_chunk(s0, COL0 + 0 * CH, scale_log2e, need_mask, kv_rows, q0, L, BN, CH, reg_c)
            p1 = _p_chunk(s1, COL0 + 1 * CH, scale_log2e, need_mask, kv_rows, q0, L, BN, CH, reg_c)
            _wt(ts_ptr, wslot, 35, pid)
            mbarrier.wait(dp_ready, it & 1)
            if PART == 0:
                _ts(ts_ptr, it, 5, pid)
            dp0 = dp_tmem.slice(COL0 + 0 * CH, CH).load(reg_c)
            dp1 = dp_tmem.slice(COL0 + 1 * CH, CH).load(reg_c)
            gl.barrier()
            mbarrier.arrive(dp_free)
            ds0 = (p0 * dp0).to(gl.bfloat16)
            ds1 = (p1 * dp1).to(gl.bfloat16)
            p016 = gl.convert_layout(p0.to(gl.bfloat16), reg_pc)
            p116 = gl.convert_layout(p1.to(gl.bfloat16), reg_pc)
            if PART == 0:
                _ts(ts_ptr, it, 6, pid)
            _wt(ts_ptr, wslot, 36, pid)
            mbarrier.wait(pk_done, (it & 1) ^ 1)           # dV/dK of the previous tile finished reading P^T/dS^T
            p_tmem.slice(COL0 + 0 * CH, CH).store(p016)
            p_tmem.slice(COL0 + 1 * CH, CH).store(p116)
            dst_tmem.slice(COL0 + 0 * CH, CH).store(gl.convert_layout(ds0, reg_pc))
            dst_tmem.slice(COL0 + 1 * CH, CH).store(gl.convert_layout(ds1, reg_pc))
            _wt(ts_ptr, wslot, 37, pid)
            mbarrier.wait(ds_free.index(sb), ((it // 2) & 1) ^ 1)   # dQ^T_{it-2} finished reading buffer sb
            ds_sb = ds_smem.index(sb)
            ds_sb.slice(COL0 + 0 * CH, CH, dim=1).store(ds0)
            ds_sb.slice(COL0 + 1 * CH, CH, dim=1).store(ds1)
            fence_async_shared()
            gl.barrier()
            mbarrier.arrive(pds_ready)
            if PART == 0:
                _ts(ts_ptr, it, 7, pid)
            else:
                _ts(ts_ptr, it, 14, pid)
            it += 1
    _done(ts_ptr, pid, 2 + PART)


# ----------------------------------------------------------------------------- main kernel: epilogue warps
@gluon.jit
def _load_acc2(acc_tmem, c0: gl.constexpr, scale, BN: gl.constexpr, num_warps: gl.constexpr):
    """Two 32-column chunks [c0, c0+64) of a TMEM [BN, D] fp32 accumulator, scaled, as bf16 register tiles."""
    CW: gl.constexpr = 32
    tl_w: gl.constexpr = TensorMemoryLayout(block=(BN, CW), col_stride=1)
    reg_w: gl.constexpr = get_tmem_reg_layout(gl.float32, [BN, CW], tl_w, num_warps)
    a = (acc_tmem.slice(c0, CW).load(reg_w) * scale).to(gl.bfloat16)
    b = (acc_tmem.slice(c0 + CW, CW).load(reg_w) * scale).to(gl.bfloat16)
    return a, b


@gluon.jit
def _stage_store(stg, a, b, desc, out_ptr, row0, col0, valid,
                 BN: gl.constexpr, D: gl.constexpr, H: gl.constexpr, num_warps: gl.constexpr):
    """bf16 chunks a | b ([BN, 32] each) -> smem staging [BN, 64] -> rows row0.. / columns col0.. of a (T, H*D) tensor.
    A full kv tile goes out with one TMA store; a partial tile (document end) with masked register stores of the
    `valid` rows. The caller must have made `stg` free (store_wait + barrier)."""
    CW: gl.constexpr = 32
    stg.slice(0, CW, dim=1).store(a)
    stg.slice(CW, CW, dim=1).store(b)
    fence_async_shared()
    gl.barrier()
    if valid >= BN:
        tma.async_copy_shared_to_global(desc, [row0, col0], stg)
    else:
        tl_w: gl.constexpr = TensorMemoryLayout(block=(BN, CW), col_stride=1)
        reg_w: gl.constexpr = get_tmem_reg_layout(gl.float32, [BN, CW], tl_w, num_warps)
        r = gl.arange(0, BN, layout=gl.SliceLayout(1, reg_w))
        c = gl.arange(0, CW, layout=gl.SliceLayout(0, reg_w))
        offs = gl.expand_dims((row0 + r) * (H * D) + col0, 1) + gl.expand_dims(c, 0)
        m2 = gl.expand_dims(r < valid, 1)
        gl.store(out_ptr + offs, a, mask=m2)
        gl.store(out_ptr + offs + CW, b, mask=m2)


@gluon.jit
def _k_load(k_ptr, unit_tab, u, H: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, num_warps: gl.constexpr):
    """K tile of unit u as two coalesced [BN, 64] bf16 register halves (rows beyond the document are zero)."""
    HD: gl.constexpr = 64
    lay_c: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    h = gl.load(unit_tab + u * 8 + 0)
    cu_b = gl.load(unit_tab + u * 8 + 1)
    L = gl.load(unit_tab + u * 8 + 2)
    m = gl.load(unit_tab + u * 8 + 3)
    r = m * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, lay_c))
    cc = gl.arange(0, HD, layout=gl.SliceLayout(0, lay_c))
    rmask = gl.expand_dims(r < L, 1)
    offs = gl.expand_dims((cu_b + r) * (H * D) + h * D, 1) + gl.expand_dims(cc, 0)
    kc0 = gl.load(k_ptr + offs, mask=rmask, other=0.0)
    kc1 = gl.load(k_ptr + offs + HD, mask=rmask, other=0.0)
    return kc0, kc1


@gluon.jit
def _k_store_tmem(kc0, kc1, k_tmem, k_ready, stg, D: gl.constexpr, BN: gl.constexpr, num_warps: gl.constexpr):
    """Register halves from _k_load -> TMEM (A operand of the S MMAs) through the smem staging tile (layout change to
    the TMEM register layout), then k_ready. The caller must have made `stg` free (store_wait + barrier)."""
    HD: gl.constexpr = 64
    tl_32: gl.constexpr = TensorMemoryLayout(block=(BN, D // 2), col_stride=1)
    tl_h: gl.constexpr = TensorMemoryLayout(block=(BN, HD), col_stride=1)
    reg_h: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BN, HD], tl_h, num_warps)
    k32 = k_tmem._reinterpret(gl.float32, [BN, D // 2], tl_32)
    stg.store(kc0)
    gl.barrier()
    k32.slice(0, HD // 2)._reinterpret(gl.bfloat16, [BN, HD], tl_h).store(stg.load(reg_h))
    gl.barrier()
    stg.store(kc1)
    gl.barrier()
    k32.slice(HD // 2, HD // 2)._reinterpret(gl.bfloat16, [BN, HD], tl_h).store(stg.load(reg_h))
    gl.barrier()
    mbarrier.arrive(k_ready)


@gluon.jit
def _epilogue_part(unit_tab, u_begin, u_end, pid, ts_ptr, dq_desc, dv_desc, dk_desc, dbg_ptr, dk_ptr, dv_ptr, k_ptr, scale,
                   dq_smem, dv_tmem, dk_tmem, dqt_tmem, k_tmem,
                   dq_ready, dq_free, dql_done, dkv_ready, dkv_free, k_ready,
                   BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, H: gl.constexpr, num_warps: gl.constexpr):
    """Per tile: dQ^T (TMEM, fp32) -> fp16 smem staging -> TMA reduce-add into the dq accumulator. Per unit: stage
    the next unit's K into TMEM (at tile jk, when the TMEM-K S MMAs of this unit are complete) and, at the end,
    dV and dK -> bf16 -> smem staging -> TMA stores."""
    HB: gl.constexpr = BM // 2
    tl_h: gl.constexpr = TensorMemoryLayout(block=(BN, HB), col_stride=1)
    reg_h: gl.constexpr = get_tmem_reg_layout(gl.float32, [D, HB], tl_h, num_warps)
    stg = dq_smem._reinterpret(gl.bfloat16, [BN, 64], dv_desc.layout)   # [BN, 64] bf16 view (K copy, dV/dK stores)
    if u_begin < u_end:
        kc0, kc1 = _k_load(k_ptr, unit_tab, u_begin, H, D, BN, num_warps)
        _k_store_tmem(kc0, kc1, k_tmem, k_ready, stg, D, BN, num_warps)
    it = 0
    uc = 0
    for u in range(u_begin, u_end):
        h = gl.load(unit_tab + u * 8 + 0)
        cu_b = gl.load(unit_tab + u * 8 + 1)
        L = gl.load(unit_tab + u * 8 + 2)
        m = gl.load(unit_tab + u * 8 + 3)
        dqpad = gl.load(unit_tab + u * 8 + 4)
        nq = gl.load(unit_tab + u * 8 + 5)
        j0 = gl.load(unit_tab + u * 8 + 6)
        row0 = h * D
        # dQ^T_jk waited for S_{jk+1} to be loaded, i.e. the last TMEM-K S MMA (tile nq-KS-1) is complete after it
        jk = gl.maximum(nq - _KS - 2, j0)
        for j in range(j0, nq):
            _wt(ts_ptr, 4, 41, pid)
            mbarrier.wait(dq_ready, it & 1)
            _ts(ts_ptr, it, 8, pid)
            dq0 = dqt_tmem.slice(0, HB).load(reg_h)
            dq1 = dqt_tmem.slice(HB, HB).load(reg_h)
            gl.barrier()
            mbarrier.arrive(dq_free)
            if j == nq - 1:
                mbarrier.arrive(dql_done)              # last dQ^T of the unit drained from s_tmem
            _ts(ts_ptr, it, 9, pid)
            tma.store_wait(0)                          # previous reduce finished reading the staging tile
            gl.barrier()
            dq_smem.slice(0, HB, dim=1).store(dq0.to(gl.float16))
            dq_smem.slice(HB, HB, dim=1).store(dq1.to(gl.float16))
            fence_async_shared()
            gl.barrier()
            tma_reduce_add(dq_desc, [row0, dqpad + j * BM], dq_smem)
            _ts(ts_ptr, it, 10, pid)
            if j == jk:
                if u + 1 < u_end:
                    _ev(ts_ptr, 5, uc, pid)
                    kc0, kc1 = _k_load(k_ptr, unit_tab, u + 1, H, D, BN, num_warps)
                    tma.store_wait(0)
                    gl.barrier()
                    _ev(ts_ptr, 6, uc, pid)
                    _k_store_tmem(kc0, kc1, k_tmem, k_ready, stg, D, BN, num_warps)
                    _ev(ts_ptr, 7, uc, pid)
            it += 1
        # dV and dK of this unit: TMEM -> bf16 registers -> smem staging -> TMA stores (64 columns at a time)
        _wt(ts_ptr, 4, 42, pid)
        mbarrier.wait(dkv_ready, uc & 1)
        _ev(ts_ptr, 2, uc, pid)
        kv0 = m * BN
        valid = gl.minimum(L - kv0, BN)
        krow0 = cu_b + kv0
        v0, v1 = _load_acc2(dv_tmem, 0, 1.0, BN, num_warps)
        v2, v3 = _load_acc2(dv_tmem, 64, 1.0, BN, num_warps)
        tma.store_wait(0)
        gl.barrier()
        _ev(ts_ptr, 3, uc, pid)
        _stage_store(stg, v0, v1, dv_desc, dv_ptr, krow0, h * D, valid, BN, D, H, num_warps)
        _ev(ts_ptr, 4, uc, pid)
        k0, k1 = _load_acc2(dk_tmem, 0, scale, BN, num_warps)
        tma.store_wait(0)
        gl.barrier()
        _stage_store(stg, v2, v3, dv_desc, dv_ptr, krow0, h * D + 64, valid, BN, D, H, num_warps)
        k2, k3 = _load_acc2(dk_tmem, 64, scale, BN, num_warps)
        gl.barrier()
        mbarrier.arrive(dkv_free)                      # both accumulators left TMEM
        _ev(ts_ptr, 1, uc, pid)
        tma.store_wait(0)
        gl.barrier()
        _stage_store(stg, k0, k1, dk_desc, dk_ptr, krow0, h * D, valid, BN, D, H, num_warps)
        tma.store_wait(0)
        gl.barrier()
        _stage_store(stg, k2, k3, dk_desc, dk_ptr, krow0, h * D + 64, valid, BN, D, H, num_warps)
        _uts(ts_ptr, uc, 5, pid)
        uc += 1
    tma.store_wait(0)
    _done(ts_ptr, pid, 4)
    _gtimer(dbg_ptr, pid, 3)


# ----------------------------------------------------------------------------- main kernel
@gluon.jit
def _bwd_kernel(q_desc, k_desc, v_desc, do_desc, dq_desc, dv_desc, dk_desc, dbg_ptr, ts_ptr, lse_ptr, delta_ptr,
                k_ptr, dk_ptr, dv_ptr, unit_tab, cta_start, T, scale_log2e, scale, inv_scale,
                BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, H: gl.constexpr, NQS: gl.constexpr,
                KA: gl.constexpr, num_warps: gl.constexpr):
    pid = gl.program_id(0)
    _gtimer(dbg_ptr, pid, 2)
    u_begin = gl.load(cta_start + pid)
    u_end = gl.load(cta_start + pid + 1)
    # ---- shared memory (~225 KB)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [NQS, BM, D], q_desc.layout)
    do_smem = gl.allocate_shared_memory(gl.bfloat16, [NQS, BM, D], do_desc.layout)
    kv_a = gl.allocate_shared_memory(gl.bfloat16, [BN, D], k_desc.layout)        # K (dQ^T and last-KS S MMAs)
    kv_b = gl.allocate_shared_memory(gl.bfloat16, [BN, D], k_desc.layout)        # V
    ds_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BM], gl.bfloat16)
    ds_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BN, BM], ds_layout)      # dS^T, double buffered
    dq_smem = gl.allocate_shared_memory(gl.float16, [D, BM], dq_desc.layout)      # dQ^T staging (fp16)
    b_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, KA], gl.bfloat16)
    lse_smem = gl.allocate_shared_memory(gl.bfloat16, [NQS, BM, KA], b_lay)      # -lse/scale bias B tiles
    delta_smem = gl.allocate_shared_memory(gl.bfloat16, [NQS, BM, KA], b_lay)    # -delta bias B tiles
    ones_lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, KA], gl.bfloat16)
    ones_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, KA], ones_lay)
    o_layout: gl.constexpr = gl.BlockedLayout([1, KA], [32, 1], [num_warps, 1], [1, 0])
    okk = gl.arange(0, KA, layout=gl.SliceLayout(0, o_layout))
    ones_smem.store(gl.where(gl.expand_dims(okk < 2, 0) & gl.expand_dims(gl.arange(0, BN, layout=gl.SliceLayout(1, o_layout)) >= 0, 1), 1.0, 0.0).to(gl.bfloat16))
    fence_async_shared()
    # ---- barriers
    q_ready = mbarrier.allocate_mbarrier(batch=NQS)
    q_free = mbarrier.allocate_mbarrier(batch=NQS)
    ds_free = mbarrier.allocate_mbarrier(batch=2)
    sing = mbarrier.allocate_mbarrier(batch=16)
    kv_ready = sing.index(0)
    kv_free = sing.index(1)
    v_ready = sing.index(2)
    v_free = sing.index(3)
    dp_ready = sing.index(4)
    dp_free = sing.index(5)
    pds_ready = sing.index(6)
    dq_ready = sing.index(7)
    dq_free = sing.index(8)
    dkv_ready = sing.index(9)
    dkv_free = sing.index(10)
    k_ready = sing.index(11)
    s_ready = sing.index(12)
    s_free = sing.index(13)
    pk_done = sing.index(14)
    dql_done = sing.index(15)
    for i in gl.static_range(NQS):
        mbarrier.init(q_ready.index(i), count=2)
        mbarrier.init(q_free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(ds_free.index(i), count=1)
    for i in gl.static_range(16):
        mbarrier.init(sing.index(i), count=1)
    mbarrier.init(s_free, count=2)          # both softmax partitions
    mbarrier.init(dp_free, count=2)
    mbarrier.init(pds_ready, count=2)
    fence_async_shared()
    # ---- tensor memory (512 columns total)
    tl_s: gl.constexpr = TensorMemoryLayout(block=(BN, BM), col_stride=1)
    tl_acc: gl.constexpr = TensorMemoryLayout(block=(BN, D), col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [BN, BM], tl_s)     # S^T, later dQ^T            (64 cols)
    dp_tmem = allocate_tensor_memory(gl.float32, [BN, BM], tl_s)    # dP^T                       (64 cols)
    pd_tmem = allocate_tensor_memory(gl.float32, [BN, BM], tl_s)    # P^T | dS^T (bf16)          (64 cols)
    dv_tmem = allocate_tensor_memory(gl.float32, [BN, D], tl_acc)   # dV accumulator            (128 cols)
    dk_tmem = allocate_tensor_memory(gl.float32, [BN, D], tl_acc)   # dK accumulator            (128 cols)
    k_tmem = allocate_tensor_memory(gl.bfloat16, [BN, D], tl_acc)   # K (A operand of S)         (64 cols)
    gl.barrier()
    gl.warp_specialize([
        (_softmax_part, (unit_tab, u_begin, u_end, ts_ptr, pid, ds_smem, s_tmem, dp_tmem, pd_tmem,
                         s_ready, s_free, dp_ready, dp_free, pds_ready, ds_free, pk_done, scale_log2e, 0, BM, BN, NQS, num_warps)),
        (_softmax_part, (unit_tab, u_begin, u_end, ts_ptr, pid, ds_smem, s_tmem, dp_tmem, pd_tmem,
                         s_ready, s_free, dp_ready, dp_free, pds_ready, ds_free, pk_done, scale_log2e, 1, BM, BN, NQS, 4)),
        (_epilogue_part, (unit_tab, u_begin, u_end, pid, ts_ptr, dq_desc, dv_desc, dk_desc, dbg_ptr, dk_ptr, dv_ptr, k_ptr, scale,
                          dq_smem, dv_tmem, dk_tmem, s_tmem, k_tmem,
                          dq_ready, dq_free, dql_done, dkv_ready, dkv_free, k_ready, BM, BN, D, H, 4)),
        (_load_part, (q_desc, k_desc, v_desc, do_desc, unit_tab, u_begin, u_end, ts_ptr, pid, lse_ptr, delta_ptr, T, inv_scale,
                      q_smem, do_smem, kv_a, kv_b, lse_smem, delta_smem, q_ready, q_free, kv_ready, kv_free, v_ready, v_free,
                      BM, BN, D, NQS, KA)),
        (_mma_sdp_part, (unit_tab, u_begin, u_end, ts_ptr, pid, q_smem, do_smem, kv_b, kv_a, ones_smem, lse_smem, delta_smem,
                         s_tmem, dp_tmem, k_tmem, q_ready, v_ready, v_free, k_ready, kv_ready, s_ready, s_free, dp_ready, dp_free,
                         dq_free, dql_done, BM, BN, D, NQS)),
        (_mma_dkvq_part, (unit_tab, u_begin, u_end, ts_ptr, pid, q_smem, do_smem, kv_a, ds_smem, s_tmem, pd_tmem, dv_tmem, dk_tmem,
                          q_free, kv_ready, kv_free, s_free, pds_ready, pk_done, dq_ready, dq_free, ds_free, dkv_ready, dkv_free,
                          BM, BN, D, NQS)),
    ], [4, 4, 1, 1, 1], [_R_SB, _R_EP, _R_LD, _R_M1, _R_M2])


# ----------------------------------------------------------------------------- host side
_sched_cache = {}


def build_schedule(seqlens, H, n_ctas):
    """Units (doc b, kv tile m, head h) with their causal q-tile range, balanced over the CTAs by greedy LPT plus
    local refinement on the cost model (tiles + UC). Returns (unit_tab [n_units, 8] int32 rows
    [h, cu_b, L, m, dqpad, nq, j0, nq], cta_start [n_ctas+1], qtile_tab [n_qtiles, 3], Tpad, n_qtiles)."""
    import heapq
    key = (tuple(seqlens), H, n_ctas)
    if key in _sched_cache:
        return _sched_cache[key]
    cu = [0]
    for L in seqlens:
        cu.append(cu[-1] + L)
    pad = [0]
    for L in seqlens:
        pad.append(pad[-1] + (L + BM - 1) // BM * BM)
    units = []
    for b, L in enumerate(seqlens):
        nq = (L + BM - 1) // BM
        for m in range((L + BN - 1) // BN):
            j0 = (m * BN) // BM
            cost = (nq - j0) + UC
            for h in range(H):
                units.append((cost, [h, cu[b], L, m, pad[b], nq, j0, nq]))
    units.sort(key=lambda x: -x[0])
    heap = [(0, c) for c in range(n_ctas)]
    heapq.heapify(heap)
    lists = [[] for _ in range(n_ctas)]
    for cost, rec in units:
        load, c = heapq.heappop(heap)
        lists[c].append((cost, rec))
        heapq.heappush(heap, (load + cost, c))
    for _ in range(20000):
        tot = [sum(c for c, _ in l) for l in lists]
        hi = max(range(n_ctas), key=lambda p: tot[p])
        lo = min(range(n_ctas), key=lambda p: tot[p])
        gap = tot[hi] - tot[lo]
        best = None
        for idx, (c, _) in enumerate(lists[hi]):
            if c < gap and (best is None or c > best[0]):
                best = (c, idx)
        if best is None:
            break
        lists[lo].append(lists[hi].pop(best[1]))
    tab, start = [], [0]
    for c in range(n_ctas):
        for _, rec in lists[c]:
            tab.append(rec)
        start.append(len(tab))
    qtiles = []
    for b, L in enumerate(seqlens):
        for jq in range((L + BM - 1) // BM):
            qtiles.append([cu[b] + jq * BM, min(BM, L - jq * BM), pad[b] + jq * BM])
    res = (torch.tensor(tab, dtype=torch.int32, device="cuda").flatten(),
           torch.tensor(start, dtype=torch.int32, device="cuda"),
           torch.tensor(qtiles, dtype=torch.int32, device="cuda").flatten(),
           pad[-1], len(qtiles))
    _sched_cache[key] = res
    return res


class VarlenBwd:
    """Backward of varlen causal attention for one packed layout (seqlens fixed at construction).
    __call__(q, k, v, o, do, lse) -> (dq, dk, dv); q/k/v/o/do are (T, H, D) bf16, lse is (H, T) fp32."""

    def __init__(self, seqlens, H, D, device="cuda"):
        assert D == D_
        self.seqlens, self.H, self.D = list(seqlens), H, D
        self.T = sum(seqlens)
        self.n_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.scale = D ** -0.5
        self.delta = torch.empty(H, self.T, device=device, dtype=torch.float32)
        (self.unit_tab, self.cta_start, self.qtile_tab, self.Tpad, self.n_qtiles) = build_schedule(self.seqlens, H, self.n_ctas)
        self.dq_acc = torch.zeros(H * D, self.Tpad, device=device, dtype=torch.float16)
        self.kernel = None
        self.dbg = torch.zeros(self.n_ctas, 4, device=device, dtype=torch.int32)          # global timers (DBGT)
        self.ts = torch.zeros(4096 + self.n_ctas * 16, device=device, dtype=torch.int64)  # timestamps (DBGT/DBGW/DBGD)

    def _args(self, q, k, v, do, lse, dk, dv):
        T, H, D = self.T, self.H, self.D
        lay_q = gl.NVMMASharedLayout.get_default_for([BM, D], gl.bfloat16)
        lay_k = gl.NVMMASharedLayout.get_default_for([BN, D], gl.bfloat16)
        lay_dq = gl.NVMMASharedLayout.get_default_for([D, BM], gl.float16)
        lay64 = gl.NVMMASharedLayout.get_default_for([BN, 64], gl.bfloat16)
        q2, k2, v2, do2, dk2, dv2 = (x.view(T, H * D) for x in (q, k, v, do, dk, dv))
        return (TensorDescriptor.from_tensor(q2, [BM, D], lay_q), TensorDescriptor.from_tensor(k2, [BN, D], lay_k),
                TensorDescriptor.from_tensor(v2, [BN, D], lay_k), TensorDescriptor.from_tensor(do2, [BM, D], lay_q),
                TensorDescriptor.from_tensor(self.dq_acc, [D, BM], lay_dq),
                TensorDescriptor.from_tensor(dv2, [BN, 64], lay64), TensorDescriptor.from_tensor(dk2, [BN, 64], lay64),
                self.dbg, self.ts, lse, self.delta, k, dk, dv, self.unit_tab, self.cta_start, T,
                self.scale * LOG2E, self.scale, 1.0 / self.scale)

    def compile_only(self, q, k, v, o, do, lse):
        """Compile the main kernel without launching it (also used by the diagnostics scripts)."""
        if self.kernel is None:
            dk, dv = torch.empty_like(k), torch.empty_like(v)
            self.kernel = _bwd_kernel.warmup(*self._args(q, k, v, do, lse, dk, dv), BM=BM, BN=BN, D=self.D, H=self.H,
                                             NQS=NQS, KA=KA, num_warps=4, grid=(1,))

    def __call__(self, q, k, v, o, do, lse):
        T, H, D = self.T, self.H, self.D
        assert q.shape == (T, H, D) and lse.shape == (H, T) and lse.dtype == torch.float32
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        BT = 32
        _preprocess_kernel[(triton.cdiv(T, BT), H)](o, do, self.delta, T, H=H, D=D, BT=BT, num_warps=4)
        _zero_rows_kernel[(H * D,)](self.dq_acc, self.Tpad, BLOCK=4096, num_warps=8)
        self.compile_only(q, k, v, o, do, lse)
        self.kernel[(self.n_ctas, 1, 1)](*self._args(q, k, v, do, lse, dk, dv), BM, BN, D, H, NQS, KA, 4)
        _postprocess_kernel[(self.n_qtiles, H)](self.dq_acc, dq, self.qtile_tab, self.Tpad, self.scale, H=H, D=D, BM=BM, num_warps=8)
        return dq, dk, dv
