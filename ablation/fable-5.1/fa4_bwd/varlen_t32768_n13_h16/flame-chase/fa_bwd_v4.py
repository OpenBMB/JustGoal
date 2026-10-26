"""FlashAttention-style varlen causal backward for Blackwell (sm100) in Gluon -- v4.

Structure (per CTA = one 128-key block of one (document, head); CTAs run in clusters of 2 = adjacent key blocks):
  per 128-query block j (from the diagonal to the end of the document):
     S^T  = K Q^T      [128 x 128]  (A = K smem, B = Q smem)              -> S region (TMEM)
     P^T  = exp2(S^T*c - lse[q]) masked          -> bf16 into the S region (cols 0..63)
     dP^T = V dO^T     [128 x 128]                                        -> R region
     dS^T = P^T (dP^T - delta[q])                -> bf16 into the S region (cols 64..127) + smem
     dV  += P^T dO ; dQ^T = K^T dS^T -> R ; dK += dS^T Q
     dQ^T: each CTA of the pair owns 64 of the 128 query columns; it sends the other 64 columns (fp32) to its
           partner through distributed shared memory (into the partner's dS buffer, which is idle at that time),
           adds the partner's contribution for its own columns and reduces them into dq_accum (fp32 atomics).
  Warp roles: softmax group 0 (8 warps, default partition), softmax group 1 (8 warps), TMA producer (1),
  MMA issuer (1), dQ drain/exchange (4).
"""
import math
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, tcgen05_mma, tcgen05_commit,
    mbarrier, tma, fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from kernel.cluster_launch import register_cluster

NORED = gl.constexpr(int(__import__("os").environ.get("FA_NORED", "0")))   # ablation: skip dQ reductions
BLOCK_N = 128   # keys per CTA
BLOCK_M = 128   # queries per iteration
HEAD_D = 128
Q_STAGES = 2
XBUF_MAGIC = 0x5EED1234


# ----------------------------------------------------------------------------------------------
# inline-PTX helpers (cluster / DSMEM / raw mbarriers)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _asm_u32(asm: gl.constexpr, a):
    return gl.inline_asm_elementwise(asm, "=r,r", [a], dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _asm_u32_2(asm: gl.constexpr, a, b):
    return gl.inline_asm_elementwise(asm, "=r,r,r", [a, b], dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _scan_smem(word_a, word_b, n_scan, smem_bytes, tid, NT: gl.constexpr):
    """Find the byte offsets (in dynamic smem) of the 16B-aligned words equal to word_a / word_b."""
    fa = gl.zeros_like(tid) + 0x7FFFFFFF
    fb = gl.zeros_like(tid) + 0x7FFFFFFF
    for k in range(n_scan):
        cand = (k * NT + tid) * 16
        v = gl.inline_asm_elementwise(
            "{ .reg .pred p; .reg .u32 b; setp.lt.u32 p, $1, $2; mov.u32 b, global_smem; add.u32 b, b, $1; mov.u32 $0, 0; @p ld.shared.b32 $0, [b]; }",
            "=r,r,r", [cand, gl.zeros_like(cand) + smem_bytes], dtype=gl.int32, is_pure=False, pack=1)
        fa = gl.where((v == word_a) & (cand < fa), cand, fa)
        fb = gl.where((v == word_b) & (cand < fb), cand, fb)
    # the whole buffer is filled with the sentinel -> the buffer start is the FIRST matching chunk
    return gl.min(fa, axis=0), gl.min(fb, axis=0)


@gluon.jit
def _smem_addr(off_u32):
    return _asm_u32("{ .reg .u32 b; mov.u32 b, global_smem; add.u32 $0, b, $1; }", off_u32)


@gluon.jit
def _mapa(addr_u32, peer_u32):
    return _asm_u32_2("mapa.shared::cluster.u32 $0, $1, $2;", addr_u32, peer_u32)


@gluon.jit
def _ctarank(dummy):
    return _asm_u32("mov.u32 $0, %cluster_ctarank;", dummy)


@gluon.jit
def _cluster_sync(dummy):
    return _asm_u32("{ barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned; mov.u32 $0, $1; }", dummy)


@gluon.jit
def _mbar_init_pred(addr_u32, tid):
    return _asm_u32_2("{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared::cta.b64 [$1], 1; mov.u32 $0, $1; }", addr_u32, tid)


@gluon.jit
def _mbar_remote_arrive_pred(addr_u32, tid):
    return _asm_u32_2("{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$1]; mov.u32 $0, $1; }", addr_u32, tid)


@gluon.jit
def _mbar_wait_local(addr_u32, phase_u32):
    return _asm_u32_2(
        "{ .reg .pred p; LAB_WAIT: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2; @!p bra LAB_WAIT; mov.u32 $0, $1; }",
        addr_u32, phase_u32)


@gluon.jit
def _st_cluster_f32(base_u32, idx_i32, vals):
    """st.shared::cluster.f32 [base + 4*idx], vals (element-wise)."""
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $2, 4, $1; st.shared::cluster.f32 [a], $3; mov.u32 $0, a; }",
        "=r,r,r,f", [base_u32, idx_i32, vals], dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _st_local_v4(base_u32, idx_i32, vals):
    """4 consecutive elements per thread: st.shared.v4.f32 [base + 4*idx_first], {vals}"""
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $8, 4, $4; st.shared.v4.f32 [a], {$12, $13, $14, $15}; "
        "mov.u32 $0, a; mov.u32 $1, a; mov.u32 $2, a; mov.u32 $3, a; }",
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,f,f,f,f", [base_u32, idx_i32, vals], dtype=gl.uint32, is_pure=False, pack=4)


@gluon.jit
def _ld_cluster_v4(base_u32, idx_i32):
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $8, 4, $4; ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [a]; }",
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r", [base_u32, idx_i32], dtype=gl.float32, is_pure=False, pack=4)


@gluon.jit
def _ld_cluster_f32(base_u32, idx_i32):
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $2, 4, $1; ld.shared::cluster.f32 $0, [a]; }",
        "=f,r,r", [base_u32, idx_i32], dtype=gl.float32, is_pure=False, pack=1)


@gluon.jit
def _st_local_f32(base_u32, idx_i32, vals):
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $2, 4, $1; st.shared.f32 [a], $3; mov.u32 $0, a; }",
        "=r,r,r,f", [base_u32, idx_i32, vals], dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _ld_local_f32(base_u32, idx_i32):
    return gl.inline_asm_elementwise(
        "{ .reg .u32 a; mad.lo.u32 a, $2, 4, $1; ld.shared.f32 $0, [a]; }",
        "=f,r,r", [base_u32, idx_i32], dtype=gl.float32, is_pure=False, pack=1)


@gluon.jit
def _red_add_f32(vals, base_i64, offs_i32):
    """red.global.add.f32 [base + 4*offs], vals   (base: scalar int64, offs: int32 tensor)"""
    if NORED:
        return offs_i32
    return gl.inline_asm_elementwise(
        "{ .reg .u64 a; mul.wide.s32 a, $3, 4; add.u64 a, a, $2; red.global.add.f32 [a], $1; mov.b32 $0, $3; }",
        "=r,f,l,r", [vals, base_i64, offs_i32], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _fence_before(dummy):
    """tcgen05.fence::before_thread_sync - order prior tcgen05 ops (TMEM st/ld) before a thread sync."""
    return gl.inline_asm_elementwise("{ tcgen05.fence::before_thread_sync; mov.u32 $0, $1; }", "=r,r", [dummy],
                                     dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _fence_after(dummy):
    """tcgen05.fence::after_thread_sync - order subsequent tcgen05 ops after a thread sync / mbarrier wait."""
    return gl.inline_asm_elementwise("{ tcgen05.fence::after_thread_sync; mov.u32 $0, $1; }", "=r,r", [dummy],
                                     dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _clk(dummy):
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [dummy], dtype=gl.int64, is_pure=False, pack=1)


@gluon.jit
def _dbg_add(dbg_ptr, idx: gl.constexpr, val, lane, DEBUG: gl.constexpr):
    if DEBUG:
        gl.atomic_add(dbg_ptr + idx + lane * 0, val, mask=lane == 0)


@gluon.jit
def _trace(trace_ptr, cta, part: gl.constexpr, code, lane, TRACE: gl.constexpr):
    if TRACE:
        gl.store(trace_ptr + cta * 8 + part + lane * 0, gl.zeros_like(lane) + code, mask=lane == 0)


@gluon.jit
def _exp2_emul(x):
    """exp2 without MUFU / float->int conversion: 2^round(x) * poly(frac); x clamped at -126."""
    x = gl.maximum(x, -126.0)
    t = x + 12582912.0
    r = t - 12582912.0
    f = x - r
    p = 0.99995134 + f * (0.69325305 + f * (0.24225698 + f * 0.05502927))
    return (t.to(gl.int32, bitcast=True) * 8388608 + p.to(gl.int32, bitcast=True)).to(gl.float32, bitcast=True)


# ----------------------------------------------------------------------------------------------
# producer warp
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _producer(q_desc, do_desc, q_smem, do_smem, lse_smem, delta_smem, q_bar, q_empty, do_bar, do_empty,
              lse_ptr, delta_ptr, q_start, h, col0, it0, n_iters, kb, T, trace_ptr, cta, dbg_ptr,
              Q_STAGES: gl.constexpr, BLOCK_M: gl.constexpr, XCHG: gl.constexpr, TRACE: gl.constexpr,
              DEBUG: gl.constexpr):
    vl: gl.constexpr = gl.BlockedLayout([4], [32], [1], [0])
    offs = gl.arange(0, BLOCK_M, layout=vl)
    ctid = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [1], [0]))
    t_start = _clk(ctid)
    tw = gl.zeros_like(ctid).to(gl.int64)
    _trace(trace_ptr, cta, 2, 2, ctid, TRACE)
    for it in range(it0, n_iters):
        itr = it - it0
        s = itr % Q_STAGES
        use = itr // Q_STAGES
        row = q_start + (kb + it) * BLOCK_M
        # ---- Q(it) (+ lse/delta) into the 2-stage ring
        if itr >= Q_STAGES:
            _trace(trace_ptr, cta, 2, 100 + it * 16 + 1, ctid, TRACE)
            if DEBUG:
                t0 = _clk(ctid)
            mbarrier.wait(q_empty.index(s), (use - 1) & 1)
            if DEBUG:
                tw += _clk(ctid) - t0
        _trace(trace_ptr, cta, 2, 100 + it * 16 + 2, ctid, TRACE)
        mbarrier.expect(q_bar.index(s), q_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(q_desc, [row, col0], q_bar.index(s), q_smem.index(s))
        q_glob = row + offs
        ok = q_glob < T
        lse = gl.load(lse_ptr + h * T + q_glob, mask=ok, other=0.0)
        delta = gl.load(delta_ptr + h * T + q_glob, mask=ok, other=0.0)
        lse_smem.index(s).store(lse * 1.4426950408889634)
        delta_smem.index(s).store(delta)
        mbarrier.arrive(q_bar.index(s), count=1)
        # ---- dO(it): single stage, free once dV(it-1) is done
        if itr >= 1:
            _trace(trace_ptr, cta, 2, 100 + it * 16 + 3, ctid, TRACE)
            mbarrier.wait(do_empty, (itr - 1) & 1)
        mbarrier.expect(do_bar, do_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(do_desc, [row, col0], do_bar, do_smem)
    _trace(trace_ptr, cta, 2, 9999, ctid, TRACE)
    _dbg_add(dbg_ptr, 17, _clk(ctid) - t_start, ctid, DEBUG)
    _dbg_add(dbg_ptr, 18, tw, ctid, DEBUG)


# ----------------------------------------------------------------------------------------------
# MMA warp
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _mma(k_smem, v_smem, q_smem, do_smem, ds_smem, s_tmem, r_tmem, dk_tmem, dv_tmem,
         q_bar, q_empty, do_bar, do_empty, s_ready, dp_ready, p_ready, ds_ready, dq_ready, dq_done, it0, n_iters,
         Q_STAGES: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr, XCHG: gl.constexpr,
         TRACE: gl.constexpr, trace_ptr, cta, dbg_ptr, DEBUG: gl.constexpr):
    """Issue order per iteration (S region holds only P^T, so S^T(it+1) can start right after dV(it)):
         [S^T(it0)]  ...  dP^T(it) -> dV(it) -> S^T(it+1) -> dQ^T(it) -> dK(it) -> dP^T(it+1) ...
       The P phase of it+1 is then hidden behind dQ^T(it) + dK(it) + dP^T(it+1)."""
    ctid = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [1], [0]))
    t_start = _clk(ctid)
    tw_q = gl.zeros_like(ctid).to(gl.int64)
    tw_dqd = gl.zeros_like(ctid).to(gl.int64)
    tw_p = gl.zeros_like(ctid).to(gl.int64)
    tw_ds = gl.zeros_like(ctid).to(gl.int64)
    _trace(trace_ptr, cta, 3, 2, ctid, TRACE)
    p_tmem = s_tmem._reinterpret(gl.bfloat16, [BLOCK_N, 2 * BLOCK_M],
                                 TensorMemoryLayout(block=[BLOCK_N, 2 * BLOCK_M], col_stride=1))
    HM: gl.constexpr = BLOCK_M // 2
    p_bf0 = p_tmem.slice(0, HM)              # P^T for queries [0, 64)   (group 0 region)
    p_bf1 = p_tmem.slice(2 * HM, HM)         # P^T for queries [64, 128) (group 1 region)
    # prologue: S^T(it0)
    mbarrier.wait(q_bar.index(0), 0)
    _fence_after(ctid)
    tcgen05_mma(k_smem, q_smem.index(0).permute((1, 0)), s_tmem, use_acc=False)
    tcgen05_commit(s_ready)
    for it in range(it0, n_iters):
        itr = it - it0
        sq = itr % Q_STAGES
        ph = itr & 1
        _trace(trace_ptr, cta, 3, 100 + it * 16 + 1, ctid, TRACE)
        # dP^T(it) -> R  (R held dQ^T(it-1): wait until drained)
        if itr > 0:
            if DEBUG:
                t0 = _clk(ctid)
            mbarrier.wait(dq_done, (itr - 1) & 1)
            if DEBUG:
                tw_dqd += _clk(ctid) - t0
            _fence_after(ctid)
        mbarrier.wait(do_bar, ph)
        tcgen05_mma(v_smem, do_smem.permute((1, 0)), r_tmem, use_acc=False)
        tcgen05_commit(dp_ready)
        # dV += P^T dO
        _trace(trace_ptr, cta, 3, 100 + it * 16 + 3, ctid, TRACE)
        if DEBUG:
            t0 = _clk(ctid)
        mbarrier.wait(p_ready, ph)
        if DEBUG:
            tw_p += _clk(ctid) - t0
        _fence_after(ctid)
        tcgen05_mma(p_bf0, do_smem.slice(0, HM, dim=0), dv_tmem, use_acc=(it > it0))
        tcgen05_mma(p_bf1, do_smem.slice(HM, HM, dim=0), dv_tmem, use_acc=True)
        tcgen05_commit(do_empty)
        # dS(it) ready -> S^T(it+1) can overwrite the S region (P consumed by dV in-order)
        _trace(trace_ptr, cta, 3, 100 + it * 16 + 4, ctid, TRACE)
        if DEBUG:
            t0 = _clk(ctid)
        mbarrier.wait(ds_ready, ph)
        if DEBUG:
            tw_ds += _clk(ctid) - t0
        _fence_after(ctid)
        if it + 1 < n_iters:
            sq1 = (itr + 1) % Q_STAGES
            if DEBUG:
                t0 = _clk(ctid)
            mbarrier.wait(q_bar.index(sq1), ((itr + 1) // Q_STAGES) & 1)
            if DEBUG:
                tw_q += _clk(ctid) - t0
            _fence_after(ctid)
            tcgen05_mma(k_smem, q_smem.index(sq1).permute((1, 0)), s_tmem, use_acc=False)
            tcgen05_commit(s_ready)
        # dQ^T(it) -> R, dK += dS^T Q (dS^T from smem)
        tcgen05_mma(k_smem.permute((1, 0)), ds_smem, r_tmem, use_acc=False)
        tcgen05_commit(dq_ready)
        tcgen05_mma(ds_smem, q_smem.index(sq), dk_tmem, use_acc=(it > it0))
        tcgen05_commit(q_empty.index(sq))
    _trace(trace_ptr, cta, 3, 9999, ctid, TRACE)
    _dbg_add(dbg_ptr, 0, _clk(ctid) - t_start, ctid, DEBUG)
    _dbg_add(dbg_ptr, 1, tw_q, ctid, DEBUG)
    _dbg_add(dbg_ptr, 2, tw_dqd, ctid, DEBUG)
    _dbg_add(dbg_ptr, 3, tw_p, ctid, DEBUG)
    _dbg_add(dbg_ptr, 4, tw_ds, ctid, DEBUG)


# ----------------------------------------------------------------------------------------------
# softmax group (8 warps): columns [64*grp, 64*grp + 64), processed as two 32-column passes
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _softmax(s_tmem, r_tmem, ds_smem, lse_smem, delta_smem, q_bar, s_ready, dp_ready, p_ready, ds_ready,
             xb_local, seqlen, kb0, kb, it0, n_iters, scale_log2, xchg, swap_ds,
             grp: gl.constexpr, Q_STAGES: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr,
             num_warps: gl.constexpr, EXP_EMUL: gl.constexpr, XCHG: gl.constexpr, TRACE: gl.constexpr,
             trace_ptr, cta, dbg_ptr, DEBUG: gl.constexpr):
    CW: gl.constexpr = 32
    ctid = gl.arange(0, 32 * num_warps, layout=gl.BlockedLayout([1], [32], [num_warps], [0]))
    t_start = _clk(ctid)
    tw_s = gl.zeros_like(ctid).to(gl.int64)
    tw_dp = gl.zeros_like(ctid).to(gl.int64)
    tp_P = gl.zeros_like(ctid).to(gl.int64)
    tp_dS = gl.zeros_like(ctid).to(gl.int64)
    tw_rc = gl.zeros_like(ctid).to(gl.int64)
    _trace(trace_ptr, cta, grp, 1, ctid, TRACE)
    _trace(trace_ptr, cta, grp, 2, ctid, TRACE)
    c_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, CW], col_stride=1)
    rl: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, CW], c_tl, num_warps)
    prl: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BLOCK_N, CW], c_tl, num_warps)
    col_l: gl.constexpr = gl.SliceLayout(0, rl)
    C0: gl.constexpr = 64 * grp
    C1: gl.constexpr = 64 * grp + CW
    # bf16 view of the S region: group g owns bf16 columns [128g, 128g+128) = the 32-bit columns [64g, 64g+64)
    # holding its own S^T slices: P_g at [128g, 128g+64), dS_g at [128g+64, 128g+128)  (no cross-group overlap)
    PB0: gl.constexpr = 128 * grp
    pbuf = s_tmem._reinterpret(gl.bfloat16, [BLOCK_N, 2 * BLOCK_M],
                               TensorMemoryLayout(block=[BLOCK_N, 2 * BLOCK_M], col_stride=1))
    k_loc = kb * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, rl))
    k_valid = k_loc < seqlen
    col = gl.arange(0, CW, layout=col_l)

    for it in range(it0, n_iters):
        itr = it - it0
        sq = itr % Q_STAGES
        ph = itr & 1
        j = kb0 + it                                   # query block index; diagonal block is j == kb
        qa = j * BLOCK_M + C0 + col
        qb = j * BLOCK_M + C1 + col
        need_mask = (j == kb) | ((j + 1) * BLOCK_M > seqlen) | ((kb + 1) * BLOCK_N > seqlen)
        _trace(trace_ptr, cta, grp, 100 + it * 16 + 1, ctid, TRACE)
        mbarrier.wait(q_bar.index(sq), (itr // Q_STAGES) & 1)
        # ---------------- P phase
        _trace(trace_ptr, cta, grp, 100 + it * 16 + 2, ctid, TRACE)
        if DEBUG:
            t0 = _clk(ctid)
        mbarrier.wait(s_ready, ph)
        if DEBUG:
            t1 = _clk(ctid)
            tw_s += t1 - t0
        _fence_after(ctid)
        _trace(trace_ptr, cta, grp, 100 + it * 16 + 3, ctid, TRACE)
        st_a = s_tmem.slice(C0, CW).load(rl)
        st_b = s_tmem.slice(C1, CW).load(rl)
        lse_a = lse_smem.index(sq).slice(C0, CW).load(col_l)
        p_a = gl.exp2(st_a * scale_log2 - lse_a[None, :])
        if need_mask:
            p_a = gl.where((qa[None, :] >= k_loc[:, None]) & (qa[None, :] < seqlen) & k_valid[:, None], p_a, 0.0)
        pa_bf = gl.convert_layout(p_a.to(gl.bfloat16), prl)
        pbuf.slice(PB0, CW).store(pa_bf)
        lse_b = lse_smem.index(sq).slice(C1, CW).load(col_l)
        if EXP_EMUL:
            p_b = _exp2_emul(st_b * scale_log2 - lse_b[None, :])
        else:
            p_b = gl.exp2(st_b * scale_log2 - lse_b[None, :])
        if need_mask:
            p_b = gl.where((qb[None, :] >= k_loc[:, None]) & (qb[None, :] < seqlen) & k_valid[:, None], p_b, 0.0)
        pb_bf = gl.convert_layout(p_b.to(gl.bfloat16), prl)
        pbuf.slice(PB0 + CW, CW).store(pb_bf)
        _fence_before(ctid)
        gl.barrier()
        mbarrier.arrive(p_ready, count=1)
        if DEBUG:
            t2 = _clk(ctid)
            tp_P += t2 - t1
        # ---------------- dS phase
        _trace(trace_ptr, cta, grp, 100 + it * 16 + 4, ctid, TRACE)
        mbarrier.wait(dp_ready, ph)
        if DEBUG:
            t3 = _clk(ctid)
            tw_dp += t3 - t2
        _fence_after(ctid)
        _trace(trace_ptr, cta, grp, 100 + it * 16 + 5, ctid, TRACE)
        if DEBUG:
            t4 = _clk(ctid)
            tw_rc += t4 - t3
        dp_a = r_tmem.slice(C0, CW).load(rl)
        dp_b = r_tmem.slice(C1, CW).load(rl)
        delta_a = delta_smem.index(sq).slice(C0, CW).load(col_l)
        ds_a = (gl.convert_layout(pa_bf, rl).to(gl.float32) * (dp_a - delta_a[None, :])).to(gl.bfloat16)
        if XCHG and (swap_ds & (it >= 1)):
            ds_smem.slice(64 - C0, CW, dim=1).store(ds_a)
        else:
            ds_smem.slice(C0, CW, dim=1).store(ds_a)
        delta_b = delta_smem.index(sq).slice(C1, CW).load(col_l)
        ds_b = (gl.convert_layout(pb_bf, rl).to(gl.float32) * (dp_b - delta_b[None, :])).to(gl.bfloat16)
        if XCHG and (swap_ds & (it >= 1)):
            ds_smem.slice(64 - C0 + CW, CW, dim=1).store(ds_b)
        else:
            ds_smem.slice(C1, CW, dim=1).store(ds_b)
        fence_async_shared()
        _fence_before(ctid)
        gl.barrier()
        mbarrier.arrive(ds_ready, count=1)
        if DEBUG:
            tp_dS += _clk(ctid) - t4
    _trace(trace_ptr, cta, grp, 9999, ctid, TRACE)
    if grp == 0:
        _dbg_add(dbg_ptr, 5, _clk(ctid) - t_start, ctid, DEBUG)
        _dbg_add(dbg_ptr, 6, tw_s, ctid, DEBUG)
        _dbg_add(dbg_ptr, 7, tw_dp, ctid, DEBUG)
        _dbg_add(dbg_ptr, 8, tp_P, ctid, DEBUG)
        _dbg_add(dbg_ptr, 9, tp_dS, ctid, DEBUG)
        _dbg_add(dbg_ptr, 10, tw_rc, ctid, DEBUG)
    _trace(trace_ptr, cta, grp, 10000, ctid, TRACE)


# ----------------------------------------------------------------------------------------------
# dQ drain / exchange warps (4 warps)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _dq(r_tmem, dq_ready, dq_done, dq_acc_ptr, q_start, seqlen, h, kb, it0, n_iters, H,
        rank, xchg, xs_local, xs_remote, xb_local, xb_remote,
        BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr, HEAD_D: gl.constexpr, DQ_WARPS: gl.constexpr,
        XCHG: gl.constexpr, TRACE: gl.constexpr, trace_ptr, cta, dbg_ptr, DEBUG: gl.constexpr):
    """Drains the dQ^T tile (R region, [128 d x 128 q] fp32) in 8-column slices.
    Paired CTAs: tile columns are [partner-owned 64 | own 64] (the softmax wrote dS^T accordingly).
      1. (wait until the partner consumed our previous staging)  partner half: TMEM -> our staging buffer
         (local 16B stores; d-major with a 16B-chunk XOR swizzle); remote-arrive on the partner's READY.
      2. own half: TMEM -> registers; release the R region (dq_done) - no partner dependency.
      3. wait our READY, pull our columns from the partner's staging (16B remote loads, 2 batches), add, reduce
         into dq_accum; remote-arrive on the partner's CONSUMED.
    dq_accum has >= 128 padding rows, so no masking is needed (masked-out query columns are exactly zero)."""
    CW: gl.constexpr = 8
    NS: gl.constexpr = 64 // CW                  # slices per 64-column half
    c_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, CW], col_stride=1)
    rl: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, CW], c_tl, DQ_WARPS)
    d_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, rl))      # lane = d
    col = gl.arange(0, CW, layout=gl.SliceLayout(0, rl))
    tid = gl.arange(0, 32 * DQ_WARPS, layout=gl.BlockedLayout([1], [32], [DQ_WARPS], [0]))
    base = gl.cast(dq_acc_ptr, gl.int64, bitcast=True)
    goff = col[None, :] * (H * HEAD_D) + d_idx[:, None]              # dq_accum offset of (q=col, d) within a row
    c32_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, 32], col_stride=1)
    rl32: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, 32], c32_tl, DQ_WARPS)
    col32 = gl.arange(0, 32, layout=gl.SliceLayout(0, rl32))
    d32 = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, rl32))
    goff32 = col32[None, :] * (H * HEAD_D) + d32[:, None]
    drow = d_idx[:, None] * 64                                        # staging: 64 floats (256B) per d row
    dsw = d_idx[:, None] & 7                                          # 16B-chunk swizzle
    cl4 = col[None, :] & 3
    rd_local = xb_local + 16                                          # READY (partner arrives), CONSUMED at +24
    rd_remote = xb_remote + 16
    cs_local = xb_local + 24
    cs_remote = xb_remote + 24
    own_q0 = 64 * rank
    t_start = _clk(tid)
    tw_r = gl.zeros_like(tid).to(gl.int64)
    tw_cs = gl.zeros_like(tid).to(gl.int64)
    tw_rf = gl.zeros_like(tid).to(gl.int64)
    tp_send = gl.zeros_like(tid).to(gl.int64)
    tp_red = gl.zeros_like(tid).to(gl.int64)
    _trace(trace_ptr, cta, 4, 2, tid, TRACE)
    for it in range(it0, n_iters):
        itr = it - it0
        _trace(trace_ptr, cta, 4, 100 + it * 16 + 1, tid, TRACE)
        if DEBUG:
            t0 = _clk(tid)
        mbarrier.wait(dq_ready, itr & 1)
        if DEBUG:
            t1 = _clk(tid)
            tw_r += t1 - t0
        _fence_after(tid)
        _trace(trace_ptr, cta, 4, 100 + it * 16 + 2, tid, TRACE)
        row_off = ((q_start + (kb + it) * BLOCK_M) * H + h) * HEAD_D
        do_x = (xchg != 0) & (it >= 1)
        x = it - 1
        if XCHG and do_x:
            if x >= 1:
                _mbar_wait_local(cs_local + gl.zeros_like(tid), gl.zeros_like(tid) + ((x - 1) & 1))
            if DEBUG:
                t2 = _clk(tid)
                tw_cs += t2 - t1
            for c in gl.static_range(NS):                                        # 1. stage partner's columns
                v = r_tmem.slice(c * CW, CW).load(rl)
                sidx = drow + ((((c * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
                _st_local_v4(xs_local + gl.zeros_like(sidx), sidx, v)
            gl.barrier()
            _mbar_remote_arrive_pred(rd_remote + gl.zeros_like(tid), tid)
            if DEBUG:
                t3 = _clk(tid)
                tp_send += t3 - t2
            w0 = r_tmem.slice(64 + 0 * CW, CW).load(rl)                         # 2. own columns -> registers
            w1 = r_tmem.slice(64 + 1 * CW, CW).load(rl)
            w2 = r_tmem.slice(64 + 2 * CW, CW).load(rl)
            w3 = r_tmem.slice(64 + 3 * CW, CW).load(rl)
            w4 = r_tmem.slice(64 + 4 * CW, CW).load(rl)
            w5 = r_tmem.slice(64 + 5 * CW, CW).load(rl)
            w6 = r_tmem.slice(64 + 6 * CW, CW).load(rl)
            w7 = r_tmem.slice(64 + 7 * CW, CW).load(rl)
            _fence_before(tid)
            gl.barrier()
            mbarrier.arrive(dq_done, count=1)                                    # R fully drained
            _mbar_wait_local(rd_local + gl.zeros_like(tid), gl.zeros_like(tid) + (x & 1))
            if DEBUG:
                t4 = _clk(tid)
                tw_rf += t4 - t3
            rbase = base + 4 * (row_off + own_q0 * (H * HEAD_D)).to(gl.int64)
            # 3. pull partner's contribution in two batches of 4 slices, add, reduce
            s0 = drow + ((((0 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s1 = drow + ((((1 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s2 = drow + ((((2 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s3 = drow + ((((3 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            u0 = _ld_cluster_v4(xs_remote + gl.zeros_like(s0), s0)
            u1 = _ld_cluster_v4(xs_remote + gl.zeros_like(s1), s1)
            u2 = _ld_cluster_v4(xs_remote + gl.zeros_like(s2), s2)
            u3 = _ld_cluster_v4(xs_remote + gl.zeros_like(s3), s3)
            _red_add_f32(w0 + u0, rbase + (0 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w1 + u1, rbase + (1 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w2 + u2, rbase + (2 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w3 + u3, rbase + (3 * CW * (H * HEAD_D) * 4), goff)
            s4 = drow + ((((4 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s5 = drow + ((((5 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s6 = drow + ((((6 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            s7 = drow + ((((7 * CW + col[None, :]) >> 2) ^ dsw) << 2) + cl4
            u4 = _ld_cluster_v4(xs_remote + gl.zeros_like(s4), s4)
            u5 = _ld_cluster_v4(xs_remote + gl.zeros_like(s5), s5)
            u6 = _ld_cluster_v4(xs_remote + gl.zeros_like(s6), s6)
            u7 = _ld_cluster_v4(xs_remote + gl.zeros_like(s7), s7)
            gl.barrier()
            _mbar_remote_arrive_pred(cs_remote + gl.zeros_like(tid), tid)       # partner staging consumed
            _red_add_f32(w4 + u4, rbase + (4 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w5 + u5, rbase + (5 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w6 + u6, rbase + (6 * CW * (H * HEAD_D) * 4), goff)
            _red_add_f32(w7 + u7, rbase + (7 * CW * (H * HEAD_D) * 4), goff)
            if DEBUG:
                tp_red += _clk(tid) - t4
        else:
            # plain path: drain the whole tile (4 x32 TMEM loads) and release R before any reduction
            rb = base + 4 * row_off.to(gl.int64)
            v0 = r_tmem.slice(0, 32).load(rl32)
            v1 = r_tmem.slice(32, 32).load(rl32)
            v2 = r_tmem.slice(64, 32).load(rl32)
            v3 = r_tmem.slice(96, 32).load(rl32)
            _fence_before(tid)
            gl.barrier()
            mbarrier.arrive(dq_done, count=1)
            _red_add_f32(v0, rb, goff32)
            _red_add_f32(v1, rb + (32 * (H * HEAD_D) * 4), goff32)
            _red_add_f32(v2, rb + (64 * (H * HEAD_D) * 4), goff32)
            _red_add_f32(v3, rb + (96 * (H * HEAD_D) * 4), goff32)
    _trace(trace_ptr, cta, 4, 9999, tid, TRACE)
    _dbg_add(dbg_ptr, 11, _clk(tid) - t_start, tid, DEBUG)
    _dbg_add(dbg_ptr, 12, tw_r, tid, DEBUG)
    _dbg_add(dbg_ptr, 13, tw_cs, tid, DEBUG)
    _dbg_add(dbg_ptr, 14, tw_rf, tid, DEBUG)
    _dbg_add(dbg_ptr, 15, tp_send, tid, DEBUG)
    _dbg_add(dbg_ptr, 16, tp_red, tid, DEBUG)


# ----------------------------------------------------------------------------------------------
# main kernel
# ----------------------------------------------------------------------------------------------
@gluon.jit(do_not_specialize=["gen", "smem_bytes", "ds_word", "xb_word", "trace_addr", "T", "H"])
def fa_bwd_v4_kernel(q_desc, k_desc, v_desc, do_desc, lse_ptr, delta_ptr, dq_acc_ptr, dk_ptr, dv_ptr,
                     cu_ptr, items_ptr, T, H, scale, scale_log2, smem_bytes, ds_word, xb_word, trace_addr, ctr_ptr, gen,
                     dbg_ptr, off_ptr,
                     Q_STAGES: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr, HEAD_D: gl.constexpr,
                     DQ_WARPS: gl.constexpr, XCHG: gl.constexpr, EXP_EMUL: gl.constexpr, TRACE: gl.constexpr,
                     DEBUG: gl.constexpr, num_warps: gl.constexpr):
    pid = gl.program_id(0)
    lane_all = gl.arange(0, 32 * num_warps, layout=gl.BlockedLayout([1], [32], [num_warps], [0]))
    t_cta0 = _clk(lane_all)
    if TRACE:
        trace_ptr = gl.cast(trace_addr, gl.pointer_type(gl.int32), bitcast=True)
    else:
        trace_ptr = gl.cast(dq_acc_ptr, gl.pointer_type(gl.int32), bitcast=True)   # unused
    item = pid // 2
    rank = pid % 2
    doc = gl.load(items_ptr + item * 4 + 0)
    kb0 = gl.load(items_ptr + item * 4 + 1)
    hh = gl.load(items_ptr + item * 4 + 2)
    xchg = gl.load(items_ptr + item * 4 + 3)
    kb = kb0 + rank * xchg                       # paired: partner has the next key block; lone: same kb, other head
    h = (hh & 0xFF) * (1 - rank) + (hh >> 8) * rank
    q_start = gl.load(cu_ptr + doc)
    seqlen = gl.load(cu_ptr + doc + 1) - q_start
    n_iters = gl.cdiv(seqlen, BLOCK_M) - kb0     # iteration index it = j - kb0 (shared by the pair)
    it0 = rank * xchg                            # the partner (kb0+1) starts one block later
    swap_ds = (xchg != 0) & (rank == 0)          # paired rank 0: write dS^T halves swapped -> dQ^T = [partner | own]
    col0 = h * HEAD_D

    # ---- shared memory
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, HEAD_D], k_desc.layout)
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, HEAD_D], v_desc.layout)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [Q_STAGES, BLOCK_M, HEAD_D], q_desc.layout)
    do_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, HEAD_D], do_desc.layout)          # single stage
    ds_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_M], gl.bfloat16)
    ds_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, BLOCK_M], ds_layout)
    vec_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    lse_smem = gl.allocate_shared_memory(gl.float32, [Q_STAGES, BLOCK_M], vec_layout)
    delta_smem = gl.allocate_shared_memory(gl.float32, [Q_STAGES, BLOCK_M], vec_layout)
    xbuf = gl.allocate_shared_memory(gl.int32, [64], vec_layout)
    xs_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_N * 64], vec_layout)                   # exchange staging (32KB)

    # ---- barriers
    kv_bar = mbarrier.allocate_mbarrier()
    q_bar = mbarrier.allocate_mbarrier(batch=Q_STAGES)
    q_empty = mbarrier.allocate_mbarrier(batch=Q_STAGES)
    do_bar = mbarrier.allocate_mbarrier()
    do_empty = mbarrier.allocate_mbarrier()
    s_ready = mbarrier.allocate_mbarrier()
    dp_ready = mbarrier.allocate_mbarrier()
    p_ready = mbarrier.allocate_mbarrier()
    ds_ready = mbarrier.allocate_mbarrier()
    dq_ready = mbarrier.allocate_mbarrier()
    dq_done = mbarrier.allocate_mbarrier()
    mbarrier.init(kv_bar, count=1)
    for s in gl.static_range(Q_STAGES):
        mbarrier.init(q_bar.index(s), count=2)
        mbarrier.init(q_empty.index(s), count=1)
    mbarrier.init(do_bar, count=1)
    mbarrier.init(do_empty, count=1)
    mbarrier.init(s_ready, count=1)
    mbarrier.init(dp_ready, count=1)
    mbarrier.init(p_ready, count=2)
    mbarrier.init(ds_ready, count=2)
    mbarrier.init(dq_ready, count=1)
    mbarrier.init(dq_done, count=1)
    fence_async_shared()

    # ---- tensor memory: S 128 + R 128 + dK 128 + dV 128 = 512 cols
    s_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, BLOCK_M], col_stride=1)
    acc_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, HEAD_D], col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, BLOCK_M], s_tl)
    r_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, BLOCK_M], s_tl)
    dk_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, HEAD_D], acc_tl)
    dv_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, HEAD_D], acc_tl)

    # ---- K, V loads
    mbarrier.expect(kv_bar, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
    k_row = q_start + kb * BLOCK_N
    tma.async_copy_global_to_shared(k_desc, [k_row, col0], kv_bar, k_smem)
    tma.async_copy_global_to_shared(v_desc, [k_row, col0], kv_bar, v_smem)

    # ---- locate the dS buffer and xbuf in the dynamic smem window (sentinel scan); set up DSMEM addresses
    tid = gl.arange(0, 32 * num_warps, layout=gl.BlockedLayout([1], [32], [num_warps], [0]))
    _trace(trace_ptr, pid, 5, 1, tid, TRACE)
    xbuf.store(gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [num_warps], [0])) * 0 + xb_word)
    xs_smem.store(gl.full([BLOCK_N * 64], 7777.0, gl.float32, gl.BlockedLayout([4], [32], [num_warps], [0])))
    gl.barrier()
    if gen == 1:
        n_scan = (smem_bytes // 16 + 32 * num_warps - 1) // (32 * num_warps)
        ds_off, xb_off = _scan_smem(ds_word, xb_word, n_scan, smem_bytes, tid, 32 * num_warps)
        if pid == 0:
            gl.store(off_ptr + 0 + tid * 0, gl.zeros_like(tid) + ds_off, mask=tid == 0)
            gl.store(off_ptr + 1 + tid * 0, gl.zeros_like(tid) + xb_off, mask=tid == 0)
    else:
        ds_off = gl.load(off_ptr + 0)
        xb_off = gl.load(off_ptr + 1)
    xs_local = gl.max(_smem_addr(ds_off + gl.zeros_like(tid)), axis=0)
    xb_local = gl.max(_smem_addr(xb_off + gl.zeros_like(tid)), axis=0)
    peer = (1 - rank).to(gl.uint32)
    xs_remote = gl.max(_mapa(xs_local + gl.zeros_like(tid), peer + gl.zeros_like(tid)), axis=0)
    xb_remote = gl.max(_mapa(xb_local + gl.zeros_like(tid), peer + gl.zeros_like(tid)), axis=0)
    _mbar_init_pred(xb_local + 16 + gl.zeros_like(tid), tid)
    _mbar_init_pred(xb_local + 24 + gl.zeros_like(tid), tid)
    _asm_u32("{ fence.mbarrier_init.release.cluster; mov.u32 $0, $1; }", tid)
    fence_async_shared()
    gl.barrier()
    if XCHG and (xchg != 0):
        # cluster handshake through a global counter: the partner must have initialized its raw barriers before
        # we ever arrive on them remotely (cannot use barrier.cluster: the CTA has idle padding warps)
        gl.atomic_add(ctr_ptr + item, 1, sem="release")
        target = 2 * gen
        seen = gl.atomic_add(ctr_ptr + item, 0, sem="acquire")
        while seen < target:
            seen = gl.atomic_add(ctr_ptr + item, 0, sem="acquire")
    _trace(trace_ptr, pid, 6, ds_off, tid, TRACE)
    _trace(trace_ptr, pid, 7, xb_off, tid, TRACE)
    _trace(trace_ptr, pid, 5, 2, tid, TRACE)
    mbarrier.wait(kv_bar, 0)
    _trace(trace_ptr, pid, 5, 3, tid, TRACE)
    _dbg_add(dbg_ptr, 19, _clk(lane_all) - t_cta0, lane_all, DEBUG)

    gl.warp_specialize([
        (_softmax, (s_tmem, r_tmem, ds_smem, lse_smem, delta_smem, q_bar, s_ready, dp_ready, p_ready, ds_ready,
                    xb_local, seqlen, kb0, kb, it0, n_iters, scale_log2, xchg, swap_ds,
                    0, Q_STAGES, BLOCK_N, BLOCK_M, num_warps, EXP_EMUL, XCHG, TRACE, trace_ptr, pid, dbg_ptr, DEBUG)),
        (_softmax, (s_tmem, r_tmem, ds_smem, lse_smem, delta_smem, q_bar, s_ready, dp_ready, p_ready, ds_ready,
                    xb_local, seqlen, kb0, kb, it0, n_iters, scale_log2, xchg, swap_ds,
                    1, Q_STAGES, BLOCK_N, BLOCK_M, num_warps, EXP_EMUL, XCHG, TRACE, trace_ptr, pid, dbg_ptr, DEBUG)),
        (_producer, (q_desc, do_desc, q_smem, do_smem, lse_smem, delta_smem, q_bar, q_empty, do_bar, do_empty,
                     lse_ptr, delta_ptr, q_start, h, col0, it0, n_iters, kb0, T, trace_ptr, pid, dbg_ptr,
                     Q_STAGES, BLOCK_M, XCHG, TRACE, DEBUG)),
        (_mma, (k_smem, v_smem, q_smem, do_smem, ds_smem, s_tmem, r_tmem, dk_tmem, dv_tmem,
                q_bar, q_empty, do_bar, do_empty, s_ready, dp_ready, p_ready, ds_ready, dq_ready, dq_done, it0, n_iters,
                Q_STAGES, BLOCK_N, BLOCK_M, XCHG, TRACE, trace_ptr, pid, dbg_ptr, DEBUG)),
        (_dq, (r_tmem, dq_ready, dq_done, dq_acc_ptr, q_start, seqlen, h, kb0, it0, n_iters, H,
               rank, xchg, xs_local, xs_remote, xb_local, xb_remote,
               BLOCK_N, BLOCK_M, HEAD_D, DQ_WARPS, XCHG, TRACE, trace_ptr, pid, dbg_ptr, DEBUG)),
    ], worker_num_warps=[num_warps, 1, 1, DQ_WARPS], worker_num_regs=[128, 56, 40, 168])

    # ---- epilogue: wait for the last dK MMA, store dK, dV
    last = n_iters - 1 - it0
    _trace(trace_ptr, pid, 5, 4, tid, TRACE)
    mbarrier.wait(q_empty.index(last % Q_STAGES), (last // Q_STAGES) & 1)
    _fence_after(tid)
    _trace(trace_ptr, pid, 5, 5, tid, TRACE)
    HD2: gl.constexpr = HEAD_D // 2
    e_tl: gl.constexpr = TensorMemoryLayout(block=[BLOCK_N, HD2], col_stride=1)
    e_rl: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, HD2], e_tl, num_warps)
    rows = kb * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, e_rl))
    cols = gl.arange(0, HD2, layout=gl.SliceLayout(0, e_rl))
    mask = (rows[:, None] < seqlen) & (cols[None, :] < HD2)
    offs = ((q_start + rows[:, None]) * H + h) * HEAD_D + cols[None, :]
    for half in gl.static_range(2):
        dk = dk_tmem.slice(half * HD2, HD2).load(e_rl) * scale
        gl.store(dk_ptr + offs + half * HD2, dk.to(gl.bfloat16), mask=mask)
    for half in gl.static_range(2):
        dv = dv_tmem.slice(half * HD2, HD2).load(e_rl)
        gl.store(dv_ptr + offs + half * HD2, dv.to(gl.bfloat16), mask=mask)
    _trace(trace_ptr, pid, 5, 10000, tid, TRACE)
    _dbg_add(dbg_ptr, 20, _clk(lane_all) - t_cta0, lane_all, DEBUG)
    _dbg_add(dbg_ptr, 21, gl.zeros_like(lane_all).to(gl.int64) + (n_iters - it0), lane_all, DEBUG)
    _dbg_add(dbg_ptr, 22, gl.zeros_like(lane_all).to(gl.int64) + 1, lane_all, DEBUG)
    # keep the sentinel/exchange buffers alive (they are accessed only through inline PTX)
    xbuf._keep_alive()
    xs_smem._keep_alive()


# ----------------------------------------------------------------------------------------------
# pre/post kernels (plain Triton)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _preprocess_kernel(o_ptr, do_ptr, delta_ptr, T, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    m = rows < T
    offs = (rows[:, None] * H + h) * D + cols[None, :]
    o = tl.load(o_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32)
    tl.store(delta_ptr + h * T + rows, tl.sum(o * do, axis=1), mask=m)


@triton.jit
def _postprocess_kernel(acc_ptr, dq_ptr, scale, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    x = tl.load(acc_ptr + offs, mask=m, other=0.0)
    tl.store(dq_ptr + offs, (x * scale).to(tl.bfloat16), mask=m)


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------
def build_pair_schedule(cu_seqlens_cpu, H, block=128, head_group=2):
    """Items of 2 CTAs (one cluster).  Paired item: key blocks (kb, kb+1) of the same (doc, head), exchanging dQ
    partials.  Lone item (odd #key-blocks): the last key block for heads (h, h+1), no exchange.
    Ordered by head group (outer) and descending work (inner).  Row = [doc, kb0, h0 | h1<<8, xchg]."""
    cu = cu_seqlens_cpu.tolist()
    groups = [[] for _ in range(H // head_group)]
    for d in range(len(cu) - 1):
        L = cu[d + 1] - cu[d]
        nkb = (L + block - 1) // block
        for kb in range(0, nkb, 2):
            if kb + 1 < nkb:
                work = 2 * (nkb - kb) - 1
                for h in range(H):
                    groups[h // head_group].append((work, d, kb, h | (h << 8), 1))
            else:
                work = 2 * (nkb - kb)
                for h in range(0, H, 2):
                    groups[h // head_group].append((work, d, kb, h | ((h + 1) << 8), 0))
    rows = []
    for g in groups:
        g.sort(key=lambda x: -x[0])
        rows += [x[1:] for x in g]
    return torch.tensor(rows, dtype=torch.int32)


class FaBwd4:
    def __init__(self, cu_seqlens, T, H, D=HEAD_D, xchg=0, exp_emul=0, trace=0, debug=0):
        assert D == HEAD_D and H % 2 == 0
        self.T, self.H, self.D = T, H, D
        self.cu = cu_seqlens
        self.items = build_pair_schedule(cu_seqlens.cpu(), H).cuda()
        self.n_items = self.items.shape[0]
        self.xchg = xchg
        self.exp_emul = exp_emul
        self.dq_acc = torch.empty(T + BLOCK_M, H, D, device="cuda", dtype=torch.float32)   # +128 padding rows
        self.delta = torch.empty(H, T, device="cuda", dtype=torch.float32)
        self.q_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_D], gl.bfloat16)
        self.k_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_D], gl.bfloat16)
        self.ds_word = torch.tensor(7777.0, dtype=torch.float32).view(torch.int32).item()   # xs staging sentinel
        self.xb_word = XBUF_MAGIC
        self.smem_bytes = None
        self.trace = trace
        self.trace_buf = torch.zeros(2 * self.n_items * 8, dtype=torch.int32, pin_memory=True) if trace else None
        self.ctr = torch.zeros(self.n_items, dtype=torch.int32, device="cuda")
        self.gen = 0
        self.debug = debug
        self.dbg = torch.zeros(32, dtype=torch.int64, device="cuda")
        self.offs = torch.zeros(4, dtype=torch.int32, device="cuda")
        register_cluster("fa_bwd_v4_kernel", 2)

    def _args(self, q, k, v, do, lse, dk, dv, scale):
        T, H, D = q.shape
        q2 = q.view(T, H * D); k2 = k.view(T, H * D); v2 = v.view(T, H * D); do2 = do.view(T, H * D)
        q_desc = TensorDescriptor.from_tensor(q2, [BLOCK_M, HEAD_D], self.q_layout)
        do_desc = TensorDescriptor.from_tensor(do2, [BLOCK_M, HEAD_D], self.q_layout)
        k_desc = TensorDescriptor.from_tensor(k2, [BLOCK_N, HEAD_D], self.k_layout)
        v_desc = TensorDescriptor.from_tensor(v2, [BLOCK_N, HEAD_D], self.k_layout)
        return (q_desc, k_desc, v_desc, do_desc, lse, self.delta, self.dq_acc, dk, dv, self.cu, self.items, T, H,
                scale, scale * 1.4426950408889634, self.smem_bytes if self.smem_bytes else 0, self.ds_word, self.xb_word,
                self.trace_buf.data_ptr() if self.trace else 0, self.ctr, self.gen, self.dbg, self.offs)

    def _kw(self):
        return dict(Q_STAGES=Q_STAGES, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M, HEAD_D=HEAD_D, DQ_WARPS=4,
                    XCHG=self.xchg, EXP_EMUL=self.exp_emul, TRACE=self.trace, DEBUG=self.debug, num_warps=4)

    def __call__(self, q, k, v, o, do, lse, scale, dq=None, dk=None, dv=None):
        T, H, D = q.shape
        assert lse.shape == (H, T)
        if dq is None:
            dq = torch.empty_like(q)
        if dk is None:
            dk = torch.empty_like(k)
        if dv is None:
            dv = torch.empty_like(v)
        if self.smem_bytes is None:
            compiled = fa_bwd_v4_kernel.warmup(*self._args(q, k, v, do, lse, dk, dv, scale), grid=(self.n_items,), **self._kw())
            self.smem_bytes = compiled.metadata.shared
        _preprocess_kernel[(triton.cdiv(T, 64), H)](o, do, self.delta, T, H=H, D=D, BLOCK=64, num_warps=4)
        self.dq_acc.zero_()
        self.gen += 1
        # NOTE: with a cluster launch Triton multiplies the grid by the cluster size -> grid is in clusters
        fa_bwd_v4_kernel[(self.n_items,)](*self._args(q, k, v, do, lse, dk, dv, scale), **self._kw())
        n = T * H * D
        _postprocess_kernel[(triton.cdiv(n, 4096),)](self.dq_acc, dq, scale, n, BLOCK=4096, num_warps=4)
        return dq, dk, dv


DBG_NAMES = {0: "mma_total", 1: "mma_wait_qbar", 2: "mma_wait_dqdone", 3: "mma_wait_pready", 4: "mma_wait_dsready",
             5: "sm0_total", 6: "sm0_wait_sready", 7: "sm0_wait_dpready", 8: "sm0_phase_P", 9: "sm0_phase_dS",
             10: "sm0_wait_recvc", 11: "dq_total", 12: "dq_wait_dqready", 13: "dq_wait_consumed", 14: "dq_wait_ready",
             15: "dq_send", 16: "dq_recv_red", 17: "prod_total", 18: "prod_wait_qempty", 19: "cta_prologue",
             20: "cta_total"}


def report_debug(f):
    d = f.dbg.cpu().tolist()
    n_cta = 2 * f.n_items
    rep = max(d[22] // n_cta, 1)          # replication factor of the per-CTA debug stores
    iters = d[21] // rep
    d = [x // rep for x in d]
    print(f"  CTAs={n_cta} iterations={iters} avg iters/CTA={iters / max(n_cta, 1):.1f}")
    for i, n in DBG_NAMES.items():
        print(f"  {n:18s}: {d[i] / max(n_cta, 1):12.0f} clk/CTA   {d[i] / max(iters, 1):8.0f} clk/iter")
