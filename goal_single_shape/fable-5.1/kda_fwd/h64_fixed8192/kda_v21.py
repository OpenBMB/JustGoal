"""KDA forward, Gluon V21: V20 + (a) beta sigmoid deferred to its use (raw prefetch), (b) the O epilogue of chunk j-2
and the S pass of chunk j run in the gap while the T-inverse of chunk j is computed (O staging tile = the consumed
g_s stage of chunk j; M'^T has its own buffer), (c) the gates are no longer serialised across WGs (smem sequence counters gseq/g2seq order the loader and the g2 update;
used for the g2 ordering in part0's continuation region; the loader waits on tile_empty only; a WG arrives
the T-inverse warp is a slave of its WG via tcmd), (d) part1 publishes progress every 8 WG-chunks + at the end.
V20: symmetric dual-workgroup design (session 4).

Why: V18 (5 warp roles: chainA/chainB/gates/3x tinv/load) is instruction-FETCH bound (ncu: no_instruction = 43% of
stall samples, one miss per 128-byte line, ~106 KB of hot code in 5 distinct streams, 4 streams per scheduler).
V20 has two identical 4-warp workgroups (WG0 = default partition, WG1 = worker) that each run the WHOLE per-chunk
program for alternating chunks (WG w: chunks j = w, w+2, ...): prep (gates -> MMA1 -> acc1 processing -> E-MMAs ->
G epilogue) -> chain step (S pass -> chain MMA -> U epilogue -> S-update/O MMAs; serialised across WGs by s_bar) ->
O epilogue.  Two 1-warp T-inverse partitions (warp K serves WG K) and one loader warp.  Every per-WG buffer is
single-buffered (one chunk in flight per WG); pipelining comes from the two WGs being offset by one chunk.
Per scheduler: one warp of each WG (same 2 x ~40 KB program), one tinv or loader warp.

Math = kda_model3.py (identical to V13..V18): split-T (P=2) with part0 continuation, transposed full-v chain,
S^T [v, d] fp32 in TMEM, St^T = bf16(S^T) TMEM A operand, G^T = f * [Tn@L ; Q]^T (the Q half is now produced by the
gates directly: bf16(q * p * f)), U epilogue in place, S^T = Dc*S^T + Ub^T @ Kbar, O^T += Ub^T @ Lq^T.

Grid 2*H: pid < H -> part1 (chunks [NC0, NC) from S=0, writes the final state, publishes per-WG progress counters);
pid >= H -> part0 (chunks [0, NC0) from the initial state, continues while max_d log2(cumdecay since NC0) >= -THR).
Dynamic loop end n_end (smem): decided by the gates of chunk j' (n_end = j'+1); every partition re-reads it after
each wait; a WG signals its exit to its T-inverse warp (tcmd = -1) and to the loader (gseq = huge).
TMEM (512 cols): S_t 128 | St_t 64 | acc1 x2 (64 each) | G_t x2 (32 each) | UO x2 (64 each).
"""
import math
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, tma, mbarrier, tcgen05_mma, tcgen05_commit,
    fence_async_shared)

LOG2E = gl.constexpr(1.4426950408889634)
NEG5LOG2E = gl.constexpr(-5.0 * 1.4426950408889634)
INV_SQRT_D = gl.constexpr(1.0 / math.sqrt(128.0))
C = gl.constexpr(32)
D = gl.constexpr(128)
THR = gl.constexpr(64.0)
NS = gl.constexpr(3)       # input stages (q,k,g,v tiles) shared by the two WGs
PF = gl.constexpr(3)       # L2 prefetch distance (chunks)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _rcp(x):
    return gl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _sigmoid(x):
    return _rcp(1.0 + gl.exp2(x * (-LOG2E)))


@gluon.jit
def _tanh(x):
    return gl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _prefetch_l2(ptrs):
    return gl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _ld_acquire(ptr):
    return gl.inline_asm_elementwise("ld.acquire.gpu.global.s32 $0, [$1];", "=r,l", [ptr], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _st_release(ptr, val):
    return gl.inline_asm_elementwise("st.release.gpu.global.s32 [$1], $2; mov.u32 $0, 0;", "=r,l,r", [ptr, val],
                                     dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _st_volatile(ptr, val):
    return gl.inline_asm_elementwise("st.volatile.global.s32 [$1], $2; mov.u32 $0, 0;", "=r,l,r", [ptr, val],
                                     dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _fence_cta():
    return gl.inline_asm_elementwise("membar.cta; mov.u32 $0, 0;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _spin_pause():
    # side-effecting (keeps a spin loop alive, forces the re-load) and cheap: nanosleep instead of a fence
    return gl.inline_asm_elementwise("nanosleep.u32 32; mov.u32 $0, 0;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _smid():
    return gl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _clock():
    return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=gl.int64, is_pure=False, pack=1)


@gluon.jit
def _stamp(dbg, part: gl.constexpr, c, pt: gl.constexpr, NC: gl.constexpr, DEBUG: gl.constexpr):
    # dbg = (dbg_ptr, wd_ptr): dbg layout [pid, 5 partitions, NC + 8, 16]; wd (pinned host) [pid, 8] = last code
    # DEBUG: 0 = off, 1 = clock stamps + watchdog codes, 2 = watchdog codes only (near-production timing)
    if DEBUG == 1:
        dbg_ptr, wd_ptr = dbg
        gl.store(dbg_ptr + (part * (NC + 8) + c) * 16 + pt, _clock())
        _st_volatile(wd_ptr + part, c * 16 + pt)
    elif DEBUG == 2:
        dbg_ptr, wd_ptr = dbg
        _st_volatile(wd_ptr + part, c * 16 + pt)


@gluon.jit
def _ld_scalar(sm, layout: gl.constexpr):
    return gl.max(sm.load(layout), axis=0)


@gluon.jit
def _st_scalar(sm, val, layout: gl.constexpr):
    sm.store(gl.full([1], 0, gl.int32, layout) + val)


# ----------------------------------------------------------------------------------------------------------------
@gluon.jit
def load_partition(descs, smem, bars, q_ptr, k_ptr, v_ptr, g_ptr, h, c_base, n_max, dbg_ptr,
                   NC: gl.constexpr, H: gl.constexpr, DEBUG: gl.constexpr):
    q_desc, k_desc, v_desc, g_desc = descs
    qk_s, g_s, v_s, nend_s, gseq_s = smem
    tile_full, tile_empty = bars
    lane = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [1], [0]))
    row_off = lane.to(gl.int64) * (H * D)
    s1: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    # prologue: chunks 0 .. NS-1
    for jj in gl.static_range(NS):
        if jj < n_max:
            mbarrier.expect(tile_full.index(jj), 4 * q_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(q_desc, [(c_base + jj) * C, h * D], tile_full.index(jj), qk_s.index(jj).slice(0, C, dim=0))
            tma.async_copy_global_to_shared(k_desc, [(c_base + jj) * C, h * D], tile_full.index(jj), qk_s.index(jj).slice(C, C, dim=0))
            tma.async_copy_global_to_shared(g_desc, [(c_base + jj) * C, h * D], tile_full.index(jj), g_s.index(jj))
            tma.async_copy_global_to_shared(v_desc, [(c_base + jj) * C, h * D], tile_full.index(jj), v_s.index(jj))
    for jj in gl.static_range(NS, NS + PF):
        if jj < n_max:
            base = ((c_base + jj) * C).to(gl.int64) * (H * D) + h * D + row_off
            _prefetch_l2(q_ptr + base)
            _prefetch_l2(q_ptr + base + 64)
            _prefetch_l2(k_ptr + base)
            _prefetch_l2(k_ptr + base + 64)
            _prefetch_l2(g_ptr + base)
            _prefetch_l2(g_ptr + base + 64)
            _prefetch_l2(v_ptr + base)
            _prefetch_l2(v_ptr + base + 64)
    j = NS
    last = NS - 1                                                              # last chunk loaded (loads form a prefix)
    n_end = _ld_scalar(nend_s, s1)
    while j < n_end:
        stage = j % NS
        c = c_base + j
        _stamp(dbg_ptr, 4, j, 0, NC, DEBUG)
        # gates(j-2) done (gseq = last chunk whose gates finished; a WG stores a huge value on exit): bounds the loader
        # by the WGs' progress so that it never waits for the consumption of a chunk nobody will process
        gv = _ld_scalar(gseq_s.index((j - 2) & 1), s1)
        while gv < j - 2:
            _spin_pause()                       # side effect: keeps the spin loop (LLVM deletes side-effect-free loops)
            gv = _ld_scalar(gseq_s.index((j - 2) & 1), s1)
        n_end = _ld_scalar(nend_s, s1)
        if j < n_end:
            _stamp(dbg_ptr, 4, j, 1, NC, DEBUG)
            mbarrier.wait(tile_empty.index(stage), ((j // NS) - 1) & 1)      # chunk j-NS consumed (its U' E-MMA done)
            _stamp(dbg_ptr, 4, j, 2, NC, DEBUG)
            mbarrier.expect(tile_full.index(stage), 4 * q_desc.block_type.nbytes)
            last = j
            tma.async_copy_global_to_shared(q_desc, [c * C, h * D], tile_full.index(stage), qk_s.index(stage).slice(0, C, dim=0))
            tma.async_copy_global_to_shared(k_desc, [c * C, h * D], tile_full.index(stage), qk_s.index(stage).slice(C, C, dim=0))
            tma.async_copy_global_to_shared(g_desc, [c * C, h * D], tile_full.index(stage), g_s.index(stage))
            tma.async_copy_global_to_shared(v_desc, [c * C, h * D], tile_full.index(stage), v_s.index(stage))
            if j + PF < n_max:
                base = ((c + PF) * C).to(gl.int64) * (H * D) + h * D + row_off
                _prefetch_l2(q_ptr + base)
                _prefetch_l2(q_ptr + base + 64)
                _prefetch_l2(k_ptr + base)
                _prefetch_l2(k_ptr + base + 64)
                _prefetch_l2(g_ptr + base)
                _prefetch_l2(g_ptr + base + 64)
                _prefetch_l2(v_ptr + base)
                _prefetch_l2(v_ptr + base + 64)
        j += 1
    # no TMA load may be in flight when the CTA joins / invalidates the barriers (a phantom chunk's tile is never
    # waited for by anybody): wait for the latest load of every stage (already-complete phases return at once)
    for s in gl.static_range(NS):
        jw = last - s
        if jw >= 0:
            mbarrier.wait(tile_full.index(jw % NS), (jw // NS) & 1)
    _stamp(dbg_ptr, 4, j, 3, NC, DEBUG)


# ----------------------------------------------------------------------------------------------------------------
@gluon.jit
def tinv_partition(smem, bars, beta_ptr, h, c_base, n_max, dbg_ptr, K: gl.constexpr, NC: gl.constexpr, H: gl.constexpr,
                   DEBUG: gl.constexpr, ABL: gl.constexpr):
    """T-inverse warp K: a slave of WG K.  Each m_ready[K] arrival comes with tcmd_s[K] = chunk index (or -1 = stop);
    solves X <- M'^-1 diag(n beta), stores T^T / Tn^T tiles for the E-MMAs, arrives t_ready[K]."""
    TB, TN, MT, n_s, invn_s, tcmd_s = smem
    m_ready, t_ready = bars
    col_layout: gl.constexpr = gl.BlockedLayout([C, 1], [1, 32], [1, 1], [0, 1])
    s1: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    ri = gl.arange(0, C, layout=gl.SliceLayout(1, col_layout))[:, None]
    lane = gl.arange(0, C, layout=gl.SliceLayout(0, col_layout))
    MTk = MT.index(K)
    beta_next = gl.load(beta_ptr + ((c_base + K) * C + lane) * H + h)
    it = 0
    cont = True
    while cont:
        mbarrier.wait(m_ready.index(K), it & 1)
        jt = _ld_scalar(tcmd_s.index(K), s1)
        _stamp(dbg_ptr, 2 + K, it * 2 + K, 0, NC, DEBUG)
        if jt < 0:
            cont = False
        else:
            _stamp(dbg_ptr, 2 + K, jt, 1, NC, DEBUG)
            beta_lane = _sigmoid(beta_next.to(gl.float32))
            beta_next = gl.load(beta_ptr + ((c_base + jt + 2) * C + lane) * H + h, mask=(lane >= 0) & (jt + 2 < n_max),
                                other=0.0)
            n_lane = n_s.index(K).slice(0, C).load(gl.SliceLayout(0, col_layout))
            X = gl.where(ri == lane[None, :], (n_lane * beta_lane)[None, :], 0.0)
            m0 = MTk.slice(0, 1, dim=0).slice(0, C, dim=1).permute((1, 0)).load(col_layout)
            m1 = MTk.slice(1, 1, dim=0).slice(0, C, dim=1).permute((1, 0)).load(col_layout)
            for r in gl.static_range(C - 1 if not (ABL & 1) else 0):
                trow = gl.sum(gl.where(ri == r, X, -0.0), axis=0)
                if r % 2 == 0:
                    X = gl.where(ri > r, X - m0 * gl.expand_dims(trow, 0), X)
                    if r + 2 < C - 1:
                        m0 = MTk.slice(r + 2, 1, dim=0).slice(0, C, dim=1).permute((1, 0)).load(col_layout)
                else:
                    X = gl.where(ri > r, X - m1 * gl.expand_dims(trow, 0), X)
                    if r + 2 < C - 1:
                        m1 = MTk.slice(r + 2, 1, dim=0).slice(0, C, dim=1).permute((1, 0)).load(col_layout)
            invn_rows = invn_s.index(K).slice(0, C).load(gl.SliceLayout(1, col_layout))
            Tm = X * invn_rows[:, None]
            _stamp(dbg_ptr, 2 + K, jt, 2, NC, DEBUG)
            TB.index(K).store(gl.permute(Tm.to(gl.bfloat16), (1, 0)))                       # TB[t', t] = T[t, t']
            TN.index(K).store(gl.permute((-(Tm * n_lane[None, :])).to(gl.bfloat16), (1, 0)))  # TN[t', t] = Tn[t, t']
            fence_async_shared()
            mbarrier.arrive(t_ready.index(K), count=1)
            _stamp(dbg_ptr, 2 + K, jt, 3, NC, DEBUG)
        it += 1
    _stamp(dbg_ptr, 2 + K, it * 2 + K, 4, NC, DEBUG)


# ----------------------------------------------------------------------------------------------------------------
@gluon.jit
def _state_init_q(S_t, state_ptr, h, col: gl.constexpr, zero: gl.constexpr, s_reg: gl.constexpr):
    sv = gl.arange(0, D, layout=gl.SliceLayout(1, s_reg))
    sd = gl.arange(0, C, layout=gl.SliceLayout(0, s_reg))
    if zero:
        S_t.slice(col, 32).store(gl.zeros([D, C], gl.float32, s_reg))
    else:
        ptrs = state_ptr + h * D * D + sv[:, None] * D + col + sd[None, :]
        S_t.slice(col, 32).store(gl.load(ptrs).to(gl.float32))


@gluon.jit
def _state_store_q(S_t, state_ptr, h, col: gl.constexpr, s_reg: gl.constexpr):
    sv = gl.arange(0, D, layout=gl.SliceLayout(1, s_reg))
    sd = gl.arange(0, C, layout=gl.SliceLayout(0, s_reg))
    ptrs = state_ptr + h * D * D + sv[:, None] * D + col + sd[None, :]
    gl.store(ptrs, S_t.slice(col, 32).load(s_reg).to(gl.bfloat16))


@gluon.jit
def _s_pass_q(S_t, St_t, dc_vec, c0: gl.constexpr, s_reg: gl.constexpr, st_reg: gl.constexpr, ABL: gl.constexpr):
    """Columns c0:c0+32: St^T = bf16(S^T), then S^T *= Dc."""
    dc_layout: gl.constexpr = gl.SliceLayout(0, s_reg)
    x0 = S_t.slice(c0, 32).load(s_reg)
    St_t.slice(c0, 32).store(gl.convert_layout(x0.to(gl.bfloat16), st_reg))
    if not (ABL & 8):
        d0 = dc_vec.slice(c0, 32).load(dc_layout)
        S_t.slice(c0, 32).store(x0 * d0[None, :])


@gluon.jit
def _o_epilogue(o_desc, O_w, UOw, nq_v, prog_ptr, h, jm, cm, part1, epoch_base, s_reg: gl.constexpr, W: gl.constexpr,
                NC0: gl.constexpr, ABL: gl.constexpr):
    """O epilogue of my chunk jm (= c_base + ... = cm): o = nq_t * O^T -> O_w [t, v] -> TMA store.  part0's
    continuation chunks wait until part1's store of the same chunk has landed."""
    o = UOw.slice(C, C).load(s_reg)
    nq_col = nq_v.slice(C, C).load(gl.SliceLayout(0, s_reg))
    ob = (o * nq_col[None, :]).to(gl.bfloat16)
    if not part1:
        if cm >= NC0:
            i1 = cm - NC0
            target = epoch_base + (i1 >> 1) + 1
            v = _ld_acquire(prog_ptr + 2 * h + (i1 & 1))
            while v < target:
                v = _ld_acquire(prog_ptr + 2 * h + (i1 & 1))
    O_w.store(gl.permute(ob, (1, 0)))
    fence_async_shared()
    gl.barrier()
    if ABL & 4:
        tma.async_copy_shared_to_global(o_desc, [cm * C, h * D], O_w.slice(0, 1, dim=0))
    else:
        tma.async_copy_shared_to_global(o_desc, [cm * C, h * D], O_w)


@gluon.jit
def wg_partition(o_desc, smem, tmem, bars, state_ptr, prog_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, h, c_base, part1,
                 part0_cont, wstate, epoch_base, dbg_ptr, W: gl.constexpr, NC: gl.constexpr, NC0: gl.constexpr,
                 MID: gl.constexpr, H: gl.constexpr, DEBUG: gl.constexpr, ABL: gl.constexpr):
    (qk_s, g_s, v_s, LQ, KB, G_sm, TB, TN, LQ0, MT, f_s, dc_s, n_s, nq_s, invn_s, g2_s, nend_s, tcmd_s, gseq_s,
     g2seq_s) = smem
    S_t, St_t, acc1, G_t, UO = tmem
    (tile_full, tile_empty, m_ready, t_ready, mma1_bar, g_bar, uinit, chain_bar, o_bar, s_bar, init_bar,
     wg_done) = bars
    num_warps: gl.constexpr = 4
    # ---- layouts ----
    col_layout: gl.constexpr = gl.BlockedLayout([C, 1], [1, 32], [1, num_warps], [0, 1])   # gates: [32 t, 128 d]
    lq_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([D, 2 * C], gl.bfloat16)
    s1: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    s_tl: gl.constexpr = TensorMemoryLayout((D, C), col_stride=1)
    s_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (D, C), s_tl, num_warps)
    st_reg: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, (D, C), s_tl, num_warps)
    acc1_tl: gl.constexpr = TensorMemoryLayout((2 * C, C), col_stride=1)
    acc1_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (2 * C, C), acc1_tl, num_warps)
    ub_tl: gl.constexpr = TensorMemoryLayout((D, D), col_stride=1)
    dvec: gl.constexpr = gl.SliceLayout(0, col_layout)      # per-d vectors in the gates
    # ---- per-WG buffers ----
    LQw = LQ.index(W)
    KBw = KB.index(W)
    Gw = G_sm.index(W)
    LQ0w = LQ0.index(W)
    MTw = MT.index(W)
    acc1w = acc1.index(W)
    G_tw = G_t.index(W)
    UOw = UO.index(W)
    # ---- state init: my column half ----
    if part1:
        _state_init_q(S_t, state_ptr, h, 64 * W, True, s_reg)
        _state_init_q(S_t, state_ptr, h, 64 * W + 32, True, s_reg)
    else:
        _state_init_q(S_t, state_ptr, h, 64 * W, False, s_reg)
        _state_init_q(S_t, state_ptr, h, 64 * W + 32, False, s_reg)
    mbarrier.arrive(init_bar, count=1)
    if DEBUG == 1:
        gl.store(dbg_ptr[0] + 15, _smid().to(gl.int64))
    # ---- gates constants ----
    d_idx = gl.arange(0, D, layout=dvec)
    t_idx = gl.arange(0, C, layout=gl.SliceLayout(1, col_layout))
    a_h = gl.exp(gl.load(A_log_ptr + h))
    bias = gl.load(dt_bias_ptr + h * D + d_idx)
    a_half = gl.full([D], 1.0, gl.float32, dvec) * (a_h * 0.5)
    bias_half = bias * (a_h * 0.5)
    # ---- acc1 constants ----
    r64 = gl.arange(0, 2 * C, layout=gl.SliceLayout(1, acc1_reg))[:, None]
    cs = gl.arange(0, C, layout=gl.SliceLayout(0, acc1_reg))[None, :]
    r64v = gl.arange(0, 2 * C, layout=gl.SliceLayout(1, acc1_reg))
    beta_raw = gl.load(beta_ptr + ((c_base + W) * C + r64v) * H + h, mask=r64v < C, other=0.0)
    mbarrier.wait(init_bar, 0)
    j = W
    jl = -1                                                                    # last chunk I processed
    n_end = _ld_scalar(nend_s, s1)
    while j < n_end:
        stage = j % NS
        ps = (j // NS) & 1
        jj = j >> 1
        pw = jj & 1
        c = c_base + j
        _stamp(dbg_ptr, W, j, 0, NC, DEBUG)
        if part0_cont & (c >= NC0):
            # continuation region: wait for the other WG's decision at chunk j-1 before reading n_end, so that this
            # chunk is never a phantom (a WG entering a cancelled chunk correlates with a join hang, see NOTES)
            gv0 = _ld_scalar(g2seq_s, s1)
            while gv0 < j - 1:
                _spin_pause()
                gv0 = _ld_scalar(g2seq_s, s1)
            _fence_cta()
        n_end = _ld_scalar(nend_s, s1)
        if j < n_end:
            jl = j
            # ================= P1: gates =================
            mbarrier.wait(tile_full.index(stage), ps)
            _stamp(dbg_ptr, W, j, 1, NC, DEBUG)
            qb = qk_s.index(stage).slice(0, C, dim=0).load(col_layout)
            kb = qk_s.index(stage).slice(C, C, dim=0).load(col_layout)
            g = g_s.index(stage).load(col_layout).to(gl.float32)
            if ABL & 2:
                u_m = gl.sum(g, axis=0) * 0.001
                u_e = u_m
                p = g
                ip = g
                cK = u_m
                f = u_m
            else:
                xh = g * a_half[None, :] + bias_half[None, :]
                sg = _tanh(xh) * 0.5 + 0.5
                u = gl.associative_scan(sg, 0, _add)
                u_m = gl.sum(gl.where(t_idx[:, None] == MID, u, -0.0), axis=0)
                u_e = gl.sum(gl.where(t_idx[:, None] == C - 1, u, -0.0), axis=0)
                y = u * NEG5LOG2E - (u_m * NEG5LOG2E)[None, :]
                p = gl.exp2(y)
                if ABL & 32:
                    ip = (0x7EF311C7 - p.to(gl.int32, bitcast=True)).to(gl.float32, bitcast=True)
                    ip = ip * (2.0 - p * ip)
                    ip = ip * (2.0 - p * ip)
                else:
                    ip = _rcp(p)
                cK = gl.exp2((u_e - u_m) * NEG5LOG2E)
                f = gl.exp2(u_m * NEG5LOG2E)
            pb = p.to(gl.bfloat16)
            ipb = ip.to(gl.bfloat16)
            Lt = kb * pb
            Rt = kb * ipb
            Qt = qb * pb
            Qrt = qb * ipb
            Kbar = Rt * cK.to(gl.bfloat16)[None, :]
            Qf = (qb.to(gl.float32) * (p * f[None, :])).to(gl.bfloat16)          # f-scaled Q for G^T cols 32:64
            _stamp(dbg_ptr, W, j, 2, NC, DEBUG)
            # continuation criterion (part0, chunks >= NC0): stop after the chunk whose cumulative decay < 2^-THR.
            # g2 is accumulated in chunk order: wait for the other WG's previous continuation chunk.
            if part0_cont & (c >= NC0):
                gv2 = _ld_scalar(g2seq_s, s1)           # g2 is accumulated in chunk order (g2seq = last chunk in)
                while gv2 < j - 1:
                    _spin_pause()
                    gv2 = _ld_scalar(g2seq_s, s1)
                _fence_cta()
                n_end = _ld_scalar(nend_s, s1)          # fresh: the other WG may have decided at chunk j-1
                G2 = g2_s.load(dvec) + u_e * NEG5LOG2E
                g2_s.store(G2)
                if gl.max(G2, axis=0) < -THR:
                    if j + 1 < n_end:                    # monotone update (never raise n_end)
                        n_end = j + 1
                        _st_scalar(nend_s, n_end, s1)
                _fence_cta()
                gl.barrier()
                _st_scalar(g2seq_s, j, s1)
            _fence_cta()
            _st_scalar(gseq_s.index(W), j, s1)
            f_s.index(W).store(f)
            dc_s.index(W).store(gl.exp2(u_e * NEG5LOG2E))
            KBw.store(gl.permute(Kbar, (1, 0)))
            LQw.slice(0, C, dim=1).store(gl.permute(Lt, (1, 0)))
            LQw.slice(C, C, dim=1).store(gl.permute(Qt, (1, 0)))
            Gw.slice(C, C, dim=1).store(gl.permute(Qf, (1, 0)))
            gl.barrier()                                                       # all warps done reading q/k
            R_s = qk_s.index(stage)._reinterpret(gl.bfloat16, [D, 2 * C], lq_layout)
            R_s.slice(0, C, dim=1).store(gl.permute(Rt, (1, 0)))
            R_s.slice(C, C, dim=1).store(gl.permute(Qrt, (1, 0)))
            fence_async_shared()
            _stamp(dbg_ptr, W, j, 3, NC, DEBUG)
            tcgen05_mma(LQw.permute((1, 0)), R_s, acc1w, use_acc=False, mbarriers=[mma1_bar.index(W)])
            # ================= P2: acc1 processing =================
            beta_row = _sigmoid(beta_raw.to(gl.float32))
            beta_raw = gl.load(beta_ptr + ((c + 2) * C + r64v) * H + h, mask=(r64v < C) & (j + 2 < n_end), other=0.0)
            mbarrier.wait(mma1_bar.index(W), pw)
            _stamp(dbg_ptr, W, j, 4, NC, DEBUG)
            acc_lo = acc1w.slice(0, C).load(acc1_reg)
            acc_hi = acc1w.slice(C, C).load(acc1_reg)
            nk2 = gl.sum(gl.where(r64 == cs, acc_lo, -0.0), axis=1)
            nq2 = gl.sum(gl.where(r64 == cs + C, acc_hi, -0.0), axis=1)
            n_row = gl.where(r64v < C, gl.rsqrt(nk2), 1.0)
            invn_row = gl.where(r64v < C, gl.sqrt(nk2), 1.0)
            nq_row = gl.where(r64v >= C, gl.rsqrt(nq2) * INV_SQRT_D, 1.0)
            Mm = gl.where(r64 == cs, 1.0, 0.0) + gl.where((r64 > cs) & (r64 < C), acc_lo, 0.0) * (beta_row * n_row * n_row)[:, None]
            Lq = gl.where((r64 >= C) & (r64 - C >= cs), acc_lo, 0.0).to(gl.bfloat16)
            n_s.index(W).store(n_row)
            nq_s.index(2 * W + (jj & 1)).store(nq_row)
            invn_s.index(W).store(invn_row)
            LQ0w.store(Lq)
            MTw.store(gl.permute(Mm, (1, 0)))                                  # MT[r, i] = M'[i, r]
            _st_scalar(tcmd_s.index(W), j, s1)
            fence_async_shared()
            mbarrier.arrive(m_ready.index(W), count=1)
            _stamp(dbg_ptr, W, j, 5, NC, DEBUG)
            # ================= gap (T-inverse running): O epilogue of chunk j-2, S pass of chunk j =================
            if jj >= 1:
                mbarrier.wait(o_bar.index(W), (jj - 1) & 1)
                _o_epilogue(o_desc, g_s.index(stage), UOw, nq_s.index(2 * W + ((jj - 1) & 1)), prog_ptr, h, j - 2, c - 2,
                            part1, epoch_base, s_reg, W, NC0, ABL)
            _stamp(dbg_ptr, W, j, 6, NC, DEBUG)
            if j >= 1:
                mbarrier.wait(s_bar, (j & 1) ^ 1)                              # S-update(j-1) done
            _stamp(dbg_ptr, W, j, 7, NC, DEBUG)
            dcw = dc_s.index(W)
            _s_pass_q(S_t, St_t, dcw, 0, s_reg, st_reg, ABL)
            _s_pass_q(S_t, St_t, dcw, 32, s_reg, st_reg, ABL)
            _s_pass_q(S_t, St_t, dcw, 64, s_reg, st_reg, ABL)
            _s_pass_q(S_t, St_t, dcw, 96, s_reg, st_reg, ABL)
            _stamp(dbg_ptr, W, j, 8, NC, DEBUG)
            # ================= P4: E-MMAs + G epilogue =================
            mbarrier.wait(t_ready.index(W), pw)
            _stamp(dbg_ptr, W, j, 9, NC, DEBUG)
            tcgen05_mma(LQw.slice(0, C, dim=1), TN.index(W), G_tw, use_acc=False, mbarriers=[g_bar.index(W)])
            tma.store_wait(0)                                                  # O store(j-2) done reading g_s[stage]
            gl.barrier()
            tcgen05_mma(v_s.index(stage).permute((1, 0)), TB.index(W), UOw.slice(0, C), use_acc=False,
                        mbarriers=[uinit.index(W), tile_empty.index(stage)])
            if part1:
                if (jj & 7) == 7:
                    _st_release(prog_ptr + 2 * h + W, epoch_base + jj)         # my chunks with index < j are stored
            mbarrier.wait(g_bar.index(W), pw)
            f_row = f_s.index(W).load(gl.SliceLayout(1, s_reg))
            yg = G_tw.load(s_reg)
            Gw.slice(0, C, dim=1).store((yg * f_row[:, None]).to(gl.bfloat16))
            fence_async_shared()
            _stamp(dbg_ptr, W, j, 10, NC, DEBUG)
            # ================= P5: chain step =================
            mbarrier.wait(uinit.index(W), pw)
            _stamp(dbg_ptr, W, j, 11, NC, DEBUG)
            tcgen05_mma(St_t, Gw.slice(0, C, dim=1), UOw.slice(0, C), use_acc=True)
            tcgen05_mma(St_t, Gw.slice(C, C, dim=1), UOw.slice(C, C), use_acc=False, mbarriers=[chain_bar.index(W)])
            mbarrier.wait(chain_bar.index(W), pw)
            _stamp(dbg_ptr, W, j, 12, NC, DEBUG)
            u = UOw.slice(0, C).load(s_reg)
            n_col = n_s.index(W).slice(0, C).load(gl.SliceLayout(0, s_reg))
            Ub_t = UOw._reinterpret(gl.bfloat16, [D, D], ub_tl).slice(0, C)
            Ub_t.store(gl.convert_layout((u * n_col[None, :]).to(gl.bfloat16), st_reg))
            tcgen05_mma(Ub_t, KBw.permute((1, 0)), S_t, use_acc=True)
            tcgen05_mma(Ub_t, LQ0w.slice(C, C, dim=0).permute((1, 0)), UOw.slice(C, C), use_acc=True,
                        mbarriers=[s_bar, o_bar.index(W)])
            _stamp(dbg_ptr, W, j, 13, NC, DEBUG)
        j += 2
    # ---- exit: stop my T-inverse warp; wake the loader if it waits for a chunk I never started ----
    _st_scalar(tcmd_s.index(W), -1, s1)
    mbarrier.arrive(m_ready.index(W), count=1)
    _st_scalar(gseq_s.index(W), NC + 1024, s1)
    # ---- tail: O epilogue of my last chunk, progress, final state ----
    _stamp(dbg_ptr, W, j, 14, NC, DEBUG)
    if jl >= 0:
        mbarrier.wait(o_bar.index(W), (jl >> 1) & 1)
        _stamp(dbg_ptr, W, j, 15, NC, DEBUG)
        _o_epilogue(o_desc, g_s.index(jl % NS), UOw, nq_s.index(2 * W + ((jl >> 1) & 1)), prog_ptr, h, jl, c_base + jl,
                    part1, epoch_base, s_reg, W, NC0, ABL)
    tma.store_wait(0)
    gl.barrier()
    _stamp(dbg_ptr, W, j + 1, 14, NC, DEBUG)
    if part1:
        cnt = gl.maximum((n_end - 1 - W) // 2 + 1, 0)
        _st_release(prog_ptr + 2 * h + W, epoch_base + cnt)
    if wstate:
        mbarrier.wait(s_bar, (n_end - 1) & 1)                                  # last S-update done
        _stamp(dbg_ptr, W, j + 1, 15, NC, DEBUG)
        _state_store_q(S_t, state_ptr, h, 64 * W, s_reg)
        _state_store_q(S_t, state_ptr, h, 64 * W + 32, s_reg)
    # the default partition (WG0) must not sit in the warp-specialize join while WG1 still runs a chunk: that
    # pattern hangs the CTA at the join (observed on B200; cause unknown).  WG1 signals its end, WG0 waits for it.
    if W == 1:
        mbarrier.arrive(wg_done, count=1)
    else:
        mbarrier.wait(wg_done, 0)
    _stamp(dbg_ptr, W, j + 2, 15, NC, DEBUG)


# ----------------------------------------------------------------------------------------------------------------
@gluon.jit
def kda_v21_kernel(q_desc, k_desc, v_desc, g_desc, o_desc, q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, beta_ptr, A_log_ptr, dt_bias_ptr,
                   state_ptr, prog_ptr, epoch_base, dbg_ptr, wd_addr, NC: gl.constexpr, NC0: gl.constexpr, H: gl.constexpr,
                   MID: gl.constexpr, SPLIT: gl.constexpr, DEBUG: gl.constexpr, ABL: gl.constexpr, num_warps: gl.constexpr):
    pid = gl.program_id(0)
    if SPLIT:
        part1 = pid < H
        h = pid % H
        c_base = gl.where(part1, NC0, 0)
        n_init = gl.where(part1, NC - NC0, NC)
        part0_cont = pid >= H
        wstate = part1
    else:
        part1 = pid < 0
        h = pid
        c_base = 0
        n_init = NC
        part0_cont = pid < 0
        wstate = pid >= 0
    dbg_ptr = (dbg_ptr + pid * (5 * (NC + 8) * 16), wd_addr.to(gl.pointer_type(gl.int32)) + pid * 8)

    # ---------------- shared memory ----------------
    qk_s = gl.allocate_shared_memory(gl.bfloat16, [NS, 2 * C, D], q_desc.layout)
    g_s = gl.allocate_shared_memory(gl.bfloat16, [NS, C, D], g_desc.layout)
    v_s = gl.allocate_shared_memory(gl.bfloat16, [NS, C, D], v_desc.layout)
    lq_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([D, 2 * C], gl.bfloat16)
    t32_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([D, C], gl.bfloat16)
    a64_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([2 * C, C], gl.bfloat16)
    a32_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([C, C], gl.bfloat16)
    LQ = gl.allocate_shared_memory(gl.bfloat16, [2, D, 2 * C], lq_layout)    # per WG: cols 0:32 = L^T, 32:64 = Q^T
    KB = gl.allocate_shared_memory(gl.bfloat16, [2, D, C], t32_layout)       # Kbar^T [d, t]
    G_sm = gl.allocate_shared_memory(gl.bfloat16, [2, D, 2 * C], lq_layout)  # G^T [d, t]: cols 0:32 f*W'n^T, 32:64 f*Q^T
    TB = gl.allocate_shared_memory(gl.bfloat16, [2, C, C], a32_layout)       # T^T
    TN = gl.allocate_shared_memory(gl.bfloat16, [2, C, C], a32_layout)       # Tn^T
    LQ0 = gl.allocate_shared_memory(gl.bfloat16, [2, 2 * C, C], a64_layout)  # [0 ; tril(Aq)]
    f32_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    MT = gl.allocate_shared_memory(gl.float32, [2, C, 2 * C], f32_layout)    # M'^T (cols < 32 used)
    vec_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    f_s = gl.allocate_shared_memory(gl.float32, [2, D], vec_layout)
    dc_s = gl.allocate_shared_memory(gl.float32, [2, D], vec_layout)
    n_s = gl.allocate_shared_memory(gl.float32, [2, 2 * C], vec_layout)      # [n_t ; 1]
    nq_s = gl.allocate_shared_memory(gl.float32, [4, 2 * C], vec_layout)     # [1 ; nq_t], x2 per WG (O epilogue lag)
    invn_s = gl.allocate_shared_memory(gl.float32, [2, 2 * C], vec_layout)   # [1/n_t ; 1]
    g2_s = gl.allocate_shared_memory(gl.float32, [D], vec_layout)            # cumulative log2 decay since NC0 (part0)
    nend_s = gl.allocate_shared_memory(gl.int32, [1], vec_layout)
    tcmd_s = gl.allocate_shared_memory(gl.int32, [2, 1], vec_layout)         # per-WG command for its T-inverse warp
    gseq_s = gl.allocate_shared_memory(gl.int32, [2, 1], vec_layout)         # per-WG: last chunk whose gates finished
    g2seq_s = gl.allocate_shared_memory(gl.int32, [1], vec_layout)           # last chunk accumulated into g2

    # ---------------- barriers ----------------
    bl: gl.constexpr = mbarrier.MBarrierLayout()
    tile_full = gl.allocate_shared_memory(gl.int64, [NS, 1], bl)
    tile_empty = gl.allocate_shared_memory(gl.int64, [NS, 1], bl)
    m_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    t_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    mma1_bar = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    g_bar = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    uinit = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    chain_bar = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    o_bar = gl.allocate_shared_memory(gl.int64, [2, 1], bl)
    s_bar = gl.allocate_shared_memory(gl.int64, [1], bl)
    init_bar = gl.allocate_shared_memory(gl.int64, [1], bl)
    wg_done = gl.allocate_shared_memory(gl.int64, [1], bl)
    for i in gl.static_range(NS):
        mbarrier.init(tile_full.index(i), count=1)
        mbarrier.init(tile_empty.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(m_ready.index(i), count=1)
        mbarrier.init(t_ready.index(i), count=1)
        mbarrier.init(mma1_bar.index(i), count=1)
        mbarrier.init(g_bar.index(i), count=1)
        mbarrier.init(uinit.index(i), count=1)
        mbarrier.init(chain_bar.index(i), count=1)
        mbarrier.init(o_bar.index(i), count=1)
    mbarrier.init(s_bar, count=1)
    mbarrier.init(init_bar, count=2)
    mbarrier.init(wg_done, count=1)

    # ---------------- tensor memory (512 columns) ----------------
    S_t = allocate_tensor_memory(gl.float32, [D, D], TensorMemoryLayout((D, D), col_stride=1))
    St_t = allocate_tensor_memory(gl.bfloat16, [D, D], TensorMemoryLayout((D, D), col_stride=1))
    acc1 = allocate_tensor_memory(gl.float32, [2, 2 * C, 2 * C], TensorMemoryLayout((2 * C, 2 * C), col_stride=1))
    G_t = allocate_tensor_memory(gl.float32, [2, D, C], TensorMemoryLayout((D, C), col_stride=1))
    UO = allocate_tensor_memory(gl.float32, [2, D, 2 * C], TensorMemoryLayout((D, 2 * C), col_stride=1))

    # ---------------- n_end, g2 ----------------
    s1: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    _st_scalar(nend_s, n_init, s1)
    _st_scalar(gseq_s.index(0), -1, s1)
    _st_scalar(gseq_s.index(1), -1, s1)
    _st_scalar(g2seq_s, NC0 - 1, s1)
    g2_s.store(gl.zeros([D], gl.float32, gl.BlockedLayout([1], [32], [num_warps], [0])))
    fence_async_shared()
    gl.barrier()

    descs = (q_desc, k_desc, v_desc, g_desc)
    wg_smem = (qk_s, g_s, v_s, LQ, KB, G_sm, TB, TN, LQ0, MT, f_s, dc_s, n_s, nq_s, invn_s, g2_s, nend_s, tcmd_s, gseq_s,
               g2seq_s)
    tmem = (S_t, St_t, acc1, G_t, UO)
    wg_bars = (tile_full, tile_empty, m_ready, t_ready, mma1_bar, g_bar, uinit, chain_bar, o_bar, s_bar, init_bar,
               wg_done)
    ti_smem = (TB, TN, MT, n_s, invn_s, tcmd_s)
    ti_bars = (m_ready, t_ready)
    ld_smem = (qk_s, g_s, v_s, nend_s, gseq_s)
    ld_bars = (tile_full, tile_empty)
    gl.warp_specialize([
        (wg_partition, (o_desc, wg_smem, tmem, wg_bars, state_ptr, prog_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, h, c_base,
                        part1, part0_cont, wstate, epoch_base, dbg_ptr, 0, NC, NC0, MID, H, DEBUG, ABL)),
        (wg_partition, (o_desc, wg_smem, tmem, wg_bars, state_ptr, prog_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, h, c_base,
                        part1, part0_cont, wstate, epoch_base, dbg_ptr, 1, NC, NC0, MID, H, DEBUG, ABL)),
        (tinv_partition, (ti_smem, ti_bars, beta_ptr, h, c_base, n_init, dbg_ptr, 0, NC, H, DEBUG, ABL)),
        (tinv_partition, (ti_smem, ti_bars, beta_ptr, h, c_base, n_init, dbg_ptr, 1, NC, H, DEBUG, ABL)),
        (load_partition, (descs, ld_smem, ld_bars, q_ptr, k_ptr, v_ptr, g_ptr, h, c_base, n_init, dbg_ptr, NC, H, DEBUG)),
    ], [4, 1, 1, 1], [168, 88, 88, 88])

    _stamp(dbg_ptr, 0, NC + 4, 0, NC, DEBUG)      # after the partition join
    for i in gl.static_range(NS):
        mbarrier.invalidate(tile_full.index(i))
        mbarrier.invalidate(tile_empty.index(i))
    for i in gl.static_range(2):
        mbarrier.invalidate(m_ready.index(i))
        mbarrier.invalidate(t_ready.index(i))
        mbarrier.invalidate(mma1_bar.index(i))
        mbarrier.invalidate(g_bar.index(i))
        mbarrier.invalidate(uinit.index(i))
        mbarrier.invalidate(chain_bar.index(i))
        mbarrier.invalidate(o_bar.index(i))
    mbarrier.invalidate(s_bar)
    mbarrier.invalidate(init_bar)
    mbarrier.invalidate(wg_done)
    _stamp(dbg_ptr, 0, NC + 4, 1, NC, DEBUG)      # after the invalidations (TMEM dealloc follows)


_PROG = {}
_EPOCH = [0]
NC0_DEFAULT = 126


def kda_fwd(q, k, v, g, beta, A_log, dt_bias, state, dbg=None, NC0=None, split=None, wd_addr=0, abl=0):
    B, T, H, Dd = q.shape
    assert B == 1 and Dd == 128 and T % 32 == 0
    NC = T // 32
    if split is None:
        split = NC >= 64
    if NC0 is None:
        NC0 = NC0_DEFAULT if NC == 256 else NC // 2 - 2
    if not split:
        NC0 = NC
    o = torch.empty_like(q)
    key = q.device
    if key not in _PROG:
        _PROG[key] = torch.zeros(2 * H, dtype=torch.int32, device=q.device)
    _EPOCH[0] += 1
    layout128 = gl.NVMMASharedLayout.get_default_for([32, 128], gl.bfloat16)
    mk = lambda x: TensorDescriptor(x.view(T, H * Dd), shape=[T, H * Dd], strides=[H * Dd, 1], block_shape=[32, 128],
                                    layout=layout128)
    grid = (2 * H,) if split else (H,)
    wd_flag = wd_addr
    if not wd_addr:
        wd_addr = 1 << 40      # dummy 64-bit value (only dereferenced in DEBUG builds)
    kda_v21_kernel[grid](mk(q), mk(k), mk(v), mk(g), mk(o), q, k, v, g, o, beta, A_log, dt_bias, state, _PROG[key],
                         _EPOCH[0] * 1024,
                         dbg if dbg is not None else torch.empty(1, dtype=torch.int64, device=q.device), wd_addr,
                         NC=NC, NC0=NC0, H=H, MID=15, SPLIT=split, DEBUG=(1 if dbg is not None else (2 if wd_flag else 0)), ABL=abl, num_warps=4, maxnreg=168)
    return o, state
