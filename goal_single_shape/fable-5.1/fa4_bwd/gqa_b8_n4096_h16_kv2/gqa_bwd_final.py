"""GQA causal attention backward for Blackwell (sm_100a), written from scratch in Triton Gluon.

Problem: B=8, N=4096, H=16, H_KV=2, D=128, bf16, causal, layout (B, N, H, D).

Design (one CTA per (batch, kv-head, kv-block of 128 rows); tiles = (q-block of 128 rows) x (one of the
8 q-heads of the group), causal-pruned):
  * K/V block resident in SMEM; dK/dV accumulate in TMEM across all 8 q-heads and all q-blocks
    (GQA-aware: dK/dV never leave the SM until the very end).
  * Per tile five full-rate tcgen05 MMAs (all 128x128x128), issued by a single MMA warp in the order
        dP^T = V dO^T        -> TMEM Y                 (Y previously held S^T(t), read by the softmax)
        dV  += P^T dO        A = P^T  packed bf16 in TMEM X[0:64]
        S^T(t+1) = K Q^T     -> TMEM Y
        dK  += dS^T Q        A = dS^T packed bf16 in TMEM X[64:128]
        dQ^T = K^T dS^T      -> TMEM X                 (B = dS^T bf16 in SMEM)
  * Two 8-warp softmax partitions (64 q-columns each) read S^T / dP^T from TMEM, compute P (exp2 via
    MUFU) and dS = P * (dP - delta) with bf16x2 math, and write P^T / dS^T straight into TMEM as MMA
    A-operands; only dS^T also goes to SMEM (into the tile's own dO stage, once dV has consumed dO).
  * An 8-warp drain partition scales dQ^T, stages it as fp16 in a dedicated SMEM buffer and issues
    two 128x64 TMA bulk reduce-adds into an fp16 (B, H, D, N) accumulator; it also issues the Q/lse/delta
    TMA loads and the dO cp.async loads (completion via cp.async.mbarrier.arrive.noinc; the MMA warp
    executes a proxy fence after the wait).  The per-SM TMA unit therefore only serves the reduce and Q.
  * SMEM: K 32K + V 32K + Q 2x32K + dO 2x32K + staging 32K = 224 KB; TMEM: X 128 + Y 128 + dK 128 +
    dV 128 = 512 columns.
  * Pre-processing kernel: delta = rowsum(O * dO), lse * log2(e), zero the fp16 accumulator.
    Post-processing kernel: transpose the accumulator to (B, N, H, D) bf16.
"""
import math
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
)

_BLOCK_N = 128  # kv rows per CTA
_BLOCK_M = 128  # q rows per tile
_HALF = 64
_QUARTER = 32
_HEAD_DIM = 128
_NUM_Q_STAGES = 2
_NUM_DO_STAGES = 2
_LOG2E = 1.4426950408889634
BLOCK_N = gl.constexpr(_BLOCK_N)
BLOCK_M = gl.constexpr(_BLOCK_M)
HALF = gl.constexpr(_HALF)
QUARTER = gl.constexpr(_QUARTER)
HEAD_DIM = gl.constexpr(_HEAD_DIM)
NUM_Q_STAGES = gl.constexpr(_NUM_Q_STAGES)
NUM_DO_STAGES = gl.constexpr(_NUM_DO_STAGES)


# ----------------------------------------------------------------------------------------------
# Small inline-PTX helpers
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _red_add_v4_f32(ptrs, vals):
    """Fire-and-forget 16-byte fp32 atomic add; `vals` must hold 4 contiguous elements per thread."""
    addr = ptrs.to(gl.int64)
    gl.inline_asm_elementwise(
        "red.relaxed.gpu.global.add.v4.f32 [$4], {$8, $9, $10, $11};",
        "=r,=r,=r,=r,l,l,l,l,f,f,f,f", [addr, vals], dtype=gl.int32, is_pure=False, pack=4)


@gluon.jit
def _red_add_v2_f16x2(ptrs, vals):
    """4 contiguous fp32 values per thread -> 4 fp16 -> one 8-byte packed atomic add (red.v2.f16x2)."""
    addr = ptrs.to(gl.int64)
    gl.inline_asm_elementwise(
        """{
        .reg .b32 h0, h1;
        cvt.rn.f16x2.f32 h0, $9, $8;
        cvt.rn.f16x2.f32 h1, $11, $10;
        red.relaxed.gpu.global.add.noftz.v2.f16x2 [$4], {h0, h1};
        }""",
        "=r,=r,=r,=r,l,l,l,l,f,f,f,f", [addr, vals], dtype=gl.int32, is_pure=False, pack=4)


from triton.experimental.gluon.language._core import builtin as _gl_builtin
import triton.experimental.gluon.language._core as _ttgl


@_gl_builtin
def _tma_reduce_add(tensor_desc, coord, src, _semantic=None):
    """cp.reduce.async.bulk.tensor (add) from shared memory into global memory."""
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(_ttgl.ir.DESCRIPTOR_REDUCE_KIND.ADD, tensor_desc.handle, coord, src.handle)


@gluon.jit
def _pair_red_f16x2(ptrs, vals):
    """Lane pairs (d even/odd) exchange one value so each lane owns (d, d+1) of one q; 4-byte f16x2 atomic add.

    vals: int32 bit patterns of fp32 values, 2 consecutive registers (q, q+1) per call; ptrs: target of the pair."""
    addr = ptrs.to(gl.int64)
    gl.inline_asm_elementwise(
        """{
        .reg .pred podd;
        .reg .u32 lid;
        .reg .b32 snd, rcv, lo, hi, h;
        mov.u32 lid, %laneid;
        and.b32 lid, lid, 1;
        setp.ne.u32 podd, lid, 0;
        selp.b32 snd, $4, $5, podd;
        shfl.sync.bfly.b32 rcv, snd, 1, 0x1f, 0xffffffff;
        selp.b32 lo, rcv, $4, podd;
        selp.b32 hi, $5, rcv, podd;
        mov.b32 $0, lo;
        mov.b32 $1, hi;
        cvt.rn.f16x2.f32 h, hi, lo;
        red.relaxed.gpu.global.add.noftz.f16x2 [$2], h;
        }""",
        "=r,=r,l,l,r,r", [addr, vals], dtype=gl.int32, is_pure=False, pack=2)


@gluon.jit
def _to_bf16x2(x):
    """fp32 -> bf16 with the packed cvt.rn.bf16x2.f32 (one instruction per two elements)."""
    return gl.inline_asm_elementwise("cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", [x], dtype=gl.bfloat16,
                                     is_pure=True, pack=2)


@gluon.jit
def _exp2_f16x2_to_bf16(x):
    """exp2 of two fp32 inputs computed with the packed half-precision MUFU (ex2.approx.f16x2), output bf16."""
    return gl.inline_asm_elementwise(
        """{
        .reg .b32 h;
        cvt.rn.f16x2.f32 h, $2, $1;
        ex2.approx.f16x2 $0, h;
        }""",
        "=r,f,f", [x], dtype=gl.float16, is_pure=True, pack=2)


@gluon.jit
def _quad_transpose_i32(x):
    """Transpose 4x4 blocks spread over 4 consecutive lanes x 4 consecutive registers (int32 bit patterns).

    Input: lane k (k = laneid & 3) holds a[k][0..3].  Output: lane k holds a[0..3][k]  (butterfly shuffles)."""
    return gl.inline_asm_elementwise(
        """{
        .reg .u32 lid;
        .reg .pred pk1, pk2;
        .reg .b32 t0, t1, t2, t3, s0, s1, s2, s3;
        mov.u32 lid, %laneid;
        and.b32 lid, lid, 3;
        setp.ge.u32 pk2, lid, 2;
        and.b32 lid, lid, 1;
        setp.ne.u32 pk1, lid, 0;
        shfl.sync.bfly.b32 s0, $6, 2, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s1, $7, 2, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s2, $4, 2, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s3, $5, 2, 0x1f, 0xffffffff;
        selp.b32 t0, s0, $4, pk2;
        selp.b32 t1, s1, $5, pk2;
        selp.b32 t2, $6, s2, pk2;
        selp.b32 t3, $7, s3, pk2;
        shfl.sync.bfly.b32 s0, t1, 1, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s1, t0, 1, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s2, t3, 1, 0x1f, 0xffffffff;
        shfl.sync.bfly.b32 s3, t2, 1, 0x1f, 0xffffffff;
        selp.b32 $0, s0, t0, pk1;
        selp.b32 $1, t1, s1, pk1;
        selp.b32 $2, s2, t2, pk1;
        selp.b32 $3, t3, s3, pk1;
        }""",
        "=r,=r,=r,=r,r,r,r,r", [x], dtype=gl.int32, is_pure=True, pack=4)


@gluon.jit
def _mul_bf16x2(a, b):
    """Packed bf16 multiply (mul.rn.bf16x2), two elements per instruction."""
    return gl.inline_asm_elementwise("mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], dtype=gl.bfloat16,
                                     is_pure=True, pack=2)


@gluon.jit
def _warp_arrive(smem_off):
    """One mbarrier arrival per warp (lane 0) on the barrier at byte offset `smem_off` of dynamic smem."""
    gl.inline_asm_elementwise(
        """{
        .reg .pred p;
        .reg .u32 lid, a;
        mov.u32 lid, %laneid;
        setp.eq.u32 p, lid, 0;
        mov.u32 a, global_smem;
        add.u32 a, a, $1;
        @p mbarrier.arrive.shared::cta.b64 _, [a];
        mov.u32 $0, 0;
        }""",
        "=r,r", [smem_off], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _exp2_approx(x):
    """Hardware exp2 (MUFU.EX2) via inline PTX; gl.exp2 lowers to a slow software sequence."""
    return gl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=gl.float32,
                                     is_pure=True, pack=1)


@gluon.jit
def _tc_fence_before():
    gl.inline_asm_elementwise("tcgen05.fence::before_thread_sync; mov.b32 $0, 0;", "=r", [],
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _tc_fence_after():
    gl.inline_asm_elementwise("tcgen05.fence::after_thread_sync; mov.b32 $0, 0;", "=r", [],
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _clock():
    return gl.inline_asm_elementwise("mov.u32 $0, %clock;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _tile_coords(t, GROUP: gl.constexpr, NUM_Q_BLOCKS: gl.constexpr):
    # q-block descending (from the end of the sequence down to the diagonal), heads inner.
    i = NUM_Q_BLOCKS - 1 - t // GROUP
    h = t % GROUP
    return i, h


# ----------------------------------------------------------------------------------------------
# MMA + TMA partition (1 warp)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _issue_q_loads(t, a, SEQ: gl.constexpr, NUM_HEADS: gl.constexpr, GROUP: gl.constexpr, NUM_Q_BLOCKS: gl.constexpr):
    """Q tile + lse + delta of tile t into stage t % NUM_Q_STAGES (one expect-arrival on q_ready)."""
    descs, smem, tmem, bars, scalars = a
    q_desc, do_ptr, k_desc, v_desc, lse_desc, delta_desc = descs
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    b, hk, j, num_tiles = scalars
    q_ready = bars[1]
    qs = t % NUM_Q_STAGES
    i, h = _tile_coords(t, GROUP, NUM_Q_BLOCKS)
    hq = hk * GROUP + h
    q_row = b * SEQ + i * BLOCK_M
    vec_base = (b * NUM_HEADS + hq) * SEQ + i * BLOCK_M
    bar = q_ready.index(qs)
    mbarrier.expect(bar, q_desc.block_type.nbytes + lse_desc.block_type.nbytes + delta_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [hq, q_row, 0], bar, q_smem.index(qs))
    tma.async_copy_global_to_shared(lse_desc, [vec_base], bar, lse_smem.index(qs))
    tma.async_copy_global_to_shared(delta_desc, [vec_base], bar, delta_smem.index(qs))


@gluon.jit
def _issue_do_load(t, a, SEQ: gl.constexpr, NUM_HEADS: gl.constexpr, GROUP: gl.constexpr, NUM_Q_BLOCKS: gl.constexpr):
    """cp.async of the dO tile of tile t into dO stage t % NUM_DO_STAGES (issued by the 8 drain warps)."""
    descs, smem, tmem, bars, scalars = a
    q_desc, do_ptr, k_desc, v_desc, lse_desc, delta_desc = descs
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    b, hk, j, num_tiles = scalars
    do_ready = bars[13]
    ds = t % NUM_DO_STAGES
    i, h = _tile_coords(t, GROUP, NUM_Q_BLOCKS)
    hq = hk * GROUP + h
    tile_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [8, 1], [1, 0])
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, tile_layout))
    cols = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, tile_layout))
    ptrs = do_ptr + (((b * SEQ + i * BLOCK_M + rows[:, None]) * NUM_HEADS + hq) * HEAD_DIM + cols[None, :])
    async_copy.async_copy_global_to_shared(do_smem.index(ds), ptrs)
    async_copy.mbarrier_arrive(do_ready.index(ds), increment_count=False)   # fires when this thread's copies land


@gluon.jit
def _signal_loaded(bar, pending):
    """Wait until at most `pending` cp.async groups are outstanding, publish to the async proxy, arrive."""
    async_copy.wait_group(pending)
    fence_async_shared()
    mbarrier.arrive(bar, count=1)


@gluon.jit
def _mma_partition(a, dbg_ptr, SEQ: gl.constexpr, NUM_HEADS: gl.constexpr, GROUP: gl.constexpr,
                   NUM_Q_BLOCKS: gl.constexpr, DBG: gl.constexpr, ABL: gl.constexpr):
    descs, smem, tmem, bars, scalars = a
    q_desc, do_ptr, k_desc, v_desc, lse_desc, delta_desc = descs
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    s_tmem, dp_tmem, dk_tmem, dv_tmem = tmem
    kv_ready, q_ready, s_ready, dp_ready, p_ready, ds_ready, dv_done, dk_done, dq_done, dq_empty, dkv_ready, s_cons, dp_cons, do_ready = bars
    b, hk, j, num_tiles = scalars

    pfull_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, BLOCK_M), col_stride=1)
    p_tmem = s_tmem.slice(0, BLOCK_M // 2)._reinterpret(gl.bfloat16, [BLOCK_N, BLOCK_M], pfull_layout)        # X[0:64]  = P^T
    dst_tmem = s_tmem.slice(BLOCK_M // 2, BLOCK_M // 2)._reinterpret(gl.bfloat16, [BLOCK_N, BLOCK_M], pfull_layout)  # X[64:128] = dS^T

    kv_row = b * SEQ + j * BLOCK_N
    mbarrier.expect(kv_ready, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(k_desc, [hk, kv_row, 0], kv_ready, k_smem)
    tma.async_copy_global_to_shared(v_desc, [hk, kv_row, 0], kv_ready, v_smem)
    mbarrier.wait(kv_ready, 0)
    mbarrier.wait(q_ready.index(0), 0)
    tcgen05_mma(k_smem, q_smem.index(0).permute((1, 0)), dp_tmem, use_acc=False)     # S(0) -> Y
    tcgen05_commit(s_ready)

    m0 = 0
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    m5 = 0
    m6 = 0
    for t in range(num_tiles):
        ph = t & 1
        qs = t % NUM_Q_STAGES
        qs1 = (t + 1) % NUM_Q_STAGES
        ds = t % NUM_DO_STAGES
        q_stage = q_smem.index(qs)
        do_stage = do_smem.index(ds)            # dO(t); later also holds dS^T(t) (B operand of dQ^T)
        use_acc = t > 0

        if DBG:
            k0 = _clock()
        mbarrier.wait(do_ready.index(ds), (t // NUM_DO_STAGES) & 1)
        fence_async_shared()                    # cp.async (generic proxy) writes of dO(t) -> async proxy
        mbarrier.wait(s_cons, ph)               # S(t) read by both halves -> X free
        _tc_fence_after()
        if DBG:
            k1 = _clock()
        tcgen05_mma(v_smem, do_stage.permute((1, 0)), dp_tmem, use_acc=False)         # dP(t) -> Y
        tcgen05_commit(dp_ready)
        mbarrier.wait(p_ready, ph)              # both halves of P^T in X[0:64]
        _tc_fence_after()
        if DBG:
            k2 = _clock()
        tcgen05_mma(p_tmem, do_stage, dv_tmem, use_acc=use_acc)                         # dV += P^T dO (A: TMEM)
        tcgen05_commit(dv_done)
        if t + 1 < num_tiles:
            mbarrier.wait(q_ready.index(qs1), ((t + 1) // NUM_Q_STAGES) & 1)
            mbarrier.wait(dp_cons, ph)          # dP(t) read by both halves -> X free
            _tc_fence_after()
            tcgen05_mma(k_smem, q_smem.index(qs1).permute((1, 0)), dp_tmem, use_acc=False)  # S(t+1) -> Y
            tcgen05_commit(s_ready)
        mbarrier.wait(ds_ready, ph)             # both halves of dS^T in X[64:128] and in smem
        _tc_fence_after()
        if DBG:
            k3 = _clock()
            k4 = k3
        tcgen05_mma(dst_tmem, q_stage, dk_tmem, use_acc=use_acc)                        # dK += dS^T Q (A: TMEM)
        tcgen05_commit(dk_done)
        if DBG:
            k5 = _clock()
        mbarrier.wait(dv_done, ph)              # dV(t) done reading P^T from X[0:64]
        _tc_fence_after()
        if DBG:
            k6 = _clock()
        tcgen05_mma(k_smem.permute((1, 0)), do_stage, s_tmem, use_acc=False)           # dQ^T(t) = K^T dS^T -> X
        tcgen05_commit(dq_done)
        if DBG:
            k7 = _clock()
            k8 = k7
        if DBG:
            m0 += k1 - k0
            m1 += k2 - k1
            m2 += k3 - k2
            m3 += k4 - k3
            m4 += k6 - k5
            m5 += k8 - k7
            m6 += k8 - k0
    tcgen05_commit(dkv_ready)
    if DBG:
        pid = gl.program_id(0)
        dbase = dbg_ptr + pid * 64 + 32
        gl.store(dbase + 0, m0)
        gl.store(dbase + 1, m1)
        gl.store(dbase + 2, m2)
        gl.store(dbase + 3, m3)
        gl.store(dbase + 4, m4)
        gl.store(dbase + 5, m5)
        gl.store(dbase + 6, m6)
        gl.store(dbase + 8, num_tiles)


# ----------------------------------------------------------------------------------------------
# Softmax partitions (2 x 8 warps): partition HALF_IDX handles q-columns [64*HALF_IDX, +64)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _softmax_tile(t, a, scale_log2e, p_off, ds_off, HALF_IDX: gl.constexpr, GROUP: gl.constexpr,
                  NUM_Q_BLOCKS: gl.constexpr, ABL: gl.constexpr, DBG: gl.constexpr, MASK: gl.constexpr):
    descs, smem, tmem, bars, scalars = a
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    s_tmem, dp_tmem, dk_tmem, dv_tmem = tmem
    kv_ready, q_ready, s_ready, dp_ready, p_ready, ds_ready, dv_done, dk_done, dq_done, dq_empty, dkv_ready, s_cons, dp_cons, do_ready = bars
    b, hk, j, num_tiles = scalars

    q_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, QUARTER), col_stride=1)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, QUARTER], q_layout, 8)
    vec_layout: gl.constexpr = gl.SliceLayout(0, reg_layout)
    C0: gl.constexpr = HALF * HALF_IDX
    pds_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HALF], gl.bfloat16)
    # dS^T of this half goes into the tile's dO stage once dV(t) has consumed dO(t) (B operand of dQ^T)
    my_pds = do_smem.index(t % NUM_DO_STAGES)._reinterpret(gl.bfloat16, [2, BLOCK_N, HALF], pds_layout).index(HALF_IDX)
    p_layout: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BLOCK_N, QUARTER], q_layout, 8)
    # packed-bf16 quarter views in X: P^T at fp32 cols [C0/2 + 16*qq), dS^T at [64 + C0/2 + 16*qq)
    p_q0 = s_tmem.slice(C0 // 2, QUARTER // 2)._reinterpret(gl.bfloat16, [BLOCK_N, QUARTER], q_layout)
    p_q1 = s_tmem.slice(C0 // 2 + QUARTER // 2, QUARTER // 2)._reinterpret(gl.bfloat16, [BLOCK_N, QUARTER], q_layout)
    ds_q0 = s_tmem.slice(BLOCK_M // 2 + C0 // 2, QUARTER // 2)._reinterpret(gl.bfloat16, [BLOCK_N, QUARTER], q_layout)
    ds_q1 = s_tmem.slice(BLOCK_M // 2 + C0 // 2 + QUARTER // 2, QUARTER // 2)._reinterpret(gl.bfloat16, [BLOCK_N, QUARTER], q_layout)

    ph = t & 1
    qs = t % NUM_Q_STAGES
    i, h = _tile_coords(t, GROUP, NUM_Q_BLOCKS)

    if DBG:
        c0 = _clock()
    mbarrier.wait(q_ready.index(qs), (t // NUM_Q_STAGES) & 1)   # TMA-written lse/delta visible
    mbarrier.wait(s_ready, ph)
    _tc_fence_after()
    if DBG:
        c1 = _clock()
    # ---- P quarter 0 (lse arrives pre-scaled by log2 e)
    lse0 = lse_smem.index(qs).slice(C0, QUARTER).load(vec_layout)
    lse1 = lse_smem.index(qs).slice(C0 + QUARTER, QUARTER).load(vec_layout)
    st0 = dp_tmem.slice(C0, QUARTER).load(reg_layout)
    st1 = dp_tmem.slice(C0 + QUARTER, QUARTER).load(reg_layout)
    _tc_fence_before()
    mbarrier.arrive(s_cons, count=1)     # both quarters of S(t) are in registers -> Y may be overwritten by dP(t)
    x0 = st0 * scale_log2e - lse0[None, :]
    if MASK:
        kv_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, reg_layout))
        q_idx = gl.arange(0, QUARTER, layout=gl.SliceLayout(0, reg_layout))
        x0 = gl.where((i * BLOCK_M + C0 + q_idx)[None, :] >= (j * BLOCK_N + kv_idx)[:, None], x0, -1.0e30)
    if ABL & 512:
        pb0 = _exp2_f16x2_to_bf16(x0).to(gl.bfloat16)
    elif ABL & 1:
        pb0 = _to_bf16x2(x0)
    else:
        pb0 = _to_bf16x2(_exp2_approx(x0))
    # ---- P quarter 1
    x1 = st1 * scale_log2e - lse1[None, :]
    if MASK:
        x1 = gl.where((i * BLOCK_M + C0 + QUARTER + q_idx)[None, :] >= (j * BLOCK_N + kv_idx)[:, None], x1, -1.0e30)
    if ABL & 512:
        pb1 = _exp2_f16x2_to_bf16(x1).to(gl.bfloat16)
    elif ABL & 1:
        pb1 = _to_bf16x2(x1)
    else:
        pb1 = _to_bf16x2(_exp2_approx(x1))
    mbarrier.wait(dq_empty, ph ^ 1)      # dQ^T(t-1) drained out of X -> P^T(t) may be written there
    _tc_fence_after()
    if not (ABL & 2):
        p_q0.store(gl.convert_layout(pb0, p_layout))
        p_q1.store(gl.convert_layout(pb1, p_layout))
    _tc_fence_before()
    _warp_arrive(p_off)
    if DBG:
        c2 = _clock()

    # ---- dS = P * (dP - delta)
    mbarrier.wait(dp_ready, ph)
    _tc_fence_after()
    if DBG:
        c3 = _clock()
    dpt0 = dp_tmem.slice(C0, QUARTER).load(reg_layout)            # dP(t) lives in Y (the S region)
    dpt1 = dp_tmem.slice(C0 + QUARTER, QUARTER).load(reg_layout)
    _tc_fence_before()
    mbarrier.arrive(dp_cons, count=1)
    delta0 = delta_smem.index(qs).slice(C0, QUARTER).load(vec_layout)
    ds0 = _mul_bf16x2(pb0, _to_bf16x2(dpt0 - delta0[None, :]))
    delta1 = delta_smem.index(qs).slice(C0 + QUARTER, QUARTER).load(vec_layout)
    ds1 = _mul_bf16x2(pb1, _to_bf16x2(dpt1 - delta1[None, :]))
    if not (ABL & 2):
        ds_q0.store(gl.convert_layout(ds0, p_layout))          # TMEM copy: A operand of dK
        ds_q1.store(gl.convert_layout(ds1, p_layout))
        mbarrier.wait(dv_done, ph)                             # dO(t) consumed -> its stage may hold dS^T(t)
        my_pds.slice(0, QUARTER, dim=1).store(ds0)             # smem copy: B operand of dQ^T
        my_pds.slice(QUARTER, QUARTER, dim=1).store(ds1)
        fence_async_shared()
    _tc_fence_before()
    _warp_arrive(ds_off)
    if DBG:
        c4 = _clock()
        return c1 - c0, c2 - c1, c3 - c2, c4 - c3
    return 0, 0, 0, 0


@gluon.jit
def _softmax_half(a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off, HALF_IDX: gl.constexpr,
                  SEQ: gl.constexpr, NUM_KV_HEADS: gl.constexpr, GROUP: gl.constexpr,
                  NUM_Q_BLOCKS: gl.constexpr, ABL: gl.constexpr, DBG: gl.constexpr):
    descs, smem, tmem, bars, scalars = a
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    s_tmem, dp_tmem, dk_tmem, dv_tmem = tmem
    kv_ready, q_ready, s_ready, dp_ready, p_ready, ds_ready, dv_done, dk_done, dq_done, dq_empty, dkv_ready, s_cons, dp_cons, do_ready = bars
    b, hk, j, num_tiles = scalars

    C0: gl.constexpr = HALF * HALF_IDX

    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0
    # main loop: tiles strictly above the diagonal (no causal mask); the last GROUP tiles (i == j) are peeled
    for t in range(num_tiles - GROUP):
        e0, e1, e2, e3 = _softmax_tile(t, a, scale_log2e, p_off, ds_off, HALF_IDX, GROUP, NUM_Q_BLOCKS, ABL, DBG, False)
        a0 += e0
        a1 += e1
        a2 += e2
        a3 += e3
    for t in range(num_tiles - GROUP, num_tiles):
        e0, e1, e2, e3 = _softmax_tile(t, a, scale_log2e, p_off, ds_off, HALF_IDX, GROUP, NUM_Q_BLOCKS, ABL, DBG, True)
        a0 += e0
        a1 += e1
        a2 += e2
        a3 += e3
    if DBG:
        pid = gl.program_id(0)
        dbase = dbg_ptr + pid * 64 + HALF_IDX * 16
        gl.store(dbase + 0, a0)
        gl.store(dbase + 1, a1)
        gl.store(dbase + 2, a2)
        gl.store(dbase + 3, a3)

    # ---- final dK / dV: this partition writes d-columns [C0, C0+64)
    KCHUNK: gl.constexpr = 32
    kchunk_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, KCHUNK), col_stride=1)
    kreg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, KCHUNK], kchunk_layout, 8)
    krows = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, kreg_layout))  # kv row
    kcols = gl.arange(0, KCHUNK, layout=gl.SliceLayout(0, kreg_layout))   # d within chunk
    mbarrier.wait(dkv_ready, 0)
    _tc_fence_after()
    kv_base = ((b * SEQ + j * BLOCK_N) * NUM_KV_HEADS + hk) * HEAD_DIM + C0
    for c in gl.static_range(HALF // KCHUNK):
        offs = kv_base + krows[:, None] * (NUM_KV_HEADS * HEAD_DIM) + (c * KCHUNK + kcols)[None, :]
        dk = dk_tmem.slice(C0 + c * KCHUNK, KCHUNK).load(kreg_layout)
        gl.store(dk_ptr + offs, (dk * sm_scale).to(gl.bfloat16))
        dv = dv_tmem.slice(C0 + c * KCHUNK, KCHUNK).load(kreg_layout)
        gl.store(dv_ptr + offs, dv.to(gl.bfloat16))


@gluon.jit
def _softmax_partition_a(a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off,
                         SEQ: gl.constexpr, NUM_KV_HEADS: gl.constexpr, GROUP: gl.constexpr,
                         NUM_Q_BLOCKS: gl.constexpr, ABL: gl.constexpr, DBG: gl.constexpr, num_warps: gl.constexpr):
    _softmax_half(a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off, 0, SEQ, NUM_KV_HEADS, GROUP,
                  NUM_Q_BLOCKS, ABL, DBG)


@gluon.jit
def _softmax_partition_b(a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off,
                         SEQ: gl.constexpr, NUM_KV_HEADS: gl.constexpr, GROUP: gl.constexpr,
                         NUM_Q_BLOCKS: gl.constexpr, ABL: gl.constexpr, DBG: gl.constexpr):
    _softmax_half(a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off, 1, SEQ, NUM_KV_HEADS, GROUP,
                  NUM_Q_BLOCKS, ABL, DBG)


# ----------------------------------------------------------------------------------------------
# Drain partition (8 warps): dQ^T(t) -> registers (bf16) -> coalesced fp32 red.v4 into dq_accum (B, N, H, D)
# ----------------------------------------------------------------------------------------------
@gluon.jit
def _drain_partition(a, dq_desc, sm_scale, dbg_ptr, SEQ: gl.constexpr, NUM_HEADS: gl.constexpr,
                     GROUP: gl.constexpr, NUM_Q_BLOCKS: gl.constexpr, ABL: gl.constexpr, DBG: gl.constexpr):
    descs, smem, tmem, bars, scalars = a
    k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem = smem
    s_tmem, dp_tmem, dk_tmem, dv_tmem = tmem
    dq_done, dq_empty = bars[8], bars[9]
    b, hk, j, num_tiles = scalars
    CH: gl.constexpr = 32
    ch_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, CH), col_stride=1)
    c_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_N, CH], ch_layout, 8)
    f16_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_M], gl.float16)
    kv_ready, q_ready, do_ready, dk_done = bars[0], bars[1], bars[13], bars[7]

    # prologue: Q(0), Q(1), dO(0), dO(1)
    _issue_q_loads(0, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)
    _issue_q_loads(1, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)
    _issue_do_load(0, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)
    _issue_do_load(1, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)

    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    for t in range(num_tiles):
        ph = t & 1
        qs = t % NUM_Q_STAGES
        ds = t % NUM_DO_STAGES
        i, h = _tile_coords(t, GROUP, NUM_Q_BLOCKS)
        hq = hk * GROUP + h
        if DBG:
            c0 = _clock()
        if t + 2 < num_tiles:
            mbarrier.wait(dk_done, ph)          # dK(t) done reading Q(t) -> refill that Q stage with Q(t+2)
            _issue_q_loads(t + 2, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)
        mbarrier.wait(dq_done, ph)              # dQ^T(t) in X; dO stage ds (dO(t), then dS^T(t)) is free
        _tc_fence_after()
        if t + 2 < num_tiles:
            _issue_do_load(t + 2, a, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS)
        if DBG:
            c1 = _clock()
        x0 = (s_tmem.slice(0 * CH, CH).load(c_layout) * sm_scale).to(gl.float16)
        x1 = (s_tmem.slice(1 * CH, CH).load(c_layout) * sm_scale).to(gl.float16)
        x2 = (s_tmem.slice(2 * CH, CH).load(c_layout) * sm_scale).to(gl.float16)
        x3 = (s_tmem.slice(3 * CH, CH).load(c_layout) * sm_scale).to(gl.float16)
        _tc_fence_before()
        mbarrier.arrive(dq_empty, count=1)
        if DBG:
            c2 = _clock()
        if not (ABL & 16):
            # stage dQ^T (d rows, q contiguous) as fp16 in the dedicated staging buffer, then bulk reduce-add into
            # dq_accum[b, hq, d, i*128 + q]
            stage = stage_smem
            tma.store_wait(0)                   # reduce(t-1) has finished reading the staging buffer
            stage.slice(0 * CH, CH, dim=1).store(x0)
            stage.slice(1 * CH, CH, dim=1).store(x1)
            stage.slice(2 * CH, CH, dim=1).store(x2)
            stage.slice(3 * CH, CH, dim=1).store(x3)
            fence_async_shared()
            _tma_reduce_add(dq_desc, [b * NUM_HEADS + hq, 0, i * BLOCK_M], stage.slice(0, HALF, dim=1))
            _tma_reduce_add(dq_desc, [b * NUM_HEADS + hq, 0, i * BLOCK_M + HALF], stage.slice(HALF, HALF, dim=1))
        if DBG:
            cX = _clock()
            cY = cX
            cZ = cX
        if DBG:
            c3 = _clock()
            a0 += c1 - c0
            a1 += c2 - c1
            a2 += cX - c2
            a3 += cY - cX
            a4 += cZ - cY
            a5 += c3 - cZ
    tma.store_wait(0)
    if DBG:
        pid = gl.program_id(0)
        dbase = dbg_ptr + pid * 64 + 48
        gl.store(dbase + 0, a0)
        gl.store(dbase + 1, a1)
        gl.store(dbase + 2, a2)
        gl.store(dbase + 3, a3)
        gl.store(dbase + 4, a4)
        gl.store(dbase + 5, a5)


# ----------------------------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------------------------
@gluon.jit(do_not_specialize=["p_off", "ds_off"])
def gqa_bwd_kernel(q_desc, k_desc, v_desc, do_ptr, lse_desc, delta_desc, dq_desc, dk_ptr, dv_ptr,
                   sm_scale, scale_log2e, dbg_ptr, p_off, ds_off,
                   SEQ: gl.constexpr, NUM_HEADS: gl.constexpr, NUM_KV_HEADS: gl.constexpr,
                   ABL: gl.constexpr, DBG: gl.constexpr, num_warps: gl.constexpr):
    GROUP: gl.constexpr = NUM_HEADS // NUM_KV_HEADS
    NUM_Q_BLOCKS: gl.constexpr = SEQ // BLOCK_M

    pid = gl.program_id(0)
    num_groups = gl.num_programs(0) // (SEQ // BLOCK_N)
    j = pid // num_groups                 # kv block, ascending -> longest CTAs first
    bh = pid % num_groups
    b = bh // NUM_KV_HEADS
    hk = bh % NUM_KV_HEADS
    num_tiles = (NUM_Q_BLOCKS - j) * GROUP

    # ---- shared memory (K 32K + V 32K + Q 2x32K + dO 2x32K + staging 32K = 224 KB)
    tile_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl.bfloat16)
    vec_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M], gl.float32)
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, HEAD_DIM], v_desc.layout)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_Q_STAGES, BLOCK_M, HEAD_DIM], q_desc.layout)
    do_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_DO_STAGES, BLOCK_M, HEAD_DIM], tile_smem_layout)
    stage_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_N, BLOCK_M], gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_M], gl.float16))
    lse_smem = gl.allocate_shared_memory(gl.float32, [NUM_Q_STAGES, BLOCK_M], lse_desc.layout)
    delta_smem = gl.allocate_shared_memory(gl.float32, [NUM_Q_STAGES, BLOCK_M], delta_desc.layout)

    # ---- barriers
    kv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    q_ready = gl.allocate_shared_memory(gl.int64, [NUM_Q_STAGES, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dp_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    p_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    ds_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dv_done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dk_done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dq_done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dq_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dkv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    s_cons = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dp_cons = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(s_cons, count=2)
    mbarrier.init(dp_cons, count=2)
    mbarrier.init(kv_ready, count=1)
    do_ready = gl.allocate_shared_memory(gl.int64, [NUM_DO_STAGES, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(NUM_Q_STAGES):
        mbarrier.init(q_ready.index(s), count=1)
    for s in gl.static_range(NUM_DO_STAGES):
        mbarrier.init(do_ready.index(s), count=256)  # one noinc arrival per drain thread's cp.async batch
    mbarrier.init(s_ready, count=1)
    mbarrier.init(dp_ready, count=1)
    mbarrier.init(p_ready, count=16)     # one arrival per softmax warp (2 partitions x 8 warps)
    mbarrier.init(ds_ready, count=16)
    mbarrier.init(dv_done, count=1)
    mbarrier.init(dk_done, count=1)
    mbarrier.init(dq_done, count=1)
    mbarrier.init(dq_empty, count=1)
    mbarrier.init(dkv_ready, count=1)

    # ---- tensor memory (512 columns): S region | dP / dQ^T region | dK | dV
    stage_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, BLOCK_M), col_stride=1)
    acc_layout: gl.constexpr = TensorMemoryLayout((BLOCK_N, HEAD_DIM), col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, BLOCK_M], stage_layout)
    dp_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, BLOCK_M], stage_layout)
    dk_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, HEAD_DIM], acc_layout)
    dv_tmem = allocate_tensor_memory(gl.float32, [BLOCK_N, HEAD_DIM], acc_layout)

    a = ((q_desc, do_ptr, k_desc, v_desc, lse_desc, delta_desc),
         (k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem, stage_smem),
         (s_tmem, dp_tmem, dk_tmem, dv_tmem),
         (kv_ready, q_ready, s_ready, dp_ready, p_ready, ds_ready, dv_done, dk_done, dq_done, dq_empty, dkv_ready,
          s_cons, dp_cons, do_ready),
         (b, hk, j, num_tiles))

    gl.warp_specialize([
        (_softmax_partition_a, (a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off, SEQ, NUM_KV_HEADS,
                                GROUP, NUM_Q_BLOCKS, ABL, DBG, num_warps)),
        (_softmax_partition_b, (a, dk_ptr, dv_ptr, sm_scale, scale_log2e, dbg_ptr, p_off, ds_off, SEQ, NUM_KV_HEADS,
                                GROUP, NUM_Q_BLOCKS, ABL, DBG)),
        (_drain_partition, (a, dq_desc, sm_scale, dbg_ptr, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS, ABL, DBG)),
        (_mma_partition, (a, dbg_ptr, SEQ, NUM_HEADS, GROUP, NUM_Q_BLOCKS, DBG, ABL)),
    ], [8, 8, 1], [72, 72, 64])


# ----------------------------------------------------------------------------------------------
# Pre/post-processing kernels (plain Triton)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _preprocess_kernel(o_ptr, do_ptr, lse_ptr, delta_ptr, lse2_ptr, dq_accum_ptr, SEQ: tl.constexpr,
                       NUM_HEADS: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr, LOG2E: tl.constexpr):
    """delta[b,h,n] = sum_d O*dO ; lse2 = lse*log2(e) ; zero dq_accum, for ROWS consecutive (b,n,h) rows."""
    pid = tl.program_id(0)
    row = pid * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, D)
    offs = row[:, None] * D + col[None, :]
    o = tl.load(o_ptr + offs).to(tl.float32)
    do = tl.load(do_ptr + offs).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    h = row % NUM_HEADS
    bn = row // NUM_HEADS
    b = bn // SEQ
    n = bn % SEQ
    vidx = (b * NUM_HEADS + h) * SEQ + n
    tl.store(delta_ptr + vidx, delta)
    tl.store(lse2_ptr + vidx, tl.load(lse_ptr + vidx) * LOG2E)
    tl.store(dq_accum_ptr + offs, tl.zeros([ROWS, D], dtype=tl.float16))


@triton.jit
def _postprocess_dq_kernel(dq_accum_ptr, dq_ptr, SEQ: tl.constexpr, NUM_HEADS: tl.constexpr, D: tl.constexpr,
                           BN: tl.constexpr):
    """dq[b, n, h, d] = dq_accum[b, h, d, n]  (fp16 -> bf16, tile transpose)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS
    d = tl.arange(0, D)
    n = pid_n * BN + tl.arange(0, BN)
    x = tl.load(dq_accum_ptr + (pid_bh * D + d[:, None]) * SEQ + n[None, :])   # [D, BN]
    dst = dq_ptr + ((b * SEQ + n[:, None]) * NUM_HEADS + h) * D + d[None, :]
    tl.store(dst, tl.trans(x).to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------
def _head_major_desc(x, block_rows):
    """x: (B, N, Hx, D) contiguous -> TMA descriptor over the (Hx, B*N, D) strided view."""
    Bsz, N, Hx, Dd = x.shape
    x3 = x.view(Bsz * N, Hx, Dd).permute(1, 0, 2)
    layout = gl.NVMMASharedLayout.get_default_for([block_rows, Dd], gl.bfloat16)  # rank-2 layout, 3D box
    return TensorDescriptor.from_tensor(x3, [1, block_rows, Dd], layout)


_BARRIER_OFFSETS = {}


def _find_barrier_offsets(ptx, count, n):
    """Byte offsets (from the dynamic smem base) of the first n mbarriers initialised with `count`, in PTX order."""
    import re
    lines = ptx.split("\n")
    regs = []
    for l in lines:
        m = re.search(r"mbarrier\.init\.shared::cta\.b64 \[(%r\d+)\], (\d+);", l)
        if m and int(m.group(2)) == count:
            regs.append(m.group(1))
    assert len(regs) >= n, f"found only {len(regs)} barriers with count {count}"
    offs = []
    for r in regs[:n]:
        pat = re.compile(r"add\.s32\s+" + re.escape(r) + r",\s*%r\d+,\s*(\d+);")
        found = [int(m.group(1)) for l in lines for m in [pat.search(l)] if m]
        assert len(found) == 1, f"barrier register {r}: {found}"
        offs.append(found[0])
    return offs


def gqa_bwd(q, k, v, o, do, lse, sm_scale=None, abl=0, dbg=None):
    """Returns (dq, dk, dv) in bf16. lse: (B, H, N) fp32 natural-log logsumexp from the forward."""
    Bsz, N, H, Dd = q.shape
    H_KV = k.shape[2]
    assert Dd == _HEAD_DIM and N % _BLOCK_N == 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Dd)
    for t in (q, k, v, o, do):
        assert t.is_contiguous() and t.dtype == torch.bfloat16
    lse = lse.contiguous().float()

    delta = torch.empty(Bsz, H, N, device=q.device, dtype=torch.float32)
    lse2 = torch.empty(Bsz, H, N, device=q.device, dtype=torch.float32)
    dq_accum = torch.empty(Bsz, H, Dd, N, device=q.device, dtype=torch.float16)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    ROWS = 16
    _preprocess_kernel[(Bsz * N * H // ROWS, )](o, do, lse, delta, lse2, dq_accum, SEQ=N, NUM_HEADS=H, D=Dd,
                                                ROWS=ROWS, LOG2E=_LOG2E, num_warps=4)

    vec_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_M], gl.float32)
    lse_desc = TensorDescriptor.from_tensor(lse2.view(-1), [_BLOCK_M], vec_layout)
    delta_desc = TensorDescriptor.from_tensor(delta.view(-1), [_BLOCK_M], vec_layout)
    q_desc = _head_major_desc(q, _BLOCK_M)
    dq_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_N, _HALF], gl.float16)
    dq_desc = TensorDescriptor.from_tensor(dq_accum.view(Bsz * H, Dd, N), [1, _BLOCK_N, _HALF], dq_layout)
    k_desc = _head_major_desc(k, _BLOCK_N)
    v_desc = _head_major_desc(v, _BLOCK_N)

    grid = ((N // _BLOCK_N) * Bsz * H_KV, )
    dbg_t = dbg if dbg is not None else torch.zeros(1, device=q.device, dtype=torch.int32)
    kw = dict(SEQ=N, NUM_HEADS=H, NUM_KV_HEADS=H_KV, ABL=abl, DBG=dbg is not None, num_warps=8, maxnreg=72)
    key = (N, H, H_KV, abl, dbg is not None)
    if key not in _BARRIER_OFFSETS:
        compiled = gqa_bwd_kernel.warmup(q_desc, k_desc, v_desc, do, lse_desc, delta_desc, dq_desc, dk, dv,
                                         sm_scale, sm_scale * _LOG2E, dbg_t, 0, 0, grid=grid, **kw)
        _BARRIER_OFFSETS[key] = _find_barrier_offsets(compiled.asm["ptx"], count=16, n=2)
    p_off, ds_off = _BARRIER_OFFSETS[key]
    gqa_bwd_kernel[grid](q_desc, k_desc, v_desc, do, lse_desc, delta_desc, dq_desc, dk, dv,
                         sm_scale, sm_scale * _LOG2E, dbg_t, p_off, ds_off, **kw)
    dq = torch.empty_like(q)
    BN = 32
    _postprocess_dq_kernel[(N // BN, Bsz * H)](dq_accum, dq, SEQ=N, NUM_HEADS=H, D=Dd, BN=BN, num_warps=4)
    return dq, dk, dv
