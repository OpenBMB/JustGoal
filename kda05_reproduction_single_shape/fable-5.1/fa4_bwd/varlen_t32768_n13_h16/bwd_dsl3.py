"""CuTe DSL (Blackwell sm_100a) warp-specialized attention backward main kernel, v9 "persistent half-tile pipeline".

One 512-thread CTA per SM loops over work items (kv tile, head) in LPT order; the loader warp claims items with an atomic
counter and publishes them through an SMEM ring. Per item the kernel is the v8e half-tile pipeline:
q tiles of 128 rows are processed as two 64-row halves (a: rows 0-63, b: rows 64-127) for S^T, dP^T and dQ^T so
that TMEM holds one half of each plus the packed P tile; MMA warp A issues the next tile's S/dP halves while
the compute warps still work on the current tile.

  TMEM (512 cols): dK[0:128] | dV[128:256] | P bf16 packed [256:320] | S half [320:384] | dP half [384:448] | dQ^T half [448:512]
  warps 0-3   : reduce  (dQ^T half: TMEM (lane=d, col=q) -> f16 SMEM (q, d) per warp -> TMA reduce-add into the f16 dq accumulator)
                + per-item dV/dK epilogue (TMEM -> bf16 global)
  warps 4-7   : compute WG0: P = exp2(S*scale*log2e - lse*log2e) masked -> TMEM bf16 (both halves)
  warps 8-11  : compute WG1: dS = P*(dP-delta) -> SMEM bf16 (both halves); P read back from TMEM
  warp 12     : MMA warp A: S_x^T = K Q_x^T, dP_x^T = V dO_x^T (N=64)
  warp 13     : item fetch (atomic counter -> ring) + TMA loads of Q / dO half-slots and the LSE/delta rows
  warp 14     : MMA warp B: dV += P^T dO, dK += dS^T Q (K=64 halves), dQ_x^T = K^T dS_x^T (M=128 d, N=64 q)
  warp 15     : K/V loader: the two 32 KB buffers alternate roles per item (item n: K in buffer n&1, V in the other);
                ATT_KV_BUFS=3 selects a 3-way rotation (V(n+1) a full item ahead) that measured slower (Q/dO slot cost)
  SMEM: K/V 64K | Q 64K | dO 48K | dS^T 32K | dQ staging 4 warps x 2K = 216K + extra blocks / barriers / ring
  v9f: the dK/dV epilogue (TMEM -> bf16 global) runs in WG1 after its last dS of the item (ATT_EPI_WG1=1), and warp A
  issues S_b before waiting for V (ATT_PRO_SB=1)
"""

import os as _os

import torch
import triton
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm

TILE = 128          # kv rows (MMA M) == q rows per iteration
HALF = 64           # q rows per half (MMA N of S/dP/dQ^T, K of dV/dK halves)
HEAD_DIM = 128
KV_BUFS = int(_os.environ.get("ATT_KV_BUFS", "2"))       # 32 KB K/V buffers: 2 = alternating pair (default), 3 = 3-way role rotation (measured slower: the displaced Q/dO slots cost more than the V wait)
KV_ROT = int(_os.environ.get("ATT_KV_ROT", "1"))         # 3-buffer rotation: 0 = K(n) in n%3 (V(n+1) loads a full item ahead), 1 = V(n) in n%3 (K(n+1) loads a full item ahead)
PRO_SB_FIRST = _os.environ.get("ATT_PRO_SB", "1") == "1"   # warp A prologue: issue S_b before waiting for V (S_a, S_b, dP_a, dP_b)
Q_STAGES = int(_os.environ.get("ATT_Q_STAGES", "3" if KV_BUFS == 3 else "4"))      # Q half-slots (16 KB each)
DO_STAGES = int(_os.environ.get("ATT_DO_STAGES", "2" if KV_BUFS == 3 else "3"))    # dO half-slots
DQ_COLS = 32                                              # d columns per reduce warp (= TMEM lanes per warp)
DQ_QCOLS = int(_os.environ.get("ATT_DQ_QCOLS", "32"))     # q columns per TMA reduce box (staging = 4 warps x 32 d x DQ_QCOLS f16)
DQ_ROWS_D = 16                                            # direct mode: q rows per TMA reduce box (16 x 64 d bf16 = 2 KB per warp)
Q_REVERSE = _os.environ.get("ATT_QREV", "1") == "1"
RING = 8                                                  # work-item ring entries
RING_W = 8                                                # int32 words per ring entry
NUM_WARPS = 16
NUM_THREADS = NUM_WARPS * 32
REDUCE_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7, 8, 9, 10, 11)
MMA_WARP = 12       # issues S^T, dP^T
LOAD_WARP = 13      # item fetch + Q/dO loads
MMA2_WARP = 14      # issues dV, dK, dQ^T
KV_WARP = 15        # K/V loads
NUM_COMPUTE_THREADS = 256
NUM_WG_THREADS = 128
NUM_REDUCE_THREADS = 128

TMEM_DK = 0
TMEM_DV = 128
TMEM_P = 256
TMEM_S = 320
TMEM_DP = 384
TMEM_DQ = 448

LOG2E = 1.4426950408889634

IKET = _os.environ.get("ATT_IKET", "0") == "1"
EXP_NOCOMPUTE = _os.environ.get("ATT_NOCOMPUTE", "0") == "1"  # experiment: compute warps only run the pipeline skeleton
EXP_NOREDUCE = _os.environ.get("ATT_NOREDUCE", "0") == "1"    # experiment: reduce warps only run the pipeline skeleton
EXP_NOEXP = _os.environ.get("ATT_NOEXP", "0") == "1"
EXP_NOTAIL = _os.environ.get("ATT_NOTAIL", "0") == "1"      # experiment: skip the 9th k-step MMAs (timing only)
EXP_NOEPI = _os.environ.get("ATT_NOEPI", "0") == "1"        # experiment: skip the dV/dK epilogue stores (timing only)
EXP_KV_EARLY = _os.environ.get("ATT_KV_EARLY", "0") == "1"  # experiment (timing only, races): load the next V without waiting for the K buffer release
XK = 512          # bf16 elements per extra B block (64 rows x 8: first K core-matrix column of the 9th k-step)
XBLK_BYTES = 1024
EXP_SAMEQ = _os.environ.get("ATT_SAMEQ", "0") == "1"
EXP_K1 = _os.environ.get("ATT_MMA_K1", "0") == "1"           # experiment: one k-step per GEMM (timing only)
NKSTEPS = int(_os.environ.get("ATT_MMA_K", "8"))
NK8 = 1 if EXP_K1 else NKSTEPS
NK4 = 1 if EXP_K1 else min(4, NKSTEPS)
KV_BLK16 = 2048   # 16-byte units per 32 KB K/V buffer
KV_PREFETCH = _os.environ.get("ATT_KV_PREFETCH", "1") == "1"   # L2 prefetch of the next item's K/V when the item is claimed
DQ_RED = _os.environ.get("ATT_DQ_RED", "0") == "1"            # experiment: dQ partials via red.global.add.v4.f16x2 from registers (no staging); measured 1.92 ms vs 1.54: the per-SM 16-B atomic rate limits
DQ_DIRECT = _os.environ.get("ATT_DQ_DIRECT", "1") == "1"      # dQ = dS K computed with q in TMEM lanes and reduce-added (bf16) straight into dq (T, H, D)
DQ_DIRECT_F16 = DQ_DIRECT and _os.environ.get("ATT_DQ_DIRECT_F16", "0") == "1"   # experiment: direct mode into an fp16 (T, H, D) accumulator (flat convert)
EPI_WG1 = _os.environ.get("ATT_EPI_WG1", "1") == "1"     # dK/dV epilogue done by the dS warpgroup (idle at item boundaries while V(n+1) loads) instead of the reduce warps (busy draining dQ)
EPI_TMA = DQ_DIRECT and not EPI_WG1 and _os.environ.get("ATT_EPI_TMA", "0") == "1"   # dK/dV epilogue: (16 kv x 64 d) boxes staged in the dQ staging buffer + TMA store (full 128-B lines) instead of row-per-thread 16-B global stores
DQ_ORDER = int(_os.environ.get("ATT_DQ_ORDER", "1"))      # DQ_DIRECT issue order: 0 = dK_0, dK_1, dQ_0, dQ_1; 1 = dK_0, dQ_0, dK_1, dQ_1
DQ_TILED = (not DQ_DIRECT) and _os.environ.get("ATT_DQ_TILED", "1") == "1"   # accumulator (H, T_cap/64, D, 64): contiguous 16 KB per 64-token block
CVT_INK = DQ_TILED and _os.environ.get("ATT_CVT_INK", "0") == "1"   # experiment (rejected, see draft): in-kernel dQ conversion by the diagonal tile's KV warp after per-block completion counting
CVT_DBG = int(_os.environ.get("ATT_CVT_DBG", "0"))        # debug: 1 = count only (never push jobs), 2 = push/pop without converting
CVT_TAIL = int(_os.environ.get("ATT_CVT_TAIL", "2"))      # newest halves of an item whose completion is counted one item later (even)
CVT_STATS = _os.environ.get("ATT_CVT_STATS", "0") == "1"   # debug: per-CTA counters appended to blk_cnt (8 per CTA)
CVT_ATOM = _os.environ.get("ATT_CVT_ATOM", "release")     # counter atomic semantics: release (default) | acqrel | relaxed (debug only)        # halves between a dQ reduce issue and its completion check
CVTQ = 64                                                 # conversion job queue entries per CTA (global memory ring)
CVTQ_W = 6                                                # int32 words per job entry (h, blk, t0, nvalid, thr, seq)
TABLE_FULL = int(_os.environ.get("ATT_ORDER", "2")) == 2       # item table holds (head, seq, kv_block) entries (one per item)
QDO_PF = int(_os.environ.get("ATT_QDO_PF", "0"))                # L2 prefetch distance (q blocks) for Q/dO halves, 0 = off (measured: +3% slower)


def kv_sched(n):
    """Per-item K/V buffer schedule. Returns (kb, vb, k_full_par, v_full_par, k_empty_par, v_empty_par):
    K(n) lives in buffer kb, V(n) in vb; *_full_par are the kv_full parities the consumers wait for; the loader waits
    kv_empty_v[kb] with k_empty_par before loading K(n) and kv_empty_k[vb] with v_empty_par before loading V(n).
    KV_BUFS == 3: K(n) -> n % 3, V(n) -> (n + 1) % 3, so a buffer's uses alternate K, V, K, ... and V(n+1) replaces
    K(n-1) (released at the end of item n-1) instead of K(n) (released at the end of item n)."""
    if KV_BUFS == 3 and KV_ROT == 0:
        kb = n % 3
        vb = (n + 1) % 3
        ck_k = n // 3                 # prior K uses of buffer kb
        cv_k = (n + 2) // 3           # prior V uses of buffer kb
        ck_v = (n + 1) // 3           # prior K uses of buffer vb
        cv_v = n // 3                 # prior V uses of buffer vb
        return kb, vb, (ck_k + cv_k) & 1, (ck_v + cv_v) & 1, (cv_k - 1) & 1, (ck_v - 1) & 1
    if KV_BUFS == 3:
        # mirrored rotation: V(n) -> n % 3, K(n) -> (n + 1) % 3; K(n) replaces V(n-2) (released after item n-2's last dP),
        # V(n) replaces K(n-1) (released at the end of item n-1)
        vb = n % 3
        kb = (n + 1) % 3
        ck_k = n // 3                 # prior K uses of buffer kb
        cv_k = (n + 1) // 3           # prior V uses of buffer kb
        ck_v = (n + 2) // 3           # prior K uses of buffer vb
        cv_v = n // 3                 # prior V uses of buffer vb
        return kb, vb, (ck_k + cv_k) & 1, (ck_v + cv_v) & 1, (cv_k - 1) & 1, (ck_v - 1) & 1
    kb = n & 1
    vb = 1 - kb
    role_phase = (((n + 1) >> 1) - 1) & 1
    return kb, vb, n & 1, n & 1, role_phase, role_phase


def _rpush(name):
    if IKET:
        cute.experimental.iket.range_push(name)


def _rpop():
    if IKET:
        cute.experimental.iket.range_pop()


# named barrier ids
BAR_TMEM_ALLOC = 1
BAR_FINAL = 4

# setmaxnreg budget (kernel compiled at 128 regs/thread x 512 threads = 64K)
REGS_WG3 = int(_os.environ.get("ATT_REGS_WG3", "56"))        # mma / load / kv warps (12-15)
REGS_REDUCE = int(_os.environ.get("ATT_REGS_REDUCE", "104"))  # warps 0-3
REGS_COMPUTE = int(_os.environ.get("ATT_REGS_COMPUTE", "176"))  # warps 4-11
assert REGS_WG3 * 128 + REGS_REDUCE * 128 + REGS_COMPUTE * 256 <= 65536


def _set_regs(n):
    """setmaxnreg relative to the 128-register launch value (inc above, dec below)."""
    if n > 128:
        cute.arch.warpgroup_reg_alloc(n)
    elif n < 128:
        cute.arch.warpgroup_reg_dealloc(n)


# ---------------------------------------------------------------- UMMA descriptor encoding (PTX ISA 9.7.16.x)
_SWZ_LAYOUT_TYPE = {0: 0, 1: 6, 2: 4, 3: 2}   # swizzle bits -> layout_type field (NONE, 32B, 64B, 128B)
_SWZ_ATOM_MN = {0: 1, 6: 2, 4: 4, 2: 8}


def _umma_smem_desc_base(layout128, swz, k_major: bool) -> int:
    """64-bit smem descriptor without start address for a canonical UMMA layout (in 16-byte units)."""
    B, M, S = swz.num_bits, swz.num_base, swz.num_shift
    assert M == 4 and S == 3, (B, M, S)
    lt = _SWZ_LAYOUT_TYPE[B]
    atom_mn = _SWZ_ATOM_MN[lt]
    if k_major:
        canon = cute.logical_divide(layout128, (8, 2))
        sbo = canon.stride[0][1]
        lbo = canon.stride[1][0]
    else:
        canon = cute.logical_divide(layout128, (atom_mn, 8))
        s01, s11 = canon.stride[0][1], canon.stride[1][1]
        if lt == 0:
            sbo, lbo = s01, s11
        else:
            sbo, lbo = s11, s01
    desc = 0
    desc |= (int(lbo) & 0x3FFF) << 16
    desc |= (int(sbo) & 0x3FFF) << 32
    desc |= 1 << 46          # version (sm100)
    desc |= (lt & 0x7) << 61
    return desc & 0xFFFFFFFFFFFFFFFF


def _umma_idesc(op) -> int:
    fmt = {cutlass.Float16: 0, cutlass.BFloat16: 1}
    a_major = 0 if op.a_major_mode == cute.nvgpu.OperandMajorMode.K else 1
    b_major = 0 if op.b_major_mode == cute.nvgpu.OperandMajorMode.K else 1
    M, N = op.shape_mnk[0], op.shape_mnk[1]
    d = 0
    d |= 1 << 4                      # c_format f32
    d |= fmt[op.a_dtype] << 7
    d |= fmt[op.b_dtype] << 10
    d |= a_major << 15
    d |= b_major << 16
    d |= ((N >> 3) & 0x3F) << 17
    d |= ((M >> 4) & 0x1F) << 24
    return d & 0xFFFFFFFF


def _desc_base_of(smem_stage_tensor, k_major: bool) -> int:
    lay = cute.recast_layout(128, smem_stage_tensor.element_type.width, smem_stage_tensor.layout[0])
    return _umma_smem_desc_base(lay, smem_stage_tensor.iterator.type.swizzle_type, k_major)


def _kblock_offsets_16B(smem_stage_tensor):
    nk = cute.size(smem_stage_tensor.shape[2])
    offs = [(cute.crd2idx((0, 0, k), smem_stage_tensor.layout) * smem_stage_tensor.element_type.width // 8) >> 4 for k in range(nk)]
    return offs


def _asm_smem_a(hi_a, hi_b, offs_a, offs_b, idesc, ks_a, ks_b):
    asm = ("{\n\t.reg .pred lt;\n\t.reg .pred p;\n\t.reg .b32 alo, blo;\n\t.reg .b64 da, db;\n\t"
           "elect.sync _|lt, -1;\n\tmov.b32 alo, $1;\n\tmov.b32 blo, $2;\n\tsetp.ne.b32 p, $3, 0;\n\t"
           f"mov.b64 da, {{alo, {hex(hi_a)}}};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
           f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, {hex(idesc)}, p;\n\t")
    for j in range(1, len(ks_a)):
        asm += (f"add.u32 alo, alo, {hex(offs_a[ks_a[j]] - offs_a[ks_a[j - 1]])};\n\t"
                f"add.u32 blo, blo, {hex(offs_b[ks_b[j]] - offs_b[ks_b[j - 1]])};\n\t"
                f"mov.b64 da, {{alo, {hex(hi_a)}}};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
                f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, {hex(idesc)}, 1;\n\t")
    return asm + "}\n"


def _asm_tmem_a(hi_b, offs_b, idesc, ks_b):
    asm = ("{\n\t.reg .pred lt;\n\t.reg .pred p;\n\t.reg .b32 blo;\n\t.reg .b64 db;\n\t"
           "elect.sync _|lt, -1;\n\tmov.b32 blo, $2;\n\tsetp.ne.b32 p, $3, 0;\n\t"
           f"mov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
           f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], db, {hex(idesc)}, p;\n\t")
    for j in range(1, len(ks_b)):
        asm += (f"add.u32 blo, blo, {hex(offs_b[ks_b[j]] - offs_b[ks_b[j - 1]])};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
                f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], [$1 + {hex(8 * j)}], db, {hex(idesc)}, 1;\n\t")
    return asm + "}\n"


@cute.jit
def gemm_ptx(op, acc: cute.Tensor, sA_stage, sB_stage, tmem_a, accumulate, ks_a, ks_b, a_off16):
    """Issue the k-blocks ks_a (A) x ks_b (B) of one GEMM with tcgen05.mma from the elected lane.
    sA_stage/sB_stage: smem tensors of ONE pipeline stage with layout (MMA, M|N, K); tmem_a: Int32 TMEM address of the
    first A k-block (or None; then 8 columns per k-block). accumulate: Int32/Boolean (0 => overwrite on the first k-block).
    a_off16: dynamic Int32 offset (16-byte units) added to the A start address (K/V buffer selection)."""
    is_ts = tmem_a is not None
    idesc = _umma_idesc(op)
    b_kmajor = op.b_major_mode == cute.nvgpu.OperandMajorMode.K
    base_b = _desc_base_of(sB_stage, b_kmajor)
    hi_b = (base_b >> 32) & 0xFFFFFFFF
    lo_b = base_b & 0xFFFFFFFF
    offs_b = _kblock_offsets_16B(sB_stage)
    b_lo = cutlass.Int32(lo_b) + cutlass.Int32(((sB_stage.iterator.toint() & 0x3FFFF) >> 4) + offs_b[ks_b[0]])
    acc_flag = cutlass.Int32(accumulate)
    if cutlass.const_expr(not is_ts):
        a_kmajor = op.a_major_mode == cute.nvgpu.OperandMajorMode.K
        base_a = _desc_base_of(sA_stage, a_kmajor)
        hi_a = (base_a >> 32) & 0xFFFFFFFF
        lo_a = base_a & 0xFFFFFFFF
        offs_a = _kblock_offsets_16B(sA_stage)
        a_lo = cutlass.Int32(lo_a) + cutlass.Int32(((sA_stage.iterator.toint() & 0x3FFFF) >> 4) + offs_a[ks_a[0]]) + cutlass.Int32(a_off16)
        asm = _asm_smem_a(hi_a, hi_b, offs_a, offs_b, idesc, ks_a, ks_b)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), a_lo.ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    else:
        asm = _asm_tmem_a(hi_b, offs_b, idesc, ks_b)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), cutlass.Int32(tmem_a).ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def l2_prefetch_line(addr):
    """prefetch.global.L2 of the 128-byte line at the global byte address addr (Int64)."""
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value()], "prefetch.global.L2 [$0];", "l",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_atom_add_acqrel_gpu(addr, v):
    """old = atom.acq_rel.gpu.global.add.s32 [addr], v"""
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                                         "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_atom_add_relaxed_gpu(addr, v):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                                         "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_red_add(addr, v):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                    "red.relaxed.gpu.global.add.s32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_atom_add_release_gpu(addr, v):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                                         "atom.release.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def cvt_atomic(addr, v):
    if cutlass.const_expr(CVT_ATOM == "relaxed"):
        return ptx_atom_add_relaxed_gpu(addr, v)
    elif cutlass.const_expr(CVT_ATOM == "acqrel"):
        return ptx_atom_add_acqrel_gpu(addr, v)
    else:
        return ptx_atom_add_release_gpu(addr, v)


@cute.jit
def ptx_st_global_b32(addr, v):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                    "st.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_st_release_gpu_b32(addr, v):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(), cutlass.Int32(v).ir_value()],
                    "st.release.gpu.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_ld_acquire_gpu_b32(addr):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int64(addr).ir_value()],
                                         "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_ld_cg_b32(addr):
    """ld.global.cg.b32 (L2-coherent, bypasses L1)"""
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int64(addr).ir_value()],
                                         "ld.global.cg.b32 $0, [$1];", "=r,l",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_ld_cg_v4(addr, rT):
    """ld.global.cg.v4.b32 of 16 bytes into the Int32 (4,) register tensor rT (four outputs via an LLVM struct)"""
    from cutlass._mlir import ir
    i32 = ir.IntegerType.get_signless(32)
    sty = llvm.StructType.get_literal([i32, i32, i32, i32])
    r = llvm.inline_asm(sty, [cutlass.Int64(addr).ir_value()], "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    for i in cutlass.range_constexpr(4):
        rT[i] = cutlass.Int32(llvm.extractvalue(i32, r, [i]))


@cute.jit
def ptx_st_shared_v4(saddr, a, b, c, d):
    llvm.inline_asm(None, [cutlass.Int32(saddr).ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value(),
                           cutlass.Int32(c).ir_value(), cutlass.Int32(d).ir_value()],
                    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_st_v2_b32(addr, a, b):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()],
                    "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_st_v4_zero(addr):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value()],
                    "st.global.v4.b32 [$0], {0, 0, 0, 0};", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_shared_atom_add(saddr, v):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int32(saddr).ir_value(), cutlass.Int32(v).ir_value()],
                                         "atom.shared.add.u32 $0, [$1], $2;", "=r,r,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_shared_ld_acquire(saddr):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int32(saddr).ir_value()],
                                         "ld.acquire.cta.shared.b32 $0, [$1];", "=r,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_shared_st_release(saddr, v):
    llvm.inline_asm(None, [cutlass.Int32(saddr).ir_value(), cutlass.Int32(v).ir_value()],
                    "st.release.cta.shared.b32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_shared_st(saddr, v):
    llvm.inline_asm(None, [cutlass.Int32(saddr).ir_value(), cutlass.Int32(v).ir_value()],
                    "st.volatile.shared.b32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_shared_ld(saddr):
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int32(saddr).ir_value()],
                                         "ld.volatile.shared.b32 $0, [$1];", "=r,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_mbar_try_wait(saddr, parity):
    """1 if the mbarrier phase with the given parity has completed (single probe), else 0"""
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int32(saddr).ir_value(), cutlass.Int32(parity).ir_value()],
                                         "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [$1], $2;\n\tselp.u32 $0, 1, 0, p;\n\t}", "=r,r,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def ptx_shfl0(v):
    """broadcast lane 0's value to the warp"""
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [cutlass.Int32(v).ir_value()],
                                         "shfl.sync.idx.b32 $0, $1, 0, 0x1f, 0xffffffff;", "=r,r",
                                         has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


CVT_WD = _os.environ.get("ATT_CVT_WD", "0") == "1"          # debug: watchdog traps with a diagnostic print after ~2^22 spin iterations


@cute.jit
def ptx_trap():
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def ptx_nanosleep(ns):
    llvm.inline_asm(None, [cutlass.Int32(ns).ir_value()], "nanosleep.u32 $0;", "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def cvt_push_job(h_, blk_, t0_, nv_, thr_, cvt_push_addr, cvt_pop_addr, cvt_q_base, stats_base):
    """reduce warp 0, lane 0: queue a conversion job for this CTA's KV warp in the per-CTA global ring
    (entry: h, blk, t0, nvalid, threshold, seq); the ring holds CVTQ pending jobs (jobs wait for the head's slowest tile)"""
    job = ptx_shared_ld(cvt_push_addr)
    wd = cutlass.Int32(0)
    while job - ptx_shared_ld(cvt_pop_addr) >= CVTQ:
        if cutlass.const_expr(CVT_STATS):
            ptx_red_add(stats_base + 4, 1)
        ptx_nanosleep(256)
        if cutlass.const_expr(CVT_WD):
            wd += 1
            if wd == 4000000:
                cute.printf("WD push-full: job %d pop %d\n", job, ptx_shared_ld(cvt_pop_addr))
                ptx_trap()
    e_addr = cvt_q_base + cutlass.Int64(job % CVTQ) * (CVTQ_W * 4)
    ptx_st_global_b32(e_addr, h_)
    ptx_st_global_b32(e_addr + 4, blk_)
    ptx_st_global_b32(e_addr + 8, t0_)
    ptx_st_global_b32(e_addr + 12, nv_)
    ptx_st_global_b32(e_addr + 16, thr_)
    ptx_st_release_gpu_b32(e_addr + 20, job + 1)
    ptx_shared_st_release(cvt_push_addr, job + 1)
    if cutlass.const_expr(CVT_STATS):
        ptx_red_add(stats_base, 1)


@cute.jit
def cvt_ready(cst, job, cvt_q_base, cnt_base, nblk_acc):
    """KV warp: is job's block complete (all contributors counted)? If so, load the entry into cst and clear its seq word."""
    e_addr = cvt_q_base + cutlass.Int64(job % CVTQ) * (CVTQ_W * 4)
    ready = cutlass.Int32(0)
    if ptx_shfl0(ptx_ld_acquire_gpu_b32(e_addr + 20)) == job + 1:
        jh = ptx_shfl0(ptx_ld_cg_b32(e_addr))
        jblk_ = ptx_shfl0(ptx_ld_cg_b32(e_addr + 4))
        thr_ = ptx_shfl0(ptx_ld_cg_b32(e_addr + 16))
        cnt = ptx_shfl0(ptx_ld_cg_b32(cnt_base + cutlass.Int64(jh * nblk_acc + jblk_) * 4))
        if cutlass.const_expr(CVT_DBG == 2):
            cnt = thr_
        if cnt >= thr_:
            cst[2] = jh
            cst[3] = jblk_
            cst[4] = ptx_shfl0(ptx_ld_cg_b32(e_addr + 8))
            cst[5] = ptx_shfl0(ptx_ld_cg_b32(e_addr + 12))
            cst[0] = 1
            cst[1] = 0
            if cute.arch.lane_idx() == 0:
                ptx_st_global_b32(e_addr + 20, 0)        # consumed: the seq word is 0 again for the next run
            cute.arch.fence_acq_rel_gpu()
            ready = cutlass.Int32(1)
    return ready


@cute.jit
def cvt_chunk(cst, acc_base, dqo_base, dqo_rowb, nblk_acc, lane, sm_scale,
              rIn0, rIn1, rIn2, rIn3, rInH0, rInH1, rInH2, rInH3, rOutB, rOutW, stats_base):
    """KV warp: one step of the active job. Steps 2c (c < 8) issue the loads of q chunk c (8 q rows of the 128 d x 64 q f16 block,
    lane l: d = 4l..4l+3), steps 2c+1 convert and store them (dq[t0 + i, h, 4l..4l+3] = bf16(sm_scale * acc), 8-byte pieces),
    step 16 re-zeroes the block. The K/V barriers are polled between steps."""
    st = cst[1]
    jh = cst[2]
    jnv = cst[5]
    if cutlass.const_expr(CVT_STATS):
        if lane == 0:
            ptx_red_add(stats_base + 8, 1)
    src = acc_base + (cutlass.Int64(jh) * nblk_acc + cst[3]) * (HEAD_DIM * 64 * 2)
    if st < 16:
        c = st >> 1
        if c * 8 < jnv:
            if (st & 1) == 0:
                ptx_ld_cg_v4(src + (4 * lane + 0) * 128 + c * 16, rIn0)
                ptx_ld_cg_v4(src + (4 * lane + 1) * 128 + c * 16, rIn1)
                ptx_ld_cg_v4(src + (4 * lane + 2) * 128 + c * 16, rIn2)
                ptx_ld_cg_v4(src + (4 * lane + 3) * 128 + c * 16, rIn3)
                cst[1] = st + 1
            else:
                dst_col = dqo_base + cutlass.Int64(jh * HEAD_DIM + 4 * lane) * 2
                for i in cutlass.range_constexpr(8):
                    if c * 8 + i < jnv:
                        rOutB[0] = (rInH0[i].to(cutlass.Float32) * sm_scale).to(cutlass.BFloat16)
                        rOutB[1] = (rInH1[i].to(cutlass.Float32) * sm_scale).to(cutlass.BFloat16)
                        rOutB[2] = (rInH2[i].to(cutlass.Float32) * sm_scale).to(cutlass.BFloat16)
                        rOutB[3] = (rInH3[i].to(cutlass.Float32) * sm_scale).to(cutlass.BFloat16)
                        ptx_st_v2_b32(dst_col + cutlass.Int64(cst[4] + c * 8 + i) * dqo_rowb, rOutW[0], rOutW[1])
                cst[1] = st + 1
        else:
            cst[1] = 16
    else:
        if cutlass.const_expr(CVT_DBG != 2):
            for j in cutlass.range_constexpr(32):
                ptx_st_v4_zero(src + (j * 32 + lane) * 16)
        cst[0] = 0


@cute.jit
def red_f16x8(addr, r0, r1, r2, r3):
    """red.global.add.noftz.v4.f16x2 of 8 fp16 values (four packed 32-bit registers) at the 16-byte-aligned global address addr."""
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(), cutlass.Int32(r0).ir_value(), cutlass.Int32(r1).ir_value(),
                           cutlass.Int32(r2).ir_value(), cutlass.Int32(r3).ir_value()],
                    "red.global.add.noftz.v4.f16x2 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def mma_tail_ptx(op, acc: cute.Tensor, a_lo, b_lo):
    """One extra K=16 MMA (accumulate) with explicit SWIZZLE_NONE K-major descriptors: a_lo/b_lo hold the start address
    and LBO fields; SBO = 128 B (8-row core matrices), version 1."""
    if cutlass.const_expr(EXP_NOTAIL):
        return
    idesc = _umma_idesc(op)
    hi = 0x4008
    asm = ("{\n\t.reg .pred lt;\n\t.reg .b64 da, db;\n\telect.sync _|lt, -1;\n\t"
           f"mov.b64 da, {{$1, {hex(hi)}}};\n\tmov.b64 db, {{$2, {hex(hi)}}};\n\t"
           f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, {hex(idesc)}, 1;\n\t}}\n")
    llvm.inline_asm(None, [acc.iterator.toint().ir_value(), cutlass.Int32(a_lo).ir_value(), cutlass.Int32(b_lo).ir_value()],
                    asm, "r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.struct
class SharedStorage:
    q_bars: cute.struct.MemRange[cutlass.Int64, 2 * Q_STAGES]     # load->mma Q half-slots
    do_bars: cute.struct.MemRange[cutlass.Int64, 2 * DO_STAGES]   # load->mma dO half-slots
    kv_full: cute.struct.MemRange[cutlass.Int64, KV_BUFS]         # K/V buffer loaded (per buffer, once per use)
    kv_empty_k: cute.struct.MemRange[cutlass.Int64, KV_BUFS]      # buffer free after a K use (2 arrivals: S by warp A, dQ^T by warp B)
    kv_empty_v: cute.struct.MemRange[cutlass.Int64, KV_BUFS]      # buffer free after a V use (1 arrival: dP by warp A)
    cb_bar: cute.struct.MemRange[cutlass.Int64, 1]                # constant block loaded (once)
    s_bars: cute.struct.MemRange[cutlass.Int64, 2]                # mma->WG0 S half (1 stage)
    dp_bars: cute.struct.MemRange[cutlass.Int64, 2]               # mma->WG1 dP half (1 stage)
    dq_bars: cute.struct.MemRange[cutlass.Int64, 2]               # mma->reduce dQ^T half (1 stage)
    p_bars: cute.struct.MemRange[cutlass.Int64, 4]                # WG0->mma P half (2 stages: a, b)
    p2_bars: cute.struct.MemRange[cutlass.Int64, 4]               # WG0->WG1 P half (2 stages: a, b)
    ds_bars: cute.struct.MemRange[cutlass.Int64, 4]               # WG1->mma dS half (2 stages: a, b)
    dkdv_bars: cute.struct.MemRange[cutlass.Int64, 4]             # mma B <-> reduce: dK/dV TMEM regions (2 stages)
    tile_bars: cute.struct.MemRange[cutlass.Int64, RING]          # work-item ring entry published
    tile_ring: cute.struct.MemRange[cutlass.Int32, RING * RING_W]  # (tile, head, seq_start, seqlen, acc_start, ...)
    cvt_ctr: cute.struct.MemRange[cutlass.Int32, 4]                 # push count, pop count, reduce warps done
    tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]


class AttnBwdSm100V9:
    def __init__(self):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        K, MN = cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN
        cg = tcgen05.CtaGroup.ONE
        # S_x^T = K Q_x^T and dP_x^T = V dO_x^T : M=kv 128, N=64 q, A (K/V) K-major SMEM, B (Q/dO half) K-major SMEM
        self.op_s = tcgen05.MmaF16BF16Op(bf16, f32, (TILE, HALF, 16), cg, tcgen05.OperandSource.SMEM, K, K)
        # dV += P^T dO : A (P^T) from TMEM K-major, B (dO half) MN-major view (N=d 128, K=q 64)
        self.op_pdo = tcgen05.MmaF16BF16Op(bf16, f32, (TILE, HEAD_DIM, 16), cg, tcgen05.OperandSource.TMEM, K, MN)
        # dK += dS^T Q : A (dS^T) K-major SMEM, B (Q half) MN-major view
        self.op_dsq = tcgen05.MmaF16BF16Op(bf16, f32, (TILE, HEAD_DIM, 16), cg, tcgen05.OperandSource.SMEM, K, MN)
        # dQ_x^T = K^T dS_x^T : A (K^T) MN-major view of the K tile (M=128 d, K=kv), B (dS_x^T) MN-major view (N=64 q, K=kv)
        self.op_dq = tcgen05.MmaF16BF16Op(bf16, f32, (TILE, HALF, 16), cg, tcgen05.OperandSource.SMEM, MN, MN)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDO: cute.Tensor,
                 mDQacc: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor, tile_table: cute.Tensor,
                 seq_acc: cute.Tensor, counter: cute.Tensor, n_items: cutlass.Int32, num_heads: cutlass.Int32,
                 sm_scale: cutlass.Float32, mXQ: cute.Tensor, mXDO: cute.Tensor, mCB: cute.Tensor,
                 grid_x: cutlass.Int32, stream: cuda.CUstream, mDQout: cute.Tensor, blk_cnt: cute.Tensor, nblk_acc: cutlass.Int32,
                 cvt_q: cute.Tensor):
        mDQacc_raw = mDQacc
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        mma_s = cute.make_tiled_mma(self.op_s)
        mma_pdo = cute.make_tiled_mma(self.op_pdo)
        mma_dsq = cute.make_tiled_mma(self.op_dsq)
        mma_dq = cute.make_tiled_mma(self.op_dq)

        tiler_kv = (TILE, HALF, HALF)            # K/V: A tile (128 kv, 64 d) per d-block
        tiler_s = (TILE, HALF, HEAD_DIM)         # S half: B tile (64 q, 128 d)
        tiler_dk = (TILE, HEAD_DIM, HALF)        # dK/dV half: B tile (128 d, 64 q) MN-major view
        tiler_dqT = (TILE, HALF, TILE)           # dQ^T half: A (128 d, 128 kv), B (64 q, 128 kv)
        tiler_ds = (TILE, HEAD_DIM, TILE)        # dS^T K-major A (128 kv, 128 q)

        kv_layout = sm100_utils.make_smem_layout_a(mma_s, tiler_kv, bf16, 2 * KV_BUFS)   # (MMA, M, K=4, d-blocks: KV_BUFS buffers x 2)
        q_layout = sm100_utils.make_smem_layout_b(mma_s, tiler_s, bf16, Q_STAGES)      # (MMA, N=64, K=8, slots)
        do_layout = sm100_utils.make_smem_layout_b(mma_s, tiler_s, bf16, DO_STAGES)
        qT_layout = sm100_utils.make_smem_layout_b(mma_dsq, tiler_dk, bf16, Q_STAGES)  # B: n=d, k=q (4 k-steps)
        doT_layout = sm100_utils.make_smem_layout_b(mma_pdo, tiler_dk, bf16, DO_STAGES)
        kT_layout = sm100_utils.make_smem_layout_a(mma_dq, tiler_dqT, bf16, KV_BUFS)   # A: m=128 d, k=kv, per buffer (byte-identical to kv_layout)
        ds_layout = sm100_utils.make_smem_layout_a(mma_dsq, tiler_ds, bf16, 1)         # A: m=kv, k=q (K-major)
        dsT_layout = sm100_utils.make_smem_layout_b(mma_dq, tiler_dqT, bf16, 2)        # B: n=64 q, k=kv, 2 stages (16 KB each)
        kH_layout = sm100_utils.make_smem_layout_b(mma_dq, tiler_dqT, bf16, 2 * KV_BUFS)   # B: n=64 d (K/V d-block), k=kv: blocks [b0 d0, b0 d1, b1 d0, ...]
        p_layout = sm100_utils.make_smem_layout_a(mma_pdo, tiler_ds, bf16, 1)          # TMEM A layout for P^T (k=q 128)
        # dQ^T staging: per reduce warp one (32 d, DQ_QCOLS q) f16 box (128-B rows, SW128); the accumulator is (H, D, T)
        if cutlass.const_expr(DQ_DIRECT):
            # direct mode: per reduce warp one (16 q, 64 d) bf16 box = 128-B rows (full lines in dq), written in two passes of 16 lanes
            dq_epi_layout = sm100_utils.make_smem_layout_epi(cutlass.Float16 if DQ_DIRECT_F16 else cutlass.BFloat16, utils.LayoutEnum.ROW_MAJOR, (DQ_ROWS_D, HALF), 4)
        else:
            dq_epi_layout = sm100_utils.make_smem_layout_epi(cutlass.Float16, utils.LayoutEnum.ROW_MAJOR, (DQ_COLS, DQ_QCOLS), 4)

        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_atom, k_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(kv_layout, (None, None, None, 0)), tiler_kv, mma_s)
        v_atom, v_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mV, cute.slice_(kv_layout, (None, None, None, 0)), tiler_kv, mma_s)
        q_atom, q_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), tiler_s, mma_s)
        do_atom, do_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), tiler_s, mma_s)
        # 9th k-step operands: (8 cols, 64 rows) bf16 boxes of the per-token extra rows and the 4 KB constant block
        x_layout = cute.make_layout((8, HALF, Q_STAGES), stride=(1, 8, 8 * HALF))
        xq_atom, xq_tma = cpasync.make_tiled_tma_atom(tma_op, mXQ, cute.slice_(x_layout, (None, None, 0)), (8, HALF))
        xdo_atom, xdo_tma = cpasync.make_tiled_tma_atom(tma_op, mXDO, cute.slice_(x_layout, (None, None, 0)), (8, HALF))
        cb_layout = cute.make_layout((8, 192), stride=(1, 8))                 # ones block (128 rows) + zero block (64 rows)
        cb_atom, cb_tma = cpasync.make_tiled_tma_atom(tma_op, mCB, cb_layout, (8, 192))
        if cutlass.const_expr(DQ_TILED):
            # (64 q, D, nblk, H) tensor viewed as (D, (64 q, nblk), H): the q mode is hierarchical (block-contiguous)
            mDQacc = cute.make_tensor(mDQacc.iterator, cute.make_layout(
                (mDQacc.shape[1], (64, mDQacc.shape[2]), mDQacc.shape[3]),
                stride=(mDQacc.stride[1], (1, mDQacc.stride[2]), mDQacc.stride[3])))
        dq_red_atom, dq_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionKind.ADD), mDQacc,
            cute.slice_(dq_epi_layout, (None, None, 0)), (DQ_ROWS_D, HALF) if DQ_DIRECT else (DQ_COLS, DQ_QCOLS))

        if cutlass.const_expr(EPI_TMA):
            # dK / dV epilogue TMA stores: (16 kv rows, 64 d) bf16 boxes from the per-warp staging box into (T, D, H)
            st_op = cpasync.CopyBulkTensorTileS2GOp()
            dk_st_atom, dk_tma = cpasync.make_tiled_tma_atom(st_op, mDK, cute.slice_(dq_epi_layout, (None, None, 0)), (DQ_ROWS_D, HALF))
            dv_st_atom, dv_tma = cpasync.make_tiled_tma_atom(st_op, mDV, cute.slice_(dq_epi_layout, (None, None, 0)), (DQ_ROWS_D, HALF))
        else:
            dk_st_atom, dk_tma, dv_st_atom, dv_tma = dq_red_atom, dq_tma, dq_red_atom, dq_tma

        if cutlass.const_expr(_os.environ.get("ATT_PRINT_LAYOUTS", "0") == "1"):
            print("kv_layout", kv_layout)
            print("kT_layout", kT_layout)
            print("q_layout", q_layout)
            print("qT_layout", qT_layout)
            print("do_layout", do_layout)
            print("doT_layout", doT_layout)
            print("ds_layout", ds_layout)
            print("dsT_layout", dsT_layout)
            print("p_layout", p_layout)
            print("dq_epi_layout", dq_epi_layout)

        self.kernel(mma_s, mma_pdo, mma_dsq, mma_dq,
                    k_atom, k_tma, v_atom, v_tma, q_atom, q_tma, do_atom, do_tma, dq_red_atom, dq_tma,
                    mDK, mDV, cu_seqlens, tile_table, seq_acc, counter, n_items, num_heads,
                    kv_layout, q_layout, do_layout, qT_layout, kT_layout, doT_layout, ds_layout, dsT_layout, kH_layout,
                    p_layout, dq_epi_layout, sm_scale,
                    xq_atom, xq_tma, xdo_atom, xdo_tma, cb_atom, cb_tma, x_layout, cb_layout, mK, mV, mQ, mDO, mDQacc_raw, mDQout, blk_cnt, nblk_acc, cvt_q,
                    dk_st_atom, dk_tma, dv_st_atom, dv_tma).launch(
            grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    @cute.jit
    def epi_acc(self, tiled_ld, tTR_tX, rDQ0, rDQ1, rHa, rHb, rHaw, rHbw, scale, do_scale, stg_base, lane, warp_idx, kv0, jblk, seqlen,
                kv_valid, kv_tok, head, st_atom, tXs, tXg, mX, vec_atom, vec_layout):
        """dK / dV epilogue of one reduce warp (32 kv rows = TMEM lanes): per 64-d half, TMEM -> bf16 (x scale) -> the warp's
        (16, 64) SW128 staging box in two 16-lane passes -> TMA store of full 128-B lines into (T, D, H). Boxes that cross the
        sequence end fall back to per-lane 16-B stores of the valid rows."""
        row16 = lane & 15
        for pair in cutlass.range_constexpr(2):
            cute.copy(tiled_ld, tTR_tX[(None, None, None, 2 * pair)], rDQ0)
            cute.copy(tiled_ld, tTR_tX[(None, None, None, 2 * pair + 1)], rDQ1)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(do_scale):
                rHa.store((rDQ0.load() * scale).to(cutlass.BFloat16))
                rHb.store((rDQ1.load() * scale).to(cutlass.BFloat16))
            else:
                rHa.store(rDQ0.load().to(cutlass.BFloat16))
                rHb.store(rDQ1.load().to(cutlass.BFloat16))
            if cutlass.const_expr(not EXP_NOEPI):
                for p in cutlass.range_constexpr(2):
                    box_ok = kv0 + 32 * warp_idx + 16 * p + 16 <= seqlen        # warp-uniform
                    if box_ok:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)       # previous TMA store / reduce finished reading the box
                        cute.arch.sync_warp()
                        if (lane >> 4) == p:
                            for v in cutlass.range_constexpr(8):
                                w = (rHaw, rHbw)[v // 4]
                                j = 4 * (v % 4)
                                ptx_st_shared_v4(stg_base + row16 * 128 + ((v ^ (row16 & 7)) << 4), w[j], w[j + 1], w[j + 2], w[j + 3])
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        cute.copy(st_atom, tXs[(None, warp_idx)], tXg[(None, 8 * jblk + 2 * warp_idx + p, pair)])
                        cute.arch.cp_async_bulk_commit_group()
                    else:
                        if (lane >> 4) == p:
                            if kv_valid:
                                for c in cutlass.range_constexpr(2):
                                    rOv = cute.make_tensor((rHa, rHb)[c].iterator, vec_layout)
                                    dstv = cute.make_tensor((mX.iterator + mX.layout((kv_tok, 64 * pair + 32 * c, head))).align(min_align=16), vec_layout)
                                    cute.copy(vec_atom, rOv, dstv)

    @cute.jit
    def p_chunk(self, rS, rPb, chunk, masked, qbase, kv_row, kv_valid, seqlen, scale_log2):
        """P = exp2(scale*log2e * S') for one 32-column chunk held in rS (fp32; S' = S - lse/scale from the MMA) -> rPb (bf16).
        masked (constexpr): apply the causal / sequence-end / kv-tail mask."""
        if cutlass.const_expr(masked):
            for i in cutlass.range_constexpr(32):
                q_pos = qbase + chunk * 32 + i
                p = cute.math.exp2(rS[i] * scale_log2, fastmath=True)
                ok = (kv_row <= q_pos) and (q_pos < seqlen) and kv_valid
                rS[i] = p if ok else cutlass.Float32(0.0)
        else:
            for i in cutlass.range_constexpr(32):
                if cutlass.const_expr(EXP_NOEXP):
                    rS[i] = rS[i] * scale_log2
                else:
                    rS[i] = cute.math.exp2(rS[i] * scale_log2, fastmath=True)
        rPb.store(rS.load().to(cutlass.BFloat16))

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, mma_s: cute.TiledMma, mma_pdo: cute.TiledMma, mma_dsq: cute.TiledMma, mma_dq: cute.TiledMma,
               k_atom: cute.CopyAtom, mK: cute.Tensor, v_atom: cute.CopyAtom, mV: cute.Tensor,
               q_atom: cute.CopyAtom, mQ: cute.Tensor, do_atom: cute.CopyAtom, mDO: cute.Tensor,
               dq_atom: cute.CopyAtom, mDQ: cute.Tensor,
               mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor, tile_table: cute.Tensor,
               seq_acc: cute.Tensor, counter: cute.Tensor, n_items: cutlass.Int32, num_heads: cutlass.Int32,
               kv_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               qT_layout: cute.ComposedLayout, kT_layout: cute.ComposedLayout, doT_layout: cute.ComposedLayout,
               ds_layout: cute.ComposedLayout, dsT_layout: cute.ComposedLayout, kH_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               dq_epi_layout: cute.ComposedLayout, sm_scale: cutlass.Float32,
               xq_atom: cute.CopyAtom, mXQ: cute.Tensor, xdo_atom: cute.CopyAtom, mXDO: cute.Tensor,
               cb_atom: cute.CopyAtom, mCB: cute.Tensor, x_layout: cute.Layout, cb_layout: cute.Layout,
               mK_raw: cute.Tensor, mV_raw: cute.Tensor, mQ_raw: cute.Tensor, mDO_raw: cute.Tensor, mDQ_raw: cute.Tensor,
               mDQout: cute.Tensor, blk_cnt: cute.Tensor, nblk_acc: cutlass.Int32, cvt_q: cute.Tensor,
               dk_st_atom: cute.CopyAtom, mDKt: cute.Tensor, dv_st_atom: cute.CopyAtom, mDVt: cute.Tensor):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tiler_kv = (TILE, HALF, HALF)
        tiler_s = (TILE, HALF, HEAD_DIM)
        if cutlass.const_expr(IKET):
            setup_tok = cute.experimental.iket.range_start("setup")

        # ---------------- shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sKV = smem.allocate_tensor(bf16, kv_layout.outer, 1024, swizzle=kv_layout.inner)    # (MMA, M, K, 4): [buf0 db0, buf0 db1, buf1 db0, buf1 db1]
        sQ = smem.allocate_tensor(bf16, q_layout.outer, 1024, swizzle=q_layout.inner)      # (MMA, N, K, Q_STAGES)
        sDO = smem.allocate_tensor(bf16, do_layout.outer, 1024, swizzle=do_layout.inner)   # (MMA, N, K, DO_STAGES)
        sDS = smem.allocate_tensor(bf16, ds_layout.outer, 1024, swizzle=ds_layout.inner)
        sDQ = smem.allocate_tensor(cutlass.BFloat16 if (DQ_DIRECT and not DQ_DIRECT_F16) else cutlass.Float16, dq_epi_layout.outer, 1024, swizzle=dq_epi_layout.inner)  # (rows, cols, 4 warps)
        # 9th k-step operands (SWIZZLE_NONE K-major, 8x8 bf16 core matrices of 128 B):
        #   sOnes: A first K core column (16 M-cores: 128 kv rows; cols 0,1 = 1.0)      sZero: shared all-zero second K core column
        #   sQx / sDOx: per-slot B first K core column (8 N-cores: 64 q rows; cols 0,1 = (hi, lo) of -lse/scale resp. -delta)
        sQx = smem.allocate_tensor(bf16, x_layout, 128)                       # (8, 64, Q_STAGES)
        sDOx = smem.allocate_tensor(bf16, cute.make_layout((8, HALF, DO_STAGES), stride=(1, 8, 8 * HALF)), 128)
        sCB = smem.allocate_tensor(bf16, cb_layout, 128)                       # (8, 192): ones block (rows 0-127) + 1 KB zero block
        ones_addr = sCB.iterator.toint() & 0x3FFFF
        zero_addr = ones_addr + 2048
        qx_addr = sQx.iterator.toint() & 0x3FFFF
        dox_addr = sDOx.iterator.toint() & 0x3FFFF
        ones_lo = ones_addr >> 4          # LBO = 0: A's second K core aliases the first (B's second core is the zero block)
        # alias views (same bytes, different canonical layouts)
        sQT = cute.make_tensor(cute.recast_ptr(sQ.iterator, qT_layout.inner), qT_layout.outer)
        sKT = cute.make_tensor(cute.recast_ptr(sKV.iterator, kT_layout.inner), kT_layout.outer)   # (MMA, 128 d, K=8, 2 buffers)
        sDOT = cute.make_tensor(cute.recast_ptr(sDO.iterator, doT_layout.inner), doT_layout.outer)
        sDST = cute.make_tensor(cute.recast_ptr(sDS.iterator, dsT_layout.inner), dsT_layout.outer)
        sDSA = cute.make_tensor(cute.recast_ptr(sDS.iterator, kT_layout.inner), kT_layout.outer)   # dS as MN-major A (M=128 q, K=kv); stage 0 only
        sKH = cute.make_tensor(cute.recast_ptr(sKV.iterator, kH_layout.inner), kH_layout.outer)   # K d-halves as MN-major B (N=64 d, K=kv)
        ring = cute.make_tensor(storage.tile_ring.data_ptr(), cute.make_layout(RING * RING_W))

        if warp_idx == LOAD_WARP:
            cpasync.prefetch_descriptor(q_atom)
            cpasync.prefetch_descriptor(do_atom)
            cpasync.prefetch_descriptor(xq_atom)
            cpasync.prefetch_descriptor(xdo_atom)
        if warp_idx == KV_WARP:
            cpasync.prefetch_descriptor(k_atom)
            cpasync.prefetch_descriptor(v_atom)
            cpasync.prefetch_descriptor(cb_atom)
        if warp_idx == REDUCE_WARPS[0]:
            cpasync.prefetch_descriptor(dq_atom)
            if cutlass.const_expr(EPI_TMA):
                cpasync.prefetch_descriptor(dk_st_atom)
                cpasync.prefetch_descriptor(dv_st_atom)

        # ---------------- pipelines
        q_bytes = cute.size_in_bytes(bf16, cute.select(q_layout, mode=[0, 1, 2])) + XBLK_BYTES
        do_bytes = cute.size_in_bytes(bf16, cute.select(do_layout, mode=[0, 1, 2])) + XBLK_BYTES
        kv_buf_bytes = 2 * cute.size_in_bytes(bf16, cute.select(kv_layout, mode=[0, 1, 2]))   # one K or V tile (2 d-blocks)
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
        two_issuers = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=2)   # releases by both MMA-issuing warps
        load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=32)
        compute_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_COMPUTE_THREADS)
        wg0_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_WG_THREADS)
        wg1_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_WG_THREADS)
        reduce_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_REDUCE_THREADS)

        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_bars.data_ptr(), num_stages=Q_STAGES, producer_group=one_thread,
            consumer_group=two_issuers, tx_count=q_bytes, defer_sync=True).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_bars.data_ptr(), num_stages=DO_STAGES, producer_group=one_thread,
            consumer_group=two_issuers, tx_count=do_bytes, defer_sync=True).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_bars.data_ptr(), num_stages=1, producer_group=one_thread,
            consumer_group=wg0_threads, defer_sync=True).make_participants()
        dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dp_bars.data_ptr(), num_stages=1, producer_group=one_thread,
            consumer_group=wg1_threads, defer_sync=True).make_participants()
        dq_prod, dq_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dq_bars.data_ptr(), num_stages=1, producer_group=one_thread,
            consumer_group=reduce_threads, defer_sync=True).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_bars.data_ptr(), num_stages=2, producer_group=wg0_threads,
            consumer_group=one_thread, defer_sync=True).make_participants()
        p2_prod, p2_cons = pipeline.PipelineAsync.create(
            barrier_storage=storage.p2_bars.data_ptr(), num_stages=2, producer_group=wg0_threads,
            consumer_group=wg1_threads, defer_sync=True).make_participants()
        ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ds_bars.data_ptr(), num_stages=2, producer_group=wg1_threads,
            consumer_group=one_thread, defer_sync=True).make_participants()
        # second issuer's views of the same pipelines (same barriers, own state)
        q_prod2, q_cons2 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_bars.data_ptr(), num_stages=Q_STAGES, producer_group=one_thread,
            consumer_group=two_issuers, tx_count=q_bytes, defer_sync=True).make_participants()
        do_prod2, do_cons2 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_bars.data_ptr(), num_stages=DO_STAGES, producer_group=one_thread,
            consumer_group=two_issuers, tx_count=do_bytes, defer_sync=True).make_participants()
        dkdv_prod, dkdv_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dkdv_bars.data_ptr(), num_stages=2, producer_group=one_thread,
            consumer_group=wg1_threads if EPI_WG1 else reduce_threads, defer_sync=True).make_participants()
        kv_full = storage.kv_full.data_ptr()
        kv_empty_k = storage.kv_empty_k.data_ptr()
        kv_empty_v = storage.kv_empty_v.data_ptr()
        cb_bar = storage.cb_bar.data_ptr()
        tile_bars = storage.tile_bars.data_ptr()
        if tidx == 0:
            for r in cutlass.range_constexpr(KV_BUFS):
                cute.arch.mbarrier_init(kv_full + r, 1)
                cute.arch.mbarrier_init(kv_empty_k + r, 2)
                cute.arch.mbarrier_init(kv_empty_v + r, 1)
            cute.arch.mbarrier_init(cb_bar, 1)
            for r in cutlass.range_constexpr(RING):
                cute.arch.mbarrier_init(tile_bars + r, 1)
            cvt_ctr = cute.make_tensor(storage.cvt_ctr.data_ptr(), cute.make_layout(4))
            for r in cutlass.range_constexpr(4):
                cvt_ctr[r] = 0

        cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(
            storage.tmem_holding.data_ptr(),
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=BAR_TMEM_ALLOC,
                                                       num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))),
            allocator_warp_id=MMA_WARP)
        cute.arch.barrier()  # all 512 threads: barriers initialized

        # ---------------- item-independent fragments / partitions
        thr_s = mma_s.get_slice(0)
        tSrK = mma_s.make_fragment_A(sKV)        # (MMA, M, K=4, 4)
        tDVrP_layout = mma_pdo.make_fragment_A(p_layout.outer).layout  # (MMA, M, K=8, 1)
        tS_fake = mma_s.make_fragment_C(mma_s.partition_shape_C((TILE, HALF)))
        tDV_fake = mma_pdo.make_fragment_C(mma_pdo.partition_shape_C((TILE, HEAD_DIM)))
        tDK_fake = mma_dsq.make_fragment_C(mma_dsq.partition_shape_C((TILE, HEAD_DIM)))
        tDQ_fake = mma_dq.make_fragment_C(mma_dq.partition_shape_C((TILE, HALF)))
        scale_log2 = sm_scale * LOG2E
        n_final = 32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))

        # ============================================================ LOAD WARP: item fetch + Q / dO half-slots
        if warp_idx == LOAD_WARP:
            _set_regs(REGS_WG3)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            lane = tidx % 32
            row_bytes = cutlass.Int64(mQ_raw.stride[0]) * 2
            n = cutlass.Int32(0)
            active = cutlass.Boolean(True)
            while active:
                slot = n % RING
                # claim the next work item (one lane) and publish it through the ring
                with cute.arch.elect_one():
                    item = cute.arch.atomic_add(counter.iterator, cutlass.Int32(1))
                    item_c = min(item, n_items - 1)
                    if cutlass.const_expr(TABLE_FULL):
                        entry = tile_table[item_c]
                        head_c = (entry >> 24) & 0xFF
                        tile_e = (entry & 0xFFFFFF) | (entry >> 31)      # padding entries (-1) stay negative
                    else:
                        tile_e = tile_table[item_c // num_heads]
                        head_c = item_c % num_heads
                    tile = cutlass.Int32(-1)
                    if item < n_items:
                        tile = tile_e
                    tile_c = max(tile, 0)
                    seq = (tile_c >> 16) & 0xFF
                    ring[slot * RING_W + 0] = tile
                    ring[slot * RING_W + 1] = head_c
                    ring[slot * RING_W + 2] = cu_seqlens[seq]
                    ring[slot * RING_W + 3] = cu_seqlens[seq + 1]
                    ring[slot * RING_W + 4] = seq_acc[seq]
                    cute.arch.mbarrier_arrive(tile_bars + slot)
                cute.arch.sync_warp()
                tile = ring[slot * RING_W + 0]
                if tile < 0:
                    active = False
                else:
                    head = ring[slot * RING_W + 1]
                    seq_start = ring[slot * RING_W + 2]
                    seqlen = ring[slot * RING_W + 3] - seq_start
                    jblk = tile & 0xFFFF
                    n_q_blk = (seqlen + TILE - 1) // TILE
                    n_iters = n_q_blk - jblk
                    mQ_h = cute.domain_offset((seq_start, 0), mQ[(None, None, head)])
                    mDO_h = cute.domain_offset((seq_start, 0), mDO[(None, None, head)])
                    gQ = cute.local_tile(mQ_h, tiler_s, (0, None, 0), proj=(None, 1, 1))       # (64, D, nQhalf)
                    gDO = cute.local_tile(mDO_h, tiler_s, (0, None, 0), proj=(None, 1, 1))
                    tQgQ = thr_s.partition_B(gQ)
                    tDOgDO = thr_s.partition_B(gDO)
                    tQsQ, tQgQ_tma = cpasync.tma_partition(q_atom, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tQgQ, 0, 3))
                    tDOsDO, tDOgDO_tma = cpasync.tma_partition(do_atom, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(tDOgDO, 0, 3))
                    mXQ_h = cute.domain_offset((0, seq_start), mXQ[(None, None, head)])          # (8, T)
                    mXDO_h = cute.domain_offset((0, seq_start), mXDO[(None, None, head)])
                    gXQ = cute.local_tile(mXQ_h, (8, HALF), (0, None))                            # (8, 64, nQhalf)
                    gXDO = cute.local_tile(mXDO_h, (8, HALF), (0, None))
                    tXQs, tXQg = cpasync.tma_partition(xq_atom, 0, cute.make_layout(1), cute.group_modes(sQx, 0, 2), cute.group_modes(gXQ, 0, 2))
                    tXDOs, tXDOg = cpasync.tma_partition(xdo_atom, 0, cute.make_layout(1), cute.group_modes(sDOx, 0, 2), cute.group_modes(gXDO, 0, 2))
                    if cutlass.const_expr(QDO_PF > 0):
                        q_base = cutlass.Int64(cute.domain_offset((seq_start, 0, head), mQ_raw).iterator.toint())
                        do_base = cutlass.Int64(cute.domain_offset((seq_start, 0, head), mDO_raw).iterator.toint())
                    for it in cutlass.range(n_iters, unroll=1):
                        iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                        if cutlass.const_expr(EXP_SAMEQ):
                            iq = jblk
                        if cutlass.const_expr(QDO_PF > 0):
                            # warm L2 with the Q / dO rows of the block QDO_PF iterations ahead (4 + 4 lines per lane)
                            iq_pf = (iq - QDO_PF) if Q_REVERSE else (iq + QDO_PF)
                            if (iq_pf >= jblk) and (iq_pf < n_q_blk):
                                for j in cutlass.range_constexpr(4):
                                    row = iq_pf * TILE + lane + 32 * j
                                    if row < seqlen:
                                        l2_prefetch_line(q_base + row * row_bytes)
                                        l2_prefetch_line(q_base + row * row_bytes + 128)
                                        l2_prefetch_line(do_base + row * row_bytes)
                                        l2_prefetch_line(do_base + row * row_bytes + 128)
                        for hx in cutlass.range_constexpr(2):
                            _rpush("ld_wait_q_empty")
                            qh = q_prod.acquire_and_advance()
                            _rpop()
                            cute.copy(q_atom, tQgQ_tma[(None, 2 * iq + hx)], tQsQ[(None, qh.index)], tma_bar_ptr=qh.barrier)
                            cute.copy(xq_atom, tXQg[(None, 2 * iq + hx)], tXQs[(None, qh.index)], tma_bar_ptr=qh.barrier)
                        for hx in cutlass.range_constexpr(2):
                            _rpush("ld_wait_do_empty")
                            dh = do_prod.acquire_and_advance()
                            _rpop()
                            cute.copy(do_atom, tDOgDO_tma[(None, 2 * iq + hx)], tDOsDO[(None, dh.index)], tma_bar_ptr=dh.barrier)
                            cute.copy(xdo_atom, tXDOg[(None, 2 * iq + hx)], tXDOs[(None, dh.index)], tma_bar_ptr=dh.barrier)
                n += 1
            q_prod.tail()
            do_prod.tail()

        # ============================================================ KV WARP: K / V buffers (alternating roles)
        elif warp_idx == KV_WARP:
            _set_regs(REGS_WG3)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            lane = tidx % 32
            gCB = cute.local_tile(mCB, (8, 192), (0, 0))
            tCBs, tCBg = cpasync.tma_partition(cb_atom, 0, cute.make_layout(1), cute.group_modes(sCB, 0, 2), cute.group_modes(gCB, 0, 2))
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(cb_bar, 3072)
            cute.copy(cb_atom, tCBg, tCBs, tma_bar_ptr=cb_bar)
            cvt_pop = cutlass.Int32(0)
            cvt_push_addr = cutlass.Int32(storage.cvt_ctr.data_ptr().toint() & 0x3FFFF)
            cvt_pop_addr = cvt_push_addr + 4
            cvt_done_addr = cvt_push_addr + 8
            cvt_q_base = cutlass.Int64(cvt_q.iterator.toint()) + cutlass.Int64(bidx) * (CVTQ * CVTQ_W * 4)
            acc_base = cutlass.Int64(mDQ_raw.iterator.toint())
            cnt_base = cutlass.Int64(blk_cnt.iterator.toint())
            stats_base = cutlass.Int64(blk_cnt.iterator.toint()) + cutlass.Int64(num_heads * nblk_acc + bidx * 8) * 4
            dqo_base = cutlass.Int64(mDQout.iterator.toint())
            dqo_rowb = cutlass.Int64(mDQout.stride[0]) * 2
            rIn0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            rIn1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            rIn2 = cute.make_rmem_tensor((4,), cutlass.Int32)
            rIn3 = cute.make_rmem_tensor((4,), cutlass.Int32)
            rInH0 = cute.make_tensor(cute.recast_ptr(rIn0.iterator, dtype=cutlass.Float16), cute.make_layout(8))
            rInH1 = cute.make_tensor(cute.recast_ptr(rIn1.iterator, dtype=cutlass.Float16), cute.make_layout(8))
            rInH2 = cute.make_tensor(cute.recast_ptr(rIn2.iterator, dtype=cutlass.Float16), cute.make_layout(8))
            rInH3 = cute.make_tensor(cute.recast_ptr(rIn3.iterator, dtype=cutlass.Float16), cute.make_layout(8))
            rOutB = cute.make_rmem_tensor((4,), bf16)
            rOutW = cute.make_tensor(cute.recast_ptr(rOutB.iterator, dtype=cutlass.Int32), cute.make_layout(2))
            cst = cute.make_rmem_tensor((6,), cutlass.Int32)       # [active, chunk, h, blk, t0, nvalid]
            for r in cutlass.range_constexpr(6):
                cst[r] = 0

            n = cutlass.Int32(0)
            active = cutlass.Boolean(True)
            while active:
                slot = n % RING
                if cutlass.const_expr(CVT_INK):
                    # the next item is claimed by the loader only near the end of the current one: poll, converting meanwhile
                    ring_bar_addr = cutlass.Int32((tile_bars + slot).toint() & 0x3FFFF)
                    wd1 = cutlass.Int32(0)
                    while ptx_shfl0(ptx_mbar_try_wait(ring_bar_addr, (n // RING) & 1)) == 0:
                        if cutlass.const_expr(CVT_WD):
                            wd1 += 1
                            if wd1 == 4000000:
                                cute.printf("WD ring-poll: cta %d n %d pop %d push %d cst0 %d chunk %d\n", bidx, n, cvt_pop, ptx_shared_ld(cvt_push_addr), cst[0], cst[1])
                                ptx_trap()
                        if cst[0] != 0:
                            cvt_chunk(cst, acc_base, dqo_base, dqo_rowb, nblk_acc, lane, sm_scale, rIn0, rIn1, rIn2, rIn3, rInH0, rInH1, rInH2, rInH3, rOutB, rOutW, stats_base)
                        elif cvt_pop < ptx_shfl0(ptx_shared_ld_acquire(cvt_push_addr)):
                            if cvt_ready(cst, cvt_pop, cvt_q_base, cnt_base, nblk_acc) != 0:
                                cvt_pop += 1
                                ptx_shared_st(cvt_pop_addr, cvt_pop)
                            else:
                                ptx_nanosleep(128)
                        else:
                            if cutlass.const_expr(CVT_STATS):
                                if lane == 0:
                                    ptx_red_add(stats_base + 12, 1)
                            ptx_nanosleep(128)
                else:
                    cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                tile = ring[slot * RING_W + 0]
                if tile < 0:
                    active = False
                    if cutlass.const_expr(CVT_INK):
                        # drain: jobs may still be pushed until all four reduce warps are done
                        draining = cutlass.Boolean(True)
                        wd4 = cutlass.Int32(0)
                        while draining:
                            if cutlass.const_expr(CVT_WD):
                                wd4 += 1
                                if wd4 == 4000000:
                                    cute.printf("WD drain: cta %d n %d pop %d push %d done %d cst0 %d chunk %d\n", bidx, n, cvt_pop, ptx_shared_ld(cvt_push_addr), ptx_shared_ld(cvt_done_addr), cst[0], cst[1])
                                    ptx_trap()
                            if cst[0] != 0:
                                cvt_chunk(cst, acc_base, dqo_base, dqo_rowb, nblk_acc, lane, sm_scale, rIn0, rIn1, rIn2, rIn3, rInH0, rInH1, rInH2, rInH3, rOutB, rOutW, stats_base)
                            elif cvt_pop < ptx_shfl0(ptx_shared_ld_acquire(cvt_push_addr)):
                                if cvt_ready(cst, cvt_pop, cvt_q_base, cnt_base, nblk_acc) != 0:
                                    cvt_pop += 1
                                    ptx_shared_st(cvt_pop_addr, cvt_pop)
                                else:
                                    ptx_nanosleep(256)
                            elif ptx_shfl0(ptx_shared_ld_acquire(cvt_done_addr)) == len(REDUCE_WARPS):
                                draining = False
                            else:
                                ptx_nanosleep(256)
                else:
                    head = ring[slot * RING_W + 1]
                    seq_start = ring[slot * RING_W + 2]
                    jblk = tile & 0xFFFF
                    kv0 = jblk * TILE
                    kb, vb, _kfp, _vfp, k_empty_par, v_empty_par = kv_sched(n)
                    mK_h = cute.domain_offset((seq_start + kv0, 0), mK[(None, None, head)])
                    mV_h = cute.domain_offset((seq_start + kv0, 0), mV[(None, None, head)])
                    gK = cute.local_tile(mK_h, tiler_kv, (0, 0, None), proj=(1, None, 1))      # (TILE, 64, 2 d-blocks)
                    gV = cute.local_tile(mV_h, tiler_kv, (0, 0, None), proj=(1, None, 1))
                    tKgK = thr_s.partition_A(gK)
                    tVgV = thr_s.partition_A(gV)
                    tKsK, tKgK_tma = cpasync.tma_partition(k_atom, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 3), cute.group_modes(tKgK, 0, 3))
                    tVsV, tVgV_tma = cpasync.tma_partition(v_atom, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 3), cute.group_modes(tVgV, 0, 3))
                    if cutlass.const_expr(KV_PREFETCH):
                        # warm L2 with the 128 rows (2 x 128 B each) of the K and V tiles while the buffers are still busy
                        k_base = cutlass.Int64(cute.domain_offset((seq_start + kv0, 0, head), mK_raw).iterator.toint())
                        v_base = cutlass.Int64(cute.domain_offset((seq_start + kv0, 0, head), mV_raw).iterator.toint())
                        row_bytes = cutlass.Int64(mK_raw.stride[0]) * 2
                        for j in cutlass.range_constexpr(4):
                            row = lane + 32 * j
                            l2_prefetch_line(k_base + row * row_bytes)
                            l2_prefetch_line(k_base + row * row_bytes + 128)
                            l2_prefetch_line(v_base + row * row_bytes)
                            l2_prefetch_line(v_base + row * row_bytes + 128)
                    # K -> buffer kb: its previous use was a V (released by warp A after that item's last dP MMA)
                    role_phase = k_empty_par
                    _rpush("kv_wait_k_empty")
                    if cutlass.const_expr(CVT_INK):
                        # poll the buffer barrier; run queued conversion jobs while it is busy
                        kv_bar_addr = cutlass.Int32((kv_empty_v + kb).toint() & 0x3FFFF)
                        wd2 = cutlass.Int32(0)
                        while ptx_shfl0(ptx_mbar_try_wait(kv_bar_addr, role_phase)) == 0:
                            if cutlass.const_expr(CVT_WD):
                                wd2 += 1
                                if wd2 == 4000000:
                                    cute.printf("WD kv-k-poll: cta %d n %d pop %d push %d cst0 %d chunk %d\n", bidx, n, cvt_pop, ptx_shared_ld(cvt_push_addr), cst[0], cst[1])
                                    ptx_trap()
                            if cst[0] != 0:
                                cvt_chunk(cst, acc_base, dqo_base, dqo_rowb, nblk_acc, lane, sm_scale, rIn0, rIn1, rIn2, rIn3, rInH0, rInH1, rInH2, rInH3, rOutB, rOutW, stats_base)
                            elif cvt_pop < ptx_shfl0(ptx_shared_ld_acquire(cvt_push_addr)):
                                if cvt_ready(cst, cvt_pop, cvt_q_base, cnt_base, nblk_acc) != 0:
                                    cvt_pop += 1
                                    ptx_shared_st(cvt_pop_addr, cvt_pop)
                                else:
                                    ptx_nanosleep(128)
                            else:
                                ptx_nanosleep(128)
                    else:
                        cute.arch.mbarrier_wait(kv_empty_v + kb, role_phase)
                    _rpop()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(kv_full + kb, kv_buf_bytes)
                    for db in cutlass.range_constexpr(2):
                        cute.copy(k_atom, tKgK_tma[(None, db)], tKsK[(None, 2 * kb + db)], tma_bar_ptr=kv_full + kb)
                    # V -> buffer vb: its previous use was a K (released after the last S of warp A and the last dQ^T of warp B)
                    role_phase = v_empty_par
                    _rpush("kv_wait_v_empty")
                    if cutlass.const_expr(CVT_INK):
                        kv_bar_addr = cutlass.Int32((kv_empty_k + vb).toint() & 0x3FFFF)
                        wd3 = cutlass.Int32(0)
                        while ptx_shfl0(ptx_mbar_try_wait(kv_bar_addr, role_phase)) == 0:
                            if cutlass.const_expr(CVT_WD):
                                wd3 += 1
                                if wd3 == 4000000:
                                    cute.printf("WD kv-v-poll: cta %d n %d pop %d push %d cst0 %d chunk %d\n", bidx, n, cvt_pop, ptx_shared_ld(cvt_push_addr), cst[0], cst[1])
                                    ptx_trap()
                            if cst[0] != 0:
                                cvt_chunk(cst, acc_base, dqo_base, dqo_rowb, nblk_acc, lane, sm_scale, rIn0, rIn1, rIn2, rIn3, rInH0, rInH1, rInH2, rInH3, rOutB, rOutW, stats_base)
                            elif cvt_pop < ptx_shfl0(ptx_shared_ld_acquire(cvt_push_addr)):
                                if cvt_ready(cst, cvt_pop, cvt_q_base, cnt_base, nblk_acc) != 0:
                                    cvt_pop += 1
                                    ptx_shared_st(cvt_pop_addr, cvt_pop)
                                else:
                                    ptx_nanosleep(128)
                            else:
                                ptx_nanosleep(128)
                    elif cutlass.const_expr(not EXP_KV_EARLY):
                        cute.arch.mbarrier_wait(kv_empty_k + vb, role_phase)
                    _rpop()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(kv_full + vb, kv_buf_bytes)
                    for db in cutlass.range_constexpr(2):
                        cute.copy(v_atom, tVgV_tma[(None, db)], tVsV[(None, 2 * vb + db)], tma_bar_ptr=kv_full + vb)
                n += 1

        # ============================================================ MMA WARP A: S^T, dP^T
        elif warp_idx == MMA_WARP:
            _set_regs(REGS_WG3)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            _rpush("mma_alloc")
            tmem.allocate(512)
            tmem.wait_for_alloc()
            _rpop()
            tmem_ptr = tmem.retrieve_ptr(f32)
            tS = cute.make_tensor(tmem_ptr + TMEM_S, tS_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + TMEM_DP, tS_fake.layout)
            sKV_all = cute.group_modes(sKV, 2, 4)        # (MMA, M, (K=4, 4 blocks)) -> 16 k-blocks; buffer b = k-blocks 8b..8b+7
            K8 = list(range(NK8))
            cute.arch.mbarrier_wait(cb_bar, 0)
            n = cutlass.Int32(0)
            active = cutlass.Boolean(True)
            while active:
                slot = n % RING
                _rpush("mma_wait_item")
                cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                _rpop()
                tile = ring[slot * RING_W + 0]
                if tile < 0:
                    active = False
                else:
                    seq_start = ring[slot * RING_W + 2]
                    seqlen = ring[slot * RING_W + 3] - seq_start
                    jblk = tile & 0xFFFF
                    n_q_blk = (seqlen + TILE - 1) // TILE
                    n_iters = n_q_blk - jblk
                    kb, vb, k_full_par, v_full_par, _kep, _vep = kv_sched(n)
                    k_off = kb * KV_BLK16
                    v_off = vb * KV_BLK16
                    _rpush("mma_wait_kv")
                    cute.arch.mbarrier_wait(kv_full + kb, k_full_par)
                    _rpop()
                    # ---- prologue: S_a(0), dP_a(0), S_b(0), dP_b(0)
                    _rpush("mma_pro")
                    if cutlass.const_expr(PRO_SB_FIRST):
                        # S_a, S_b (K only; S_b waits for WG0 to drain S_a), then the V wait, dP_a, dP_b
                        for hx in cutlass.range_constexpr(2):
                            qh = q_cons.wait_and_advance()
                            sh = s_prod.acquire_and_advance()
                            gemm_ptx(self.op_s, tS, sKV_all, sQ[(None, None, None, qh.index)], None, 0, K8, K8, k_off)
                            mma_tail_ptx(self.op_s, tS, ones_lo, ((qx_addr + qh.index * XBLK_BYTES) >> 4) | (((zero_addr - qx_addr - qh.index * XBLK_BYTES) >> 4) << 16))
                            sh.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: S done (1 of 2)
                        _rpush("mma_wait_v")
                        cute.arch.mbarrier_wait(kv_full + vb, v_full_par)
                        _rpop()
                        for hx in cutlass.range_constexpr(2):
                            dh = do_cons.wait_and_advance()
                            dph = dp_prod.acquire_and_advance()
                            gemm_ptx(self.op_s, tDP, sKV_all, sDO[(None, None, None, dh.index)], None, 0, K8, K8, v_off)
                            mma_tail_ptx(self.op_s, tDP, ones_lo, ((dox_addr + dh.index * XBLK_BYTES) >> 4) | (((zero_addr - dox_addr - dh.index * XBLK_BYTES) >> 4) << 16))
                            dph.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)     # dO slot: dP done (1 of 2)
                    else:
                        for hx in cutlass.range_constexpr(2):
                            qh = q_cons.wait_and_advance()
                            sh = s_prod.acquire_and_advance()
                            gemm_ptx(self.op_s, tS, sKV_all, sQ[(None, None, None, qh.index)], None, 0, K8, K8, k_off)
                            mma_tail_ptx(self.op_s, tS, ones_lo, ((qx_addr + qh.index * XBLK_BYTES) >> 4) | (((zero_addr - qx_addr - qh.index * XBLK_BYTES) >> 4) << 16))
                            sh.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: S done (1 of 2)
                            if hx == 0:
                                _rpush("mma_wait_v")
                                cute.arch.mbarrier_wait(kv_full + vb, v_full_par)
                                _rpop()
                            dh = do_cons.wait_and_advance()
                            dph = dp_prod.acquire_and_advance()
                            gemm_ptx(self.op_s, tDP, sKV_all, sDO[(None, None, None, dh.index)], None, 0, K8, K8, v_off)
                            mma_tail_ptx(self.op_s, tDP, ones_lo, ((dox_addr + dh.index * XBLK_BYTES) >> 4) | (((zero_addr - dox_addr - dh.index * XBLK_BYTES) >> 4) << 16))
                            dph.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)     # dO slot: dP done (1 of 2)
                    _rpop()
                    # ---- main loop. Per half x of tile it: [S_x'] [dP_x']  (' = next tile)
                    for it in cutlass.range(n_iters, unroll=1):
                        has_next = it + 1 < n_iters
                        for hx in cutlass.range_constexpr(2):
                            if has_next:
                                _rpush("mma_wait_q")
                                qh = q_cons.wait_and_advance()
                                _rpop()
                                _rpush("mma_wait_s_empty")
                                sh = s_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mma_S")
                                gemm_ptx(self.op_s, tS, sKV_all, sQ[(None, None, None, qh.index)], None, 0, K8, K8, k_off)
                                mma_tail_ptx(self.op_s, tS, ones_lo, ((qx_addr + qh.index * XBLK_BYTES) >> 4) | (((zero_addr - qx_addr - qh.index * XBLK_BYTES) >> 4) << 16))
                                sh.commit()
                                with cute.arch.elect_one():
                                    tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)
                                _rpop()
                            if has_next:
                                _rpush("mma_wait_do")
                                dh = do_cons.wait_and_advance()
                                _rpop()
                                _rpush("mma_wait_dp_empty")
                                dph = dp_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mma_dP")
                                gemm_ptx(self.op_s, tDP, sKV_all, sDO[(None, None, None, dh.index)], None, 0, K8, K8, v_off)
                                mma_tail_ptx(self.op_s, tDP, ones_lo, ((dox_addr + dh.index * XBLK_BYTES) >> 4) | (((zero_addr - dox_addr - dh.index * XBLK_BYTES) >> 4) << 16))
                                dph.commit()
                                with cute.arch.elect_one():
                                    tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)
                                _rpop()
                    # release both K/V buffers (this warp's share): all S / dP MMAs of the item retired
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty_k + kb)
                        tcgen05.commit(kv_empty_v + vb)
                n += 1
            # wait for all TMEM readers then free
            _rpush("mma_fin")
            pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=n_final).arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            _rpop()

        # ============================================================ MMA WARP B: dV, dK, dQ^T
        elif warp_idx == MMA2_WARP:
            _set_regs(REGS_WG3)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(f32)
            tDV = cute.make_tensor(tmem_ptr + TMEM_DV, tDV_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
            tDVrP = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_P, dtype=bf16), tDVrP_layout)
            tDQ = cute.make_tensor(tmem_ptr + TMEM_DQ, tDQ_fake.layout)
            sDS0 = sDS[(None, None, None, 0)]
            sKT0 = sKT[(None, None, None, 0)]
            tmem_p = tDVrP.iterator.toint()
            K4 = list(range(NK4))
            K8 = list(range(NK8))
            n = cutlass.Int32(0)
            active = cutlass.Boolean(True)
            while active:
                slot = n % RING
                _rpush("mmb_wait_item")
                cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                _rpop()
                tile = ring[slot * RING_W + 0]
                if tile < 0:
                    active = False
                else:
                    seq_start = ring[slot * RING_W + 2]
                    seqlen = ring[slot * RING_W + 3] - seq_start
                    jblk = tile & 0xFFFF
                    n_q_blk = (seqlen + TILE - 1) // TILE
                    n_iters = n_q_blk - jblk
                    kb, vb, k_full_par, _vfp, _kep, _vep = kv_sched(n)
                    k_off = kb * KV_BLK16
                    # dK / dV TMEM regions: wait until the previous item's epilogue has read them
                    _rpush("mmb_wait_dkdv_free")
                    kh = dkdv_prod.acquire_and_advance()
                    kh2 = dkdv_prod.acquire_and_advance()
                    _rpop()
                    _rpush("mmb_wait_kv")
                    cute.arch.mbarrier_wait(kv_full + kb, k_full_par)
                    _rpop()
                    # ---- first iteration: both dV halves first (P arrives early; dS waits for V), then dK/dQ^T per half
                    for hx in cutlass.range_constexpr(2):
                        _rpush("mmb_wait_p")
                        ph = p_cons.wait_and_advance()
                        _rpop()
                        dh = do_cons2.wait_and_advance()
                        _rpush("mmb_dV")
                        gemm_ptx(self.op_pdo, tDV, None, sDOT[(None, None, None, dh.index)], tmem_p + hx * 32,
                                 cutlass.Int32(hx != 0), K4, K4, 0)
                        ph.release()
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)     # dO slot: dV done (2 of 2)
                        if cutlass.const_expr(hx == 1):
                            if n_iters == 1:
                                kh.commit()                                                       # dV ready
                        _rpop()
                    for hx in cutlass.range_constexpr(2):
                        _rpush("mmb_wait_ds")
                        dsh = ds_cons.wait_and_advance()
                        _rpop()
                        qh = q_cons2.wait_and_advance()
                        _rpush("mmb_dK")
                        if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 1 and hx == 1):
                            # dS_1 has arrived: issue dQ_0 before dK_1 so that dK_1 covers part of the dQ_0 drain
                            _rpush("mmb_wait_dq_empty")
                            dqh = dq_prod.acquire_and_advance()
                            _rpop()
                            _rpush("mmb_dQ")
                            gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb)], None, 0, K8, K8, 0)
                            dqh.commit()
                            _rpop()
                        gemm_ptx(self.op_dsq, tDK, sDS0, sQT[(None, None, None, qh.index)], None,
                                 cutlass.Int32(hx != 0), [hx * 4 + kk for kk in K4], K4, 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: dK done (2 of 2)
                        if cutlass.const_expr(hx == 1):
                            if n_iters == 1:
                                kh2.commit()                                                      # dK ready
                        _rpop()
                        if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 1 and hx == 1):
                            _rpush("mmb_wait_dq_empty")
                            dqh = dq_prod.acquire_and_advance()
                            _rpop()
                            _rpush("mmb_dQ")
                            gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb + 1)], None, 0, K8, K8, 0)
                            dqh.commit()
                            _rpop()
                        if cutlass.const_expr(not DQ_DIRECT):
                            _rpush("mmb_wait_dq_empty")
                            dqh = dq_prod.acquire_and_advance()
                            _rpop()
                            _rpush("mmb_dQ")
                            gemm_ptx(self.op_dq, tDQ, sKT0, sDST[(None, None, None, hx)], None, 0, K8, K8, k_off)
                            dqh.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.ds_bars.data_ptr() + 2 + hx)                  # dS stage x free (dK_x, dQ_x done)
                            _rpop()
                    if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 0):
                        # dQ[:, d-half] = dS (128 q x 128 kv, both dS^T stages as MN-major A) . K[:, d-half] (MN-major B)
                        for dh in cutlass.range_constexpr(2):
                            _rpush("mmb_wait_dq_empty")
                            dqh = dq_prod.acquire_and_advance()
                            _rpop()
                            _rpush("mmb_dQ")
                            gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb + dh)], None, 0, K8, K8, 0)
                            dqh.commit()
                            _rpop()
                    if cutlass.const_expr(DQ_DIRECT):
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.ds_bars.data_ptr() + 2)                      # dS stages free (dK_0/1, dQ done)
                            tcgen05.commit(storage.ds_bars.data_ptr() + 3)
                    for it in cutlass.range(1, n_iters, unroll=1):
                        last = it == n_iters - 1
                        for hx in cutlass.range_constexpr(2):
                            _rpush("mmb_wait_p")
                            ph = p_cons.wait_and_advance()
                            _rpop()
                            dh = do_cons2.wait_and_advance()
                            _rpush("mmb_dV")
                            gemm_ptx(self.op_pdo, tDV, None, sDOT[(None, None, None, dh.index)], tmem_p + hx * 32,
                                     cutlass.Int32(1), K4, K4, 0)
                            ph.release()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)     # dO slot: dV done (2 of 2)
                            if cutlass.const_expr(hx == 1):
                                if last:
                                    kh.commit()                                                       # dV ready (epilogue may start)
                            _rpop()
                            _rpush("mmb_wait_ds")
                            dsh = ds_cons.wait_and_advance()
                            _rpop()
                            qh = q_cons2.wait_and_advance()
                            _rpush("mmb_dK")
                            if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 1 and hx == 1):
                                _rpush("mmb_wait_dq_empty")
                                dqh = dq_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mmb_dQ")
                                gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb)], None, 0, K8, K8, 0)
                                dqh.commit()
                                _rpop()
                            gemm_ptx(self.op_dsq, tDK, sDS0, sQT[(None, None, None, qh.index)], None,
                                     cutlass.Int32(1), [hx * 4 + kk for kk in K4], K4, 0)
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: dK done (2 of 2)
                            if cutlass.const_expr(hx == 1):
                                if last:
                                    kh2.commit()                                                      # dK ready
                            _rpop()
                            if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 1 and hx == 1):
                                _rpush("mmb_wait_dq_empty")
                                dqh = dq_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mmb_dQ")
                                gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb + 1)], None, 0, K8, K8, 0)
                                dqh.commit()
                                _rpop()
                            if cutlass.const_expr(not DQ_DIRECT):
                                _rpush("mmb_wait_dq_empty")
                                dqh = dq_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mmb_dQ")
                                gemm_ptx(self.op_dq, tDQ, sKT0, sDST[(None, None, None, hx)], None, 0, K8, K8, k_off)
                                dqh.commit()
                                with cute.arch.elect_one():
                                    tcgen05.commit(storage.ds_bars.data_ptr() + 2 + hx)                  # dS stage x free (dK_x, dQ_x done)
                                _rpop()
                        if cutlass.const_expr(DQ_DIRECT and DQ_ORDER == 0):
                            for dh in cutlass.range_constexpr(2):
                                _rpush("mmb_wait_dq_empty")
                                dqh = dq_prod.acquire_and_advance()
                                _rpop()
                                _rpush("mmb_dQ")
                                gemm_ptx(self.op_dq, tDQ, sDSA[(None, None, None, 0)], sKH[(None, None, None, 2 * kb + dh)], None, 0, K8, K8, 0)
                                dqh.commit()
                                _rpop()
                        if cutlass.const_expr(DQ_DIRECT):
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.ds_bars.data_ptr() + 2)
                                tcgen05.commit(storage.ds_bars.data_ptr() + 3)
                    # ---- item tail: K buffer free (dQ done); dV / dK were committed after their last MMAs
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty_k + kb)
                n += 1
            _rpush("mmb_fin")
            pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=n_final).arrive_and_wait()
            _rpop()

        # ============================================================ COMPUTE WARPS
        elif warp_idx >= COMPUTE_WARPS[0] and warp_idx <= COMPUTE_WARPS[-1]:
            _set_regs(REGS_COMPUTE)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(f32)
            tS = cute.make_tensor(tmem_ptr + TMEM_S, tS_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + TMEM_DP, tS_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + TMEM_DV, tDV_fake.layout)
            ctid = tidx - COMPUTE_WARPS[0] * 32          # 0..255
            wg = cute.arch.make_warp_uniform(ctid // 128)  # warpgroup 0 (P) / 1 (dS)
            dp_idx = ctid % 128                           # lane / row within tile

            # TMEM load atoms over a (TILE, 32-column) chunk of the S / dP half regions
            tS2d = tS[((None, None), 0, 0)]               # (TILE, 64)
            tDP2d = tDP[((None, None), 0, 0)]
            chunk_tiler = (TILE, 32)
            tS_chunks = cute.local_tile(tS2d, chunk_tiler, (0, None))      # (TILE, 32, 2)
            tDP_chunks = cute.local_tile(tDP2d, chunk_tiler, (0, None))
            ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), f32)
            tiled_ld = tcgen05.make_tmem_copy(ld_atom, tS_chunks[(None, None, 0)])
            thr_ld = tiled_ld.get_slice(dp_idx)
            cChunk = cute.make_identity_tensor(chunk_tiler)
            tTR_c = thr_ld.partition_D(cChunk)            # coords within a chunk ((32,1),1,1)
            tTR_tS = thr_ld.partition_S(tS_chunks)        # (((32,32),1),1,1,2)
            tTR_tDP = thr_ld.partition_S(tDP_chunks)
            frag_shape = tTR_c.shape
            # P region: packed bf16 pairs as f32 words: (TILE, 64) f32 view, 16-col chunks = 32 q columns
            tDV2d_full = tDV[((None, None), 0, 0)]       # (TILE, 128) f32 TMEM layout
            tP32 = cute.make_tensor(tmem_ptr + TMEM_P, cute.composition(tDV2d_full.layout, cute.make_layout((TILE, TILE // 2))))
            p_chunk_tiler = (TILE, 16)
            tP_chunks = cute.local_tile(tP32, p_chunk_tiler, (0, None))     # (TILE, 16, 4)
            st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(16)), f32)
            tiled_st = tcgen05.make_tmem_copy(st_atom, tP_chunks[(None, None, 0)])
            thr_st = tiled_st.get_slice(dp_idx)
            tRT_tP = thr_st.partition_D(tP_chunks)        # (..., 4)
            cPChunk = cute.make_identity_tensor(p_chunk_tiler)
            p_frag_shape = thr_st.partition_S(cPChunk).shape
            ldp_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), f32)
            tiled_ldp = tcgen05.make_tmem_copy(ldp_atom, tP_chunks[(None, None, 0)])
            thr_ldp = tiled_ldp.get_slice(dp_idx)
            tTR_tP = thr_ldp.partition_S(tP_chunks)       # (..., 4)
            # 2-D (kv, q) view of the dS^T smem buffer (K-major SW128 canonical layout, swizzle via iterator)
            sDS2d = cute.make_tensor(sDS.iterator, cute.make_layout((TILE, (16, 4, 2)), stride=(64, (1, 16, 8192))))
            row0 = tTR_c[0][0]                            # this thread's kv row within tile
            # register -> swizzled smem tiled copy for dS^T chunks (same thread/value tiling as the TMEM load)
            sDS_chunks = cute.local_tile(sDS2d, chunk_tiler, (0, None))        # (TILE, 32, 4) bf16
            r2s_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, bf16, f32, tiled_ld)
            tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_ld)
            thr_r2s = tiled_r2s.get_slice(dp_idx)
            tRS_sDS = thr_r2s.partition_D(sDS_chunks)                           # (R2S, M, N, 4)

            if wg == 0:
                # ---------------- WG0: P producer
                rS0 = cute.make_rmem_tensor(frag_shape, f32)
                rS1 = cute.make_rmem_tensor(frag_shape, f32)
                rPb0 = cute.make_rmem_tensor(frag_shape, bf16)
                rPb1 = cute.make_rmem_tensor(frag_shape, bf16)
                rP16_0 = cute.make_tensor(cute.recast_tensor(rPb0, f32).iterator, cute.make_layout(p_frag_shape))
                rP16_1 = cute.make_tensor(cute.recast_tensor(rPb1, f32).iterator, cute.make_layout(p_frag_shape))
                n = cutlass.Int32(0)
                active = cutlass.Boolean(True)
                while active:
                    slot = n % RING
                    cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                    tile = ring[slot * RING_W + 0]
                    if tile < 0:
                        active = False
                    else:
                        seq_start = ring[slot * RING_W + 2]
                        seqlen = ring[slot * RING_W + 3] - seq_start
                        jblk = tile & 0xFFFF
                        kv0 = jblk * TILE
                        n_q_blk = (seqlen + TILE - 1) // TILE
                        n_iters = n_q_blk - jblk
                        kv_row = kv0 + row0
                        kv_valid = kv_row < seqlen
                        kv_tail = (kv0 + TILE) > seqlen
                        # masked tiles: the first (last q tile of the sequence) and the last (diagonal) iteration, or all
                        # iterations for a kv-tail block; the interior loop has no data-dependent branches.
                        n_head = cutlass.Int32(1)
                        if kv_tail:
                            n_head = n_iters
                        n_end = max(n_head, n_iters - 1)
                        for it in cutlass.range(n_head, unroll=1):
                            iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                            qbase = iq * TILE
                            for hx in cutlass.range_constexpr(2):
                                _rpush("cmp_wait_s")
                                sh = s_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp0_ldS")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 0)], rS0)
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 1)], rS1)
                                cute.arch.fence_view_async_tmem_load()
                                sh.release()
                                _rpop()
                                _rpush("cmp0_exp")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    self.p_chunk(rS0, rPb0, hx * 2 + 0, True, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                    self.p_chunk(rS1, rPb1, hx * 2 + 1, True, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                _rpop()
                                _rpush("cmp_wait_p_empty")
                                ph = p_prod.acquire_and_advance()
                                p2h = p2_prod.acquire_and_advance()
                                _rpop()
                                _rpush("cmp0_stP")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_st, rP16_0, tRT_tP[(None, None, None, hx * 2 + 0)])
                                    cute.copy(tiled_st, rP16_1, tRT_tP[(None, None, None, hx * 2 + 1)])
                                cute.arch.fence_view_async_tmem_store()
                                ph.commit()
                                p2h.commit()
                                _rpop()
                        for it in cutlass.range(n_head, n_end, unroll=1):
                            iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                            qbase = iq * TILE
                            for hx in cutlass.range_constexpr(2):
                                _rpush("cmp_wait_s")
                                sh = s_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp0_ldS")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 0)], rS0)
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 1)], rS1)
                                cute.arch.fence_view_async_tmem_load()
                                sh.release()
                                _rpop()
                                _rpush("cmp0_exp")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    self.p_chunk(rS0, rPb0, hx * 2 + 0, False, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                    self.p_chunk(rS1, rPb1, hx * 2 + 1, False, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                _rpop()
                                _rpush("cmp_wait_p_empty")
                                ph = p_prod.acquire_and_advance()
                                p2h = p2_prod.acquire_and_advance()
                                _rpop()
                                _rpush("cmp0_stP")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_st, rP16_0, tRT_tP[(None, None, None, hx * 2 + 0)])
                                    cute.copy(tiled_st, rP16_1, tRT_tP[(None, None, None, hx * 2 + 1)])
                                cute.arch.fence_view_async_tmem_store()
                                ph.commit()
                                p2h.commit()
                                _rpop()
                        for it in cutlass.range(n_end, n_iters, unroll=1):
                            iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                            qbase = iq * TILE
                            for hx in cutlass.range_constexpr(2):
                                _rpush("cmp_wait_s")
                                sh = s_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp0_ldS")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 0)], rS0)
                                    cute.copy(tiled_ld, tTR_tS[(None, None, None, 1)], rS1)
                                cute.arch.fence_view_async_tmem_load()
                                sh.release()
                                _rpop()
                                _rpush("cmp0_exp")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    self.p_chunk(rS0, rPb0, hx * 2 + 0, True, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                    self.p_chunk(rS1, rPb1, hx * 2 + 1, True, qbase, kv_row, kv_valid, seqlen, scale_log2)
                                _rpop()
                                _rpush("cmp_wait_p_empty")
                                ph = p_prod.acquire_and_advance()
                                p2h = p2_prod.acquire_and_advance()
                                _rpop()
                                _rpush("cmp0_stP")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_st, rP16_0, tRT_tP[(None, None, None, hx * 2 + 0)])
                                    cute.copy(tiled_st, rP16_1, tRT_tP[(None, None, None, hx * 2 + 1)])
                                cute.arch.fence_view_async_tmem_store()
                                ph.commit()
                                p2h.commit()
                                _rpop()
                    n += 1
            else:
                # ---------------- WG1: dS producer
                rDP0 = cute.make_rmem_tensor(frag_shape, f32)
                rDP1 = cute.make_rmem_tensor(frag_shape, f32)
                rPb0 = cute.make_rmem_tensor(frag_shape, bf16)
                rPb1 = cute.make_rmem_tensor(frag_shape, bf16)
                rP16_0 = cute.make_tensor(cute.recast_tensor(rPb0, f32).iterator, cute.make_layout(p_frag_shape))
                rP16_1 = cute.make_tensor(cute.recast_tensor(rPb1, f32).iterator, cute.make_layout(p_frag_shape))
                rDSb0 = cute.make_rmem_tensor(frag_shape, bf16)
                rDSb1 = cute.make_rmem_tensor(frag_shape, bf16)
                tRS_rDS0 = tiled_r2s.retile(rDSb0)
                tRS_rDS1 = tiled_r2s.retile(rDSb1)
                if cutlass.const_expr(EPI_WG1):
                    # dK / dV epilogue views: (TILE kv lanes, 32-col chunks); this thread's TMEM lane dp_idx = kv row
                    tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
                    tDV_chunks_e = cute.local_tile(tDV[((None, None), 0, 0)], chunk_tiler, (0, None))     # (TILE, 32, 4)
                    tDK_chunks_e = cute.local_tile(tDK[((None, None), 0, 0)], chunk_tiler, (0, None))
                    tTR_tDVe = thr_ld.partition_S(tDV_chunks_e)
                    tTR_tDKe = thr_ld.partition_S(tDK_chunks_e)
                    vec_atom_e = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bf16, num_bits_per_copy=128)
                    vec_layout_e = cute.make_layout((8, 4), stride=(1, 8))      # 32 bf16 as 4 x 16B
                    rEb = cute.make_rmem_tensor((32,), bf16)
                n = cutlass.Int32(0)
                active = cutlass.Boolean(True)
                while active:
                    slot = n % RING
                    cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                    tile = ring[slot * RING_W + 0]
                    if tile < 0:
                        active = False
                    else:
                        head = ring[slot * RING_W + 1]
                        seq_start = ring[slot * RING_W + 2]
                        seqlen = ring[slot * RING_W + 3] - seq_start
                        jblk = tile & 0xFFFF
                        n_q_blk = (seqlen + TILE - 1) // TILE
                        n_iters = n_q_blk - jblk
                        kv_row_e = jblk * TILE + row0
                        kv_valid_e = kv_row_e < seqlen
                        kv_tok_e = seq_start + kv_row_e
                        for it in cutlass.range(n_iters, unroll=1):
                            for hx in cutlass.range_constexpr(2):
                                _rpush("cmp_wait_dp")
                                dph = dp_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp1_ldDP")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_ld, tTR_tDP[(None, None, None, 0)], rDP0)
                                    cute.copy(tiled_ld, tTR_tDP[(None, None, None, 1)], rDP1)
                                cute.arch.fence_view_async_tmem_load()
                                dph.release()
                                _rpop()
                                _rpush("cmp_wait_p2")
                                p2h = p2_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp1_ldP")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_ldp, tTR_tP[(None, None, None, hx * 2 + 0)], rP16_0)
                                    cute.copy(tiled_ldp, tTR_tP[(None, None, None, hx * 2 + 1)], rP16_1)
                                cute.arch.fence_view_async_tmem_load()
                                p2h.release()
                                _rpop()
                                _rpush("cmp1_math")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    for c in cutlass.range_constexpr(2):
                                        rDP = (rDP0, rDP1)[c]
                                        rPb = (rPb0, rPb1)[c]
                                        rDSb = (rDSb0, rDSb1)[c]
                                        for i in cutlass.range_constexpr(32):
                                            rDP[i] = rPb[i].to(f32) * rDP[i]
                                        rDSb.store(rDP.load().to(bf16))
                                _rpop()
                                _rpush("cmp_wait_ds_empty")
                                dsh = ds_prod.acquire_and_advance()
                                _rpop()
                                _rpush("cmp1_stDS")
                                if cutlass.const_expr(not EXP_NOCOMPUTE):
                                    cute.copy(tiled_r2s, tRS_rDS0, tRS_sDS[(None, None, None, hx * 2 + 0)])
                                    cute.copy(tiled_r2s, tRS_rDS1, tRS_sDS[(None, None, None, hx * 2 + 1)])
                                cute.arch.fence_view_async_shared()
                                dsh.commit()
                                _rpop()
                        if cutlass.const_expr(EPI_WG1):
                            # ---- item epilogue (this warpgroup idles here until the next item's V / dP arrive):
                            # dV (ready after the last dV GEMM) then dK (ready after the last dK GEMM) -> bf16 global rows
                            for acc_i in cutlass.range_constexpr(2):
                                _rpush("cmp1_epi_wait")
                                kh = dkdv_cons.wait_and_advance()
                                _rpop()
                                _rpush("cmp1_epi")
                                for pair in cutlass.range_constexpr(2):
                                    if cutlass.const_expr(acc_i == 0):
                                        cute.copy(tiled_ld, tTR_tDVe[(None, None, None, 2 * pair)], rDP0)
                                        cute.copy(tiled_ld, tTR_tDVe[(None, None, None, 2 * pair + 1)], rDP1)
                                    else:
                                        cute.copy(tiled_ld, tTR_tDKe[(None, None, None, 2 * pair)], rDP0)
                                        cute.copy(tiled_ld, tTR_tDKe[(None, None, None, 2 * pair + 1)], rDP1)
                                    cute.arch.fence_view_async_tmem_load()
                                    for c in cutlass.range_constexpr(2):
                                        chunk = 2 * pair + c
                                        rO = (rDP0, rDP1)[c]
                                        if cutlass.const_expr(acc_i == 0):
                                            rEb.store(rO.load().to(bf16))
                                        else:
                                            rEb.store((rO.load() * sm_scale).to(bf16))
                                        if cutlass.const_expr(not EXP_NOEPI):
                                            if kv_valid_e:
                                                rOv = cute.make_tensor(rEb.iterator, vec_layout_e)
                                                if cutlass.const_expr(acc_i == 0):
                                                    dstv = cute.make_tensor((mDV.iterator + mDV.layout((kv_tok_e, chunk * 32, head))).align(min_align=16), vec_layout_e)
                                                else:
                                                    dstv = cute.make_tensor((mDK.iterator + mDK.layout((kv_tok_e, chunk * 32, head))).align(min_align=16), vec_layout_e)
                                                cute.copy(vec_atom_e, rOv, dstv)
                                kh.release()
                                _rpop()
                    n += 1
            _rpush("cmp_fin")
            pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=n_final).arrive_and_wait()
            _rpop()

        # ============================================================ REDUCE WARPS (dQ^T halves) + dV/dK epilogue
        elif warp_idx <= REDUCE_WARPS[-1]:
            _set_regs(REGS_REDUCE)
            if cutlass.const_expr(IKET):
                cute.experimental.iket.range_end(setup_tok)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(f32)
            tDQ = cute.make_tensor(tmem_ptr + TMEM_DQ, tDQ_fake.layout)
            tDV = cute.make_tensor(tmem_ptr + TMEM_DV, tDV_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
            rtid = tidx                                  # 0..127
            tDQ2d = tDQ[((None, None), 0, 0)]            # (TILE d-lanes, 64 q-cols)
            dq_tiler = (TILE, 32)
            tDQ_chunks = cute.local_tile(tDQ2d, dq_tiler, (0, None))        # (TILE, 32, 2)
            ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), f32)
            tiled_ld = tcgen05.make_tmem_copy(ld_atom, tDQ_chunks[(None, None, 0)])
            thr_ld = tiled_ld.get_slice(rtid)
            tTR_tDQ = thr_ld.partition_S(tDQ_chunks)
            cDQ = cute.make_identity_tensor(dq_tiler)
            tTR_cDQ = thr_ld.partition_D(cDQ)
            rDQ0 = cute.make_rmem_tensor(tTR_cDQ.shape, f32)
            rDQ1 = cute.make_rmem_tensor(tTR_cDQ.shape, f32)
            # epilogue views: (TILE kv-lanes, 32-col chunks) of dV / dK
            tDV2d = tDV[((None, None), 0, 0)]
            tDK2d = tDK[((None, None), 0, 0)]
            tDV_chunks = cute.local_tile(tDV2d, dq_tiler, (0, None))        # (TILE, 32, 4)
            tDK_chunks = cute.local_tile(tDK2d, dq_tiler, (0, None))
            tTR_tDV = thr_ld.partition_S(tDV_chunks)
            tTR_tDK = thr_ld.partition_S(tDK_chunks)
            row0 = tTR_cDQ[0][0]                         # this thread's TMEM lane (= kv row in the epilogue)
            vec_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bf16, num_bits_per_copy=128)
            vec_layout = cute.make_layout((8, 4), stride=(1, 8))      # 32 bf16 as 4 x 16B
            # (128 d, DQ_QCOLS q) view of the staging buffer (same bytes as the 4 per-warp (32, DQ_QCOLS) boxes)
            sDQv = cute.make_tensor(cute.recast_ptr(sDQ.iterator, dq_epi_layout.inner),
                                    cute.make_layout((TILE, DQ_QCOLS), stride=(DQ_QCOLS, 1)))
            sDQ_chunks = cute.local_tile(sDQv, dq_tiler, (0, None))                    # (TILE, 32, DQ_QCOLS/32)
            NBOX = HALF // DQ_QCOLS                                                     # TMA boxes per half (1 or 2)
            r2s_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, cutlass.Float16, f32, tiled_ld)
            tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_ld)
            thr_r2s = tiled_r2s.get_slice(rtid)
            tRS_sDQ = thr_r2s.partition_D(sDQ_chunks)                                  # (R2S, M, N, 2)
            tRS_rDQ0 = tiled_r2s.retile(rDQ0)
            tRS_rDQ1 = tiled_r2s.retile(rDQ1)
            rDQc0 = cute.make_rmem_tensor(tRS_rDQ0.shape, cutlass.BFloat16 if DQ_DIRECT else cutlass.Float16)
            rDQc1 = cute.make_rmem_tensor(tRS_rDQ1.shape, cutlass.BFloat16 if DQ_DIRECT else cutlass.Float16)
            store_pipe = pipeline.PipelineTmaStore.create(num_stages=1, producer_group=load_threads)
            rHa = cute.make_rmem_tensor((32,), cutlass.Float16 if DQ_DIRECT_F16 else cutlass.BFloat16)
            rHb = cute.make_rmem_tensor((32,), cutlass.Float16 if DQ_DIRECT_F16 else cutlass.BFloat16)
            rHaw = cute.make_tensor(cute.recast_ptr(rHa.iterator, dtype=cutlass.Int32), cute.make_layout(16))
            rHbw = cute.make_tensor(cute.recast_ptr(rHb.iterator, dtype=cutlass.Int32), cute.make_layout(16))
            stg_base = cutlass.Int32((sDQ.iterator.toint() & 0x3FFFF) + warp_idx * (DQ_ROWS_D * HALF * 2))   # this warp's 2 KB box
            rDQh0 = cute.make_rmem_tensor((32,), cutlass.Float16)
            rDQh1 = cute.make_rmem_tensor((32,), cutlass.Float16)
            rDQw0 = cute.make_tensor(cute.recast_ptr(rDQh0.iterator, dtype=cutlass.Int32), cute.make_layout(16))
            rDQw1 = cute.make_tensor(cute.recast_ptr(rDQh1.iterator, dtype=cutlass.Int32), cute.make_layout(16))
            dq_rowb = cutlass.Int64(mDQ_raw.stride[0]) * 2            # (DQ_RED experiment path; untiled layout only)
            lane = tidx % 32
            cvt_push_addr = cutlass.Int32(storage.cvt_ctr.data_ptr().toint() & 0x3FFFF)
            cvt_pop_addr = cvt_push_addr + 4
            cvt_done_addr = cvt_push_addr + 8
            cvt_q_base = cutlass.Int64(cvt_q.iterator.toint()) + cutlass.Int64(bidx) * (CVTQ * CVTQ_W * 4)
            cnt_base = cutlass.Int64(blk_cnt.iterator.toint())
            stats_base = cnt_base + cutlass.Int64(num_heads * nblk_acc + bidx * 8) * 4
            carry_base = cutlass.Int64(0)              # counters of the previous item's newest CVT_TAIL halves (counted one item late)
            carry_n = cutlass.Int32(0)
            n = cutlass.Int32(0)
            active = cutlass.Boolean(True)
            while active:
                slot = n % RING
                _rpush("red_wait_item")
                cute.arch.mbarrier_wait(tile_bars + slot, (n // RING) & 1)
                _rpop()
                tile = ring[slot * RING_W + 0]
                if tile < 0:
                    active = False
                else:
                    head = ring[slot * RING_W + 1]
                    seq_start = ring[slot * RING_W + 2]
                    seqlen = ring[slot * RING_W + 3] - seq_start
                    acc_start = ring[slot * RING_W + 4]
                    jblk = tile & 0xFFFF
                    kv0 = jblk * TILE
                    n_q_blk = (seqlen + TILE - 1) // TILE
                    n_iters = n_q_blk - jblk
                    if cutlass.const_expr(DQ_DIRECT):
                        mDQ_h = cute.domain_offset((seq_start, 0), mDQ[(None, None, head)])     # (T, D) view of the head: dq rows of this sequence
                    else:
                        mDQ_h = cute.domain_offset((0, acc_start), mDQ[(None, None, head)])      # (D, T_pad) view of the head
                    if cutlass.const_expr(DQ_DIRECT):
                        gDQ = cute.local_tile(mDQ_h, (DQ_ROWS_D, HALF), (None, None))           # boxes (16 q, 64 d): (q-box, d-box)
                    else:
                        gDQ = cute.local_tile(mDQ_h, (DQ_COLS, DQ_QCOLS), (None, None))         # boxes (32 d, 32 q): (d-chunk, q-box)
                    tDQsDQ, tDQgDQ = cpasync.tma_partition(dq_atom, 0, cute.make_layout(1),
                                                           cute.group_modes(sDQ, 0, 2), cute.group_modes(gDQ, 0, 2))
                    kv_row = kv0 + row0
                    kv_valid = kv_row < seqlen
                    kv_tok = seq_start + kv_row
                    if cutlass.const_expr(EPI_TMA):
                        mDK_h = cute.domain_offset((seq_start, 0), mDKt[(None, None, head)])
                        mDV_h = cute.domain_offset((seq_start, 0), mDVt[(None, None, head)])
                        gDK = cute.local_tile(mDK_h, (DQ_ROWS_D, HALF), (None, None))           # boxes (16 kv, 64 d)
                        gDV = cute.local_tile(mDV_h, (DQ_ROWS_D, HALF), (None, None))
                        tDKs, tDKg = cpasync.tma_partition(dk_st_atom, 0, cute.make_layout(1),
                                                           cute.group_modes(sDQ, 0, 2), cute.group_modes(gDK, 0, 2))
                        tDVs, tDVg = cpasync.tma_partition(dv_st_atom, 0, cute.make_layout(1),
                                                           cute.group_modes(sDQ, 0, 2), cute.group_modes(gDV, 0, 2))
                    dq_row_base = cutlass.Int64(0)
                    if cutlass.const_expr(DQ_RED):      # experiment path (untiled (D, T_pad, H) accumulator only)
                        dq_row_base = cutlass.Int64(cute.domain_offset((0, acc_start, head), mDQ_raw).iterator.toint()) + cutlass.Int64(row0) * dq_rowb
                    for it in cutlass.range(n_iters, unroll=1):
                        iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                        for hx in cutlass.range_constexpr(2):
                            _rpush("red_wait_dq")
                            dqh = dq_cons.wait_and_advance()
                            _rpop()
                            if cutlass.const_expr(EXP_NOREDUCE):
                                dqh.release()
                            elif cutlass.const_expr(DQ_RED):
                                _rpush("red_tmem_ld")
                                cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 0)], rDQ0)
                                cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 1)], rDQ1)
                                cute.arch.fence_view_async_tmem_load()
                                dqh.release()
                                _rpop()
                                _rpush("red_red")
                                rDQh0.store(rDQ0.load().to(cutlass.Float16))
                                rDQh1.store(rDQ1.load().to(cutlass.Float16))
                                q0 = iq * TILE + hx * HALF
                                for g in cutlass.range_constexpr(8):
                                    qg = q0 + 8 * g
                                    if qg < seqlen:
                                        w = (rDQw0, rDQw1)[g // 4]
                                        j = 4 * (g % 4)
                                        red_f16x8(dq_row_base + cutlass.Int64(qg) * 2, w[j], w[j + 1], w[j + 2], w[j + 3])
                                _rpop()
                            else:
                                _rpush("red_tmem_ld")
                                cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 0)], rDQ0)
                                cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 1)], rDQ1)
                                cute.arch.fence_view_async_tmem_load()
                                dqh.release()
                                _rpop()
                                if cutlass.const_expr(DQ_DIRECT):
                                    # thread = q row (TMEM lane): 64 d values -> bf16 * sm_scale, 8 x 16-B chunks of a 128-B row;
                                    # pass p: lanes 16p..16p+15 write rows 0..15 of the (16, 64) SW128 box, then one TMA reduce
                                    if cutlass.const_expr(DQ_DIRECT_F16):
                                        rHa.store(rDQ0.load().to(cutlass.Float16))
                                        rHb.store(rDQ1.load().to(cutlass.Float16))
                                    else:
                                        rHa.store((rDQ0.load() * sm_scale).to(cutlass.BFloat16))
                                        rHb.store((rDQ1.load() * sm_scale).to(cutlass.BFloat16))
                                    row16 = lane & 15
                                    for p in cutlass.range_constexpr(2):
                                        _rpush("red_acquire")
                                        store_pipe.producer_acquire()
                                        cute.arch.sync_warp()
                                        _rpop()
                                        _rpush("red_sts")
                                        if (lane >> 4) == p:
                                            for v in cutlass.range_constexpr(8):
                                                w = (rHaw, rHbw)[v // 4]
                                                j = 4 * (v % 4)
                                                ptx_st_shared_v4(stg_base + row16 * 128 + ((v ^ (row16 & 7)) << 4), w[j], w[j + 1], w[j + 2], w[j + 3])
                                        cute.arch.fence_view_async_shared()
                                        cute.arch.sync_warp()
                                        _rpop()
                                        _rpush("red_tma")
                                        cute.copy(dq_atom, tDQsDQ[(None, warp_idx)], tDQgDQ[(None, 8 * iq + 2 * warp_idx + p, hx)])
                                        store_pipe.producer_commit()
                                        _rpop()
                                else:
                                    rDQc0.store(tRS_rDQ0.load().to(cutlass.Float16))
                                    rDQc1.store(tRS_rDQ1.load().to(cutlass.Float16))
                                for c in cutlass.range_constexpr(NBOX if not DQ_DIRECT else 0):
                                    _rpush("red_acquire")
                                    store_pipe.producer_acquire()        # previous TMA reduce finished reading this warp's box
                                    cute.arch.sync_warp()
                                    _rpop()
                                    _rpush("red_sts")
                                    if cutlass.const_expr(NBOX == 1):
                                        cute.copy(tiled_r2s, rDQc0, tRS_sDQ[(None, None, None, 0)])
                                        cute.copy(tiled_r2s, rDQc1, tRS_sDQ[(None, None, None, 1)])
                                    elif cutlass.const_expr(c == 0):
                                        cute.copy(tiled_r2s, rDQc0, tRS_sDQ[(None, None, None, 0)])
                                    else:
                                        cute.copy(tiled_r2s, rDQc1, tRS_sDQ[(None, None, None, 0)])
                                    cute.arch.fence_view_async_shared()
                                    cute.arch.sync_warp()
                                    _rpop()
                                    _rpush("red_tma")
                                    cute.copy(dq_atom, tDQsDQ[(None, warp_idx)], tDQgDQ[(None, warp_idx, (2 * iq + hx) * NBOX + c)])
                                    store_pipe.producer_commit()
                                    _rpop()
                            if cutlass.const_expr(hx == 0 and not EPI_WG1):
                                if it == n_iters - 1:
                                    # dV epilogue (dV_b retired before dK_b / dQ_b^T): TMEM lane = kv row, 4 chunks of 32 d columns
                                    _rpush("red_epi_wait")
                                    kh = dkdv_cons.wait_and_advance()
                                    _rpop()
                                    _rpush("red_epi")
                                    if cutlass.const_expr(EPI_TMA):
                                        self.epi_acc(tiled_ld, tTR_tDV, rDQ0, rDQ1, rHa, rHb, rHaw, rHbw, sm_scale, False, stg_base, lane, warp_idx,
                                                     kv0, jblk, seqlen, kv_valid, kv_tok, head, dv_st_atom, tDVs, tDVg, mDV, vec_atom, vec_layout)
                                    for pair in cutlass.range_constexpr(0 if EPI_TMA else 2):
                                        cute.copy(tiled_ld, tTR_tDV[(None, None, None, 2 * pair)], rDQ0)
                                        cute.copy(tiled_ld, tTR_tDV[(None, None, None, 2 * pair + 1)], rDQ1)
                                        cute.arch.fence_view_async_tmem_load()
                                        for c in cutlass.range_constexpr(2):
                                            chunk = 2 * pair + c
                                            rO = (rDQ0, rDQ1)[c]
                                            rOb = cute.make_rmem_tensor((32,), bf16)
                                            rOb.store(rO.load().to(bf16))
                                            if cutlass.const_expr(not EXP_NOEPI):
                                                if kv_valid:
                                                    rOv = cute.make_tensor(rOb.iterator, vec_layout)
                                                    dstv = cute.make_tensor((mDV.iterator + mDV.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                                                    cute.copy(vec_atom, rOv, dstv)
                                    kh.release()
                                    _rpop()
                    # ---- item epilogue: dK (dV was written inside the last iteration)
                    if cutlass.const_expr(not EPI_WG1):
                        _rpush("red_epi_wait2")
                        kh2 = dkdv_cons.wait_and_advance()
                        _rpop()
                    _rpush("red_epi2")
                    if cutlass.const_expr(EPI_TMA):
                        self.epi_acc(tiled_ld, tTR_tDK, rDQ0, rDQ1, rHa, rHb, rHaw, rHbw, sm_scale, True, stg_base, lane, warp_idx,
                                     kv0, jblk, seqlen, kv_valid, kv_tok, head, dk_st_atom, tDKs, tDKg, mDK, vec_atom, vec_layout)
                    for pair in cutlass.range_constexpr(0 if (EPI_TMA or EPI_WG1) else 2):
                        cute.copy(tiled_ld, tTR_tDK[(None, None, None, 2 * pair)], rDQ0)
                        cute.copy(tiled_ld, tTR_tDK[(None, None, None, 2 * pair + 1)], rDQ1)
                        cute.arch.fence_view_async_tmem_load()
                        for c in cutlass.range_constexpr(2):
                            chunk = 2 * pair + c
                            rO = (rDQ0, rDQ1)[c]
                            rOb = cute.make_rmem_tensor((32,), bf16)
                            rOb.store((rO.load() * sm_scale).to(bf16))
                            if cutlass.const_expr(not EXP_NOEPI):
                                if kv_valid:
                                    rOv = cute.make_tensor(rOb.iterator, vec_layout)
                                    dstv = cute.make_tensor((mDK.iterator + mDK.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                                    cute.copy(vec_atom, rOv, dstv)
                    if cutlass.const_expr(not EPI_WG1):
                        kh2.release()
                    _rpop()
                    if cutlass.const_expr(CVT_INK):
                        # item end: all dQ reduces of this item complete -> count them (one relaxed red per (block, warp), lanes in
                        # parallel); the two 64-token blocks of this tile's diagonal q block are converted by this CTA's KV warp
                        # once their counters reach 4 * (jblk + 1) (all kv tiles 0..jblk, all 4 reduce warps)
                        # all but the newest NBOX groups (the last half, hx = 1 of the diagonal block) are complete; that half is
                        # counted at the next item end (carried record), so the wait rarely stalls
                        # all but the newest CVT_TAIL halves are complete. Halves are issued top-down (Q_REVERSE), so the newest
                        # ones are the lowest block indices idx = 2*(iq - jblk) + hx < CVT_TAIL; those are counted at the next
                        # item end (carried), so this wait rarely stalls
                        cute.arch.cp_async_bulk_wait_group(NBOX * CVT_TAIL, read=False)
                        if cutlass.const_expr(CVT_ATOM == "release"):
                            cute.arch.fence_acq_rel_gpu()
                        cute.arch.sync_warp()
                        if lane == 0:
                            for cidx in cutlass.range(carry_n, unroll=1):
                                ptx_red_add(carry_base + cutlass.Int64(cidx) * 4, 1)
                        g0 = (acc_start >> 6) + 2 * jblk
                        c_base = cnt_base + cutlass.Int64(head * nblk_acc + g0) * 4
                        n_half = 2 * n_iters
                        n_valid = (seqlen - jblk * TILE + HALF - 1) // HALF            # halves of this item with data
                        for idx in cutlass.range(CVT_TAIL + lane, n_half, 32, unroll=1):
                            if idx < n_valid:
                                ptx_red_add(c_base + cutlass.Int64(idx) * 4, 1)
                        carry_base = c_base
                        carry_n = min(min(CVT_TAIL, n_half), n_valid)
                        cute.arch.sync_warp()
                        if warp_idx == REDUCE_WARPS[0] and CVT_DBG != 1:
                            if lane == 0:
                                for hx2 in cutlass.range_constexpr(2):
                                    nv2 = min(seqlen - (jblk * TILE + hx2 * HALF), HALF)
                                    if nv2 > 0:
                                        cvt_push_job(head, g0 + hx2, seq_start + jblk * TILE + hx2 * HALF, nv2,
                                                     len(REDUCE_WARPS) * (jblk + 1), cvt_push_addr, cvt_pop_addr, cvt_q_base, stats_base)
                        cute.arch.sync_warp()
                n += 1
            _rpush("red_tail")
            store_pipe.producer_tail()
            if cutlass.const_expr(EPI_TMA):
                cute.arch.cp_async_bulk_wait_group(0, read=False)
            if cutlass.const_expr(CVT_INK):
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                if cutlass.const_expr(CVT_ATOM == "release"):
                    cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_warp()
                if lane == 0:
                    for cidx in cutlass.range(carry_n, unroll=1):
                        ptx_red_add(carry_base + cutlass.Int64(cidx) * 4, 1)
                    ptx_shared_atom_add(cvt_done_addr, 1)
                cute.arch.sync_warp()
            _rpop()
            _rpush("red_fin")
            pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=n_final).arrive_and_wait()
            _rpop()


# ---------------------------------------------------------------------------------------- host wrapper
_KERNEL = None
_COMPILED = {}
_NUM_SMS = None


def _as_tdh(t):
    """(T, H, D) torch tensor -> cute (T, D, H) view."""
    return from_dlpack(t.permute(0, 2, 1), assumed_align=16).mark_layout_dynamic(leading_dim=1)


def attention_bwd_prepare(q, k, v, dout, dq_accum, dk, dv, cu_seqlens, tile_table, max_tiles, sm_scale, xq, xdo, cb, seq_acc, counter,
                          dq=None, blk_cnt=None, cvt_q=None):
    """Host-side argument preparation (DLPack conversions); call before launching any kernel."""
    global _KERNEL, _NUM_SMS
    if _KERNEL is None:
        _KERNEL = AttnBwdSm100V9()
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count
    T, H, D = q.shape
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    grid_x = min(_NUM_SMS, H * max_tiles)
    args = (_as_tdh(q), _as_tdh(k), _as_tdh(v), _as_tdh(dout),
            (_as_tdh(dq_accum) if DQ_DIRECT else                                                                          # dq itself: (T, D, H) bf16
             from_dlpack(dq_accum.permute(3, 2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0) if DQ_TILED   # (64, D, nblk, H)
             else from_dlpack(dq_accum.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)),           # (D, T_pad, H)
            _as_tdh(dk), _as_tdh(dv),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            from_dlpack(tile_table, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            from_dlpack(seq_acc, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            from_dlpack(counter, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            cutlass.Int32(H * max_tiles), cutlass.Int32(H), cutlass.Float32(sm_scale),
            from_dlpack(xq.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),      # (8, T, H)
            from_dlpack(xdo.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(cb.view(192, 8).permute(1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),  # (8, 192)
            cutlass.Int32(grid_x), stream,
            _as_tdh(dq if dq is not None else q),
            from_dlpack(blk_cnt if blk_cnt is not None else counter, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            cutlass.Int32(dq_accum.shape[1] if DQ_TILED else 1),
            from_dlpack(cvt_q if cvt_q is not None else counter, assumed_align=16).mark_layout_dynamic(leading_dim=0))
    key = "main"
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_KERNEL, *args)
        _COMPILED[key] = fn
    return fn, args


def attention_bwd_launch(prepared):
    fn, args = prepared
    fn(*args)
