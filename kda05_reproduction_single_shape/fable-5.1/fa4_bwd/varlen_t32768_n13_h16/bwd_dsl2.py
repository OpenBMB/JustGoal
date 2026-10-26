"""CuTe DSL (Blackwell sm_100a) warp-specialized attention backward main kernel, v8b "half-tile pipeline, split roles".

Per CTA = one 128-row kv block of one sequence, one head; grid (H, tiles) over a device-built LPT tile table.
q tiles of 128 rows are processed as two 64-row halves (a: rows 0-63, b: rows 64-127) for S^T, dP^T and dQ^T so
that TMEM holds one half of each plus the packed P tile; the MMA warp issues the next tile's S/dP halves while
the compute warps still work on the current tile.

  TMEM (512 cols): dK[0:128] | dV[128:256] | P bf16 packed [256:320] | S half [320:384] | dP half [384:448] | dQ^T half [448:512]
  warps 0-3   : reduce  (dQ^T half: TMEM (lane=d, col=q) -> f16 SMEM (q, d) per warp -> TMA reduce-add into the f16 dq accumulator)
  warps 4-7   : compute WG0: P = exp2(S*scale*log2e - lse*log2e) masked -> TMEM bf16 (both halves)
  warps 8-11  : compute WG1: dS = P*(dP-delta) -> SMEM bf16 (both halves); P read back from TMEM
  warp 12     : MMA issue  S_x^T = K Q_x^T, dP_x^T = V dO_x^T (N=64), dV += P^T dO, dK += dS^T Q (K=64 halves),
                           dQ_x^T = K^T dS_x^T (M=128 d, N=64 q) so that dS stage x is free right after dK_x/dQ_x
  warp 13     : TMA loads (K,V once as 2 d-blocks each; Q 4 half-slots; dO 3 half-slots) + LSE/delta vectors
  warps 14-15 : idle
  SMEM: K 32K | V 32K | Q 64K | dO 48K | dS^T 32K | dQ staging 4 warps x 4K = 224K
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
Q_STAGES = int(_os.environ.get("ATT_Q_STAGES", "4"))      # Q half-slots (16 KB each)
DO_STAGES = int(_os.environ.get("ATT_DO_STAGES", "3"))    # dO half-slots
DQ_COLS = 32                                              # d columns per reduce warp (= TMEM lanes per warp)
DQ_QCOLS = int(_os.environ.get("ATT_DQ_QCOLS", "32"))     # q columns per TMA reduce box (staging = 4 warps x 32 d x DQ_QCOLS f16)
Q_REVERSE = _os.environ.get("ATT_QREV", "1") == "1"
NUM_WARPS = 16
NUM_THREADS = NUM_WARPS * 32
REDUCE_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7, 8, 9, 10, 11)
MMA_WARP = 12       # issues S^T, dP^T, dQ^T
LOAD_WARP = 13
MMA2_WARP = 14      # issues dV, dK
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
EXP_NOMASK = _os.environ.get("ATT_NOMASK", "0") == "1"      # experiment: never take the masked path (timing only)
EXP_NOTAIL = _os.environ.get("ATT_NOTAIL", "0") == "1"      # experiment: skip the 9th k-step MMAs (timing only)
XK = 512          # bf16 elements per extra B block (64 rows x 8: first K core-matrix column of the 9th k-step)
XBLK_BYTES = 1024
EXP_SAMEQ = _os.environ.get("ATT_SAMEQ", "0") == "1"
EXP_K1 = _os.environ.get("ATT_MMA_K1", "0") == "1"           # experiment: one k-step per GEMM (timing only)
NKSTEPS = int(_os.environ.get("ATT_MMA_K", "8"))
NK8 = 1 if EXP_K1 else NKSTEPS
NK4 = 1 if EXP_K1 else min(4, NKSTEPS)


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
REGS_WG3 = int(_os.environ.get("ATT_REGS_WG3", "56"))        # mma / load / idle warps (12-15)
REGS_REDUCE = int(_os.environ.get("ATT_REGS_REDUCE", "104"))  # warps 0-3
REGS_COMPUTE = int(_os.environ.get("ATT_REGS_COMPUTE", "176"))  # warps 4-11
assert REGS_WG3 * 128 + REGS_REDUCE * 128 + REGS_COMPUTE * 256 <= 65536


def _set_regs(n):
    """setmaxnreg relative to the 128-register launch value (inc above, dec below)."""
    if n > 128:
        cute.arch.warpgroup_reg_alloc(n)
    elif n < 128:
        cute.arch.warpgroup_reg_dealloc(n)


MMA_MODE = _os.environ.get("ATT_MMA", "ptx")   # "ptx": inline tcgen05.mma with precomputed descriptors; "dsl": cute.gemm


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
def gemm_ptx(op, acc: cute.Tensor, sA_stage, sB_stage, tmem_a, accumulate, ks_a, ks_b):
    """Issue the k-blocks ks_a (A) x ks_b (B) of one GEMM with tcgen05.mma from the elected lane.
    sA_stage/sB_stage: smem tensors of ONE pipeline stage with layout (MMA, M|N, K); tmem_a: Int32 TMEM address of the
    first A k-block (or None; then 8 columns per k-block). accumulate: Int32/Boolean (0 => overwrite on the first k-block)."""
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
        a_lo = cutlass.Int32(lo_a) + cutlass.Int32(((sA_stage.iterator.toint() & 0x3FFFF) >> 4) + offs_a[ks_a[0]])
        asm = _asm_smem_a(hi_a, hi_b, offs_a, offs_b, idesc, ks_a, ks_b)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), a_lo.ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    else:
        asm = _asm_tmem_a(hi_b, offs_b, idesc, ks_b)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), cutlass.Int32(tmem_a).ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


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
    kv_bar: cute.struct.MemRange[cutlass.Int64, 1]                # K,V loaded (once)
    s_bars: cute.struct.MemRange[cutlass.Int64, 2]                # mma->WG0 S half (1 stage)
    dp_bars: cute.struct.MemRange[cutlass.Int64, 2]               # mma->WG1 dP half (1 stage)
    dq_bars: cute.struct.MemRange[cutlass.Int64, 2]               # mma->reduce dQ^T half (1 stage)
    p_bars: cute.struct.MemRange[cutlass.Int64, 4]                # WG0->mma P half (2 stages: a, b)
    p2_bars: cute.struct.MemRange[cutlass.Int64, 4]               # WG0->WG1 P half (2 stages: a, b)
    ds_bars: cute.struct.MemRange[cutlass.Int64, 4]               # WG1->mma dS half (2 stages: a, b)
    dkdv_bars: cute.struct.MemRange[cutlass.Int64, 4]             # mma->compute dK/dV ready (2 stages)
    tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]


class AttnBwdSm100V8:
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
                 mLSE: cute.Tensor, mDelta: cute.Tensor, mDQacc: cute.Tensor,
                 mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor, tile_table: cute.Tensor,
                 max_tiles: cutlass.Int32, total_tokens: cutlass.Int32,
                 sm_scale: cutlass.Float32, mXQ: cute.Tensor, mXDO: cute.Tensor, mCB: cute.Tensor, stream: cuda.CUstream):
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

        k_layout = sm100_utils.make_smem_layout_a(mma_s, tiler_kv, bf16, 2)            # (MMA, M, K=4, 2 d-blocks)
        q_layout = sm100_utils.make_smem_layout_b(mma_s, tiler_s, bf16, Q_STAGES)      # (MMA, N=64, K=8, slots)
        do_layout = sm100_utils.make_smem_layout_b(mma_s, tiler_s, bf16, DO_STAGES)
        qT_layout = sm100_utils.make_smem_layout_b(mma_dsq, tiler_dk, bf16, Q_STAGES)  # B: n=d, k=q (4 k-steps)
        doT_layout = sm100_utils.make_smem_layout_b(mma_pdo, tiler_dk, bf16, DO_STAGES)
        kT_layout = sm100_utils.make_smem_layout_a(mma_dq, tiler_dqT, bf16, 1)         # A: m=128 d, k=kv (byte-identical to k_layout)
        ds_layout = sm100_utils.make_smem_layout_a(mma_dsq, tiler_ds, bf16, 1)         # A: m=kv, k=q (K-major)
        dsT_layout = sm100_utils.make_smem_layout_b(mma_dq, tiler_dqT, bf16, 2)        # B: n=64 q, k=kv, 2 stages (16 KB each)
        p_layout = sm100_utils.make_smem_layout_a(mma_pdo, tiler_ds, bf16, 1)          # TMEM A layout for P^T (k=q 128)
        # dQ^T staging: per reduce warp one (32 d, 64 q) f16 box (4 KB, 128-B rows, SW128); the accumulator is (H, D, T)
        dq_epi_layout = sm100_utils.make_smem_layout_epi(cutlass.Float16, utils.LayoutEnum.ROW_MAJOR, (DQ_COLS, DQ_QCOLS), 4)

        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_atom, k_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), tiler_kv, mma_s)
        v_atom, v_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mV, cute.slice_(k_layout, (None, None, None, 0)), tiler_kv, mma_s)
        q_atom, q_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), tiler_s, mma_s)
        do_atom, do_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), tiler_s, mma_s)
        # 9th k-step operands: (8 cols, 64 rows) bf16 boxes of the per-token extra rows and the 4 KB constant block
        x_layout = cute.make_layout((8, HALF, Q_STAGES), stride=(1, 8, 8 * HALF))
        xq_atom, xq_tma = cpasync.make_tiled_tma_atom(tma_op, mXQ, cute.slice_(x_layout, (None, None, 0)), (8, HALF))
        xdo_atom, xdo_tma = cpasync.make_tiled_tma_atom(tma_op, mXDO, cute.slice_(x_layout, (None, None, 0)), (8, HALF))
        cb_layout = cute.make_layout((8, 192), stride=(1, 8))                 # ones block (128 rows) + zero block (64 rows)
        cb_atom, cb_tma = cpasync.make_tiled_tma_atom(tma_op, mCB, cb_layout, (8, 192))
        dq_red_atom, dq_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionKind.ADD), mDQacc,
            cute.slice_(dq_epi_layout, (None, None, 0)), (DQ_COLS, DQ_QCOLS))

        if cutlass.const_expr(_os.environ.get("ATT_PRINT_LAYOUTS", "0") == "1"):
            print("k_layout", k_layout)
            print("kT_layout", kT_layout)
            print("q_layout", q_layout)
            print("qT_layout", qT_layout)
            print("do_layout", do_layout)
            print("doT_layout", doT_layout)
            print("ds_layout", ds_layout)
            print("dsT_layout", dsT_layout)
            print("p_layout", p_layout)
            print("dq_epi_layout", dq_epi_layout)

        H = mQ.shape[2]
        self.kernel(mma_s, mma_pdo, mma_dsq, mma_dq,
                    k_atom, k_tma, v_atom, v_tma, q_atom, q_tma, do_atom, do_tma, dq_red_atom, dq_tma,
                    mLSE, mDelta, mDK, mDV, cu_seqlens, tile_table,
                    k_layout, q_layout, do_layout, qT_layout, kT_layout, doT_layout, ds_layout, dsT_layout,
                    p_layout, dq_epi_layout, total_tokens, sm_scale,
                    xq_atom, xq_tma, xdo_atom, xdo_tma, cb_atom, cb_tma, x_layout, cb_layout).launch(
            grid=(H, max_tiles, 1), block=(NUM_THREADS, 1, 1), stream=stream)

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
               mLSE: cute.Tensor, mDelta: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor,
               tile_table: cute.Tensor, k_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               qT_layout: cute.ComposedLayout, kT_layout: cute.ComposedLayout, doT_layout: cute.ComposedLayout,
               ds_layout: cute.ComposedLayout, dsT_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               dq_epi_layout: cute.ComposedLayout, total_tokens: cutlass.Int32, sm_scale: cutlass.Float32,
               xq_atom: cute.CopyAtom, mXQ: cute.Tensor, xdo_atom: cute.CopyAtom, mXDO: cute.Tensor,
               cb_atom: cute.CopyAtom, mCB: cute.Tensor, x_layout: cute.Layout, cb_layout: cute.Layout):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        head, tile_idx, _ = cute.arch.block_idx()
        tiler_kv = (TILE, HALF, HALF)
        tiler_s = (TILE, HALF, HEAD_DIM)

        if cutlass.const_expr(IKET):
            setup_tok = cute.experimental.iket.range_start("setup")
        tile = tile_table[tile_idx]
        tile_c = max(tile, 0)              # padding entries are -1: never index cu_seqlens with them
        seq = tile_c >> 16
        jblk = tile_c & 0xFFFF
        seq_start = cu_seqlens[seq]
        seqlen = cu_seqlens[seq + 1] - seq_start
        kv0 = jblk * TILE
        n_q_blk = (seqlen + TILE - 1) // TILE
        n_iters = n_q_blk - jblk  # q blocks jblk .. n_q_blk-1 (causal)

        if tile >= 0:
            # ---------------- shared memory
            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sK = smem.allocate_tensor(bf16, k_layout.outer, 1024, swizzle=k_layout.inner)      # (MMA, M, K, 2)
            sV = smem.allocate_tensor(bf16, k_layout.outer, 1024, swizzle=k_layout.inner)
            sQ = smem.allocate_tensor(bf16, q_layout.outer, 1024, swizzle=q_layout.inner)      # (MMA, N, K, Q_STAGES)
            sDO = smem.allocate_tensor(bf16, do_layout.outer, 1024, swizzle=do_layout.inner)   # (MMA, N, K, DO_STAGES)
            sDS = smem.allocate_tensor(bf16, ds_layout.outer, 1024, swizzle=ds_layout.inner)
            sDQ = smem.allocate_tensor(cutlass.Float16, dq_epi_layout.outer, 1024, swizzle=dq_epi_layout.inner)  # (32 d, 64 q, 4 warps)
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
            sKT = cute.make_tensor(cute.recast_ptr(sK.iterator, kT_layout.inner), kT_layout.outer)
            sDOT = cute.make_tensor(cute.recast_ptr(sDO.iterator, doT_layout.inner), doT_layout.outer)
            sDST = cute.make_tensor(cute.recast_ptr(sDS.iterator, dsT_layout.inner), dsT_layout.outer)

            if warp_idx == LOAD_WARP:
                cpasync.prefetch_descriptor(k_atom)
                cpasync.prefetch_descriptor(v_atom)
                cpasync.prefetch_descriptor(q_atom)
                cpasync.prefetch_descriptor(do_atom)
                cpasync.prefetch_descriptor(dq_atom)
                cpasync.prefetch_descriptor(xq_atom)
                cpasync.prefetch_descriptor(xdo_atom)
                cpasync.prefetch_descriptor(cb_atom)

            # ---------------- pipelines
            q_bytes = cute.size_in_bytes(bf16, cute.select(q_layout, mode=[0, 1, 2])) + XBLK_BYTES
            do_bytes = cute.size_in_bytes(bf16, cute.select(do_layout, mode=[0, 1, 2])) + XBLK_BYTES
            kv_bytes = 4 * cute.size_in_bytes(bf16, cute.select(k_layout, mode=[0, 1, 2])) + 3072   # K and V (2 d-blocks each) + constant block
            one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
            two_issuers = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=2)   # releases by both MMA-issuing warps
            load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=32)
            mma_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=32)
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
                consumer_group=compute_threads, defer_sync=True).make_participants()
            kv_bar = storage.kv_bar.data_ptr()
            if tidx == 0:
                cute.arch.mbarrier_init(kv_bar, 1)
            cute.arch.mbarrier_init_fence()

            tmem = utils.TmemAllocator(
                storage.tmem_holding.data_ptr(),
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=BAR_TMEM_ALLOC,
                                                           num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))),
                allocator_warp_id=MMA_WARP)
            cute.arch.barrier()  # all 512 threads: barriers initialized

            # ---------------- tensors / fragments (shared by roles; cheap)
            mK_h = cute.domain_offset((seq_start + kv0, 0), mK[(None, None, head)])
            mV_h = cute.domain_offset((seq_start + kv0, 0), mV[(None, None, head)])
            mQ_h = cute.domain_offset((seq_start, 0), mQ[(None, None, head)])
            mDO_h = cute.domain_offset((seq_start, 0), mDO[(None, None, head)])
            # padded start of this sequence in the transposed dQ accumulator (every sequence 64-aligned)
            acc_start = cutlass.Int32(0)
            for sq in cutlass.range(seq, unroll=1):
                acc_start += ((cu_seqlens[sq + 1] - cu_seqlens[sq] + HALF - 1) // HALF) * HALF
            mDQ_h = cute.domain_offset((0, acc_start), mDQ[(None, None, head)])                   # (D, T_pad) view of the head
            gK = cute.local_tile(mK_h, tiler_kv, (0, 0, None), proj=(1, None, 1))      # (TILE, 64, 2 d-blocks)
            gV = cute.local_tile(mV_h, tiler_kv, (0, 0, None), proj=(1, None, 1))
            gQ = cute.local_tile(mQ_h, tiler_s, (0, None, 0), proj=(None, 1, 1))       # (64, D, nQhalf)
            gDO = cute.local_tile(mDO_h, tiler_s, (0, None, 0), proj=(None, 1, 1))
            gDQ = cute.local_tile(mDQ_h, (DQ_COLS, DQ_QCOLS), (None, None))             # (32 d, DQ_QCOLS q, 4 d-chunks, nQbox)

            thr_s = mma_s.get_slice(0)
            tKgK = thr_s.partition_A(gK)
            tVgV = thr_s.partition_A(gV)
            tQgQ = thr_s.partition_B(gQ)
            tDOgDO = thr_s.partition_B(gDO)
            tKsK, tKgK_tma = cpasync.tma_partition(k_atom, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tKgK, 0, 3))
            tVsV, tVgV_tma = cpasync.tma_partition(v_atom, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tVgV, 0, 3))
            tQsQ, tQgQ_tma = cpasync.tma_partition(q_atom, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tQgQ, 0, 3))
            tDOsDO, tDOgDO_tma = cpasync.tma_partition(do_atom, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(tDOgDO, 0, 3))
            mXQ_h = cute.domain_offset((0, seq_start), mXQ[(None, None, head)])          # (8, T)
            mXDO_h = cute.domain_offset((0, seq_start), mXDO[(None, None, head)])
            gXQ = cute.local_tile(mXQ_h, (8, HALF), (0, None))                            # (8, 64, nQhalf)
            gXDO = cute.local_tile(mXDO_h, (8, HALF), (0, None))
            tXQs, tXQg = cpasync.tma_partition(xq_atom, 0, cute.make_layout(1), cute.group_modes(sQx, 0, 2), cute.group_modes(gXQ, 0, 2))
            tXDOs, tXDOg = cpasync.tma_partition(xdo_atom, 0, cute.make_layout(1), cute.group_modes(sDOx, 0, 2), cute.group_modes(gXDO, 0, 2))
            gCB = cute.local_tile(mCB, (8, 192), (0, 0))
            tCBs, tCBg = cpasync.tma_partition(cb_atom, 0, cute.make_layout(1), cute.group_modes(sCB, 0, 2), cute.group_modes(gCB, 0, 2))

            # MMA fragments
            tSrK = mma_s.make_fragment_A(sK)         # (MMA, M, K=4, 2)
            tSrQ = mma_s.make_fragment_B(sQ)         # (MMA, N, K=8, Q_STAGES)
            tDPrV = mma_s.make_fragment_A(sV)
            tDPrDO = mma_s.make_fragment_B(sDO)
            tDVrDOT = mma_pdo.make_fragment_B(sDOT)  # (MMA, N=128, K=4, DO_STAGES)
            tDKrDST = mma_dsq.make_fragment_A(sDS)   # (MMA, M, K=8, 1)
            tDKrQT = mma_dsq.make_fragment_B(sQT)    # (MMA, N=128, K=4, Q_STAGES)
            tDQrKT = mma_dq.make_fragment_A(sKT)     # (MMA, M=128 d, K=8, 1)
            tDQrDST = mma_dq.make_fragment_B(sDST)   # (MMA, N=64 q, K=8, 2 stages)

            tS_fake = mma_s.make_fragment_C(mma_s.partition_shape_C((TILE, HALF)))
            tDV_fake = mma_pdo.make_fragment_C(mma_pdo.partition_shape_C((TILE, HEAD_DIM)))
            tDK_fake = mma_dsq.make_fragment_C(mma_dsq.partition_shape_C((TILE, HEAD_DIM)))
            tDQ_fake = mma_dq.make_fragment_C(mma_dq.partition_shape_C((TILE, HALF)))
            tDVrP_layout = mma_pdo.make_fragment_A(p_layout.outer).layout  # (MMA, M, K=8, 1)

            scale_log2 = sm_scale * LOG2E

            # ============================================================ LOAD WARP
            if warp_idx == LOAD_WARP:
                _set_regs(REGS_WG3)
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_end(setup_tok)
                lane = tidx % 32
                # init the 9th-k-step blocks: zeros everywhere, ones in columns 0,1 of sOnes (row r of M-core g at g*64 + r*8)
                # K, V once (2 d-blocks each)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(kv_bar, kv_bytes)
                for db in cutlass.range_constexpr(2):
                    cute.copy(k_atom, tKgK_tma[(None, db)], tKsK[(None, db)], tma_bar_ptr=kv_bar)
                    cute.copy(v_atom, tVgV_tma[(None, db)], tVsV[(None, db)], tma_bar_ptr=kv_bar)
                cute.copy(cb_atom, tCBg, tCBs, tma_bar_ptr=kv_bar)
                for it in cutlass.range(n_iters, unroll=1):
                    iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                    if cutlass.const_expr(EXP_SAMEQ):
                        iq = jblk
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
                q_prod.tail()
                do_prod.tail()

            # ============================================================ MMA WARP A: S^T, dP^T, dQ^T
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
                sK_all = cute.group_modes(sK, 2, 4)        # (MMA, M, (K=4, 2 d-blocks)) -> 8 k-blocks
                sV_all = cute.group_modes(sV, 2, 4)
                K8 = list(range(NK8))

                _rpush("mma_wait_kv")
                cute.arch.mbarrier_wait(kv_bar, 0)
                _rpop()

                # ---- prologue: S_a(0), dP_a(0), S_b(0), dP_b(0)
                _rpush("mma_pro")
                for hx in cutlass.range_constexpr(2):
                    qh = q_cons.wait_and_advance()
                    _rpush("mma_wait_s_empty")
                    sh = s_prod.acquire_and_advance()
                    _rpop()
                    gemm_ptx(self.op_s, tS, sK_all, sQ[(None, None, None, qh.index)], None, 0, K8, K8)
                    mma_tail_ptx(self.op_s, tS, ones_lo, ((qx_addr + qh.index * XBLK_BYTES) >> 4) | (((zero_addr - qx_addr - qh.index * XBLK_BYTES) >> 4) << 16))
                    sh.commit()
                    with cute.arch.elect_one():
                        tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: S done (1 of 2)
                    dh = do_cons.wait_and_advance()
                    _rpush("mma_wait_dp_empty")
                    dph = dp_prod.acquire_and_advance()
                    _rpop()
                    gemm_ptx(self.op_s, tDP, sV_all, sDO[(None, None, None, dh.index)], None, 0, K8, K8)
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
                            gemm_ptx(self.op_s, tS, sK_all, sQ[(None, None, None, qh.index)], None, 0, K8, K8)
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
                            gemm_ptx(self.op_s, tDP, sV_all, sDO[(None, None, None, dh.index)], None, 0, K8, K8)
                            mma_tail_ptx(self.op_s, tDP, ones_lo, ((dox_addr + dh.index * XBLK_BYTES) >> 4) | (((zero_addr - dox_addr - dh.index * XBLK_BYTES) >> 4) << 16))
                            dph.commit()
                            with cute.arch.elect_one():
                                tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)
                            _rpop()
                # wait for all TMEM readers then free
                _rpush("mma_fin")
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
                _rpop()

            # ============================================================ MMA WARP B: dV, dK
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
                _rpush("mmb_wait_kv")
                cute.arch.mbarrier_wait(kv_bar, 0)
                _rpop()
                for it in cutlass.range(n_iters, unroll=1):
                    for hx in cutlass.range_constexpr(2):
                        _rpush("mmb_wait_p")
                        ph = p_cons.wait_and_advance()
                        _rpop()
                        dh = do_cons2.wait_and_advance()
                        _rpush("mmb_dV")
                        gemm_ptx(self.op_pdo, tDV, None, sDOT[(None, None, None, dh.index)], tmem_p + hx * 32,
                                 cutlass.Int32((it != 0) or (hx != 0)), K4, K4)
                        ph.release()
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.do_bars.data_ptr() + DO_STAGES + dh.index)     # dO slot: dV done (2 of 2)
                        _rpop()
                        _rpush("mmb_wait_ds")
                        dsh = ds_cons.wait_and_advance()
                        _rpop()
                        qh = q_cons2.wait_and_advance()
                        _rpush("mmb_dK")
                        gemm_ptx(self.op_dsq, tDK, sDS0, sQT[(None, None, None, qh.index)], None,
                                 cutlass.Int32((it != 0) or (hx != 0)), [hx * 4 + kk for kk in K4], K4)
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.q_bars.data_ptr() + Q_STAGES + qh.index)       # Q slot: dK done (2 of 2)
                        _rpop()
                        _rpush("mmb_wait_dq_empty")
                        dqh = dq_prod.acquire_and_advance()
                        _rpop()
                        _rpush("mmb_dQ")
                        gemm_ptx(self.op_dq, tDQ, sKT0, sDST[(None, None, None, hx)], None, 0, K8, K8)
                        dqh.commit()
                        with cute.arch.elect_one():
                            tcgen05.commit(storage.ds_bars.data_ptr() + 2 + hx)                  # dS stage x free (dK_x, dQ_x done)
                        _rpop()
                # ---- tail: dV and dK ready
                kh = dkdv_prod.acquire_and_advance()
                kh.commit()                                  # dV ready
                kh2 = dkdv_prod.acquire_and_advance()
                kh2.commit()                                 # dK ready
                _rpush("mmb_fin")
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
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
                tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
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
                kv_row = kv0 + row0
                kv_valid = kv_row < seqlen
                vec_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bf16, num_bits_per_copy=128)
                vec_layout = cute.make_layout((8, 4), stride=(1, 8))      # 32 bf16 as 4 x 16B
                kv_tail = (kv0 + TILE) > seqlen
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
                                    chunk = hx * 2 + c
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

                # ---- epilogue: dV then dK (both warpgroups; chunk = wg*2 + half)
                rO = cute.make_rmem_tensor(frag_shape, f32)
                tDV2d = tDV[((None, None), 0, 0)]
                tDK2d = tDK[((None, None), 0, 0)]
                tDV_chunks = cute.local_tile(tDV2d, chunk_tiler, (0, None))
                tDK_chunks = cute.local_tile(tDK2d, chunk_tiler, (0, None))
                tTR_tDV = thr_ld.partition_S(tDV_chunks)
                tTR_tDK = thr_ld.partition_S(tDK_chunks)
                kv_tok = seq_start + kv_row
                _rpush("cmp_epi_wait")
                kh = dkdv_cons.wait_and_advance()
                _rpop()
                _rpush("cmp_epi")
                for half in cutlass.range_constexpr(2):
                    chunk = wg * 2 + half
                    cute.copy(tiled_ld, tTR_tDV[(None, None, None, chunk)], rO)
                    rOb = cute.make_rmem_tensor((32,), bf16)
                    rOb.store(rO.load().to(bf16))
                    if kv_valid:
                        rOv = cute.make_tensor(rOb.iterator, vec_layout)
                        dstv = cute.make_tensor((mDV.iterator + mDV.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                        cute.copy(vec_atom, rOv, dstv)
                cute.arch.fence_view_async_tmem_load()
                kh.release()
                kh2 = dkdv_cons.wait_and_advance()
                for half in cutlass.range_constexpr(2):
                    chunk = wg * 2 + half
                    cute.copy(tiled_ld, tTR_tDK[(None, None, None, chunk)], rO)
                    rOb = cute.make_rmem_tensor((32,), bf16)
                    rOb.store((rO.load() * sm_scale).to(bf16))
                    if kv_valid:
                        rOv = cute.make_tensor(rOb.iterator, vec_layout)
                        dstv = cute.make_tensor((mDK.iterator + mDK.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                        cute.copy(vec_atom, rOv, dstv)
                cute.arch.fence_view_async_tmem_load()
                kh2.release()
                _rpop()
                _rpush("cmp_fin")
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
                _rpop()

            # ============================================================ REDUCE WARPS (dQ^T halves)
            elif warp_idx <= REDUCE_WARPS[-1]:
                _set_regs(REGS_REDUCE)
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_end(setup_tok)
                tmem.allocate(512)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(f32)
                tDQ = cute.make_tensor(tmem_ptr + TMEM_DQ, tDQ_fake.layout)
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
                # (128 d, 64 q) view of the staging buffer (same bytes as the 4 per-warp (32, 64) boxes)
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
                rDQc0 = cute.make_rmem_tensor(tRS_rDQ0.shape, cutlass.Float16)
                rDQc1 = cute.make_rmem_tensor(tRS_rDQ1.shape, cutlass.Float16)
                tDQsDQ, tDQgDQ = cpasync.tma_partition(dq_atom, 0, cute.make_layout(1),
                                                       cute.group_modes(sDQ, 0, 2), cute.group_modes(gDQ, 0, 2))
                store_pipe = pipeline.PipelineTmaStore.create(num_stages=1, producer_group=load_threads)
                for it in cutlass.range(n_iters, unroll=1):
                    iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                    for hx in cutlass.range_constexpr(2):
                        _rpush("red_wait_dq")
                        dqh = dq_cons.wait_and_advance()
                        _rpop()
                        if cutlass.const_expr(EXP_NOREDUCE):
                            dqh.release()
                        else:
                            _rpush("red_tmem_ld")
                            cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 0)], rDQ0)
                            cute.copy(tiled_ld, tTR_tDQ[(None, None, None, 1)], rDQ1)
                            cute.arch.fence_view_async_tmem_load()
                            dqh.release()
                            _rpop()
                            rDQc0.store(tRS_rDQ0.load().to(cutlass.Float16))
                            rDQc1.store(tRS_rDQ1.load().to(cutlass.Float16))
                            for c in cutlass.range_constexpr(NBOX):
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
                _rpush("red_tail")
                store_pipe.producer_tail()
                _rpop()
                _rpush("red_fin")
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (2 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
                _rpop()
            else:
                # idle warp 15: participates in the warpgroup register deallocation only
                _set_regs(REGS_WG3)
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_end(setup_tok)


# ---------------------------------------------------------------------------------------- host wrapper
_KERNEL = None
_COMPILED = {}


def _as_tdh(t):
    """(T, H, D) torch tensor -> cute (T, D, H) view."""
    return from_dlpack(t.permute(0, 2, 1), assumed_align=16).mark_layout_dynamic(leading_dim=1)


def attention_bwd_prepare(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, tile_table, max_tiles, sm_scale, xq, xdo, cb):
    """Host-side argument preparation (DLPack conversions); call before launching any kernel."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = AttnBwdSm100V8()
    T, H, D = q.shape
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_as_tdh(q), _as_tdh(k), _as_tdh(v), _as_tdh(dout),
            from_dlpack(lse, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(delta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dq_accum.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1),   # (D, T, H)
            _as_tdh(dk), _as_tdh(dv),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            from_dlpack(tile_table, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            cutlass.Int32(max_tiles), cutlass.Int32(T), cutlass.Float32(sm_scale),
            from_dlpack(xq.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),      # (8, T, H)
            from_dlpack(xdo.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(cb.view(192, 8).permute(1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0),  # (8, 192)
            stream)
    key = "main"
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_KERNEL, *args)
        _COMPILED[key] = fn
    return fn, args


def attention_bwd_launch(prepared):
    fn, args = prepared
    fn(*args)


def attention_bwd_main(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, max_seqlen, sm_scale):
    """Standalone entry (builds the tile table itself)."""
    from bwd_triton import _build_tile_table
    T, H, D = q.shape
    num_seqs = cu_seqlens.numel() - 1
    max_tiles = (T + TILE - 1) // TILE + num_seqs
    table = torch.empty(max_tiles, dtype=torch.int32, device=q.device)
    prepared = attention_bwd_prepare(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, table, max_tiles, sm_scale)
    _build_tile_table[(1,)](cu_seqlens, num_seqs, table, max_tiles, BLOCK_N=TILE,
                            MAX_NBLK=(int(max_seqlen) + TILE - 1) // TILE, NSEQ_PAD=max(16, triton.next_power_of_2(num_seqs + 1)), num_warps=1)
    attention_bwd_launch(prepared)
