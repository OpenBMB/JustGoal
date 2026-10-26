"""CuTe DSL (Blackwell sm_100a) warp-specialized attention backward main kernel.

Design (per CTA = one 128-row kv block of one sequence, one head; grid (H, num_seqs, max_kv_blocks)):
  TMEM (512 cols): dK[0:128] | dV[128:256] | dQ = dP [256:384] | S = P [384:512]
  warps 0-3   : reduce  (dQ: TMEM -> SMEM -> TMA reduce-add into fp32 dq_accum)
  warps 4-11  : compute (P = exp2(S*scale*log2e - lse*log2e) masked, P->TMEM bf16, dS = P*(dP-delta) -> SMEM bf16)
  warp 12     : MMA issue (S^T=K Q^T, dP^T=V dO^T, dV+=P^T dO, dK+=dS^T Q, dQ=dS K)
  warp 13     : TMA loads (K,V once; Q 2 stages; dO 1 stage) + LSE/delta vectors
  warps 14-15 : idle
Reference structure: CUTLASS examples/77_blackwell_fmha (sm100_fmha_bwd_kernel_tma_warpspecialized.hpp).
"""

import math

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

TILE = 128          # kv rows (MMA M) == q cols (MMA N) per iteration
HEAD_DIM = 128
import os as _os0
DQ_COLS = int(_os0.environ.get("ATT_DQ_COLS", "64"))      # dQ staging chunk (columns of d) per TMA reduce
DQ_STAGES = int(_os0.environ.get("ATT_DQ_STAGES", "2"))   # staging buffers (DQ_COLS*TILE*4 bytes each)
RED_PRELOAD = _os0.environ.get("ATT_RED_PRELOAD", "0") == "1"  # load the whole dQ tile to registers before staging
Q_REVERSE = _os0.environ.get("ATT_QREV", "1") == "1"   # iterate q blocks from the sequence end down to the diagonal (L2 locality)
MMA_ORDER = int(_os0.environ.get("ATT_MMA_ORDER", "1"))
DQ_F16 = _os0.environ.get("ATT_DQ_F16", "1") == "1"      # accumulate dQ partials in fp16 (halves reduce traffic)
NUM_WARPS = 16
NUM_THREADS = NUM_WARPS * 32
REDUCE_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7, 8, 9, 10, 11)
MMA_WARP = 12
LOAD_WARP = 13
NUM_COMPUTE_THREADS = 256
NUM_REDUCE_THREADS = 128

TMEM_DK = 0
TMEM_DV = 128
TMEM_DQ = 256
TMEM_DP = 256
TMEM_S = 384
TMEM_P = 384

LOG2E = 1.4426950408889634

import os as _os
IKET = _os.environ.get("ATT_IKET", "0") == "1"
REDUCE_MODE = _os.environ.get("ATT_REDUCE", "tma")
EXP_NOEXP = _os.environ.get("ATT_NOEXP", "0") == "1"      # experiment: skip MUFU exp2
EXP_NOPSTORE = _os.environ.get("ATT_NOPSTORE", "0") == "1"  # experiment: skip P TMEM store + barrier
EXP_NOTMEMLD = _os.environ.get("ATT_NOTMEMLD", "0") == "1"  # experiment: skip S TMEM loads
EXP_NOLDS = _os.environ.get("ATT_NOLDS", "0") == "1"        # experiment: skip LSE smem loads
EXP_NOCOMPUTE = _os.environ.get("ATT_NOCOMPUTE", "0") == "1"  # experiment: compute warps only run the pipeline skeleton
EXP_NOREDUCE = _os.environ.get("ATT_NOREDUCE", "0") == "1"    # experiment: reduce warps only run the pipeline skeleton
MBAR_TICKS = int(_os.environ.get("ATT_MBAR_TICKS", "-1"))     # <0: DSL default (10 ms suspend hint); >=0: custom suspend hint (ns)

if MBAR_TICKS >= 0:
    from cutlass._mlir.dialects import nvvm as _nvvm
    from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
    import cutlass.cute.arch.mbar as _mbar_mod

    @_dsl_user_op
    def _fast_mbarrier_wait(mbar_ptr, phase, *, loc=None, ip=None):
        _nvvm.mbarrier_try_wait_parity(
            mbar_ptr.to_llvm_ptr(loc=loc, ip=ip),
            cutlass.Int32(phase).ir_value(loc=loc, ip=ip),
            cutlass.Int32(MBAR_TICKS).ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip)

    cute.arch.mbarrier_wait = _fast_mbarrier_wait
    _mbar_mod.mbarrier_wait = _fast_mbarrier_wait   # "red": red.global.add.v4.f32 from registers; "tma": smem + TMA reduce-add


@cute.jit
def red_add_v4_f32(ptr_i64: cutlass.Int64, a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, d: cutlass.Float32):
    llvm.inline_asm(
        None,
        [ptr_i64.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "red.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _rpush(name):
    if IKET:
        cute.experimental.iket.range_push(name)


def _rpop():
    if IKET:
        cute.experimental.iket.range_pop()

# named barrier ids
BAR_TMEM_ALLOC = 1
BAR_COMPUTE = 2
BAR_REDUCE = 3
BAR_FINAL = 4

# setmaxnreg budget (kernel compiled at 128 regs/thread x 512 threads = 64K)
REGS_WG3 = int(_os0.environ.get("ATT_REGS_WG3", "56"))        # mma / load / idle warps (12-15)
REGS_REDUCE = int(_os0.environ.get("ATT_REGS_REDUCE", "104"))  # warps 0-3
REGS_COMPUTE = int(_os0.environ.get("ATT_REGS_COMPUTE", "176"))  # warps 4-11
assert REGS_WG3 * 128 + REGS_REDUCE * 128 + REGS_COMPUTE * 256 <= 65536


def _set_regs(n):
    """setmaxnreg relative to the 128-register launch value (inc above, dec below)."""
    if n > 128:
        cute.arch.warpgroup_reg_alloc(n)
    elif n < 128:
        cute.arch.warpgroup_reg_dealloc(n)

MMA_MODE = _os.environ.get("ATT_MMA", "dsl")   # "ptx": inline tcgen05.mma with precomputed descriptors; "dsl": cute.gemm


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


def _asm_smem_a(hi_a, hi_b, offs_a, offs_b, idesc, nk):
    asm = ("{\n\t.reg .pred lt;\n\t.reg .pred p;\n\t.reg .b32 alo, blo;\n\t.reg .b64 da, db;\n\t"
           "elect.sync _|lt, -1;\n\tmov.b32 alo, $1;\n\tmov.b32 blo, $2;\n\tsetp.ne.b32 p, $3, 0;\n\t"
           f"mov.b64 da, {{alo, {hex(hi_a)}}};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
           f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, {hex(idesc)}, p;\n\t")
    for k in range(1, nk):
        asm += (f"add.u32 alo, alo, {hex(offs_a[k] - offs_a[k-1])};\n\tadd.u32 blo, blo, {hex(offs_b[k] - offs_b[k-1])};\n\t"
                f"mov.b64 da, {{alo, {hex(hi_a)}}};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
                f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, {hex(idesc)}, 1;\n\t")
    return asm + "}\n"


def _asm_tmem_a(hi_b, offs_b, idesc, nk):
    asm = ("{\n\t.reg .pred lt;\n\t.reg .pred p;\n\t.reg .b32 blo;\n\t.reg .b64 db;\n\t"
           "elect.sync _|lt, -1;\n\tmov.b32 blo, $2;\n\tsetp.ne.b32 p, $3, 0;\n\t"
           f"mov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
           f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], db, {hex(idesc)}, p;\n\t")
    for k in range(1, nk):
        asm += (f"add.u32 blo, blo, {hex(offs_b[k] - offs_b[k-1])};\n\tmov.b64 db, {{blo, {hex(hi_b)}}};\n\t"
                f"@lt tcgen05.mma.cta_group::1.kind::f16 [$0], [$1 + {hex(8 * k)}], db, {hex(idesc)}, 1;\n\t")
    return asm + "}\n"


@cute.jit
def gemm_ptx(op, acc: cute.Tensor, sA_stage, sB_stage, tmem_a, accumulate):
    """Issue all k-blocks of one 128x128x(K) GEMM with tcgen05.mma from the elected lane.
    sA_stage/sB_stage: smem tensors of ONE pipeline stage with layout (MMA, M|N, K); tmem_a: Int32 TMEM address of A (or None).
    accumulate: Int32/Boolean (0 => overwrite on the first k-block)."""
    is_ts = tmem_a is not None
    idesc = _umma_idesc(op)
    b_kmajor = op.b_major_mode == cute.nvgpu.OperandMajorMode.K
    base_b = _desc_base_of(sB_stage, b_kmajor)
    hi_b = (base_b >> 32) & 0xFFFFFFFF
    lo_b = base_b & 0xFFFFFFFF
    offs_b = _kblock_offsets_16B(sB_stage)
    nk = len(offs_b)
    b_lo = cutlass.Int32(lo_b) | cutlass.Int32((sB_stage.iterator.toint() & 0x3FFFF) >> 4)
    acc_flag = cutlass.Int32(accumulate)
    if cutlass.const_expr(not is_ts):
        a_kmajor = op.a_major_mode == cute.nvgpu.OperandMajorMode.K
        base_a = _desc_base_of(sA_stage, a_kmajor)
        hi_a = (base_a >> 32) & 0xFFFFFFFF
        lo_a = base_a & 0xFFFFFFFF
        offs_a = _kblock_offsets_16B(sA_stage)
        a_lo = cutlass.Int32(lo_a) | cutlass.Int32((sA_stage.iterator.toint() & 0x3FFFF) >> 4)
        asm = _asm_smem_a(hi_a, hi_b, offs_a, offs_b, idesc, nk)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), a_lo.ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    else:
        asm = _asm_tmem_a(hi_b, offs_b, idesc, nk)
        llvm.inline_asm(None, [acc.iterator.toint().ir_value(), cutlass.Int32(tmem_a).ir_value(), b_lo.ir_value(), acc_flag.ir_value()],
                        asm, "r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.struct
class SharedStorage:
    # pipelines: each needs 2*stages Int64 barriers
    q_bars: cute.struct.MemRange[cutlass.Int64, 4]      # load->mma Q (2 stages)
    do_bars: cute.struct.MemRange[cutlass.Int64, 2]     # load->mma dO (1 stage)
    kv_bar: cute.struct.MemRange[cutlass.Int64, 1]      # K,V loaded (once)
    lse_bars: cute.struct.MemRange[cutlass.Int64, 2]    # load->compute LSE (1 stage)
    delta_bars: cute.struct.MemRange[cutlass.Int64, 2]  # load->compute delta (1 stage)
    s_bars: cute.struct.MemRange[cutlass.Int64, 2]      # mma->compute S
    dp_bars: cute.struct.MemRange[cutlass.Int64, 2]     # mma->compute dP
    dq_bars: cute.struct.MemRange[cutlass.Int64, 2]     # mma->reduce dQ
    p_bars: cute.struct.MemRange[cutlass.Int64, 2]      # compute->mma P
    ds_bars: cute.struct.MemRange[cutlass.Int64, 2]     # compute->mma dS
    dkdv_bars: cute.struct.MemRange[cutlass.Int64, 4]   # mma->compute dK/dV ready (2 stages)
    tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
    lse: cute.struct.MemRange[cutlass.Float32, TILE]
    delta: cute.struct.MemRange[cutlass.Float32, TILE]


class AttnBwdSm100:
    def __init__(self):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        K, MN = cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN
        shape = (TILE, TILE, 16)
        cg = tcgen05.CtaGroup.ONE
        # S^T = K Q^T and dP^T = V dO^T : A (K/V) K-major SMEM, B (Q/dO) K-major SMEM
        self.op_kq = tcgen05.MmaF16BF16Op(bf16, f32, shape, cg, tcgen05.OperandSource.SMEM, K, K)
        # dV += P^T dO : A (P^T) from TMEM K-major, B (dO) MN-major view
        self.op_pdo = tcgen05.MmaF16BF16Op(bf16, f32, shape, cg, tcgen05.OperandSource.TMEM, K, MN)
        # dK += dS^T Q : A (dS^T) K-major SMEM, B (Q) MN-major view
        self.op_dsq = tcgen05.MmaF16BF16Op(bf16, f32, shape, cg, tcgen05.OperandSource.SMEM, K, MN)
        # dQ = dS K : A (dS) MN-major view of dS^T buffer, B (K) MN-major view
        self.op_dsk = tcgen05.MmaF16BF16Op(bf16, f32, shape, cg, tcgen05.OperandSource.SMEM, MN, MN)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mDO: cute.Tensor,
                 mLSE: cute.Tensor, mDelta: cute.Tensor, mDQacc: cute.Tensor,
                 mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor, tile_table: cute.Tensor,
                 max_tiles: cutlass.Int32, total_tokens: cutlass.Int32,
                 sm_scale: cutlass.Float32, stream: cuda.CUstream):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        tiler = (TILE, TILE, HEAD_DIM)
        mma_kq = cute.make_tiled_mma(self.op_kq)
        mma_pdo = cute.make_tiled_mma(self.op_pdo)
        mma_dsq = cute.make_tiled_mma(self.op_dsq)
        mma_dsk = cute.make_tiled_mma(self.op_dsk)

        # SMEM layouts (K-major views for TMA loads)
        k_layout = sm100_utils.make_smem_layout_a(mma_kq, tiler, bf16, 1)     # (MMA,M,K,1) K rows x d
        q_layout = sm100_utils.make_smem_layout_b(mma_kq, tiler, bf16, 2)     # q rows x d, 2 stages
        do_layout = sm100_utils.make_smem_layout_b(mma_kq, tiler, bf16, 1)
        # MN-major views of the same buffers (operands of dK / dQ / dV GEMMs)
        qT_layout = sm100_utils.make_smem_layout_b(mma_dsq, tiler, bf16, 2)   # B: n=d, k=q
        kT_layout = sm100_utils.make_smem_layout_b(mma_dsk, tiler, bf16, 1)   # B: n=d, k=kv
        doT_layout = sm100_utils.make_smem_layout_b(mma_pdo, tiler, bf16, 1)  # B: n=d, k=q
        ds_layout = sm100_utils.make_smem_layout_a(mma_dsq, tiler, bf16, 1)   # A: m=kv, k=q (K-major)
        dsT_layout = sm100_utils.make_smem_layout_a(mma_dsk, tiler, bf16, 1)  # A: m=q, k=kv (MN-major)
        p_layout = sm100_utils.make_smem_layout_a(mma_pdo, tiler, bf16, 1)    # TMEM A layout for P^T

        # dQ staging for TMA reduce: (TILE, DQ_COLS) fp32 row-major, 2 stages
        dq_dtype = cutlass.Float16 if DQ_F16 else f32
        dq_epi_layout = sm100_utils.make_smem_layout_epi(dq_dtype, utils.LayoutEnum.ROW_MAJOR, (TILE, DQ_COLS), DQ_STAGES)

        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_atom, k_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), tiler, mma_kq)
        v_atom, v_tma = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mV, cute.slice_(k_layout, (None, None, None, 0)), tiler, mma_kq)
        q_atom, q_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), tiler, mma_kq)
        do_atom, do_tma = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), tiler, mma_kq)
        dq_red_atom, dq_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionKind.ADD), mDQacc,
            cute.slice_(dq_epi_layout, (None, None, 0)), (TILE, DQ_COLS))

        if cutlass.const_expr(False):
            print("q_layout", q_layout)
            print("qT_layout", qT_layout)
            print("ds_layout", ds_layout)
            print("dsT_layout", dsT_layout)
            print("dq_epi_layout", dq_epi_layout)

        H = mQ.shape[2]
        self.kernel(mma_kq, mma_pdo, mma_dsq, mma_dsk,
                    k_atom, k_tma, v_atom, v_tma, q_atom, q_tma, do_atom, do_tma, dq_red_atom, dq_tma, mDQacc,
                    mLSE, mDelta, mDK, mDV, cu_seqlens, tile_table,
                    k_layout, q_layout, do_layout, qT_layout, kT_layout, doT_layout, ds_layout, dsT_layout,
                    p_layout, dq_epi_layout, total_tokens, sm_scale).launch(
            grid=(H, max_tiles, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    @cute.jit
    def p_chunk(self, rS, rPb, chunk, need_mask, qbase, kv_row, kv_valid, seqlen, sLSE4, scale_log2):
        """P = exp2(S*scale*log2e - lse*log2e) for one 32-column chunk held in rS (fp32) -> rPb (bf16)."""
        if need_mask:
            for quad in cutlass.range_constexpr(8):
                lv = sLSE4[(None, chunk * 8 + quad)].load()
                for kq in cutlass.range_constexpr(4):
                    i = quad * 4 + kq
                    q_pos = qbase + chunk * 32 + i
                    p = cute.math.exp2(rS[i] * scale_log2 - lv[kq], fastmath=True)
                    ok = (kv_row <= q_pos) and (q_pos < seqlen) and kv_valid
                    rS[i] = p if ok else cutlass.Float32(0.0)
        else:
            for quad in cutlass.range_constexpr(8):
                lv = sLSE4[(None, chunk * 8 + quad)].load()
                for kq in cutlass.range_constexpr(4):
                    i = quad * 4 + kq
                    rS[i] = cute.math.exp2(rS[i] * scale_log2 - lv[kq], fastmath=True)
        rPb.store(rS.load().to(cutlass.BFloat16))

    @cute.jit
    def p_store(self, rPb, p_frag_shape, tiled_st, tRT_tP, chunk):
        rP16 = cute.make_tensor(cute.recast_tensor(rPb, cutlass.Float32).iterator, cute.make_layout(p_frag_shape))
        cute.copy(tiled_st, rP16, tRT_tP[(None, None, None, chunk)])

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, mma_kq: cute.TiledMma, mma_pdo: cute.TiledMma, mma_dsq: cute.TiledMma, mma_dsk: cute.TiledMma,
               k_atom: cute.CopyAtom, mK: cute.Tensor, v_atom: cute.CopyAtom, mV: cute.Tensor,
               q_atom: cute.CopyAtom, mQ: cute.Tensor, do_atom: cute.CopyAtom, mDO: cute.Tensor,
               dq_atom: cute.CopyAtom, mDQ: cute.Tensor, mDQp: cute.Tensor,
               mLSE: cute.Tensor, mDelta: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, cu_seqlens: cute.Tensor,
               tile_table: cute.Tensor, k_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               qT_layout: cute.ComposedLayout, kT_layout: cute.ComposedLayout, doT_layout: cute.ComposedLayout,
               ds_layout: cute.ComposedLayout, dsT_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               dq_epi_layout: cute.ComposedLayout, total_tokens: cutlass.Int32, sm_scale: cutlass.Float32):
        bf16, f32 = cutlass.BFloat16, cutlass.Float32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        head, tile_idx, _ = cute.arch.block_idx()
        tiler = (TILE, TILE, HEAD_DIM)

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
            sK = smem.allocate_tensor(bf16, k_layout.outer, 1024, swizzle=k_layout.inner)
            sV = smem.allocate_tensor(bf16, k_layout.outer, 1024, swizzle=k_layout.inner)
            sQ = smem.allocate_tensor(bf16, q_layout.outer, 1024, swizzle=q_layout.inner)
            sDO = smem.allocate_tensor(bf16, do_layout.outer, 1024, swizzle=do_layout.inner)
            sDS = smem.allocate_tensor(bf16, ds_layout.outer, 1024, swizzle=ds_layout.inner)
            sDQ = smem.allocate_tensor(cutlass.Float16 if DQ_F16 else f32, dq_epi_layout.outer, 1024, swizzle=dq_epi_layout.inner)
            # alias views (same bytes, different canonical layouts)
            sQT = cute.make_tensor(cute.recast_ptr(sQ.iterator, qT_layout.inner), qT_layout.outer)
            sKT = cute.make_tensor(cute.recast_ptr(sK.iterator, kT_layout.inner), kT_layout.outer)
            sDOT = cute.make_tensor(cute.recast_ptr(sDO.iterator, doT_layout.inner), doT_layout.outer)
            sDST = cute.make_tensor(cute.recast_ptr(sDS.iterator, dsT_layout.inner), dsT_layout.outer)
            sLSE = storage.lse.get_tensor(cute.make_layout((TILE,)))
            sDelta = storage.delta.get_tensor(cute.make_layout((TILE,)))

            if warp_idx == LOAD_WARP:
                cpasync.prefetch_descriptor(k_atom)
                cpasync.prefetch_descriptor(v_atom)
                cpasync.prefetch_descriptor(q_atom)
                cpasync.prefetch_descriptor(do_atom)
                cpasync.prefetch_descriptor(dq_atom)

            # ---------------- pipelines
            q_bytes = cute.size_in_bytes(bf16, cute.select(q_layout, mode=[0, 1, 2]))
            do_bytes = cute.size_in_bytes(bf16, cute.select(do_layout, mode=[0, 1, 2]))
            kv_bytes = 2 * cute.size_in_bytes(bf16, cute.select(k_layout, mode=[0, 1, 2]))
            one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
            load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=32)
            compute_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_COMPUTE_THREADS)
            reduce_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=NUM_REDUCE_THREADS)

            q_prod, q_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.q_bars.data_ptr(), num_stages=2, producer_group=one_thread,
                consumer_group=one_thread, tx_count=q_bytes, defer_sync=True).make_participants()
            do_prod, do_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.do_bars.data_ptr(), num_stages=1, producer_group=one_thread,
                consumer_group=one_thread, tx_count=do_bytes, defer_sync=True).make_participants()
            lse_prod, lse_cons = pipeline.PipelineAsync.create(
                barrier_storage=storage.lse_bars.data_ptr(), num_stages=1, producer_group=load_threads,
                consumer_group=compute_threads, defer_sync=True).make_participants()
            delta_prod, delta_cons = pipeline.PipelineAsync.create(
                barrier_storage=storage.delta_bars.data_ptr(), num_stages=1, producer_group=load_threads,
                consumer_group=compute_threads, defer_sync=True).make_participants()
            s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.s_bars.data_ptr(), num_stages=1, producer_group=one_thread,
                consumer_group=compute_threads, defer_sync=True).make_participants()
            dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.dp_bars.data_ptr(), num_stages=1, producer_group=one_thread,
                consumer_group=compute_threads, defer_sync=True).make_participants()
            dq_prod, dq_cons = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.dq_bars.data_ptr(), num_stages=1, producer_group=one_thread,
                consumer_group=reduce_threads, defer_sync=True).make_participants()
            p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.p_bars.data_ptr(), num_stages=1, producer_group=compute_threads,
                consumer_group=one_thread, defer_sync=True).make_participants()
            ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.ds_bars.data_ptr(), num_stages=1, producer_group=compute_threads,
                consumer_group=one_thread, defer_sync=True).make_participants()
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
                                                           num_threads=32 * (1 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))),
                allocator_warp_id=MMA_WARP)
            cute.arch.barrier()  # all 512 threads: barriers initialized

            # ---------------- tensors / fragments (shared by roles; cheap)
            mK_h = cute.domain_offset((seq_start + kv0, 0), mK[(None, None, head)])
            mV_h = cute.domain_offset((seq_start + kv0, 0), mV[(None, None, head)])
            mQ_h = cute.domain_offset((seq_start, 0), mQ[(None, None, head)])
            mDO_h = cute.domain_offset((seq_start, 0), mDO[(None, None, head)])
            mDQ_h = cute.domain_offset((seq_start, 0), mDQ[(None, None, head)])
            gK = cute.local_tile(mK_h, tiler, (0, 0, None), proj=(1, None, 1))      # (TILE, D, 1)
            gV = cute.local_tile(mV_h, tiler, (0, 0, None), proj=(1, None, 1))
            gQ = cute.local_tile(mQ_h, tiler, (0, None, 0), proj=(None, 1, 1))      # (TILE, D, nQblk)
            gDO = cute.local_tile(mDO_h, tiler, (0, None, 0), proj=(None, 1, 1))
            gDQ = cute.local_tile(mDQ_h, (TILE, DQ_COLS), (None, None))              # (TILE, 64, nQblk, 2)

            thr_kq = mma_kq.get_slice(0)
            tKgK = thr_kq.partition_A(gK)
            tVgV = thr_kq.partition_A(gV)
            tQgQ = thr_kq.partition_B(gQ)
            tDOgDO = thr_kq.partition_B(gDO)
            tKsK, tKgK_tma = cpasync.tma_partition(k_atom, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tKgK, 0, 3))
            tVsV, tVgV_tma = cpasync.tma_partition(v_atom, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tVgV, 0, 3))
            tQsQ, tQgQ_tma = cpasync.tma_partition(q_atom, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tQgQ, 0, 3))
            tDOsDO, tDOgDO_tma = cpasync.tma_partition(do_atom, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(tDOgDO, 0, 3))

            # MMA fragments
            tSrK = mma_kq.make_fragment_A(sK)       # (MMA, M, K, 1)
            tSrQ = mma_kq.make_fragment_B(sQ)       # (MMA, N, K, 2)
            tDPrV = mma_kq.make_fragment_A(sV)
            tDPrDO = mma_kq.make_fragment_B(sDO)
            tDVrDOT = mma_pdo.make_fragment_B(sDOT)
            tDKrDST = mma_dsq.make_fragment_A(sDS)
            tDKrQT = mma_dsq.make_fragment_B(sQT)
            tDQrDS = mma_dsk.make_fragment_A(sDST)
            tDQrKT = mma_dsk.make_fragment_B(sKT)

            acc_shape = mma_kq.partition_shape_C((TILE, TILE))
            tS_fake = mma_kq.make_fragment_C(acc_shape)
            tDV_fake = mma_pdo.make_fragment_C(mma_pdo.partition_shape_C((TILE, HEAD_DIM)))
            tDK_fake = mma_dsq.make_fragment_C(mma_dsq.partition_shape_C((TILE, HEAD_DIM)))
            tDQ_fake = mma_dsk.make_fragment_C(mma_dsk.partition_shape_C((TILE, HEAD_DIM)))
            tDVrP_layout = mma_pdo.make_fragment_A(p_layout.outer).layout  # (MMA, M, K, 1)

            scale_log2 = sm_scale * LOG2E

            # ============================================================ LOAD WARP
            if warp_idx == LOAD_WARP:
                _set_regs(REGS_WG3)
                lane = tidx % 32
                # K, V once
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(kv_bar, kv_bytes)
                cute.copy(k_atom, tKgK_tma[(None, 0)], tKsK[(None, 0)], tma_bar_ptr=kv_bar)
                cute.copy(v_atom, tVgV_tma[(None, 0)], tVsV[(None, 0)], tma_bar_ptr=kv_bar)
                for it in cutlass.range(n_iters, unroll=1):
                    iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                    _rpush("ld_wait_q_empty")
                    qh = q_prod.acquire_and_advance()
                    _rpop()
                    cute.copy(q_atom, tQgQ_tma[(None, iq)], tQsQ[(None, qh.index)], tma_bar_ptr=qh.barrier)
                    _rpush("ld_wait_lse_empty")
                    lh = lse_prod.acquire_and_advance()
                    _rpop()
                    _rpush("ld_lse")
                    for k in cutlass.range_constexpr(4):
                        c = lane * 4 + k
                        tok = seq_start + iq * TILE + c
                        val = cutlass.Float32(0.0)
                        if tok < total_tokens:
                            val = mLSE[(head, tok)] * LOG2E
                        sLSE[c] = val
                    lh.commit()
                    _rpop()
                    _rpush("ld_wait_do_empty")
                    dh = do_prod.acquire_and_advance()
                    _rpop()
                    cute.copy(do_atom, tDOgDO_tma[(None, iq)], tDOsDO[(None, dh.index)], tma_bar_ptr=dh.barrier)
                    deh = delta_prod.acquire_and_advance()
                    for k in cutlass.range_constexpr(4):
                        c = lane * 4 + k
                        tok = seq_start + iq * TILE + c
                        val = cutlass.Float32(0.0)
                        if tok < total_tokens:
                            val = mDelta[(head, tok)]
                        sDelta[c] = val
                    deh.commit()
                q_prod.tail()
                do_prod.tail()
                lse_prod.tail()
                delta_prod.tail()

            # ============================================================ MMA WARP
            elif warp_idx == MMA_WARP:
                _set_regs(REGS_WG3)
                tmem.allocate(512)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(f32)
                tS = cute.make_tensor(tmem_ptr + TMEM_S, tS_fake.layout)
                tDP = cute.make_tensor(tmem_ptr + TMEM_DP, tS_fake.layout)
                tDV = cute.make_tensor(tmem_ptr + TMEM_DV, tDV_fake.layout)
                tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
                tDQ = cute.make_tensor(tmem_ptr + TMEM_DQ, tDQ_fake.layout)
                tDVrP = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_P, dtype=bf16), tDVrP_layout)

                num_k = cute.size(tSrK, mode=[2])
                _rpush("mma_wait_kv")
                cute.arch.mbarrier_wait(kv_bar, 0)
                _rpop()
                # ---- first iteration: S, dP, dV
                _rpush("mma_wait_q")
                qh = q_cons.wait_and_advance()
                _rpop()
                q_prev_idx = qh.index
                sh = s_prod.acquire_and_advance()
                _rpush("mma_S")
                if cutlass.const_expr(MMA_MODE == "ptx"):
                    gemm_ptx(self.op_kq, tS, sK[(None, None, None, 0)], sQ[(None, None, None, qh.index)], None, 0)
                else:
                    for kk in cutlass.range_constexpr(num_k):
                        mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(mma_kq, tS, tSrK[(None, None, kk, 0)], tSrQ[(None, None, kk, qh.index)], tS)
                sh.commit()
                _rpop()
                _rpush("mma_wait_do")
                dh = do_cons.wait_and_advance()
                _rpop()
                dph = dp_prod.acquire_and_advance()
                dqh = dq_prod.acquire_and_advance()
                if cutlass.const_expr(MMA_ORDER == 1):
                    _rpush("mma_dP")
                    for kk in cutlass.range_constexpr(num_k):
                        mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(mma_kq, tDP, tDPrV[(None, None, kk, 0)], tDPrDO[(None, None, kk, dh.index)], tDP)
                    dph.commit()
                    _rpop()
                _rpush("mma_wait_p")
                ph = p_cons.wait_and_advance()
                _rpop()
                _rpush("mma_dV")
                for kk in cutlass.range_constexpr(num_k):
                    mma_pdo.set(tcgen05.Field.ACCUMULATE, kk != 0)
                    cute.gemm(mma_pdo, tDV, tDVrP[(None, None, kk, 0)], tDVrDOT[(None, None, kk, dh.index)], tDV)
                ph.release()
                _rpop()
                if cutlass.const_expr(MMA_ORDER == 2):
                    _rpush("mma_dP")
                    for kk in cutlass.range_constexpr(num_k):
                        mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(mma_kq, tDP, tDPrV[(None, None, kk, 0)], tDPrDO[(None, None, kk, dh.index)], tDP)
                    dph.commit()
                    _rpop()
                dh.release()
                # ---- remaining iterations
                for it in cutlass.range(1, n_iters, 1, unroll=1):
                    _rpush("mma_wait_q")
                    qh = q_cons.wait_and_advance()
                    _rpop()
                    _rpush("mma_wait_s_empty")
                    sh = s_prod.acquire_and_advance()
                    _rpop()
                    _rpush("mma_S")
                    if cutlass.const_expr(MMA_MODE == "ptx"):
                        gemm_ptx(self.op_kq, tS, sK[(None, None, None, 0)], sQ[(None, None, None, qh.index)], None, 0)
                    else:
                        for kk in cutlass.range_constexpr(num_k):
                            mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                            cute.gemm(mma_kq, tS, tSrK[(None, None, kk, 0)], tSrQ[(None, None, kk, qh.index)], tS)
                    sh.commit()
                    _rpop()
                    _rpush("mma_wait_ds")
                    dsh = ds_cons.wait_and_advance()
                    _rpop()
                    _rpush("mma_wait_dp_empty")
                    dph = dp_prod.acquire_and_advance()   # compute finished reading dP (same TMEM as dQ)
                    _rpop()
                    _rpush("mma_dQ_dK")
                    if cutlass.const_expr(MMA_MODE == "ptx"):
                        gemm_ptx(self.op_dsk, tDQ, sDST[(None, None, None, 0)], sKT[(None, None, None, 0)], None, 0)
                        dqh.commit()
                        gemm_ptx(self.op_dsq, tDK, sDS[(None, None, None, 0)], sQT[(None, None, None, q_prev_idx)], None, cutlass.Int32(it != 1))
                    else:
                        for kk in cutlass.range_constexpr(num_k):
                            mma_dsk.set(tcgen05.Field.ACCUMULATE, kk != 0)
                            cute.gemm(mma_dsk, tDQ, tDQrDS[(None, None, kk, 0)], tDQrKT[(None, None, kk, 0)], tDQ)
                        dqh.commit()
                        for kk in cutlass.range_constexpr(num_k):
                            mma_dsq.set(tcgen05.Field.ACCUMULATE, (it != 1) or (kk != 0))
                            cute.gemm(mma_dsq, tDK, tDKrDST[(None, None, kk, 0)], tDKrQT[(None, None, kk, q_prev_idx)], tDK)
                    # release previous Q stage and dS (MMA completion tracked by commit on empty barriers)
                    with cute.arch.elect_one():
                        tcgen05.commit(storage.q_bars.data_ptr() + 2 + q_prev_idx)
                    dsh.release()
                    _rpop()
                    q_prev_idx = qh.index
                    if cutlass.const_expr(MMA_ORDER == 1):
                        _rpush("mma_wait_dq_empty")
                        dqh = dq_prod.acquire_and_advance()   # reduce finished reading dQ (same TMEM as dP)
                        _rpop()
                    _rpush("mma_wait_do")
                    dh = do_cons.wait_and_advance()
                    _rpop()
                    if cutlass.const_expr(MMA_ORDER == 1):
                        _rpush("mma_dP")
                        for kk in cutlass.range_constexpr(num_k):
                            mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                            cute.gemm(mma_kq, tDP, tDPrV[(None, None, kk, 0)], tDPrDO[(None, None, kk, dh.index)], tDP)
                        dph.commit()
                        _rpop()
                    _rpush("mma_wait_p")
                    ph = p_cons.wait_and_advance()
                    _rpop()
                    _rpush("mma_dV")
                    for kk in cutlass.range_constexpr(num_k):
                        mma_pdo.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(mma_pdo, tDV, tDVrP[(None, None, kk, 0)], tDVrDOT[(None, None, kk, dh.index)], tDV)
                    ph.release()
                    _rpop()
                    if cutlass.const_expr(MMA_ORDER == 2):
                        _rpush("mma_wait_dq_empty")
                        dqh = dq_prod.acquire_and_advance()   # reduce finished reading dQ (same TMEM as dP)
                        _rpop()
                        _rpush("mma_dP")
                        for kk in cutlass.range_constexpr(num_k):
                            mma_kq.set(tcgen05.Field.ACCUMULATE, kk != 0)
                            cute.gemm(mma_kq, tDP, tDPrV[(None, None, kk, 0)], tDPrDO[(None, None, kk, dh.index)], tDP)
                        dph.commit()
                        _rpop()
                    dh.release()
                # ---- tail: dV done; last dK and dQ
                kh = dkdv_prod.acquire_and_advance()
                kh.commit()                                  # dV ready
                kh2 = dkdv_prod.acquire_and_advance()
                dsh = ds_cons.wait_and_advance()
                if cutlass.const_expr(MMA_MODE == "ptx"):
                    gemm_ptx(self.op_dsq, tDK, sDS[(None, None, None, 0)], sQT[(None, None, None, q_prev_idx)], None, cutlass.Int32(n_iters != 1))
                    kh2.commit()                                 # dK ready
                    gemm_ptx(self.op_dsk, tDQ, sDST[(None, None, None, 0)], sKT[(None, None, None, 0)], None, 0)
                else:
                    for kk in cutlass.range_constexpr(num_k):
                        mma_dsq.set(tcgen05.Field.ACCUMULATE, (n_iters != 1) or (kk != 0))
                        cute.gemm(mma_dsq, tDK, tDKrDST[(None, None, kk, 0)], tDKrQT[(None, None, kk, q_prev_idx)], tDK)
                    kh2.commit()                                 # dK ready
                    for kk in cutlass.range_constexpr(num_k):
                        mma_dsk.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(mma_dsk, tDQ, tDQrDS[(None, None, kk, 0)], tDQrKT[(None, None, kk, 0)], tDQ)
                dqh.commit()
                with cute.arch.elect_one():
                    tcgen05.commit(storage.q_bars.data_ptr() + 2 + q_prev_idx)
                dsh.release()
                # wait for all TMEM readers then free
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (1 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)

            # ============================================================ COMPUTE WARPS
            elif warp_idx >= COMPUTE_WARPS[0] and warp_idx <= COMPUTE_WARPS[-1]:
                _set_regs(REGS_COMPUTE)
                tmem.allocate(512)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(f32)
                tS = cute.make_tensor(tmem_ptr + TMEM_S, tS_fake.layout)
                tDP = cute.make_tensor(tmem_ptr + TMEM_DP, tS_fake.layout)
                tDV = cute.make_tensor(tmem_ptr + TMEM_DV, tDV_fake.layout)
                tDK = cute.make_tensor(tmem_ptr + TMEM_DK, tDK_fake.layout)
                ctid = tidx - COMPUTE_WARPS[0] * 32          # 0..255
                wg = ctid // 128                              # warpgroup 0/1 -> column halves
                dp_idx = ctid % 128                           # lane / row within tile
                compute_bar = pipeline.NamedBarrier(barrier_id=BAR_COMPUTE, num_threads=NUM_COMPUTE_THREADS)

                # TMEM load/store atoms over a (TILE, 32-column) chunk
                tS2d = tS[((None, None), 0, 0)]               # (TILE, TILE)
                tDP2d = tDP[((None, None), 0, 0)]
                chunk_tiler = (TILE, 32)
                tS_chunks = cute.local_tile(tS2d, chunk_tiler, (0, None))      # (TILE, 32, 4)
                tDP_chunks = cute.local_tile(tDP2d, chunk_tiler, (0, None))
                ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), f32)
                tiled_ld = tcgen05.make_tmem_copy(ld_atom, tS_chunks[(None, None, 0)])
                thr_ld = tiled_ld.get_slice(dp_idx)
                cChunk = cute.make_identity_tensor(chunk_tiler)
                tTR_c = thr_ld.partition_D(cChunk)            # coords within a chunk ((32,1),1,1)
                tTR_tS = thr_ld.partition_S(tS_chunks)        # (((32,32),1),1,1,4)
                tTR_tDP = thr_ld.partition_S(tDP_chunks)
                frag_shape = tTR_c.shape
                rS0 = cute.make_rmem_tensor(frag_shape, f32)
                rS1 = cute.make_rmem_tensor(frag_shape, f32)
                rPb0 = cute.make_rmem_tensor(frag_shape, bf16)
                rPb1 = cute.make_rmem_tensor(frag_shape, bf16)
                rDP0 = cute.make_rmem_tensor(frag_shape, f32)
                rDP = rDP0
                # P store: packed bf16 pairs as f32 words into the P region: (TILE, 64) f32 view of S region
                tP32 = cute.make_tensor(tmem_ptr + TMEM_P, cute.composition(tS2d.layout, cute.make_layout((TILE, TILE // 2))))
                p_chunk_tiler = (TILE, 16)
                tP_chunks = cute.local_tile(tP32, p_chunk_tiler, (0, None))     # (TILE, 16, 4)
                st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(16)), f32)
                tiled_st = tcgen05.make_tmem_copy(st_atom, tP_chunks[(None, None, 0)])
                thr_st = tiled_st.get_slice(dp_idx)
                tRT_tP = thr_st.partition_D(tP_chunks)        # (..., 4)
                cPChunk = cute.make_identity_tensor(p_chunk_tiler)
                p_frag_shape = thr_st.partition_S(cPChunk).shape
                # 2-D (kv, q) view of the dS^T smem buffer (K-major SW128 canonical layout, swizzle via iterator)
                sDS2d = cute.make_tensor(sDS.iterator, cute.make_layout((TILE, (16, 4, 2)), stride=(64, (1, 16, 8192))))
                row0 = tTR_c[0][0]                            # this thread's kv row within tile
                # 4-wide views of the per-column vectors (128-bit shared loads)
                sLSE4 = cute.make_tensor(sLSE.iterator, cute.make_layout((4, TILE // 4), stride=(1, 4)))
                sDelta4 = cute.make_tensor(sDelta.iterator, cute.make_layout((4, TILE // 4), stride=(1, 4)))
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

                for it in cutlass.range(n_iters, unroll=1):
                    iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                    qbase = iq * TILE
                    _rpush("cmp_wait_s")
                    sh = s_cons.wait_and_advance()
                    _rpop()
                    _rpush("cmp_wait_p_empty")
                    ph = p_prod.acquire_and_advance()
                    _rpop()
                    _rpush("cmp_wait_lse")
                    lh = lse_cons.wait_and_advance()
                    _rpop()
                    _rpush("cmp_P")
                    # Chunk ownership: WG0 -> S chunks {0,1} (P cols 0-31 inside own S chunk 0), WG1 -> {2,3} (P cols 32-63
                    # inside WG0's S chunk 1).  WG0 loads chunk 1 first and signals; WG1 waits before storing its P.
                    need_mask = (iq == jblk) or (iq == n_q_blk - 1) or kv_tail
                    if cutlass.const_expr(not EXP_NOCOMPUTE):
                        if wg == 0:
                            cute.copy(tiled_ld, tTR_tS[(None, None, None, 1)], rS1)
                            cute.arch.fence_view_async_tmem_load()
                            compute_bar.arrive()
                            self.p_chunk(rS1, rPb1, 1, need_mask, qbase, kv_row, kv_valid, seqlen, sLSE4, scale_log2)
                            cute.copy(tiled_ld, tTR_tS[(None, None, None, 0)], rS0)
                            self.p_chunk(rS0, rPb0, 0, need_mask, qbase, kv_row, kv_valid, seqlen, sLSE4, scale_log2)
                            self.p_store(rPb0, p_frag_shape, tiled_st, tRT_tP, 0)
                            self.p_store(rPb1, p_frag_shape, tiled_st, tRT_tP, 1)
                        else:
                            cute.copy(tiled_ld, tTR_tS[(None, None, None, 2)], rS0)
                            self.p_chunk(rS0, rPb0, 2, need_mask, qbase, kv_row, kv_valid, seqlen, sLSE4, scale_log2)
                            cute.copy(tiled_ld, tTR_tS[(None, None, None, 3)], rS1)
                            self.p_chunk(rS1, rPb1, 3, need_mask, qbase, kv_row, kv_valid, seqlen, sLSE4, scale_log2)
                            cute.arch.fence_view_async_tmem_load()
                            compute_bar.arrive_and_wait()
                            self.p_store(rPb0, p_frag_shape, tiled_st, tRT_tP, 2)
                            self.p_store(rPb1, p_frag_shape, tiled_st, tRT_tP, 3)
                    cute.arch.fence_view_async_tmem_store()
                    ph.commit()
                    sh.release()
                    lh.release()
                    _rpop()
                    # ---- dS
                    _rpush("cmp_wait_delta")
                    deh = delta_cons.wait_and_advance()
                    _rpop()
                    _rpush("cmp_wait_dp")
                    dph = dp_cons.wait_and_advance()
                    _rpop()
                    _rpush("cmp_wait_ds_empty")
                    dsh = ds_prod.acquire_and_advance()
                    _rpop()
                    _rpush("cmp_dS")
                    for half in cutlass.range_constexpr(2 if not EXP_NOCOMPUTE else 0):
                        chunk = wg * 2 + half
                        rPb = rPb0 if half == 0 else rPb1
                        cute.copy(tiled_ld, tTR_tDP[(None, None, None, chunk)], rDP0)
                        rS0.store(rPb.load().to(f32))
                        for quad in cutlass.range_constexpr(8):
                            dv4 = sDelta4[(None, chunk * 8 + quad)].load()
                            for kq in cutlass.range_constexpr(4):
                                i = quad * 4 + kq
                                rDP0[i] = rS0[i] * (rDP0[i] - dv4[kq])
                        rDSb = cute.make_rmem_tensor(frag_shape, bf16)
                        rDSb.store(rDP0.load().to(bf16))
                        # write dS^T[row0, chunk*32 : +32] into the swizzled sDS tile
                        tRS_rDS = tiled_r2s.retile(rDSb)
                        cute.copy(tiled_r2s, tRS_rDS, tRS_sDS[(None, None, None, chunk)])
                    cute.arch.fence_view_async_tmem_load()
                    dph.release()
                    cute.arch.fence_view_async_shared()
                    dsh.commit()
                    deh.release()
                    _rpop()

                # ---- epilogue: dV then dK
                tDV2d = tDV[((None, None), 0, 0)]
                tDK2d = tDK[((None, None), 0, 0)]
                tDV_chunks = cute.local_tile(tDV2d, chunk_tiler, (0, None))
                tDK_chunks = cute.local_tile(tDK2d, chunk_tiler, (0, None))
                tTR_tDV = thr_ld.partition_S(tDV_chunks)
                tTR_tDK = thr_ld.partition_S(tDK_chunks)
                kv_tok = seq_start + kv_row
                kh = dkdv_cons.wait_and_advance()
                for half in cutlass.range_constexpr(2):
                    chunk = wg * 2 + half
                    cute.copy(tiled_ld, tTR_tDV[(None, None, None, chunk)], rDP)
                    rOb = cute.make_rmem_tensor((32,), bf16)
                    rOb.store(rDP.load().to(bf16))
                    if kv_valid:
                        rOv = cute.make_tensor(rOb.iterator, vec_layout)
                        dstv = cute.make_tensor((mDV.iterator + mDV.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                        cute.copy(vec_atom, rOv, dstv)
                cute.arch.fence_view_async_tmem_load()
                kh.release()
                kh2 = dkdv_cons.wait_and_advance()
                for half in cutlass.range_constexpr(2):
                    chunk = wg * 2 + half
                    cute.copy(tiled_ld, tTR_tDK[(None, None, None, chunk)], rDP)
                    rOb = cute.make_rmem_tensor((32,), bf16)
                    rOb.store((rDP.load() * sm_scale).to(bf16))
                    if kv_valid:
                        rOv = cute.make_tensor(rOb.iterator, vec_layout)
                        dstv = cute.make_tensor((mDK.iterator + mDK.layout((kv_tok, chunk * 32, head))).align(min_align=16), vec_layout)
                        cute.copy(vec_atom, rOv, dstv)
                cute.arch.fence_view_async_tmem_load()
                kh2.release()
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (1 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()

            # ============================================================ REDUCE WARPS (dQ)
            elif warp_idx <= REDUCE_WARPS[-1]:
                _set_regs(REGS_REDUCE)
                tmem.allocate(512)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(f32)
                tDQ = cute.make_tensor(tmem_ptr + TMEM_DQ, tDQ_fake.layout)
                rtid = tidx                                  # 0..127
                reduce_bar = pipeline.NamedBarrier(barrier_id=BAR_REDUCE, num_threads=NUM_REDUCE_THREADS)
                tDQ2d = tDQ[((None, None), 0, 0)]
                dq_tiler = (TILE, DQ_COLS)
                tDQ_chunks = cute.local_tile(tDQ2d, dq_tiler, (0, None))        # (TILE, 32, 4)
                ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(DQ_COLS)), f32)
                tiled_ld = tcgen05.make_tmem_copy(ld_atom, tDQ_chunks[(None, None, 0)])
                thr_ld = tiled_ld.get_slice(rtid)
                tTR_tDQ = thr_ld.partition_S(tDQ_chunks)
                cDQ = cute.make_identity_tensor(dq_tiler)
                tTR_cDQ = thr_ld.partition_D(cDQ)
                NCHUNK = HEAD_DIM // DQ_COLS
                rDQ = cute.make_rmem_tensor(tTR_cDQ.shape, f32)
                rDQs = [cute.make_rmem_tensor(tTR_cDQ.shape, f32) for _ in range(NCHUNK)] if RED_PRELOAD else None
                # RMEM -> SMEM staging copy (fp32 registers -> dq_dtype smem)
                dq_dt = cutlass.Float16 if DQ_F16 else f32
                r2s_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, dq_dt, f32, tiled_ld)
                tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_ld)
                thr_r2s = tiled_r2s.get_slice(rtid)
                tRS_sDQ = thr_r2s.partition_D(sDQ)                               # (R2S, M, N, stages)
                tRS_rDQ32 = tiled_r2s.retile(rDQ)
                rDQc = cute.make_rmem_tensor(tRS_rDQ32.shape, dq_dt)
                tRS_rDQ = rDQc
                # TMA reduce partition
                tDQsDQ, tDQgDQ = cpasync.tma_partition(dq_atom, 0, cute.make_layout(1),
                                                       cute.group_modes(sDQ, 0, 2), cute.group_modes(gDQ, 0, 2))
                store_pipe = pipeline.PipelineTmaStore.create(num_stages=DQ_STAGES, producer_group=reduce_threads)
                stage = cutlass.Int32(0)
                row_dq = tTR_cDQ[0][0]
                for it in cutlass.range(n_iters, unroll=1):
                    iq = (n_q_blk - 1 - it) if Q_REVERSE else (jblk + it)
                    _rpush("red_wait_dq")
                    dqh = dq_cons.wait_and_advance()
                    _rpop()
                    _rpush("red_chunks")
                    if cutlass.const_expr(EXP_NOREDUCE):
                        dqh.release()
                    if cutlass.const_expr(REDUCE_MODE == "red" and not EXP_NOREDUCE):
                        tok = seq_start + iq * TILE + row_dq
                        for chunk in cutlass.range_constexpr(HEAD_DIM // DQ_COLS):
                            cute.copy(tiled_ld, tTR_tDQ[(None, None, None, chunk)], rDQ)
                            if chunk == HEAD_DIM // DQ_COLS - 1:
                                cute.arch.fence_view_async_tmem_load()
                                dqh.release()
                            if tok < total_tokens:
                                base = (mDQp.iterator + mDQp.layout((tok, chunk * DQ_COLS, head))).toint()
                                for v in cutlass.range_constexpr(DQ_COLS // 4):
                                    red_add_v4_f32(base + v * 16, rDQ[4 * v], rDQ[4 * v + 1], rDQ[4 * v + 2], rDQ[4 * v + 3])
                    if cutlass.const_expr(REDUCE_MODE == "tma" and not EXP_NOREDUCE and RED_PRELOAD):
                        # 1) pull the whole tile out of TMEM and release it, 2) stage + TMA-reduce chunk by chunk
                        for chunk in cutlass.range_constexpr(NCHUNK):
                            cute.copy(tiled_ld, tTR_tDQ[(None, None, None, chunk)], rDQs[chunk])
                        cute.arch.fence_view_async_tmem_load()
                        dqh.release()
                        for chunk in cutlass.range_constexpr(NCHUNK):
                            store_pipe.producer_acquire()
                            reduce_bar.arrive_and_wait()
                            rDQc.store(tiled_r2s.retile(rDQs[chunk]).load().to(dq_dt))
                            cute.copy(tiled_r2s, rDQc, tRS_sDQ[(None, None, None, chunk % DQ_STAGES)])
                            cute.arch.fence_view_async_shared()
                            reduce_bar.arrive_and_wait()
                            if warp_idx == 0:
                                cute.copy(dq_atom, tDQsDQ[(None, chunk % DQ_STAGES)], tDQgDQ[(None, iq, chunk)])
                                store_pipe.producer_commit()
                    for chunk in cutlass.range_constexpr(NCHUNK if (REDUCE_MODE == "tma" and not EXP_NOREDUCE and not RED_PRELOAD) else 0):
                        _rpush("red_tmem_ld")
                        cute.copy(tiled_ld, tTR_tDQ[(None, None, None, chunk)], rDQ)
                        if chunk == NCHUNK - 1:
                            cute.arch.fence_view_async_tmem_load()
                            dqh.release()
                        _rpop()
                        _rpush("red_acquire")
                        store_pipe.producer_acquire()
                        reduce_bar.arrive_and_wait()
                        _rpop()
                        _rpush("red_sts")
                        rDQc.store(tRS_rDQ32.load().to(dq_dt))
                        cute.copy(tiled_r2s, tRS_rDQ, tRS_sDQ[(None, None, None, chunk % DQ_STAGES)])
                        cute.arch.fence_view_async_shared()
                        reduce_bar.arrive_and_wait()
                        _rpop()
                        _rpush("red_tma")
                        if warp_idx == 0:
                            cute.copy(dq_atom, tDQsDQ[(None, chunk % DQ_STAGES)], tDQgDQ[(None, iq, chunk)])
                            store_pipe.producer_commit()
                        _rpop()
                    _rpop()
                store_pipe.producer_tail()
                pipeline.NamedBarrier(barrier_id=BAR_FINAL, num_threads=32 * (1 + len(COMPUTE_WARPS) + len(REDUCE_WARPS))).arrive_and_wait()
            else:
                # idle warps 14, 15: participate in the warpgroup register deallocation only
                _set_regs(REGS_WG3)


# ---------------------------------------------------------------------------------------- host wrapper
_KERNEL = None
_COMPILED = {}


def _as_tdh(t):
    """(T, H, D) torch tensor -> cute (T, D, H) view."""
    return from_dlpack(t.permute(0, 2, 1), assumed_align=16).mark_layout_dynamic(leading_dim=1)


def attention_bwd_prepare(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, tile_table, max_tiles, sm_scale):
    """Host-side argument preparation (DLPack conversions); call before launching any kernel."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = AttnBwdSm100()
    T, H, D = q.shape
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_as_tdh(q), _as_tdh(k), _as_tdh(v), _as_tdh(dout),
            from_dlpack(lse, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(delta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            _as_tdh(dq_accum), _as_tdh(dk), _as_tdh(dv),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            from_dlpack(tile_table, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            cutlass.Int32(max_tiles), cutlass.Int32(T), cutlass.Float32(sm_scale), stream)
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
