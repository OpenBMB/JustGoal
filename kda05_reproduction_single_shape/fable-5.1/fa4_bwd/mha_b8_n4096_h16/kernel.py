"""Causal attention backward for B200 (sm_100a) in CuTe DSL — own implementation, v11.4 (cp.async Q/dO loads, late dQ reduce knobs; v11.0:
128-row Q trips with every TMEM slot single-buffered (N=128 GEMMs are math-bound; the N=64 GEMMs of
v10.x were shared-memory-fetch-bound).

Architecture (warp-specialised, persistent CTAs, one 128-row K/V tile at a time):
  warps 0-3   reduce : dQ (q lanes x d columns) TMEM -> regs -> bf16 rows in the trip's dO stage ->
                       TMA reduce-add (.add.bf16) into dq; dK/dV epilogue through the same staging
  warps 4-19  compute: 4 warpgroups x 32 S/dP columns: P = exp2(S*c + lse2) -> TMEM (bf16, overlays S),
                       dS^T = P*(dP*scale + delta) -> swizzled smem
  warp 20     mma    : per trip j: dV(j), S(j+1), dQ(j), dK(j), dP(j+1) on tcgen05
  warp 21     load   : TMA K/V once per tile, Q/dO per trip; cp.async lse2/delta rows
TMEM (512 cols): dK[0:128) dV[128:256) X = S^T/P^T [256:384) Y = dP^T/dQ [384:512).
smem: K 32 | V 32 | Q 2x32 | dO 2x32 | dS 32 KB (+ lse2/delta vectors); the dQ staging aliases the
dO stage of the same trip (dO(j) is dead after dV(j), which precedes dQ(j)).
Pre-kernel computes delta = -sm_scale*sum(O*dO), lse2 = -lse*log2(e) and zeroes dq (bf16).
The CuTe DSL idioms (pipeline classes, TMA/TMEM copy partitioning, smem layout helpers)
follow NVIDIA's BSD-3 CUTLASS example `fmha_bwd.py`; the kernel organisation, causal
trip logic, GQA head loop, workspace handling and pre-kernel are this solution's own.
"""
import math
import os as _os
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float32, Float16, BFloat16, Boolean
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

LOG2E = 1.4426950408889634


@dsl_user_op
def _scale_by_floor(r_rounded: Float32, frac_pow: Float32, *, loc=None, ip=None) -> Float32:
    """frac_pow * 2^(low 8 bits of r_rounded): shift the integer into the exponent field and add."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(r_rounded).ir_value(loc=loc, ip=ip), Float32(frac_pow).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .s32 ri, fi, re, oi;\n\tmov.b32 ri, $1;\n\tmov.b32 fi, $2;\n\t"
            "shl.b32 re, ri, 23;\n\tadd.s32 oi, re, fi;\n\tmov.b32 $0, oi;\n\t}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _red_add_v4_bf16x2(gaddr: cutlass.Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None) -> None:
    """red.global.add.noftz.v4.bf16x2 [gaddr], {a, b, c, d}: eight bf16 adds in L2 with one 16-byte request."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
         Int32(c).ir_value(loc=loc, ip=ip), Int32(d).ir_value(loc=loc, ip=ip)],
        "red.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
                                         is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


class CausalAttnBwdSm100:
    def __init__(self):
        self.io_dtype = BFloat16
        self.acc_dtype = Float32
        self.TK = 128  # K/V rows per CTA tile
        self.TQ = 128  # Q rows per trip
        self.D = 128
        TK, TQ, D = self.TK, self.TQ, self.D
        # (M, N, K) per GEMM
        self.tiler_S = (TK, TQ, D)    # S^T  = K  Q^T    A=K    (K-major)         B=Q  (K-major)
        self.tiler_dP = (TK, TQ, D)   # dP^T = V  dO^T   A=V    (K-major)         B=dO (K-major)
        self.tiler_dV = (TK, D, TQ)   # dV   = P^T dO    A=P^T  (TMEM)            B=dO (MN-major)
        self.tiler_dK = (TK, D, TQ)   # dK   = dS^T Q    A=dS^T smem (K-major)    B=Q  (MN-major)
        self.tiler_dQ = (TQ, D, TK)   # dQ   = dS K      A=dS smem (MN-major)     B=K  (MN-major)

        self.reduce_warps = (0, 1, 2, 3)
        self.compute_warps = tuple(range(4, 20))
        self.mma_warp = int(_os.environ.get("FA_MMAW", 20))
        self.load_warp = 21 if self.mma_warp != 21 else 20
        self.num_reduce_warps = 4
        self.num_compute_warps = 16
        self.num_warps = 24
        self.threads_per_cta = 32 * self.num_warps

        self.tmem_cols = 512
        self.tmem_dK = 0
        self.tmem_dV = 128
        self.tmem_X = 256          # S^T (P^T bf16 overlays)
        self.tmem_Y = 384          # dP^T, then dQ

        self.st_Q = 2
        self.st_dO = 2
        self.st_vec = 2
        self.st_dKdV = 2
        self.PRE_ROWS = int(_os.environ.get("FA_PRE_ROWS", 64))   # rows per pre-kernel CTA (one thread per row)
        regs = [int(x) for x in _os.environ.get("FA_REGS", "88,96,32").split(",")]
        self.regs_compute, self.regs_reduce, self.regs_other = regs
        self.regs_launch = (65536 // self.threads_per_cta) // 8 * 8     # per-thread registers at launch (CTA pool = this x threads)
        self.DEBUG = _os.environ.get("FA_DEBUG") == "1"
        self.TRACE = _os.environ.get("FA_TRACE") == "1"
        self.ABL = int(_os.environ.get("FA_ABL", 0))     # ablation knobs (timing only, wrong results)
        self.TAIL_GROUPS = int(_os.environ.get("FA_TAILG", 16))
        self.TAIL_MODE = int(_os.environ.get("FA_TAILMODE", 1))
        self.ENDTIME = int(_os.environ.get("FA_ENDTIME", 0))
        self.PERSIST = int(_os.environ.get("FA_PERSIST", 1))
        self.NUM_SM = int(_os.environ.get("FA_NSM", 148))
        self.PRINT = _os.environ.get("FA_PRINT") == "1"
        self.DRAINX = int(_os.environ.get("FA_DRAINX", 32))     # TMEM columns per reduce-warp load
        self.DRAINDYN = int(_os.environ.get("FA_DRAINDYN", 0))  # 1: dynamic chunk loop in the drain (no load hoisting)
        self.PBF16 = int(_os.environ.get("FA_PBF16", 0))        # 1: dS uses the bf16-rounded P (frees 32 fp32 registers per compute thread)
        self.EMUL = int(_os.environ.get("FA_EMUL", 0))          # exp2 on the FMA pipe for every EMUL-th pair (0: all MUFU)
        self.DQRED = int(_os.environ.get("FA_DQRED", 0))        # 1: dq via red.global.add.v4.bf16x2 from the reduce warps (staging readback) instead of the TMA reduce-add
        self.DRAINPIPE = int(_os.environ.get("FA_DRAINPIPE", 1))  # 1: next TMEM chunk load issued before converting the current one
        self.QCP = int(_os.environ.get("FA_QCP", 0))            # 1: Q tiles via cp.async (load warp, 64 x 16 B per lane) instead of TMA
        self.DOCP = int(_os.environ.get("FA_DOCP", 0))          # 1: dO tiles via cp.async
        self.REDLATE = int(_os.environ.get("FA_REDLATE", 0))    # 1: issue the dQ TMA reduce once P(j+1) is ready (reads overlap dV(j+1))
        self.WARR = int(_os.environ.get("FA_WARR", 1))          # 1: one mbarrier arrival per warp (elected lane after a warp sync) instead of per thread
        self.ORDER = int(_os.environ.get("FA_ORDER", 0))     # 0: dV S dQ dK dP ; 2: dV dQ S dK dP ; 3: dV dQ dK S dP

        self.bar_cta = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.bar_tmem = pipeline.NamedBarrier(barrier_id=2, num_threads=32)
        self.bar_reduce = pipeline.NamedBarrier(barrier_id=5, num_threads=self.num_reduce_warps * 32)
        self.align = 1024

    @cute.jit
    def mark(self, mProg, base, code, trip):
        if cutlass.const_expr(self.DEBUG):
            mProg[base] = Int32(code)
            mProg[base + 1] = Int32(trip)
        if cutlass.const_expr(self.TRACE):
            cute.experimental.iket.mark("c%d" % code)

    # ------------------------------------------------------------------ host (jit)
    @cute.jit
    def __call__(
        self,
        prob: Tuple[Int32, Int32, Int32, Int32],   # (S, G, Hkv, B)
        Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, O: cute.Tensor, dO: cute.Tensor,
        LSE: cute.Tensor,
        dQ: cute.Tensor, dK: cute.Tensor, dV: cute.Tensor,
        workspace: cute.Tensor,
        sm_scale: Float32,
        prog: cute.Tensor,
        cnt: cute.Tensor,
        stream: cuda.CUstream,
    ):
        S, G, Hkv, B = prob
        D = self.D
        H = G * Hkv

        def qlike(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[0], t.shape[1], ((t.shape[2], t.shape[3]), t.shape[4])),
                    stride=(t.stride[0], t.stride[1], ((t.stride[2], t.stride[3]), t.stride[4])),
                ),
            )

        def klike(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[0], t.shape[1], (t.shape[2], t.shape[3])),
                    stride=(t.stride[0], t.stride[1], (t.stride[2], t.stride[3])),
                ),
            )
        mQ, mO, mdO, mdQ = qlike(Q), qlike(O), qlike(dO), qlike(dQ)
        mK, mV, mdK, mdV = klike(K), klike(V), klike(dK), klike(dV)
        mLSE = cute.make_tensor(
            LSE.iterator,
            cute.make_layout(
                (LSE.shape[0], ((LSE.shape[1], LSE.shape[2]), LSE.shape[3])),
                stride=(LSE.stride[0], ((LSE.stride[1], LSE.stride[2]), LSE.stride[3])),
            ),
        )
        nvec = cute.assume(S * H * B, divby=8)
        f32 = cute.recast_ptr(workspace.iterator, dtype=Float32)
        vec_layout = cute.make_layout((S, ((G, Hkv), B)), stride=(1, ((S, S * G), S * H)))
        mDelta = cute.make_tensor(f32, vec_layout)
        mLSE2 = cute.make_tensor(f32 + nvec, vec_layout)

        cta_group = tcgen05.CtaGroup.ONE
        io, acc = self.io_dtype, self.acc_dtype
        mma_S = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.K, OperandMajorMode.K, acc, cta_group, self.tiler_S[:2])
        mma_dP = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.K, OperandMajorMode.K, acc, cta_group, self.tiler_dP[:2])
        mma_dV = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.K, OperandMajorMode.MN, acc, cta_group, self.tiler_dV[:2], tcgen05.OperandSource.TMEM)
        mma_dK = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.K, OperandMajorMode.MN, acc, cta_group, self.tiler_dK[:2])
        mma_dQ = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.MN, OperandMajorMode.MN, acc, cta_group, self.tiler_dQ[:2])
        mma_dV_c = sm100_utils.make_trivial_tiled_mma(io, io, OperandMajorMode.K, OperandMajorMode.MN, acc, cta_group, self.tiler_dV[:2], tcgen05.OperandSource.TMEM)

        cluster_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_S.thr_id.shape,))

        # smem layouts (staged); views of the same bytes are byte-identical (checked: scripts/dev/v11/layout_check.py)
        lK = sm100_utils.make_smem_layout_a(mma_S, self.tiler_S, io, 1)             # K   (keys, d)   K-major A of S^T
        lKB = sm100_utils.make_smem_layout_b(mma_dQ, self.tiler_dQ, io, 1)          # K   as MN-major B of dQ
        lQ = sm100_utils.make_smem_layout_b(mma_S, self.tiler_S, io, self.st_Q)     # Q   (q, d)      K-major B of S^T
        lQT = sm100_utils.make_smem_layout_b(mma_dK, self.tiler_dK, io, self.st_Q)  # Q   as MN-major B of dK
        lV = sm100_utils.make_smem_layout_a(mma_dP, self.tiler_dP, io, 1)           # V   (keys, d)   K-major A of dP^T
        ldO = sm100_utils.make_smem_layout_b(mma_dP, self.tiler_dP, io, self.st_dO) # dO  (q, d)      K-major B of dP^T; also dQ/dK/dV staging
        ldOT = sm100_utils.make_smem_layout_b(mma_dV, self.tiler_dV, io, self.st_dO)   # dO as MN-major B of dV
        ldST = sm100_utils.make_smem_layout_a(mma_dK, self.tiler_dK, io, 1)         # dS^T (keys, q)  K-major A of dK
        ldSA = sm100_utils.make_smem_layout_a(mma_dQ, self.tiler_dQ, io, 1)         # dS   as MN-major A of dQ
        lP = sm100_utils.make_smem_layout_a(mma_dV, self.tiler_dV, io, 1)           # P^T layout (TMEM)
        lvec = cute.make_layout((self.TQ, self.st_vec))

        tma_load = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_red = cpasync.CopyReduceBulkTensorTileS2GOp()
        tma_store = cpasync.CopyBulkTensorTileS2GOp()
        lK1 = cute.select(lK, mode=[0, 1, 2])
        lV1 = cute.select(lV, mode=[0, 1, 2])
        lQ1 = cute.select(lQ, mode=[0, 1, 2])
        ldO1 = cute.select(ldO, mode=[0, 1, 2])
        tmaK, tK = cute.nvgpu.make_tiled_tma_atom_A(tma_load, mK, lK1, self.tiler_S, mma_S, cluster_vmnk.shape)
        tmaV, tV = cute.nvgpu.make_tiled_tma_atom_A(tma_load, mV, lV1, self.tiler_dP, mma_dP, cluster_vmnk.shape)
        tmaQ, tQ = cute.nvgpu.make_tiled_tma_atom_B(tma_load, mQ, lQ1, self.tiler_S, mma_S, cluster_vmnk.shape)
        tmadO, tdO = cute.nvgpu.make_tiled_tma_atom_B(tma_load, mdO, ldO1, self.tiler_dP, mma_dP, cluster_vmnk.shape)
        self.bytes_Q = cute.size_in_bytes(io, lQ1)
        self.bytes_K = cute.size_in_bytes(io, lK1)
        self.bytes_V = cute.size_in_bytes(io, lV1)
        self.bytes_dO = cute.size_in_bytes(io, ldO1)
        # staging: a plain (128 rows, 128 d) SW128 layout over the bytes of each dO stage (dO(j) is dead
        # when dQ(j) is staged); rows tiled first so a TMA box is (128 rows x 64 d), two boxes per tile
        lStg_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(OperandMajorMode.K, io, (self.TQ, D)), io)
        lStg = cute.tile_to_shape(lStg_atom, (self.TQ, D, self.st_dO), order=(0, 1, 2))
        lStg2 = cute.select(lStg, mode=[0, 1])
        tmadQ, tdQacc = cpasync.make_tiled_tma_atom(tma_red, mdQ, lStg2, (self.TQ, D))
        tmadK, tdKt = cpasync.make_tiled_tma_atom(tma_store, mdK, lStg2, (self.TK, D))
        tmadV, tdVt = cpasync.make_tiled_tma_atom(tma_store, mdV, lStg2, (self.TK, D))
        if cutlass.const_expr(self.PRINT):
            print("lStg", lStg, "ldO", ldO)

        @cute.struct
        class Smem:
            sK: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(lK)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(lV)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(lQ)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(ldO)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(ldST)], 1024]
            sLSE: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(lvec)], 1024]
            sDelta: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(lvec)], 1024]
            bar_Qf: cute.struct.MemRange[cutlass.Int64, 2]
            bar_Qe: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dVgo: cute.struct.MemRange[cutlass.Int64, 1]
            bar_dOf: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dOe: cute.struct.MemRange[cutlass.Int64, 2]
            bar_lse: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            bar_S: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dP: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dQ: cute.struct.MemRange[cutlass.Int64, 2]
            bar_P: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dS: cute.struct.MemRange[cutlass.Int64, 2]
            bar_dKdV: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            bar_K: cute.struct.MemRange[cutlass.Int64, 2]
            bar_V: cute.struct.MemRange[cutlass.Int64, 2]
            bar_tile: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            tmem_holder: cutlass.Int32
            tile_ids: cute.struct.MemRange[cutlass.Int32, 2]

        self.smem_t = Smem

        # ---- pre-kernel: delta, lse2, zero dq (TMA tile loads/stores, one thread per row)
        PR = self.PRE_ROWS
        lOt_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(OperandMajorMode.K, io, (PR, D)), io)
        lOt = cute.tile_to_shape(lOt_atom, (PR, D, 1), order=(1, 0, 2))
        lOt1 = cute.select(lOt, mode=[0, 1])
        tmaO, tOt = cpasync.make_tiled_tma_atom(tma_load, mO, lOt1, (PR, D))
        tmadO2, tdOt = cpasync.make_tiled_tma_atom(tma_load, mdO, lOt1, (PR, D))
        tmaZ, tZt = cpasync.make_tiled_tma_atom(tma_store, mdQ, lOt1, (PR, D))
        self.pre_bytes = 2 * cute.size_in_bytes(io, lOt1)

        @cute.struct
        class PreSmem:
            bar: cutlass.Int64
            sO: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(lOt)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(lOt)], 1024]

        self.pre_smem_t = PreSmem
        nk = cute.ceil_div(S, self.TK)
        n_hb = Hkv * B
        n_tiles = nk * n_hb
        n_cta = n_tiles
        if cutlass.const_expr(self.PERSIST > 0):
            n_cta = cute.min(n_tiles, Int32(self.NUM_SM * self.PERSIST))
        self.pre_kernel(tmaO, tOt, tmadO2, tdOt, mLSE, mDelta, mLSE2, sm_scale, S, lOt, cnt, n_cta, tmaZ, tZt).launch(
            grid=(cute.ceil_div(S, PR), H, B), block=(PR, 1, 1), cluster=(1, 1, 1), stream=stream)

        # ---- main kernel
        self.bwd_kernel(
            mma_S, mma_dP, mma_dV, mma_dK, mma_dQ, mma_dV_c,
            tmaK, tK, tmaV, tV, tmaQ, tQ, tmadO, tdO, tmadQ, tdQacc,
            tdKt, tdVt, mLSE2, mDelta, sm_scale, S, G, prog, Hkv, n_hb, nk, n_tiles, cnt, tmadK, tmadV,
            lK, lKB, lQ, lQT, lV, ldO, ldOT, ldST, ldSA, lP, lvec, lStg, mdQ, mQ, mdO,
        ).launch(grid=(n_cta, 1, 1), block=(self.threads_per_cta, 1, 1), cluster=(1, 1, 1),
                 stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------ pre-kernel
    @cute.kernel
    def pre_kernel(self, tmaO: cute.CopyAtom, mO: cute.Tensor, tmadO: cute.CopyAtom, mdO: cute.Tensor,
                   mLSE: cute.Tensor, mDelta: cute.Tensor,
                   mLSE2: cute.Tensor, sm_scale: Float32, S: Int32, lOt: cute.ComposedLayout, mCnt: cute.Tensor, n_cta: Int32,
                   tmaZ: cute.CopyAtom, mZ: cute.Tensor):
        """Per PR-row tile: TMA-load O and dO, one thread per row computes delta = -sm_scale*sum(O*dO)
        and lse2 = -lse*log2e, then the O buffer is zeroed and TMA-stored as the bf16 dq tile."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        G = cute.size(mLSE.shape[1][0][0])
        hb = ((bidy % G, bidy // G), bidz)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tmaO)
            cpasync.prefetch_descriptor(tmadO)
            cpasync.prefetch_descriptor(tmaZ)
        if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0:
            mCnt[0] = n_cta
        smem = utils.SmemAllocator()
        st = smem.allocate(self.pre_smem_t)
        bar = st.bar.ptr
        if tidx == 0:
            cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sO = st.sO.get_tensor(lOt.outer, swizzle=lOt.inner)
        sdO = st.sdO.get_tensor(lOt.outer, swizzle=lOt.inner)
        sZ = sO
        PR = self.PRE_ROWS
        gO = cute.local_tile(mO, (PR, self.D), (None, None, None))
        gdO = cute.local_tile(mdO, (PR, self.D), (None, None, None))
        gZ = cute.local_tile(mZ, (PR, self.D), (None, None, None))
        tOs, tOg = cpasync.tma_partition(tmaO, 0, cute.make_layout(1), cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2))
        tdOs, tdOg = cpasync.tma_partition(tmadO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdO, 0, 2))
        tZs, tZg = cpasync.tma_partition(tmaZ, 0, cute.make_layout(1), cute.group_modes(sZ, 0, 2), cute.group_modes(gZ, 0, 2))
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, self.pre_bytes)
            cute.copy(tmaO, tOg[(None, bidx, 0, hb)], tOs[None, 0], tma_bar_ptr=bar)
            cute.copy(tmadO, tdOg[(None, bidx, 0, hb)], tdOs[None, 0], tma_bar_ptr=bar)
        cute.arch.mbarrier_wait(bar, 0)
        row = tidx
        s_ = bidx * PR + row
        o_row = cute.logical_divide(sO[row, None, 0], cute.make_layout(8))
        d_row = cute.logical_divide(sdO[row, None, 0], cute.make_layout(8))
        acc = Float32(0.0)
        for j in cutlass.range_constexpr(self.D // 8):
            prod = o_row[None, j].load().to(Float32) * d_row[None, j].load().to(Float32)
            for e in cutlass.range_constexpr(8):
                acc = acc + prod[e]
        if s_ < S:
            mDelta[s_, hb] = -acc * sm_scale
            mLSE2[s_, hb] = -mLSE[s_, hb] * Float32(LOG2E)
        cute.arch.sync_threads()
        zero8 = cute.make_rmem_tensor((8,), self.io_dtype)
        for e in cutlass.range_constexpr(8):
            zero8[e] = self.io_dtype(0.0)
        z_row = cute.logical_divide(sZ[row, None, 0], cute.make_layout(8))
        for j in cutlass.range_constexpr(self.D // 8):
            z_row[None, j].store(zero8.load())
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.copy(tmaZ, tZs[None, 0], tZg[(None, bidx, 0, hb)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    # ------------------------------------------------------------------ tile order
    @cute.jit
    def tile_coords(self, lin, n_hb, nk, Hkv):
        # group-major (all KV tiles of one (kv-head, batch) group adjacent -> Q/dO L2 reuse); the last R
        # groups are dispatched kv-major so that tile cost decreases smoothly to the end (causal tail)
        R = cute.min(Int32(self.TAIL_GROUPS), n_hb)
        n_std = (n_hb - R) * nk
        half = nk // 2
        grp = Int32(0)
        bidx = Int32(0)
        gi = Int32(0)
        if lin < n_std:
            grp = lin // nk
            bidx = lin - grp * nk
        else:
            rem = lin - n_std
            if cutlass.const_expr(self.TAIL_MODE == 1):
                bidx = rem // R
                gi = rem - bidx * R
            else:
                n_heavy = R * half
                if rem < n_heavy:
                    gi = rem // half
                    bidx = rem - gi * half
                else:
                    rem2 = rem - n_heavy
                    n_light = nk - half
                    gi = rem2 // n_light
                    bidx = half + (rem2 - gi * n_light)
            grp = n_hb - R + gi
        bidy = grp % Hkv
        bidz = grp // Hkv
        return bidx, bidy, bidz

    # NOTE: pipeline states must be advanced in the loop bodies themselves (not inside helper
    # functions): the DSL only treats names assigned or method-invoked in the loop body as loop-carried.

    @cute.kernel
    def bwd_kernel(
        self,
        mma_S: cute.TiledMma, mma_dP: cute.TiledMma, mma_dV: cute.TiledMma,
        mma_dK: cute.TiledMma, mma_dQ: cute.TiledMma, mma_dV_c: cute.TiledMma,
        tmaK: cute.CopyAtom, mK: cute.Tensor, tmaV: cute.CopyAtom, mV: cute.Tensor,
        tmaQ: cute.CopyAtom, mQ: cute.Tensor, tmadO: cute.CopyAtom, mdO: cute.Tensor,
        tmadQ: cute.CopyAtom, mdQacc: cute.Tensor,
        mdK: cute.Tensor, mdV: cute.Tensor, mLSE2: cute.Tensor, mDelta: cute.Tensor,
        sm_scale: Float32, S: Int32, G: Int32, mProg: cute.Tensor, Hkv: Int32, n_hb: Int32, nk: Int32,
        n_tiles: Int32, mCnt: cute.Tensor, tmadK: cute.CopyAtom, tmadV: cute.CopyAtom,
        lK: cute.ComposedLayout, lKB: cute.ComposedLayout, lQ: cute.ComposedLayout, lQT: cute.ComposedLayout,
        lV: cute.ComposedLayout, ldO: cute.ComposedLayout, ldOT: cute.ComposedLayout,
        ldST: cute.ComposedLayout, ldSA: cute.ComposedLayout, lP: cute.ComposedLayout, lvec: cute.Layout,
        lStg: cute.ComposedLayout, mdQraw: cute.Tensor, mQraw: cute.Tensor, mdOraw: cute.Tensor,
    ):
        lin, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        pbase = (lin * self.num_warps + warp_idx) * 2
        if warp_idx == self.load_warp:
            cpasync.prefetch_descriptor(tmaK)
            cpasync.prefetch_descriptor(tmaV)
            cpasync.prefetch_descriptor(tmaQ)
            cpasync.prefetch_descriptor(tmadO)
        if warp_idx == self.reduce_warps[0]:
            cpasync.prefetch_descriptor(tmadQ)
            cpasync.prefetch_descriptor(tmadK)
            cpasync.prefetch_descriptor(tmadV)

        smem = utils.SmemAllocator()
        st = smem.allocate(self.smem_t)
        if cutlass.const_expr(self.ENDTIME):
            tidx0 = cute.arch.thread_idx()[0]
            if tidx0 == 0:
                t_start = _globaltimer()
                mProg[4 * lin] = Int32(t_start & 0xffffffff)
                mProg[4 * lin + 1] = Int32(t_start >> 32)

        grp_load = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        grp_mma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        grp_loadwarp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        grp_set = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps * 32)
        grp_reduce = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_reduce_warps * 32)
        if cutlass.const_expr(self.WARR):
            grp_set_a = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps)     # arrivals: one per warp
            grp_reduce_a = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_reduce_warps)
        else:
            grp_set_a = grp_set
            grp_reduce_a = grp_reduce
        pipe_lse = pipeline.PipelineCpAsync.create(barrier_storage=st.bar_lse.data_ptr(), num_stages=self.st_vec,
                                                   producer_group=grp_loadwarp, consumer_group=grp_set_a)
        pipe_S = pipeline.PipelineUmmaAsync.create(barrier_storage=st.bar_S.data_ptr(), num_stages=1,
                                                   producer_group=grp_mma, consumer_group=grp_set)
        pipe_dP = pipeline.PipelineUmmaAsync.create(barrier_storage=st.bar_dP.data_ptr(), num_stages=1,
                                                    producer_group=grp_mma, consumer_group=grp_set)
        pipe_dQ = pipeline.PipelineUmmaAsync.create(barrier_storage=st.bar_dQ.data_ptr(), num_stages=1,
                                                    producer_group=grp_mma, consumer_group=grp_reduce_a)
        pipe_P = pipeline.PipelineAsyncUmma.create(barrier_storage=st.bar_P.data_ptr(), num_stages=1,
                                                   producer_group=grp_set_a, consumer_group=grp_mma)
        pipe_dS = pipeline.PipelineAsyncUmma.create(barrier_storage=st.bar_dS.data_ptr(), num_stages=1,
                                                    producer_group=grp_set_a, consumer_group=grp_mma)
        pipe_dKdV = pipeline.PipelineUmmaAsync.create(barrier_storage=st.bar_dKdV.data_ptr(), num_stages=self.st_dKdV,
                                                      producer_group=grp_mma, consumer_group=grp_reduce_a)
        pipe_K = pipeline.PipelineTmaUmma.create(barrier_storage=st.bar_K.data_ptr(), num_stages=1,
                                                 producer_group=grp_load, consumer_group=grp_mma, tx_count=self.bytes_K)
        pipe_V = pipeline.PipelineTmaUmma.create(barrier_storage=st.bar_V.data_ptr(), num_stages=1,
                                                 producer_group=grp_load, consumer_group=grp_mma, tx_count=self.bytes_V)
        grp_tile_cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, (1 + self.num_compute_warps + self.num_reduce_warps) * 32)
        pipe_tile = pipeline.PipelineAsync.create(barrier_storage=st.bar_tile.data_ptr(), num_stages=2,
                                                  producer_group=grp_loadwarp, consumer_group=grp_tile_cons)
        # dO ring: raw mbarriers. full[s]: 1 arrival + tx bytes (load warp); empty[s]: 1 arrival (reduce
        # warp 0 after the TMA reduce/store finished reading the staging that aliases the stage)
        bar_dOf = st.bar_dOf.data_ptr()
        bar_dOe = st.bar_dOe.data_ptr()
        bar_Qf = st.bar_Qf.data_ptr()
        bar_Qe = st.bar_Qe.data_ptr()
        bar_dVgo = st.bar_dVgo.data_ptr()           # MMA warp arrives right before dV(j+1): the reduce warps issue the dQ(j) reduce then (REDLATE)
        if warp_idx == self.load_warp:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(bar_dVgo, 1)
                for s_ in cutlass.range_constexpr(self.st_dO):
                    cute.arch.mbarrier_init(bar_dOf + s_, 32 if self.DOCP else 1)   # cp.async: one arrival per load-warp lane
                    cute.arch.mbarrier_init(bar_dOe + s_, 1)
                for s_ in cutlass.range_constexpr(self.st_Q):
                    cute.arch.mbarrier_init(bar_Qf + s_, 32 if self.QCP else 1)
                    cute.arch.mbarrier_init(bar_Qe + s_, 1)
        cute.arch.mbarrier_init_fence()
        sTile = st.tile_ids.get_tensor(cute.make_layout(2))
        self.bar_cta.arrive_and_wait()

        sK = st.sK.get_tensor(lK.outer, swizzle=lK.inner)
        sV = st.sV.get_tensor(lV.outer, swizzle=lV.inner)
        sQ = st.sQ.get_tensor(lQ.outer, swizzle=lQ.inner)
        sdO = st.sdO.get_tensor(ldO.outer, swizzle=ldO.inner)
        sdS = st.sdS.get_tensor(ldSA.outer, swizzle=ldSA.inner)      # (q-contiguous view: flat index = 128*key + q, the compute warps' row order)
        sLSE = st.sLSE.get_tensor(lvec)
        sDelta = st.sDelta.get_tensor(lvec)
        tmem_holder = st.tmem_holder.ptr
        sKB = cute.make_tensor(cute.recast_ptr(sK.iterator, lKB.inner), lKB.outer)
        sQT = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQT.inner), lQT.outer)
        sdST = cute.make_tensor(cute.recast_ptr(sdS.iterator, ldST.inner), ldST.outer)     # dS^T as K-major A of dK
        sdOT = cute.make_tensor(cute.recast_ptr(sdO.iterator, ldOT.inner), ldOT.outer)
        sStg = cute.make_tensor(cute.recast_ptr(sdO.iterator, lStg.inner), lStg.outer)     # (row, d, stage) staging over the dO stages
        sQrows = cute.make_tensor(cute.recast_ptr(sQ.iterator, lStg.inner), lStg.outer)    # (row, d, stage) view of the Q stages (cp.async loads)
        tP_ptr0 = cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.tmem)
        tP0 = cute.make_tensor(tP_ptr0, lP.outer)

        frK = mma_S.make_fragment_A(sK)
        frQ = mma_S.make_fragment_B(sQ)
        frV = mma_dP.make_fragment_A(sV)
        frdO = mma_dP.make_fragment_B(sdO)
        frdSA = mma_dQ.make_fragment_A(sdS)
        frKB = mma_dQ.make_fragment_B(sKB)
        frdST = mma_dK.make_fragment_A(sdST)
        frQT = mma_dK.make_fragment_B(sQT)
        frdOT = mma_dV.make_fragment_B(sdOT)

        tS_ = mma_S.make_fragment_C(mma_S.partition_shape_C(self.tiler_S[:2]))
        tS = cute.make_tensor(tS_.iterator + self.tmem_X, tS_.layout)
        tdP_ = mma_dP.make_fragment_C(mma_dP.partition_shape_C(self.tiler_dP[:2]))
        tdP = cute.make_tensor(tdP_.iterator + self.tmem_Y, tdP_.layout)
        tdQ_ = mma_dQ.make_fragment_C(mma_dQ.partition_shape_C(self.tiler_dQ[:2]))
        tdQ = cute.make_tensor(cute.make_ptr(self.acc_dtype, self.tmem_Y, cute.AddressSpace.tmem, assumed_align=256), tdQ_.layout)
        tdK = mma_dK.make_fragment_C(mma_dK.partition_shape_C(self.tiler_dK[:2]))
        tdK = cute.make_tensor(tdK.iterator + self.tmem_dK, tdK.layout)
        tdV = mma_dV.make_fragment_C(mma_dV.partition_shape_C(self.tiler_dV[:2]))
        tdV = cute.make_tensor(tdV.iterator + self.tmem_dV, tdV.layout)
        frP_ = mma_dV.make_fragment_A(tP0)[None, None, None, 0]
        # P^T (bf16) overlays the fp32 S^T slot with the k-block stride doubled: k-block kb (16 q
        # columns) is packed into fp32 columns [16 kb, 16 kb + 8), inside the columns that the
        # warpgroup owning those q columns reads, so no cross-warpgroup barrier is needed before the store.
        s2 = frP_.stride[2]
        if cutlass.const_expr(isinstance(s2, tuple)):
            s2d = tuple(2 * x for x in s2)
        else:
            s2d = 2 * s2
        frP_layout = cute.make_layout(frP_.shape, stride=(frP_.stride[0], frP_.stride[1], s2d))
        frP = cute.make_tensor(cute.recast_ptr(tS.iterator, dtype=self.io_dtype), frP_layout)
        if cutlass.const_expr(self.PRINT):
            print("frP_", frP_.layout, "frP", frP.layout)

        nq = cute.ceil_div(S, self.TQ)
        tile_env = (n_tiles, n_hb, nk, Hkv, nq, G, S)

        if warp_idx == self.load_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.load_role(mK, mV, mQ, mdO, mQraw, mdOraw, mLSE2, mDelta, sK, sV, sQ, sdO, sQrows, sStg, sLSE, sDelta,
                           mma_S, mma_dP, tmaK, tmaV, tmaQ, tmadO, tile_env, pipe_tile, sTile, mCnt, lin,
                           pipe_K, pipe_V, bar_Qf, bar_Qe, bar_dOf, bar_dOe, pipe_lse, mProg, pbase)
            self.mark(mProg, pbase, 69, 0)
        elif warp_idx == self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
            self.mma_role(mma_S, mma_dP, mma_dV, mma_dK, mma_dQ,
                          tS, frK, frQ, tdP, frV, frdO, tdV, frP, frdOT,
                          tdQ, frdSA, frKB, tdK, frdST, frQT, tmem_holder, tile_env, pipe_tile, sTile,
                          pipe_K, pipe_V, bar_Qf, bar_Qe, bar_dOf, bar_dVgo, pipe_S, pipe_dP, pipe_dQ, pipe_P, pipe_dS, pipe_dKdV, mProg, pbase)
            self.mark(mProg, pbase, 29, 0)
        elif warp_idx >= self.compute_warps[0] and warp_idx <= self.compute_warps[-1]:
            if cutlass.const_expr(self.regs_compute > self.regs_launch):
                cute.arch.setmaxregister_increase(self.regs_compute)
            else:
                cute.arch.setmaxregister_decrease(self.regs_compute)
            self.compute_role(tS, tdP, frP, sLSE, sDelta, sdS, mma_dV_c,
                              tile_env, pipe_tile, sTile, sm_scale,
                              pipe_S, pipe_P, pipe_lse, pipe_dP, pipe_dS, mProg, pbase)
            self.mark(mProg, pbase, 49, 0)
        elif warp_idx <= self.reduce_warps[-1]:
            if cutlass.const_expr(self.regs_reduce > self.regs_launch):
                cute.arch.setmaxregister_increase(self.regs_reduce)
            else:
                cute.arch.setmaxregister_decrease(self.regs_reduce)
            self.reduce_role(tdQ, tmadQ, mdQacc, sStg, tile_env, pipe_tile, sTile,
                             pipe_dQ, bar_dOe, bar_dVgo, mProg, pbase, mdK, mdV, pipe_dKdV, tmadK, tmadV, mdQraw)
            self.mark(mProg, pbase, 59, 0)
        else:
            cute.arch.setmaxregister_decrease(self.regs_other)
        self.bar_cta.arrive_and_wait()
        if cutlass.const_expr(self.ENDTIME):
            tidx1 = cute.arch.thread_idx()[0]
            if tidx1 == 0:
                t_end = _globaltimer()
                mProg[4 * lin + 2] = Int32(t_end & 0xffffffff)
                mProg[4 * lin + 3] = Int32(t_end >> 32)
        if warp_idx == self.compute_warps[0]:
            tptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holder)
            cute.arch.dealloc_tmem(tptr, self.tmem_cols)

    # ------------------------------------------------------------------ load warp
    @cute.jit
    def load_vec(self, atom, src_f, dst_f, base, lane, per_thr, S, hb, stage):
        for i in cutlass.range_constexpr(per_thr):
            r = base + lane * per_thr + i
            if r < S:
                cute.copy(atom, src_f[None, r, hb], dst_f[None, lane * per_thr + i, stage])
            else:
                dst_f[None, lane * per_thr + i, stage].fill(0.0)

    @cute.jit
    def load_role(self, mK, mV, mQ, mdO, mQraw, mdOraw, mLSE2, mDelta, sK, sV, sQ, sdO, sQr, sdOr, sLSE, sDelta,
                  mma_S, mma_dP, tmaK, tmaV, tmaQ, tmadO, tile_env, pipe_tile, sTile, mCnt, lin,
                  pipe_K, pipe_V, bar_Qf, bar_Qe, bar_dOf, bar_dOe, pipe_lse, mProg, pbase):
        tidx = cute.arch.thread_idx()[0]
        lane = tidx % 32
        gK = cute.local_tile(mK, cute.select(self.tiler_S, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(mQ, cute.select(self.tiler_S, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(mV, cute.select(self.tiler_dP, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(mdO, cute.select(self.tiler_dP, mode=[1, 2]), (None, None, None))
        thrS = mma_S.get_slice(0)
        thrdP = mma_dP.get_slice(0)
        tKg = thrS.partition_A(gK)
        tQg = thrS.partition_B(gQ)
        tVg = thrdP.partition_A(gV)
        tdOg = thrdP.partition_B(gdO)
        tKs, tKg = cpasync.tma_partition(tmaK, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tKg, 0, 3))
        tQs, tQg = cpasync.tma_partition(tmaQ, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tQg, 0, 3))
        tVs, tVg = cpasync.tma_partition(tmaV, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tVg, 0, 3))
        tdOs, tdOg = cpasync.tma_partition(tmadO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tdOg, 0, 3))

        pQ = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.st_Q)
        pdO = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.st_dO)
        rows_per_lane = self.TQ // 32
        plse = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.st_vec)

        vec_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS), Float32, num_bits_per_copy=32)
        tile_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS), self.io_dtype, num_bits_per_copy=128)
        gQt = cute.local_tile(mQraw, (self.TQ, self.D), (None, None, None))    # (q, d, qi, 0, hb) raw gmem (cp.async source)
        gdOt = cute.local_tile(mdOraw, (self.TQ, self.D), (None, None, None))
        sLSE_f = cute.flat_divide(sLSE, (1,))
        sDelta_f = cute.flat_divide(sDelta, (1,))
        mLSE2_f = cute.flat_divide(mLSE2, (1,))
        mDelta_f = cute.flat_divide(mDelta, (1,))
        per_thr = self.TQ // 32
        n_tiles, n_hb, nk, Hkv, nq, G, S = tile_env
        pT = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        pK = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pV = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pipe_tile.producer_acquire(pT)
        if lane == 0:
            sTile[pT.index] = lin
        cute.arch.sync_warp()
        tile = sTile[pT.index]
        cute.arch.sync_warp()
        pipe_tile.producer_commit(pT)
        pT.advance()
        while tile < n_tiles:
            bidx, hk, b = self.tile_coords(tile, n_hb, nk, Hkv)
            q_start = bidx * (self.TK // self.TQ)
            n_qtiles = nq - q_start
            trips = n_qtiles * G
            kvb = (hk, b)
            self.mark(mProg, pbase, 64, tile)
            pipe_K.producer_acquire(pK)            # K buffer free: the previous tile's last dQ completed
            self.mark(mProg, pbase, 65, tile)
            bar = pipe_K.producer_get_barrier(pK)
            cute.copy(tmaK, tKg[(None, bidx, 0, kvb)], tKs[None, 0], tma_bar_ptr=bar)
            pK.advance()
            g = Int32(0)
            qi = q_start
            for it in cutlass.range(trips):
                hb = ((g, hk), b)
                self.mark(mProg, pbase, 60, it)
                cute.arch.mbarrier_wait(bar_Qe + pQ.index, pQ.phase)
                self.mark(mProg, pbase, 61, it)
                if cutlass.const_expr(self.QCP):
                    for rr in cutlass.range_constexpr(rows_per_lane):
                        r = lane * rows_per_lane + rr
                        g_row8 = cute.logical_divide(gQt[(r, None, qi, 0, hb)], cute.make_layout(8))
                        s_row8 = cute.logical_divide(sQr[r, None, pQ.index], cute.make_layout(8))
                        for c8 in cutlass.range_constexpr(self.D // 8):
                            cute.copy(tile_atom, g_row8[None, c8], s_row8[None, c8])
                    cute.arch.cp_async_mbarrier_arrive_noinc(bar_Qf + pQ.index)
                elif cutlass.const_expr(self.ABL & 128):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_Qf + pQ.index)      # ablation: no Q load (stale smem, wrong results)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_Qf + pQ.index, self.bytes_Q)
                    cute.copy(tmaQ, tQg[(None, qi, 0, hb)], tQs[None, pQ.index], tma_bar_ptr=bar_Qf + pQ.index)
                pQ.advance()
                self.mark(mProg, pbase, 62, it)
                pipe_lse.producer_acquire(plse)
                self.load_vec(vec_atom, mLSE2_f, sLSE_f, self.TQ * qi, lane, per_thr, S, hb, plse.index)
                self.load_vec(vec_atom, mDelta_f, sDelta_f, self.TQ * qi, lane, per_thr, S, hb, plse.index)
                pipe_lse.producer_commit(plse)
                plse.advance()
                self.mark(mProg, pbase, 63, it)
                cute.arch.mbarrier_wait(bar_dOe + pdO.index, pdO.phase)     # staging/dO stage released by the reduce warps
                self.mark(mProg, pbase, 66, it)
                if cutlass.const_expr(self.DOCP):
                    for rr in cutlass.range_constexpr(rows_per_lane):
                        r = lane * rows_per_lane + rr
                        g_row8 = cute.logical_divide(gdOt[(r, None, qi, 0, hb)], cute.make_layout(8))
                        s_row8 = cute.logical_divide(sdOr[r, None, pdO.index], cute.make_layout(8))
                        for c8 in cutlass.range_constexpr(self.D // 8):
                            cute.copy(tile_atom, g_row8[None, c8], s_row8[None, c8])
                    cute.arch.cp_async_mbarrier_arrive_noinc(bar_dOf + pdO.index)
                elif cutlass.const_expr(self.ABL & 64):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_dOf + pdO.index)    # ablation: no dO load (stale smem, wrong results)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_dOf + pdO.index, self.bytes_dO)
                    cute.copy(tmadO, tdOg[(None, qi, 0, hb)], tdOs[None, pdO.index], tma_bar_ptr=bar_dOf + pdO.index)
                pdO.advance()
                if it == 0:
                    pipe_V.producer_acquire(pV)        # V free after the previous tile's last dP
                    bar = pipe_V.producer_get_barrier(pV)
                    cute.copy(tmaV, tVg[(None, bidx, 0, kvb)], tVs[None, 0], tma_bar_ptr=bar)
                    pV.advance()
                qi += 1
                if qi == nq:
                    qi = q_start
                    g += 1
            pipe_tile.producer_acquire(pT)
            if lane == 0:
                sTile[pT.index] = cute.arch.atomic_add(mCnt.iterator, Int32(1))
            cute.arch.sync_warp()
            tile = sTile[pT.index]
            cute.arch.sync_warp()
            pipe_tile.producer_commit(pT)
            pT.advance()

    # ------------------------------------------------------------------ MMA warp
    @cute.jit
    def gemm_S(self, mma_S, tS, frK, frQ_s, nkS):
        mma_S.set(tcgen05.Field.ACCUMULATE, False)
        cute.gemm(mma_S, tS, frK[None, None, 0], frQ_s[None, None, 0], tS)
        mma_S.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range_constexpr(1, nkS):
            cute.gemm(mma_S, tS, frK[None, None, kb], frQ_s[None, None, kb], tS)

    @cute.jit
    def gemm_dP(self, mma_dP, tdP, frV, frdO_s, nkdP):
        mma_dP.set(tcgen05.Field.ACCUMULATE, False)
        cute.gemm(mma_dP, tdP, frV[None, None, 0], frdO_s[None, None, 0], tdP)
        mma_dP.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range_constexpr(1, nkdP):
            cute.gemm(mma_dP, tdP, frV[None, None, kb], frdO_s[None, None, kb], tdP)

    @cute.jit
    def gemm_dV(self, mma_dV, tdV, frP, frdOT_s, nkdV):
        cute.gemm(mma_dV, tdV, frP[None, None, 0], frdOT_s[None, None, 0], tdV)
        mma_dV.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range_constexpr(1, nkdV):
            cute.gemm(mma_dV, tdV, frP[None, None, kb], frdOT_s[None, None, kb], tdV)

    @cute.jit
    def gemm_dK(self, mma_dK, tdK, frdST_s, frQT_s, nkdK):
        cute.gemm(mma_dK, tdK, frdST_s[None, None, 0], frQT_s[None, None, 0], tdK)
        mma_dK.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range_constexpr(1, nkdK):
            cute.gemm(mma_dK, tdK, frdST_s[None, None, kb], frQT_s[None, None, kb], tdK)

    @cute.jit
    def gemm_dQ(self, mma_dQ, tdQ, frdSA_s, frKB_s, nkdQ):
        mma_dQ.set(tcgen05.Field.ACCUMULATE, False)
        cute.gemm(mma_dQ, tdQ, frdSA_s[None, None, 0], frKB_s[None, None, 0], tdQ)
        mma_dQ.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range_constexpr(1, nkdQ):
            cute.gemm(mma_dQ, tdQ, frdSA_s[None, None, kb], frKB_s[None, None, kb], tdQ)

    @cute.jit
    def mma_role(self, mma_S, mma_dP, mma_dV, mma_dK, mma_dQ,
                 tS, frK, frQ, tdP, frV, frdO, tdV, frP, frdOT,
                 tdQ, frdSA, frKB, tdK, frdST, frQT, tmem_holder, tile_env, pipe_tile, sTile,
                 pipe_K, pipe_V, bar_Qf, bar_Qe, bar_dOf, bar_dVgo, pipe_S, pipe_dP, pipe_dQ, pipe_P, pipe_dS, pipe_dKdV, mProg, pbase):
        cute.arch.alloc_tmem(cutlass.Int32(self.tmem_cols), tmem_holder)
        self.bar_tmem.arrive_and_wait()
        self.mark(mProg, pbase, 10, 0)
        cQ = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.st_Q)
        cQ_rel = cQ.clone()
        cdO = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.st_dO)
        cP = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        cdS = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        pS = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pdP = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pdQ_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pdQ_com = pdQ_acq.clone()
        pdKdV = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.st_dKdV)
        pdKdV_com = pdKdV.clone()
        nkS = cute.size(frQ, mode=[2])
        nkdP = cute.size(frdO, mode=[2])
        nkdV = cute.size(frP, mode=[2])
        nkdQ = cute.size(frKB, mode=[2])
        nkdK = cute.size(frdST, mode=[2])
        n_tiles, n_hb, nk, Hkv, nq, G, S = tile_env
        cT = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        cK = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        cV = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        frK_t = frK[None, None, None, 0]
        frKB_t = frKB[None, None, None, 0]
        frV_t = frV[None, None, None, 0]
        frdSA_t = frdSA[None, None, None, 0]
        frdST_t = frdST[None, None, None, 0]
        pipe_tile.consumer_wait(cT)
        tile = sTile[cT.index]
        pipe_tile.consumer_release(cT)
        cT.advance()
        while tile < n_tiles:
            bidx, bidy, bidz = self.tile_coords(tile, n_hb, nk, Hkv)
            q_start = bidx * (self.TK // self.TQ)
            n_qtiles = nq - q_start
            trips = n_qtiles * G
            mma_dV.set(tcgen05.Field.ACCUMULATE, False)
            mma_dK.set(tcgen05.Field.ACCUMULATE, False)
            self.mark(mProg, pbase, 24, tile)
            pipe_K.consumer_wait(cK)
            self.mark(mProg, pbase, 25, tile)
            # prologue: S(0) -> X, dP(0) -> Y
            self.mark(mProg, pbase, 11, 0)
            cute.arch.mbarrier_wait(bar_Qf + cQ.index, cQ.phase)
            self.mark(mProg, pbase, 12, 0)
            self.gemm_S(mma_S, tS, frK_t, frQ[None, None, None, cQ.index], nkS)
            mma_S.set(tcgen05.Field.ACCUMULATE, True)
            cQ.advance()
            pipe_S.producer_commit(pS)
            pS.advance()
            self.mark(mProg, pbase, 17, 0)
            pipe_dQ.producer_acquire(pdQ_acq)          # Y drained (previous tile's last dQ)
            pdQ_acq.advance()
            self.mark(mProg, pbase, 18, 0)
            pipe_V.consumer_wait(cV)
            cute.arch.mbarrier_wait(bar_dOf + cdO.index, cdO.phase)
            self.mark(mProg, pbase, 19, 0)
            self.gemm_dP(mma_dP, tdP, frV_t, frdO[None, None, None, cdO.index], nkdP)
            mma_dP.set(tcgen05.Field.ACCUMULATE, True)
            cdO.advance()
            pipe_dP.producer_commit(pdP)
            pdP.advance()
            if trips == 1:
                pipe_V.consumer_release(cV)        # one-trip tile: dP(0) was the last dP
                cV.advance()
            # dK/dV TMEM must have been drained by the previous tile's epilogue before the first dV/dK GEMM
            self.mark(mProg, pbase, 23, tile)
            pipe_dKdV.producer_acquire(pdKdV)
            pdKdV.advance()
            pipe_dKdV.producer_acquire(pdKdV)
            pdKdV.advance()

            for j in cutlass.range(trips - 1):
                # dV(j): P(j) overlays X; releases nothing (dO(j) is released by the reduce warps)
                self.mark(mProg, pbase, 20, j)
                pipe_P.consumer_wait(cP)
                cP.advance()
                if cutlass.const_expr(self.REDLATE):
                    if j > 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(bar_dVgo)
                self.mark(mProg, pbase, 21, j)
                self.gemm_dV(mma_dV, tdV, frP, frdOT[None, None, None, cQ_rel.index], nkdV)   # dO(j) stage index == trip parity == Q(j) stage
                mma_dV.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(self.ORDER == 0):
                    # S(j+1) -> X (in order after dV(j), which read P(j) from X)
                    self.mark(mProg, pbase, 11, j + 1)
                    cute.arch.mbarrier_wait(bar_Qf + cQ.index, cQ.phase)
                    self.mark(mProg, pbase, 12, j + 1)
                    self.gemm_S(mma_S, tS, frK_t, frQ[None, None, None, cQ.index], nkS)
                    mma_S.set(tcgen05.Field.ACCUMULATE, True)
                    cQ.advance()
                    pipe_S.producer_commit(pS)
                    pS.advance()
                    # dQ(j) -> Y (dS(j) ready => compute loaded dP(j)); dK(j)
                    self.mark(mProg, pbase, 13, j)
                    pipe_dS.consumer_wait(cdS)
                    cdS.advance()
                    self.mark(mProg, pbase, 15, j)
                    self.gemm_dQ(mma_dQ, tdQ, frdSA_t, frKB_t, nkdQ)
                    mma_dQ.set(tcgen05.Field.ACCUMULATE, True)
                    pipe_dQ.producer_commit(pdQ_com)
                    pdQ_com.advance()
                    self.gemm_dK(mma_dK, tdK, frdST_t, frQT[None, None, None, cQ_rel.index], nkdK)
                    mma_dK.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(bar_Qe + cQ_rel.index)    # Q(j) free (dK(j) done)
                cQ_rel.advance()
                # dP(j+1) -> Y after the reduce warps drained dQ(j)
                self.mark(mProg, pbase, 17, j + 1)
                pipe_dQ.producer_acquire(pdQ_acq)
                pdQ_acq.advance()
                self.mark(mProg, pbase, 18, j + 1)
                cute.arch.mbarrier_wait(bar_dOf + cdO.index, cdO.phase)
                self.mark(mProg, pbase, 19, j + 1)
                self.gemm_dP(mma_dP, tdP, frV_t, frdO[None, None, None, cdO.index], nkdP)
                mma_dP.set(tcgen05.Field.ACCUMULATE, True)
                cdO.advance()
                pipe_dP.producer_commit(pdP)
                pdP.advance()
                if j + 2 == trips:
                    pipe_V.consumer_release(cV)        # last dP of the tile: V free for the next tile
                    cV.advance()

            # tail: dV, dQ, dK of the last trip
            self.mark(mProg, pbase, 20, trips - 1)
            pipe_P.consumer_wait(cP)
            cP.advance()
            if cutlass.const_expr(self.REDLATE):
                if trips > 1:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_dVgo)
            self.mark(mProg, pbase, 21, trips - 1)
            self.gemm_dV(mma_dV, tdV, frP, frdOT[None, None, None, cQ_rel.index], nkdV)
            mma_dV.set(tcgen05.Field.ACCUMULATE, True)
            self.mark(mProg, pbase, 13, trips - 1)
            pipe_dS.consumer_wait(cdS)
            cdS.advance()
            self.mark(mProg, pbase, 15, trips - 1)
            self.gemm_dQ(mma_dQ, tdQ, frdSA_t, frKB_t, nkdQ)
            mma_dQ.set(tcgen05.Field.ACCUMULATE, True)
            pipe_dQ.producer_commit(pdQ_com)
            pdQ_com.advance()
            pipe_K.consumer_release(cK)               # K free once the last dQ completed (dK does not read K)
            cK.advance()
            self.gemm_dK(mma_dK, tdK, frdST_t, frQT[None, None, None, cQ_rel.index], nkdK)
            mma_dK.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(bar_Qe + cQ_rel.index)
            cQ_rel.advance()
            self.mark(mProg, pbase, 22, 0)
            pipe_dKdV.producer_commit(pdKdV_com)
            pdKdV_com.advance()
            pipe_dKdV.producer_commit(pdKdV_com)
            pdKdV_com.advance()
            self.mark(mProg, pbase, 26, tile)
            pipe_tile.consumer_wait(cT)
            tile = sTile[cT.index]
            pipe_tile.consumer_release(cT)
            cT.advance()
            self.mark(mProg, pbase, 27, tile)

    # ------------------------------------------------------------------ compute warps
    @cute.jit
    def split_wg(self, t, nwg, wg):
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], (nwg, cute.size(t, mode=[2]) // nwg))))
            return p[None, None, (wg, None)]
        else:
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], t.shape[2], (nwg, cute.size(t, mode=[3]) // nwg))))
            return p[None, None, None, (wg, None)]

    @cute.jit
    def exp2_pair_emul(self, x, y):
        """2^x, 2^y on the FMA/ALU pipes (no MUFU). Inputs <= 0 (clamped at -126)."""
        lo = Float32(-126.0)
        xc = cute.arch.fmax(x, lo)
        yc = cute.arch.fmax(y, lo)
        magic = Float32(12582912.0)                      # 1.5 * 2^23
        r = cute.arch.add_packed_f32x2((xc, yc), (magic, magic), rnd="rm")   # floor(x) in low mantissa bits
        back = cute.arch.sub_packed_f32x2(r, (magic, magic))
        f = cute.arch.sub_packed_f32x2((xc, yc), back)                        # fraction in [0, 1)
        c3 = Float32(0.0771190896630287)
        c2 = Float32(0.2275643944740295)
        c1 = Float32(0.6951461434364319)
        c0 = Float32(1.0)
        p = cute.arch.fma_packed_f32x2((c3, c3), f, (c2, c2))
        p = cute.arch.fma_packed_f32x2(p, f, (c1, c1))
        p = cute.arch.fma_packed_f32x2(p, f, (c0, c0))                         # 2^f in [1, 2)
        return _scale_by_floor(r[0], p[0]), _scale_by_floor(r[1], p[1])

    @cute.jit
    def to_bf16(self, src):
        out = cute.make_rmem_tensor(src.shape, self.io_dtype)
        n = cute.size(src) // 4
        s4 = cute.logical_divide(src, cute.make_layout(4))
        o4 = cute.make_tensor(out.iterator, s4.layout)
        for i in cutlass.range(n, unroll_full=True):
            o4[None, i].store(s4[None, i].load().to(self.io_dtype))
        return out

    @cute.jit
    def compute_role(self, tS, tdP, frP, sLSE, sDelta, sdS, mma_dV,
                     tile_env, pipe_tile, sTile, sm_scale,
                     pipe_S, pipe_P, pipe_lse, pipe_dP, pipe_dS, mProg, pbase):
        tidx = cute.arch.thread_idx()[0]
        cS = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        cdP = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        pP = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        pdS = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        clse = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.st_vec)

        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        st_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), self.io_dtype)
        cS_id = cute.make_identity_tensor(cute.select(self.tiler_S, mode=[0, 1]))
        nwg = self.num_compute_warps // 4
        lane128 = tidx % 128
        wg = ((tidx - self.compute_warps[0] * 32) % (self.num_compute_warps * 32)) // 128
        tS_2 = tS[(None, None), 0, 0]
        tdP_2 = tdP[(None, None), 0, 0]
        t2r = tcgen05.make_tmem_copy(ld_atom, tS_2)
        thr_t2r = t2r.get_slice(lane128)
        tTR_c = self.split_wg(thr_t2r.partition_D(cS_id), nwg, wg)          # coords (k, q)
        tTR_rS = cute.make_rmem_tensor(tTR_c.shape, self.acc_dtype)
        tTR_rdP = cute.make_rmem_tensor(tTR_c.shape, self.acc_dtype)
        tTR_tS = self.split_wg(thr_t2r.partition_S(tS_2), nwg, wg)
        tTR_tdP = self.split_wg(thr_t2r.partition_S(tdP_2), nwg, wg)
        cP_id = mma_dV.get_slice(0).partition_A(cS_id)
        r2t = tcgen05.make_tmem_copy(st_atom, frP)
        thr_r2t = r2t.get_slice(lane128)
        tRT_tP = self.split_wg(thr_r2t.partition_D(frP), nwg, wg)
        tRT_c = self.split_wg(thr_r2t.partition_S(cP_id), nwg, wg)
        dS_thread_layout = cute.make_ordered_layout((self.TK, self.TQ), (1, 0))
        tTR_c_full = thr_t2r.partition_D(cS_id)
        sdS_st = sdS[None, None, None, 0]
        sdS_rows = cute.composition(sdS_st, dS_thread_layout)
        sdS_mine = cute.composition(sdS_rows[lane128, None], cute.make_layout(tTR_c_full.shape))
        sdS_mine = self.split_wg(sdS_mine, nwg, wg)
        if cutlass.const_expr(self.PRINT):
            print("frP.layout", frP.layout)
            print("tRT_tP.layout", tRT_tP.layout, "full", thr_r2t.partition_D(frP).layout)
            print("tRT_c.layout", tRT_c.layout)
            print("tTR_c.layout", tTR_c.layout, "full", tTR_c_full.layout)
            print("tTR_tS.layout", tTR_tS.layout)
            print("sdS_mine", sdS_mine.layout)
        c = sm_scale * Float32(LOG2E)
        sc = sm_scale

        n_tiles, n_hb, nk, Hkv, nq, G, S = tile_env
        cT = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        pipe_tile.consumer_wait(cT)
        tile = sTile[cT.index]
        pipe_tile.consumer_release(cT)
        cT.advance()
        while tile < n_tiles:
            bidx, bidy, bidz = self.tile_coords(tile, n_hb, nk, Hkv)
            q_start = bidx * (self.TK // self.TQ)
            n_qtiles = nq - q_start
            trips = n_qtiles * G
            g = Int32(0)
            qi = q_start
            for t in cutlass.range(trips):
                self.mark(mProg, pbase, 30, t)
                pipe_S.consumer_wait(cS)
                cS.advance()
                self.mark(mProg, pbase, 32, t)
                pipe_lse.consumer_wait(clse)
                self.mark(mProg, pbase, 31, t)
                if cutlass.const_expr(not (self.ABL & 32)):
                    cute.copy(t2r, tTR_tS, tTR_rS)
                    if qi * self.TQ < (bidx + 1) * self.TK:   # diagonal block: mask k > q
                        for i in cutlass.range(cute.size(tTR_rS), unroll_full=True):
                            kk = cute.get(tTR_c[i], mode=[0]) + bidx * self.TK
                            qq = cute.get(tTR_c[i], mode=[1]) + qi * self.TQ
                            if kk > qq:
                                tTR_rS[i] = -Float32(float("inf"))
                    sL4 = cute.logical_divide(sLSE[None, clse.index], cute.make_layout(4))
                    for j in cutlass.range_constexpr(cute.size(tTR_rS) // 4):
                        l4 = sL4[None, cute.get(tTR_c[4 * j], mode=[1]) // 4].load()
                        for h in cutlass.range_constexpr(0, 4, 2):
                            i = 4 * j + h
                            tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2((tTR_rS[i], tTR_rS[i + 1]), (c, c), (l4[h], l4[h + 1]))
                            if cutlass.const_expr(self.EMUL > 0 and ((2 * j + h // 2) % self.EMUL) == 0):
                                tTR_rS[i], tTR_rS[i + 1] = self.exp2_pair_emul(tTR_rS[i], tTR_rS[i + 1])
                            else:
                                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)
                    rP = self.to_bf16(tTR_rS)
                    rP2 = cute.make_tensor(rP.iterator, cute.make_layout(tRT_c.shape))
                    cute.arch.fence_view_async_tmem_load()
                    cute.copy(r2t, rP2, tRT_tP)
                    cute.arch.fence_view_async_tmem_store()
                if cutlass.const_expr(self.WARR):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_P.producer_commit(pP)
                else:
                    pipe_P.producer_commit(pP)
                pP.advance()

                self.mark(mProg, pbase, 36, t)
                pipe_dP.consumer_wait(cdP)
                cdP.advance()
                self.mark(mProg, pbase, 37, t)
                if cutlass.const_expr(not (self.ABL & 32)):
                    cute.copy(t2r, tTR_tdP, tTR_rdP)
                    sD4 = cute.logical_divide(sDelta[None, clse.index], cute.make_layout(4))
                    for j in cutlass.range_constexpr(cute.size(tTR_rdP) // 4):
                        d4 = sD4[None, cute.get(tTR_c[4 * j], mode=[1]) // 4].load()
                        for h in cutlass.range_constexpr(0, 4, 2):
                            i = 4 * j + h
                            tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.fma_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (sc, sc), (d4[h], d4[h + 1]))
                            if cutlass.const_expr(self.PBF16):
                                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (rP[i].to(Float32), rP[i + 1].to(Float32)))
                            else:
                                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1]))
                    rdS = self.to_bf16(tTR_rdP)
                    cute.arch.fence_view_async_tmem_load()
                    cute.autovec_copy(rdS, sdS_mine)
                    cute.arch.fence_proxy("async.shared", space="cta")
                if cutlass.const_expr(self.WARR):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_dS.producer_commit(pdS)
                        pipe_lse.consumer_release(clse)
                else:
                    pipe_dS.producer_commit(pdS)
                    pipe_lse.consumer_release(clse)
                pdS.advance()
                clse.advance()
                self.mark(mProg, pbase, 38, t)
                qi += 1
                if qi == nq:
                    qi = q_start
                    g += 1
            self.mark(mProg, pbase, 44, tile)
            pipe_tile.consumer_wait(cT)
            tile = sTile[cT.index]
            pipe_tile.consumer_release(cT)
            cT.advance()
            self.mark(mProg, pbase, 45, tile)

    # ------------------------------------------------------------------ reduce warps
    @cute.jit
    def drain_rows(self, t2r, tt_src, tc_shape, sRow, nchunk):
        """TMEM (lane = row) -> fp32 regs -> bf16 -> 16-byte stores into this lane's swizzled staging row."""
        nv = cute.size(tc_shape) // 8
        row8 = cute.logical_divide(sRow, cute.make_layout(8))
        if cutlass.const_expr(self.DRAINDYN):
            for ch in cutlass.range(nchunk):
                tr = cute.make_rmem_tensor(tc_shape, self.acc_dtype)
                cute.copy(t2r, tt_src[None, None, ch], tr)
                trb = self.to_bf16(tr)
                if cutlass.const_expr(not (self.ABL & 2)):
                    tb8 = cute.logical_divide(trb, cute.make_layout(8))
                    for v in cutlass.range_constexpr(nv):
                        row8[None, ch * nv + v].store(tb8[None, v].load())
        elif cutlass.const_expr(self.DRAINPIPE):
            tr = cute.make_rmem_tensor(tc_shape, self.acc_dtype)
            cute.copy(t2r, tt_src[None, None, 0], tr)
            for ch in cutlass.range_constexpr(nchunk):
                if cutlass.const_expr(ch + 1 < nchunk):
                    tr_next = cute.make_rmem_tensor(tc_shape, self.acc_dtype)
                    cute.copy(t2r, tt_src[None, None, ch + 1], tr_next)
                trb = self.to_bf16(tr)
                if cutlass.const_expr(not (self.ABL & 2)):
                    tb8 = cute.logical_divide(trb, cute.make_layout(8))
                    for v in cutlass.range_constexpr(nv):
                        row8[None, ch * nv + v].store(tb8[None, v].load())
                if cutlass.const_expr(ch + 1 < nchunk):
                    tr = tr_next
        else:
            for ch in cutlass.range_constexpr(nchunk):
                tr = cute.make_rmem_tensor(tc_shape, self.acc_dtype)
                cute.copy(t2r, tt_src[None, None, ch], tr)
                trb = self.to_bf16(tr)
                if cutlass.const_expr(not (self.ABL & 2)):
                    tb8 = cute.logical_divide(trb, cute.make_layout(8))
                    for v in cutlass.range_constexpr(nv):
                        row8[None, ch * nv + v].store(tb8[None, v].load())

    @cute.jit
    def reduce_role(self, tdQ, tmadQ, mdQacc, sdO, tile_env, pipe_tile, sTile,
                    pipe_dQ, bar_dOe, bar_dVgo, mProg, pbase, mdK, mdV, pipe_dKdV, tmadK, tmadV, mdQraw):
        sStg_all = sdO
        dvgo_ph = Int32(0)
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.reduce_warps[0]
        thread_idx = tidx % (self.num_reduce_warps * 32)
        lane = thread_idx % 32
        cdQ = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        cdO = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.st_dO)   # tracks the dO ring (staging)
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.DRAINX)), self.acc_dtype)
        gdQ = cute.local_tile(mdQacc, (self.TQ, self.D), (None, None, None))    # (q-tile, 0, hb)
        cid = cute.make_identity_tensor((self.TQ, self.D))
        tdQ_2 = tdQ[(None, None), 0, 0]
        t2r = tcgen05.make_tmem_copy(ld_atom, tdQ_2)
        thr = t2r.get_slice(thread_idx)
        tc = thr.partition_D(cid)
        tt = thr.partition_S(tdQ_2)
        nchunk = cute.size(tt, mode=[2])
        tdV_2 = cute.make_tensor(cute.make_ptr(self.acc_dtype, self.tmem_dV, cute.AddressSpace.tmem, assumed_align=256), tdQ_2.layout)
        tdK_2 = cute.make_tensor(cute.make_ptr(self.acc_dtype, self.tmem_dK, cute.AddressSpace.tmem, assumed_align=256), tdQ_2.layout)
        ttV = thr.partition_S(tdV_2)
        ttK = thr.partition_S(tdK_2)
        tdQs, tdQg = cpasync.tma_partition(tmadQ, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdQ, 0, 2))
        gdK = cute.local_tile(mdK, (self.TK, self.D), (None, None, None))
        gdV = cute.local_tile(mdV, (self.TK, self.D), (None, None, None))
        tKs, tKg = cpasync.tma_partition(tmadK, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdK, 0, 2))
        tVs, tVg = cpasync.tma_partition(tmadV, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdV, 0, 2))
        tc_shape = tc[None, None, 0].shape
        if cutlass.const_expr(self.PRINT):
            print("reduce tc", tc.layout, "tt", tt.layout, "tc_shape", tc_shape)
        n_tiles, n_hb, nk, Hkv, nq, G, S = tile_env
        cT = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        cdKdV = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.st_dKdV)
        pipe_tile.consumer_wait(cT)
        tile = sTile[cT.index]
        pipe_tile.consumer_release(cT)
        cT.advance()
        while tile < n_tiles:
            bidx, bidy, bidz = self.tile_coords(tile, n_hb, nk, Hkv)
            q_start = bidx * (self.TK // self.TQ)
            n_qtiles = nq - q_start
            trips = n_qtiles * G
            kvb = (bidy, bidz)
            g = Int32(0)
            qi = q_start
            stg = Int32(0)
            for it in cutlass.range(trips):
                hb = ((g, bidy), bidz)
                stg = cdO.index
                self.mark(mProg, pbase, 50, it)
                pipe_dQ.consumer_wait(cdQ)
                self.mark(mProg, pbase, 51, it)
                # dO(it) (same stage) is dead: dV(it) completed before dQ(it) (in-order MMA)
                self.drain_rows(t2r, tt, tc_shape, sdO[thread_idx, None, stg], nchunk)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.WARR):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_dQ.consumer_release(cdQ)      # Y free: the MMA warp may issue dP(it+1)
                else:
                    pipe_dQ.consumer_release(cdQ)
                cdQ.advance()
                cute.arch.fence_proxy("async.shared", space="cta")
                self.bar_reduce.arrive_and_wait()
                self.mark(mProg, pbase, 52, it)
                if cutlass.const_expr(self.DQRED):
                    # each warp: its 32 staged rows, two 256-byte rows per red instruction (16 lanes x 16 B each)
                    r2 = lane // 16
                    d8 = lane % 16
                    qrow0 = warp_idx * 32 + r2
                    off0 = cute.crd2idx((qi * self.TQ + qrow0, 8 * d8, hb), mdQraw.layout)
                    sq = mdQraw.stride[0]
                    for k in cutlass.range_constexpr(16):
                        qrow = qrow0 + 2 * k
                        row8 = cute.logical_divide(sStg_all[qrow, None, stg], cute.make_layout(8))
                        r8 = cute.make_rmem_tensor((8,), self.io_dtype)
                        r8.store(row8[None, d8].load())
                        r4 = cute.make_tensor(cute.recast_ptr(r8.iterator, dtype=Int32), cute.make_layout(4))
                        if qi * self.TQ + qrow < S:
                            _red_add_v4_bf16x2((mdQraw.iterator + (off0 + 2 * k * sq)).toint(), r4[0], r4[1], r4[2], r4[3])
                    self.bar_reduce.arrive_and_wait()          # staging fully read back
                    if warp_idx == 0:
                        if it + 1 < trips:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bar_dOe + stg)      # stage free for dO(it+2)
                elif warp_idx == 0:
                    if cutlass.const_expr(self.ABL & 8):
                        if it + 1 < trips:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bar_dOe + stg)      # ablation: release before the TMA read (wrong results)
                    if cutlass.const_expr(self.REDLATE):
                        if it + 1 < trips:
                            cute.arch.mbarrier_wait(bar_dVgo, dvgo_ph)    # dV(it+1) (A from TMEM) is being issued
                            dvgo_ph = dvgo_ph ^ 1
                    if cutlass.const_expr(not (self.ABL & 4)):
                        cute.copy(tmadQ, tdQs[None, stg], tdQg[None, qi, 0, hb])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    if cutlass.const_expr(not (self.ABL & 8)):
                        if it + 1 < trips:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bar_dOe + stg)      # stage free for dO(it+2)
                self.mark(mProg, pbase, 53, it)
                cdO.advance()
                qi += 1
                if qi == nq:
                    qi = q_start
                    g += 1
            # ---- epilogue: dV then dK through the last trip's stage (its dQ reduce has finished reading)
            self.mark(mProg, pbase, 55, 0)
            self.bar_reduce.arrive_and_wait()          # warp 0's wait_group of the last trip is done
            for which in cutlass.range_constexpr(2):
                pipe_dKdV.consumer_wait(cdKdV)
                if cutlass.const_expr(which == 0):
                    self.drain_rows(t2r, ttV, tc_shape, sdO[thread_idx, None, stg], nchunk)
                else:
                    self.drain_rows(t2r, ttK, tc_shape, sdO[thread_idx, None, stg], nchunk)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.WARR):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_dKdV.consumer_release(cdKdV)
                else:
                    pipe_dKdV.consumer_release(cdKdV)
                cdKdV.advance()
                cute.arch.fence_proxy("async.shared", space="cta")
                self.bar_reduce.arrive_and_wait()
                if warp_idx == 0:
                    if cutlass.const_expr(which == 0):
                        cute.copy(tmadV, tVs[None, stg], tVg[None, bidx, 0, kvb])
                    else:
                        cute.copy(tmadK, tKs[None, stg], tKg[None, bidx, 0, kvb])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    if cutlass.const_expr(which == 1):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(bar_dOe + stg)      # stage free for the next tile's dO
                if cutlass.const_expr(which == 0):
                    self.bar_reduce.arrive_and_wait()  # dV store finished reading before dK is staged
            self.mark(mProg, pbase, 56, 0)
            pipe_tile.consumer_wait(cT)
            tile = sTile[cT.index]
            pipe_tile.consumer_release(cT)
            cT.advance()


# ---------------------------------------------------------------------- torch entry
_KERNEL = CausalAttnBwdSm100()
_CACHE = {}


def _prep(q, k, v, out, dout, lse):
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    G = H // Hkv

    def qlike(t):  # (B,S,H,D) -> (S, D, G, Hkv, B)
        t5 = t.view(B, S, Hkv, G, D).permute(1, 4, 3, 2, 0)
        ct = from_dlpack(t5, assumed_align=16)
        ct.element_type = BFloat16
        return ct.mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(
            mode=1, stride_order=(4, 0, 3, 2, 1), divisibility=64)

    def klike(t):  # (B,S,Hkv,D) -> (S, D, Hkv, B)
        t4 = t.permute(1, 3, 2, 0)
        ct = from_dlpack(t4, assumed_align=16)
        ct.element_type = BFloat16
        return ct.mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(
            mode=1, stride_order=(3, 0, 2, 1), divisibility=64)

    lse4 = lse.view(B, Hkv, G, S).permute(3, 2, 1, 0)   # (S, G, Hkv, B)
    clse = from_dlpack(lse4, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    return qlike, klike, clse, (S, G, Hkv, B)


def run(q, k, v, out, dout, lse, sm_scale):
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    assert D == 128 and H % Hkv == 0
    q, k, v, out, dout = [t if t.is_contiguous() else t.contiguous() for t in (q, k, v, out, dout)]
    lse = lse.contiguous().float()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    key = (B, S, H, Hkv)
    ent = _CACHE.get(key)
    nvec = B * H * S
    if ent is None:
        ws = torch.empty(2 * nvec * 4, dtype=torch.uint8, device=q.device)
        ent = {"ws": ws}
        _CACHE[key] = ent
    ws = ent["ws"]
    if "cnt" not in ent:
        ent["cnt"] = torch.zeros(8, dtype=torch.int32, device=q.device)
    if "prog" not in ent:
        n_cta = (S // 128 + 1) * Hkv * B
        if _KERNEL.DEBUG:
            ent["prog"] = torch.zeros(n_cta * 24 * 2, dtype=torch.int32, pin_memory=True)
        else:
            ent["prog"] = torch.zeros(4 * 1024, dtype=torch.int32, device=q.device)
    prog = ent["prog"]
    qlike, klike, clse, prob = _prep(q, k, v, out, dout, lse)
    cws = from_dlpack(ws, assumed_align=16).mark_layout_dynamic()
    cprog = from_dlpack(prog, assumed_align=16).mark_layout_dynamic()
    ccnt = from_dlpack(ent["cnt"], assumed_align=16).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (prob, qlike(q), klike(k), klike(v), qlike(out), qlike(dout), clse,
            qlike(dq), klike(dk), klike(dv), cws, Float32(float(sm_scale)), cprog, ccnt, stream)
    fn = ent.get("fn")
    if fn is None:
        fn = cute.compile(_KERNEL, *args, options="--opt-level=2 iket" if _KERNEL.TRACE else "--opt-level=2")
        ent["fn"] = fn
    fn(*args)
    return dq, dk, dv
