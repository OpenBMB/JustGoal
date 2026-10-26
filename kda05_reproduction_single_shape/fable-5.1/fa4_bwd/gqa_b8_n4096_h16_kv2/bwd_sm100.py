"""Blackwell (sm_100a) causal GQA attention backward in CuTe DSL.

Own implementation of the FlashAttention backward for the fixed problem family
(head_dim 128, 16 query heads, bshd layout, causal). Architecture: KV-stationary
warp-specialized kernel with tcgen05 MMAs, TMEM accumulators, TMA loads and a
TMA reduce-add path for dQ (design reference: NVIDIA CUTLASS CuTe DSL
blackwell/fmha_bwd.py example, BSD-3).

Kernels launched by `AttnBwdSm100.__call__` (one host JIT function):
  1. preprocess : ndelta = -rowsum(O*dO), lse2 = -lse*log2(e), dq_acc = 0
  2. bwd        : one CTA per (kv tile, kv head, batch); loops over the query
                  heads sharing the kv head and the causal q tiles.
  3. convert    : dq = bf16(sm_scale * dq_acc)
"""
import math
import os as _os
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int32, Float32, Boolean

LOG2E = 1.4426950408889634


def _ik_push(flag, name):
    """Optional IKET range start (compiled in only when the kernel is built with ABWD_IKET=1)."""
    if flag:
        cute.experimental.iket.range_push(name)


def _ik_pop(flag):
    if flag:
        cute.experimental.iket.range_pop()


class AttnBwdSm100:
    def __init__(self, B: int, S: int, H: int, HKV: int, D: int = 128,
                 q_stages: int = 2, do_stages: int = 1):
        assert D == 128 and H % HKV == 0 and S % 128 == 0
        self.B, self.S, self.H, self.HKV, self.D = B, S, H, HKV, D
        self.h_r = H // HKV
        self.h_k = HKV
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        # dq accumulation buffer: fp16 (sm_scale applied before accumulation, |dq| <= ~4 on this family) halves the
        # TMA reduce-add / staging traffic; fp32 path kept for A/B (ABWD_DQ_F16=0)
        self.dq_f16 = int(_os.environ.get("ABWD_DQ_F16", 1))
        self.dq_dtype = cutlass.Float16 if self.dq_f16 else cutlass.Float32

        self.tile_q = 128
        self.tile_k = 128
        self.num_q_tiles = S // self.tile_q
        self.num_k_tiles = S // self.tile_k

        # MMA tilers (M, N, K)
        self.KQ_tiler = (self.tile_k, self.tile_q, D)     # S^T  = K  Q^T
        self.VdO_tiler = (self.tile_k, self.tile_q, D)    # dP^T = V  dO^T
        self.PdO_tiler = (self.tile_k, D, self.tile_q)    # dV  += P^T dO
        self.dSQ_tiler = (self.tile_k, D, self.tile_q)    # dK  += dS^T Q
        self.dSK_tiler = (self.tile_q, D, self.tile_k)    # dQ   = dS  K

        # warp roles
        self.reduce_warps = (0, 1, 2, 3)
        self.compute_warps = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp = 12
        self.load_warp = 13
        self.num_reduce_warps = 4
        self.num_compute_warps = 8
        self.threads_per_cta = 32 * 16

        # pipeline stages
        self.q_stages = q_stages
        self.do_stages = do_stages
        self.lse_stages = 1
        self.delta_stages = 1
        self.S_stages = 1
        self.dP_stages = 1
        self.dQ_stages = 1
        self.P_stages = 1
        self.dS_stages = 1
        self.dKdV_stages = 2
        # columns per dQ staging subtile (TMA reduce box): 64 fp16 columns = 16 KB per stage (2 TMA ops per tile)
        self.dq_sub_cols = int(_os.environ.get("ABWD_DQ_SUB", 64 if self.dq_f16 else 32))
        self.dq_store_stages = int(_os.environ.get("ABWD_DQ_STAGES", 2))
        self.no_dq_store = int(_os.environ.get("ABWD_NO_DQ_STORE", 0))   # diagnostic only: skip the TMA reduce-add of dQ (wrong dq)

        # TMEM column map (512 columns of fp32)
        self.tmem_dK = 0
        self.tmem_dV = 128
        self.tmem_dQ = 256
        self.tmem_dP = 256   # dP^T and dQ alias (dP is consumed before dQ is produced)
        self.tmem_S = 384    # P^T (bf16) overlays S^T
        self.tmem_cols = 512

        # setmaxnreg is warpgroup-collective: WG0 = reduce warps, WG1/WG2 = compute warps,
        # WG3 = mma + load + 2 idle warps (all four must use the same value)
        # 152/144/56: MMA warp no longer spills ~150 LDL/STL in its issue loop (40 regs); main kernel 1287 -> ~1236 us with split dS
        self.regs_compute = int(_os.environ.get("ABWD_REGS_COMPUTE", 152))
        self.regs_reduce = int(_os.environ.get("ABWD_REGS_REDUCE", 144))
        self.regs_wg3 = int(_os.environ.get("ABWD_REGS_WG3", 56))
        self.regs_mma = self.regs_load = self.regs_empty = self.regs_wg3
        assert 8 * self.regs_compute + 4 * self.regs_reduce + 4 * self.regs_wg3 <= 2048

        self.iket = bool(int(_os.environ.get("ABWD_IKET", 0)))
        self.p_var = int(_os.environ.get("ABWD_PVAR", 0))
        self.p1_off = int(_os.environ.get("ABWD_P1OFF", 64))
        self.ds_tmem = int(_os.environ.get("ABWD_DS_TMEM", 1))
        self.split_ds = self.ds_tmem and int(_os.environ.get("ABWD_SPLIT_DS", 1))   # separate ready signals for dS^T (TMEM, dK) and dS (smem, dQ)
        # each compute warpgroup writes its P^T / dS^T half into the first 32 columns of its own S^T / dP^T read region
        # (k-blocks 4..7 of the dV / dK A operand then start 32 columns later): no cross-warpgroup barrier needed
        self.wg_place = int(_os.environ.get("ABWD_WG_PLACE", 0))
        self.wg_bar = int(_os.environ.get("ABWD_WG_BAR", 0))        # debug: keep the P (1) / dS (2) compute barriers with wg_place
        self.swp = int(_os.environ.get("ABWD_SWP", 0))              # software-pipelined compute loop (TMEM loads issued ahead of use when data is already there)
        self.swp_nowait = int(_os.environ.get("ABWD_SWP_NOWAIT", 0))  # skip tcgen05.wait::ld before the P^T store when a dP load is in flight
        self.p_bf16 = int(_os.environ.get("ABWD_P_BF16", 0))        # dS uses the bf16-rounded P (frees 32 registers)
        if self.wg_place:
            self.p_var = 0
        self.vec_consts = int(_os.environ.get("ABWD_VEC", 1))        # LSE/delta column constants via 16-byte smem loads
        self.cvt_alu = int(_os.environ.get("ABWD_CVT_ALU", 0))        # of every 4 groups of 4 values, this many are converted to bf16 with integer ops (ALU pipe)
        self.cvt_trunc = int(_os.environ.get("ABWD_CVT_TRUNC", 0))    # ALU conversion path truncates instead of rounding to nearest even
        self.exp_fma = int(_os.environ.get("ABWD_EXP_FMA", 0))        # of every 4 groups of 4 values, this many use the FMA-pipe exp2 emulation
        self.shfl_consts = int(_os.environ.get("ABWD_SHFL", 0))      # LSE/delta column constants via warp shuffles     # dK reads dS^T from TMEM (A operand) instead of smem   # TMEM column offset of WG1's P^T half inside the S region
        self.cta_sync = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.tmem_alloc_sync = pipeline.NamedBarrier(barrier_id=2, num_threads=32)
        self.compute_sync = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_compute_warps)
        self.epilogue_sync = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * self.num_compute_warps)
        self.reduce_sync = pipeline.NamedBarrier(barrier_id=5, num_threads=32 * self.num_reduce_warps)

        # preprocess / convert geometry
        self.pre_threads = 128
        self.pre_rows_per_block = 64
        self.pre_elems = 8          # bf16 elements per 16-byte load
        self.pre_threads_per_row = D // self.pre_elems   # 16

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, Q4: cute.Tensor, K4: cute.Tensor, V4: cute.Tensor, O4: cute.Tensor,
                 dO4: cute.Tensor, LSE3: cute.Tensor, dQ4: cute.Tensor, dK4: cute.Tensor,
                 dV4: cute.Tensor, dQacc4: cute.Tensor, LSE2w: cute.Tensor, NDELTAw: cute.Tensor,
                 sm_scale: Float32, stream: cuda.CUstream):
        B, S, H, HKV, D = self.B, self.S, self.H, self.HKV, self.D
        h_r, h_k = self.h_r, self.h_k
        hb = ((h_r, h_k), B)

        # (s, d, ((h_r, h_k), b)) views of the bshd tensors
        def qview(t):
            return cute.make_tensor(t.iterator, cute.make_layout(
                (S, D, hb), stride=(H * D, 1, ((D, D * h_r), S * H * D))))

        def kview(t):
            return cute.make_tensor(t.iterator, cute.make_layout(
                (S, D, ((1, h_k), B)), stride=(HKV * D, 1, ((0, D), S * HKV * D))))

        def lview(t):  # (b, h, s) fp32 -> (s, ((h_r, h_k), b))
            return cute.make_tensor(t.iterator, cute.make_layout(
                (S, hb), stride=(1, ((S, S * h_r), S * H))))

        Q, O, dO, dQ, dQacc = qview(Q4), qview(O4), qview(dO4), qview(dQ4), qview(dQacc4)
        K, V, dK, dV = kview(K4), kview(V4), kview(dK4), kview(dV4)
        LSE, LSE2, NDELTA = lview(LSE3), lview(LSE2w), lview(NDELTAw)

        io = self.io_dtype
        acc = self.acc_dtype
        cta_group = tcgen05.CtaGroup.ONE
        KMaj, MNMaj = tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN

        KQ_mma = sm100_utils.make_trivial_tiled_mma(io, KMaj, KMaj, acc, cta_group, self.KQ_tiler[:2])
        VdO_mma = sm100_utils.make_trivial_tiled_mma(io, KMaj, KMaj, acc, cta_group, self.VdO_tiler[:2])
        PdO_mma = sm100_utils.make_trivial_tiled_mma(io, KMaj, MNMaj, acc, cta_group, self.PdO_tiler[:2],
                                                     tcgen05.OperandSource.TMEM)
        dS_src = tcgen05.OperandSource.TMEM if self.ds_tmem else tcgen05.OperandSource.SMEM
        dSQ_mma = sm100_utils.make_trivial_tiled_mma(io, KMaj, MNMaj, acc, cta_group, self.dSQ_tiler[:2], dS_src)
        dSK_mma = sm100_utils.make_trivial_tiled_mma(io, MNMaj, MNMaj, acc, cta_group, self.dSK_tiler[:2])

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (KQ_mma.thr_id.shape,))

        K_sl = sm100_utils.make_smem_layout_a(KQ_mma, self.KQ_tiler, io, 1)
        Q_sl = sm100_utils.make_smem_layout_b(KQ_mma, self.KQ_tiler, io, self.q_stages)
        V_sl = sm100_utils.make_smem_layout_a(VdO_mma, self.VdO_tiler, io, 1)
        dO_sl = sm100_utils.make_smem_layout_b(VdO_mma, self.VdO_tiler, io, self.do_stages)
        dS_sl = sm100_utils.make_smem_layout_a(dSK_mma, self.dSK_tiler, io, self.dS_stages)
        KT_sl = sm100_utils.make_smem_layout_b(dSK_mma, self.dSK_tiler, io, 1)
        dST_sl = sm100_utils.make_smem_layout_a(dSQ_mma, self.dSQ_tiler, io, self.dS_stages)
        QT_sl = sm100_utils.make_smem_layout_b(dSQ_mma, self.dSQ_tiler, io, self.q_stages)
        # P^T lives in TMEM, one half (64 q columns) per compute warpgroup inside that warpgroup's own S^T columns
        self.PdO_tiler_half = (self.tile_k, D, self.tile_q // 2)
        P_tiler = self.PdO_tiler_half if self.p_var == 1 else self.PdO_tiler
        P_tl = sm100_utils.make_smem_layout_a(PdO_mma, P_tiler, io, self.P_stages)
        dOT_sl = sm100_utils.make_smem_layout_b(PdO_mma, self.PdO_tiler, io, self.do_stages)
        LSE_sl = cute.make_layout((self.tile_q, self.lse_stages))
        DELTA_sl = cute.make_layout((self.tile_q, self.delta_stages))

        dQ_s_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(KMaj, self.dq_dtype, (self.tile_q, self.dq_sub_cols)), self.dq_dtype)
        dQ_sl = cute.tile_to_shape(dQ_s_atom, (self.tile_q, self.dq_sub_cols, self.dq_store_stages), order=(1, 0, 2))

        tma_load = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce = cpasync.CopyReduceBulkTensorTileS2GOp()

        tma_K, tK = cute.nvgpu.make_tiled_tma_atom_A(tma_load, K, cute.select(K_sl, mode=[0, 1, 2]),
                                                     self.KQ_tiler, KQ_mma, cluster_layout_vmnk.shape)
        tma_V, tV = cute.nvgpu.make_tiled_tma_atom_A(tma_load, V, cute.select(V_sl, mode=[0, 1, 2]),
                                                     self.VdO_tiler, VdO_mma, cluster_layout_vmnk.shape)
        tma_Q, tQ = cute.nvgpu.make_tiled_tma_atom_B(tma_load, Q, cute.select(Q_sl, mode=[0, 1, 2]),
                                                     self.KQ_tiler, KQ_mma, cluster_layout_vmnk.shape)
        tma_dO, tdO = cute.nvgpu.make_tiled_tma_atom_B(tma_load, dO, cute.select(dO_sl, mode=[0, 1, 2]),
                                                       self.VdO_tiler, VdO_mma, cluster_layout_vmnk.shape)
        tma_dQ, tdQacc = cpasync.make_tiled_tma_atom(tma_reduce, dQacc, cute.select(dQ_sl, mode=[0, 1]),
                                                     (self.tile_q, self.dq_sub_cols))

        self.bytes_Q = cute.size_in_bytes(io, cute.select(Q_sl, mode=[0, 1, 2]))
        self.bytes_K = cute.size_in_bytes(io, cute.select(K_sl, mode=[0, 1, 2]))
        self.bytes_V = cute.size_in_bytes(io, cute.select(V_sl, mode=[0, 1, 2]))
        self.bytes_dO = cute.size_in_bytes(io, cute.select(dO_sl, mode=[0, 1, 2]))

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.do_stages * 2]
            lse_mbar: cute.struct.MemRange[cutlass.Int64, self.lse_stages * 2]
            delta_mbar: cute.struct.MemRange[cutlass.Int64, self.delta_stages * 2]
            S_mbar: cute.struct.MemRange[cutlass.Int64, self.S_stages * 2]
            dP_mbar: cute.struct.MemRange[cutlass.Int64, self.dP_stages * 2]
            dQ_mbar: cute.struct.MemRange[cutlass.Int64, self.dQ_stages * 2]
            P_mbar: cute.struct.MemRange[cutlass.Int64, self.P_stages * 2]
            dS_mbar: cute.struct.MemRange[cutlass.Int64, self.dS_stages * 2]
            dST_mbar: cute.struct.MemRange[cutlass.Int64, self.dS_stages * 2]
            dKdV_mbar: cute.struct.MemRange[cutlass.Int64, self.dKdV_stages * 2]
            tmem_holding_buf: cutlass.Int32
            sK: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(K_sl)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(V_sl)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(Q_sl)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(dO_sl)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(dS_sl)], 1024]
            sdQ: cute.struct.Align[cute.struct.MemRange[self.dq_dtype, cute.cosize(dQ_sl)], 1024]
            sLSE: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(LSE_sl)], 1024]
            sDELTA: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(DELTA_sl)], 1024]

        self.shared_storage = SharedStorage

        # ---- 1. preprocess
        pre_grid = (S // self.pre_rows_per_block, H, B)
        self.preprocess(O, dO, LSE, LSE2, NDELTA, dQacc, Float32(-LOG2E)).launch(
            grid=pre_grid, block=[self.pre_threads, 1, 1], cluster=[1, 1, 1], stream=stream)

        # ---- 2. main backward
        self.bwd(KQ_mma, VdO_mma, PdO_mma, dSQ_mma, dSK_mma,
                 tma_K, tK, tma_V, tV, tma_Q, tQ, tma_dO, tdO, tma_dQ, tdQacc,
                 dK, dV, LSE2, NDELTA, sm_scale,
                 K_sl, Q_sl, V_sl, dO_sl, dS_sl, KT_sl, dST_sl, QT_sl, dOT_sl, dQ_sl, P_tl, LSE_sl, DELTA_sl,
                 ).launch(grid=(HKV * B, self.num_k_tiles, 1), block=[self.threads_per_cta, 1, 1],
                          cluster=[1, 1, 1], stream=stream, min_blocks_per_mp=1)

        # ---- 3. convert
        self.convert(dQacc, dQ, sm_scale).launch(
            grid=pre_grid, block=[self.pre_threads, 1, 1], cluster=[1, 1, 1], stream=stream)

    # ------------------------------------------------------------ preprocess
    @cute.kernel
    def preprocess(self, O: cute.Tensor, dO: cute.Tensor, LSE: cute.Tensor, LSE2: cute.Tensor,
                   NDELTA: cute.Tensor, dQacc: cute.Tensor, lse_scale: Float32):
        # 128 threads: 16 threads per row (16 B of O / dO each), 8 rows per step, 64 rows per block.
        # Loads of all 8 steps are issued before the reductions so the memory system sees 8 rows in flight.
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        h = bidy
        hb = ((h % self.h_r, h // self.h_r), bidz)
        seg = tidx % self.pre_threads_per_row
        row_in_blk = tidx // self.pre_threads_per_row
        rows_per_iter = self.pre_threads // self.pre_threads_per_row
        n_it = self.pre_rows_per_block // rows_per_iter
        row0 = bidx * self.pre_rows_per_block + row_in_blk
        o_frg = cute.make_rmem_tensor((self.pre_elems, n_it), self.io_dtype)
        do_frg = cute.make_rmem_tensor((self.pre_elems, n_it), self.io_dtype)
        for it in cutlass.range_constexpr(n_it):
            row = row0 + it * rows_per_iter
            O_row = cute.logical_divide(O[row, None, hb], cute.make_layout(self.pre_elems))
            dO_row = cute.logical_divide(dO[row, None, hb], cute.make_layout(self.pre_elems))
            o_frg[None, it].store(O_row[None, seg].load())
            do_frg[None, it].store(dO_row[None, seg].load())
        z = cute.make_rmem_tensor((self.pre_elems,), self.dq_dtype)
        z.fill(0.0)
        for it in cutlass.range_constexpr(n_it):
            row = row0 + it * rows_per_iter
            prod = o_frg[None, it].load().to(self.acc_dtype) * do_frg[None, it].load().to(self.acc_dtype)
            partial = prod.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
            total = cute.arch.warp_reduction_sum(partial, threads_in_group=self.pre_threads_per_row)
            if seg == 0:
                NDELTA[row, hb] = -total
                LSE2[row, hb] = lse_scale * LSE[row, hb]
            dq_row = cute.logical_divide(dQacc[row, None, hb], cute.make_layout(self.pre_elems))
            dq_row[None, seg].store(z.load())

    # --------------------------------------------------------------- convert
    @cute.kernel
    def convert(self, dQacc: cute.Tensor, dQ: cute.Tensor, sm_scale: Float32):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        h = bidy
        hb = ((h % self.h_r, h // self.h_r), bidz)
        seg = tidx % self.pre_threads_per_row
        row_in_blk = tidx // self.pre_threads_per_row
        rows_per_iter = self.pre_threads // self.pre_threads_per_row
        n_it = self.pre_rows_per_block // rows_per_iter
        row0 = bidx * self.pre_rows_per_block + row_in_blk
        acc_frg = cute.make_rmem_tensor((self.pre_elems, n_it), self.dq_dtype)
        for it in cutlass.range_constexpr(n_it):
            row = row0 + it * rows_per_iter
            src = cute.logical_divide(dQacc[row, None, hb], cute.make_layout(self.pre_elems))
            acc_frg[None, it].store(src[None, seg].load())
        for it in cutlass.range_constexpr(n_it):
            row = row0 + it * rows_per_iter
            dst = cute.logical_divide(dQ[row, None, hb], cute.make_layout(self.pre_elems))
            if cutlass.const_expr(self.dq_f16):
                dst[None, seg].store(acc_frg[None, it].load().to(self.acc_dtype).to(self.io_dtype))   # scale already applied
            else:
                dst[None, seg].store((acc_frg[None, it].load() * sm_scale).to(self.io_dtype))

    # ------------------------------------------------------------- main kernel
    @cute.kernel
    def bwd(self, KQ_mma: cute.TiledMma, VdO_mma: cute.TiledMma, PdO_mma: cute.TiledMma,
            dSQ_mma: cute.TiledMma, dSK_mma: cute.TiledMma,
            tma_K: cute.CopyAtom, mK: cute.Tensor, tma_V: cute.CopyAtom, mV: cute.Tensor,
            tma_Q: cute.CopyAtom, mQ: cute.Tensor, tma_dO: cute.CopyAtom, mdO: cute.Tensor,
            tma_dQ: cute.CopyAtom, mdQacc: cute.Tensor,
            dK: cute.Tensor, dV: cute.Tensor, LSE2: cute.Tensor, NDELTA: cute.Tensor, sm_scale: Float32,
            K_sl: cute.ComposedLayout, Q_sl: cute.ComposedLayout, V_sl: cute.ComposedLayout,
            dO_sl: cute.ComposedLayout, dS_sl: cute.ComposedLayout, KT_sl: cute.ComposedLayout,
            dST_sl: cute.ComposedLayout, QT_sl: cute.ComposedLayout, dOT_sl: cute.ComposedLayout,
            dQ_sl: cute.ComposedLayout, P_tl: cute.ComposedLayout, LSE_sl: cute.Layout, DELTA_sl: cute.Layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # grid = (HKV*B, num_k_tiles): kv tile 0 (heaviest, causal) is launched first for every (b, hk)
        kv_tile = bidy
        blk_hk = bidx % self.HKV
        blk_b = bidx // self.HKV

        if warp_idx == self.load_warp:
            cpasync.prefetch_descriptor(tma_K)
            cpasync.prefetch_descriptor(tma_Q)
            cpasync.prefetch_descriptor(tma_V)
            cpasync.prefetch_descriptor(tma_dO)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # pipelines
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        one_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        compute_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_compute_warps)
        reduce_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_reduce_warps)

        q_pipe = pipeline.PipelineTmaUmma.create(barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.q_stages,
                                                 producer_group=one_thread, consumer_group=one_thread, tx_count=self.bytes_Q)
        do_pipe = pipeline.PipelineTmaUmma.create(barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.do_stages,
                                                  producer_group=one_thread, consumer_group=one_thread, tx_count=self.bytes_dO)
        lse_pipe = pipeline.PipelineCpAsync.create(barrier_storage=storage.lse_mbar.data_ptr(), num_stages=self.lse_stages,
                                                   producer_group=one_warp, consumer_group=compute_grp)
        delta_pipe = pipeline.PipelineCpAsync.create(barrier_storage=storage.delta_mbar.data_ptr(), num_stages=self.delta_stages,
                                                     producer_group=one_warp, consumer_group=compute_grp)
        S_pipe = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.S_mbar.data_ptr(), num_stages=self.S_stages,
                                                   producer_group=one_thread, consumer_group=compute_grp)
        dP_pipe = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.dP_mbar.data_ptr(), num_stages=self.dP_stages,
                                                    producer_group=one_thread, consumer_group=compute_grp)
        dQ_pipe = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.dQ_mbar.data_ptr(), num_stages=self.dQ_stages,
                                                    producer_group=one_thread, consumer_group=reduce_grp)
        P_pipe = pipeline.PipelineAsyncUmma.create(barrier_storage=storage.P_mbar.data_ptr(), num_stages=self.P_stages,
                                                   producer_group=compute_grp, consumer_group=one_thread)
        dS_pipe = pipeline.PipelineAsyncUmma.create(barrier_storage=storage.dS_mbar.data_ptr(), num_stages=self.dS_stages,
                                                    producer_group=compute_grp, consumer_group=one_thread)
        dST_pipe = pipeline.PipelineAsyncUmma.create(barrier_storage=storage.dST_mbar.data_ptr(), num_stages=self.dS_stages,
                                                     producer_group=compute_grp, consumer_group=one_thread)
        dKdV_pipe = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.dKdV_mbar.data_ptr(), num_stages=self.dKdV_stages,
                                                      producer_group=one_thread, consumer_group=compute_grp)
        dq_store_pipe = pipeline.PipelineTmaStore.create(num_stages=self.dq_store_stages, producer_group=reduce_grp)

        self.cta_sync.arrive_and_wait()

        # smem tensors
        sQ = storage.sQ.get_tensor(Q_sl.outer, swizzle=Q_sl.inner)
        sK = storage.sK.get_tensor(K_sl.outer, swizzle=K_sl.inner)
        sV = storage.sV.get_tensor(V_sl.outer, swizzle=V_sl.inner)
        sdO = storage.sdO.get_tensor(dO_sl.outer, swizzle=dO_sl.inner)
        sdQ = storage.sdQ.get_tensor(dQ_sl.outer, swizzle=dQ_sl.inner)
        sLSE = storage.sLSE.get_tensor(LSE_sl)
        sDELTA = storage.sDELTA.get_tensor(DELTA_sl)
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        sQT = cute.make_tensor(cute.recast_ptr(sQ.iterator, QT_sl.inner), QT_sl.outer)
        sKT = cute.make_tensor(cute.recast_ptr(sK.iterator, KT_sl.inner), KT_sl.outer)
        sdS = storage.sdS.get_tensor(dS_sl.outer, swizzle=dS_sl.inner)
        sdST = cute.make_tensor(cute.recast_ptr(sdS.iterator, dST_sl.inner), dST_sl.outer)
        sdOT = cute.make_tensor(cute.recast_ptr(sdO.iterator, dOT_sl.inner), dOT_sl.outer)
        tP = cute.make_tensor(cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.tmem), P_tl.outer)

        # MMA fragments
        tSTrK = KQ_mma.make_fragment_A(sK)
        tSTrQ = KQ_mma.make_fragment_B(sQ)
        tdPTrV = VdO_mma.make_fragment_A(sV)
        tdPTrdO = VdO_mma.make_fragment_B(sdO)
        tdQrdS = dSK_mma.make_fragment_A(sdS)
        tdQrKT = dSK_mma.make_fragment_B(sKT)
        tdKrQT = dSQ_mma.make_fragment_B(sQT)

        tSTtST = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C(cute.select(self.KQ_tiler, mode=[0, 1])))
        tSTtST = cute.make_tensor(tSTtST.iterator + self.tmem_S, tSTtST.layout)

        tdVrP_layout = PdO_mma.make_fragment_A(tP)[None, None, None, 0].layout
        # WG0: q columns 0..63 at S^T columns [0,32) of the S region; WG1: q 64..127 at S^T columns [64,96)
        tdVrP0 = cute.make_tensor(cute.recast_ptr(tSTtST.iterator, dtype=self.io_dtype), tdVrP_layout)
        if cutlass.const_expr(self.p_var == 1):
            tdVrP1 = cute.make_tensor(cute.recast_ptr(tSTtST.iterator + self.p1_off, dtype=self.io_dtype), tdVrP_layout)
        elif cutlass.const_expr(self.wg_place):
            # k-blocks 0..3 at columns [0,32) and 4..7 at [64,96) of the S region (each warpgroup's own S^T half)
            tdVrP0 = cute.make_tensor(tdVrP0.iterator, self.gapped_a_layout(tdVrP_layout))
            tdVrP1 = tdVrP0
        else:
            tdVrP1 = tdVrP0
        tdVrdOT = PdO_mma.make_fragment_B(sdOT)

        tdPTtdPT = VdO_mma.make_fragment_C(VdO_mma.partition_shape_C(cute.select(self.VdO_tiler, mode=[0, 1])))
        tdPTtdPT = cute.make_tensor(tdPTtdPT.iterator + self.tmem_dP, tdPTtdPT.layout)

        if cutlass.const_expr(self.ds_tmem):
            # dS^T (bf16) overlays the dP^T columns; A operand of the dK MMA straight from TMEM
            tdST_fake = cute.make_tensor(cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.tmem), dST_sl.outer)
            tdKrdST_layout = dSQ_mma.make_fragment_A(tdST_fake)[None, None, None, 0].layout
            if cutlass.const_expr(self.wg_place):
                tdKrdST_layout = self.gapped_a_layout(tdKrdST_layout)
            tdKrdST = cute.make_tensor(cute.recast_ptr(tdPTtdPT.iterator, dtype=self.io_dtype), tdKrdST_layout)
            tdKrdST1 = tdKrdST
        else:
            tdKrdST = dSQ_mma.make_fragment_A(sdST)
            tdKrdST1 = tdKrdST

        tdQtdQ = dSK_mma.make_fragment_C(dSK_mma.partition_shape_C(cute.select(self.dSK_tiler, mode=[0, 1])))
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator + self.tmem_dQ, tdQtdQ.layout)

        tdKtdK = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C(cute.select(self.dSQ_tiler, mode=[0, 1])))
        tdKtdK = cute.make_tensor(tdKtdK.iterator + self.tmem_dK, tdKtdK.layout)

        tdVtdV = PdO_mma.make_fragment_C(PdO_mma.partition_shape_C(cute.select(self.PdO_tiler, mode=[0, 1])))
        tdVtdV = cute.make_tensor(tdVtdV.iterator + self.tmem_dV, tdVtdV.layout)

        # causal trip bookkeeping (tile_q == tile_k): q tiles [kv_tile, num_q_tiles)
        trip_start = kv_tile
        trip_end = Int32(self.num_q_tiles)
        trips_per_head = trip_end - trip_start
        iter_count = trips_per_head * self.h_r

        if warp_idx == self.load_warp:
            cute.arch.setmaxregister_decrease(self.regs_load)
            self.load(mK, mV, mQ, mdO, LSE2, NDELTA, sK, sQ, sV, sdO, sLSE, sDELTA,
                      KQ_mma, VdO_mma, tma_K, tma_Q, tma_V, tma_dO,
                      kv_tile, blk_hk, blk_b, iter_count, trip_start, trip_end,
                      (q_pipe, lse_pipe, do_pipe, delta_pipe))
        elif warp_idx == self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_mma)
            self.mma(KQ_mma, VdO_mma, PdO_mma, dSK_mma, dSQ_mma,
                     tSTtST, tSTrQ, tSTrK, tdPTtdPT, tdPTrV, tdPTrdO, tdVtdV, (tdVrP0, tdVrP1), tdVrdOT,
                     tdQtdQ, tdQrdS, tdQrKT, (tdKrdST, tdKrdST1), tdKtdK, tdKrQT, tmem_holding_buf, iter_count,
                     (q_pipe, S_pipe, do_pipe, dP_pipe, dQ_pipe, P_pipe, dS_pipe, dKdV_pipe, dST_pipe))
        elif warp_idx >= self.compute_warps[0] and warp_idx <= self.compute_warps[-1]:
            cute.arch.setmaxregister_increase(self.regs_compute)
            self.compute(tSTtST, tdPTtdPT, (tdVrP0, tdVrP1), (tdKrdST, tdKrdST1), sLSE, sdS, sDELTA, dK, dV, tdKtdK, tdVtdV,
                         PdO_mma, dSQ_mma, kv_tile, blk_hk, blk_b, iter_count, trip_start, trip_end, sm_scale,
                         (S_pipe, P_pipe, lse_pipe, delta_pipe, dP_pipe, dS_pipe, dKdV_pipe, dST_pipe))
            self.epilogue_sync.arrive_and_wait()
            if warp_idx == self.compute_warps[0]:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
        elif warp_idx <= self.reduce_warps[-1]:
            cute.arch.setmaxregister_increase(self.regs_reduce)
            self.reduce(dSK_mma, tdQtdQ, tma_dQ, mdQacc, sdQ, blk_hk, blk_b, iter_count, trip_start, trip_end, sm_scale,
                        (dQ_pipe, dq_store_pipe))
        else:
            cute.arch.setmaxregister_decrease(self.regs_empty)

    # ------------------------------------------------------------------ load
    @cute.jit
    def load(self, mK, mV, mQ, mdO, LSE2, NDELTA, sK, sQ, sV, sdO, sLSE, sDELTA,
             KQ_mma, VdO_mma, tma_K, tma_Q, tma_V, tma_dO,
             kv_tile, blk_hk, blk_b, iter_count, trip_start, trip_end, pipes):
        tidx, _, _ = cute.arch.thread_idx()
        q_pipe, lse_pipe, do_pipe, delta_pipe = pipes
        hr = Int32(0)
        hb = ((hr, blk_hk), blk_b)

        gK = cute.local_tile(mK, cute.select(self.KQ_tiler, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(mQ, cute.select(self.KQ_tiler, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(mV, cute.select(self.VdO_tiler, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(mdO, cute.select(self.VdO_tiler, mode=[1, 2]), (None, None, None))

        KQ_thr = KQ_mma.get_slice(0)
        VdO_thr = VdO_mma.get_slice(0)
        tSTgK = KQ_thr.partition_A(gK)
        tSTgQ = KQ_thr.partition_B(gQ)
        tdPTgV = VdO_thr.partition_A(gV)
        tdPTgdO = VdO_thr.partition_B(gdO)

        tKsK, tKgK = cpasync.tma_partition(tma_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSTgK, 0, 3))
        tQsQ, tQgQ = cpasync.tma_partition(tma_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSTgQ, 0, 3))
        tVsV, tVgV = cpasync.tma_partition(tma_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tdPTgV, 0, 3))
        tdOsdO, tdOgdO = cpasync.tma_partition(tma_dO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tdPTgdO, 0, 3))

        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stages)
        lse_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.lse_stages)
        do_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.do_stages)
        delta_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.delta_stages)

        lane = tidx % 32
        n_per_thread = self.tile_q // 32   # 4 fp32 per lane
        cp_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                                      self.acc_dtype, num_bits_per_copy=self.acc_dtype.width)
        sLSE_c = cute.flat_divide(sLSE, (1,))
        LSE2_c = cute.flat_divide(LSE2, (1,))
        sDELTA_c = cute.flat_divide(sDELTA, (1,))
        NDELTA_c = cute.flat_divide(NDELTA, (1,))

        q_tile = trip_start
        first = True
        while iter_count > 0:
            if q_tile == trip_end:
                q_tile = trip_start
                hr += 1
                hb = ((hr, blk_hk), blk_b)

            # Q (+K on the first trip)
            _ik_push(self.iket, "ld_a_q")
            q_pipe.producer_acquire(q_state)
            _ik_pop(self.iket)
            bar = q_pipe.producer_get_barrier(q_state)
            if first:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(bar, self.bytes_K)
                cute.copy(tma_K, tKgK[(None, kv_tile, 0, hb)], tKsK[None, 0], tma_bar_ptr=bar)
            cute.copy(tma_Q, tQgQ[(None, q_tile, 0, hb)], tQsQ[None, q_state.index], tma_bar_ptr=bar)
            q_state.advance()

            # LSE row block
            _ik_push(self.iket, "ld_a_lse")
            lse_pipe.producer_acquire(lse_state)
            _ik_pop(self.iket)
            for i in cutlass.range_constexpr(n_per_thread):
                idx = self.tile_q * q_tile + lane * n_per_thread + i
                cute.copy(cp_atom, LSE2_c[None, idx, hb], sLSE_c[None, lane * n_per_thread + i, lse_state.index])
            lse_pipe.producer_commit(lse_state)
            lse_state.advance()

            # dO (+V on the first trip)
            _ik_push(self.iket, "ld_a_dO")
            do_pipe.producer_acquire(do_state)
            _ik_pop(self.iket)
            bar = do_pipe.producer_get_barrier(do_state)
            if first:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(bar, self.bytes_V)
                cute.copy(tma_V, tVgV[(None, kv_tile, 0, hb)], tVsV[None, 0], tma_bar_ptr=bar)
            cute.copy(tma_dO, tdOgdO[(None, q_tile, 0, hb)], tdOsdO[None, do_state.index], tma_bar_ptr=bar)
            do_state.advance()

            # delta row block
            _ik_push(self.iket, "ld_a_delta")
            delta_pipe.producer_acquire(delta_state)
            _ik_pop(self.iket)
            for i in cutlass.range_constexpr(n_per_thread):
                idx = self.tile_q * q_tile + lane * n_per_thread + i
                cute.copy(cp_atom, NDELTA_c[None, idx, hb], sDELTA_c[None, lane * n_per_thread + i, delta_state.index])
            delta_pipe.producer_commit(delta_state)
            delta_state.advance()

            first = False
            iter_count -= 1
            q_tile += 1

    # ------------------------------------------------------------------- mma
    @cute.jit
    def mma(self, KQ_mma, VdO_mma, PdO_mma, dSK_mma, dSQ_mma,
            tSTtST, tSTrQ, tSTrK, tdPTtdPT, tdPTrV, tdPTrdO, tdVtdV, tdVrP, tdVrdOT,
            tdQtdQ, tdQrdS, tdQrKT, tdKrdST, tdKtdK, tdKrQT, tmem_holding_buf, iter_count, pipes):
        q_pipe, S_pipe, do_pipe, dP_pipe, dQ_pipe, P_pipe, dS_pipe, dKdV_pipe, dST_pipe = pipes
        tdVrP0, tdVrP1 = tdVrP
        tdKrdST, tdKrdST1 = tdKrdST
        nk_half = cute.size(tdVrP0, mode=[2])
        nk_dk = cute.size(tdKrdST, mode=[2])

        cute.arch.alloc_tmem(cutlass.Int32(self.tmem_cols), tmem_holding_buf)
        self.tmem_alloc_sync.arrive_and_wait()

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stages)
        q_rel = q_cons.clone()
        S_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.S_stages)
        dS_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dS_stages)
        dST_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dS_stages)
        dP_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dP_stages)
        dQ_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dQ_stages)
        do_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.do_stages)
        P_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.P_stages)
        dKdV_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dKdV_stages)

        nk = cute.size(tSTrQ, mode=[2])

        # ---- prologue: S_0, dP_0, dV_0
        q_pipe.consumer_wait(q_cons)
        S_pipe.producer_acquire(S_prod)
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range(nk, unroll_full=True):
            cute.gemm(KQ_mma, tSTtST, tSTrK[None, None, kb, 0], tSTrQ[None, None, kb, q_cons.index], tSTtST)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        q_cons.advance()
        S_pipe.producer_commit(S_prod)
        S_prod.advance()

        do_pipe.consumer_wait(do_cons)
        dP_pipe.producer_acquire(dP_prod)
        dQ_pipe.producer_acquire(dQ_prod)
        VdO_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range(nk, unroll_full=True):
            cute.gemm(VdO_mma, tdPTtdPT, tdPTrV[None, None, kb, 0], tdPTrdO[None, None, kb, do_cons.index], tdPTtdPT)
            VdO_mma.set(tcgen05.Field.ACCUMULATE, True)
        dP_pipe.producer_commit(dP_prod)
        dP_prod.advance()

        P_pipe.consumer_wait(P_cons)
        if cutlass.const_expr(self.p_var == 1):
            for kb in cutlass.range(nk_half, unroll_full=True):
                cute.gemm(PdO_mma, tdVtdV, tdVrP0[None, None, kb], tdVrdOT[None, None, kb, do_cons.index], tdVtdV)
                PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
            for kb in cutlass.range(nk_half, unroll_full=True):
                cute.gemm(PdO_mma, tdVtdV, tdVrP1[None, None, kb], tdVrdOT[None, None, kb + nk_half, do_cons.index], tdVtdV)
                PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
        else:
            for kb in cutlass.range(cute.size(tdVrP0, mode=[2]), unroll_full=True):
                cute.gemm(PdO_mma, tdVtdV, tdVrP0[None, None, kb], tdVrdOT[None, None, kb, do_cons.index], tdVtdV)
                PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
        P_pipe.consumer_release(P_cons)
        P_cons.advance()
        do_pipe.consumer_release(do_cons)
        do_cons.advance()
        iter_count -= 1

        # ---- steady state (PdO/dSQ accumulators start from ACCUMULATE=False default)
        while iter_count > 0:
            _ik_push(self.iket, "mma_w_q")
            q_pipe.consumer_wait(q_cons)
            _ik_pop(self.iket)
            _ik_push(self.iket, "mma_a_S")
            S_pipe.producer_acquire(S_prod)
            _ik_pop(self.iket)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range(nk, unroll_full=True):
                cute.gemm(KQ_mma, tSTtST, tSTrK[None, None, kb, 0], tSTrQ[None, None, kb, q_cons.index], tSTtST)
                KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
            q_cons.advance()
            S_pipe.producer_commit(S_prod)
            S_prod.advance()

            _ik_push(self.iket, "mma_w_dS")
            if cutlass.const_expr(self.split_ds):
                dST_pipe.consumer_wait(dST_cons)   # dS^T in TMEM
            else:
                dS_pipe.consumer_wait(dS_cons)
            _ik_pop(self.iket)
            # dK first: releases the previous Q stage as early as possible (TMA lead time for Q_{i+1})
            for kb in cutlass.range(nk_dk, unroll_full=True):
                cute.gemm(dSQ_mma, tdKtdK, self.dst_frag(tdKrdST, kb, dS_cons.index), tdKrQT[None, None, kb, q_rel.index], tdKtdK)
                dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(self.split_ds):
                dST_pipe.consumer_release(dST_cons)
                dST_cons.advance()
            q_pipe.consumer_release(q_rel)
            q_rel.advance()

            _ik_push(self.iket, "mma_a_dP")
            dP_pipe.producer_acquire(dP_prod)      # dQ aliases dP in TMEM
            _ik_pop(self.iket)
            if cutlass.const_expr(self.split_ds):
                _ik_push(self.iket, "mma_w_dSs")
                dS_pipe.consumer_wait(dS_cons)     # dS in smem
                _ik_pop(self.iket)
            dSK_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range(cute.size(tdQrdS, mode=[2]), unroll_full=True):
                cute.gemm(dSK_mma, tdQtdQ, tdQrdS[None, None, kb, dS_cons.index], tdQrKT[None, None, kb, 0], tdQtdQ)
                dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
            dQ_pipe.producer_commit(dQ_prod)
            dQ_prod.advance()
            dS_pipe.consumer_release(dS_cons)
            dS_cons.advance()

            _ik_push(self.iket, "mma_a_dQ")
            dQ_pipe.producer_acquire(dQ_prod)
            _ik_pop(self.iket)
            _ik_push(self.iket, "mma_w_dO")
            do_pipe.consumer_wait(do_cons)
            _ik_pop(self.iket)
            VdO_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range(nk, unroll_full=True):
                cute.gemm(VdO_mma, tdPTtdPT, tdPTrV[None, None, kb, 0], tdPTrdO[None, None, kb, do_cons.index], tdPTtdPT)
                VdO_mma.set(tcgen05.Field.ACCUMULATE, True)
            dP_pipe.producer_commit(dP_prod)
            dP_prod.advance()

            _ik_push(self.iket, "mma_w_P")
            P_pipe.consumer_wait(P_cons)
            _ik_pop(self.iket)
            if cutlass.const_expr(self.p_var == 1):
                for kb in cutlass.range(nk_half, unroll_full=True):
                    cute.gemm(PdO_mma, tdVtdV, tdVrP0[None, None, kb], tdVrdOT[None, None, kb, do_cons.index], tdVtdV)
                    PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range(nk_half, unroll_full=True):
                    cute.gemm(PdO_mma, tdVtdV, tdVrP1[None, None, kb], tdVrdOT[None, None, kb + nk_half, do_cons.index], tdVtdV)
                    PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for kb in cutlass.range(cute.size(tdVrP0, mode=[2]), unroll_full=True):
                    cute.gemm(PdO_mma, tdVtdV, tdVrP0[None, None, kb], tdVrdOT[None, None, kb, do_cons.index], tdVtdV)
                    PdO_mma.set(tcgen05.Field.ACCUMULATE, True)
            P_pipe.consumer_release(P_cons)
            P_cons.advance()
            do_pipe.consumer_release(do_cons)
            do_cons.advance()
            iter_count -= 1

        # ---- tail: dV done; last dK and dQ
        dKdV_pipe.producer_acquire(dKdV_prod)
        dKdV_pipe.producer_commit(dKdV_prod)
        dKdV_prod.advance()

        dKdV_pipe.producer_acquire(dKdV_prod)
        if cutlass.const_expr(self.split_ds):
            dST_pipe.consumer_wait(dST_cons)
        else:
            dS_pipe.consumer_wait(dS_cons)
        for kb in cutlass.range(nk_dk, unroll_full=True):
            cute.gemm(dSQ_mma, tdKtdK, self.dst_frag(tdKrdST, kb, dS_cons.index), tdKrQT[None, None, kb, q_rel.index], tdKtdK)
            dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        dKdV_pipe.producer_commit(dKdV_prod)
        dKdV_prod.advance()
        if cutlass.const_expr(self.split_ds):
            dST_pipe.consumer_release(dST_cons)
            dST_cons.advance()
            dS_pipe.consumer_wait(dS_cons)

        dSK_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range(cute.size(tdQrdS, mode=[2]), unroll_full=True):
            cute.gemm(dSK_mma, tdQtdQ, tdQrdS[None, None, kb, dS_cons.index], tdQrKT[None, None, kb, 0], tdQtdQ)
            dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
        dQ_pipe.producer_commit(dQ_prod)
        dQ_prod.advance()
        q_pipe.consumer_release(q_rel)
        q_rel.advance()
        dS_pipe.consumer_release(dS_cons)
        dS_cons.advance()

    # --------------------------------------------------------------- compute
    @cute.jit
    def compute(self, tSTtST, tdPTtdPT, tdVrP, tdKrdST, sLSE, sdS, sDELTA, dK, dV, tdKtdK, tdVtdV,
                PdO_mma, dSQ_mma, kv_tile, blk_hk, blk_b, iter_count, trip_start, trip_end, sm_scale, pipes):
        tidx, _, _ = cute.arch.thread_idx()
        S_pipe, P_pipe, lse_pipe, delta_pipe, dP_pipe, dS_pipe, dKdV_pipe, dST_pipe = pipes
        tdKrdST, tdKrdST1 = tdKrdST

        S_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.S_stages)
        P_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.P_stages)
        lse_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.lse_stages)
        delta_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.delta_stages)
        dP_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dP_stages)
        dS_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dS_stages)
        dST_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dS_stages)
        dKdV_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dKdV_stages)

        t2r_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        r2t_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), self.io_dtype)

        tSTtST_full = tSTtST
        tdPTtdPT_full = tdPTtdPT
        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]
        cST = cute.make_identity_tensor(cute.select(self.KQ_tiler, mode=[0, 1]))
        cdPT = cute.make_identity_tensor(cute.select(self.VdO_tiler, mode=[0, 1]))

        nwg = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * 32)) // 128

        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)
        tTR_cST = self.split_wg(thr_t2r.partition_D(cST), nwg, wg_idx)
        tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)
        tTR_tST = self.split_wg(thr_t2r.partition_S(tSTtST), nwg, wg_idx)
        tTR_cdPT_full = thr_t2r.partition_D(cdPT)
        tTR_cdPT = self.split_wg(tTR_cdPT_full, nwg, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)
        tTR_tdPT = self.split_wg(thr_t2r.partition_S(tdPTtdPT), nwg, wg_idx)

        tdVrP0, tdVrP1 = tdVrP
        tiled_r2t = tcgen05.make_tmem_copy(r2t_atom, tdVrP0)
        thr_r2t = tiled_r2t.get_slice(dp_idx)
        if cutlass.const_expr(self.p_var == 1):
            # this warpgroup's P^T half: TMEM column offset wg_idx * 64 (computed on the fp32 S^T pointer, then recast)
            tdVrP_w = cute.make_tensor(
                cute.recast_ptr(tSTtST_full.iterator + wg_idx * self.p1_off, dtype=self.io_dtype), tdVrP0.layout)
            tRT_tP0 = thr_r2t.partition_D(tdVrP_w)
            cST_half = cute.make_identity_tensor((self.tile_k, self.tile_q // 2))
            tRT_cST = thr_r2t.partition_S(PdO_mma.get_slice(0).partition_A(cST_half))
        else:
            tRT_tP0 = self.split_wg(thr_r2t.partition_D(tdVrP0), nwg, wg_idx)
            tRT_cST = self.split_wg(thr_r2t.partition_S(PdO_mma.get_slice(0).partition_A(cST)), nwg, wg_idx)

        if cutlass.const_expr(self.ds_tmem):
            tiled_r2t_ds = tcgen05.make_tmem_copy(r2t_atom, tdKrdST)
            thr_r2t_ds = tiled_r2t_ds.get_slice(dp_idx)
            tRT_tdST = self.split_wg(thr_r2t_ds.partition_D(tdKrdST), nwg, wg_idx)
            cdPT_A = dSQ_mma.get_slice(0).partition_A(cdPT)
            tRT_cdPT = self.split_wg(thr_r2t_ds.partition_S(cdPT_A), nwg, wg_idx)

        scale_log2 = sm_scale * Float32(LOG2E)
        q_tile = trip_start
        kv_base = kv_tile * self.tile_k
        lane = dp_idx % 32
        col_base = wg_idx * (self.tile_q // nwg)

        if cutlass.const_expr(self.swp):
            self.compute_loop_swp(tTR_tST, tTR_rST, tTR_cST, tTR_tdPT, tTR_rdPT, tTR_cdPT, tTR_cdPT_full, tRT_tP0, tRT_cST,
                                  tiled_t2r, tiled_r2t, sLSE, sDELTA, sdS, kv_tile, trip_start, trip_end, iter_count,
                                  scale_log2, dp_idx, wg_idx, nwg,
                                  (tiled_r2t_ds, tRT_tdST, tRT_cdPT) if self.ds_tmem else None,
                                  (S_pipe, P_pipe, lse_pipe, delta_pipe, dP_pipe, dS_pipe, dST_pipe),
                                  (S_cons, P_prod, lse_cons, delta_cons, dP_cons, dS_prod, dST_prod))
        else:
            while iter_count > 0:
                _ik_push(self.iket, "cmp_w_S")
                S_pipe.consumer_wait(S_cons)
                _ik_pop(self.iket)
                _ik_push(self.iket, "cmp_a_P")
                P_pipe.producer_acquire(P_prod)
                _ik_pop(self.iket)
                _ik_push(self.iket, "cmp_w_lse")
                lse_pipe.consumer_wait(lse_cons)
                _ik_pop(self.iket)

                _ik_push(self.iket, "cmp_softmax")
                cute.copy(tiled_t2r, tTR_tST, tTR_rST)

                # causal mask only on the diagonal tile (q_tile == kv_tile)
                if q_tile == kv_tile:
                    q_base = q_tile * self.tile_q
                    for i in cutlass.range_constexpr(cute.size(tTR_rST)):
                        ik = cute.get(tTR_cST[i], mode=[0]) + kv_base
                        iq = cute.get(tTR_cST[i], mode=[1]) + q_base
                        if ik > iq:
                            tTR_rST[i] = -Float32.inf

                if cutlass.const_expr(self.shfl_consts):
                    # lane j owns columns (2j, 2j+1) of this warpgroup's 64-column half; broadcast by shuffle
                    lse_a = sLSE[col_base + 2 * lane, lse_cons.index]
                    lse_b = sLSE[col_base + 2 * lane + 1, lse_cons.index]
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rST), 2):
                        l2 = (cute.arch.shuffle_sync(lse_a, i // 2), cute.arch.shuffle_sync(lse_b, i // 2))
                        tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
                            (tTR_rST[i], tTR_rST[i + 1]), (scale_log2, scale_log2), l2)
                        tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
                        tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)
                elif cutlass.const_expr(self.vec_consts):
                    # column constants as 16-byte broadcast loads: 4 consecutive q columns per LDS.128
                    sLSE_v = cute.logical_divide(sLSE[None, lse_cons.index], cute.make_layout(4))
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rST), 4):
                        c0 = self.col_group(tTR_cST, i)
                        l4 = cute.make_rmem_tensor((4,), Float32)
                        cute.autovec_copy(sLSE_v[None, c0 // 4], l4)
                        for k in cutlass.range_constexpr(0, 4, 2):
                            tTR_rST[i + k], tTR_rST[i + k + 1] = cute.arch.fma_packed_f32x2(
                                (tTR_rST[i + k], tTR_rST[i + k + 1]), (scale_log2, scale_log2), (l4[k], l4[k + 1]))
                        if cutlass.const_expr((i // 4) % 4 < self.exp_fma):
                            self.exp2_emul4(tTR_rST, i)
                        else:
                            for k in cutlass.range_constexpr(4):
                                tTR_rST[i + k] = cute.math.exp2(tTR_rST[i + k], fastmath=True)
                else:
                    for i in cutlass.range(0, cute.size(tTR_rST), 2, unroll_full=True):
                        l2 = (sLSE[cute.get(tTR_cST[i], mode=[1]), lse_cons.index],
                              sLSE[cute.get(tTR_cST[i + 1], mode=[1]), lse_cons.index])
                        tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
                            (tTR_rST[i], tTR_rST[i + 1]), (scale_log2, scale_log2), l2)
                        tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
                        tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)

                tRT_rP = self.quantize_mix(tTR_rST)
                tRT_rP = cute.make_tensor(tRT_rP.iterator, cute.make_layout(tRT_cST.shape))

                # each warpgroup overwrites only its own S^T columns with its P^T half: no cross-warp barrier needed
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.p_var == 0 and (not self.wg_place or self.wg_bar & 1)):
                    self.compute_sync.arrive_and_wait()
                cute.copy(tiled_r2t, tRT_rP, tRT_tP0)
                cute.arch.fence_view_async_tmem_store()

                P_pipe.producer_commit(P_prod)
                P_prod.advance()
                S_pipe.consumer_release(S_cons)
                S_cons.advance()
                lse_pipe.consumer_release(lse_cons)
                lse_cons.advance()
                _ik_pop(self.iket)

                _ik_push(self.iket, "cmp_w_delta")
                delta_pipe.consumer_wait(delta_cons)
                _ik_pop(self.iket)
                _ik_push(self.iket, "cmp_w_dP")
                dP_pipe.consumer_wait(dP_cons)
                _ik_pop(self.iket)
                if cutlass.const_expr(not self.split_ds):
                    _ik_push(self.iket, "cmp_a_dS")
                    dS_pipe.producer_acquire(dS_prod)
                    _ik_pop(self.iket)

                _ik_push(self.iket, "cmp_ds")
                cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)
                if cutlass.const_expr(self.shfl_consts):
                    nd_a = sDELTA[col_base + 2 * lane, delta_cons.index]
                    nd_b = sDELTA[col_base + 2 * lane + 1, delta_cons.index]
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rdPT), 2):
                        nd = (cute.arch.shuffle_sync(nd_a, i // 2), cute.arch.shuffle_sync(nd_b, i // 2))
                        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2((tTR_rdPT[i], tTR_rdPT[i + 1]), nd)
                        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2(
                            (tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1]))
                elif cutlass.const_expr(self.vec_consts):
                    sDELTA_v = cute.logical_divide(sDELTA[None, delta_cons.index], cute.make_layout(4))
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rdPT), 4):
                        c0 = self.col_group(tTR_cdPT, i)
                        d4 = cute.make_rmem_tensor((4,), Float32)
                        cute.autovec_copy(sDELTA_v[None, c0 // 4], d4)
                        for k in cutlass.range_constexpr(0, 4, 2):
                            tTR_rdPT[i + k], tTR_rdPT[i + k + 1] = cute.arch.add_packed_f32x2(
                                (tTR_rdPT[i + k], tTR_rdPT[i + k + 1]), (d4[k], d4[k + 1]))
                            tTR_rdPT[i + k], tTR_rdPT[i + k + 1] = cute.arch.mul_packed_f32x2(
                                (tTR_rdPT[i + k], tTR_rdPT[i + k + 1]), (tTR_rST[i + k], tTR_rST[i + k + 1]))
                else:
                    for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
                        nd = (sDELTA[cute.get(tTR_cdPT[i], mode=[1]), delta_cons.index],
                              sDELTA[cute.get(tTR_cdPT[i + 1], mode=[1]), delta_cons.index])
                        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2((tTR_rdPT[i], tTR_rdPT[i + 1]), nd)
                        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2(
                            (tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1]))
                tTR_rdS = self.quantize_mix(tTR_rdPT)

                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.ds_tmem):
                    # all compute warps must have read dP^T before dS^T overwrites its first 64 columns
                    if cutlass.const_expr(not self.wg_place or self.wg_bar & 2):
                        self.compute_sync.arrive_and_wait()
                    if cutlass.const_expr(self.split_ds):
                        _ik_push(self.iket, "cmp_a_dST")
                        dST_pipe.producer_acquire(dST_prod)    # dK_{i-1} has finished reading dS^T
                        _ik_pop(self.iket)
                    tRT_rdS_t = cute.make_tensor(tTR_rdS.iterator, cute.make_layout(tRT_cdPT.shape))
                    cute.copy(tiled_r2t_ds, tRT_rdS_t, tRT_tdST)
                    cute.arch.fence_view_async_tmem_store()
                    if cutlass.const_expr(self.split_ds):
                        dST_pipe.producer_commit(dST_prod)
                        dST_prod.advance()
                dP_pipe.consumer_release(dP_cons)
                dP_cons.advance()
                if cutlass.const_expr(self.split_ds):
                    _ik_push(self.iket, "cmp_a_dS")
                    dS_pipe.producer_acquire(dS_prod)          # dQ_{i-1} has finished reading the smem dS
                    _ik_pop(self.iket)

                sdS_stage = sdS[None, None, None, dS_prod.index]
                thread_layout = cute.make_ordered_layout((self.tile_k, self.tile_q), (1, 0))
                sdS_rows = cute.composition(sdS_stage, thread_layout)
                sdS_row = cute.composition(sdS_rows[dp_idx, None], cute.make_layout(tTR_cdPT_full.shape))
                sdS_row = self.split_wg(sdS_row, nwg, wg_idx)
                cute.autovec_copy(tTR_rdS, sdS_row)

                cute.arch.fence_proxy("async.shared", space="cta")
                dS_pipe.producer_commit(dS_prod)
                dS_prod.advance()
                delta_pipe.consumer_release(delta_cons)
                delta_cons.advance()
                _ik_pop(self.iket)

                iter_count -= 1
                q_tile += 1
                if q_tile == trip_end:
                    q_tile = trip_start

        _ik_push(self.iket, "cmp_epi")
        self.epilogue(kv_tile, blk_hk, blk_b, dK, dV, tdKtdK, tdVtdV, sm_scale, (dKdV_pipe, dKdV_cons))
        _ik_pop(self.iket)

    @cute.jit
    def compute_loop_swp(self, tTR_tST, tTR_rST, tTR_cST, tTR_tdPT, tTR_rdPT, tTR_cdPT, tTR_cdPT_full, tRT_tP0, tRT_cST,
                         tiled_t2r, tiled_r2t, sLSE, sDELTA, sdS, kv_tile, trip_start, trip_end, iter_count,
                         scale_log2, dp_idx, wg_idx, nwg, ds_t, pipes, states):
        """Software-pipelined compute loop. Per iteration i:
             softmax(S_i) [S_i registers were loaded at the end of iteration i-1]
             wait dP_i, issue TMEM load of its first half; quantize/store P^T_i (hides the load latency)
             issue the second half; dS on half A; dS on half B; store dS^T_i (TMEM); release dP_i
             wait S_{i+1}, issue its TMEM load; store dS_i to smem (hides the S load latency)."""
        S_pipe, P_pipe, lse_pipe, delta_pipe, dP_pipe, dS_pipe, dST_pipe = pipes
        S_cons, P_prod, lse_cons, delta_cons, dP_cons, dS_prod, dST_prod = states
        if cutlass.const_expr(self.ds_tmem):
            tiled_r2t_ds, tRT_tdST, tRT_cdPT_A = ds_t
        n_rep = cute.size(tTR_rST, mode=[2])          # 4 x 16 columns per thread
        n_el = cute.size(tTR_rST)                     # 64
        half = n_el // 2
        tTR_rdS = cute.make_rmem_tensor(tTR_rdPT.shape, self.io_dtype)
        tRT_rP = cute.make_rmem_tensor(tTR_rST.shape, self.io_dtype)
        q_tile = trip_start
        kv_base = kv_tile * self.tile_k

        # prologue: S_0
        _ik_push(self.iket, "cmp_w_S")
        S_pipe.consumer_wait(S_cons)
        _ik_pop(self.iket)
        self.ld_reps(tiled_t2r, tTR_tST, tTR_rST, 0, n_rep)

        while iter_count > 0:
            _ik_push(self.iket, "cmp_w_lse")
            lse_pipe.consumer_wait(lse_cons)
            _ik_pop(self.iket)
            _ik_push(self.iket, "cmp_softmax")
            if q_tile == kv_tile:
                q_base = q_tile * self.tile_q
                for i in cutlass.range_constexpr(n_el):
                    ik = cute.get(tTR_cST[i], mode=[0]) + kv_base
                    iq = cute.get(tTR_cST[i], mode=[1]) + q_base
                    if ik > iq:
                        tTR_rST[i] = -Float32.inf
            sLSE_v = cute.logical_divide(sLSE[None, lse_cons.index], cute.make_layout(4))
            for i in cutlass.range_constexpr(0, n_el, 4):
                c0 = self.col_group(tTR_cST, i)
                l4 = cute.make_rmem_tensor((4,), Float32)
                cute.autovec_copy(sLSE_v[None, c0 // 4], l4)
                for k in cutlass.range_constexpr(0, 4, 2):
                    tTR_rST[i + k], tTR_rST[i + k + 1] = cute.arch.fma_packed_f32x2(
                        (tTR_rST[i + k], tTR_rST[i + k + 1]), (scale_log2, scale_log2), (l4[k], l4[k + 1]))
                for k in cutlass.range_constexpr(4):
                    tTR_rST[i + k] = cute.math.exp2(tTR_rST[i + k], fastmath=True)
            lse_pipe.consumer_release(lse_cons)
            lse_cons.advance()
            _ik_pop(self.iket)

            # dP_i: if it has already landed, issue the first half of its TMEM load before the P^T store
            tok = dP_pipe.consumer_try_wait(dP_cons)
            if tok:
                self.ld_reps(tiled_t2r, tTR_tdPT, tTR_rdPT, 0, n_rep // 2)

            _ik_push(self.iket, "cmp_a_P")
            P_pipe.producer_acquire(P_prod)
            _ik_pop(self.iket)
            _ik_push(self.iket, "cmp_pst")
            self.quantize_into(tTR_rST, tRT_rP, 0, n_el // 4)
            tRT_rP_v = cute.make_tensor(tRT_rP.iterator, cute.make_layout(tRT_cST.shape))
            if cutlass.const_expr(self.swp_nowait):
                if not tok:
                    cute.arch.fence_view_async_tmem_load()
            else:
                cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.p_var == 0 and (not self.wg_place or self.wg_bar & 1)):
                self.compute_sync.arrive_and_wait()
            cute.copy(tiled_r2t, tRT_rP_v, tRT_tP0)
            cute.arch.fence_view_async_tmem_store()
            P_pipe.producer_commit(P_prod)
            P_prod.advance()
            S_pipe.consumer_release(S_cons)
            S_cons.advance()
            _ik_pop(self.iket)

            # dS_i
            _ik_push(self.iket, "cmp_w_dP")
            dP_pipe.consumer_wait(dP_cons, tok)
            _ik_pop(self.iket)
            if not tok:
                self.ld_reps(tiled_t2r, tTR_tdPT, tTR_rdPT, 0, n_rep // 2)
            self.ld_reps(tiled_t2r, tTR_tdPT, tTR_rdPT, n_rep // 2, n_rep)
            _ik_push(self.iket, "cmp_w_delta")
            delta_pipe.consumer_wait(delta_cons)
            _ik_pop(self.iket)
            _ik_push(self.iket, "cmp_ds")
            sDELTA_v = cute.logical_divide(sDELTA[None, delta_cons.index], cute.make_layout(4))
            for hf in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(hf * half, (hf + 1) * half, 4):
                    c0 = self.col_group(tTR_cdPT, i)
                    d4 = cute.make_rmem_tensor((4,), Float32)
                    cute.autovec_copy(sDELTA_v[None, c0 // 4], d4)
                    if cutlass.const_expr(self.p_bf16):
                        p4 = cute.make_rmem_tensor((4,), Float32)
                        p4.store(cute.make_tensor(tRT_rP.iterator + i, cute.make_layout(4)).load().to(Float32))
                        pa, pb, pc, pd = p4[0], p4[1], p4[2], p4[3]
                    else:
                        pa, pb, pc, pd = tTR_rST[i], tTR_rST[i + 1], tTR_rST[i + 2], tTR_rST[i + 3]
                    tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2((tTR_rdPT[i], tTR_rdPT[i + 1]), (d4[0], d4[1]))
                    tTR_rdPT[i + 2], tTR_rdPT[i + 3] = cute.arch.add_packed_f32x2((tTR_rdPT[i + 2], tTR_rdPT[i + 3]), (d4[2], d4[3]))
                    tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdPT[i], tTR_rdPT[i + 1]), (pa, pb))
                    tTR_rdPT[i + 2], tTR_rdPT[i + 3] = cute.arch.mul_packed_f32x2((tTR_rdPT[i + 2], tTR_rdPT[i + 3]), (pc, pd))
                self.quantize_into(tTR_rdPT, tTR_rdS, hf * half // 4, (hf + 1) * half // 4)

            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.ds_tmem):
                if cutlass.const_expr(not self.wg_place or self.wg_bar & 2):
                    self.compute_sync.arrive_and_wait()
                if cutlass.const_expr(self.split_ds):
                    _ik_push(self.iket, "cmp_a_dST")
                    dST_pipe.producer_acquire(dST_prod)
                    _ik_pop(self.iket)
                else:
                    _ik_push(self.iket, "cmp_a_dS")
                    dS_pipe.producer_acquire(dS_prod)
                    _ik_pop(self.iket)
                tRT_rdS_t = cute.make_tensor(tTR_rdS.iterator, cute.make_layout(tRT_cdPT_A.shape))
                cute.copy(tiled_r2t_ds, tRT_rdS_t, tRT_tdST)
                cute.arch.fence_view_async_tmem_store()
                if cutlass.const_expr(self.split_ds):
                    dST_pipe.producer_commit(dST_prod)
                    dST_prod.advance()
            dP_pipe.consumer_release(dP_cons)
            dP_cons.advance()
            delta_pipe.consumer_release(delta_cons)
            delta_cons.advance()
            _ik_pop(self.iket)

            iter_count -= 1
            q_tile += 1
            if q_tile == trip_end:
                q_tile = trip_start

            # S_{i+1}: if it has already landed, issue its TMEM load before the dS_i smem store
            tokS = Boolean(False)
            if iter_count > 0:
                tokS = S_pipe.consumer_try_wait(S_cons)
                if tokS:
                    self.ld_reps(tiled_t2r, tTR_tST, tTR_rST, 0, n_rep)

            _ik_push(self.iket, "cmp_sts")
            if cutlass.const_expr(self.split_ds or not self.ds_tmem):
                _ik_push(self.iket, "cmp_a_dS")
                dS_pipe.producer_acquire(dS_prod)
                _ik_pop(self.iket)
            sdS_stage = sdS[None, None, None, dS_prod.index]
            thread_layout = cute.make_ordered_layout((self.tile_k, self.tile_q), (1, 0))
            sdS_rows = cute.composition(sdS_stage, thread_layout)
            sdS_row = cute.composition(sdS_rows[dp_idx, None], cute.make_layout(tTR_cdPT_full.shape))
            sdS_row = self.split_wg(sdS_row, nwg, wg_idx)
            cute.autovec_copy(tTR_rdS, sdS_row)
            cute.arch.fence_proxy("async.shared", space="cta")
            dS_pipe.producer_commit(dS_prod)
            dS_prod.advance()
            _ik_pop(self.iket)

            if iter_count > 0:
                _ik_push(self.iket, "cmp_w_S")
                S_pipe.consumer_wait(S_cons, tokS)
                _ik_pop(self.iket)
                if not tokS:
                    self.ld_reps(tiled_t2r, tTR_tST, tTR_rST, 0, n_rep)

    @cute.jit
    def ld_reps(self, tiled_t2r, src, dst, r0, r1):
        """Issue the TMEM->register copies for repetitions [r0, r1) (16 columns each)."""
        for r in cutlass.range_constexpr(r0, r1):
            cute.copy(tiled_t2r, src[None, None, r], dst[None, None, r])

    @cute.jit
    def quantize_into(self, inp, out, g0, g1):
        """bf16-round groups [g0, g1) of 4 consecutive fp32 values of `inp` into `out` (same shape)."""
        t_frg = cute.logical_divide(inp, cute.make_layout(4))
        o_frg = cute.make_tensor(out.iterator, t_frg.layout)
        for g in cutlass.range_constexpr(g0, g1):
            o_frg[None, g].store(t_frg[None, g].load().to(self.io_dtype))

    # ---------------------------------------------------------------- reduce
    @cute.jit
    def reduce(self, dSK_mma, tdQtdQ, tma_dQ, mdQacc, sdQ, blk_hk, blk_b, iter_count, trip_start, trip_end, sm_scale, pipes):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        dQ_pipe, dq_store_pipe = pipes
        hr = Int32(0)
        hb = ((hr, blk_hk), blk_b)
        q_tile = trip_start

        dQ_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.dQ_stages)
        st_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.dq_store_stages)

        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dq_sub_cols)), self.acc_dtype)
        gdQ = cute.local_tile(mdQacc, (self.tile_q, self.dq_sub_cols), (None, None, None))
        cdQ = cute.make_identity_tensor((self.dSK_tiler[0], self.dSK_tiler[1]))
        thread_idx = tidx % (self.num_reduce_warps * 32)
        tdQtdQ = tdQtdQ[(None, None), 0, 0]
        tiled_t2r = tcgen05.make_tmem_copy(ld_atom, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)
        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_sdQ = thr_t2r.partition_D(sdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)
        tdQsdQ, tdQgdQ = cpasync.tma_partition(tma_dQ, 0, cute.make_layout(1),
                                               cute.group_modes(sdQ, 0, 2), cute.group_modes(gdQ, 0, 2))

        while iter_count > 0:
            _ik_push(self.iket, "red_w_dQ")
            dQ_pipe.consumer_wait(dQ_cons)
            _ik_pop(self.iket)
            _ik_push(self.iket, "red_ld")
            tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
            cute.copy(tiled_t2r, tTR_tdQ, tTR_rdQ)
            cute.arch.fence_view_async_tmem_load()
            dQ_pipe.consumer_release(dQ_cons)
            dQ_cons.advance()
            _ik_pop(self.iket)

            _ik_push(self.iket, "red_st")
            if cutlass.const_expr(self.dq_f16):
                tTR_rdQh = cute.make_rmem_tensor(tTR_rdQ.shape, self.dq_dtype)
                t_frg = cute.logical_divide(tTR_rdQ, cute.make_layout(4))
                o_frg = cute.make_tensor(tTR_rdQh.iterator, t_frg.layout)
                for g in cutlass.range_constexpr(cute.size(tTR_rdQ) // 4):
                    o_frg[None, g].store((t_frg[None, g].load() * sm_scale).to(self.dq_dtype))
                tTR_rdQs = tTR_rdQh
            else:
                tTR_rdQs = tTR_rdQ
            for i in cutlass.range(cute.size(tTR_cdQ, mode=[2]), unroll_full=True):
                if warp_idx == 0:
                    dq_store_pipe.producer_acquire()
                self.reduce_sync.arrive_and_wait()
                cute.autovec_copy(tTR_rdQs[None, None, i], tTR_sdQ[None, None, 0, st_state.index])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.reduce_sync.arrive_and_wait()
                if warp_idx == 0:
                    if cutlass.const_expr(not self.no_dq_store):
                        cute.copy(tma_dQ, tdQsdQ[None, st_state.index], tdQgdQ[None, q_tile, i, hb])
                    dq_store_pipe.producer_commit()
                st_state.advance()
            _ik_pop(self.iket)

            iter_count -= 1
            q_tile += 1
            if q_tile == trip_end:
                q_tile = trip_start
                hr += 1
                hb = ((hr, blk_hk), blk_b)
        dq_store_pipe.producer_tail()

    # -------------------------------------------------------------- epilogue
    @cute.jit
    def epilogue(self, kv_tile, blk_hk, blk_b, dK, dV, tdKtdK, tdVtdV, sm_scale, pipes):
        tidx, _, _ = cute.arch.thread_idx()
        dKdV_pipe, dKdV_cons = pipes
        hb = ((Int32(0), blk_hk), blk_b)
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        nwg = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * 32)) // 128

        tdKtdK = tdKtdK[(None, None), 0, 0]
        tdVtdV = tdVtdV[(None, None), 0, 0]
        gdK = cute.local_tile(dK, (self.tile_k, self.D), (None, None, None))[None, None, kv_tile, 0, hb]
        gdV = cute.local_tile(dV, (self.tile_k, self.D), (None, None, None))[None, None, kv_tile, 0, hb]
        cKV = cute.make_identity_tensor((self.tile_k, self.D))

        tiled_dK = tcgen05.make_tmem_copy(ld_atom, tdKtdK)
        thr_dK = tiled_dK.get_slice(dp_idx)
        tTR_cdK = self.split_wg(thr_dK.partition_D(cKV), nwg, wg_idx)
        tTR_gdK = self.split_wg(thr_dK.partition_D(gdK), nwg, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = self.split_wg(thr_dK.partition_S(tdKtdK), nwg, wg_idx)

        tiled_dV = tcgen05.make_tmem_copy(ld_atom, tdVtdV)
        thr_dV = tiled_dV.get_slice(dp_idx)
        tTR_cdV = self.split_wg(thr_dV.partition_D(cKV), nwg, wg_idx)
        tTR_gdV = self.split_wg(thr_dV.partition_D(gdV), nwg, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = self.split_wg(thr_dV.partition_S(tdVtdV), nwg, wg_idx)

        dKdV_pipe.consumer_wait(dKdV_cons)
        cute.copy(tiled_dV, tTR_tdV, tTR_rdV)
        self.store_tile(tTR_gdV, tTR_rdV)
        cute.arch.fence_view_async_tmem_load()
        dKdV_pipe.consumer_release(dKdV_cons)
        dKdV_cons.advance()

        dKdV_pipe.consumer_wait(dKdV_cons)
        cute.copy(tiled_dK, tTR_tdK, tTR_rdK)
        for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
            tTR_rdK[i] = sm_scale * tTR_rdK[i]
        self.store_tile(tTR_gdK, tTR_rdK)
        cute.arch.fence_view_async_tmem_load()
        dKdV_pipe.consumer_release(dKdV_cons)
        dKdV_cons.advance()

    @cute.jit
    def store_tile(self, gmem, regs):
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype, num_bits_per_copy=128)
        copy_op = cute.make_cotiled_copy(copy_atom, cute.make_layout((1, 128 // self.io_dtype.width)), regs.layout)
        thr_copy = copy_op.get_slice(0)
        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(self.quantize(regs, 4))
        cute.copy(copy_atom, tCr, tCg)

    # --------------------------------------------------------------- helpers
    def gapped_a_layout(self, frag_layout):
        """TMEM A-operand fragment layout ((128,16),1,(4,2)):((lane,1),0,(16,64)) with the second k-block group
        moved from +64 to +128 bf16 (= 64 columns): k-blocks 4..7 then live in the second warpgroup's region."""
        shape = frag_layout.shape
        stride = frag_layout.stride
        assert cute.size(shape[2]) == 8 and stride[2][1] == 4 * stride[2][0], (shape, stride)
        new_stride = (stride[0], stride[1], (stride[2][0], 2 * stride[2][1]))
        return cute.make_layout(shape, stride=new_stride)

    @cute.jit
    def dst_frag(self, tdKrdST, kb, stage):
        ret = None
        if cutlass.const_expr(self.ds_tmem):
            ret = tdKrdST[None, None, kb]
        else:
            ret = tdKrdST[None, None, kb, stage]
        return ret

    def col_group(self, coord_t, i):
        """q column of fragment element i; checks that elements i..i+3 are 4 consecutive, 16B-aligned columns."""
        cols = [cute.get(coord_t[i + k], mode=[1]) for k in range(4)]
        if all(isinstance(c, int) for c in cols):
            assert cols[0] % 4 == 0 and cols == list(range(cols[0], cols[0] + 4)), cols
        # dynamic coordinates (warpgroup split): alignment follows from the TMEM copy layout (16 consecutive columns per lane)
        return cols[0]

    @cute.jit
    def split_wg(self, t, nwg, wg_idx):
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], (nwg, cute.size(t, mode=[2]) // nwg))))
            ret = p[None, None, (wg_idx, None)]
        else:
            p = cute.composition(t, cute.make_layout((t.shape[0], t.shape[1], t.shape[2], (nwg, cute.size(t, mode=[3]) // nwg))))
            ret = p[None, None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def exp2_emul4(self, t, i):
        """2^x for t[i..i+3] on the FMA pipe: Cody-Waite split x = j + f, degree-3 polynomial for 2^f on
        [-0.5, 0.5] (max rel err 7.7e-5), 2^j folded into the exponent field with an integer add. x <= 0 expected
        (masked entries are -inf): clamp at -126 so the result underflows to ~0 instead of producing NaN."""
        tf = cute.make_rmem_tensor((8,), Float32)
        ti = cute.recast_tensor(tf, Int32)
        t_i = cute.recast_tensor(t, Int32)
        for k in cutlass.range_constexpr(4):
            x = cute.arch.fmax(t[i + k], Float32(-126.0))
            tf[k] = x + Float32(12582912.0)                 # 1.5 * 2^23: low mantissa bits hold round(x)
            jf = tf[k] - Float32(12582912.0)
            f = x - jf
            p = Float32(0.0550886838) * f + Float32(0.242604051)
            p = p * f + Float32(0.693276242)
            p = p * f + Float32(0.99992894)
            tf[4 + k] = p
            t_i[i + k] = ti[4 + k] + ((ti[k] - Int32(0x4B400000)) << 23)

    @cute.jit
    def quantize_mix(self, inp):
        """fp32 -> bf16; of every 4 groups of 4 values, `cvt_alu` groups are packed with integer ops (ALU pipe,
        round-to-nearest-even unless cvt_trunc) and the rest with cvt.rn.bf16x2.f32 (XU pipe)."""
        out = cute.make_rmem_tensor(inp.shape, self.io_dtype)
        n = cute.size(inp) // 4
        t_frg = cute.logical_divide(inp, cute.make_layout(4))
        o_frg = cute.make_tensor(out.iterator, t_frg.layout)
        inp_i = cute.recast_tensor(inp, Int32)
        out_i = cute.recast_tensor(out, Int32)
        for g in cutlass.range_constexpr(n):
            if cutlass.const_expr(g % 4 < self.cvt_alu):
                for k in cutlass.range_constexpr(2):
                    a = inp_i[4 * g + 2 * k]
                    b = inp_i[4 * g + 2 * k + 1]
                    if cutlass.const_expr(self.cvt_trunc == 0):
                        a = a + (Int32(0x7FFF) + ((a >> 16) & Int32(1)))
                        b = b + (Int32(0x7FFF) + ((b >> 16) & Int32(1)))
                    out_i[2 * g + k] = ((a >> 16) & Int32(0xFFFF)) | (b & Int32(-65536))
            else:
                o_frg[None, g].store(t_frg[None, g].load().to(self.io_dtype))
        return out

    @cute.jit
    def quantize(self, inp, frg_cnt):
        out = cute.make_rmem_tensor(inp.shape, self.io_dtype)
        n = cute.size(inp) // frg_cnt
        t_frg = cute.logical_divide(inp, cute.make_layout(frg_cnt))
        o_frg = cute.make_tensor(out.iterator, t_frg.layout)
        for i in cutlass.range(n, unroll_full=True):
            o_frg[None, i].store(t_frg[None, i].load().to(self.io_dtype))
        return out
