# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted for the FlashInfer-Bench attention_bwd task from the NVIDIA CUTLASS 4.5.0
# CuTe DSL example examples/python/CuTeDSL/blackwell/fmha_bwd.py (BSD-3-Clause).
# Modifications: standalone module, mask helpers moved to fmha_mask.py, test harness removed.

import math
import os
from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32, Float16, BFloat16, Boolean

from . import fmha_mask as fmha_utils
from . import umma_raw

class BlackwellFusedMultiHeadAttentionBackward:
    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        varlen: bool,
        mask_type: fmha_utils.MaskEnum,
    ):
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.tile_shape_Q = mma_tiler[0]
        self.tile_shape_K = mma_tiler[1]
        self.tile_shape_dQ_K = mma_tiler[2]
        self.tile_shape_dV_dO = mma_tiler[2]
        # S^T as N=64 halves through a 64-column TMEM region, P^T in its own region
        # (FMHA_BWD_SHALF=1); 0 = full-width S with P^T aliasing it (v006)
        self.s_halves = os.environ.get("FMHA_BWD_SHALF", "0") == "1"
        self.s_n = mma_tiler[0] // 2 if self.s_halves else mma_tiler[0]
        self.n_s_halves = mma_tiler[0] // self.s_n
        # For S: (keys, q per S half, d); Q is loaded in (s_n x d) half tiles
        self.KQ_mma_tiler = (
            mma_tiler[1],
            self.s_n,
            mma_tiler[2],
        )
        # dK B operand (Q^T) per Q entry: (keys, d, q per entry)
        self.dSQ_B_tiler = (mma_tiler[1], mma_tiler[2], self.s_n)
        # Layout mode. fullw: dP and dQ as full N=128 MMAs through one 128-column TMEM region
        # and V as a smem A operand (2 Q stages). Otherwise (v005): V in TMEM (64 cols), dP and
        # dQ as N=64 halves through a shared 64-column region (3 Q stages).
        self.fullw = os.environ.get("FMHA_BWD_FULLW", "1") == "1"
        self.n_halves = 1 if self.fullw else 2
        # D23: hand-built tcgen05.mma issue (inline PTX; descriptors built once, advanced by
        # immediates) for the listed MMA groups: FMHA_BWD_RAWMMA=s,dp,dv,dq,dk (1 = all)
        raw = os.environ.get("FMHA_BWD_RAWMMA", "").replace("+", ",")
        self.raw_mma = (
            {"s", "dp", "dv", "dq", "dk"} if raw in ("1", "all") else set(raw.split(",")) - {""}
        )
        if self.raw_mma and not self.fullw:
            raise ValueError("FMHA_BWD_RAWMMA needs the fullw layout")
        # D24: dQ partial tiles added into the fp16 workspace with 16-byte vector atomics
        # straight from the reduce warps' registers (no smem staging, no TMA reduce-add):
        # FMHA_BWD_DQRED=atomic | tma
        self.dq_atomic = os.environ.get("FMHA_BWD_DQRED", "tma") == "atomic"
        if self.dq_atomic:
            os.environ.setdefault("FMHA_BWD_FUSE_DQ", "0")
        # D25: each compute warpgroup stores its bf16 P^T into its own S half (warpgroup 1 at
        # columns 96-127 of the S region instead of 32-63), so no warpgroup overwrites S columns
        # the other one still reads and the P-store barrier disappears; the dV MMA reads P^T
        # through the hand-built path with split k-block addresses (FMHA_BWD_PSPLIT=1)
        self.p_split = os.environ.get("FMHA_BWD_PSPLIT", "0") == "1"
        if self.p_split:
            self.raw_mma = self.raw_mma | {"dv"}
            if self.s_halves or not self.fullw:
                raise ValueError("FMHA_BWD_PSPLIT needs the fullw layout without S halves")
        # experiment: keep one never-executed cute.gemm in the kernel (FMHA_BWD_RAWMMA_DUMMY=1)
        self.raw_dummy_gemm = os.environ.get("FMHA_BWD_RAWMMA_DUMMY", "0") == "1"
        # For dP: (keys, q half, d)
        self.half_n = mma_tiler[0] // self.n_halves
        self.VdO_mma_tiler = (
            mma_tiler[1],
            self.half_n,
            mma_tiler[2],
        )
        # For dV
        self.PdO_mma_tiler = (
            mma_tiler[1],
            mma_tiler[2],
            mma_tiler[0],
        )
        # For dK
        self.dSQ_mma_tiler = (
            mma_tiler[1],
            mma_tiler[2],
            mma_tiler[0],
        )
        # For dQ: (q, d half, keys)
        self.dSK_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2] // self.n_halves,
            mma_tiler[1],
        )
        # full-width tilers used only to build smem layouts / TMA atoms for dO and K^T
        self.dO_layout_tiler = (mma_tiler[1], mma_tiler[0], mma_tiler[2])
        self.KT_layout_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1])
        self.cluster_size = 1  # persistent kernel: one CTA per cluster
        self.sched_stages = 2  # scheduler ring slots (load warp -> other warps)
        self.sched_bytes = 256  # global work counter at the start of the workspace
        if varlen:
            self.cluster_size = 1  # the compact varlen kv-block list is not pair-aligned
        self.cluster_shape_mn = (self.cluster_size, 1)
        self.varlen = varlen
        self.mask_type = mask_type

        # =================== Sum OdO ================================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = (
            self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        )
        self.sum_OdO_elem_per_load = 8  # 16-byte bf16 loads
        # Global dQ accumulator type (TMA reduce-add target). fp16 halves the reduce
        # traffic; dQ magnitudes in these workloads are O(1), 16 partial sums are safe.
        dq_acc = os.environ.get("FMHA_BWD_DQACC", "f16")
        self.dq_acc_dtype = {"f16": Float16, "bf16": BFloat16, "f32": Float32}[dq_acc]
        self.pre_acc_elem_per_store = 4  # 16-byte stores of the fp32-recast accumulator rows
        # D26: bandwidth-oriented preprocess / convert kernels (FMHA_BWD_PRE=v2, default) or
        # the original row-per-thread-group kernels (v1)
        self.pre_v2 = os.environ.get("FMHA_BWD_PRE", "v2") == "v2"
        # row mapping of the v2 kernels: "head" = a block covers 64 consecutive rows of one
        # head (contiguous in the head-major accumulator, 256-byte pieces at a 4 KB stride in
        # the bshd tensors); "tok" = a block covers 64 consecutive (row, head) pairs of the
        # bshd row space (contiguous in O/dO/dQ, scattered in the accumulator)
        self.pre_map_tok = os.environ.get("FMHA_BWD_PRE_MAP", "head") == "tok"
        # timing-only ablation: skip the accumulator zeroing (results are wrong)
        self.pre_nozero = os.environ.get("FMHA_BWD_PRE_NOZERO", "0") == "1"
        # D26c: dQ convert fused into the main kernel (FMHA_BWD_FUSE_DQ=1): per q tile an
        # arrival counter; the CTA whose reduce-add completes the tile's last kv block loads
        # the fp16 tile back, scales/converts it into dQ and (FMHA_BWD_ACC_ZERO=main) zeroes
        # the accumulator tile for the next call, so the preprocess only zeroes the counters.
        # Needs the fullw layout, the TMA reduce-add path and the causal mask without windows.
        # default off (D26c result, 2026-10-21): correct, but the bf16 output writes inside the
        # latency-sensitive main kernel cost more than the standalone convert kernel
        self.fuse_dq = os.environ.get("FMHA_BWD_FUSE_DQ", "0") == "1"
        self.acc_zero_main = os.environ.get("FMHA_BWD_ACC_ZERO", "main") == "main"
        # converts are released per 128-row block once all H heads of the block are complete
        # (the 16 head tiles are pushed together and converted concurrently, which gives the
        # scattered bshd output writes the DRAM page locality of the standalone convert)
        self.fz_rowblock = os.environ.get("FMHA_BWD_FZ_ROWBLOCK", "1") == "1" and not varlen
        if not self.fuse_dq:
            self.acc_zero_main = False
        if self.fuse_dq and not self.pre_v2:
            raise ValueError("FMHA_BWD_FUSE_DQ needs the v2 preprocess (FMHA_BWD_PRE=v2)")
        # timing-only ablations of the fused path: nowait (no bulk wait_group), nofence,
        # noatomic (counter never incremented: no tile is ever converted), noconv (counted
        # but never converted)
        self.fz_dbg = set(os.environ.get("FMHA_BWD_FZDBG", "").replace("+", ",").split(",")) - {""}
        # finisher warp (the first spare warp): register budget and depth of the tile ring
        # from the reduce warps. The budget must not exceed the 80 of the mma/load warps that
        # share its warpgroup (setmaxnreg is a warpgroup-level instruction): 96 hangs the CTA
        # at the first tile even when the finisher does nothing (2026-10-21 hang probes)
        self.num_regs_finisher = int(os.environ.get("FMHA_BWD_FZ_REGS", "80"))
        # FMHA_BWD_FZSTATS=1: global counters in the scheduler word block (workspace ints
        # 1..4): converts, total convert ns, max convert ns, ns reduce warp 0 blocked on the ring
        self.fz_stats = os.environ.get("FMHA_BWD_FZSTATS", "0") == "1"
        self.fz_stages = int(os.environ.get("FMHA_BWD_FZ_STAGES", "8"))
        # block shapes (measured 2026-10-21, MHA breakdown): preprocess 8 warps x 4 steps =
        # 78.7 us (4 warps x 8 steps: 84.2), convert 8 warps x 2 steps = 48.4 us (51.5).
        # A step = two 256-byte rows per warp (16 lanes x 16 B each); all loads of a lane are
        # issued before the math, so more steps mean more loads in flight but more registers
        self.pre_threads = int(os.environ.get("FMHA_BWD_PRE_THREADS", "256"))  # warps x 32
        self.pre_rows_per_warp = 2 * int(os.environ.get("FMHA_BWD_PRE_STEPS", "4"))
        self.pre_rows_per_block = self.pre_rows_per_warp * (self.pre_threads // 32)
        # raw-pointer loads/stores in the preprocess (NCU v009: the DSL-indexed version is
        # issue bound, 78% issue-active at 43% memory throughput; FMHA_BWD_PRE_RAW=0 = old path)
        self.pre_raw = os.environ.get("FMHA_BWD_PRE_RAW", "1") == "1"
        self.cvt_threads = int(os.environ.get("FMHA_BWD_CVT_THREADS", "256"))
        self.cvt_rows_per_warp = 2 * int(os.environ.get("FMHA_BWD_CVT_STEPS", "2"))
        self.cvt_rows_per_block = self.cvt_rows_per_warp * (self.cvt_threads // 32)

        # Warp layout (FMHA_BWD_WARPS). c16: four compute warpgroups (16 warps, warpgroup g
        # owns the 16-column chunks g and g+4 of S^T / dP^T / dQ) that also drain dQ and run
        # the dK/dV epilogue, plus the MMA and load warps (18 warps, 576 threads). v6: the
        # v006 layout (4 reduce, 8 compute, mma, load, empty = 15 warps).
        warps_mode = os.environ.get("FMHA_BWD_WARPS", "v6")
        self.c16 = warps_mode in ("c16", "c16r")   # 16 compute warps
        self.c16r = warps_mode == "c16r"           # ... plus 4 reduce warps (dQ drain, epilogue)
        if self.c16r:
            self.reduce_warp_id = (0, 1, 2, 3)
            self.compute_warp_id = tuple(range(4, 20))
            self.mma_warp_id = 20
            self.load_warp_id = 21
            self.empty_warp_id = 22
            self.num_reduce_warps = 4
            self.num_compute_warps = 16
            self.num_aux_warps = 2
        elif self.c16:
            self.reduce_warp_id = ()
            self.compute_warp_id = tuple(range(16))
            self.mma_warp_id = 16
            self.load_warp_id = 17
            self.empty_warp_id = 18
            self.num_reduce_warps = 0
            self.num_compute_warps = 16
            self.num_aux_warps = 2
        else:
            self.reduce_warp_id = (0, 1, 2, 3)
            self.compute_warp_id = (4, 5, 6, 7, 8, 9, 10, 11)
            self.mma_warp_id = 12
            self.load_warp_id = 13
            self.empty_warp_id = 14
            self.num_reduce_warps = 4
            self.num_compute_warps = 8
            self.num_aux_warps = 4
        # dQ staging for the TMA reduce-add: per-warpgroup 128 x 16 fp16 buffers (c16) or two
        # 128 x 32 stages shared by the reduce warps (v6); 16 KB either way
        self.dq_stage_cols = 32
        # Warp-level pipeline commits by the compute warps: one mbarrier arrival per warp
        # (after a warp sync) instead of one per thread (FMHA_BWD_WCOMMIT=1|0)
        self.warp_commit = os.environ.get("FMHA_BWD_WCOMMIT", "1" if self.c16 else "0") == "1"
        self.compute_group_threads = (
            self.num_compute_warps if self.warp_commit else self.num_compute_warps * 32
        )
        # c16: one 8 KB chunk per compute warpgroup; v6 / c16r: two stages for the reduce warps
        # (four 8 KB chunks aliasing the dS buffer with FMHA_BWD_DO2=1)
        self.dq_stage_bufs = 2

        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (
            self.num_reduce_warps + self.num_compute_warps + self.num_aux_warps
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=max(self.num_reduce_warps, 4) * self.threads_per_warp,
        )
        # c16: per-warpgroup barriers (ids 9..12) around the dQ staging buffer hand-off
        self.dq_stage_barrier_base = 9
        # compute warps -> mma warp: V has been stored into TMEM
        self.v_tmem_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=(self.num_compute_warps + 1) * self.threads_per_warp,
        )
        # P^T (bf16) overwrites the S region that both compute warpgroups read: warpgroup
        # g arrives on p_load_barrier[g] once its S loads are done and syncs on the other
        # group's barrier right before its P store (128 arrive + 128 sync = 256)
        self.p_load_barrier = (
            pipeline.NamedBarrier(barrier_id=7, num_threads=2 * 128),
            pipeline.NamedBarrier(barrier_id=8, num_threads=2 * 128),
        )

        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.tmem_dK_offset + mma_tiler[2]
        self.tmem_dQ_offset = self.tmem_dV_offset + mma_tiler[2]
        self.tmem_dP_offset = self.tmem_dQ_offset
        if self.fullw:
            self.tmem_V_offset = self.tmem_dQ_offset  # unused (V stays in smem)
            self.tmem_S_offset = self.tmem_dQ_offset + mma_tiler[2]
            # P^T (bf16, 64 columns) after the S half region, or aliasing S (s_halves=0)
            self.tmem_P_offset = self.tmem_S_offset + (self.s_n if self.s_halves else 0)
        else:
            self.tmem_P_offset = None
            self.tmem_V_offset = self.tmem_dQ_offset + mma_tiler[2] // 2
            self.tmem_S_offset = self.tmem_V_offset + mma_tiler[2] // 2

        # Register budget per role (setmaxnreg); overridable for experiments via
        # FMHA_BWD_REGS="reduce,compute,mma,empty,load". Budget: 4*reduce + 8*compute +
        # mma + empty + load warps, 32 threads each, must stay <= 65536 registers.
        # c16r: 704 threads cap the compile at 88 registers; 0 = no setmaxnreg for that role
        regs_default = ("0:0:0:0:0" if self.c16r else "24:0:80:24:80") if self.c16 else "128:152:80:40:80"
        regs = os.environ.get("FMHA_BWD_REGS", regs_default).replace(",", ":").split(":")
        (
            self.num_regs_reduce,
            self.num_regs_compute,
            self.num_regs_mma,
            self.num_regs_empty,
            self.num_regs_load,
        ) = [int(x) for x in regs]
        # Grouped longest-processing-time-first block order: (kv head, batch) pairs are
        # processed in groups of lpt_group pairs (runtime argument); inside a group the kv
        # blocks with the most causal trips (low kv index) are dispatched first. Group 1
        # reproduces the plain grid order; the group size bounds the concurrent Q/dO
        # working set so that it stays L2 resident.
        self.lpt_order = os.environ.get("FMHA_BWD_LPT", "1") == "1"
        # specialized causal/residual mask (window_size_left None, window_size_right 0)
        self.fast_mask = os.environ.get("FMHA_BWD_FASTMASK", "0") == "1"
        # In-kernel event tracing (IKET) instrumentation of pipeline waits
        self.iket = os.environ.get("FMHA_BWD_IKET", "0") == "1"
        # IKET-only: the load warp waits for each Q TMA it issued (measures the flight time;
        # slows the kernel down)
        self.iket_tmawait = os.environ.get("FMHA_BWD_IKET_TMAWAIT", "0") == "1"
        # MMA issue order inside the main loop: "dkq_first" issues dQ/dK of the previous
        # tile (releasing its Q stage, so the next Q TMA load starts earlier) before the
        # S MMA of the current tile; "s_first" is the original order.
        self.mma_order = os.environ.get("FMHA_BWD_MMA_ORDER", "s_first")
        # Fraction of P elements (pairs) whose exp2 is emulated on the FMA pipe
        emu = os.environ.get("FMHA_BWD_EX2_EMU", "0/1").split("/")
        self.ex2_emu_num, self.ex2_emu_den = int(emu[0]), int(emu[1])
        # Timing-only ablations (results become wrong): comma separated subset of
        # noexp (skip exp2), nolse (skip lse smem loads), nodsst (skip the dS smem store)
        self.dbg = set(os.environ.get("FMHA_BWD_DBG", "").replace("+", ",").split(",")) - {""}
        # L2 prefetch distance (q tiles ahead) for the Q and dO tiles, 0 disables
        self.prefetch_dist = int(os.environ.get("FMHA_BWD_PF", "0"))
        self.pf_rows_per_lane = 4  # 128 rows x 2 lines of 128 B per tile = 8 prefetches per lane
        # dS smem store through a flat per-row view (vectorizable), FMHA_BWD_DSVEC=1|0
        self.ds_store_vec = os.environ.get("FMHA_BWD_DSVEC", "0") == "1"
        # reduce warps: drain both dQ halves from TMEM before staging (needs 128 regs) or
        # one half at a time (64 regs, frees registers for the compute warps)
        self.reduce_drain_both = os.environ.get("FMHA_BWD_RED_BOTH", "0") == "1"
        self.mma_dkq_first = self.mma_order == "dkq_first"
        # deferred dK(i-1) before dQ(i-1): releases the Q stage 0.28 us earlier and lets the
        # reduce warps' dQ drain imply that dK has retired (FMHA_BWD_DK_FIRST=1|0)
        self.dk_first_order = os.environ.get("FMHA_BWD_DK_FIRST", "0") == "1"
        # two dO stages, dQ staged through the dS buffer, lse/delta 2 stages (FMHA_BWD_DO2=1)
        self.do2 = os.environ.get("FMHA_BWD_DO2", "0") == "1"
        self.stage_in_ds = self.do2
        # compute warps: vectorized (16 x fp32) smem loads of lse / delta per column chunk
        # instead of one scalar load per element (FMHA_BWD_LSEVEC=1|0)
        self.lse_vec = os.environ.get("FMHA_BWD_LSEVEC", "1") == "1"
        # P store hazard: "split" = per-warpgroup arrive/sync barriers (no full sync),
        # "full" = one 256-thread sync before the store
        self.p_bar_split = os.environ.get("FMHA_BWD_PBAR", "split") == "split"
        if self.c16:
            # the bf16 P^T store of warpgroup g clobbers S columns read by warpgroups g//2
            # and 2+g//2: one full 512-thread barrier for now
            self.p_bar_split = False
        # lse / delta pipeline depth (load warp -> compute warps)
        self.lse_stages = int(
            os.environ.get(
                "FMHA_BWD_LSE_STAGES", "2" if ((self.c16 and not self.c16r) or self.do2) else "3"
            )
        )
        if (self.c16 and not self.c16r) or self.stage_in_ds:
            self.dq_stage_bufs = 4
        # load warp issue order per tile: "dofirst" = Q, dO, lse, delta (the dO TMA is not
        # held back by the lse stage acquire); "orig" = Q, lse, dO, delta
        self.load_do_first = os.environ.get("FMHA_BWD_LOAD_ORDER", "orig") == "dofirst"
        # fp32 -> bf16 packing of P and dS: "rz" = one PRMT (truncation, ALU pipe) with the
        # values pre-scaled by (1 + 2^-9) through the lse term so that the truncation is
        # unbiased on average; "rn" = F2FP round-to-nearest (XU pipe, shared with MUFU)
        self.cvt_rz = os.environ.get("FMHA_BWD_CVT", "rz") == "rz"
        # dK/dV epilogue executed by the reduce warps (idle at item boundaries) instead of
        # the compute warps (FMHA_BWD_EPI=reduce|compute)
        self.epi_reduce = os.environ.get("FMHA_BWD_EPI", "reduce") == "reduce"
        if self.c16 and not self.c16r:
            self.epi_reduce = False  # no reduce warps: the compute warps run the epilogue
        self.lse_bias = math.log2(1.0 + 2.0 ** -9) if self.cvt_rz else 0.0

        self.buffer_align_bytes = 1024

    def _ik_push(self, name):
        if self.iket:
            cute.experimental.iket.range_push(name)

    def _ik_pop(self):
        if self.iket:
            cute.experimental.iket.range_pop()

    def _setup_attributes(self):
        self.load_mma_Q_stage = (
            int(os.environ.get("FMHA_BWD_Q_STAGES", "2" if self.fullw else "3")) * self.n_s_halves
        )
        self.load_mma_dO_stage = int(os.environ.get("FMHA_BWD_DO_STAGES", "2" if self.do2 else "1"))
        self.load_compute_LSE_stage = self.lse_stages
        self.load_compute_sum_OdO_stage = self.lse_stages
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_reduce_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2
        self.reduce_tma_store_stage = 4 if self.stage_in_ds else 2

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        LSE: cute.Tensor,
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        lpt_group: Int32,
        num_sms: Int32,
        workspace: cute.Tensor,
        dbg_ptr: Union[cute.Pointer, None],
        item_perm: Union[cute.Tensor, None],
        stream: cuda.CUstream,
    ):
        q_seq_max, k_seq_max, d, hb = problem_shape
        h, b = hb
        h_r, h_k = h
        # (s, d, h_r, h_k, b) -> (s, d, ((h_r, h_k), b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[0], Q.shape[1], hb),
                stride=(
                    Q.stride[0],
                    Q.stride[1],
                    (
                        (Q.shape[1], Q.shape[1] * Q.shape[2]),
                        (
                            0
                            if self.varlen
                            else cute.assume(
                                Q.shape[0] * Q.shape[1] * h_r * h_k, divby=64
                            )
                        ),
                    ),
                ),
            ),
        )
        # (s, d, 1, h_k, b) -> (s, d, ((1, h_k), b))
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[0], K.shape[1], hb),
                stride=(
                    K.stride[0],
                    K.stride[1],
                    (
                        (0, K.shape[1]),
                        (
                            0
                            if self.varlen
                            else cute.assume(
                                K.shape[0] * K.shape[1] * 1 * h_k, divby=64
                            )
                        ),
                    ),
                ),
            ),
        )
        # (s, d, 1, h_k, b) -> (s, d, ((1, h_k), b))
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[0], V.shape[1], hb),
                stride=(
                    V.stride[0],
                    V.stride[1],
                    (
                        (0, V.shape[1]),
                        (
                            0
                            if self.varlen
                            else cute.assume(
                                V.shape[0] * V.shape[1] * 1 * h_k, divby=64
                            )
                        ),
                    ),
                ),
            ),
        )
        O = cute.make_tensor(O.iterator, Q.layout)

        dQ = cute.make_tensor(dQ.iterator, Q.layout)
        dK = cute.make_tensor(dK.iterator, K.layout)
        dV = cute.make_tensor(dV.iterator, V.layout)
        dO = cute.make_tensor(dO.iterator, O.layout)

        # (s, h_r, h_k, b) -> (s, ((h_r, h_k), b))
        LSE = cute.make_tensor(
            LSE.iterator,
            cute.make_layout(
                (LSE.shape[0], hb),
                stride=(
                    LSE.stride[0],
                    (
                        (LSE.shape[0], LSE.shape[0] * LSE.shape[1]),
                        (
                            0
                            if LSE.shape[3] == 1
                            else LSE.shape[0] * LSE.shape[1] * LSE.shape[2]
                        ),
                    ),
                ),
            ),
        )

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.dQ_major_mode = utils.LayoutEnum.from_tensor(dQ).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        self.O_major_mode = utils.LayoutEnum.from_tensor(O).mma_major_mode()
        self.dO_major_mode = utils.LayoutEnum.from_tensor(dO).mma_major_mode()
        self.dQ_layout = utils.LayoutEnum.from_tensor(dQ)

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.dQ_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dq is not supported")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # compute S
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        # compute dP (A = V from smem (fullw) or TMEM; N = 128 or 64 halves)
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
            tcgen05.OperandSource.SMEM if self.fullw else tcgen05.OperandSource.TMEM,
        )
        # compute dV
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        # compute dK
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        # compute dQ (N = 64 halves)
        dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSK_mma_tiler[:2],
        )
        # full-width (N=128) MMAs used only to build the dS / K^T smem layouts
        dSK_layout_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.KT_layout_tiler[:2],
        )

        self.cluster_shape_mn = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mn),
            (KQ_tiled_mma.thr_id.shape,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            self.element_dtype,
            1,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        # fullw: V has its own smem buffer (A operand of dP). Otherwise V is staged in smem
        # (through the dS buffer) with the K layout and then copied to TMEM; its TMEM layout
        # is that of a K-major A operand (same as P^T)
        V_smem_layout_staged = K_smem_layout_staged
        V_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            self.element_dtype,
            1,
        )
        if cutlass.const_expr(self.fullw):
            V_smem_layout_staged = V_tmem_layout_staged
        # N=64 MMAs over full-width (N=128) tiles: the layouts carry an MMA_N = 2 mode
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma,
            self.dO_layout_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSK_tiled_mma,
            self.KT_layout_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        KT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSK_tiled_mma,
            self.KT_layout_tiler,
            self.element_dtype,
            1,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_B_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        P_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.element_dtype,
            self.compute_mma_P_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        LSE_smem_layout = cute.make_layout(
            (self.cta_tiler[0], self.load_compute_LSE_stage)
        )
        sum_OdO_smem_layout = cute.make_layout(
            (self.cta_tiler[0], self.load_compute_sum_OdO_stage)
        )

        dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.dq_acc_dtype,
                (self.tile_shape_Q, self.dq_stage_cols),
            ),
            self.dq_acc_dtype,
        )
        dQ_smem_layout_staged = cute.tile_to_shape(
            dQ_smem_layout_atom,
            (self.tile_shape_Q, self.dq_stage_cols, self.dq_stage_bufs),
            order=(1, 0, 2),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        tma_load_op_B = sm100_utils.cluster_shape_to_tma_atom_B(
            (self.cluster_size, 1), KQ_tiled_mma.thr_id
        )
        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op_B,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op_B,
            dO,
            dO_smem_layout,
            self.dO_layout_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(self.element_dtype, K_smem_layout)
        self.tma_copy_V_bytes = cute.size_in_bytes(self.element_dtype, V_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_Q_stage * 2
            ]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_dO_stage * 2
            ]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_LSE_stage * 2
            ]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_sum_OdO_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dP_stage * 2
            ]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_reduce_dQ_stage * 2
            ]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_P_stage * 2
            ]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_dS_stage * 2
            ]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dKdV_stage * 2
            ]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sched_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stages * 2]
            item_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            sched_ring: cute.struct.MemRange[Int32, self.sched_stages]
            # D26c: tile ring reduce warp 0 -> finisher warp (q tile, packed head/batch)
            fz_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fz_stages * 2]
            fz_ring: cute.struct.MemRange[Int32, self.fz_stages * 2]
            tmem_holding_buf: cutlass.Int32
            # dQ staging: v6 two 128 x 32 fp16 stages (16 KB); c16 one 128 x 32 chunk per
            # compute warpgroup (32 KB)
            stage_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.dq_acc_dtype,
                    8 if self.stage_in_ds else cute.cosize(dQ_smem_layout_staged),
                ],
                16 if self.stage_in_ds else self.buffer_align_bytes,
            ]
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(K_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(Q_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dO_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dS_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(V_smem_layout_staged) if self.fullw else 8,
                ],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                128,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                128,
            ]

        self.shared_storage = SharedStorage

        # Packed row space of the workspace: per-batch rows for dense, all tokens for varlen
        workspace_rows = Q.shape[0] if self.varlen else problem_shape[0]
        sum_OdO, scaled_LSE, dQ_acc, dQ_acc_words, sched_counter, tile_counters, fz_queue = self.get_workspace_tensor(
            problem_shape, workspace, self.acc_dtype, workspace_rows
        )

        dQ_smem_layout = cute.select(dQ_smem_layout_staged, mode=[0, 1])

        tma_atom_dQ_acc, tma_tensor_dQ_acc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dQ_acc,
            dQ_smem_layout,
            (self.tile_shape_Q, self.dq_stage_cols),
        )

        # =============================== Sum OdO ===============================
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        # varlen: the row-wise preprocess/convert kernels run over the packed token rows
        # (total_tokens x H) directly, so no sequence lookup and no empty blocks
        rows_shape = problem_shape
        if cutlass.const_expr(self.varlen):
            rows_shape = (
                workspace_rows,
                workspace_rows,
                problem_shape[2],
                (problem_shape[3][0], Int32(1)),
            )
        if cutlass.const_expr(self.pre_v2):
            pre_H = cute.size(rows_shape[3][0])
            if cutlass.const_expr(self.pre_map_tok):
                pre_grid = (
                    cute.ceil_div(rows_shape[0] * pre_H, self.pre_rows_per_block),
                    Int32(1),
                    cute.size(rows_shape[3][1]),
                )
            else:
                # head fastest: the 16 blocks of one 64-row range read the complementary
                # 256-byte column slots of the same 4 KB token rows together
                pre_grid = (
                    pre_H,
                    cute.ceil_div(rows_shape[0], self.pre_rows_per_block),
                    cute.size(rows_shape[3][1]),
                )
            self.sum_OdO_v2(
                O,
                dO,
                sum_OdO,
                LSE,
                scaled_LSE,
                dQ_acc_words,
                sched_counter,
                tile_counters,
                fz_queue,
                sum_OdO_scale,
                LSE_scale,
                rows_shape[0],
                pre_H,
                pre_grid[0],
            ).launch(
                grid=pre_grid,
                block=[self.pre_threads, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
            )
        else:
            sum_OdO_grid = self._compute_sum_OdO_grid(rows_shape, self.sum_OdO_block_q)

            self.sum_OdO(
                O,
                dO,
                sum_OdO,
                LSE,
                scaled_LSE,
                dQ_acc_words,
                None,
                sched_counter,
                sum_OdO_scale,
                LSE_scale,
                rows_shape,
            ).launch(
                grid=sum_OdO_grid,
                block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

        # =============================== Bwd ===============================
        # Work items = kv blocks x kv heads x batch (dense) or the compact kv-block list x kv
        # heads (varlen; at most ceil(total_tokens/block_k) + num_seqs blocks, the invalid
        # ones are skipped by the scheduler). Persistent grid: one CTA per SM.
        vgrid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])
        if cutlass.const_expr(self.varlen):
            n_blk = cute.ceil_div(workspace_rows, self.cta_tiler[1]) + cute.size(problem_shape[3][1])
            vgrid = (n_blk, cute.size(problem_shape[3][0][1]), Int32(1))
        n_items = vgrid[0] * vgrid[1] * vgrid[2]
        if cutlass.const_expr(item_perm is not None):
            # host-sorted list of valid items (heaviest first); invalid blocks excluded
            n_items = cute.size(item_perm)
        bwd_grid = (cutlass.min(n_items, num_sms), 1, 1)

        self.bwd(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            dSK_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            Q,
            dO,
            tma_atom_dQ_acc,
            tma_tensor_dQ_acc,
            dQ_acc,
            dQ,
            tile_counters,
            fz_queue,
            bwd_grid[0],
            dK,
            dV,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            window_size_left,
            window_size_right,
            lpt_group,
            sched_counter,
            vgrid,
            n_items,
            item_perm,
            dbg_ptr,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            KT_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            dQ_smem_layout_staged,
            P_tmem_layout_staged,
            V_tmem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[self.cluster_size, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        # =============================== Convert ===============================
        self.block_seq = 32
        self.convert_elem_per_load = 128 // self.dq_acc_dtype.width  # 16-byte loads
        self.num_threads_D_convert = self.cta_tiler[2] // self.convert_elem_per_load
        self.num_threads_seq = 256 // self.num_threads_D_convert
        self.iter_seq = self.block_seq // self.num_threads_seq

        max_seq_in_qk = max(rows_shape[0], rows_shape[1])
        convert_grid_z = (max_seq_in_qk + self.block_seq - 1) // self.block_seq
        convert_grid = [
            cute.size(rows_shape[3][0]),
            cute.size(rows_shape[3][1]),
            convert_grid_z,
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]

        if cutlass.const_expr(self.fuse_dq):
            pass  # dQ was written by the main kernel
        elif cutlass.const_expr(self.pre_v2):
            cvt_H = cute.size(rows_shape[3][0])
            if cutlass.const_expr(self.pre_map_tok):
                cvt_grid = (
                    cute.ceil_div(rows_shape[0] * cvt_H, self.cvt_rows_per_block),
                    Int32(1),
                    cute.size(rows_shape[3][1]),
                )
            else:
                cvt_grid = (
                    cvt_H,
                    cute.ceil_div(rows_shape[0], self.cvt_rows_per_block),
                    cute.size(rows_shape[3][1]),
                )
            self.convert_v2(dQ_acc, dQ, rows_shape[0], cvt_H, scale_softmax).launch(
                grid=cvt_grid,
                block=[self.cvt_threads, 1, 1],
                cluster=[1, 1, 1],
                smem=0,
                stream=stream,
            )
        else:
            self.convert(
                dQ_acc,
                dQ,
                rows_shape[0],
                rows_shape[2],
                None,
                scale_softmax,
            ).launch(
                grid=convert_grid,
                block=convert_block,
                cluster=[1, 1, 1],
                smem=0,
                stream=stream,
            )

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        dQ_acc: cute.Tensor,
        cumulative_s_q: Union[cute.Tensor, None],
        sched_counter: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        """Preprocess: delta = -sum(O * dO) per row, scaled lse, and zeroing of the fp32 dQ
        accumulator rows (fused here so that no separate memset launch is needed).
        Rows are addressed in the packed token space (row = idx_q + sequence offset)."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, tidz = cute.arch.thread_idx()
        if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0 and tidy == 0:
            sched_counter[0] = Int32(0)

        seqlen_q = problem_shape[0]
        offset = 0
        if cutlass.const_expr(cumulative_s_q is not None):
            offset = cumulative_s_q[bidz]
            seqlen_q = cumulative_s_q[bidz + 1] - offset

        zero_frag = cute.make_rmem_tensor((self.pre_acc_elem_per_store,), self.acc_dtype)
        zero_frag.fill(0.0)

        for idx_q_t in cutlass.range(
            tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True
        ):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                row = idx_q + offset
                O_bhq = O[row, None, (bidy, bidz)]
                O_bhq = cute.logical_divide(
                    O_bhq, cute.make_layout(self.sum_OdO_elem_per_load)
                )
                dO_bhq = dO[row, None, (bidy, bidz)]
                dO_bhq = cute.logical_divide(
                    dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load)
                )

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(
                    idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step
                ):
                    O_frag = O_bhq[None, idx_d].load()
                    dO_frag = dO_bhq[None, idx_d].load()
                    prod_frag = O_frag * dO_frag
                    prod_frag = prod_frag.to(self.acc_dtype)
                    acc += prod_frag.reduce(
                        cute.ReductionOp.ADD, 0.0, reduction_profile=0
                    )

                acc = cute.arch.warp_reduction_sum(
                    acc, threads_in_group=self.sum_OdO_num_threads_d
                )

                if tidx == 0:
                    lse_bhq = lse[row, (bidy, bidz)]
                    sum_OdO[row, (bidy, bidz)] = sum_OdO_scale * acc
                    scaled_lse[row, (bidy, bidz)] = lse_scale * lse_bhq + Float32(self.lse_bias)

                # zero the accumulator row (fp32 word view, 16-byte vector stores)
                dq_row = dQ_acc[row, None, (bidy, bidz)]
                dq_row = cute.logical_divide(
                    dq_row, cute.make_layout(self.pre_acc_elem_per_store)
                )
                for idx_d in cutlass.range(
                    idx_d_start,
                    cute.size(dq_row, mode=[1]),
                    idx_d_step,
                ):
                    dq_row[None, idx_d].store(zero_frag.load())

    @cute.kernel
    def bwd(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        Q_gmem: cute.Tensor,
        dO_gmem: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        dQ_acc: cute.Tensor,
        dQ_acc_gmem: cute.Tensor,
        dQ_out: cute.Tensor,
        tile_counters: cute.Tensor,
        fz_queue: cute.Tensor,
        n_ctas: Int32,
        dK: cute.Tensor,
        dV: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        sum_OdO: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        lpt_group: Int32,
        sched_counter: cute.Tensor,
        vgrid: Tuple[Int32, Int32, Int32],
        n_items: Int32,
        item_perm: Union[cute.Tensor, None],
        dbg_ptr: Union[cute.Pointer, None],
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        P_tmem_layout_staged: cute.ComposedLayout,
        V_tmem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx0, bidy0, bidz0 = cute.arch.block_idx()
        dbg = None
        if cutlass.const_expr(dbg_ptr is not None):
            dbg = cute.make_tensor(dbg_ptr, cute.make_layout(64))
        self.dbg_on = (bidx0 == 0) and (tidx % 32 == 0)
        cta_rank = Int32(0)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((self.cluster_size, 1, 1)), (KQ_tiled_mma.thr_id.shape,)
        )
        cluster_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        v_mbar = storage.v_mbar_ptr.data_ptr()
        k_mbar = storage.k_mbar_ptr.data_ptr()
        sched_ring = storage.sched_ring.get_tensor(cute.make_layout(self.sched_stages))
        fz_ring = storage.fz_ring.get_tensor(cute.make_layout(self.fz_stages * 2))
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(v_mbar, 1)
                cute.arch.mbarrier_init(k_mbar, 1)
                cute.arch.mbarrier_init(storage.stage_mbar_ptr.data_ptr(), 1)

        # work scheduler ring: load warp -> mma, compute and reduce warps
        sched_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.sched_mbar_ptr.data_ptr(),
            num_stages=self.sched_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (1 + self.num_compute_warps + self.num_reduce_warps) * self.threads_per_warp,
            ),
        )
        # D26c: completed q tiles: reduce warp 0 -> finisher warp
        fz_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.fz_mbar_ptr.data_ptr(),
            num_stages=self.fz_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
        )
        # item done (all MMAs of the item retired): mma warp -> load warp (K/V/dS free)
        item_done_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.item_done_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
        )

        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(
            storage.load_mma_Q_mbar_ptr.data_ptr(), cluster_layout_vmnk
        )
        load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(
            storage.load_mma_dO_mbar_ptr.data_ptr(), cluster_layout_vmnk
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_lse_mbar_ptr.data_ptr()
        )
        load_compute_sum_OdO_pipeline = (
            self.make_and_init_load_compute_sum_OdO_pipeline(
                storage.load_compute_sum_OdO_mbar_ptr.data_ptr()
            )
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr()
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr()
        )
        mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(
            storage.mma_reduce_dQ_mbar_ptr.data_ptr()
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr()
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr()
        )
        mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(
            storage.mma_compute_dKdV_mbar_ptr.data_ptr()
        )
        reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

        # Q/dO pipelines were created with defer_sync: fence + cluster-wide sync so that
        # remote mbarriers are initialized before any multicast / remote arrive
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        self.cta_sync_barrier.arrive_and_wait()
        q_mcast_mask = None
        do_mcast_mask = None
        if cutlass.const_expr(self.cluster_size > 1):
            q_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, cluster_coord, mcast_mode=1
            )
            do_mcast_mask = q_mcast_mask

        # setup mma
        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner
        )
        sdO = storage.sdO.get_tensor(
            dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner
        )
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
        sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)
        sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
        sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)
        sdS = storage.sdS.get_tensor(
            dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner
        )
        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
        sdQ = None
        if cutlass.const_expr(self.stage_in_ds):
            # dQ staging aliases the dS buffer (free once dK/dQ(i) retired, until dS(i+1))
            sdQ_ptr = cute.recast_ptr(
                sdS.iterator, dQ_smem_layout_staged.inner, dtype=self.dq_acc_dtype
            )
            sdQ = cute.make_tensor(sdQ_ptr, dQ_smem_layout_staged.outer)
        else:
            sdQ = storage.sdQ.get_tensor(
                dQ_smem_layout_staged.outer, swizzle=dQ_smem_layout_staged.inner
            )
        stage_mbar = storage.stage_mbar_ptr.data_ptr()
        # V buffer: own smem buffer (fullw) or the dS buffer viewed with the K layout
        sV = None
        if cutlass.const_expr(self.fullw):
            sV = storage.sV.get_tensor(
                V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner
            )
        else:
            sV_ptr = cute.recast_ptr(sdS.iterator, V_smem_layout_staged.inner)
            sV = cute.make_tensor(sV_ptr, V_smem_layout_staged.outer)
        tP_fake_ptr = cute.make_ptr(self.element_dtype, 0, cute.AddressSpace.tmem)
        tP = cute.make_tensor(tP_fake_ptr, P_tmem_layout_staged.outer)
        tV_fake = cute.make_tensor(tP_fake_ptr, V_tmem_layout_staged.outer)
        sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
        sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

        # (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        # (MMA, MMA_N, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # (MMA, MMA_N(=2 halves), MMA_K, STAGE)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdQrdS = dSK_tiled_mma.make_fragment_A(sdS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(
            cute.select(self.KQ_mma_tiler, mode=[0, 1])
        )
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        # (MMA, MMA_M, MMA_N)
        tSTtST = cute.make_tensor(tSTtST.iterator + self.tmem_S_offset, tSTtST.layout)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = PdO_tiled_mma.make_fragment_A(tP)
        tdVrP = tdVrP[None, None, None, 0]
        if cutlass.const_expr(self.s_halves):
            # P^T region directly after the S half region
            tdVrP_iter = cute.recast_ptr(
                tSTtST.iterator + (self.tmem_P_offset - self.tmem_S_offset),
                dtype=self.element_dtype,
            )
        else:
            tdVrP_iter = cute.recast_ptr(tSTtST.iterator, dtype=self.element_dtype)
        tdVrP = cute.make_tensor(tdVrP_iter, tdVrP.layout)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)

        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(
            cute.select(self.VdO_mma_tiler, mode=[0, 1])
        )
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        # (MMA, MMA_M, MMA_N)
        tdPTtdPT = cute.make_tensor(
            tdPTtdPT.iterator + self.tmem_dP_offset, tdPTtdPT.layout
        )

        tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(
            cute.select(self.dSK_mma_tiler, mode=[0, 1])
        )
        tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
        # (MMA, MMA_M, MMA_N)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator + self.tmem_dQ_offset, tdQtdQ.layout)

        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(
            cute.select(self.dSQ_mma_tiler, mode=[0, 1])
        )
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        # (MMA, MMA_M, MMA_N)
        tdKtdK = cute.make_tensor(tdKtdK.iterator + self.tmem_dK_offset, tdKtdK.layout)
        # V as the A operand of dP: smem fragment (fullw) or bf16 TMEM region
        tdPTrV = None
        if cutlass.const_expr(self.fullw):
            tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
            tdPTrV = tdPTrV[None, None, None, 0]
        else:
            tdPTrV = VdO_tiled_mma.make_fragment_A(tV_fake)
            tdPTrV = tdPTrV[None, None, None, 0]
            tdPTrV_iter = cute.recast_ptr(
                tdKtdK.iterator + self.tmem_V_offset, dtype=self.element_dtype
            )
            tdPTrV = cute.make_tensor(tdPTrV_iter, tdPTrV.layout)

        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(
            cute.select(self.PdO_mma_tiler, mode=[0, 1])
        )
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        # (MMA, MMA_M, MMA_N)
        tdVtdV = cute.make_tensor(tdVtdV.iterator + self.tmem_dV_offset, tdVtdV.layout)

        if cutlass.const_expr(True):
            # ---- LOAD warp (TMA producer + work scheduler)
            if warp_idx == self.load_warp_id:
                if cutlass.const_expr(self.num_regs_load > 0):
                    cute.arch.setmaxregister_decrease(self.num_regs_load)

                self.load(
                    K_in,
                    V_in,
                    Q_in,
                    dO_in,
                    LSE,
                    sum_OdO,
                    sK,
                    sQ,
                    sV,
                    v_mbar,
                    k_mbar,
                    sdO,
                    sLSE,
                    sSum_OdO,
                    KQ_tiled_mma,
                    VdO_tiled_mma,
                    tma_atom_K,
                    tma_atom_Q,
                    tma_atom_V,
                    tma_atom_dO,
                    problem_shape,
                    cumulative_s_q,
                    cumulative_s_k,
                    window_size_left,
                    window_size_right,
                    vgrid,
                    n_items,
                    item_perm,
                    lpt_group,
                    sched_counter,
                    sched_ring,
                    dbg,
                    (
                        load_mma_Q_pipeline,
                        load_compute_LSE_pipeline,
                        load_mma_dO_pipeline,
                        load_compute_sum_OdO_pipeline,
                        sched_pipeline,
                        item_done_pipeline,
                    ),
                )

            # ---- MMA warp
            elif warp_idx == self.mma_warp_id:
                if cutlass.const_expr(self.num_regs_mma > 0):
                    cute.arch.setmaxregister_decrease(self.num_regs_mma)

                self.mma(
                    KQ_tiled_mma,
                    VdO_tiled_mma,
                    PdO_tiled_mma,
                    dSK_tiled_mma,
                    dSQ_tiled_mma,
                    tSTtST,
                    tSTrQ,
                    tSTrK,
                    tdPTtdPT,
                    tdPTrV,
                    tdPTrdO,
                    tdVtdV,
                    tdVrP,
                    tdVrdOT,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    tdKrdST,
                    tdKtdK,
                    tdKrQT,
                    tmem_holding_buf,
                    k_mbar,
                    v_mbar,
                    problem_shape,
                    cumulative_s_q,
                    cumulative_s_k,
                    window_size_left,
                    window_size_right,
                    vgrid,
                    lpt_group,
                    sched_ring,
                    dbg,
                    (
                        load_mma_Q_pipeline,
                        mma_compute_S_pipeline,
                        load_mma_dO_pipeline,
                        mma_compute_dP_pipeline,
                        mma_reduce_dQ_pipeline,
                        compute_mma_P_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                        sched_pipeline,
                        item_done_pipeline,
                    ),
                )

            # ---- Compute warps (softmax, dS, epilogue)
            elif (
                warp_idx >= self.compute_warp_id[0]
                and warp_idx <= self.compute_warp_id[-1]
            ):
                if cutlass.const_expr(self.num_regs_compute > 0):
                    cute.arch.setmaxregister_increase(self.num_regs_compute)

                self.compute(
                    tSTtST,
                    tdPTtdPT,
                    tdVrP,
                    tdPTrV,
                    sV,
                    v_mbar,
                    VdO_tiled_mma,
                    sLSE,
                    sdS,
                    sSum_OdO,
                    dK,
                    dV,
                    tdKtdK,
                    tdVtdV,
                    PdO_tiled_mma,
                    dSQ_tiled_mma,
                    problem_shape,
                    cumulative_s_q,
                    cumulative_s_k,
                    scale_softmax,
                    window_size_left,
                    window_size_right,
                    vgrid,
                    lpt_group,
                    sched_ring,
                    dbg,
                    tdQtdQ,
                    tma_atom_dQ_acc,
                    dQ_acc,
                    sdQ,
                    stage_mbar,
                    (
                        mma_compute_S_pipeline,
                        compute_mma_P_pipeline,
                        load_compute_LSE_pipeline,
                        load_compute_sum_OdO_pipeline,
                        mma_compute_dP_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                        sched_pipeline,
                        mma_reduce_dQ_pipeline,
                        reduce_tma_store_pipeline,
                    ),
                )

                self.epilogue_sync_barrier.arrive_and_wait()
                self._dbg_mark_warp(dbg, 32, Int32(14))

                if cutlass.const_expr(not self.epi_reduce):
                    if warp_idx % self.num_compute_warps == 0:
                        tmem_ptr = cute.arch.retrieve_tmem_ptr(
                            Float32,
                            alignment=16,
                            ptr_to_buffer_holding_addr=tmem_holding_buf,
                        )
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
                self._dbg_mark_warp(dbg, 32, Int32(15))
            # ---- Reduce warps (dQ drain; v6 layout only) / spare warps
            else:
                if cutlass.const_expr(self.num_reduce_warps > 0):
                    if (
                        warp_idx >= self.reduce_warp_id[0]
                        and warp_idx <= self.reduce_warp_id[-1]
                    ):
                        if cutlass.const_expr(self.num_regs_reduce > 0):
                            cute.arch.setmaxregister_increase(self.num_regs_reduce)

                        self.reduce(
                            dSK_tiled_mma,
                            tdQtdQ,
                            tma_atom_dQ_acc,
                            dQ_acc,
                            dQ_acc_gmem,
                            (fz_pipeline, fz_ring, sched_counter),
                            sdQ,
                            dK,
                            dV,
                            tdKtdK,
                            tdVtdV,
                            scale_softmax,
                            tmem_holding_buf,
                            problem_shape,
                            cumulative_s_q,
                            cumulative_s_k,
                            window_size_left,
                            window_size_right,
                            vgrid,
                            lpt_group,
                            sched_ring,
                            dbg,
                            (
                                mma_reduce_dQ_pipeline,
                                reduce_tma_store_pipeline,
                                sched_pipeline,
                                mma_compute_dKdV_pipeline,
                                compute_mma_dS_pipeline,
                            ),
                            stage_mbar,
                        )

                    elif cutlass.const_expr(self.fuse_dq):
                        if warp_idx == self.empty_warp_id:
                            if cutlass.const_expr(self.num_regs_finisher > 0):
                                cute.arch.setmaxregister_decrease(self.num_regs_finisher)
                            if cutlass.const_expr("nopush" not in self.fz_dbg):
                                self.fused_dq_finisher(
                                    fz_pipeline, fz_ring, tile_counters, fz_queue, dQ_acc_gmem,
                                    dQ_out, cumulative_s_q, problem_shape, scale_softmax, dbg,
                                    sched_counter, n_ctas,
                                )
                        else:
                            if cutlass.const_expr(self.num_regs_empty > 0):
                                cute.arch.setmaxregister_decrease(self.num_regs_empty)
                    else:
                        if cutlass.const_expr(self.num_regs_empty > 0):
                            cute.arch.setmaxregister_decrease(self.num_regs_empty)
                else:
                    if cutlass.const_expr(self.num_regs_empty > 0):
                        cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.kernel
    def convert(
        self,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        count: Int32,
        d_dim: Int32,
        cumulative_s_q: Union[cute.Tensor, None],
        scale_softmax: Float32,
    ):
        """dQ = scale * dQ_acc (fp32 -> bf16); packed token rows, 16-byte loads."""
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        seqlen = count

        offset = 0
        if cutlass.const_expr(cumulative_s_q is not None):
            offset = cumulative_s_q[bidy]
            seqlen = cumulative_s_q[bidy + 1] - offset

        for idx_s_t in cutlass.range(tidy, self.block_seq, self.num_threads_seq):
            idx_s = idx_s_t + self.block_seq * bidz
            if idx_s < seqlen:
                row = idx_s + offset
                dQ_acc_bhs = dQ_acc[row, None, (bidx, bidy)]
                dQ_acc_bhs = cute.logical_divide(
                    dQ_acc_bhs, cute.make_layout(self.convert_elem_per_load)
                )
                dQ_bhs = dQ[row, None, (bidx, bidy)]
                dQ_bhs = cute.logical_divide(
                    dQ_bhs, cute.make_layout(self.convert_elem_per_load)
                )

                thr_start = tidx
                thr_step = self.num_threads_D_convert
                for idx_d in cutlass.range(
                    thr_start,
                    d_dim // self.convert_elem_per_load,
                    thr_step,
                ):
                    dQ_acc_frg = dQ_acc_bhs[None, idx_d].load().to(self.acc_dtype)
                    dQ_acc_frg = scale_softmax * dQ_acc_frg
                    dQ_bhs[None, idx_d].store(dQ_acc_frg.to(self.element_dtype))

    # ---- D26c: fused dQ convert (finisher warp) -----------------------------------------
    @cute.jit
    def _fz_convert(
        self,
        packed: Int32,
        mdQ_acc_gmem: cute.Tensor,
        dQ_out: cute.Tensor,
        cumulative_s_q: Union[cute.Tensor, None],
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        scale_softmax: Float32,
        lane: Int32,
    ):
        """One warp converts the completed fp16 dQ tile `packed` = q tile | h_r << 16 |
        h_k << 20 | batch << 24 into bf16 dQ (16-byte raw-pointer loads/stores, lane = (row
        parity, 16-byte segment), 8 loads in flight) and re-zeroes it for the next call."""
        q_iter = packed & 0xFFFF
        hr = (packed >> 16) & 0xF
        hk = (packed >> 20) & 0xF
        b = (packed >> 24) & 0x7F
        seg = lane % 16
        half = lane // 16
        row_off = Int32(0)
        seqlen = problem_shape[0]
        if cutlass.const_expr(cumulative_s_q is not None):
            row_off = cumulative_s_q[b]
            seqlen = cumulative_s_q[b + 1] - row_off
        first_row = q_iter * self.tile_shape_Q
        batch_coord = ((hr, hk), b)
        out_coord = batch_coord
        if cutlass.const_expr(self.varlen):
            out_coord = ((hr, hk), Int32(0))
        n_valid = cutlass.min(seqlen - first_row, Int32(self.tile_shape_Q))
        acc_bytes = self.dq_acc_dtype.width // 8
        out_bytes = self.element_dtype.width // 8
        a0 = cute.crd2idx((first_row + row_off, Int32(0), batch_coord), mdQ_acc_gmem.layout)
        a1 = cute.crd2idx((first_row + row_off + 1, Int32(0), batch_coord), mdQ_acc_gmem.layout)
        o0 = cute.crd2idx((first_row + row_off, Int32(0), out_coord), dQ_out.layout)
        o1 = cute.crd2idx((first_row + row_off + 1, Int32(0), out_coord), dQ_out.layout)
        acc_base = mdQ_acc_gmem.iterator.toint() + Int64(a0) * acc_bytes + Int64(seg) * 16
        acc_stride = Int64(a1 - a0) * acc_bytes
        out_base = dQ_out.iterator.toint() + Int64(o0) * out_bytes + Int64(seg) * 16
        out_stride = Int64(o1 - o0) * out_bytes
        zero = cute.make_rmem_tensor((8,), self.dq_acc_dtype)
        zero.fill(0.0)
        per_pass = 8
        for ps in cutlass.range_constexpr(self.tile_shape_Q // (2 * per_pass)):
            vals = []
            for j in cutlass.range_constexpr(per_pass):
                r = 2 * (ps * per_pass + j) + half
                r_c = cutlass.min(r, n_valid - 1)
                pa = cute.make_ptr(
                    self.dq_acc_dtype, acc_base + Int64(r_c) * acc_stride,
                    cute.AddressSpace.gmem, assumed_align=16,
                )
                vals.append(cute.make_tensor(pa, cute.make_layout(8)).load())
            for j in cutlass.range_constexpr(per_pass):
                r = 2 * (ps * per_pass + j) + half
                if r < n_valid:
                    v = vals[j].to(self.acc_dtype) * scale_softmax
                    po = cute.make_ptr(
                        self.element_dtype, out_base + Int64(r) * out_stride,
                        cute.AddressSpace.gmem, assumed_align=16,
                    )
                    if cutlass.const_expr("nost" not in self.fz_dbg):
                        cute.make_tensor(po, cute.make_layout(8)).store(v.to(self.element_dtype))
                    if cutlass.const_expr(self.acc_zero_main and "nozero" not in self.fz_dbg):
                        pa = cute.make_ptr(
                            self.dq_acc_dtype, acc_base + Int64(r) * acc_stride,
                            cute.AddressSpace.gmem, assumed_align=16,
                        )
                        cute.make_tensor(pa, cute.make_layout(8)).store(zero.load())

    @cute.jit
    def fused_dq_finisher(
        self,
        fz_pipeline,
        fz_ring: cute.Tensor,
        tile_counters: cute.Tensor,
        fz_queue: cute.Tensor,
        mdQ_acc_gmem: cute.Tensor,
        dQ_out: cute.Tensor,
        cumulative_s_q: Union[cute.Tensor, None],
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        scale_softmax: Float32,
        dbg: Union[cute.Tensor, None],
        sched_counter: cute.Tensor,
        n_ctas: Int32,
    ):
        """Finisher warp. Two duties, interleaved without blocking:
        1. local ring (from reduce warp 0): (q tile, head, batch) entries whose reduce-adds
           are complete. Drained in batches (lane k takes entry k) so that the acq_rel
           increments of the tiles' arrival counters are one round trip per batch; a lane
           whose increment completed its tile's last contribution (causal: kv blocks 0..q
           tile) pushes the tile into the global completed-tile queue (release store).
        2. global queue: reserve the next slot (pop counter), poll it with an acquire
           atomic whose result is consumed only after the ring batch, convert the tile (any
           CTA converts any tile, which balances the convert work).
        Exit: after the local sentinel, once every CTA is done (done counter == n_ctas) and
        the reserved slot is beyond the final push count."""
        lane = cute.arch.lane_idx()
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.fz_stages)
        push_ptr = sched_counter.iterator + 1
        pop_ptr = sched_counter.iterator + 2
        done_ptr = sched_counter.iterator + 3
        with_queue = "noconv" not in self.fz_dbg and "noatomic" not in self.fz_dbg
        H_r, H_k = problem_shape[3][0]
        H_all = H_r * H_k
        n_tiles_rb = cute.ceil_div(problem_shape[0], self.tile_shape_Q)
        # row-block counters live after the queue entries
        rb_base = H_all * cute.size(problem_shape[3][1]) * (n_tiles_rb + 256) + 512
        pending = Int32(-1)
        entry = Int32(0)
        d_cnt = Int32(0)
        p_cnt = Int32(0)
        n_conv = Int32(0)
        cnt = Int32(0)
        my_w0 = Int32(0)
        my_w1 = Int32(0)
        old = Int32(0)
        progressed = Boolean(False)
        local_done = Boolean(False)
        done_signalled = Boolean(False)
        all_done = Boolean(False)
        self._dbg_mark(dbg, 28, Int32(0))
        while not all_done:
            progressed = Boolean(False)
            # ---- global queue: reserve a slot once, issue the poll (result used below)
            entry = Int32(0)
            if cutlass.const_expr(with_queue):
                if pending < 0:
                    if lane == 0:
                        pending = Int32(cute.arch.atomic_add(pop_ptr, Int32(1)))
                    pending = cute.arch.shuffle_sync(pending, 0)
                if lane == 0:
                    entry = Int32(
                        cute.arch.atomic_add(fz_queue.iterator + pending, Int32(0), sem="acquire", scope="gpu")
                    )
            # ---- local ring: drain up to fz_stages entries, lane k keeps entry k
            cnt = Int32(0)
            if not local_done:
                for k in cutlass.range_constexpr(self.fz_stages):
                    if not local_done:
                        tok = fz_pipeline.consumer_try_wait(state)
                        if tok:
                            fz_pipeline.consumer_wait(state, tok)
                            w0 = fz_ring[2 * state.index]
                            w1 = fz_ring[2 * state.index + 1]
                            fz_pipeline.consumer_release(state)
                            state.advance()
                            if w0 < 0:
                                local_done = Boolean(True)
                            else:
                                if lane == cnt:
                                    my_w0 = w0
                                    my_w1 = w1
                                cnt += 1
            if cnt > 0:
                progressed = Boolean(True)
                self._dbg_mark(dbg, 28, cnt)
                if cutlass.const_expr("nofence" not in self.fz_dbg):
                    cute.arch.fence_proxy("async.global")
                    cute.arch.fence_acq_rel_gpu()
                if cutlass.const_expr("noatomic" not in self.fz_dbg):
                    if lane < cnt:
                        q_iter = my_w0
                        hr = my_w1 & 0xFF
                        hk = (my_w1 >> 8) & 0xFF
                        b = my_w1 >> 16
                        row_off = Int32(0)
                        if cutlass.const_expr(cumulative_s_q is not None):
                            row_off = cumulative_s_q[b]
                        first_row = q_iter * self.tile_shape_Q
                        cidx = cute.crd2idx((first_row + row_off, ((hr, hk), b)), tile_counters.layout)
                        old = Int32(
                            cute.arch.atomic_add(
                                tile_counters.iterator + cidx, Int32(1), sem="acq_rel", scope="gpu"
                            )
                        )
                        if cutlass.const_expr("noconv" not in self.fz_dbg):
                            if old == q_iter:
                                if cutlass.const_expr(self.fz_rowblock):
                                    rb_idx = rb_base + b * n_tiles_rb + q_iter
                                    rb_old = Int32(
                                        cute.arch.atomic_add(
                                            fz_queue.iterator + rb_idx, Int32(1), sem="acq_rel", scope="gpu"
                                        )
                                    )
                                    if rb_old == H_all - 1:
                                        base_slot = Int32(cute.arch.atomic_add(push_ptr, H_all))
                                        for hh in cutlass.range(0, H_all, 1):
                                            hr2 = hh % H_r
                                            hk2 = hh // H_r
                                            packed = (q_iter | (hr2 << 16) | (hk2 << 20) | (b << 24)) + 1
                                            cute.arch.atomic_add(
                                                fz_queue.iterator + (base_slot + hh), packed,
                                                sem="release", scope="gpu",
                                            )
                                else:
                                    slot = Int32(cute.arch.atomic_add(push_ptr, Int32(1)))
                                    packed = (q_iter | (hr << 16) | (hk << 20) | (b << 24)) + 1
                                    cute.arch.atomic_add(
                                        fz_queue.iterator + slot, packed, sem="release", scope="gpu"
                                    )
            if local_done and not done_signalled:
                # all of this CTA's pushes precede the done signal (same lanes, program
                # order + release); other finishers read done with acquire before the
                # final push count
                done_signalled = Boolean(True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.atomic_add(done_ptr, Int32(1), sem="release", scope="gpu")
            # ---- global queue: convert if the polled slot has arrived
            if cutlass.const_expr(with_queue):
                entry = cute.arch.shuffle_sync(entry, 0)
                if entry != 0:
                    t_conv0 = Int64(0)
                    if cutlass.const_expr(self.fz_stats):
                        t_conv0 = umma_raw.globaltimer()
                    self._fz_convert(
                        entry - 1, mdQ_acc_gmem, dQ_out, cumulative_s_q, problem_shape,
                        scale_softmax, lane,
                    )
                    pending = Int32(-1)
                    progressed = Boolean(True)
                    if cutlass.const_expr(self.fz_stats):
                        n_conv += 1
                        cute.arch.sync_warp()
                        if lane == 0:
                            dt = Int32(umma_raw.globaltimer() - t_conv0)
                            cute.arch.atomic_add(sched_counter.iterator + 8, Int32(1))
                            cute.arch.atomic_add(sched_counter.iterator + 9, dt)
                            cute.arch.atomic_max(sched_counter.iterator + 10, dt)
                elif local_done:
                    if lane == 0:
                        d_cnt = Int32(cute.arch.atomic_add(done_ptr, Int32(0), sem="acquire", scope="gpu"))
                        p_cnt = Int32(cute.arch.atomic_add(push_ptr, Int32(0)))
                    d_cnt = cute.arch.shuffle_sync(d_cnt, 0)
                    p_cnt = cute.arch.shuffle_sync(p_cnt, 0)
                    if d_cnt == n_ctas:
                        if pending >= p_cnt:
                            all_done = Boolean(True)
            else:
                if local_done:
                    all_done = Boolean(True)
            if not progressed:
                umma_raw.nanosleep(1000)
        if cutlass.const_expr(self.fz_stats):
            if lane == 0:
                cute.arch.atomic_max(sched_counter.iterator + 12, n_conv)

    # ---- D26: bandwidth-oriented preprocess / convert -----------------------------------
    @cute.kernel
    def sum_OdO_v2(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        dQ_acc_words: cute.Tensor,
        sched_counter: cute.Tensor,
        tile_counters: cute.Tensor,
        fz_queue: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        rows: Int32,
        H: Int32,
        grid_x: Int32,
    ):
        """Preprocess, bandwidth oriented: delta = -sum(O * dO) per row, scaled lse and the
        zeroing of the dQ accumulator rows. Four warps per block, 16 rows per warp; a warp
        covers two 256-byte rows per step (16 lanes x 16 B per row), issues all of its loads
        before the products, reduces over the 8 elements of a lane and then over the 16
        lanes of a row. Out-of-range rows (partial last block) load a clamped row and skip
        the stores."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if bidx == 0 and bidy == 0 and bidz == 0 and tidx < 16:
            sched_counter[tidx] = Int32(0)  # work counter, queue counters, statistics
        if cutlass.const_expr(self.fuse_dq):
            # zero the completed-tile queue (blocks of the first grid row/column)
            if bidy == 0 and bidz == 0:
                for i in cutlass.range(bidx * self.pre_threads + tidx, cute.size(fz_queue), grid_x * self.pre_threads):
                    fz_queue[i] = Int32(0)
        if cutlass.const_expr(not self.pre_map_tok):
            bidx, bidy = bidy, bidx  # grid is (H, row blocks, B)
        warp = tidx // 32
        lane = tidx % 32
        half = lane // 16
        seg = lane % 16
        row0 = bidx * self.pre_rows_per_block + warp * self.pre_rows_per_warp + half
        n_steps = self.pre_rows_per_warp // 2
        epl = self.sum_OdO_elem_per_load  # 8 bf16 = 16 B
        n_flat = rows
        if cutlass.const_expr(self.pre_map_tok):
            n_flat = rows * H
        o_frags = []
        do_frags = []
        rows_h = []
        o_base = O.iterator.toint()
        do_base = dO.iterator.toint()
        elem_b = self.element_dtype.width // 8
        for st in cutlass.range_constexpr(n_steps):
            f = row0 + 2 * st
            f_c = cutlass.min(f, n_flat - 1)
            if cutlass.const_expr(self.pre_map_tok):
                row_c = f_c // H
                head = f_c % H
            else:
                row_c = f_c
                head = bidy
            rows_h.append((f, row_c, head))
            if cutlass.const_expr(self.pre_raw):
                # one offset computation per row, 16-byte raw-pointer loads
                off_o = cute.crd2idx((row_c, Int32(0), (head, bidz)), O.layout)
                off_do = cute.crd2idx((row_c, Int32(0), (head, bidz)), dO.layout)
                po = cute.make_ptr(
                    self.element_dtype, o_base + Int64(off_o) * elem_b + Int64(seg) * 16,
                    cute.AddressSpace.gmem, assumed_align=16,
                )
                pdo = cute.make_ptr(
                    self.element_dtype, do_base + Int64(off_do) * elem_b + Int64(seg) * 16,
                    cute.AddressSpace.gmem, assumed_align=16,
                )
                o_frags.append(cute.make_tensor(po, cute.make_layout(epl)).load())
                do_frags.append(cute.make_tensor(pdo, cute.make_layout(epl)).load())
            else:
                O_row = cute.logical_divide(O[row_c, None, (head, bidz)], cute.make_layout(epl))
                dO_row = cute.logical_divide(dO[row_c, None, (head, bidz)], cute.make_layout(epl))
                o_frags.append(O_row[None, seg].load())
                do_frags.append(dO_row[None, seg].load())
        zero_frag = cute.make_rmem_tensor((self.pre_acc_elem_per_store,), self.acc_dtype)
        zero_frag.fill(0.0)
        words_per_row = self.cta_tiler[2] * self.dq_acc_dtype.width // 32
        n_acc_seg = words_per_row // self.pre_acc_elem_per_store  # 16-byte segments per row
        for st in cutlass.range_constexpr(n_steps):
            f, row, head = rows_h[st]
            prod = o_frags[st].to(self.acc_dtype) * do_frags[st].to(self.acc_dtype)
            acc = prod.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
            acc = cute.arch.warp_reduction_sum(acc, threads_in_group=16)
            if f < n_flat:
                if seg == 0:
                    sum_OdO[row, (head, bidz)] = sum_OdO_scale * acc
                    scaled_lse[row, (head, bidz)] = lse_scale * lse[row, (head, bidz)] + Float32(
                        self.lse_bias
                    )
                if cutlass.const_expr(self.fuse_dq):
                    if seg == 1:
                        tile_counters[row, (head, bidz)] = Int32(0)
                if cutlass.const_expr(not self.pre_nozero and not self.acc_zero_main):
                    if cutlass.const_expr(self.pre_raw):
                        off_q = cute.crd2idx((row, Int32(0), (head, bidz)), dQ_acc_words.layout)
                        for j in cutlass.range_constexpr(n_acc_seg // 16):
                            pq = cute.make_ptr(
                                self.acc_dtype,
                                dQ_acc_words.iterator.toint() + Int64(off_q) * 4 + Int64(seg + 16 * j) * 16,
                                cute.AddressSpace.gmem, assumed_align=16,
                            )
                            cute.make_tensor(pq, cute.make_layout(self.pre_acc_elem_per_store)).store(
                                zero_frag.load()
                            )
                    else:
                        dq_row = cute.logical_divide(
                            dQ_acc_words[row, None, (head, bidz)],
                            cute.make_layout(self.pre_acc_elem_per_store),
                        )
                        for j in cutlass.range_constexpr(n_acc_seg // 16):
                            dq_row[None, seg + 16 * j].store(zero_frag.load())

    @cute.kernel
    def convert_v2(
        self,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        rows: Int32,
        H: Int32,
        scale_softmax: Float32,
    ):
        """dQ = scale * dQ_acc (fp16/fp32 -> bf16), bandwidth oriented: same row mapping as
        sum_OdO_v2, all loads of a lane issued before the conversions."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(not self.pre_map_tok):
            bidx, bidy = bidy, bidx  # grid is (H, row blocks, B)
        warp = tidx // 32
        lane = tidx % 32
        half = lane // 16
        seg = lane % 16
        row0 = bidx * self.cvt_rows_per_block + warp * self.cvt_rows_per_warp + half
        n_steps = self.cvt_rows_per_warp // 2
        epl = self.convert_elem_per_load  # 16-byte loads of the accumulator
        n_seg = self.cta_tiler[2] // epl
        n_flat = rows
        if cutlass.const_expr(self.pre_map_tok):
            n_flat = rows * H
        frags = []
        rows_h = []
        for st in cutlass.range_constexpr(n_steps):
            f = row0 + 2 * st
            f_c = f
            if f_c >= n_flat:
                f_c = n_flat - 1
            if cutlass.const_expr(self.pre_map_tok):
                row_c = f_c // H
                head = f_c % H
            else:
                row_c = f_c
                head = bidy
            rows_h.append((f, row_c, head))
            src = cute.logical_divide(dQ_acc[row_c, None, (head, bidz)], cute.make_layout(epl))
            for j in cutlass.range_constexpr(n_seg // 16):
                frags.append(src[None, seg + 16 * j].load())
        for st in cutlass.range_constexpr(n_steps):
            f, row, head = rows_h[st]
            if f < n_flat:
                dst = cute.logical_divide(dQ[row, None, (head, bidz)], cute.make_layout(epl))
                for j in cutlass.range_constexpr(n_seg // 16):
                    v = frags[st * (n_seg // 16) + j].to(self.acc_dtype) * scale_softmax
                    dst[None, seg + 16 * j].store(v.to(self.element_dtype))

    # ---- D23: hand-built MMA issue ------------------------------------------------------
    IDESC_S = umma_raw.make_idesc(128, 128, False, False)   # A = K (K-major), B = Q (K-major)
    IDESC_DP = umma_raw.make_idesc(128, 128, False, False)  # A = V (K-major), B = dO (K-major)
    IDESC_DV = umma_raw.make_idesc(128, 128, False, True)   # A = P^T (TMEM), B = dO^T (MN-major)
    IDESC_DK = umma_raw.make_idesc(128, 128, False, True)   # A = dS^T (K-major), B = Q^T (MN-major)
    IDESC_DQ = umma_raw.make_idesc(128, 128, True, True)    # A = dS (MN-major), B = K^T (MN-major)

    def _raw_setup(
        self, tSTrK, tSTrQ, tdPTrV, tdPTrdO, tdVrP, tdVrdOT, tdQrdS, tdQrKT, tdKrdST, tdKrQT,
        tSTtST, tdPTtdPT, tdVtdV, tdQtdQ, tdKtdK,
    ):
        """Base descriptors (lo/hi words), static k-block / stage offsets (16 B units) and
        TMEM addresses for the hand-built MMA groups; evaluated once per MMA warp."""
        if self.s_halves or not self.fullw:
            raise ValueError("FMHA_BWD_RAWMMA supports the fullw layout without S halves only")
        nk = cute.size(tSTrQ, mode=[2])
        if cute.size(tdKrQT, mode=[2]) != nk or cute.size(tdVrP, mode=[2]) != nk:
            raise ValueError("umma_raw: unexpected k-block counts")
        rw = {}

        def smem(name, frag, coords, stage_mode):
            lo, hi = umma_raw.desc_parts(frag)
            rw[name + "_lo"], rw[name + "_hi"] = lo, hi
            rw[name + "_k"] = umma_raw.static_offsets(frag, coords)
            st = 0
            if stage_mode is not None and cute.size(frag, mode=[stage_mode]) > 1:
                c = [0] * cute.rank(frag.layout)
                c[stage_mode] = 1
                st = umma_raw.static_offsets(frag, [tuple(c)])[0]
            rw[name + "_st"] = st

        k4 = [(0, 0, k, 0) for k in range(nk)]
        k3 = [(0, 0, k) for k in range(nk)]
        smem("K", tSTrK, k4, None)
        smem("Q", tSTrQ, k4, 3)
        smem("V", tdPTrV, k3, None)
        smem("dO", tdPTrdO, k4, 3)
        smem("dOT", tdVrdOT, k4, 3)
        smem("dS", tdQrdS, k4, 3)
        smem("KT", tdQrKT, k4, None)
        smem("dST", tdKrdST, k4, 3)
        smem("QT", tdKrQT, k4, 3)
        p_offs = umma_raw.static_offsets(tdVrP, k3)
        rw["P_k"] = [o // 2 for o in p_offs]  # bf16 elements -> 32-bit TMEM columns
        if self.p_split:
            # k-block kb (16 queries, 8 bf16 columns) at column 16 * kb: the first 8 columns of
            # the 16-column S chunk that the owning warpgroup itself has consumed
            rw["P_k"] = [16 * i for i in range(nk)]
        rw["P_addr"] = Int32(tdVrP.iterator.toint())
        rw["S_addr"] = Int32(tSTtST.iterator.toint())
        rw["dP_addr"] = Int32(tdPTtdPT[None, 0, 0].iterator.toint())
        rw["dV_addr"] = Int32(tdVtdV.iterator.toint())
        rw["dQ_addr"] = Int32(tdQtdQ[None, 0, 0].iterator.toint())
        rw["dK_addr"] = Int32(tdKtdK.iterator.toint())
        return rw

    def _raw_s(self, rw, q_stage_u):
        b_lo = rw["Q_lo"] + q_stage_u * rw["Q_st"]
        with cute.arch.elect_one():
            umma_raw.mma_group(
                rw["S_addr"], rw["K_lo"], rw["K_hi"], b_lo, rw["Q_hi"], self.IDESC_S,
                rw["K_k"], rw["Q_k"], Int32(0),
            )

    def _raw_dp(self, rw, do_stage_u):
        b_lo = rw["dO_lo"] + do_stage_u * rw["dO_st"]
        with cute.arch.elect_one():
            umma_raw.mma_group(
                rw["dP_addr"], rw["V_lo"], rw["V_hi"], b_lo, rw["dO_hi"], self.IDESC_DP,
                rw["V_k"], rw["dO_k"], Int32(0),
            )

    def _raw_dv(self, rw, do_stage_u, acc):
        b_lo = rw["dOT_lo"] + do_stage_u * rw["dOT_st"]
        with cute.arch.elect_one():
            umma_raw.mma_group(
                rw["dV_addr"], rw["P_addr"], None, b_lo, rw["dOT_hi"], self.IDESC_DV,
                rw["P_k"], rw["dOT_k"], acc, a_in_tmem=True,
            )

    def _raw_dq(self, rw, ds_stage_u):
        a_lo = rw["dS_lo"] + ds_stage_u * rw["dS_st"]
        with cute.arch.elect_one():
            umma_raw.mma_group(
                rw["dQ_addr"], a_lo, rw["dS_hi"], rw["KT_lo"], rw["KT_hi"], self.IDESC_DQ,
                rw["dS_k"], rw["KT_k"], Int32(0),
            )

    def _raw_dk(self, rw, ds_stage_u, qr_stage_u, acc):
        a_lo = rw["dST_lo"] + ds_stage_u * rw["dST_st"]
        b_lo = rw["QT_lo"] + qr_stage_u * rw["QT_st"]
        with cute.arch.elect_one():
            umma_raw.mma_group(
                rw["dK_addr"], a_lo, rw["dST_hi"], b_lo, rw["QT_hi"], self.IDESC_DK,
                rw["dST_k"], rw["QT_k"], acc,
            )

    @cute.jit
    def _dbg_mark(self, dbg, slot: int, value: Int32):
        if cutlass.const_expr(dbg is not None):
            if self.dbg_on:
                dbg[slot] = Int32(value)

    @cute.jit
    def _dbg_mark_warp(self, dbg, base: int, value: Int32):
        # per-warp slot: base + warp index (CTA 0, lane 0)
        if cutlass.const_expr(dbg is not None):
            tidx, _, _ = cute.arch.thread_idx()
            if self.dbg_on:
                dbg[base + tidx // 32] = Int32(value)

    @cute.jit
    def item_to_block(
        self,
        item: Int32,
        vgrid: Tuple[Int32, Int32, Int32],
        lpt_group: Int32,
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
    ):
        """Linear work item -> (kv block, kv head, batch/sequence) and validity.
        Dense: grouped heaviest-first order (groups of lpt_group (head, batch) pairs, kv
        block slowest inside a group). Varlen: the compact kv-block list is scanned
        against cu_seqlens; blocks past a sequence's end are invalid."""
        gx, gy, gz = vgrid
        bidx = item % gx
        rest = item // gx
        bidy = rest % gy
        bidz = rest // gy
        valid = Boolean(True)
        if cutlass.const_expr(self.varlen):
            num_seqs = cute.size(cumulative_s_q) - 1
            lin_blk = bidx
            seq = Int32(0)
            base = Int32(0)
            nb = cute.ceil_div(cumulative_s_q[1] - cumulative_s_q[0], self.tile_shape_K)
            while seq < num_seqs - 1 and lin_blk >= base + nb:
                base = base + nb
                seq = seq + 1
                nb = cute.ceil_div(
                    cumulative_s_q[seq + 1] - cumulative_s_q[seq], self.tile_shape_K
                )
            bidx = lin_blk - base
            bidz = seq
            valid = bidx < nb
        elif cutlass.const_expr(self.lpt_order):
            hb_cnt = gy * gz
            grp_size = cutlass.min(lpt_group, hb_cnt)
            grp_idx = item // (gx * grp_size)
            grp_first = grp_idx * grp_size
            grp_cnt = cutlass.min(grp_size, hb_cnt - grp_first)
            within = item - grp_first * gx
            bidx = within // grp_cnt
            pair = grp_first + within % grp_cnt
            bidy = pair % gy
            bidz = pair // gy
        return bidx, bidy, bidz, valid

    @cute.jit
    def item_setup(
        self,
        bidx: Int32,
        bidy: Int32,
        bidz: Int32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
    ):
        """Per-item block coordinate, sequence offsets, problem shape and causal trip range
        (trip_count includes the h_r query heads of a GQA group)."""
        blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
        problem_shape_cur = problem_shape
        blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
        row_off = Int32(0)
        if cutlass.const_expr(self.varlen):
            Q_len = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
            K_len = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
            problem_shape_cur = (Q_len, K_len, problem_shape[2], problem_shape[3])
            blk_offset = (
                cumulative_s_q[bidz],
                cumulative_s_k[bidz],
                Int32(0),
                ((Int32(0), Int32(0)), Int32(0)),
            )
            row_off = cumulative_s_q[bidz]
        trip_start = fmha_utils.FusedMask.get_trip_start(
            self.mask_type,
            blk_coord,
            self.cta_tiler,
            problem_shape_cur[0],
            problem_shape_cur[1],
            window_size_left,
            window_size_right,
        )
        trip_count = fmha_utils.FusedMask.get_trip_count(
            self.mask_type,
            blk_coord,
            self.cta_tiler,
            problem_shape_cur[0],
            problem_shape_cur[1],
            window_size_left,
            window_size_right,
        )
        trip_end = trip_start + trip_count
        trip_count = trip_count * problem_shape_cur[3][0][0]
        return blk_coord, blk_offset, problem_shape_cur, trip_start, trip_count, trip_end, row_off

    @cute.jit
    def fetch_item(
        self,
        sched_counter: cute.Tensor,
        n_items: Int32,
        vgrid: Tuple[Int32, Int32, Int32],
        lpt_group: Int32,
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        lane: Int32,
        item_perm: Union[cute.Tensor, None] = None,
    ) -> Int32:
        """Load warp: next valid work item from the global counter (-1 when exhausted);
        with a host-provided permutation the counter indexes the sorted item list."""
        nxt = Int32(0)
        if lane == 0:
            nxt = cute.arch.atomic_add(sched_counter.iterator, Int32(1))
        nxt = cute.arch.shuffle_sync(nxt, 0)
        if cutlass.const_expr(item_perm is not None):
            if nxt < n_items:
                nxt = item_perm[nxt]
            else:
                nxt = Int32(-1)
        else:
            bx, by, bz, valid = self.item_to_block(nxt, vgrid, lpt_group, cumulative_s_q, cumulative_s_k)
            while nxt < n_items and not valid:
                if lane == 0:
                    nxt = cute.arch.atomic_add(sched_counter.iterator, Int32(1))
                nxt = cute.arch.shuffle_sync(nxt, 0)
                bx, by, bz, valid = self.item_to_block(
                    nxt, vgrid, lpt_group, cumulative_s_q, cumulative_s_k
                )
            if nxt >= n_items:
                nxt = Int32(-1)
        return nxt

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        dO_in: cute.Tensor,
        LSE_in: cute.Tensor,
        sum_OdO_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        v_mbar: cute.Pointer,
        k_mbar: cute.Pointer,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        vgrid: Tuple[Int32, Int32, Int32],
        n_items: Int32,
        item_perm: Union[cute.Tensor, None],
        lpt_group: Int32,
        sched_counter: cute.Tensor,
        sched_ring: cute.Tensor,
        dbg: Union[cute.Tensor, None],
        # (load_mma_Q, load_compute_LSE, load_mma_dO, load_compute_sum_OdO, sched, item_done)
        pipeline_args: tuple,
    ):
        """TMA producer warp and work scheduler. Fetches items from the global counter,
        publishes them through the smem ring, and per item loads K+V (after the previous
        item's MMAs retired), then Q/dO/lse/delta per q tile."""
        tidx, tidy, tidz = cute.arch.thread_idx()
        lane = tidx % self.threads_per_warp
        (
            load_mma_Q_pipeline,
            load_compute_LSE_pipeline,
            load_mma_dO_pipeline,
            load_compute_sum_OdO_pipeline,
            sched_pipeline,
            item_done_pipeline,
        ) = pipeline_args

        KQ_thr_mma = KQ_tiled_mma.get_slice(0)
        VdO_thr_mma = VdO_tiled_mma.get_slice(0)
        b_cta_layout = cute.make_layout(1)

        load_mma_Q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_Q_stage
        )
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage
        )
        load_mma_dO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_dO_stage
        )
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage
        )
        sched_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sched_stages
        )
        item_done_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        # lse / delta: 32 lanes x 4 fp32 per tile
        thread_idx = lane
        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )
        sLSE_for_copy = cute.flat_divide(sLSE, (1,))
        sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))

        item_idx = Int32(0)
        item = self.fetch_item(
            sched_counter, n_items, vgrid, lpt_group, cumulative_s_q, cumulative_s_k, lane, item_perm
        )
        sched_pipeline.producer_acquire(sched_producer_state)
        if lane == 0:
            sched_ring[sched_producer_state.index] = item
        sched_pipeline.producer_commit(sched_producer_state)
        sched_producer_state.advance()
        while item >= 0:
            self._dbg_mark(dbg, 0, item)
            self._dbg_mark(dbg, 1, item_idx)
            bidx, bidy, bidz, valid = self.item_to_block(
                item, vgrid, lpt_group, cumulative_s_q, cumulative_s_k
            )
            (
                blk_coord,
                blk_offset,
                problem_shape_cur,
                trip_start,
                trip_count,
                trip_end,
                row_off,
            ) = self.item_setup(
                bidx, bidy, bidz, problem_shape, cumulative_s_q, cumulative_s_k,
                window_size_left, window_size_right,
            )
            blk_coord_k = bidx
            blk_coord_h_k = bidy
            blk_coord_b = bidz
            blk_coord_h_r = Int32(0)
            blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
            iter_index = trip_start
            iter_start = trip_start
            iter_count = trip_count
            iter_end = trip_end
            seq_Q = problem_shape_cur[0]

            LSE = cute.domain_offset((row_off, ((Int32(0), Int32(0)), Int32(0))), LSE_in)
            sum_OdO = cute.domain_offset((row_off, ((Int32(0), Int32(0)), Int32(0))), sum_OdO_in)
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
            V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
            Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
            dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)

            # (bM, bK, RestM, RestK, (H, B))
            gK = cute.local_tile(
                K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None)
            )
            # (bN, bK, RestN, RestK, (H, B))
            gQ = cute.local_tile(
                Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None)
            )
            gV = cute.local_tile(
                V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None)
            )
            gdO = cute.local_tile(
                dO, cute.select(self.dO_layout_tiler, mode=[1, 2]), (None, None, None)
            )
            # full (128 x 128) q tiles of Q and dO for the L2 prefetch addresses
            gQ_full = cute.local_tile(Q, (self.tile_shape_Q, self.tile_shape_dQ_K), (None, None, None))
            gdO_full = cute.local_tile(dO, (self.tile_shape_Q, self.tile_shape_dQ_K), (None, None, None))
            tSTgK = KQ_thr_mma.partition_A(gK)
            tSTgQ = KQ_thr_mma.partition_B(gQ)
            tdPTgV = KQ_thr_mma.partition_A(gV)
            tdPTgdO = VdO_thr_mma.partition_B(gdO)
            tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSTgK, 0, 3),
            )
            tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tdPTgV, 0, 3),
            )
            tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Q,
                0,
                b_cta_layout,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSTgQ, 0, 3),
            )
            tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_dO,
                0,
                b_cta_layout,
                cute.group_modes(sdO, 0, 3),
                cute.group_modes(tdPTgdO, 0, 3),
            )

            tile_counter = Int32(0)
            while iter_count > 0:
                if iter_index == iter_end:
                    iter_index = iter_start
                    blk_coord_h_r += 1
                    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

                # ---- Q (one half tile per ring entry) ----
                self._dbg_mark(dbg, 2, tile_counter)
                self._dbg_mark(dbg, 3, Int32(1))
                for qh in cutlass.range_constexpr(self.n_s_halves):
                    self._ik_push("loa_w_lm_Q")
                    load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
                    self._ik_pop()
                    self._dbg_mark(dbg, 3, Int32(2))
                    tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
                        load_mma_Q_producer_state
                    )
                    cute.copy(
                        tma_atom_Q,
                        tQgQ_mkl[
                            (None, iter_index * self.n_s_halves + qh, 0, (blk_coord_h, blk_coord_b))
                        ],
                        tQsQ[None, load_mma_Q_producer_state.index],
                        tma_bar_ptr=tma_barrier,
                    )
                    if cutlass.const_expr(self.iket_tmawait):
                        self._ik_push("loa_tma_Q")
                        cute.arch.mbarrier_wait(tma_barrier, Int32(1) - load_mma_Q_producer_state.phase)
                        self._ik_pop()
                    load_mma_Q_producer_state.advance()

                if cutlass.const_expr(self.load_do_first):
                    # ---- dO ----
                    self._ik_push("loa_w_lm_dO")
                    load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
                    self._ik_pop()
                    self._dbg_mark(dbg, 3, Int32(4))
                    tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
                        load_mma_dO_producer_state
                    )
                    cute.copy(
                        tma_atom_dO,
                        tdOgdO_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                        tdOsdO[None, load_mma_dO_producer_state.index],
                        tma_bar_ptr=tma_barrier,
                    )
                    load_mma_dO_producer_state.advance()
                self._ik_push("loa_w_lc_LSE")
                load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)
                self._ik_pop()
                self._dbg_mark(dbg, 3, Int32(3))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    LSE_idx = (
                        self.tile_shape_Q * iter_index + thread_idx * async_copy_num_elts
                    )
                    if cute.elem_less(LSE_idx + i, seq_Q):
                        cute.copy(
                            atom_async_copy,
                            LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_LSE_producer_state.index,
                            ],
                        )
                    else:
                        sLSE_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ].fill(0.0)
                load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
                load_compute_LSE_producer_state.advance()

                if cutlass.const_expr(not self.load_do_first):
                    # ---- dO ----
                    self._ik_push("loa_w_lm_dO")
                    load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
                    self._ik_pop()
                    self._dbg_mark(dbg, 3, Int32(4))
                    tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
                        load_mma_dO_producer_state
                    )
                    cute.copy(
                        tma_atom_dO,
                        tdOgdO_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                        tdOsdO[None, load_mma_dO_producer_state.index],
                        tma_bar_ptr=tma_barrier,
                    )
                    load_mma_dO_producer_state.advance()

                load_compute_sum_OdO_pipeline.producer_acquire(
                    load_compute_sum_OdO_producer_state
                )
                self._dbg_mark(dbg, 3, Int32(5))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    sum_OdO_idx = (
                        self.tile_shape_Q * iter_index + thread_idx * async_copy_num_elts
                    )
                    if cute.elem_less(sum_OdO_idx + i, seq_Q):
                        cute.copy(
                            atom_async_copy,
                            sum_OdO_for_copy[
                                None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)
                            ],
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_sum_OdO_producer_state.index,
                            ],
                        )
                    else:
                        sSum_OdO_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ].fill(0.0)
                load_compute_sum_OdO_pipeline.producer_commit(
                    load_compute_sum_OdO_producer_state
                )
                load_compute_sum_OdO_producer_state.advance()

                if cutlass.const_expr(self.prefetch_dist > 0):
                    # L2 prefetch of the Q / dO tiles prefetch_dist q tiles ahead (same item):
                    # lane l touches rows l + 32 j, both 128-byte lines of each 256-byte row
                    pf_iter = iter_index + self.prefetch_dist
                    if pf_iter < iter_end:
                        gQ_pf = gQ_full[(None, None, pf_iter, 0, (blk_coord_h, blk_coord_b))]
                        gdO_pf = gdO_full[(None, None, pf_iter, 0, (blk_coord_h, blk_coord_b))]
                        for j in cutlass.range_constexpr(self.pf_rows_per_lane):
                            r = lane + 32 * j
                            for cc in cutlass.range_constexpr(2):
                                pq = cute.domain_offset((r, cc * 64), gQ_pf).iterator
                                cute.arch.prefetch(pq.llvm_ptr, cache_level="L2")
                                pd = cute.domain_offset((r, cc * 64), gdO_pf).iterator
                                cute.arch.prefetch(pd.llvm_ptr, cache_level="L2")

                if tile_counter == 0:
                    # K and V of this item: their smem (V is staged in the dS buffer) is
                    # free once the previous item's MMAs have all retired
                    if item_idx > 0:
                        self._dbg_mark(dbg, 3, Int32(6))
                        self._ik_push("loa_w_done")
                        item_done_pipeline.consumer_wait(item_done_consumer_state)
                        self._ik_pop()
                        item_done_pipeline.consumer_release(item_done_consumer_state)
                        item_done_consumer_state.advance()
                    self._dbg_mark(dbg, 3, Int32(7))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, self.tma_copy_K_bytes)
                        cute.arch.mbarrier_arrive_and_expect_tx(v_mbar, self.tma_copy_V_bytes)
                    cute.copy(
                        tma_atom_K,
                        tKgK_mkl[(None, blk_coord_k, 0, (blk_coord_h, blk_coord_b))],
                        tKsK[None, 0],
                        tma_bar_ptr=k_mbar,
                    )
                    cute.copy(
                        tma_atom_V,
                        tVgV_mkl[(None, blk_coord_k, 0, (blk_coord_h, blk_coord_b))],
                        tVsV[(None, 0)],
                        tma_bar_ptr=v_mbar,
                    )

                iter_count -= 1
                iter_index += 1
                tile_counter += 1

            item_idx += 1
            self._dbg_mark(dbg, 3, Int32(8))
            item = self.fetch_item(
                sched_counter, n_items, vgrid, lpt_group, cumulative_s_q, cumulative_s_k, lane, item_perm
            )
            self._dbg_mark(dbg, 3, Int32(9))
            sched_pipeline.producer_acquire(sched_producer_state)
            if lane == 0:
                sched_ring[sched_producer_state.index] = item
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()
            self._dbg_mark(dbg, 3, Int32(10))

        # keep this CTA alive until every consumer release has arrived
        self._dbg_mark(dbg, 3, Int32(11))
        load_mma_Q_pipeline.producer_tail(load_mma_Q_producer_state)
        self._dbg_mark(dbg, 3, Int32(12))
        load_mma_dO_pipeline.producer_tail(load_mma_dO_producer_state)
        self._dbg_mark(dbg, 3, Int32(13))

    @cute.jit
    def mma(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        tmem_holding_buf: Int32,
        k_mbar: cute.Pointer,
        v_mbar: cute.Pointer,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        vgrid: Tuple[Int32, Int32, Int32],
        lpt_group: Int32,
        sched_ring: cute.Tensor,
        dbg: Union[cute.Tensor, None],
        # (load_mma_Q, mma_compute_S, load_mma_dO, mma_compute_dP, mma_reduce_dQ, compute_mma_P, compute_mma_dS, mma_compute_dKdV, sched, item_done)
        pipeline_args: tuple,
    ):
        """MMA issue warp. Per item: wait for V in TMEM, then per tile i: S(i), [dQ(i-1),
        dK(i-1)], dP(i) (two N=64 halves), dV(i); dQ/dK of a tile are deferred to the next
        tile (or flushed at the end). The dKdV pipeline is acquired before the item's first
        dV / dK; 'item done' is committed after the item's last MMA."""
        (
            load_mma_Q_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
            sched_pipeline,
            item_done_pipeline,
        ) = pipeline_args
        # Alloc tmem buffer
        tmem_alloc_cols = cutlass.Int32(self.tmem_alloc_cols)
        cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
        self.tmem_alloc_barrier.arrive_and_wait()
        rw = None
        if cutlass.const_expr(len(self.raw_mma) > 0):
            rw = self._raw_setup(
                tSTrK, tSTrQ, tdPTrV, tdPTrdO, tdVrP, tdVrdOT, tdQrdS, tdQrKT, tdKrdST, tdKrQT,
                tSTtST, tdPTtdPT, tdVtdV, tdQtdQ, tdKtdK,
            )
        acc_dk = Int32(0)
        acc_dv = Int32(0)
        load_mma_Q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage
        )
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        mma_compute_S_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_S_stage
        )
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage
        )
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage
        )
        mma_reduce_dQ_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage
        )
        load_mma_dO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_dO_stage
        )
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage
        )
        mma_compute_dKdV_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage
        )
        mma_compute_dKdV_acquire_state = mma_compute_dKdV_producer_state.clone()
        sched_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.sched_stages
        )
        item_done_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        item_idx = Int32(0)

        self._ik_push("mma_w_sched")
        sched_pipeline.consumer_wait(sched_consumer_state)
        self._ik_pop()
        item = sched_ring[sched_consumer_state.index]
        sched_pipeline.consumer_release(sched_consumer_state)
        sched_consumer_state.advance()
        while item >= 0:
            self._dbg_mark(dbg, 8, item)
            self._dbg_mark(dbg, 9, item_idx)
            bidx, bidy, bidz, valid = self.item_to_block(
                item, vgrid, lpt_group, cumulative_s_q, cumulative_s_k
            )
            (
                blk_coord,
                blk_offset,
                problem_shape_cur,
                trip_start,
                trip_count,
                trip_end,
                row_off,
            ) = self.item_setup(
                bidx, bidy, bidz, problem_shape, cumulative_s_q, cumulative_s_k,
                window_size_left, window_size_right,
            )
            iter_count = trip_count
            iter_index = trip_start
            iter_start = trip_start
            iter_end = trip_end
            # K of this item (TMA)
            self._dbg_mark(dbg, 11, Int32(1))
            self._ik_push("mma_w_K")
            cute.arch.mbarrier_wait(k_mbar, item_idx % 2)
            self._ik_pop()
            self._dbg_mark(dbg, 11, Int32(2))
            PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            acc_dk = Int32(0)
            acc_dv = Int32(0)
            dk_first = Boolean(True)
            tile_counter = Int32(0)
            pending = Boolean(False)
            while iter_count > 0:
                if iter_index == iter_end:
                    iter_index = iter_start
                if cutlass.const_expr(self.mma_dkq_first):
                    # deferred dQ(i-1), dK(i-1)
                    if pending:
                        self._dbg_mark(dbg, 11, Int32(6))
                        self._ik_push("mma_w_cm_dS")
                        compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                        self._ik_pop()
                        ds_stage_u = cute.arch.make_warp_uniform(compute_mma_dS_consumer_state.index)
                        tdQrdS_s = tdQrdS[None, None, None, ds_stage_u]
                        tdKrdST_s = tdKrdST[None, None, None, ds_stage_u]
                        self._dbg_mark(dbg, 11, Int32(7))
                        if cutlass.const_expr(self.dk_first_order):
                            if dk_first:
                                # the compute warps have stored dK of the previous item
                                self._ik_push("mma_w_mc_dKdV")
                                mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                                self._ik_pop()
                                mma_compute_dKdV_acquire_state.advance()
                                dk_first = Boolean(False)
                            self._dbg_mark(dbg, 11, Int32(9))
                            for qh in cutlass.range_constexpr(self.n_s_halves):
                              qr_stage_u = cute.arch.make_warp_uniform(load_mma_Q_release_state.index)
                              tdKrQT_s = tdKrQT[None, None, None, qr_stage_u]
                              if cutlass.const_expr("nodk" not in self.dbg):
                               if cutlass.const_expr("dk" in self.raw_mma):
                                   self._raw_dk(rw, ds_stage_u, qr_stage_u, acc_dk)
                                   acc_dk = Int32(1)
                               else:
                                   for k_block in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                                    cute.gemm(
                                        dSQ_tiled_mma,
                                        tdKtdK,
                                        tdKrdST_s[None, None, k_block + qh * cute.size(tdKrQT, mode=[2])],
                                        tdKrQT_s[None, None, k_block],
                                        tdKtdK,
                                    )
                                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                              load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                              load_mma_Q_release_state.advance()
                        # dQ = dS * K (one N=128 MMA, or two N=64 halves drained one by one)
                        for n_half in cutlass.range_constexpr(self.n_halves):
                            if cutlass.const_expr(n_half > 0):
                                self._ik_push("mma_w_mr_dQ")
                                mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                                self._ik_pop()
                            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            if cutlass.const_expr("dq" in self.raw_mma):
                                self._raw_dq(rw, ds_stage_u)
                            else:
                                for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                                    cute.gemm(
                                        dSK_tiled_mma,
                                        tdQtdQ[None, 0, 0],
                                        tdQrdS_s[None, 0, k_block],
                                        tdQrKT[None, n_half, k_block, 0],
                                        tdQtdQ[None, 0, 0],
                                    )
                                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
                            mma_reduce_dQ_producer_state.advance()
                        self._dbg_mark(dbg, 11, Int32(8))
                        if cutlass.const_expr(not self.dk_first_order):
                            if dk_first:
                                # the compute warps have stored dK of the previous item
                                self._ik_push("mma_w_mc_dKdV")
                                mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                                self._ik_pop()
                                mma_compute_dKdV_acquire_state.advance()
                                dk_first = Boolean(False)
                            self._dbg_mark(dbg, 11, Int32(9))
                            for qh in cutlass.range_constexpr(self.n_s_halves):
                              qr_stage_u = cute.arch.make_warp_uniform(load_mma_Q_release_state.index)
                              tdKrQT_s = tdKrQT[None, None, None, qr_stage_u]
                              if cutlass.const_expr("nodk" not in self.dbg):
                               if cutlass.const_expr("dk" in self.raw_mma):
                                   self._raw_dk(rw, ds_stage_u, qr_stage_u, acc_dk)
                                   acc_dk = Int32(1)
                               else:
                                   for k_block in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                                    cute.gemm(
                                        dSQ_tiled_mma,
                                        tdKtdK,
                                        tdKrdST_s[None, None, k_block + qh * cute.size(tdKrQT, mode=[2])],
                                        tdKrQT_s[None, None, k_block],
                                        tdKtdK,
                                    )
                                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                              load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                              load_mma_Q_release_state.advance()
                        compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                        compute_mma_dS_consumer_state.advance()
                # S(i) as n_s_halves N=64 MMAs through the S region; the deferred dK/dQ(i-1)
                # are issued between the halves so that S_B(i) is ready when softmax_A ends
                self._dbg_mark(dbg, 10, tile_counter)
                for sh in cutlass.range_constexpr(self.n_s_halves):
                    self._dbg_mark(dbg, 11, Int32(3))
                    self._ik_push("mma_w_lm_Q")
                    load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                    self._ik_pop()
                    self._dbg_mark(dbg, 11, Int32(4))
                    self._ik_push("mma_w_mc_S")
                    mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
                    self._ik_pop()
                    self._dbg_mark(dbg, 11, Int32(5))
                    KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    q_stage_u = cute.arch.make_warp_uniform(load_mma_Q_consumer_state.index)
                    tSTrQ_s = tSTrQ[None, None, None, q_stage_u]
                    if cutlass.const_expr("nos" not in self.dbg):
                      if cutlass.const_expr("s" in self.raw_mma):
                          self._raw_s(rw, q_stage_u)
                      else:
                          for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                            cute.gemm(
                                KQ_tiled_mma,
                                tSTtST,
                                tSTrK[None, None, k_block, 0],
                                tSTrQ_s[None, None, k_block],
                                tSTtST,
                            )
                            KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    load_mma_Q_consumer_state.advance()
                    mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
                    mma_compute_S_producer_state.advance()
                    if cutlass.const_expr(sh == 0 and not self.mma_dkq_first):
                        # deferred dQ(i-1), dK(i-1)
                        if pending:
                            self._dbg_mark(dbg, 11, Int32(6))
                            self._ik_push("mma_w_cm_dS")
                            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                            self._ik_pop()
                            ds_stage_u = cute.arch.make_warp_uniform(compute_mma_dS_consumer_state.index)
                            tdQrdS_s = tdQrdS[None, None, None, ds_stage_u]
                            tdKrdST_s = tdKrdST[None, None, None, ds_stage_u]
                            self._dbg_mark(dbg, 11, Int32(7))
                            if cutlass.const_expr(self.dk_first_order):
                                if dk_first:
                                    # the compute warps have stored dK of the previous item
                                    self._ik_push("mma_w_mc_dKdV")
                                    mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                                    self._ik_pop()
                                    mma_compute_dKdV_acquire_state.advance()
                                    dk_first = Boolean(False)
                                self._dbg_mark(dbg, 11, Int32(9))
                                for qh in cutlass.range_constexpr(self.n_s_halves):
                                  qr_stage_u = cute.arch.make_warp_uniform(load_mma_Q_release_state.index)
                                  tdKrQT_s = tdKrQT[None, None, None, qr_stage_u]
                                  if cutlass.const_expr("nodk" not in self.dbg):
                                   if cutlass.const_expr("dk" in self.raw_mma):
                                       self._raw_dk(rw, ds_stage_u, qr_stage_u, acc_dk)
                                       acc_dk = Int32(1)
                                   else:
                                       for k_block in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                                        cute.gemm(
                                            dSQ_tiled_mma,
                                            tdKtdK,
                                            tdKrdST_s[None, None, k_block + qh * cute.size(tdKrQT, mode=[2])],
                                            tdKrQT_s[None, None, k_block],
                                            tdKtdK,
                                        )
                                        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                  load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                                  load_mma_Q_release_state.advance()
                            # dQ = dS * K (one N=128 MMA, or two N=64 halves drained one by one)
                            for n_half in cutlass.range_constexpr(self.n_halves):
                                if cutlass.const_expr(n_half > 0):
                                    self._ik_push("mma_w_mr_dQ")
                                    mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                                    self._ik_pop()
                                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                if cutlass.const_expr("dq" in self.raw_mma):
                                    self._raw_dq(rw, ds_stage_u)
                                else:
                                    for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                                        cute.gemm(
                                            dSK_tiled_mma,
                                            tdQtdQ[None, 0, 0],
                                            tdQrdS_s[None, 0, k_block],
                                            tdQrKT[None, n_half, k_block, 0],
                                            tdQtdQ[None, 0, 0],
                                        )
                                        dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
                                mma_reduce_dQ_producer_state.advance()
                            self._dbg_mark(dbg, 11, Int32(8))
                            if cutlass.const_expr(not self.dk_first_order):
                                if dk_first:
                                    # the compute warps have stored dK of the previous item
                                    self._ik_push("mma_w_mc_dKdV")
                                    mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                                    self._ik_pop()
                                    mma_compute_dKdV_acquire_state.advance()
                                    dk_first = Boolean(False)
                                self._dbg_mark(dbg, 11, Int32(9))
                                for qh in cutlass.range_constexpr(self.n_s_halves):
                                  qr_stage_u = cute.arch.make_warp_uniform(load_mma_Q_release_state.index)
                                  tdKrQT_s = tdKrQT[None, None, None, qr_stage_u]
                                  if cutlass.const_expr("nodk" not in self.dbg):
                                   if cutlass.const_expr("dk" in self.raw_mma):
                                       self._raw_dk(rw, ds_stage_u, qr_stage_u, acc_dk)
                                       acc_dk = Int32(1)
                                   else:
                                       for k_block in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                                        cute.gemm(
                                            dSQ_tiled_mma,
                                            tdKtdK,
                                            tdKrdST_s[None, None, k_block + qh * cute.size(tdKrQT, mode=[2])],
                                            tdKrQT_s[None, None, k_block],
                                            tdKtdK,
                                        )
                                        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                  load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                                  load_mma_Q_release_state.advance()
                            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                            compute_mma_dS_consumer_state.advance()
                # dP = V * dO, two N=64 halves through the shared dP/dQ region
                self._dbg_mark(dbg, 11, Int32(10))
                self._ik_push("mma_w_mr_dQ")
                mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                self._ik_pop()
                self._dbg_mark(dbg, 11, Int32(11))
                self._ik_push("mma_w_lm_dO")
                load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)
                self._ik_pop()
                self._dbg_mark(dbg, 11, Int32(12))
                if tile_counter == 0:
                    if cutlass.const_expr(self.fullw):
                        # V of this item landed in smem (TMA)
                        self._ik_push("mma_w_V")
                        cute.arch.mbarrier_wait(v_mbar, item_idx % 2)
                        self._ik_pop()
                    else:
                        # compute warps copied V of this item into TMEM
                        self._ik_push("mma_w_vtmem")
                        self.v_tmem_barrier.arrive_and_wait()
                        self._ik_pop()
                self._dbg_mark(dbg, 11, Int32(13))
                do_stage_u = cute.arch.make_warp_uniform(load_mma_dO_consumer_state.index)
                tdPTrdO_s = tdPTrdO[None, None, None, do_stage_u]
                tdVrdOT_s = tdVrdOT[None, None, None, do_stage_u]
                for n_half in cutlass.range_constexpr(self.n_halves):
                    self._ik_push("mma_w_mc_dP")
                    mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
                    self._ik_pop()
                    VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if cutlass.const_expr("nodp" not in self.dbg):
                     if cutlass.const_expr("dp" in self.raw_mma):
                         self._raw_dp(rw, do_stage_u)
                     else:
                         for k_block in cutlass.range(0, cute.size(tdPTrdO, mode=[2]), unroll_full=True):
                            cute.gemm(
                                VdO_tiled_mma,
                                tdPTtdPT[None, 0, 0],
                                tdPTrV[None, 0, k_block],
                                tdPTrdO_s[None, n_half, k_block],
                                tdPTtdPT[None, 0, 0],
                            )
                            VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
                    mma_compute_dP_producer_state.advance()
                # dV(i)
                self._dbg_mark(dbg, 11, Int32(14))
                self._ik_push("mma_w_cm_P")
                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
                self._ik_pop()
                self._dbg_mark(dbg, 11, Int32(15))
                if tile_counter == 0:
                    # the compute warps have stored dV of the previous item
                    self._ik_push("mma_w_mc_dKdV")
                    mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                    self._ik_pop()
                    mma_compute_dKdV_acquire_state.advance()
                self._dbg_mark(dbg, 11, Int32(16))
                if cutlass.const_expr("nodv" not in self.dbg):
                 if cutlass.const_expr("dv" in self.raw_mma):
                     self._raw_dv(rw, do_stage_u, acc_dv)
                     acc_dv = Int32(1)
                 else:
                     for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                        cute.gemm(
                            PdO_tiled_mma,
                            tdVtdV,
                            tdVrP[None, None, k_block],
                            tdVrdOT_s[None, None, k_block],
                            tdVtdV,
                        )
                        PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
                compute_mma_P_consumer_state.advance()
                load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
                load_mma_dO_consumer_state.advance()
                pending = Boolean(True)
                iter_count -= 1
                iter_index += 1
                tile_counter += 1

            # tail: dV ready; deferred dK(last) then dK ready; dQ(last); item done
            ds_stage_u = cute.arch.make_warp_uniform(compute_mma_dS_consumer_state.index)
            tdQrdS_s = tdQrdS[None, None, None, ds_stage_u]
            tdKrdST_s = tdKrdST[None, None, None, ds_stage_u]
            self._dbg_mark(dbg, 11, Int32(20))
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()
            if pending:
                self._ik_push("mma_w_cm_dS")
                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                self._ik_pop()
                self._dbg_mark(dbg, 11, Int32(21))
                if dk_first:
                    self._ik_push("mma_w_mc_dKdV")
                    mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_acquire_state)
                    self._ik_pop()
                    mma_compute_dKdV_acquire_state.advance()
                    dk_first = Boolean(False)
                for qh in cutlass.range_constexpr(self.n_s_halves):
                  qr_stage_u = cute.arch.make_warp_uniform(load_mma_Q_release_state.index)
                  tdKrQT_s = tdKrQT[None, None, None, qr_stage_u]
                  if cutlass.const_expr("nodk" not in self.dbg):
                   if cutlass.const_expr("dk" in self.raw_mma):
                       self._raw_dk(rw, ds_stage_u, qr_stage_u, acc_dk)
                       acc_dk = Int32(1)
                   else:
                       for k_block in cutlass.range(0, cute.size(tdKrQT, mode=[2]), unroll_full=True):
                        cute.gemm(
                            dSQ_tiled_mma,
                            tdKtdK,
                            tdKrdST_s[None, None, k_block + qh * cute.size(tdKrQT, mode=[2])],
                            tdKrQT_s[None, None, k_block],
                            tdKtdK,
                        )
                        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                  if cutlass.const_expr(qh + 1 < self.n_s_halves):
                    load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                    load_mma_Q_release_state.advance()
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()
            if pending:
                # dQ = dS * K (one N=128 MMA, or two N=64 halves drained one by one)
                for n_half in cutlass.range_constexpr(self.n_halves):
                    if cutlass.const_expr(n_half > 0):
                        self._ik_push("mma_w_mr_dQ")
                        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                        self._ik_pop()
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if cutlass.const_expr("dq" in self.raw_mma):
                        self._raw_dq(rw, ds_stage_u)
                    else:
                        for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                            cute.gemm(
                                dSK_tiled_mma,
                                tdQtdQ[None, 0, 0],
                                tdQrdS_s[None, 0, k_block],
                                tdQrKT[None, n_half, k_block, 0],
                                tdQtdQ[None, 0, 0],
                            )
                            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
                    mma_reduce_dQ_producer_state.advance()
                load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                load_mma_Q_release_state.advance()
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                compute_mma_dS_consumer_state.advance()
            # all MMAs of this item retired -> K/V smem, the V staging buffer and the V
            # TMEM region may be reused by the next item
            self._dbg_mark(dbg, 11, Int32(22))
            item_done_pipeline.producer_acquire(item_done_producer_state)
            self._dbg_mark(dbg, 11, Int32(23))
            item_done_pipeline.producer_commit(item_done_producer_state)
            item_done_producer_state.advance()
            item_idx += 1
            self._dbg_mark(dbg, 11, Int32(24))

            self._ik_push("mma_w_sched")
            sched_pipeline.consumer_wait(sched_consumer_state)
            self._ik_pop()
            item = sched_ring[sched_consumer_state.index]
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()
            self._dbg_mark(dbg, 11, Int32(25))
            self._dbg_mark(dbg, 12, item)
        self._dbg_mark(dbg, 11, Int32(26))
        if cutlass.const_expr(self.raw_dummy_gemm):
            if item_idx == -7:
                cute.gemm(
                    KQ_tiled_mma, tSTtST, tSTrK[None, None, 0, 0], tSTrQ[None, None, 0, 0], tSTtST
                )

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        tdPTrV: cute.Tensor,
        sV: cute.Tensor,
        v_mbar: cute.Pointer,
        VdO_tiled_mma: cute.TiledMma,
        sLSE: cute.Tensor,
        sdS: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        problem_shape_in: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        vgrid: Tuple[Int32, Int32, Int32],
        lpt_group: Int32,
        sched_ring: cute.Tensor,
        dbg: Union[cute.Tensor, None],
        # c16 dQ drain: dQ TMEM fragment, reduce-add TMA atom, dQ accumulator, staging smem
        tdQtdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        mdQ_acc_in: cute.Tensor,
        sdQ: cute.Tensor,
        stage_mbar: cute.Pointer,
        # (mma_compute_S, compute_mma_P, load_compute_LSE, load_compute_sum_OdO, mma_compute_dP, compute_mma_dS, mma_compute_dKdV, sched, mma_reduce_dQ, reduce_tma_store)
        pipeline_args: tuple,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
            sched_pipeline,
            mma_reduce_dQ_pipeline,
            reduce_tma_store_pipeline,
        ) = pipeline_args
        mma_reduce_dQ_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage
        )
        sched_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.sched_stages
        )

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage
        )
        compute_mma_P_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_P_stage
        )
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage
        )
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage
        )
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage
        )
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage
        )
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)),
            self.element_dtype,
        )

        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        # S coordinates of one S half (keys x s_n q columns)
        cST = cute.make_identity_tensor(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        # full (128 x 128) P^T / dP coordinates for the P store and the dS smem store mapping
        cST_full = cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_Q))
        cdPT = cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_Q))
        cdPT_h = cute.make_identity_tensor(cute.select(self.VdO_mma_tiler, mode=[0, 1]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        tTR_cST = self.split_wg(tTR_cST, num_warp_groups, wg_idx)
        # fp32 S / P of the whole tile (all halves), flat; half h is a (chunk, 1, n) view
        s_half_elems = cute.size(tTR_cST)
        tTR_rST = cute.make_rmem_tensor((s_half_elems * self.n_s_halves,), self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        tTR_tST = self.split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = self.split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)
        # view of the same registers as (chunk, half): half h holds chunks 2h, 2h+1
        n_chunks = cute.size(tTR_cdPT, mode=[2])
        tTR_rdPT_hv = cute.make_tensor(
            tTR_rdPT.iterator,
            cute.make_layout(
                (tTR_cdPT.shape[0], tTR_cdPT.shape[1], (n_chunks // self.n_halves, self.n_halves)),
                stride=((1, 0), 0, (cute.size(tTR_cdPT, mode=[0]), cute.size(tTR_cdPT, mode=[0]) * (n_chunks // self.n_halves))),
            ),
        )
        half_elems = cute.size(tTR_cdPT) // self.n_halves

        # dP halves live in a (128 x 64) TMEM region
        tiled_t2r_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPTtdPT)
        thr_t2r_dP = tiled_t2r_dP.get_slice(dp_idx)
        tTR_cdPT_h = self.split_wg(thr_t2r_dP.partition_D(cdPT_h), num_warp_groups, wg_idx)
        tTR_tdPT = self.split_wg(thr_t2r_dP.partition_S(tdPTtdPT), num_warp_groups, wg_idx)

        tdVcST = PdO_tiled_mma.get_slice(0).partition_A(cST_full)

        # ---- c16: dQ drain partition (this thread: lane dp_idx, chunks wg_idx + 4j of 16 d
        # columns), per-warpgroup staging barrier and fp16 fragment
        if cutlass.const_expr(self.c16):
            # warpgroup g drains dQ columns 32g..32g+31 (32x32b loads, 32 columns per thread)
            tmem_load_atom32 = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            cdQ = cute.make_identity_tensor((self.dSK_mma_tiler[0], self.dSK_mma_tiler[1]))
            tdQtdQ_v = tdQtdQ[(None, None), 0, 0]
            tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom32, tdQtdQ_v)
            thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)
            tTR_cdQ = self.split_wg(thr_t2r_dQ.partition_D(cdQ), num_warp_groups, wg_idx)
            tTR_tdQ = self.split_wg(thr_t2r_dQ.partition_S(tdQtdQ_v), num_warp_groups, wg_idx)
            rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
            dq_chunk = cute.size(tTR_cdQ, mode=[0])
            rdQ16 = cute.make_rmem_tensor((dq_chunk,), self.dq_acc_dtype)
            dq_bar_id = Int32(self.dq_stage_barrier_base) + wg_idx
            is_wg_leader = warp_idx % 4 == 0

        tiled_r2t = tcgen05.make_tmem_copy(tmem_store_atom, tdVrP)
        thr_r2t = tiled_r2t.get_slice(dp_idx)

        tRT_tP = thr_r2t.partition_D(tdVrP)
        tRT_tP = self.split_wg(tRT_tP, num_warp_groups, wg_idx)
        if cutlass.const_expr(self.p_split):
            # the warpgroups' 8-column chunks interleave at a 16-column pitch (stride 32 bf16);
            # spread them to a 32-column pitch, warpgroup 1 offset by 16 columns (+16 bf16 on
            # top of its +8-column partition offset), so each chunk lands inside the S chunk
            # that its own warpgroup reads
            st = tRT_tP.stride
            new_stride = (*st[:-1], st[-1] * 2)
            tRT_tP = cute.make_tensor(
                tRT_tP.iterator + wg_idx * 16, cute.make_layout(tRT_tP.shape, stride=new_stride)
            )
            print("p_split tRT_tP", tRT_tP)

        tRT_cST = thr_r2t.partition_S(tdVcST)
        tRT_cST = self.split_wg(tRT_cST, num_warp_groups, wg_idx)
        # per S half: the P^T store chunks of that half (chunk j = jj + n_pch_h * h in the
        # last mode of the partitions)
        tRT_tP_hv = self.split_last(tRT_tP, self.n_s_halves)
        tRT_cST_hv = self.split_last(tRT_cST, self.n_s_halves)

        # ---- V: smem staging buffer -> registers -> TMEM (A operand of dP), once per item
        # (halves layout only; in fullw V is a smem operand)
        if cutlass.const_expr(not self.fullw):
            tiled_r2t_V = tcgen05.make_tmem_copy(tmem_store_atom, tdPTrV)
            thr_r2t_V = tiled_r2t_V.get_slice(dp_idx)
            tRT_tV = self.split_wg(thr_r2t_V.partition_D(tdPTrV), num_warp_groups, wg_idx)
            cV = cute.make_identity_tensor(cute.select(self.VdO_mma_tiler, mode=[0, 2]))
            tdPcV = VdO_tiled_mma.get_slice(0).partition_A(cV)
            tRT_cV_p = thr_r2t_V.partition_S(tdPcV)
            tRT_cV = self.split_wg(tRT_cV_p, num_warp_groups, wg_idx)
            rV = cute.make_rmem_tensor(tRT_cV.shape, self.element_dtype)
            sV_slice = sV[None, None, None, 0]
            # K-major (d contiguous) layout: mode 0 (key) is the slow mode of the flat order
            sV_tmp = cute.composition(sV_slice, cute.make_ordered_layout((128, 128), (0, 1)))
            sV_p = cute.composition(sV_tmp[dp_idx, None], cute.make_layout(tRT_cV_p.shape))
            sV_rows = self.split_wg(sV_p, num_warp_groups, wg_idx)

        item_idx = Int32(0)
        g_tile = Int32(0)  # tiles processed by this CTA (all items)
        self._ik_push("com_w_sched")
        sched_pipeline.consumer_wait(sched_consumer_state)
        self._ik_pop()
        item = sched_ring[sched_consumer_state.index]
        sched_pipeline.consumer_release(sched_consumer_state)
        sched_consumer_state.advance()
        while item >= 0:
          self._dbg_mark(dbg, 16, item)
          self._dbg_mark(dbg, 17, item_idx)
          bidx, bidy, bidz, valid = self.item_to_block(
              item, vgrid, lpt_group, cumulative_s_q, cumulative_s_k
          )
          (
              blk_coord,
              blk_offset,
              problem_shape,
              trip_start,
              trip_count,
              trip_end,
              row_off,
          ) = self.item_setup(
              bidx, bidy, bidz, problem_shape_in, cumulative_s_q, cumulative_s_k,
              window_size_left, window_size_right,
          )
          Q, K, D, HB = problem_shape
          blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
          iter_count = trip_count
          iter_index = trip_start
          iter_start = trip_start
          iter_end = trip_end
          if cutlass.const_expr(self.c16):
              # dQ accumulator tiles of this item (per q tile, 32-column chunk, (head, batch))
              dq_h_r = Int32(0)
              prev_iter = Int32(0)
              prev_h_r = Int32(0)
              tile_ctr = Int32(0)
              dq_h_k = blk_coord_batch[0][1]
              dq_b = blk_coord_batch[1]
              mdQ_acc = cute.domain_offset(
                  (row_off, Int32(0), ((Int32(0), Int32(0)), Int32(0))), mdQ_acc_in
              )
              gdQ = cute.local_tile(
                  mdQ_acc, (self.tile_shape_Q, self.dq_stage_cols), (None, None, None)
              )
              tdQsdQ, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_dQ_acc,
                  0,
                  cute.make_layout(1),
                  cute.group_modes(sdQ, 0, 2),
                  cute.group_modes(gdQ, 0, 2),
              )
          # ---- V of this item: staging buffer -> registers -> TMEM (not in fullw)
          self._dbg_mark_warp(dbg, 32, Int32(1))
          if cutlass.const_expr(not self.fullw):
              self._ik_push("com_w_V")
              cute.arch.mbarrier_wait(v_mbar, item_idx % 2)
              self._ik_pop()
              self._dbg_mark_warp(dbg, 32, Int32(2))
              cute.autovec_copy(sV_rows, rV)
              cute.copy(tiled_r2t_V, rV, tRT_tV)
              cute.arch.fence_view_async_tmem_store()
              self.v_tmem_barrier.arrive_and_wait()
          self._dbg_mark_warp(dbg, 32, Int32(3))
          if cutlass.const_expr(True):
            masked_leading_count = fmha_utils.FusedMask.get_masked_leading_count(
                self.mask_type,
                blk_coord,
                self.cta_tiler,
                Q,
                K,
                window_size_left,
                window_size_right,
            )
            unmasked_count = fmha_utils.FusedMask.get_unmasked_trip_count(
                self.mask_type,
                blk_coord,
                self.cta_tiler,
                Q,
                K,
                window_size_left,
                window_size_right,
            )
            masked_trailing_count = fmha_utils.FusedMask.get_masked_trailing_count(
                self.mask_type,
                blk_coord,
                self.cta_tiler,
                Q,
                K,
                window_size_left,
                window_size_right,
            )

            while iter_count > 0:
                self._dbg_mark(dbg, 18, iter_count)
                self._dbg_mark_warp(dbg, 32, Int32(4))
                # ---- softmax over the S halves: wait S_h, load, release S (the MMA warp may
                # issue the next half), mask, exp2, pack, store P^T_h into the P region
                self._dbg_mark_warp(dbg, 32, Int32(5))
                self._ik_push("com_w_cm_P")
                compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)
                self._ik_pop()
                self._dbg_mark_warp(dbg, 32, Int32(6))
                # Wait for LSE
                self._ik_push("com_w_lc_LSE")
                load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
                self._ik_pop()

                iter_num = iter_index - iter_start + 1
                is_residual_k = Boolean(False)
                is_residual_k = blk_coord_k * self.tile_shape_K + self.tile_shape_K >= K
                is_masked_tile = (
                    is_residual_k
                    or iter_num <= masked_leading_count
                    or (
                        iter_num > masked_leading_count + unmasked_count
                        and iter_num
                        <= masked_leading_count + unmasked_count + masked_trailing_count
                    )
                )
                log2_e = Float32(math.log2(math.e))
                softmax_scale_log2_e = scale_softmax * log2_e
                sLSE_stage = sLSE[None, load_compute_LSE_consumer_state.index]
                self._ik_push("com_P")
                for sh in cutlass.range_constexpr(self.n_s_halves):
                    q_off = sh * self.s_n
                    tTR_rST_h = cute.make_tensor(
                        tTR_rST.iterator + sh * s_half_elems, cute.make_layout(tTR_cST.shape)
                    )
                    self._ik_push("com_w_mc_S")
                    mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
                    self._ik_pop()
                    if cutlass.const_expr("nocomp" not in self.dbg):
                        self._ik_push("cP_ld")
                        cute.copy(tiled_t2r, tTR_tST, tTR_rST_h)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(not self.s_halves and self.p_bar_split and not self.p_split):
                            # this warpgroup's S reads are done: let the other group store P
                            if wg_idx == 0:
                                self.p_load_barrier[0].arrive()
                            else:
                                self.p_load_barrier[1].arrive()
                        self._ik_pop()
                    # Release S (P^T does not alias S in the s_halves layout; otherwise the
                    # region is released after the P store below)
                    if cutlass.const_expr(self.s_halves):
                        self._ik_push("cP_rel")
                        if cutlass.const_expr(self.warp_commit):
                            cute.arch.sync_warp()
                            if lane_idx == 0:
                                mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
                        else:
                            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
                        mma_compute_S_consumer_state.advance()
                        self._ik_pop()
                    if cutlass.const_expr("nocomp" not in self.dbg):
                        self._ik_push("cP_mask")
                        if is_masked_tile:
                            if cutlass.const_expr(self.fast_mask):
                                q_base = iter_index * self.tile_shape_Q + q_off
                                k_row = blk_coord_k * self.tile_shape_K + cute.get(
                                    tTR_cST[0], mode=[0]
                                )
                                row_oob = k_row >= K
                                for i in cutlass.range_constexpr(cute.size(tTR_rST_h)):
                                    q_idx = q_base + cute.get(tTR_cST[i], mode=[1])
                                    if q_idx < k_row or q_idx >= Q or row_oob:
                                        tTR_rST_h[i] = -Float32.inf
                            else:
                                fmha_utils.FusedMask.apply_mask(
                                    self.mask_type,
                                    tTR_rST_h,
                                    tTR_cST,
                                    Q,
                                    K,
                                    window_size_left,
                                    window_size_right,
                                    lambda index_k, index_q: (
                                        index_q + iter_index * self.tile_shape_Q + q_off,
                                        index_k + blk_coord_k * self.tile_shape_K,
                                    ),
                                )
                        self._ik_pop()
                        self._ik_push("cP_math")
                        # this thread holds n_chunks_s runs of `chunk` consecutive q columns
                        chunk = cute.size(tTR_cST, mode=[0])
                        n_chunks_s = cute.size(tTR_cST) // chunk
                        lse_frag = cute.make_rmem_tensor((chunk,), self.acc_dtype)
                        for j in cutlass.range_constexpr(n_chunks_s):
                            if cutlass.const_expr("nolse" in self.dbg):
                                lse_frag.fill(0.0)
                            else:
                                c0 = self._col_base(tTR_cST, j * chunk, q_off)
                                cute.autovec_copy(
                                    cute.make_tensor(
                                        sLSE_stage.iterator + c0, cute.make_layout(chunk)
                                    ),
                                    lse_frag,
                                )
                            for i2 in cutlass.range_constexpr(0, chunk, 2):
                                i = j * chunk + i2
                                lse = (lse_frag[i2], lse_frag[i2 + 1])
                                tTR_rST_h[i], tTR_rST_h[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rST_h[i], tTR_rST_h[i + 1]),
                                    (softmax_scale_log2_e, softmax_scale_log2_e),
                                    lse,
                                )
                                if cutlass.const_expr("noexp" in self.dbg):
                                    pass
                                elif (i // 2) % self.ex2_emu_den < self.ex2_emu_num:
                                    tTR_rST_h[i], tTR_rST_h[i + 1] = fmha_utils.ex2_emulation_packed_f32x2(
                                        tTR_rST_h[i], tTR_rST_h[i + 1]
                                    )
                                else:
                                    tTR_rST_h[i] = cute.math.exp2(tTR_rST_h[i], fastmath=True)
                                    tTR_rST_h[i + 1] = cute.math.exp2(tTR_rST_h[i + 1], fastmath=True)
                        # fp32 P -> bf16 P^T of this half, stored into the P region
                        tRT_rST_h = self.quantize_pairs(tTR_rST_h)
                        tRT_cST_h = self.pick_last(tRT_cST_hv, sh)
                        tRT_rST_reshaped = cute.make_tensor(
                            tRT_rST_h.iterator, cute.make_layout(tRT_cST_h.shape)
                        )
                        self._ik_pop()
                        if cutlass.const_expr(not self.s_halves and not self.p_split):
                            # P^T overwrites S: all S reads of this tile must be done
                            self._ik_push("cP_bar")
                            if cutlass.const_expr(self.p_bar_split):
                                if wg_idx == 0:
                                    self.p_load_barrier[1].arrive_and_wait()
                                else:
                                    self.p_load_barrier[0].arrive_and_wait()
                            else:
                                cute.arch.fence_view_async_tmem_load()
                                self.compute_sync_barrier.arrive_and_wait()
                            self._ik_pop()
                        self._ik_push("cP_st")
                        cute.copy(tiled_r2t, tRT_rST_reshaped, self.pick_last(tRT_tP_hv, sh))
                        cute.arch.fence_view_async_tmem_store()
                        self._ik_pop()

                if cutlass.const_expr(self.c16 and not self.c16r):
                    if tile_ctr > 0:
                        (
                            mma_reduce_dQ_consumer_state,
                            compute_mma_dS_producer_state,
                        ) = self.drain_dQ(
                            tiled_t2r_dQ, tTR_tdQ, rdQ, rdQ16, sdQ, dq_bar_id, is_wg_leader,
                            wg_idx, dp_idx, tma_atom_dQ_acc, tdQsdQ, tdQgdQ, prev_iter,
                            ((prev_h_r, dq_h_k), dq_b),
                            (mma_reduce_dQ_pipeline, compute_mma_dS_pipeline, reduce_tma_store_pipeline),
                            (mma_reduce_dQ_consumer_state, compute_mma_dS_producer_state),
                        )
                # Notify for P
                self._ik_push("cP_cmt")
                if cutlass.const_expr(self.warp_commit):
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
                else:
                    compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
                self._ik_pop()
                self._ik_pop()
                compute_mma_P_producer_state.advance()

                if cutlass.const_expr(not self.s_halves):
                    # Release S (after the P store)
                    if cutlass.const_expr(self.warp_commit):
                        cute.arch.sync_warp()
                        if lane_idx == 0:
                            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
                    else:
                        mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
                    mma_compute_S_consumer_state.advance()

                # Release LSE
                if cutlass.const_expr(self.warp_commit):
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
                else:
                    load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
                load_compute_LSE_consumer_state.advance()

                # Wait for OdO
                load_compute_sum_OdO_pipeline.consumer_wait(
                    load_compute_sum_OdO_consumer_state
                )
                # Wait for dS
                self._ik_push("com_w_cm_dS")
                compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)
                self._ik_pop()

                # Compute dS = dsoftmax(P, dP, sum_OdO); dP arrives as two N=64 halves
                self._ik_push("com_dS")
                for h in cutlass.range_constexpr(self.n_halves):
                    self._ik_push("com_w_mc_dP")
                    mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)
                    self._ik_pop()
                    if cutlass.const_expr("nocomp" not in self.dbg):
                        cute.copy(tiled_t2r_dP, tTR_tdPT, tTR_rdPT_hv[None, None, (None, h)])
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(self.warp_commit):
                        cute.arch.sync_warp()
                        if lane_idx == 0:
                            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
                    else:
                        mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
                    mma_compute_dP_consumer_state.advance()
                    if cutlass.const_expr("nocomp" not in self.dbg):
                        col_off = h * self.half_n
                        chunk_h = cute.size(tTR_cdPT_h, mode=[0])
                        n_chunks_h = half_elems // chunk_h
                        sOdO_stage = sSum_OdO[None, load_compute_sum_OdO_consumer_state.index]
                        odo_frag = cute.make_rmem_tensor((chunk_h,), self.acc_dtype)
                        for j in cutlass.range_constexpr(n_chunks_h):
                            if cutlass.const_expr(self.lse_vec):
                                c0 = self._col_base(tTR_cdPT_h, j * chunk_h, col_off)
                                cute.autovec_copy(
                                    cute.make_tensor(
                                        sOdO_stage.iterator + c0, cute.make_layout(chunk_h)
                                    ),
                                    odo_frag,
                                )
                            for i2 in cutlass.range_constexpr(0, chunk_h, 2):
                                i = j * chunk_h + i2
                                e = h * half_elems + i
                                if cutlass.const_expr(self.lse_vec):
                                    odo = (odo_frag[i2], odo_frag[i2 + 1])
                                else:
                                    odo = (
                                        sSum_OdO[
                                            cute.get(tTR_cdPT_h[i], mode=[1]) + col_off,
                                            load_compute_sum_OdO_consumer_state.index,
                                        ],
                                        sSum_OdO[
                                            cute.get(tTR_cdPT_h[i + 1], mode=[1]) + col_off,
                                            load_compute_sum_OdO_consumer_state.index,
                                        ],
                                    )
                                tTR_rdPT[e], tTR_rdPT[e + 1] = cute.arch.add_packed_f32x2(
                                    (tTR_rdPT[e], tTR_rdPT[e + 1]), odo
                                )
                                tTR_rdPT[e], tTR_rdPT[e + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rdPT[e], tTR_rdPT[e + 1]), (tTR_rST[e], tTR_rST[e + 1])
                                )
                if cutlass.const_expr("nocomp" not in self.dbg):
                    # convert fp32 dS to bf16 dS which will be used in the computation of dK and dQ
                    tTR_rdST = self.quantize_pairs(tTR_rdPT)
                    self._ik_push("cS_st")

                if cutlass.const_expr(self.stage_in_ds):
                    # the reduce warps' staging reads of the previous tile's dQ are done
                    if g_tile > 0:
                        self._ik_push("com_w_stg")
                        cute.arch.mbarrier_wait(stage_mbar, (g_tile - 1) % 2)
                        self._ik_pop()
                sdS_slice = sdS[None, None, None, compute_mma_dS_producer_state.index]

                thread_layout = cute.make_ordered_layout((128, 128), (1, 0))
                sdS_slice_tmp = cute.composition(sdS_slice, thread_layout)
                if cutlass.const_expr(self.ds_store_vec):
                    # This thread's key row. The T2R partition gives each thread chunks of
                    # `chunk` consecutive q columns; the two warpgroups take alternating chunks
                    # (wg0: chunks 0,2,4,..; wg1: 1,3,5,..). Store each chunk through a flat
                    # swizzled row view so that the copy can use 16-byte stores.
                    if cutlass.const_expr(os.environ.get("FMHA_BWD_PRINT", "0") == "1"):
                        print("tTR_cdPT layout:", tTR_cdPT.layout)
                        print("tTR_rdST layout:", tTR_rdST.layout)
                        print("tTR_cdPT_p layout:", tTR_cdPT_p.layout)
                    sdS_row = sdS_slice_tmp[dp_idx, None]
                    chunk = cute.size(tTR_cdPT, mode=[0])
                    n_chunks = cute.size(tTR_cdPT) // chunk
                    if cutlass.const_expr("nodsst" not in self.dbg and "nocomp" not in self.dbg):
                        for j in cutlass.range_constexpr(n_chunks):
                            col0 = (wg_idx + num_warp_groups * j) * chunk
                            sdS_chunk = cute.composition(
                                cute.domain_offset((col0,), sdS_row), cute.make_layout(chunk)
                            )
                            rdS_chunk = cute.make_tensor(
                                tTR_rdST.iterator + j * chunk, cute.make_layout(chunk)
                            )
                            cute.autovec_copy(rdS_chunk, sdS_chunk)
                else:
                    sdS_slice_p = cute.composition(
                        sdS_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdPT_p.shape)
                    )
                    sdS_slice = self.split_wg(sdS_slice_p, num_warp_groups, wg_idx)
                    if cutlass.const_expr("nodsst" not in self.dbg and "nocomp" not in self.dbg):
                        cute.autovec_copy(tTR_rdST, sdS_slice)

                # Notify for dS
                self._ik_pop()
                self._ik_pop()
                self._ik_push("cS_cmt")
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                if cutlass.const_expr(self.warp_commit):
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
                else:
                    compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
                compute_mma_dS_producer_state.advance()
                self._ik_pop()

                # Release OdO
                if cutlass.const_expr(self.warp_commit):
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)
                else:
                    load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)
                load_compute_sum_OdO_consumer_state.advance()

                if cutlass.const_expr(self.c16):
                    prev_iter = iter_index
                    prev_h_r = dq_h_r
                    tile_ctr += 1

                iter_count -= 1
                iter_index += 1
                g_tile += 1
                if iter_index == iter_end:
                    iter_index = iter_start
                    if cutlass.const_expr(self.c16):
                        dq_h_r += 1

          if cutlass.const_expr(self.c16 and not self.c16r):
              if tile_ctr > 0:
                  # dQ of the item's last tile (issued after dK in the MMA tail)
                  (
                      mma_reduce_dQ_consumer_state,
                      compute_mma_dS_producer_state,
                  ) = self.drain_dQ(
                      tiled_t2r_dQ, tTR_tdQ, rdQ, rdQ16, sdQ, dq_bar_id, is_wg_leader,
                      wg_idx, dp_idx, tma_atom_dQ_acc, tdQsdQ, tdQgdQ, prev_iter,
                      ((prev_h_r, dq_h_k), dq_b),
                      (mma_reduce_dQ_pipeline, compute_mma_dS_pipeline, reduce_tma_store_pipeline),
                      (mma_reduce_dQ_consumer_state, compute_mma_dS_producer_state),
                  )
          # Epilogue of this item (dK, dV from TMEM to global), unless the reduce warps do it
          self._dbg_mark_warp(dbg, 32, Int32(10))
          if cutlass.const_expr(not self.epi_reduce):
              # (the pipeline state is advanced inside and must be carried across items)
              mma_compute_dKdV_consumer_state = self.epilogue(
                  blk_coord,
                  blk_offset,
                  problem_shape,
                  dK,
                  dV,
                  tdKtdK,
                  tdVtdV,
                  scale_softmax,
                  (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
              )
          item_idx += 1
          self._dbg_mark_warp(dbg, 32, Int32(11))
          self._ik_push("com_w_sched")
          sched_pipeline.consumer_wait(sched_consumer_state)
          self._ik_pop()
          item = sched_ring[sched_consumer_state.index]
          sched_pipeline.consumer_release(sched_consumer_state)
          sched_consumer_state.advance()
          self._dbg_mark_warp(dbg, 32, Int32(12))
          self._dbg_mark(dbg, 20, item)
        self._dbg_mark_warp(dbg, 32, Int32(13))
        if cutlass.const_expr(self.c16 and not self.c16r):
            if is_wg_leader:
                reduce_tma_store_pipeline.producer_tail()

    @cute.jit
    def drain_dQ(
        self,
        tiled_t2r_dQ: cute.TiledCopy,
        tTR_tdQ: cute.Tensor,
        rdQ: cute.Tensor,
        rdQ16: cute.Tensor,
        sdQ: cute.Tensor,
        dq_bar_id: Int32,
        is_wg_leader: Boolean,
        wg_idx: Int32,
        dp_idx: Int32,
        tma_atom_dQ_acc: cute.CopyAtom,
        tdQsdQ: cute.Tensor,
        tdQgdQ: cute.Tensor,
        q_tile: Int32,
        hb_coord: cute.Coord,
        pipes: tuple,
        states: tuple,
    ) -> tuple:
        """c16: one tile's dQ (this warpgroup: 32 d-columns) TMEM -> fp16 registers ->
        this warpgroup's staging chunk -> TMA reduce-add. Returns the advanced
        (mma_reduce_dQ consumer, compute_mma_dS producer [unchanged]) states."""
        mma_reduce_dQ_pipeline, compute_mma_dS_pipeline, reduce_tma_store_pipeline = pipes
        dq_state, ds_state = states
        dq_bar = pipeline.NamedBarrier(barrier_id=dq_bar_id, num_threads=128)
        self._ik_push("com_w_mr_dQ")
        mma_reduce_dQ_pipeline.consumer_wait(dq_state)
        self._ik_pop()
        self._ik_push("com_dQ")
        self._ik_push("cQ_ld")
        cute.copy(tiled_t2r_dQ, tTR_tdQ, rdQ)
        cute.arch.fence_view_async_tmem_load()
        self._ik_pop()
        self._ik_push("cQ_rel")
        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(self.warp_commit):
            cute.arch.sync_warp()
            if lane_idx == 0:
                mma_reduce_dQ_pipeline.consumer_release(dq_state)
        else:
            mma_reduce_dQ_pipeline.consumer_release(dq_state)
        dq_state.advance()
        self._ik_pop()
        self._ik_push("cQ_cvt")
        rdQ16.store(rdQ[None, None, 0].load().to(self.dq_acc_dtype))
        self._ik_pop()
        if is_wg_leader:
            # the previous tile's reduce-add has read this warpgroup's staging chunk
            self._ik_push("cQ_acq")
            reduce_tma_store_pipeline.producer_acquire()
            self._ik_pop()
        self._ik_push("cQ_st")
        st_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dq_acc_dtype, num_bits_per_copy=128
        )
        st_op = cute.make_cotiled_copy(st_atom, cute.make_layout((1, 8)), rdQ16.layout)
        st_thr = st_op.get_slice(0)
        cute.copy(
            st_atom,
            st_thr.partition_S(rdQ16),
            st_thr.partition_D(sdQ[dp_idx, None, wg_idx]),
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        self._ik_pop()
        self._ik_push("cQ_bar")
        dq_bar.arrive_and_wait()
        self._ik_pop()
        if is_wg_leader:
            if cutlass.const_expr("noredtma" not in self.dbg):
                cute.copy(
                    tma_atom_dQ_acc,
                    tdQsdQ[None, wg_idx],
                    tdQgdQ[None, q_tile, wg_idx, hb_coord],
                )
            reduce_tma_store_pipeline.producer_commit()
        self._ik_pop()
        return dq_state, ds_state

    @cute.jit
    def reduce(
        self,
        dSK_tiled_mma: cute.TiledMma,
        tdQtdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        mdQ_acc_in: cute.Tensor,
        mdQ_acc_gmem: cute.Tensor,
        fused_dq_args: tuple,
        sdQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: Float32,
        tmem_holding_buf: Int32,
        problem_shape_in: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        cumulative_s_q: Union[cute.Tensor, None],
        cumulative_s_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        vgrid: Tuple[Int32, Int32, Int32],
        lpt_group: Int32,
        sched_ring: cute.Tensor,
        dbg: Union[cute.Tensor, None],
        # (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline, sched_pipeline, mma_compute_dKdV_pipeline)
        pipeline_args: tuple,
        stage_mbar: cute.Pointer,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        (
            mma_reduce_dQ_pipeline,
            reduce_tma_store_pipeline,
            sched_pipeline,
            mma_compute_dKdV_pipeline,
            compute_mma_dS_pipeline,
        ) = pipeline_args
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage
        )
        # dS buffer free (dK and dQ of the tile retired): the dS pipeline's consumer-release
        # barrier completes once per tile; the reduce warps wait for the t-th completion
        ds_free_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage
        )
        ds_free_bar = compute_mma_dS_pipeline.consumer_get_barrier(ds_free_state)
        r_tile = Int32(0)

        mma_reduce_dQ_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage
        )
        reduce_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.reduce_tma_store_stage
        )
        sched_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.sched_stages
        )

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        # dQ arrives as two N=64 (d) halves through one (128 x 64) TMEM region
        cdQ = cute.make_identity_tensor((self.dSK_mma_tiler[0], self.dSK_mma_tiler[1]))
        chunks_per_half = self.dSK_mma_tiler[1] // 32

        thread_idx = tidx % (self.num_compute_warps * self.threads_per_warp)

        tdQtdQ = tdQtdQ[(None, None), 0, 0]

        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)

        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)
        if cutlass.const_expr(self.c16r):
            load_op16 = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype
            )
            tiled_t2r16 = tcgen05.make_tmem_copy(load_op16, tdQtdQ)
            thr_t2r16 = tiled_t2r16.get_slice(thread_idx)
            tTR_tdQ16 = thr_t2r16.partition_S(tdQtdQ)  # (16, 1, 8) plus the TMEM lane mode
            tTR_cdQ16 = thr_t2r16.partition_D(cdQ)     # (16, 1, 8) coordinate-side shape
            rdQ16c = cute.make_rmem_tensor(tTR_cdQ16[None, None, 0].shape, self.acc_dtype)
            rdQ16h = cute.make_rmem_tensor((16,), self.dq_acc_dtype)
            st_atom16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dq_acc_dtype, num_bits_per_copy=128
            )
            st_op16 = cute.make_cotiled_copy(st_atom16, cute.make_layout((1, 8)), rdQ16h.layout)
            st_thr16 = st_op16.get_slice(0)
        if cutlass.const_expr(self.fullw):
            # (32 cols, 1, 4 chunks) -> (32, 1, (2 chunks, 2 halves)): drained half by half
            n_ch = cute.size(tTR_cdQ, mode=[2])
            tTR_cdQ = cute.composition(
                tTR_cdQ, cute.make_layout((tTR_cdQ.shape[0], tTR_cdQ.shape[1], (n_ch // 2, 2)))
            )
            tTR_tdQ = cute.composition(
                tTR_tdQ, cute.make_layout((tTR_tdQ.shape[0], tTR_tdQ.shape[1], (n_ch // 2, 2)))
            )
        # 32x32b TMEM loads: thread t of the reduce warpgroup holds row (lane) t
        chunk_cols = cute.size(tTR_cdQ[None, None, 0])
        rdQ_cvt = cute.make_rmem_tensor((chunk_cols,), self.dq_acc_dtype)

        if cutlass.const_expr(self.fullw):
            half_shape = tTR_cdQ[None, None, (None, 0)].shape
            rdQ_half = cute.make_rmem_tensor(half_shape, self.acc_dtype)
            n_chunks_tile = cute.size(tTR_cdQ, mode=[2])
            rdQ16 = cute.make_rmem_tensor((chunk_cols, n_chunks_tile), self.dq_acc_dtype)
        elif cutlass.const_expr(self.reduce_drain_both):
            rdQ_halves = [
                cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype),
                cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype),
            ]
        else:
            rdQ_one = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
            rdQ_halves = [rdQ_one, rdQ_one]

        # D26c: the tile whose reduce-add completion has not been signalled yet (always the
        # previous tile: its four bulk groups are waited for after the next tile's four
        # groups were issued, at the end of the kernel with wait_group 0)
        fz_pipeline, fz_ring, sched_counter = fused_dq_args
        fz_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.fz_stages
        )
        lane_idx = tidx % 32
        fz_pushes = Int32(0)
        fz_blocked = Int32(0)
        fz_pending = Boolean(False)
        fz_iter = Int32(0)
        fz_hr = Int32(0)
        fz_hk = Int32(0)
        fz_b = Int32(0)
        fz_rowoff = Int32(0)
        fz_seqlen = Int32(0)

        self._ik_push("red_w_sched")
        sched_pipeline.consumer_wait(sched_consumer_state)
        self._ik_pop()
        item = sched_ring[sched_consumer_state.index]
        sched_pipeline.consumer_release(sched_consumer_state)
        sched_consumer_state.advance()
        while item >= 0:
          self._dbg_mark(dbg, 24, item)
          bidx, bidy, bidz, valid = self.item_to_block(
              item, vgrid, lpt_group, cumulative_s_q, cumulative_s_k
          )
          (
              blk_coord,
              blk_offset,
              problem_shape,
              trip_start,
              trip_count,
              trip_end,
              row_off,
          ) = self.item_setup(
              bidx, bidy, bidz, problem_shape_in, cumulative_s_q, cumulative_s_k,
              window_size_left, window_size_right,
          )
          blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
          blk_coord_h, blk_coord_b = blk_coord_batch
          blk_coord_h_r, blk_coord_h_k = blk_coord_h
          iter_count = trip_count
          iter_index = trip_start
          iter_start = trip_start
          iter_end = trip_end
          mdQ_acc = cute.domain_offset(
              (row_off, Int32(0), ((Int32(0), Int32(0)), Int32(0))), mdQ_acc_in
          )
          gdQ = cute.local_tile(mdQ_acc, (self.tile_shape_Q, 32), (None, None, None))
          tdQsdQ, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
              tma_atom_dQ_acc,
              0,
              cute.make_layout(1),
              cute.group_modes(sdQ, 0, 2),
              cute.group_modes(gdQ, 0, 2),
          )
          while iter_count > 0:
            self._dbg_mark(dbg, 25, iter_count)
            if cutlass.const_expr(self.c16r):
              # lean drain (fits 64-88 registers): 16-column TMEM chunks, two per 32-column
              # staging stage, one reduce-add per stage
              self._ik_push("red_w_mr_dQ")
              mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)
              self._ik_pop()
              self._ik_push("red_dQ")
              for c in cutlass.range_constexpr(4):
                  if warp_idx == 0:
                      reduce_tma_store_pipeline.producer_acquire()
                  self.reduce_sync_barrier.arrive_and_wait()
                  sdQ_row_c = sdQ[thread_idx, None, reduce_tma_store_producer_state.index]
                  sdQ_row_h = cute.composition(sdQ_row_c, cute.make_layout((16, 2)))
                  for hh in cutlass.range_constexpr(2):
                      cute.copy(tiled_t2r16, tTR_tdQ16[None, None, 2 * c + hh], rdQ16c)
                      cute.arch.fence_view_async_tmem_load()
                      if cutlass.const_expr(c == 3 and hh == 1):
                          mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
                          mma_reduce_dQ_consumer_state.advance()
                      rdQ16h.store(rdQ16c.load().to(self.dq_acc_dtype))
                      cute.copy(
                          st_atom16,
                          st_thr16.partition_S(rdQ16h),
                          st_thr16.partition_D(sdQ_row_h[None, hh]),
                      )
                  cute.arch.fence_proxy("async.shared", space="cta")
                  self.reduce_sync_barrier.arrive_and_wait()
                  if warp_idx == 0:
                      if cutlass.const_expr("noredtma" not in self.dbg):
                          cute.copy(
                              tma_atom_dQ_acc,
                              tdQsdQ[None, reduce_tma_store_producer_state.index],
                              tdQgdQ[None, iter_index, c, blk_coord_batch],
                          )
                      reduce_tma_store_pipeline.producer_commit()
                  reduce_tma_store_producer_state.advance()
              self._ik_pop()
            elif cutlass.const_expr(self.fullw):
              # one 128-column dQ tile: load + convert half 0, load half 1, release TMEM,
              # convert half 1, then stage the four 32-column chunks
              self._ik_push("red_w_mr_dQ")
              mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)
              self._ik_pop()
              for h in cutlass.range_constexpr(2):
                  cute.copy(tiled_t2r, tTR_tdQ[None, None, (None, h)], rdQ_half)
                  cute.arch.fence_view_async_tmem_load()
                  if cutlass.const_expr(h == 1):
                      mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
                      mma_reduce_dQ_consumer_state.advance()
                  for i in cutlass.range_constexpr(n_chunks_tile // 2):
                      rdQ16[None, h * (n_chunks_tile // 2) + i].store(
                          rdQ_half[None, None, i].load().to(self.dq_acc_dtype)
                      )
              if cutlass.const_expr(self.stage_in_ds):
                  # staging goes into the dS buffer: wait until dK/dQ of this tile retired
                  self._ik_push("red_w_dsfree")
                  cute.arch.mbarrier_wait(ds_free_bar, r_tile % 2)
                  self._ik_pop()
                  r_tile += 1
              if cutlass.const_expr(self.dq_atomic):
                  # row thread_idx of this q tile: 16-byte vector reductions along d
                  if cutlass.const_expr("nored" not in self.dbg):
                      q_row = iter_index * self.tile_shape_Q + thread_idx
                      if q_row < problem_shape[0]:
                          row_elem = cute.crd2idx(
                              (q_row + row_off, Int32(0), blk_coord_batch), mdQ_acc_gmem.layout
                          )
                          row_addr = mdQ_acc_gmem.iterator.toint() + Int64(row_elem) * Int64(
                              self.dq_acc_dtype.width // 8
                          )
                          rdQ16_w = cute.recast_tensor(rdQ16, Int32)
                          words_per_chunk = chunk_cols // 2
                          packed = "bf16x2" if self.dq_acc_dtype is BFloat16 else "f16x2"
                          for c in cutlass.range_constexpr(n_chunks_tile):
                              for v in cutlass.range_constexpr(words_per_chunk // 4):
                                  umma_raw.red_add_v4_16bit(
                                      row_addr + (c * chunk_cols * 2 + v * 16),
                                      [rdQ16_w[v * 4 + k, c] for k in range(4)],
                                      packed,
                                  )
              elif cutlass.const_expr("nored" not in self.dbg):
                  for c in cutlass.range_constexpr(n_chunks_tile):
                      if warp_idx == 0:
                          reduce_tma_store_pipeline.producer_acquire()
                      self.reduce_sync_barrier.arrive_and_wait()
                      sdQ_row_c = sdQ[thread_idx, None, reduce_tma_store_producer_state.index]
                      cute.autovec_copy(rdQ16[None, c], sdQ_row_c)
                      cute.arch.fence_proxy("async.shared", space="cta")
                      self.reduce_sync_barrier.arrive_and_wait()
                      if warp_idx == 0:
                          if cutlass.const_expr("noredtma" not in self.dbg):
                              cute.copy(
                                  tma_atom_dQ_acc,
                                  tdQsdQ[None, reduce_tma_store_producer_state.index],
                                  tdQgdQ[None, iter_index, c, blk_coord_batch],
                              )
                          reduce_tma_store_pipeline.producer_commit()
                      reduce_tma_store_producer_state.advance()
                  if cutlass.const_expr(self.fuse_dq and "nopush" not in self.fz_dbg):
                      if fz_pending:
                          # the previous tile's reduce-adds are complete once at most this
                          # tile's groups are pending; hand the tile to the finisher warp
                          if warp_idx == 0:
                              if cutlass.const_expr("nowait" not in self.fz_dbg):
                                  cute.arch.cp_async_bulk_wait_group(n_chunks_tile, read=False)
                              t_acq0 = Int64(0)
                              if cutlass.const_expr(self.fz_stats):
                                  t_acq0 = umma_raw.globaltimer()
                              fz_pipeline.producer_acquire(fz_producer_state)
                              if cutlass.const_expr(self.fz_stats):
                                  if lane_idx == 0:
                                      dt_acq = Int32(umma_raw.globaltimer() - t_acq0)
                                      fz_blocked += dt_acq
                                      cute.arch.atomic_add(sched_counter.iterator + 11, dt_acq)
                              if lane_idx == 0:
                                  fz_ring[2 * fz_producer_state.index] = fz_iter
                                  fz_ring[2 * fz_producer_state.index + 1] = (
                                      fz_hr | (fz_hk << 8) | (fz_b << 16)
                                  )
                              fz_pipeline.producer_commit(fz_producer_state)
                              fz_producer_state.advance()
                              fz_pushes += 1
                              self._dbg_mark(dbg, 30, fz_pushes)
                      fz_pending = Boolean(True)
                      fz_iter = iter_index
                      fz_hr = blk_coord_h_r
                      fz_hk = blk_coord_h_k
                      fz_b = blk_coord_b
                      fz_rowoff = row_off
                      fz_seqlen = problem_shape[0]
                  if cutlass.const_expr(self.stage_in_ds):
                      # the staging chunks live in the dS buffer: once their reads are done,
                      # tell the compute warps that the next dS store may proceed
                      if warp_idx == 0:
                          cute.arch.cp_async_bulk_wait_group(0, read=True)
                          with cute.arch.elect_one():
                              cute.arch.mbarrier_arrive(stage_mbar)
            elif cutlass.const_expr(self.reduce_drain_both):
              # drain both dQ halves from TMEM first so that the MMA warp is released quickly
              for n_half in cutlass.range_constexpr(2):
                self._ik_push("red_w_mr_dQ")
                mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)
                self._ik_pop()
                cute.copy(tiled_t2r, tTR_tdQ, rdQ_halves[n_half])
                cute.arch.fence_view_async_tmem_load()
                mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
                mma_reduce_dQ_consumer_state.advance()
            for n_half in cutlass.range_constexpr(0 if self.fullw else 2):
              if cutlass.const_expr(not self.reduce_drain_both):
                self._ik_push("red_w_mr_dQ")
                mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)
                self._ik_pop()
                cute.copy(tiled_t2r, tTR_tdQ, rdQ_halves[n_half])
                cute.arch.fence_view_async_tmem_load()
                mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
                mma_reduce_dQ_consumer_state.advance()
              tTR_rdQ = rdQ_halves[n_half]
              # We don't have enough smem to dump it all to smem, so we do it in stages
              if cutlass.const_expr("nored" not in self.dbg):
                  for i in cutlass.range(0, cute.size(tTR_cdQ, mode=[2]), unroll_full=True):
                      if warp_idx == 0:
                          reduce_tma_store_pipeline.producer_acquire()
                      # Wait in all threads for the acquire to complete
                      self.reduce_sync_barrier.arrive_and_wait()

                      rdQ_cvt.store(tTR_rdQ[None, None, i].load().to(self.dq_acc_dtype))
                      sdQ_row = sdQ[thread_idx, None, reduce_tma_store_producer_state.index]
                      cute.autovec_copy(rdQ_cvt, sdQ_row)

                      # Wait for the stores to all be visible to the TMA
                      cute.arch.fence_proxy(
                          "async.shared",
                          space="cta",
                      )
                      self.reduce_sync_barrier.arrive_and_wait()

                      if warp_idx == 0:
                          if cutlass.const_expr("noredtma" not in self.dbg):
                              cute.copy(
                                  tma_atom_dQ_acc,
                                  tdQsdQ[None, reduce_tma_store_producer_state.index],
                                  tdQgdQ[None, iter_index, n_half * chunks_per_half + i, blk_coord_batch],
                              )

                          reduce_tma_store_pipeline.producer_commit()

                      reduce_tma_store_producer_state.advance()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start
                blk_coord_h_r += 1
                blk_coord_batch = ((blk_coord_h_r, blk_coord_h_k), blk_coord_b)

          if cutlass.const_expr(self.epi_reduce):
              # dK/dV epilogue of this item (the state is carried across items)
              mma_compute_dKdV_consumer_state = self.epilogue_reduce(
                  blk_coord,
                  blk_offset,
                  problem_shape,
                  dK,
                  dV,
                  tdKtdK,
                  tdVtdV,
                  scale_softmax,
                  (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
              )
          self._ik_push("red_w_sched")
          sched_pipeline.consumer_wait(sched_consumer_state)
          self._ik_pop()
          item = sched_ring[sched_consumer_state.index]
          sched_pipeline.consumer_release(sched_consumer_state)
          sched_consumer_state.advance()
          self._dbg_mark(dbg, 26, Int32(1))
          self._dbg_mark(dbg, 27, item)
        self._dbg_mark(dbg, 26, Int32(2))
        if cutlass.const_expr(self.fuse_dq and "nopush" not in self.fz_dbg):
            if warp_idx == 0:
                if fz_pending:
                    if cutlass.const_expr("nowait" not in self.fz_dbg):
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                    fz_pipeline.producer_acquire(fz_producer_state)
                    if lane_idx == 0:
                        fz_ring[2 * fz_producer_state.index] = fz_iter
                        fz_ring[2 * fz_producer_state.index + 1] = (
                            fz_hr | (fz_hk << 8) | (fz_b << 16)
                        )
                    fz_pipeline.producer_commit(fz_producer_state)
                    fz_producer_state.advance()
                # sentinel: the finisher warp exits
                fz_pipeline.producer_acquire(fz_producer_state)
                if lane_idx == 0:
                    fz_ring[2 * fz_producer_state.index] = Int32(-1)
                    fz_ring[2 * fz_producer_state.index + 1] = Int32(0)
                fz_pipeline.producer_commit(fz_producer_state)
                fz_producer_state.advance()
                self._dbg_mark(dbg, 30, Int32(99))
                if cutlass.const_expr(self.fz_stats):
                    if lane_idx == 0:
                        cute.arch.atomic_max(sched_counter.iterator + 13, fz_blocked)
        reduce_tma_store_pipeline.producer_tail()
        if cutlass.const_expr(self.epi_reduce):
            self.reduce_sync_barrier.arrive_and_wait()
            if warp_idx == 0:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
        self._dbg_mark(dbg, 26, Int32(3))

    @cute.jit
    def _col_base(self, coords: cute.Tensor, first: int, extra: int = 0) -> Int32:
        """Column (q) coordinate of element `first` of a T2R coordinate partition plus
        `extra`, marked as a multiple of 16 for vectorized smem loads."""
        c0 = cute.get(coords[first], mode=[1]) + extra
        ret = c0
        if cutlass.const_expr(not isinstance(c0, int)):
            ret = cute.assume(c0, divby=16)
        return ret

    @cute.jit
    def split_last(self, t: cute.Tensor, n: int) -> cute.Tensor:
        """Reshape the last mode of t (size m) as (m // n, n)."""
        r = cute.rank(t.layout)
        m = cute.size(t, mode=[r - 1])
        shape = tuple(t.shape[i] for i in range(r - 1)) + ((m // n, n),)
        return cute.composition(t, cute.make_layout(shape))

    @cute.jit
    def pick_last(self, t: cute.Tensor, h: int) -> cute.Tensor:
        """Slice index h of the inner split of the last mode (see split_last)."""
        r = cute.rank(t.layout)
        coord = tuple([None] * (r - 1)) + ((None, h),)
        return t[coord]

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
    ) -> cute.Tensor:
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (wg_idx, None)]
        else:
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def quantize_pairs(self, input: cute.Tensor) -> cute.Tensor:
        """fp32 register tensor -> bf16 register tensor (same layout), either by F2FP
        (round to nearest) or by PRMT truncation of pre-scaled values (self.cvt_rz)."""
        ret = None
        if cutlass.const_expr(self.cvt_rz):
            n = cute.size(input)
            out32 = cute.make_rmem_tensor((n // 2,), Int32)
            for i in cutlass.range_constexpr(n // 2):
                out32[i] = fmha_utils.pack_bf16x2_rz(input[2 * i], input[2 * i + 1])
            ret = cute.make_tensor(
                cute.recast_ptr(out32.iterator, dtype=self.element_dtype), input.layout
            )
        else:
            ret = self.quantize(input, 4)
        return ret

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
    ) -> cute.Tensor:
        tidx, tidy, tidz = cute.arch.thread_idx()
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.element_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(self.quantize(regs, 4))
        tPc = thr_copy.partition_D(coord)

        preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
        preds = cute.make_rmem_tensor(preds_shape, Boolean)
        for v in cutlass.range_constexpr(preds.shape[0]):
            for m in cutlass.range_constexpr(preds.shape[1]):
                for n in cutlass.range_constexpr(preds.shape[2]):
                    for k in cutlass.range_constexpr(preds.shape[3]):
                        lhs = tPc[(0, v), m, n, k]
                        val = cute.elem_less(lhs, tensor_shape)
                        preds[v, m, n, k] = val

        cute.copy(copy_atom, tCr, tCg, pred=preds)

    @cute.jit
    def epilogue_reduce(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: Float32,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        """dV then dK: TMEM (fp32) -> registers (16-column chunks) -> bf16 -> global, by
        the 128 reduce threads (thread t owns key row t of the block)."""
        tidx, tidy, tidz = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args
        lane_row = tidx % 128
        key_row = blk_coord_k * self.tile_shape_K + lane_row
        row_ok = key_row < K

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )
        n_cols = self.cta_tiler[2]

        for which in cutlass.range_constexpr(2):
            # which 0: dV (no scale), which 1: dK (scaled by sm_scale)
            tAcc = tdVtdV if which == 0 else tdKtdK
            gOut = dV if which == 0 else dK
            tAcc = tAcc[(None, None), 0, 0]
            mOut = cute.make_tensor(
                gOut.iterator + cute.assume(blk_offset[1] * gOut.stride[0], divby=64),
                cute.make_layout((K, n_cols, HB), stride=gOut.stride),
            )
            gOut_tile = cute.local_tile(mOut, (self.tile_shape_K, n_cols), (None, None, None))
            gOut_tile = gOut_tile[None, None, blk_coord_k, 0, blk_coord_batch]
            tiled_t2r = tcgen05.make_tmem_copy(load_op, tAcc)
            thr_t2r = tiled_t2r.get_slice(lane_row)
            cOut = cute.make_identity_tensor((self.tile_shape_K, n_cols))
            tTR_c = thr_t2r.partition_D(cOut)
            tTR_t = thr_t2r.partition_S(tAcc)
            n_chunks = cute.size(tTR_c, mode=[2])
            chunk = cute.size(tTR_c, mode=[0])
            rchunk = cute.make_rmem_tensor(tTR_c[None, None, 0].shape, self.acc_dtype)
            grow = gOut_tile[lane_row, None]

            self._ik_push("epi_w_mc_dKdV")
            mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)
            self._ik_pop()
            for c in cutlass.range_constexpr(n_chunks):
                cute.copy(tiled_t2r, tTR_t[None, None, c], rchunk)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(which == 1):
                    for i in cutlass.range_constexpr(cute.size(rchunk)):
                        rchunk[i] = scale_softmax * rchunk[i]
                rbf = self.quantize_pairs(rchunk)
                col0 = self._col_base(tTR_c, c * chunk)
                if row_ok:
                    cute.autovec_copy(
                        cute.make_tensor(rbf.iterator, cute.make_layout(chunk)),
                        cute.make_tensor(grow.iterator + col0, cute.make_layout(chunk)),
                    )
            mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
            mma_compute_dKdV_consumer_state.advance()
        return mma_compute_dKdV_consumer_state

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: Float32,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        tdKtdK = tdKtdK[(None, None), 0, 0]

        mdK = cute.make_tensor(
            dK.iterator + cute.assume(blk_offset[1] * dK.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(
            mdK, (self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1]), (None, None, None)
        )
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]

        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1])),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = self.split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = self.split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = self.split_wg(tTR_tdK, num_warp_groups, wg_idx)

        mdV_in = cute.make_tensor(
            dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride)
        )
        mdV = cute.make_tensor(
            mdV_in.iterator + cute.assume(blk_offset[1] * mdV_in.stride[0], divby=64),
            mdV_in.layout,
        )
        gdV = cute.local_tile(
            mdV, (self.PdO_mma_tiler[0], self.PdO_mma_tiler[1]), (None, None, None)
        )
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[1], 0),
            cute.make_identity_tensor((self.PdO_mma_tiler[0], self.PdO_mma_tiler[1])),
        )

        tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = self.split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV)
        tTR_gdV = self.split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = self.split_wg(tTR_tdV, num_warp_groups, wg_idx)

        self._ik_push("epi_w_mc_dKdV")
        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)
        self._ik_pop()

        # Load tdVtdV
        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)

        # Store tdVgdV
        self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        cute.arch.fence_view_async_tmem_load()

        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(self.warp_commit):
            cute.arch.sync_warp()
            if lane_idx == 0:
                mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        else:
            mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        self._ik_push("epi_w_mc_dKdV")
        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)
        self._ik_pop()

        # Load tdKtdK
        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
            tTR_rdK[i] = scale_softmax * tTR_rdK[i]

        # Store tdKgdK
        self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.warp_commit):
            cute.arch.sync_warp()
            if lane_idx == 0:
                mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        else:
            mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()
        return mma_compute_dKdV_consumer_state

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
        acc_dtype: Type[cutlass.Numeric],
        num_rows: Int32,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        """Workspace views: delta (sum_OdO), scaled lse and the fp32 dQ accumulator.

        Dense: rows = seq_len per batch entry, batch stride = rows * H (* D).
        Varlen: rows = total tokens shared by all sequences (batch stride 0); the kernels
        add the sequence's cu_seqlens offset to the row index."""
        D, HB = problem_shape[2], problem_shape[3]
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)
        Q = cute.round_up(num_rows, 8)
        B_alloc = 1 if self.varlen else B

        acc_bytes = acc_dtype.width // 8
        sum_OdO_bytes = cute.assume(B_alloc * H * Q * acc_bytes, divby=acc_bytes)
        scaled_lse_bytes = cute.assume(B_alloc * H * Q * acc_bytes, divby=acc_bytes)

        sched_counter = cute.make_tensor(
            cute.recast_ptr(workspace.iterator, dtype=Int32), cute.make_layout(16)
        )
        sum_OdO_iter = workspace.iterator + self.sched_bytes
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dQ_acc_iter = scaled_lse_iter + scaled_lse_bytes

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.dq_acc_dtype)

        vec_b_stride = 0 if self.varlen else Q * H
        acc_b_stride = 0 if self.varlen else D * Q * H

        # D26c tile counters after the accumulator region, which is sized with the fp32
        # accumulator width (see _get_workspace_size); byte offsets from the uint8 workspace
        # pointer (dQ_acc_iter is already an fp16/fp32 element pointer)
        dQ_acc_bytes = cute.assume(B_alloc * H * Q * D * (Float32.width // 8), divby=16)
        counters_iter = cute.recast_ptr(
            workspace.iterator + (self.sched_bytes + sum_OdO_bytes + scaled_lse_bytes + dQ_acc_bytes),
            dtype=Int32,
        )
        tile_counters = cute.make_tensor(
            counters_iter,
            cute.make_layout((Q, ((H_r, H_k), B)), stride=(1, ((Q, Q * H_r), vec_b_stride))),
        )
        counters_elems = B_alloc * H * Q  # counters_iter is an Int32 pointer (element offsets)
        fz_queue_len = H * B_alloc * (cute.ceil_div(Q, 128) + 256) + 512
        fz_rb_len = B_alloc * (cute.ceil_div(Q, 128) + 256)
        # queue entries followed by the row-block counters (zeroed together)
        fz_queue = cute.make_tensor(
            counters_iter + counters_elems, cute.make_layout(fz_queue_len + fz_rb_len)
        )

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout((Q, ((H_r, H_k), B)), stride=(1, ((Q, Q * H_r), vec_b_stride))),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout((Q, ((H_r, H_k), B)), stride=(1, ((Q, Q * H_r), vec_b_stride))),
        )
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (Q, D, ((H_r, H_k), B)),
                stride=(D, 1, ((D * Q, D * Q * H_r), acc_b_stride)),
            ),
        )
        # fp32 word view of the same region (used for vectorized zeroing)
        W = D * self.dq_acc_dtype.width // 32
        w_b_stride = 0 if self.varlen else W * Q * H
        dQ_acc_words = cute.make_tensor(
            cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype),
            cute.make_layout(
                (Q, W, ((H_r, H_k), B)),
                stride=(W, 1, ((W * Q, W * Q * H_r), w_b_stride)),
            ),
        )

        return sum_OdO, scaled_lse, dQ_acc, dQ_acc_words, sched_counter, tile_counters, fz_queue

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        block_q: int,
    ) -> Tuple[int, int, int]:
        grid = (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )
        return grid

    def _compute_bwd_grid(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        block_k: int,
    ) -> Tuple[int, int, int]:
        K = problem_shape[1]
        H_R, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        n_kv = cute.round_up(cute.ceil_div(K, block_k), self.cluster_size)
        return (n_kv, cute.size(H_K), cute.size(B))

    @staticmethod
    def _get_workspace_size(
        q: int, k: int, d: int, h: int, b: int, acc_dtype: Type[cutlass.Numeric]
    ):
        d = (d + 7) // 8 * 8  # round up to 8
        q = (q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 256  # persistent-scheduler work counter
        # OdO vector
        workspace_bytes += b * h * q * acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += b * h * q * acc_dtype.width // 8
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += b * h * q * d * acc_dtype.width // 8
        # D26c: per (head, row) int32 tile-arrival counters
        workspace_bytes += b * h * q * 4
        # D26c: global queue of completed q tiles (one int32 per (tile, head, batch) plus
        # slack for sequence padding and the finishers' final over-reservations)
        workspace_bytes += (h * b * ((q + 127) // 128 + 256) + 512) * 4
        # D26c: per (batch, 128-row block) counters of completed heads
        workspace_bytes += b * ((q + 127) // 128 + 256) * 4
        return workspace_bytes

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr, cluster_layout_vmnk):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.cluster_size
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr, cluster_layout_vmnk):
        load_mma_dO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.cluster_size
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(
        self, load_compute_sum_OdO_mbar_ptr
    ):
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
        )

    def make_and_init_mma_reduce_dQ_pipeline(self, mma_reduce_dQ_mbar_ptr):
        mma_reduce_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_reduce_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads
            if (self.c16 and not self.c16r)
            else self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dQ_mbar_ptr,
            num_stages=self.mma_reduce_dQ_stage,
            producer_group=mma_reduce_dQ_producer_group,
            consumer_group=mma_reduce_dQ_consumer_group,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.compute_group_threads,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
        )

    def make_and_init_mma_compute_dKdV_pipeline(self, mma_compute_dKdV_mbar_ptr):
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp
            if self.epi_reduce
            else self.compute_group_threads,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
        )

    def make_and_init_reduce_tma_store_pipeline(self):
        reduce_tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            max(self.num_reduce_warps, 4) * self.threads_per_warp,
        )
        # c16: one staging buffer per warpgroup, so no outstanding bulk group may remain
        # when it is overwritten (wait_group.read 0)
        return pipeline.PipelineTmaStore.create(
            num_stages=1 if (self.c16 and not self.c16r) else self.reduce_tma_store_stage,
            producer_group=reduce_tma_store_producer_group,
        )
