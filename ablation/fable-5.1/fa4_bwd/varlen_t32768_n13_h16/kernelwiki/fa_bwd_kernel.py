# From-scratch FlashAttention-style varlen causal MHA backward for Blackwell (SM100) in CuTe DSL.
# Design: KV-stationary persistent kernel, 128x128 tiles, tcgen05 MMAs, TMEM-resident S/dP/P/dS/dQ/dK/dV,
# warp-specialized: warp0 = TMA producer, warp1 = MMA issuer, WG1/WG2 = softmax, WG3 = dQ drain (atomics).
import math, os
import torch
DBG = os.environ.get("FA_DBG", "")
SKIP_DSSMEM = "dssmem" in DBG
SKIP_TMEMST = "tmemst" in DBG
SKIP_RED = "red" in DBG or "nodrain" in DBG
DQ_BF16 = "dqf32" not in DBG      # accumulate dQ with bf16 atomics (default; FA_DBG=dqf32 reverts to fp32+convert)
SKIP_EPI = "epi" in DBG
NO_REGALLOC = "noreg" in DBG
DEBUG_WAIT = "dbgwait" in DBG
EARLY = "early" in DBG
DUMP = "dump" in DBG
TRACE = "trace" in DBG
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

BLK = 128          # q block == kv block == head dim
D = 128
STAGES = 2
NUM_THREADS = 640
bf16 = cutlass.BFloat16
f32 = cutlass.Float32
i32 = cutlass.Int32

# TMEM column map (fp32 column units)
COL_DP = 0      # dP^T region [0,128)   ; dS^T bf16 of WG w lives in [64w, 64w+32) (its own chunk-0 columns)
COL_S = 128     # S^T region [128,256)  ; P^T bf16 of WG w lives in [128+64w, +32)
COL_DQ = 0      # dQ^T accumulator [0,128)  (issued after dK, so the dP region is fully consumed)
COL_DV = 256
COL_DK = 384

NB_S_READ = 1    # WG1 arrives after reading S^T cols [192,256); WG0 waits before storing P^T there
NB_DP_READ = 2   # WG0 arrives after reading dP^T cols [0,64) ; WG1 waits before storing dS^T there
NB_TMEM = 3
NB_DRAIN = 4


@cute.struct
class SharedStorage:
    kv_full: cute.struct.MemRange[cutlass.Int64, 1]
    qdo_full: cute.struct.MemRange[cutlass.Int64, STAGES]
    qdo_empty: cute.struct.MemRange[cutlass.Int64, STAGES]
    stage_free: cute.struct.MemRange[cutlass.Int64, STAGES]
    peer_ready: cute.struct.MemRange[cutlass.Int64, 2]
    recv_full: cute.struct.MemRange[cutlass.Int64, 2]
    ds_free: cute.struct.MemRange[cutlass.Int64, STAGES]
    send_done: cute.struct.MemRange[cutlass.Int64, STAGES]
    rx_done: cute.struct.MemRange[cutlass.Int64, STAGES]
    xinfo: cute.struct.MemRange[i32, 8]
    s_full: cute.struct.MemRange[cutlass.Int64, 1]
    dp_full: cute.struct.MemRange[cutlass.Int64, 1]
    p_ready: cute.struct.MemRange[cutlass.Int64, 1]
    ds_ready: cute.struct.MemRange[cutlass.Int64, 1]
    dq_full: cute.struct.MemRange[cutlass.Int64, STAGES]
    dq_empty: cute.struct.MemRange[cutlass.Int64, 1]
    dkv_full: cute.struct.MemRange[cutlass.Int64, 1]
    dkv_empty: cute.struct.MemRange[cutlass.Int64, 1]
    lse: cute.struct.MemRange[f32, STAGES * BLK]
    delta: cute.struct.MemRange[f32, STAGES * BLK]
    tmem_holding_buf: i32


@cute.kernel
def fa_bwd_kernel(
    mma_S: cute.TiledMma,     # S^T = K Q^T      : A=K (kv,D) K-major smem, B=Q (q,D) K-major smem
    mma_dV: cute.TiledMma,    # dV += P^T dO     : A=P^T TMEM,           B=dO (D,q) MN-major smem
    mma_dQ: cute.TiledMma,    # dQ^T = K^T dS^T  : A=K (D,kv) MN-major,  B=dS^T (q,kv) MN-major smem
    tma_q: cute.CopyAtom, mQ: cute.Tensor,
    tma_k: cute.CopyAtom, mK: cute.Tensor,
    tma_v: cute.CopyAtom, mV: cute.Tensor,
    tma_do: cute.CopyAtom, mdO: cute.Tensor,
    mLSE: cute.Tensor, mDelta: cute.Tensor,       # (H, T) fp32
    mDQacc: cute.Tensor,                          # (T, H, D) fp32
    mDK: cute.Tensor, mDV: cute.Tensor,           # (T, H, D) bf16
    cu_seqlens: cute.Tensor,                      # (B+1) int32
    items: cute.Tensor,                           # (n_items, 4) int32: b, kb, h, pad
    cta_offsets: cute.Tensor,                     # (grid+1) int32
    scale_log2e: f32, scale: f32, total_T: i32, num_heads: i32, dbg: cute.Tensor,
    kv_layout_k: cute.ComposedLayout, kv_layout_mn: cute.ComposedLayout,
    q_layout_k: cute.ComposedLayout, q_layout_mn: cute.ComposedLayout,
    ds_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    wg = warp_idx // 4
    tid_wg = tidx % 128

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sK = smem.allocate_tensor(element_type=bf16, layout=kv_layout_k.outer, byte_alignment=1024, swizzle=kv_layout_k.inner)
    sV = smem.allocate_tensor(element_type=bf16, layout=kv_layout_k.outer, byte_alignment=1024, swizzle=kv_layout_k.inner)
    sQ = smem.allocate_tensor(element_type=bf16, layout=q_layout_k.outer, byte_alignment=1024, swizzle=q_layout_k.inner)
    sdO = smem.allocate_tensor(element_type=bf16, layout=q_layout_k.outer, byte_alignment=1024, swizzle=q_layout_k.inner)
    sDS = smem.allocate_tensor(element_type=bf16, layout=ds_layout.outer, byte_alignment=1024, swizzle=ds_layout.inner)
    # alternate views on the same bytes
    sK_mn = cute.make_tensor(sK.iterator, kv_layout_mn.outer)
    sQ_mn = cute.make_tensor(sQ.iterator, q_layout_mn.outer)
    sdO_mn = cute.make_tensor(sdO.iterator, q_layout_mn.outer)
    sLSE = storage.lse.get_tensor(cute.make_layout((BLK, STAGES)))
    sDelta = storage.delta.get_tensor(cute.make_layout((BLK, STAGES)))

    kv_full = storage.kv_full.data_ptr()
    qdo_full = storage.qdo_full.data_ptr()
    qdo_empty = storage.qdo_empty.data_ptr()
    stage_free = storage.stage_free.data_ptr()
    peer_ready = storage.peer_ready.data_ptr()
    recv_full = storage.recv_full.data_ptr()
    ds_free = storage.ds_free.data_ptr()
    send_done = storage.send_done.data_ptr()
    rx_done = storage.rx_done.data_ptr()
    xinfo = storage.xinfo.data_ptr()
    s_full = storage.s_full.data_ptr()
    dp_full = storage.dp_full.data_ptr()
    p_ready = storage.p_ready.data_ptr()
    ds_ready = storage.ds_ready.data_ptr()
    dq_full = storage.dq_full.data_ptr()
    dq_empty = storage.dq_empty.data_ptr()
    dkv_full = storage.dkv_full.data_ptr()
    dkv_empty = storage.dkv_empty.data_ptr()

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(kv_full, 1)
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(qdo_full + s, 32)
                cute.arch.mbarrier_init(qdo_empty + s, 1)
                cute.arch.mbarrier_init(stage_free + s, 1)
            cute.arch.mbarrier_init(s_full, 1)
            cute.arch.mbarrier_init(dp_full, 1)
            cute.arch.mbarrier_init(p_ready, 256)
            cute.arch.mbarrier_init(ds_ready, 256)
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(dq_full + s, 1)
                cute.arch.mbarrier_init(ds_free + s, 128)
            cute.arch.mbarrier_init(dq_empty, 128)
            for p in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(peer_ready + p, 1)
                cute.arch.mbarrier_init(recv_full + p, 1)
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(send_done + s, 1)
                cute.arch.mbarrier_init(rx_done + s, 128)
            cute.arch.mbarrier_init(dkv_full, 1)
            cute.arch.mbarrier_init(dkv_empty, 256)
        cute.arch.mbarrier_init_fence()
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_k)
        cpasync.prefetch_descriptor(tma_v)
        cpasync.prefetch_descriptor(tma_do)

    tmem_bar = pipeline.NamedBarrier(barrier_id=NB_TMEM, num_threads=NUM_THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_bar, allocator_warp_id=1)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(f32)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    rank = bidx % 2
    peer = 1 - rank
    cluster_id = bidx // 2

    # TMEM tensors
    acc_shape_S = mma_S.partition_shape_C((BLK, BLK))
    tS = cute.make_tensor(tmem_ptr + COL_S, mma_S.make_fragment_C(acc_shape_S).layout)
    tdP = cute.make_tensor(tmem_ptr + COL_DP, mma_S.make_fragment_C(acc_shape_S).layout)
    tdQ = cute.make_tensor(tmem_ptr + COL_DQ, mma_dQ.make_fragment_C(mma_dQ.partition_shape_C((BLK, BLK))).layout)
    tdV = cute.make_tensor(tmem_ptr + COL_DV, mma_dV.make_fragment_C(mma_dV.partition_shape_C((BLK, BLK))).layout)
    tdK = cute.make_tensor(tmem_ptr + COL_DK, mma_dV.make_fragment_C(mma_dV.partition_shape_C((BLK, BLK))).layout)
    a_frag_shape_h = mma_dV.partition_shape_A((BLK, 64))
    a_layout_h = mma_dV.make_fragment_A(a_frag_shape_h).layout
    tP_bf_h = [cute.make_tensor(cute.recast_ptr(tmem_ptr + (COL_S + 64 * kh), dtype=bf16), a_layout_h) for kh in range(2)]
    tdS_bf_h = [cute.make_tensor(cute.recast_ptr(tmem_ptr + (COL_DP + 64 * kh), dtype=bf16), a_layout_h) for kh in range(2)]

    # MMA fragments
    thr_S = mma_S.get_slice(0)
    thr_dV = mma_dV.get_slice(0)
    thr_dQ = mma_dQ.get_slice(0)
    tCrK = mma_S.make_fragment_A(thr_S.partition_A(sK))        # (MMA, MMA_M, MMA_K)
    tCrV = mma_S.make_fragment_A(thr_S.partition_A(sV))
    tCrQ = mma_S.make_fragment_B(thr_S.partition_B(sQ))        # (MMA, MMA_N, MMA_K, STAGE)
    tCrdO = mma_S.make_fragment_B(thr_S.partition_B(sdO))
    sdO_mn_h = [cute.local_tile(sdO_mn, (D, 64, STAGES), (0, kh, 0)) for kh in range(2)]
    sQ_mn_h = [cute.local_tile(sQ_mn, (D, 64, STAGES), (0, kh, 0)) for kh in range(2)]
    tCrdO_mn_h = [mma_dV.make_fragment_B(thr_dV.partition_B(sdO_mn_h[kh])) for kh in range(2)]  # (MMA, MMA_N(D), MMA_K(q/2), STAGE)
    tCrQ_mn_h = [mma_dV.make_fragment_B(thr_dV.partition_B(sQ_mn_h[kh])) for kh in range(2)]
    tCrK_mn = mma_dQ.make_fragment_A(thr_dQ.partition_A(sK_mn))
    tCrDS = mma_dQ.make_fragment_B(thr_dQ.partition_B(sDS))

    n_my_items = cta_offsets[cluster_id + 1] - cta_offsets[cluster_id]
    item_base = cta_offsets[cluster_id]
    if cutlass.const_expr(EARLY):
        if tidx == 0:
            dbg[bidx * 16 + 0] = n_my_items + 1000
            if n_my_items > 0:
                b0 = items[(item_base, 0)]
                dbg[bidx * 16 + 1] = b0
                dbg[bidx * 16 + 2] = items[(item_base, 1)]
                dbg[bidx * 16 + 3] = items[(item_base, 2)]
                dbg[bidx * 16 + 4] = cu_seqlens[b0 + 1] - cu_seqlens[b0]
                dbg[bidx * 16 + 5] = item_base

    # ------------------------------------------------------------------ producer warp
    if cutlass.const_expr(EARLY):
        pass
    elif warp_idx == 16:
        if cutlass.const_expr(not NO_REGALLOC):
            cute.arch.warpgroup_reg_dealloc(AUX_REGS)
        lane = tidx % 32
        stage = i32(0)
        qdo_phase = i32(0)   # phase for waiting on qdo_empty: first wait must pass -> start at 1? we wait parity (phase^1)
        kv_phase = i32(0)
        n_valid = i32(0)
        p_iter = i32(0)
        xflag0 = i32(0)
        xflag1 = i32(0)
        xcnt0 = i32(0)           # exchange uses of stage 0 so far (parity of send_done[0]/rx_done[0])
        xcnt1 = i32(0)
        for it in cutlass.range(n_my_items):
            item = item_base + it
            b = items[(item * 2 + rank, 0)]
            kb = items[(item * 2 + rank, 1)]
            h = items[(item * 2 + rank, 2)]
            paired = items[(item * 2 + rank, 3)]
            kb0 = kb - rank * paired
            valid = b >= 0
            if b < 0:
                b = i32(0)
            seq_start = cu_seqlens[b]
            seq_end = cu_seqlens[b + 1]
            seqlen = seq_end - seq_start
            nqb = (seqlen + BLK - 1) // BLK
            if not valid:
                nqb = kb
            k_row = seq_start + kb * BLK
            # K/V: wait until previous item's MMAs are all done (dkv_full) — except first valid item
            if valid and n_valid > 0:
                wait_dbg(dkv_full, kv_phase, i32(1), dbg, bidx * 32 + warp_idx)
                kv_phase = kv_phase ^ 1
            if valid:
                n_valid = n_valid + 1
            if valid:
                gK = cute.local_tile(cute.domain_offset((k_row, 0, h), mK), (BLK, D), (0, 0, 0))  # (BLK, D)
                gV = cute.local_tile(cute.domain_offset((k_row, 0, h), mV), (BLK, D), (0, 0, 0))
                tKsK, tKgK = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
                tVsV, tVgV = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(kv_full, 2 * BLK * D * 2)
                cute.copy(tma_k, tKgK, tKsK, tma_bar_ptr=kv_full)
                cute.copy(tma_v, tVgV, tVsV, tma_bar_ptr=kv_full)
            n0 = nqb - kb0
            nproc = i32(0)
            for qbi in cutlass.range(n0):
                sidx = (qbi + (kb0 % ROT)) % n0
                qb = nqb - 1 - sidx
                if qb >= kb:
                    q_row = seq_start + qb * BLK
                    # wait for stage to be empty
                    do_xchg = (paired != 0) and (qb > kb0)
                    if cutlass.const_expr(NODRAIN):
                        do_xchg = False
                    wait_dbg(qdo_empty + stage, qdo_phase ^ 1, i32(2) + 1000 * p_iter, dbg, bidx * 32 + warp_idx)
                    # if the previous block that used this stage staged a DSMEM send in it, wait until the peer consumed it
                    xf = xflag0
                    xp = xcnt0 % 2
                    if stage == 1:
                        xf = xflag1
                        xp = xcnt1 % 2
                    if xf > 0:
                        wait_dbg(send_done + stage, xp, i32(3) + 1000 * p_iter, dbg, bidx * 32 + warp_idx)
                        wait_dbg(rx_done + stage, xp, i32(4) + 1000 * p_iter, dbg, bidx * 32 + warp_idx)
                        if stage == 0:
                            xcnt0 = xcnt0 + 1
                        else:
                            xcnt1 = xcnt1 + 1
                    if stage == 0:
                        xflag0 = i32(0)
                        if do_xchg:
                            xflag0 = i32(1)
                    else:
                        xflag1 = i32(0)
                        if do_xchg:
                            xflag1 = i32(1)
                    p_iter = p_iter + 1
                    gQ = cute.local_tile(cute.domain_offset((q_row, 0, h), mQ), (BLK, D), (0, 0, 0))
                    gdO = cute.local_tile(cute.domain_offset((q_row, 0, h), mdO), (BLK, D), (0, 0, 0))
                    tQsQ, tQgQ = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
                    tOsO, tOgO = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdO, 0, 2))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(qdo_full + stage, 2 * BLK * D * 2)
                    cute.copy(tma_q, tQgQ, tQsQ[(None, stage)], tma_bar_ptr=qdo_full + stage)
                    cute.copy(tma_do, tOgO, tOsO[(None, stage)], tma_bar_ptr=qdo_full + stage)
                    # LSE / delta for this q block: 128 floats each, 4 per lane
                    for r in cutlass.range_constexpr(4):
                        j = lane * 4 + r
                        row = q_row + j
                        if row < total_T:
                            sLSE[(j, stage)] = mLSE[(h, row)] * f32(-1.4426950408889634)
                            sDelta[(j, stage)] = mDelta[(h, row)]
                        else:
                            sLSE[(j, stage)] = f32(0.0)
                            sDelta[(j, stage)] = f32(0.0)
                    cute.arch.mbarrier_arrive(qdo_full + stage)
                    stage = stage + 1
                    if stage == STAGES:
                        stage = i32(0)
                        qdo_phase = qdo_phase ^ 1

    # ------------------------------------------------------------------ MMA warp
    elif warp_idx == 17:
        if cutlass.const_expr(not NO_REGALLOC):
            cute.arch.warpgroup_reg_dealloc(AUX_REGS)
        stage = i32(0)
        qdo_phase = i32(0)
        kv_phase = i32(0)
        pr_phase = i32(0)
        ds_phase = i32(0)
        dqe_phase = i32(0)
        dkve_phase = i32(0)
        n_done = i32(0)   # global q-block counter for this CTA
        n_valid = i32(0)
        for it in cutlass.range(n_my_items):
            item = item_base + it
            b = items[(item * 2 + rank, 0)]
            kb = items[(item * 2 + rank, 1)]
            paired = items[(item * 2 + rank, 3)]
            kb0 = kb - rank * paired
            valid = b >= 0
            if b < 0:
                b = i32(0)
            seq_start = cu_seqlens[b]
            seqlen = cu_seqlens[b + 1] - seq_start
            nqb = (seqlen + BLK - 1) // BLK
            if not valid:
                nqb = kb
            if valid:
                wait_dbg(kv_full, kv_phase, i32(10), dbg, bidx * 32 + warp_idx)
                kv_phase = kv_phase ^ 1
                if n_valid > 0:
                    wait_dbg(dkv_empty, dkve_phase, i32(11), dbg, bidx * 32 + warp_idx)
                    dkve_phase = dkve_phase ^ 1
                n_valid = n_valid + 1
            n0 = nqb - kb0
            nproc = i32(0)
            for qbi in cutlass.range(n0):
                sidx = (qbi + (kb0 % ROT)) % n0
                qb = nqb - 1 - sidx
                if qb >= kb:
                    trace(dbg, i32(0), i32(0), n_done, bidx == 0 and tidx == 544)
                    wait_dbg(qdo_full + stage, qdo_phase, i32(12) + 1000 * n_done, dbg, bidx * 32 + warp_idx)
                    trace(dbg, i32(0), i32(1), n_done, bidx == 0 and tidx == 544)
                    # S^T = K Q^T   (S region is free: P^T of the previous block was consumed by its dV MMA, in order)
                    mma_S.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(mma_S, tS, tCrK, tCrQ[(None, None, None, stage)], tS)
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full)
                    # dP^T = V dO^T  (its region holds dQ^T of the previous block: wait for the drain)
                    if n_done > 0:
                        wait_dbg(dq_empty, dqe_phase, i32(13), dbg, bidx * 32 + warp_idx)
                        dqe_phase = dqe_phase ^ 1
                    cute.gemm(mma_S, tdP, tCrV, tCrdO[(None, None, None, stage)], tdP)
                    with cute.arch.elect_one():
                        tcgen05.commit(dp_full)
                    trace(dbg, i32(0), i32(2), n_done, bidx == 0 and tidx == 544)
                    # wait P^T
                    wait_dbg(p_ready, pr_phase, i32(14), dbg, bidx * 32 + warp_idx)
                    pr_phase = pr_phase ^ 1
                    trace(dbg, i32(0), i32(3), n_done, bidx == 0 and tidx == 544)
                    mma_dV.set(tcgen05.Field.ACCUMULATE, nproc > 0)
                    cute.gemm(mma_dV, tdV, tP_bf_h[0], tCrdO_mn_h[0][(None, None, None, stage)], tdV)
                    mma_dV.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(mma_dV, tdV, tP_bf_h[1], tCrdO_mn_h[1][(None, None, None, stage)], tdV)
                    # wait dS^T
                    wait_dbg(ds_ready, ds_phase, i32(15), dbg, bidx * 32 + warp_idx)
                    ds_phase = ds_phase ^ 1
                    trace(dbg, i32(0), i32(4), n_done, bidx == 0 and tidx == 544)
                    mma_dV.set(tcgen05.Field.ACCUMULATE, nproc > 0)
                    cute.gemm(mma_dV, tdK, tdS_bf_h[0], tCrQ_mn_h[0][(None, None, None, stage)], tdK)
                    mma_dV.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(mma_dV, tdK, tdS_bf_h[1], tCrQ_mn_h[1][(None, None, None, stage)], tdK)
                    with cute.arch.elect_one():
                        tcgen05.commit(qdo_empty + stage)
                    mma_dQ.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(mma_dQ, tdQ, tCrK_mn, tCrDS, tdQ)
                    with cute.arch.elect_one():
                        tcgen05.commit(dq_full + stage)
                    nproc = nproc + 1
                    n_done = n_done + 1
                    stage = stage + 1
                    if stage == STAGES:
                        stage = i32(0)
                        qdo_phase = qdo_phase ^ 1
            if valid:
                with cute.arch.elect_one():
                    tcgen05.commit(dkv_full)

    # ------------------------------------------------------------------ softmax warpgroups
    elif wg == 1 or wg == 2:
        if cutlass.const_expr(not NO_REGALLOC):
            cute.arch.warpgroup_reg_alloc(SOFTMAX_REGS)
        w = wg - 1            # 0 or 1 : owns q columns [64w, 64w+64)
        lane_row = tid_wg     # kv row within block (TMEM lane)
        col0 = 64 * w
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), f32)
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Uint32)
        S_layout = cute.make_layout((BLK, 32), stride=(65536, 1))
        S16_layout = cute.make_layout((BLK, 16), stride=(65536, 1))
        tmem_ptr_u32 = cute.recast_ptr(tmem_ptr, dtype=cutlass.Uint32)
        tS_c0 = cute.make_tensor(tmem_ptr + (COL_S + col0), S_layout)
        tS_c1 = cute.make_tensor(tmem_ptr + (COL_S + col0 + 32), S_layout)
        tdP_c0 = cute.make_tensor(tmem_ptr + (COL_DP + col0), S_layout)
        tdP_c1 = cute.make_tensor(tmem_ptr + (COL_DP + col0 + 32), S_layout)
        tP_st0 = cute.make_tensor(tmem_ptr_u32 + (COL_S + col0), S16_layout)
        tP_st1 = cute.make_tensor(tmem_ptr_u32 + (COL_S + col0 + 16), S16_layout)
        tdS_st0 = cute.make_tensor(tmem_ptr_u32 + (COL_DP + col0), S16_layout)
        tdS_st1 = cute.make_tensor(tmem_ptr_u32 + (COL_DP + col0 + 16), S16_layout)
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tS_c0)
        thr_ld = tiled_ld.get_slice(tid_wg)
        tiled_st = tcgen05.make_tmem_copy(st_atom, tP_st0)
        thr_st = tiled_st.get_slice(tid_wg)
        tS_c0_p = thr_ld.partition_S(tS_c0)
        tS_c1_p = thr_ld.partition_S(tS_c1)
        tdP_c0_p = thr_ld.partition_S(tdP_c0)
        tdP_c1_p = thr_ld.partition_S(tdP_c1)
        tP_st0_p = thr_st.partition_D(tP_st0)
        tP_st1_p = thr_st.partition_D(tP_st1)
        tdS_st0_p = thr_st.partition_D(tdS_st0)
        tdS_st1_p = thr_st.partition_D(tdS_st1)
        tdV_c0 = cute.make_tensor(tmem_ptr + (COL_DV + col0), S_layout)
        tdV_c1 = cute.make_tensor(tmem_ptr + (COL_DV + col0 + 32), S_layout)
        tdK_c0 = cute.make_tensor(tmem_ptr + (COL_DK + col0), S_layout)
        tdK_c1 = cute.make_tensor(tmem_ptr + (COL_DK + col0 + 32), S_layout)
        tdV_c0_p = thr_ld.partition_S(tdV_c0)
        tdV_c1_p = thr_ld.partition_S(tdV_c1)
        tdK_c0_p = thr_ld.partition_S(tdK_c0)
        tdK_c1_p = thr_ld.partition_S(tdK_c1)

        frag32 = cute.make_layout(((32, 1), 1, 1), stride=((1, 0), 0, 0))
        frag16 = cute.make_layout(((16, 1), 1, 1), stride=((1, 0), 0, 0))
        rS0 = cute.make_rmem_tensor(frag32, f32)
        rS1 = cute.make_rmem_tensor(frag32, f32)
        rP0 = cute.make_rmem_tensor(frag32, f32)
        rP1 = cute.make_rmem_tensor(frag32, f32)
        rK = cute.make_rmem_tensor(frag16, cutlass.Uint32)   # packed bf16x2
        rL4 = cute.make_rmem_tensor((4,), f32)
        sLSE4 = cute.make_tensor(storage.lse.data_ptr(), cute.make_layout((4, BLK // 4, STAGES), stride=(1, 4, BLK)))
        sDelta4 = cute.make_tensor(storage.delta.data_ptr(), cute.make_layout((4, BLK // 4, STAGES), stride=(1, 4, BLK)))
        rK1 = cute.make_rmem_tensor(frag16, cutlass.Uint32)
        # dS^T smem: (q, kv) MN-major SW128; this thread writes row kv=lane_row, q in [col0, col0+64) as 8 x 16B
        # byte offset of 16B chunk (q0 = 8*c8, kv=t): atom=(t//8) + 16*(q0//64); r=t%8; chunk=((q0%64)//8) ^ r
        sDS_base = sDS.iterator.toint()
        r8 = lane_row % 8
        ds_row_base = sDS_base + (lane_row // 8) * 1024 + (col0 // 64) * 16384 + r8 * 128
        stage = i32(0)
        qdo_phase = i32(0)
        s_phase = i32(0)
        dp_phase = i32(0)
        dsf_ph0 = i32(1)     # first waits on ds_free[s] must pass immediately
        dsf_ph1 = i32(1)
        dkvf_phase = i32(0)
        n_sm = i32(0)
        tr_on = bidx == 0 and tidx == 128
        for it in cutlass.range(n_my_items):
            item = item_base + it
            b = items[(item * 2 + rank, 0)]
            kb = items[(item * 2 + rank, 1)]
            h = items[(item * 2 + rank, 2)]
            paired = items[(item * 2 + rank, 3)]
            kb0 = kb - rank * paired
            valid = b >= 0
            if b < 0:
                b = i32(0)
            seq_start = cu_seqlens[b]
            seqlen = cu_seqlens[b + 1] - seq_start
            nqb = (seqlen + BLK - 1) // BLK
            if not valid:
                nqb = kb
            k_row_rel = kb * BLK + lane_row      # kv position within doc
            n0 = nqb - kb0
            nproc = i32(0)
            for qbi in cutlass.range(n0):
                sidx = (qbi + (kb0 % ROT)) % n0
                qb = nqb - 1 - sidx
                if qb >= kb:
                    q0_rel = qb * BLK + col0
                    need_mask = (qb == kb) or (qb == nqb - 1)
                    wait_dbg(qdo_full + stage, qdo_phase, i32(24) + 1000 * n_sm, dbg, bidx * 32 + warp_idx)
                    # ================= P phase
                    trace(dbg, i32(1), i32(0), n_sm, tr_on)
                    wait_dbg(s_full, s_phase, i32(20), dbg, bidx * 32 + warp_idx)
                    s_phase = s_phase ^ 1
                    trace(dbg, i32(1), i32(1), n_sm, tr_on)
                    cute.copy(tiled_ld, tS_c0_p, rS0)
                    cute.copy(tiled_ld, tS_c1_p, rS1)
                    cute.arch.fence_view_async_tmem_load()
                    trace(dbg, i32(1), i32(2), n_sm, tr_on)
                    if cutlass.const_expr(DUMP):
                        if bidx == 0 and it == 0 and qb == kb:
                            for i in cutlass.range_constexpr(32):
                                dbg[(0 * 128 + lane_row) * 128 + col0 + i] = rS0[i]
                                dbg[(0 * 128 + lane_row) * 128 + col0 + 32 + i] = rS1[i]
                    # chunk 0
                    for i in cutlass.range_constexpr(16):
                        if cutlass.const_expr(i % 2 == 0):
                            cute.autovec_copy(sLSE4[(None, (col0 + 2 * i) // 4, stage)], rL4)
                        l0 = rL4[(2 * i) % 4]
                        l1 = rL4[(2 * i) % 4 + 1]
                        if cutlass.const_expr(NOPACK):
                            x0 = rS0[2 * i] * scale_log2e + l0
                            x1 = rS0[2 * i + 1] * scale_log2e + l1
                        else:
                            x0, x1 = cute.arch.fma_packed_f32x2((rS0[2 * i], rS0[2 * i + 1]), (scale_log2e, scale_log2e), (l0, l1))
                        if cutlass.const_expr(X_NOEXP):
                            rP0[2 * i] = x0
                            rP0[2 * i + 1] = x1
                        else:
                            rP0[2 * i] = cute.arch.exp2(x0)
                            if cutlass.const_expr(EXP_EMU and (i % EMU_DIV == EMU_DIV - 1)):
                                rP0[2 * i + 1] = exp2_emu(x1)
                            else:
                                rP0[2 * i + 1] = cute.arch.exp2(x1)
                    if need_mask:
                        for i in cutlass.range_constexpr(32):
                            qi = q0_rel + i
                            if (qi < k_row_rel) or (qi >= seqlen):
                                rP0[i] = f32(0.0)
                    pack_bf16x2(rP0, rK)
                    if cutlass.const_expr(not X_NOTST):
                        cute.copy(tiled_st, rK, tP_st0_p)
                    # chunk 1
                    for i in cutlass.range_constexpr(16):
                        if cutlass.const_expr(i % 2 == 0):
                            cute.autovec_copy(sLSE4[(None, (col0 + 32 + 2 * i) // 4, stage)], rL4)
                        l0 = rL4[(2 * i) % 4]
                        l1 = rL4[(2 * i) % 4 + 1]
                        if cutlass.const_expr(NOPACK):
                            x0 = rS1[2 * i] * scale_log2e + l0
                            x1 = rS1[2 * i + 1] * scale_log2e + l1
                        else:
                            x0, x1 = cute.arch.fma_packed_f32x2((rS1[2 * i], rS1[2 * i + 1]), (scale_log2e, scale_log2e), (l0, l1))
                        if cutlass.const_expr(X_NOEXP):
                            rP1[2 * i] = x0
                            rP1[2 * i + 1] = x1
                        else:
                            rP1[2 * i] = cute.arch.exp2(x0)
                            if cutlass.const_expr(EXP_EMU and (i % EMU_DIV == EMU_DIV - 1)):
                                rP1[2 * i + 1] = exp2_emu(x1)
                            else:
                                rP1[2 * i + 1] = cute.arch.exp2(x1)
                    if need_mask:
                        for i in cutlass.range_constexpr(32):
                            qi = q0_rel + 32 + i
                            if (qi < k_row_rel) or (qi >= seqlen):
                                rP1[i] = f32(0.0)
                    pack_bf16x2(rP1, rK1)
                    if cutlass.const_expr(not X_NOTST):
                        cute.copy(tiled_st, rK1, tP_st1_p)
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.mbarrier_arrive(p_ready)
                    trace(dbg, i32(1), i32(3), n_sm, tr_on)
                    # ================= dS phase
                    wait_dbg(dp_full, dp_phase, i32(21), dbg, bidx * 32 + warp_idx)
                    dp_phase = dp_phase ^ 1
                    trace(dbg, i32(1), i32(4), n_sm, tr_on)
                    cute.copy(tiled_ld, tdP_c0_p, rS0)
                    cute.arch.fence_view_async_tmem_load()
                    trace(dbg, i32(1), i32(5), n_sm, tr_on)
                    if cutlass.const_expr(DUMP):
                        if bidx == 0 and it == 0 and qb == kb:
                            for i in cutlass.range_constexpr(32):
                                dbg[(1 * 128 + lane_row) * 128 + col0 + i] = rS0[i]
                                dbg[(1 * 128 + lane_row) * 128 + col0 + 32 + i] = rS1[i]
                                dbg[(5 * 128 + lane_row) * 128 + col0 + i] = rP0[i]
                                dbg[(5 * 128 + lane_row) * 128 + col0 + 32 + i] = rP1[i]
                    # wait until the dS^T smem buffer is free (previous dQ MMA done and any received partial consumed)
                    if cutlass.const_expr(not X_NODSF):
                        if stage == 0:
                            wait_dbg(ds_free + 0, dsf_ph0, i32(22), dbg, bidx * 32 + warp_idx)
                            dsf_ph0 = dsf_ph0 ^ 1
                        else:
                            wait_dbg(ds_free + 1, dsf_ph1, i32(22), dbg, bidx * 32 + warp_idx)
                            dsf_ph1 = dsf_ph1 ^ 1
                    # chunk 0
                    for i in cutlass.range_constexpr(16):
                        if cutlass.const_expr(i % 2 == 0):
                            cute.autovec_copy(sDelta4[(None, (col0 + 2 * i) // 4, stage)], rL4)
                        d0 = rL4[(2 * i) % 4]
                        d1 = rL4[(2 * i) % 4 + 1]
                        if cutlass.const_expr(NOPACK):
                            rS0[2 * i] = (rS0[2 * i] - d0) * rP0[2 * i]
                            rS0[2 * i + 1] = (rS0[2 * i + 1] - d1) * rP0[2 * i + 1]
                        else:
                            t0, t1 = cute.arch.sub_packed_f32x2((rS0[2 * i], rS0[2 * i + 1]), (d0, d1))
                            rS0[2 * i], rS0[2 * i + 1] = cute.arch.mul_packed_f32x2((t0, t1), (rP0[2 * i], rP0[2 * i + 1]))
                    pack_bf16x2(rS0, rK)
                    if cutlass.const_expr(not X_NOTST):
                        cute.copy(tiled_st, rK, tdS_st0_p)
                    if cutlass.const_expr(not X_NOSTS):
                        for c8 in cutlass.range_constexpr(4):
                            p = cute.make_ptr(cutlass.Uint32, ds_row_base + ((c8 ^ r8) * 16), cute.AddressSpace.smem, assumed_align=16)
                            cute.arch.store(p, vec4u(rK[4 * c8], rK[4 * c8 + 1], rK[4 * c8 + 2], rK[4 * c8 + 3]))
                    # chunk 1
                    cute.copy(tiled_ld, tdP_c1_p, rS1)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(16):
                        if cutlass.const_expr(i % 2 == 0):
                            cute.autovec_copy(sDelta4[(None, (col0 + 32 + 2 * i) // 4, stage)], rL4)
                        d0 = rL4[(2 * i) % 4]
                        d1 = rL4[(2 * i) % 4 + 1]
                        if cutlass.const_expr(NOPACK):
                            rS1[2 * i] = (rS1[2 * i] - d0) * rP1[2 * i]
                            rS1[2 * i + 1] = (rS1[2 * i + 1] - d1) * rP1[2 * i + 1]
                        else:
                            t0, t1 = cute.arch.sub_packed_f32x2((rS1[2 * i], rS1[2 * i + 1]), (d0, d1))
                            rS1[2 * i], rS1[2 * i + 1] = cute.arch.mul_packed_f32x2((t0, t1), (rP1[2 * i], rP1[2 * i + 1]))
                    pack_bf16x2(rS1, rK1)
                    if cutlass.const_expr(not X_NOTST):
                        cute.copy(tiled_st, rK1, tdS_st1_p)
                    if cutlass.const_expr(not X_NOSTS):
                        for c8 in cutlass.range_constexpr(4):
                            p = cute.make_ptr(cutlass.Uint32, ds_row_base + (((c8 + 4) ^ r8) * 16), cute.AddressSpace.smem, assumed_align=16)
                            cute.arch.store(p, vec4u(rK1[4 * c8], rK1[4 * c8 + 1], rK1[4 * c8 + 2], rK1[4 * c8 + 3]))
                    cute.arch.fence_view_async_shared()
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.mbarrier_arrive(ds_ready)
                    trace(dbg, i32(1), i32(6), n_sm, tr_on)
                    n_sm = n_sm + 1
                    stage = stage + 1
                    if stage == STAGES:
                        stage = i32(0)
                        qdo_phase = qdo_phase ^ 1
            # ---- epilogue: dK / dV for this item
            if valid:
                wait_dbg(dkv_full, dkvf_phase, i32(23), dbg, bidx * 32 + warp_idx)
                dkvf_phase = dkvf_phase ^ 1
                kv_row_abs = seq_start + kb * BLK + lane_row
                kv_valid = (kb * BLK + lane_row) < seqlen
                cute.copy(tiled_ld, tdV_c0_p, rS0)
                cute.copy(tiled_ld, tdV_c1_p, rS1)
                cute.copy(tiled_ld, tdK_c0_p, rP0)
                cute.copy(tiled_ld, tdK_c1_p, rP1)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(dkv_empty)
                if cutlass.const_expr(DUMP):
                    if bidx == 0 and it == 0:
                        for i in cutlass.range_constexpr(32):
                            dbg[(2 * 128 + lane_row) * 128 + col0 + i] = rS0[i]
                            dbg[(2 * 128 + lane_row) * 128 + col0 + 32 + i] = rS1[i]
                            dbg[(3 * 128 + lane_row) * 128 + col0 + i] = rP0[i]
                            dbg[(3 * 128 + lane_row) * 128 + col0 + 32 + i] = rP1[i]
                if kv_valid and cutlass.const_expr(not SKIP_EPI):
                    gdv = cute.recast_ptr(mDV.iterator, dtype=cutlass.Uint32) + ((kv_row_abs * num_heads + h) * (D // 2) + col0 // 2)
                    gdk = cute.recast_ptr(mDK.iterator, dtype=cutlass.Uint32) + ((kv_row_abs * num_heads + h) * (D // 2) + col0 // 2)
                    pack_bf16x2(rS0, rK)
                    pack_bf16x2(rS1, rK1)
                    for c8 in cutlass.range_constexpr(4):
                        cute.arch.store(gdv + c8 * 4, vec4u(rK[4 * c8], rK[4 * c8 + 1], rK[4 * c8 + 2], rK[4 * c8 + 3]))
                        cute.arch.store(gdv + 16 + c8 * 4, vec4u(rK1[4 * c8], rK1[4 * c8 + 1], rK1[4 * c8 + 2], rK1[4 * c8 + 3]))
                    for i in cutlass.range_constexpr(32):
                        rP0[i] = rP0[i] * scale
                        rP1[i] = rP1[i] * scale
                    pack_bf16x2(rP0, rK)
                    pack_bf16x2(rP1, rK1)
                    for c8 in cutlass.range_constexpr(4):
                        cute.arch.store(gdk + c8 * 4, vec4u(rK[4 * c8], rK[4 * c8 + 1], rK[4 * c8 + 2], rK[4 * c8 + 3]))
                        cute.arch.store(gdk + 16 + c8 * 4, vec4u(rK1[4 * c8], rK1[4 * c8 + 1], rK1[4 * c8 + 2], rK1[4 * c8 + 3]))

    # ------------------------------------------------------------------ dQ drain warpgroups (WG0: even tiles, WG3: odd tiles)
    elif wg == 0 or wg == 3:
        my_par = wg // 3
        if cutlass.const_expr(not NO_REGALLOC):
            cute.arch.warpgroup_reg_dealloc(DRAIN_REGS)
        d_lane = tid_wg   # D index (TMEM lane)
        lane = tidx % 32
        r4 = lane % 4
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), f32)
        S_layout = cute.make_layout((BLK, 32), stride=(65536, 1))
        tdQ_c = [cute.make_tensor(tmem_ptr + (COL_DQ + 32 * c), S_layout) for c in range(4)]
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tdQ_c[0])
        thr_ld = tiled_ld.get_slice(tid_wg)
        tdQ_p = [thr_ld.partition_S(t) for t in tdQ_c]
        frag32 = cute.make_layout(((32, 1), 1, 1), stride=((1, 0), 0, 0))
        rQ = [cute.make_rmem_tensor(frag32, f32) for c in range(2)]
        rV4 = cute.make_rmem_tensor((4,), f32)
        rB8 = cute.make_rmem_tensor((8,), bf16)
        frag16d = cute.make_layout(((16, 1), 1, 1), stride=((1, 0), 0, 0))
        rKx0 = cute.make_rmem_tensor(frag16d, cutlass.Uint32)
        rKx1 = cute.make_rmem_tensor(frag16d, cutlass.Uint32)
        nb_drain = pipeline.NamedBarrier(barrier_id=NB_DRAIN, num_threads=128)
        dqf_phase = i32(0)
        qe_phase = i32(0)
        xph0 = i32(0)
        xph1 = i32(0)
        x_cnt = i32(0)
        stage = i32(0)
        HD = num_heads * D
        n_dr = i32(0)
        tr_on = bidx == 0 and tidx == 384
        sQ_base = sQ.iterator.toint()
        sdO_base = sdO.iterator.toint()
        for it in cutlass.range(n_my_items):
            item = item_base + it
            b = items[(item * 2 + rank, 0)]
            kb = items[(item * 2 + rank, 1)]
            h = items[(item * 2 + rank, 2)]
            paired = items[(item * 2 + rank, 3)]
            kb0 = kb - rank * paired
            valid = b >= 0
            if b < 0:
                b = i32(0)
            seq_start = cu_seqlens[b]
            seq_end = cu_seqlens[b + 1]
            seqlen = seq_end - seq_start
            nqb = (seqlen + BLK - 1) // BLK
            if not valid:
                nqb = kb
            n0 = nqb - kb0
            for qbi in cutlass.range(n0):
                sidx = (qbi + (kb0 % ROT)) % n0
                qb = nqb - 1 - sidx
                if qb >= kb:
                    q_row = seq_start + qb * BLK
                    do_xchg = (paired != 0) and (qb > kb0)
                    xp = x_cnt % 2
                    xph = xph0
                    if xp == 1:
                        xph = xph1
                    mine = (n_dr % 2) == my_par
                    if mine:
                        trace(dbg, i32(2), i32(0), n_dr, tr_on)
                        wait_dbg(dq_full + stage, dqf_phase, i32(22) + 1000 * n_dr, dbg, bidx * 32 + warp_idx)
                        dqf_phase = dqf_phase ^ 1
                        trace(dbg, i32(2), i32(1), n_dr, tr_on)
                        # (dK of this block finished before dQ, so the Q/dO stage buffers are already free)
                        wait_dbg(qdo_empty + stage, qe_phase, i32(25), dbg, bidx * 32 + warp_idx)
                        gbase = mDQacc.iterator + (h * D + (d_lane // 4) * 4)
                        cute.arch.mbarrier_arrive(ds_free + stage)
                        if cutlass.const_expr(NODRAIN):
                            do_xchg = False
                        if do_xchg:
                            # my dO stage buffer is the receive target for the peer's send: tell the peer its index
                            if tid_wg == 0:
                                cute.arch.mbarrier_arrive_and_expect_tx(peer_ready + xp, 4)
                                cute.arch.store_async_dsmem(xinfo + 4 * xp, cutlass.Int32(stage), peer_ready + xp, peer)
                            trace(dbg, i32(2), i32(3), n_dr, tr_on)
                            # ---- send the peer's half (q cols [64*peer, +64)): stage it locally right away
                            cute.copy(tiled_ld, thr_ld.partition_S(cute.make_tensor(tmem_ptr + (COL_DQ + 64 * peer), S_layout)), rQ[0])
                            cute.copy(tiled_ld, thr_ld.partition_S(cute.make_tensor(tmem_ptr + (COL_DQ + 64 * peer + 32), S_layout)), rQ[1])
                            cute.arch.fence_view_async_tmem_load()
                            s_q_stage = sQ_base + stage * (BLK * D * 2)
                            if cutlass.const_expr(BF16_XCHG):
                                # bf16 partials: 16B chunk m (q = 8m..8m+7 as bf16) of lane d at m*2048 + d*16 (16 KB total)
                                pack_bf16x2(rQ[0], rKx0)
                                pack_bf16x2(rQ[1], rKx1)
                                for m in cutlass.range_constexpr(8):
                                    pl = cute.make_ptr(cutlass.Uint32, s_q_stage + m * 2048 + d_lane * 16, cute.AddressSpace.smem, assumed_align=16)
                                    if cutlass.const_expr(m < 4):
                                        cute.arch.store(pl, vec4u(rKx0[4 * m], rKx0[4 * m + 1], rKx0[4 * m + 2], rKx0[4 * m + 3]))
                                    else:
                                        cute.arch.store(pl, vec4u(rKx1[4 * (m - 4)], rKx1[4 * (m - 4) + 1], rKx1[4 * (m - 4) + 2], rKx1[4 * (m - 4) + 3]))
                            else:
                                for m in cutlass.range_constexpr(16):
                                    c = m // 8
                                    i0 = (m % 8) * 4
                                    pl = cute.make_ptr(f32, s_q_stage + m * 2048 + d_lane * 16, cute.AddressSpace.smem, assumed_align=16)
                                    cute.arch.store(pl, vec4f(rQ[c][i0], rQ[c][i0 + 1], rQ[c][i0 + 2], rQ[c][i0 + 3]))
                            cute.arch.fence_view_async_shared()
                            # ---- my half: read it now so the TMEM dQ region is released as early as possible
                            cute.copy(tiled_ld, thr_ld.partition_S(cute.make_tensor(tmem_ptr + (COL_DQ + 64 * rank), S_layout)), rQ[0])
                            cute.copy(tiled_ld, thr_ld.partition_S(cute.make_tensor(tmem_ptr + (COL_DQ + 64 * rank + 32), S_layout)), rQ[1])
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.mbarrier_arrive(dq_empty)
                            wait_dbg(peer_ready + xp, xph, i32(26), dbg, bidx * 32 + warp_idx)
                            tstage = xinfo[4 * xp]
                            trace(dbg, i32(2), i32(4), n_dr, tr_on)
                            nb_drain.arrive_and_wait()
                            if tid_wg == 0:
                                dst = cute.arch.inline_ptx("mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};", write_only_types=[cutlass.Int32], read_only_args=[cutlass.Int32(sdO_base + tstage * (BLK * D * 2)), cutlass.Int32(peer)])
                                mb = cute.arch.inline_ptx("mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};", write_only_types=[cutlass.Int32], read_only_args=[cutlass.Int32((recv_full + xp).toint()), cutlass.Int32(peer)])
                                cute.arch.inline_ptx("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [{$r0}], [{$r1}], %d, [{$r2}];" % XCHG_BYTES,
                                                     read_only_args=[dst, cutlass.Int32(s_q_stage), mb])
                                cute.arch.mbarrier_arrive_and_expect_tx(recv_full + xp, XCHG_BYTES)
                            trace(dbg, i32(2), i32(5), n_dr, tr_on)
                            # ---- receive the peer's partial for my half (in my dO stage buffer) and add
                            wait_dbg(recv_full + xp, xph, i32(27), dbg, bidx * 32 + warp_idx)
                            trace(dbg, i32(2), i32(6), n_dr, tr_on)
                            if cutlass.const_expr(BF16_XCHG):
                                sRecvB = cute.make_tensor(cute.make_ptr(bf16, sdO_base + stage * (BLK * D * 2) + d_lane * 16, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((8, 8), stride=(1, 1024)))
                                for m in cutlass.range_constexpr(8):
                                    c = m // 4
                                    i0 = (m % 4) * 8
                                    cute.autovec_copy(sRecvB[(None, m)], rB8)
                                    for k in cutlass.range_constexpr(8):
                                        rQ[c][i0 + k] = rQ[c][i0 + k] + rB8[k].to(f32)
                            else:
                                sRecv = cute.make_tensor(cute.make_ptr(f32, sdO_base + stage * (BLK * D * 2) + d_lane * 16, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((4, 16), stride=(1, 512)))
                                for m in cutlass.range_constexpr(16):
                                    c = m // 8
                                    i0 = (m % 8) * 4
                                    cute.autovec_copy(sRecv[(None, m)], rV4)
                                    for k in cutlass.range_constexpr(4):
                                        rQ[c][i0 + k] = rQ[c][i0 + k] + rV4[k]
                            cute.arch.mbarrier_arrive(rx_done + stage)
                            nb_drain.arrive_and_wait()
                            if tid_wg == 0:
                                cute.arch.mbarrier_arrive(send_done + tstage, peer_cta_rank_in_cluster=peer)
                            trace(dbg, i32(2), i32(7), n_dr, tr_on)
                            if cutlass.const_expr(not SKIP_RED):
                                red_tile(rQ[0], rQ[1], gbase, q_row + 64 * rank, seq_end, HD, lane, r4, scale)
                        else:
                            for half in cutlass.range_constexpr(2):
                                cute.copy(tiled_ld, tdQ_p[2 * half], rQ[0])
                                cute.copy(tiled_ld, tdQ_p[2 * half + 1], rQ[1])
                                cute.arch.fence_view_async_tmem_load()
                                if half == 1:
                                    cute.arch.mbarrier_arrive(dq_empty)
                                if cutlass.const_expr(DUMP):
                                    if bidx == 0 and it == 0 and qb == kb:
                                        for c in cutlass.range_constexpr(2):
                                            for i in cutlass.range_constexpr(32):
                                                dbg[(4 * 128 + d_lane) * 128 + 64 * half + 32 * c + i] = rQ[c][i]
                                if cutlass.const_expr(not SKIP_RED):
                                    red_tile(rQ[0], rQ[1], gbase, q_row + 64 * half, seq_end, HD, lane, r4, scale)
                        trace(dbg, i32(2), i32(2), n_dr, tr_on)
                    if do_xchg:
                        x_cnt = x_cnt + 1
                        if xp == 0:
                            xph0 = xph0 ^ 1
                        else:
                            xph1 = xph1 ^ 1
                    n_dr = n_dr + 1
                    stage = stage + 1
                    if stage == STAGES:
                        stage = i32(0)
                        qe_phase = qe_phase ^ 1
    else:
        if cutlass.const_expr(not NO_REGALLOC):
            cute.arch.warpgroup_reg_dealloc(AUX_REGS)

    # teardown
    if wg == 3:
        if tid_wg == 0:
            cute.arch.inline_ptx("cp.async.bulk.wait_group 0;")
    if cutlass.const_expr(DEBUG_WAIT):
        dbg[bidx * 32 + warp_idx] = f32(99)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if cutlass.const_expr(DEBUG_WAIT):
        dbg[bidx * 32 + warp_idx] = f32(1099)
    if warp_idx == 1:
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import vector as _vector
SOFTMAX_REGS = int(os.environ.get("FA_SM_REGS", "144"))
AUX_REGS = int(os.environ.get("FA_AUX_REGS", "24"))
DRAIN_REGS = int(os.environ.get("FA_DR_REGS", "80"))
EXP_EMU = "noemu" not in DBG
EMU_DIV = int(os.environ.get("FA_EMU_DIV", "4"))
NOPACK = "nopack" in DBG
NOHOIST = "nohoist" in DBG
X_NOSTS = "nosts" in DBG
X_NOTST = "notst" in DBG
X_NOPACK = "nopk" in DBG
X_NOEXP = "noexp" in DBG
X_NODSF = "nodsf" in DBG
PACK_ASM = "packasm" in DBG
NODRAIN = "nodrain" in DBG
ROT = int(os.environ.get("FA_ROT", "1000000"))
SPIN_WAIT = "nospin" not in DBG
BF16_XCHG = "fp32x" not in DBG
XCHG_BYTES = 16384 if BF16_XCHG else 32768


def vec4u(a, b, c, d):
    vt = _ir.VectorType.get([4], cutlass.Uint32.mlir_type)
    return _vector.from_elements(vt, [a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()])


def vec4f(a, b, c, d):
    vt = _ir.VectorType.get([4], f32.mlir_type)
    return _vector.from_elements(vt, [a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()])


def vec2u(a, b):
    vt = _ir.VectorType.get([2], cutlass.Uint32.mlir_type)
    return _vector.from_elements(vt, [a.ir_value(), b.ir_value()])


@cute.jit
def _bf16x2(x0: f32, x1: f32) -> cutlass.Uint32:
    tb = cute.make_rmem_tensor((2,), bf16)
    tf = cute.make_rmem_tensor((2,), f32)
    tf[0] = x0
    tf[1] = x1
    tb.store(tf.load().to(bf16))
    return cute.recast_tensor(tb, cutlass.Uint32)[0]


@cute.jit
def exp2_emu(x: f32) -> f32:
    # 2^x for x <= ~0 via Cody-Waite + degree-3 polynomial on FMA pipes (relative error ~1e-4)
    xc = cute.arch.fmax(x, f32(-126.0))
    t = xc + f32(12582912.0)                      # 1.5 * 2^23 : rounds to nearest integer in the low mantissa bits
    n_i = t.to(cutlass.Int32, ) if False else cutlass.Int32(cute.arch.inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[cutlass.Int32], read_only_args=[t]))
    n_i = n_i - cutlass.Int32(0x4B400000)
    fpart = xc - (t - f32(12582912.0))             # in [-0.5, 0.5]
    p = f32(0.0555041086648216) * fpart + f32(0.240226506959101)
    p = p * fpart + f32(0.693147180559945)
    p = p * fpart + f32(1.0)
    pb = cutlass.Int32(cute.arch.inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[cutlass.Int32], read_only_args=[p]))
    rb = pb + (n_i << 23)
    return f32(cute.arch.inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[f32], read_only_args=[rb]))


@cute.jit
def sel4(v0: f32, v1: f32, v2: f32, v3: f32, idx: i32) -> f32:
    r = v0
    if idx == 1:
        r = v1
    if idx == 2:
        r = v2
    if idx == 3:
        r = v3
    return r


@cute.jit
def trace(dbg: cute.Tensor, role: i32, ev: i32, n: i32, on: cutlass.Boolean):
    if cutlass.const_expr(TRACE):
        if on:
            t = cute.arch.clock() & 0xFFFFFF
            dbg[(role * 8 + ev) * 64 + (n % 64)] = f32(t)


@cute.jit
def wait_dbg(bar: cute.Pointer, phase: i32, code: i32, dbg: cute.Tensor, slot: i32):
    if cutlass.const_expr(DEBUG_WAIT):
        dbg[slot] = f32(code)
        cute.arch.mbarrier_wait(bar, phase)
        dbg[slot] = f32(-code)
    elif cutlass.const_expr(SPIN_WAIT):
        done = cute.arch.mbarrier_try_wait(bar, phase)
        while done == False:
            done = cute.arch.mbarrier_try_wait(bar, phase)
    else:
        cute.arch.mbarrier_wait(bar, phase)


@cute.jit
def red_tile(rq0: cute.Tensor, rq1: cute.Tensor, gbase: cute.Pointer, qrow0: i32, seq_end: i32, HD: i32, lane: i32, r4: i32, scale: f32):
    """rq0/rq1: 64 q values (chunk c, index i -> q = qrow0 + 32c + i) of D lane d for this thread.
    Quad (4-lane) transpose via 2 rounds of shfl.xor, then red.global.add.v4.f32 per (row, 4 consecutive D).
    After the transpose, lane with r4 = r holds row (4m + r) for D = 4*(d//4) + 0..3."""
    b0 = (r4 & 1) != 0
    b1 = (r4 & 2) != 0
    for c in cutlass.range_constexpr(2):
        for m in cutlass.range_constexpr(8):
            if cutlass.const_expr(c == 0):
                a0 = rq0[4 * m + 0]
                a1 = rq0[4 * m + 1]
                a2 = rq0[4 * m + 2]
                a3 = rq0[4 * m + 3]
            else:
                a0 = rq1[4 * m + 0]
                a1 = rq1[4 * m + 1]
                a2 = rq1[4 * m + 2]
                a3 = rq1[4 * m + 3]
            # round 1: swap with lane^1 : lane with b0=0 sends a1,a3 and receives partner's a0,a2 into a1,a3 slots
            s0 = a0
            s1 = a2
            if b0:
                s0 = a1
                s1 = a3
            # partner (b0 flipped) sends its (a1,a3) if it has b0=0, i.e. we receive partner's a1/a3 when we have b0=1... 
            # symmetric formulation: send x = b0 ? a0 : a1 ; receive y from lane^1 ; then a0/a1 := b0 ? (y, a1) : (a0, y)
            t0 = a1
            if b0:
                t0 = a0
            t1 = a3
            if b0:
                t1 = a2
            y0 = cute.arch.shuffle_sync_bfly(t0, 1)
            y1 = cute.arch.shuffle_sync_bfly(t1, 1)
            if b0:
                a0 = y0
                a2 = y1
            else:
                a1 = y0
                a3 = y1
            # round 2: swap with lane^2 : send b1 ? a0 : a2 (and b1 ? a1 : a3), receive into a0/a1 (b1) or a2/a3 (!b1)
            t2 = a2
            if b1:
                t2 = a0
            t3 = a3
            if b1:
                t3 = a1
            y2 = cute.arch.shuffle_sync_bfly(t2, 2)
            y3 = cute.arch.shuffle_sync_bfly(t3, 2)
            if b1:
                a0 = y2
                a1 = y3
            else:
                a2 = y2
                a3 = y3
            row = qrow0 + 32 * c + 4 * m + r4
            if row < seq_end:
                p = gbase + row * HD
                if cutlass.const_expr(DQ_BF16):
                    cute.arch.red(p, vec2u(_bf16x2(a0 * scale, a1 * scale), _bf16x2(a2 * scale, a3 * scale)), op="add", dtype="bf16x2", sem="relaxed", scope="gpu")
                else:
                    cute.arch.red(p, vec4f(a0, a1, a2, a3), op="add", dtype="f32", sem="relaxed", scope="gpu")


@cute.jit
def release_prev(have_prev: i32, prev_xchg: i32, prev_xphase: i32, prev_stage: i32, send_done: cute.Pointer, stage_free: cute.Pointer, dbg: cute.Tensor, slot: i32):
    # called by the drain's thread 0 right after committing this iteration's bulk group:
    # wait until the previous group's smem reads are done (allow the newest group in flight), then release its stage.
    # (the peer's consumption of the previous send buffer was already awaited at the start of this iteration)
    if have_prev > 0:
        cute.arch.inline_ptx("cp.async.bulk.wait_group.read 1;")
        cute.arch.mbarrier_arrive(stage_free + prev_stage)


def pack_bf16x2(src, dst):
    # trace-time inlined: src: 32 fp32 (rmem tensor) -> dst: 16 uint32 regs holding bf16x2 (low half = even index)
    if X_NOPACK:
        for i in range(16):
            dst[i] = cutlass.Uint32(cute.arch.inline_ptx("mov.b32 {$w0}, {$r0};", write_only_types=[cutlass.Int32], read_only_args=[src[2 * i]]))
        return
    if PACK_ASM:
        for i in range(16):
            dst[i] = cute.arch.inline_ptx(
                "cvt.rn.satfinite.bf16x2.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[cutlass.Uint32],
                read_only_args=[src[2 * i + 1], src[2 * i]],
            )
        return
    # native conversion: vector truncf to bf16, then reinterpret pairs as u32
    v = src.load().to(bf16)
    tb = cute.make_rmem_tensor((32,), bf16)
    tb.store(v)
    tu = cute.recast_tensor(tb, cutlass.Uint32)
    for i in range(16):
        dst[i] = tu[i]


@cute.jit
def host_launch(
    mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
    mLSE: cute.Tensor, mDelta: cute.Tensor, mDQacc: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor,
    cu_seqlens: cute.Tensor, items: cute.Tensor, cta_offsets: cute.Tensor,
    scale_log2e: f32, scale: f32, total_T: i32, num_heads: i32, grid: i32, dbg: cute.Tensor,
):
    mma_S = sm100_utils.make_trivial_tiled_mma(bf16, bf16, OperandMajorMode.K, OperandMajorMode.K, f32, tcgen05.CtaGroup.ONE, (BLK, BLK), tcgen05.OperandSource.SMEM)
    mma_dV = sm100_utils.make_trivial_tiled_mma(bf16, bf16, OperandMajorMode.K, OperandMajorMode.MN, f32, tcgen05.CtaGroup.ONE, (BLK, BLK), tcgen05.OperandSource.TMEM)
    mma_dQ = sm100_utils.make_trivial_tiled_mma(bf16, bf16, OperandMajorMode.MN, OperandMajorMode.MN, f32, tcgen05.CtaGroup.ONE, (BLK, BLK), tcgen05.OperandSource.SMEM)
    # smem layouts (plain 2D/3D, SW128 swizzled). MN views describe the same bytes as the K-major ones.
    k_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, bf16)
    mn_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW128, bf16)
    kv_layout_k = cute.tile_to_shape(k_atom, (BLK, D), order=(0, 1))                 # (kv, D)
    kv_layout_mn = cute.tile_to_shape(mn_atom, (D, BLK), order=(1, 0))               # (D, kv) view
    q_layout_k = cute.tile_to_shape(k_atom, (BLK, D, STAGES), order=(0, 1, 2))       # (q, D, stage)
    q_layout_mn = cute.tile_to_shape(mn_atom, (D, BLK, STAGES), order=(1, 0, 2))     # (D, q, stage) view
    ds_layout = cute.tile_to_shape(mn_atom, (BLK, BLK), order=(1, 0))                # (q, kv), q contiguous
    # gmem tensors (T,H,D) -> (T, D, H) views for TMA tiles of (BLK, D)
    mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, [0, 2, 1]))
    mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, [0, 2, 1]))
    mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, [0, 2, 1]))
    mdO = cute.make_tensor(mdO.iterator, cute.select(mdO.layout, [0, 2, 1]))
    op = cpasync.CopyBulkTensorTileG2SOp()
    kv_stage = kv_layout_k
    q_stage = cute.slice_(q_layout_k, (None, None, 0))
    tma_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, q_stage, (BLK, D))
    tma_do, tdO = cpasync.make_tiled_tma_atom(op, mdO, q_stage, (BLK, D))
    tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, kv_stage, (BLK, D))
    tma_v, tV = cpasync.make_tiled_tma_atom(op, mV, kv_stage, (BLK, D))
    fa_bwd_kernel(
        mma_S, mma_dV, mma_dQ,
        tma_q, tQ, tma_k, tK, tma_v, tV, tma_do, tdO,
        mLSE, mDelta, mDQacc, mDK, mDV, cu_seqlens, items, cta_offsets,
        scale_log2e, scale, total_T, num_heads, dbg,
        kv_layout_k, kv_layout_mn, q_layout_k, q_layout_mn, ds_layout,
    ).launch(grid=[grid, 1, 1], block=[NUM_THREADS, 1, 1], cluster=[2, 1, 1])
