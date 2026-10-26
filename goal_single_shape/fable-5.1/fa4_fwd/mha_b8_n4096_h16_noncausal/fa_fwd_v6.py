"""
FlashAttention forward (non-causal, bf16, hdim=128) for Blackwell SM100 -- v6: 2-CTA, two S tiles per CTA,
each tile's softmax split by key columns across the two softmax warpgroups (two warps per SM sub-partition
work on the same rows), speculative exps with the running max, row-sum overflow guard (no per-block row max).

A cluster of 2 CTAs cooperates on 512 query rows of one (batch, head):
  * tcgen05.mma with cta_group::2 (M=256): each CTA supplies 128 rows of Q (A operand) and half of the
    K / V block (B operand), so every K/V block is loaded only once per cluster (half per CTA).
  * S = Q K^T lands in each CTA's own TMEM (128 rows x 128 keys per softmax warpgroup).
  * P (bf16, packed) is written back into TMEM (aliasing S) and consumed as the A operand of O += P V.
Warp roles per CTA (384 threads):
  warp 0  : TMA producer (Q once per unit, K/V through pipelines)
  warp 1  : MMA issuer (leader CTA only)
  warps 4-7 / 8-11 : softmax warpgroups 0 / 1 (128 query rows each)
TMEM (512 cols): O0[0:128) O1[128:256) S0[256:384) S1[384:512); P_wg aliases S_wg[0:64).
"""
import math
import os
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

BM = 128          # q rows per softmax warpgroup (per CTA)
NWG = 2           # softmax warpgroups per CTA
BN = 128          # keys per block
HD = 128          # head dim
CLUSTER = 2
UNIT_ROWS = BM * NWG * CLUSTER   # 512 q rows per work unit (per cluster)
K_STAGES = int(os.environ.get("FA_K_STAGES", "4"))
V_STAGES = int(os.environ.get("FA_V_STAGES", "4"))
THREADS = 128 * (1 + NWG)
TMA_WARP = 0
MMA_WARP = 1
TMEM_COLS = 512
O_COL = 0
S_COL = 256
RESCALE_THRESH = 8.0
SUM_LIMIT = 2.0 ** 24   # row-sum guard: if the partial row sum exceeds it, recompute the block with an updated max
NEG_BIG = -1.0e30
# exp2 emulation (Cody-Waite + degree-3 polynomial on [-0.5, 0.5])
EXP_MAGIC = 12582912.0          # 1.5 * 2^23
EXP_MAGIC_BITS = 0x4B400000
EXP_CLAMP_BITS = EXP_MAGIC_BITS - 126
EXP_C0, EXP_C1, EXP_C2, EXP_C3 = 0.999928280583146, 0.6932609262753286, 0.2426079019998498, 0.055170250079513636
EMU_EVERY = int(os.environ.get("FA_EMU_EVERY", "3"))
EMU_NUM = int(os.environ.get("FA_EMU_NUM", "1"))   # emulate pairs i with (i % EMU_EVERY) < EMU_NUM   # emulate exp2 for 1 of every EMU_EVERY pairs (0 = never)
TRUNC_MEAN_REL = 2.0 ** -8 * (0.5 / math.log(2.0))   # mean relative error of bf16 truncation (log-uniform mantissa)
TRUNC_DELTA = math.log2(1.0 + TRUNC_MEAN_REL)         # pre-scale so that truncation is unbiased on average
ABLATE = os.environ.get("FA_ABLATE", "")
TRACE = os.environ.get("FA_TRACE", "") == "1"
TRACE_UNIT_K = int(os.environ.get("FA_TRACE_UNIT", "0"))
NOMAX = os.environ.get("FA_NOMAX", "") == "1"
NOSUM = os.environ.get("FA_NOSUM", "") == "1"
NO_MMA = "nomma" in ABLATE
NO_SOFTMAX = "nosoftmax" in ABLATE
SOFTMAX_REGS = int(os.environ.get("FA_SOFTMAX_REGS", "232"))
XCHG_BAR = 2
OSTAGE_BAR = 3
OSTRIDE = 33   # padded row stride (words) of the epilogue staging buffer: 32 data words + 1 pad -> conflict-free
OTHER_REGS = int(os.environ.get("FA_OTHER_REGS", "40"))

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32


@cute.struct
class SharedStorage:
    k_mbar: cute.struct.MemRange[cutlass.Int64, K_STAGES * 2]
    v_mbar: cute.struct.MemRange[cutlass.Int64, V_STAGES * 2]
    q_full: cutlass.Int64
    q_empty: cutlass.Int64
    s_full: cute.struct.MemRange[cutlass.Int64, NWG]
    p_full: cute.struct.MemRange[cutlass.Int64, NWG]
    pv_done: cute.struct.MemRange[cutlass.Int64, NWG]
    o_empty: cute.struct.MemRange[cutlass.Int64, NWG]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32
    xval: cute.struct.MemRange[cutlass.Float32, NWG * BM]     # [wg][row]
    ostage: cute.struct.MemRange[cutlass.Int32, NWG * BM * OSTRIDE]   # [wg][row][33 words]


def _ex2(x):
    return cute.arch.inline_ptx("ex2.approx.ftz.f32 {$w0}, {$r0};", write_only_types=[F32], read_only_args=[x])


def _max3(a, b, c):
    return cute.arch.inline_ptx("max.f32 {$w0}, {$r0}, {$r1}, {$r2};", write_only_types=[F32], read_only_args=[a, b, c])


def _row_max(sv):
    n = len(sv)
    acc = [sv[0], sv[1], sv[2], sv[3]]
    idx = 4
    ch = 0
    while idx + 1 < n:
        acc[ch] = _max3(acc[ch], sv[idx], sv[idx + 1])
        idx += 2
        ch = (ch + 1) % 4
    if idx < n:
        acc[0] = cute.arch.fmax(acc[0], sv[idx])
    return _max3(cute.arch.fmax(acc[0], acc[1]), acc[2], acc[3])


def _exp2_poly2(x2):
    """Emulated exp2 of a pair (Float32, Float32) using FMA-pipe + integer ops only."""
    mg = (F32(EXP_MAGIC), F32(EXP_MAGIC))
    t = cute.arch.add_packed_f32x2(x2, mg)
    nf = cute.arch.sub_packed_f32x2(t, mg)
    f = cute.arch.sub_packed_f32x2(x2, nf)
    p = cute.arch.fma_packed_f32x2(f, (F32(EXP_C3), F32(EXP_C3)), (F32(EXP_C2), F32(EXP_C2)))
    p = cute.arch.fma_packed_f32x2(f, p, (F32(EXP_C1), F32(EXP_C1)))
    p = cute.arch.fma_packed_f32x2(f, p, (F32(EXP_C0), F32(EXP_C0)))
    out = []
    for h in range(2):
        tb = F32(t[h]).bitcast(I32)
        e = cutlass.min(cutlass.max(tb, I32(EXP_CLAMP_BITS)), I32(EXP_MAGIC_BITS + 127))
        r = e * I32(1 << 23) + F32(p[h]).bitcast(I32)
        out.append(r.bitcast(F32))
    return out[0], out[1]


def _exp_pass_chunks(sv, scale_log2, neg_m, tPr_i32, p_tiled, tPr, tPt_ch, ncw):
    """p = exp2(s*scale - m) over python list sv (2*ncw values); each ncw-chunk packed to bf16x2 and stored to TMEM."""
    c2 = (scale_log2, scale_log2)
    nm2 = (neg_m, neg_m)
    accs = [(F32(0.0), F32(0.0)) for _ in range(4)]
    for c in range(len(sv) // ncw):
        for i in range(ncw // 2):
            e = c * ncw + 2 * i
            x2 = cute.arch.fma_packed_f32x2((sv[e], sv[e + 1]), c2, nm2)
            if EMU_EVERY > 0 and (i % EMU_EVERY) == (EMU_EVERY - 1):
                p0, p1 = _exp2_poly2(x2)
            else:
                p0 = _ex2(F32(x2[0]))
                p1 = _ex2(F32(x2[1]))
            accs[i % 4] = cute.arch.add_packed_f32x2(accs[i % 4], (p0, p1))
            tPr_i32[i] = I32(cute.arch.prmt(p0.bitcast(I32), p1.bitcast(I32), 0x7632))
        cute.copy(p_tiled, tPr, tPt_ch[c])
    s01 = cute.arch.add_packed_f32x2(accs[0], accs[1])
    s23 = cute.arch.add_packed_f32x2(accs[2], accs[3])
    s03 = cute.arch.add_packed_f32x2(s01, s23)
    return F32(s03[0]) + F32(s03[1])


def _exp_pass(sv, scale_log2, neg_m, tPr_i32):
    """p = exp2(s*scale - m) over a python list of Float32 (pairs), packed to bf16x2 into tPr_i32; returns row sum."""
    c2 = (scale_log2, scale_log2)
    nm2 = (neg_m, neg_m)
    accs = [(F32(0.0), F32(0.0)) for _ in range(4)]
    for i in range(len(sv) // 2):
        x2 = cute.arch.fma_packed_f32x2((sv[2 * i], sv[2 * i + 1]), c2, nm2)
        if EMU_EVERY > 0 and (i % EMU_EVERY) < EMU_NUM:
            p0, p1 = _exp2_poly2(x2)
        else:
            p0 = _ex2(F32(x2[0]))
            p1 = _ex2(F32(x2[1]))
        accs[i % 4] = cute.arch.add_packed_f32x2(accs[i % 4], (p0, p1))
        tPr_i32[i] = I32(cute.arch.prmt(p0.bitcast(I32), p1.bitcast(I32), 0x7632))
    s01 = cute.arch.add_packed_f32x2(accs[0], accs[1])
    s23 = cute.arch.add_packed_f32x2(accs[2], accs[3])
    s03 = cute.arch.add_packed_f32x2(s01, s23)
    return F32(s03[0]) + F32(s03[1])


def _bar_red_or(flag_i32, bar_id, nthreads):
    """barrier.red.or over nthreads: returns Int32 1 if any participating thread passed a nonzero flag."""
    return cute.arch.inline_ptx(
        "{ .reg .pred p, q; setp.ne.s32 p, {$r0}, 0; bar.red.or.pred q, %d, %d, p; selp.s32 {$w0}, 1, 0, q; }" % (bar_id, nthreads),
        write_only_types=[I32], read_only_args=[flag_i32])


def _cvt_bf16x2(lo, hi):
    """round-to-nearest pack of two f32 into one bf16x2 word (lo -> low half)."""
    return cute.arch.inline_ptx("cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};", write_only_types=[I32], read_only_args=[hi, lo])


def _raw_pass(sv, tPr_i32):
    for i in range(len(sv) // 2):
        tPr_i32[i] = I32(cute.arch.prmt(sv[2 * i].bitcast(I32), sv[2 * i + 1].bitcast(I32), 0x7632))
    return F32(1.0) + F32(0.0) * sv[0]


def _load_list(t, n):
    return [t[i] for i in range(n)]


@cute.kernel
def fa_fwd_kernel(
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV: cute.Tensor,
    mO: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    q_smem_layout: cute.ComposedLayout,
    k_smem_layout: cute.ComposedLayout,
    v_smem_layout: cute.ComposedLayout,
    p_smem_layout: cute.ComposedLayout,
    scale_log2: cutlass.Float32,
    num_units: cutlass.Int32,
    nq_units: cutlass.Int32,
    num_heads: cutlass.Int32,
    n_blocks: cutlass.Int32,
    dbg: cute.Tensor,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader = cta_rank == 0
    cluster_id = bidx // CLUSTER
    num_clusters = gdim // CLUSTER
    cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)

    qk_tiler = (BM * CLUSTER, BN, HD)
    pv_tiler = (BM * CLUSTER, HD, BN)

    # ---------------- shared memory ----------------
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = smem.allocate_tensor(BF16, q_smem_layout.outer, 128, q_smem_layout.inner)
    sK = smem.allocate_tensor(BF16, k_smem_layout.outer, 128, k_smem_layout.inner)
    sV = smem.allocate_tensor(BF16, v_smem_layout.outer, 128, v_smem_layout.inner)

    q_full = storage.q_full
    q_empty = storage.q_empty
    s_full = storage.s_full.data_ptr()
    p_full = storage.p_full.data_ptr()
    pv_done = storage.pv_done.data_ptr()
    o_empty = storage.o_empty.data_ptr()

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_q)
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_v)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_full.ptr, 1)
            cute.arch.mbarrier_init(q_empty.ptr, 1)
            for wg in cutlass.range_constexpr(NWG):
                cute.arch.mbarrier_init(s_full + wg, 1)
                cute.arch.mbarrier_init(p_full + wg, 128 * NWG * CLUSTER)
                cute.arch.mbarrier_init(pv_done + wg, 1)
                cute.arch.mbarrier_init(o_empty + wg, 128 * NWG * CLUSTER)

    k_bytes = cute.size_in_bytes(BF16, cute.select(k_smem_layout, mode=[0, 1, 2])) * CLUSTER
    v_bytes = cute.size_in_bytes(BF16, cute.select(v_smem_layout, mode=[0, 1, 2])) * CLUSTER
    q_bytes = cute.size_in_bytes(BF16, q_smem_layout) * CLUSTER

    k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=K_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=k_bytes,
        barrier_storage=storage.k_mbar.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    ).make_participants()
    v_producer, v_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=V_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=v_bytes,
        barrier_storage=storage.v_mbar.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    ).make_participants()

    # ---------------- TMEM ----------------
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=TMA_WARP,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

    thr_mma_qk = tiled_mma_qk.get_slice(cta_rank)
    thr_mma_pv = tiled_mma_pv.get_slice(cta_rank)

    # accumulator layouts (per CTA: 128 x 128)
    s_acc_shape = tiled_mma_qk.partition_shape_C((BM * CLUSTER, BN))
    tStS_fake = tiled_mma_qk.make_fragment_C(s_acc_shape)
    o_acc_shape = tiled_mma_pv.partition_shape_C((BM * CLUSTER, HD))
    tOtO_fake = tiled_mma_pv.make_fragment_C(o_acc_shape)
    p_frag_layout = tiled_mma_pv.make_fragment_A(cute.select(p_smem_layout, mode=[0, 1, 2]).outer).layout

    # ---------------- global tiles ----------------
    gQ = cute.local_tile(mQ, cute.slice_(qk_tiler, (None, 0, None)), (None, None, None))
    gK = cute.local_tile(mK, cute.slice_(qk_tiler, (0, None, None)), (None, None, None))
    gV = cute.local_tile(mV, cute.slice_(pv_tiler, (0, None, None)), (None, None, None))
    tCgQ = thr_mma_qk.partition_A(gQ)
    tCgK = thr_mma_qk.partition_B(gK)
    tCgV = thr_mma_pv.partition_B(gV)

    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    tQsQ, tQgQ = cpasync.tma_partition(
        tma_atom_q, cta_coord_vmnk[2], a_cta_layout, cute.group_modes(sQ, 0, 3), cute.group_modes(tCgQ, 0, 3)
    )
    tKsK, tKgK = cpasync.tma_partition(
        tma_atom_k, cta_coord_vmnk[1], b_cta_layout, cute.group_modes(sK, 0, 3), cute.group_modes(tCgK, 0, 3)
    )
    tVsV, tVgV = cpasync.tma_partition(
        tma_atom_v, cta_coord_vmnk[1], b_cta_layout, cute.group_modes(sV, 0, 3), cute.group_modes(tCgV, 0, 3)
    )
    q_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=2)
    kv_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=1)

    tCrQ = tiled_mma_qk.make_fragment_A(sQ)  # (MMA, MMA_M, MMA_K, NWG)
    tCrK = tiled_mma_qk.make_fragment_B(sK)  # (MMA, MMA_N, MMA_K, STAGE)
    tCrV = tiled_mma_pv.make_fragment_B(sV)  # (MMA, MMA_N, MMA_K, STAGE)

    pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    tmem.allocate(TMEM_COLS)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)

    tStS = [cute.make_tensor(tmem_ptr + (S_COL + 128 * wg), tStS_fake.layout) for wg in range(NWG)]
    tOtO = [cute.make_tensor(tmem_ptr + (O_COL + 128 * wg), tOtO_fake.layout) for wg in range(NWG)]
    tOrP = [
        cute.make_tensor(cute.recast_ptr(tmem_ptr + (S_COL + 128 * wg), dtype=BF16), p_frag_layout)
        for wg in range(NWG)
    ]

    both_mask = (1 << CLUSTER) - 1

    # =====================================================================
    # Warpgroup 0: TMA warp + MMA warp
    # =====================================================================
    if warp_idx < 4:
        cute.arch.warpgroup_reg_dealloc(OTHER_REGS)
        if warp_idx == TMA_WARP:
            q_empty_phase = cutlass.Int32(1)
            for unit in cutlass.range(cluster_id, num_units, num_clusters):
                qt = unit % nq_units
                hb = unit // nq_units
                h = hb % num_heads
                b = hb // num_heads
                cute.arch.mbarrier_wait(q_empty.ptr, q_empty_phase)
                q_empty_phase = q_empty_phase ^ 1
                if is_leader:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(q_full.ptr, q_bytes)
                for wg in cutlass.range_constexpr(NWG):
                    cute.copy(
                        tma_atom_q,
                        tQgQ[(None, qt * NWG + wg, 0, (h, b))],
                        tQsQ[(None, wg)],
                        tma_bar_ptr=q_full.ptr,
                        mcast_mask=q_mcast_mask,
                    )
                for j in cutlass.range(n_blocks):
                    kh = k_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k, tKgK[(None, j, 0, (h, b))], tKsK[(None, kh.index)],
                        tma_bar_ptr=kh.barrier, mcast_mask=kv_mcast_mask,
                    )
                    vh = v_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v, tVgV[(None, 0, j, (h, b))], tVsV[(None, vh.index)],
                        tma_bar_ptr=vh.barrier, mcast_mask=kv_mcast_mask,
                    )
            k_producer.tail()
            v_producer.tail()

        if warp_idx == MMA_WARP:
            if is_leader:
                q_full_phase = cutlass.Int32(0)
                o_empty_phase = cutlass.Int32(1)
                p_phase = cutlass.Int32(0)
                for unit in cutlass.range(cluster_id, num_units, num_clusters):
                    cute.arch.mbarrier_wait(q_full.ptr, q_full_phase)
                    q_full_phase = q_full_phase ^ 1
                    for wg in cutlass.range_constexpr(NWG):
                        cute.arch.mbarrier_wait(o_empty + wg, o_empty_phase)
                    o_empty_phase = o_empty_phase ^ 1

                    # prologue: S_0 for both warpgroups
                    kh = k_consumer.wait_and_advance()
                    for wg in cutlass.range_constexpr(NWG):
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(0 if NO_MMA else cute.size(tCrQ, mode=[2])):
                            cute.gemm(
                                tiled_mma_qk, tStS[wg],
                                tCrQ[(None, None, kb, wg)], tCrK[(None, None, kb, kh.index)], tStS[wg],
                            )
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(s_full + wg, both_mask, tcgen05.CtaGroup.TWO)
                    kh.release()

                    for j in cutlass.range(n_blocks - 1):
                        vh = v_consumer.wait_and_advance()
                        for wg in cutlass.range_constexpr(NWG):
                            cute.arch.mbarrier_wait(p_full + wg, p_phase)
                            if cutlass.const_expr(TRACE):
                                if bidx == 0:
                                    if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                        dbg[(j, 0 + 2 * wg)] = cute.arch.clock64()
                            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, j > 0)
                            for kb in cutlass.range_constexpr(0 if NO_MMA else cute.size(tOrP[wg], mode=[2])):
                                cute.gemm(
                                    tiled_mma_pv, tOtO[wg],
                                    tOrP[wg][(None, None, kb)], tCrV[(None, None, kb, vh.index)], tOtO[wg],
                                )
                                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(pv_done + wg, both_mask, tcgen05.CtaGroup.TWO)
                            # next S for this warpgroup
                            if cutlass.const_expr(wg == 0):
                                kh = k_consumer.wait_and_advance()
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(0 if NO_MMA else cute.size(tCrQ, mode=[2])):
                                cute.gemm(
                                    tiled_mma_qk, tStS[wg],
                                    tCrQ[(None, None, kb, wg)], tCrK[(None, None, kb, kh.index)], tStS[wg],
                                )
                                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(s_full + wg, both_mask, tcgen05.CtaGroup.TWO)
                            if cutlass.const_expr(TRACE):
                                if bidx == 0:
                                    if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                        dbg[(j, 1 + 2 * wg)] = cute.arch.clock64()
                        kh.release()
                        vh.release()
                        p_phase = p_phase ^ 1

                    # last block: PV only
                    vh = v_consumer.wait_and_advance()
                    for wg in cutlass.range_constexpr(NWG):
                        cute.arch.mbarrier_wait(p_full + wg, p_phase)
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, n_blocks > 1)
                        for kb in cutlass.range_constexpr(0 if NO_MMA else cute.size(tOrP[wg], mode=[2])):
                            cute.gemm(
                                tiled_mma_pv, tOtO[wg],
                                tOrP[wg][(None, None, kb)], tCrV[(None, None, kb, vh.index)], tOtO[wg],
                            )
                            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(pv_done + wg, both_mask, tcgen05.CtaGroup.TWO)
                    vh.release()
                    p_phase = p_phase ^ 1
                    with cute.arch.elect_one():
                        tcgen05.commit(q_empty.ptr, both_mask, tcgen05.CtaGroup.TWO)

    # =====================================================================
    # Softmax warpgroups: wg = column half (0: keys 0..63, 1: keys 64..127) of BOTH tiles
    # =====================================================================
    if warp_idx >= 4:
        cute.arch.warpgroup_reg_alloc(SOFTMAX_REGS)
        wg = cute.arch.make_warp_uniform((warp_idx - 4) // 4)
        ltid = tidx - 128 * (wg + 1)
        CW = BN // NWG          # 64 key columns per warpgroup
        OW = HD // NWG          # 64 O columns per warpgroup

        s_ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), F32)
        o_ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), F32)
        o_st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x64), F32)
        p_st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), BF16)

        tS_full = [cute.make_tensor(tmem_ptr + (S_COL + 128 * t), tStS_fake.layout) for t in range(NWG)]
        tO_full = [cute.make_tensor(tmem_ptr + (O_COL + 128 * t), tOtO_fake.layout) for t in range(NWG)]
        tP_full = [cute.make_tensor(cute.recast_ptr(tmem_ptr + (S_COL + 128 * t), dtype=BF16), p_frag_layout) for t in range(NWG)]
        tS_h = [cute.zipped_divide(tS_full[t], ((BM, CW),))[(None, ((0, wg), 0, 0))] for t in range(NWG)]
        tO_h = [cute.zipped_divide(tO_full[t], ((BM, OW),))[(None, ((0, wg), 0, 0))] for t in range(NWG)]
        tP_h = [tP_full[t][(None, None, (None, wg))] for t in range(NWG)]

        s_tiled = tcgen05.make_tmem_copy(s_ld_atom, tS_h[0])
        o_ld_tiled = tcgen05.make_tmem_copy(o_ld_atom, tO_h[0])
        o_st_tiled = tcgen05.make_tmem_copy(o_st_atom, tO_h[0])
        p_tiled = tcgen05.make_tmem_copy(p_st_atom, tP_h[0])
        s_thr = s_tiled.get_slice(ltid)
        o_ld_thr = o_ld_tiled.get_slice(ltid)
        o_st_thr = o_st_tiled.get_slice(ltid)
        p_thr = p_tiled.get_slice(ltid)

        cS = cute.make_identity_tensor(tS_h[0].shape)
        cO = cute.make_identity_tensor(tO_h[0].shape)
        cP = cute.make_identity_tensor(tP_h[0].shape)
        tSt = [s_thr.partition_S(tS_h[t]) for t in range(NWG)]
        tOt_ld = [o_ld_thr.partition_S(tO_h[t]) for t in range(NWG)]
        tOt_st = [o_st_thr.partition_D(tO_h[t]) for t in range(NWG)]
        tPt = [p_thr.partition_D(tP_h[t]) for t in range(NWG)]
        tSr = cute.make_rmem_tensor(s_thr.partition_D(cS).shape, F32)
        tOr = cute.make_rmem_tensor(o_ld_thr.partition_D(cO).shape, F32)
        tPr = cute.make_rmem_tensor(p_thr.partition_S(cP).shape, BF16)
        tSr_flat = cute.make_tensor(tSr.iterator, cute.make_layout(CW))
        tOr_flat = cute.make_tensor(tOr.iterator, cute.make_layout(OW))
        tOut32 = cute.make_rmem_tensor(cute.make_layout(OW // 2), I32)
        tOut4 = cute.make_rmem_tensor(cute.make_layout(4), I32)
        ostage_t = cute.make_tensor(storage.ostage.data_ptr(), cute.make_layout(NWG * BM * OSTRIDE))
        tPr_i32 = cute.make_tensor(cute.recast_ptr(tPr.iterator, dtype=I32), cute.make_layout(CW // 2))

        xval_t = cute.make_tensor(storage.xval.data_ptr(), cute.make_layout(NWG * BM))

        s_phase = cutlass.Int32(0)
        for unit in cutlass.range(cluster_id, num_units, num_clusters):
            qt = unit % nq_units
            hb = unit // nq_units
            h = hb % num_heads
            b = hb // num_heads
            m_a = cutlass.Float32(NEG_BIG)
            m_b = cutlass.Float32(NEG_BIG)
            l_a = cutlass.Float32(0.0)
            l_b = cutlass.Float32(0.0)
            if cutlass.const_expr(TRACE):
                if bidx == 0:
                    if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                        if ltid == 0:
                            dbg[(0, 14 + wg)] = cute.arch.clock64()
            for j in cutlass.range(n_blocks):
                for t in cutlass.range_constexpr(NWG):
                    m_t = m_a if t == 0 else m_b
                    l_t = l_a if t == 0 else l_b
                    cute.arch.mbarrier_wait(s_full + t, s_phase)
                    if cutlass.const_expr(TRACE):
                        if bidx == 0:
                            if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                if ltid == 0:
                                    dbg[(j, 4 + 4 * t + 2 * wg)] = cute.arch.clock64()
                    cute.copy(s_tiled, tSt[t], tSr)
                    cute.arch.fence_view_async_tmem_load()
                    sv = _load_list(tSr_flat, CW)
                    # speculative pass with the running max (no row max in the common path)
                    alpha = F32(1.0)
                    m_next = m_t
                    rowsum = F32(0.0)
                    if j == 0:
                        # first block of the unit: exact row max (exchanged with the other warpgroup), no rescale
                        mx = _row_max(sv)
                        xval_t[wg * BM + ltid] = mx
                        cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)
                        mx = cute.arch.fmax(mx, xval_t[(1 - wg) * BM + ltid])
                        m_use = mx * scale_log2
                        m_next = m_use
                        rowsum = _exp_pass(sv, scale_log2, TRUNC_DELTA - m_use, tPr_i32) if not NO_SOFTMAX else _raw_pass(sv, tPr_i32)
                        cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)   # xval reuse guard + S loads done
                        cute.copy(p_tiled, tPr, tPt[t])
                    else:
                        # speculative pass with the running max (no row max in the common path)
                        rowsum = _exp_pass(sv, scale_log2, TRUNC_DELTA - m_t, tPr_i32) if not NO_SOFTMAX else _raw_pass(sv, tPr_i32)
                        if cutlass.const_expr(TRACE):
                            if bidx == 0:
                                if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                    if ltid == 0:
                                        dbg[(j, 12 + wg)] = cute.arch.clock64() if t == 0 else dbg[(j, 12 + wg)]
                        need = cutlass.not_(rowsum <= F32(SUM_LIMIT))    # true for overflow / NaN
                        # uniform decision across both warpgroups (all 256 threads): barrier with OR-reduction.
                        # The barrier also guarantees both warpgroups finished loading S before P (aliasing S) is stored.
                        need_all = _bar_red_or(cutlass.Int32(cutlass.select_(need, I32(1), I32(0))), XCHG_BAR, 128 * NWG) != 0
                        if cutlass.const_expr(TRACE):
                            if bidx == 0:
                                if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                    if ltid == 0:
                                        dbg[(j, 14 + wg)] = cute.arch.clock64() if t == 0 else dbg[(j, 14 + wg)]
                        cute.copy(p_tiled, tPr, tPt[t])
                        if need_all:
                            # rare: recompute this tile with the exact row max (exchanged with the other warpgroup)
                            mx = _row_max(sv)
                            xval_t[wg * BM + ltid] = mx
                            cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)
                            mx = cute.arch.fmax(mx, xval_t[(1 - wg) * BM + ltid])
                            m_use = cute.arch.fmax(m_t, mx * scale_log2)
                            alpha = cute.arch.exp2(m_t - m_use)
                            m_next = m_use
                            rowsum = _exp_pass(sv, scale_log2, TRUNC_DELTA - m_use, tPr_i32)
                            cute.copy(p_tiled, tPr, tPt[t])
                            cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)   # xval reuse guard
                            cute.arch.mbarrier_wait(pv_done + t, (j - 1) & 1)
                            cute.copy(o_ld_tiled, tOt_ld[t], tOr)
                            cute.arch.fence_view_async_tmem_load()
                            ov = tOr.load()
                            tOr.store(ov * alpha)
                            cute.copy(o_st_tiled, tOr, tOt_st[t])
                    cute.arch.fence_view_async_tmem_store()
                    if cutlass.const_expr(TRACE):
                        if bidx == 0:
                            if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                                if ltid == 0:
                                    dbg[(j, 5 + 4 * t + 2 * wg)] = cute.arch.clock64()
                    cute.arch.mbarrier_arrive(p_full + t, 0)
                    if cutlass.const_expr(t == 0):
                        m_a = m_next
                        l_a = l_t * alpha + rowsum
                    else:
                        m_b = m_next
                        l_b = l_t * alpha + rowsum
                s_phase = s_phase ^ 1

            # epilogue per tile: combine partial row sums, normalize this warpgroup's O columns
            for t in cutlass.range_constexpr(NWG):
                l_t = l_a if t == 0 else l_b
                cute.arch.mbarrier_wait(pv_done + t, (n_blocks - 1) & 1)
                if cutlass.const_expr(TRACE):
                    if bidx == 0:
                        if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                            if ltid == 0:
                                dbg[(2 + 6 * t, 14 + wg)] = cute.arch.clock64()
                cute.copy(o_ld_tiled, tOt_ld[t], tOr)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(o_empty + t, 0)
                if cutlass.const_expr(TRACE):
                    if bidx == 0:
                        if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                            if ltid == 0:
                                dbg[(3 + 6 * t, 14 + wg)] = cute.arch.clock64()
                xval_t[wg * BM + ltid] = l_t
                cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)
                l_tot = l_t + xval_t[(1 - wg) * BM + ltid]
                cute.arch.barrier(barrier_id=XCHG_BAR, number_of_threads=128 * NWG)
                if cutlass.const_expr(TRACE):
                    if bidx == 0:
                        if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                            if ltid == 0:
                                dbg[(4 + 6 * t, 14 + wg)] = cute.arch.clock64()
                inv_l = (1.0 + TRUNC_MEAN_REL) / l_tot
                for i in cutlass.range_constexpr(OW // 2):
                    o2 = cute.arch.mul_packed_f32x2((tOr_flat[2 * i], tOr_flat[2 * i + 1]), (inv_l, inv_l))
                    tOut32[i] = I32(_cvt_bf16x2(F32(o2[0]), F32(o2[1])))
                # stage this warpgroup's 128 rows x 64 cols (32 words/row) through padded smem, then write
                # coalesced 128-byte row segments: 8 lanes x 16 bytes per row, 4 rows per warp-store.
                for i in cutlass.range_constexpr(OW // 2):
                    ostage_t[(wg * BM + ltid) * OSTRIDE + i] = tOut32[i]
                cute.arch.barrier(barrier_id=OSTAGE_BAR + wg, number_of_threads=128)
                lane_c = ltid % 8
                for k in cutlass.range_constexpr(BM // 16):
                    r = ltid // 8 + 16 * k
                    for w in cutlass.range_constexpr(4):
                        tOut4[w] = ostage_t[(wg * BM + r) * OSTRIDE + lane_c * 4 + w]
                    row = ((qt * NWG + t) * CLUSTER + cta_rank) * BM + r
                    gO_row = mO[(row, None, (h, b))]
                    gO_seg = cute.recast_tensor(cute.local_tile(gO_row, (8,), (wg * (OW // 8) + lane_c,)), I32)
                    cute.autovec_copy(tOut4, gO_seg)
                cute.arch.barrier(barrier_id=OSTAGE_BAR + wg, number_of_threads=128)
                if cutlass.const_expr(TRACE):
                    if bidx == 0:
                        if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                            if ltid == 0:
                                dbg[(5 + 6 * t, 14 + wg)] = cute.arch.clock64()
            if cutlass.const_expr(TRACE):
                if bidx == 0:
                    if unit == cluster_id + TRACE_UNIT_K * num_clusters:
                        if ltid == 0:
                            dbg[(1, 14 + wg)] = cute.arch.clock64()

    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr)


@cute.jit
def fa_fwd_host(
    q_raw: cute.Tensor, k_raw: cute.Tensor, v_raw: cute.Tensor, o_raw: cute.Tensor,
    scale_log2: cutlass.Float32, num_ctas: cutlass.Constexpr, B: cutlass.Constexpr, S: cutlass.Constexpr,
    H: cutlass.Constexpr, stream, dbg: cute.Tensor,
):
    D = HD
    qk_layout = cute.make_layout((S, D, (H, B)), stride=(H * D, 1, (D, S * H * D)))
    v_layout = cute.make_layout((D, S, (H, B)), stride=(1, H * D, (D, S * H * D)))
    mQ = cute.make_tensor(q_raw.iterator, qk_layout)
    mK = cute.make_tensor(k_raw.iterator, qk_layout)
    mV = cute.make_tensor(v_raw.iterator, v_layout)
    mO = cute.make_tensor(o_raw.iterator, qk_layout)

    qk_op = tcgen05.MmaF16BF16Op(
        BF16, F32, (BM * CLUSTER, BN, 16), tcgen05.CtaGroup.TWO, tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
    )
    pv_op = tcgen05.MmaF16BF16Op(
        BF16, F32, (BM * CLUSTER, HD, 16), tcgen05.CtaGroup.TWO, tcgen05.OperandSource.TMEM,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
    )
    tiled_mma_qk = cute.make_tiled_mma(qk_op)
    tiled_mma_pv = cute.make_tiled_mma(pv_op)
    qk_tiler = (BM * CLUSTER, BN, HD)
    pv_tiler = (BM * CLUSTER, HD, BN)
    cluster_shape_mn = (CLUSTER, 1)
    cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*cluster_shape_mn, 1)), (tiled_mma_qk.thr_id.shape,))

    q_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma_qk, qk_tiler, BF16, NWG)
    k_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma_qk, qk_tiler, BF16, K_STAGES)
    v_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma_pv, pv_tiler, BF16, V_STAGES)
    p_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma_pv, pv_tiler, BF16, 1)

    q_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma_qk.thr_id)
    k_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma_qk.thr_id)
    v_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma_pv.thr_id)
    tma_atom_q, tma_q = cute.nvgpu.make_tiled_tma_atom_A(
        q_op, mQ, cute.select(q_smem_layout, mode=[0, 1, 2]), qk_tiler, tiled_mma_qk, cluster_layout_vmnk.shape
    )
    tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(
        k_op, mK, cute.select(k_smem_layout, mode=[0, 1, 2]), qk_tiler, tiled_mma_qk, cluster_layout_vmnk.shape
    )
    tma_atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(
        v_op, mV, cute.select(v_smem_layout, mode=[0, 1, 2]), pv_tiler, tiled_mma_pv, cluster_layout_vmnk.shape
    )

    nq_units = S // UNIT_ROWS
    num_units = B * H * nq_units
    n_blocks = S // BN
    fa_fwd_kernel(
        tiled_mma_qk, tiled_mma_pv, tma_atom_q, tma_q, tma_atom_k, tma_k, tma_atom_v, tma_v, mO,
        cluster_layout_vmnk, q_smem_layout, k_smem_layout, v_smem_layout, p_smem_layout, scale_log2,
        cutlass.Int32(num_units), cutlass.Int32(nq_units), cutlass.Int32(H), cutlass.Int32(n_blocks), dbg,
    ).launch(grid=(num_ctas, 1, 1), block=(THREADS, 1, 1), cluster=(CLUSTER, 1, 1), stream=stream)


_compiled = {}
_dbg = None


def flash_fwd(q, k, v, num_ctas=None):
    """q,k,v: (B, S, H, D) bf16 contiguous. Returns O (B, S, H, D) bf16."""
    B, S, H, D = q.shape
    assert D == HD and k.shape == (B, S, H, D) and v.shape == (B, S, H, D)
    assert S % UNIT_ROWS == 0
    o = torch.empty_like(q)
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_ctas -= num_ctas % CLUSTER
    key = (B, S, H, num_ctas)
    import cuda.bindings.driver as cuda
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    tq, tk, tv, to = [from_dlpack(t, assumed_align=16) for t in (q, k, v, o)]
    global _dbg
    if _dbg is None or _dbg.shape[0] != S // BN:
        _dbg = torch.zeros(S // BN, 16, dtype=torch.int64, device=q.device)
    tdbg = from_dlpack(_dbg, assumed_align=16)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            fa_fwd_host, tq, tk, tv, to, cutlass.Float32(1.0), num_ctas, B, S, H, stream, tdbg
        )
    scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    _compiled[key](tq, tk, tv, to, cutlass.Float32(scale_log2), stream, tdbg)
    return o
