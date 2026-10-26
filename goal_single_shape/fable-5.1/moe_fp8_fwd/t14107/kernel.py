"""MoE FP8 block-scale (DeepSeek-V3 routing) for B200, written from scratch in Triton + Gluon (tcgen05/TMA/TMEM).

Pipeline per call (everything that depends on the inputs happens inside run()):
  1. routing          (Triton)  sigmoid -> +bias -> group top-2 sums -> top-4 groups -> top-8 experts -> weights
  2. expert offsets   (Triton)  per-expert row ranges padded to 128 rows, tile prefix tables
  3. scatter          (Triton)  pair -> permuted row, per-row token id / routing weight
  4. gather           (Triton)  permuted fp8 activations + their per-128-block scales (transposed, [56, rows])
  5. GEMM1 + SwiGLU   (Gluon)   grouped block-scaled fp8 GEMM, fused silu(x2)*x1 and per-(row,128) fp8 quantization
  6. GEMM2 + combine  (Gluon)   grouped block-scaled fp8 GEMM, routing-weight scaling, bf16 atomic accumulation

The GEMM core: warp-specialized persistent kernel; TMA loads -> tcgen05.mma into one of NBUF TMEM accumulators per
128-wide K block; two independent 4-warp "promotion" sets (even / odd K blocks) read the partial products from TMEM
and accumulate them in fp32 registers with the per-(row,k-block) x per-(col-block,k-block) scales; the two partial
sums are merged through TMEM at the end of each tile.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor

# ----------------------------------------------------------------------------------------------------------------------
# constants of this operator
E_GLOBAL = 256
E_LOCAL = 32
H = 7168
I = 2048
TOPK = 8
N_GROUP = 8
TOPK_GROUP = 4
BLK = 128
KB1 = H // BLK   # 56 k-blocks for GEMM1
KB2 = I // BLK   # 16 k-blocks for GEMM2
BM = gl.constexpr(128)
BN = gl.constexpr(256)
BK = gl.constexpr(128)
H_C = gl.constexpr(H)
I_C = gl.constexpr(I)
FP8_MAX_C = gl.constexpr(448.0)
NBUF = gl.constexpr(2)          # TMEM partial-product buffers: 2 x 256 columns = all of TMEM
import os
import re
STAGES1 = int(os.environ.get('MOE_STAGES1', '4'))
STAGES2 = int(os.environ.get('MOE_STAGES2', '4'))
DBG = gl.constexpr(int(os.environ.get('MOE_DBG', '0')))
UNROLL = gl.constexpr(int(os.environ.get('MOE_UNROLL', '1')))   # K-step unroll of the promotion loop (1 or 2)
EPI2_COAL = gl.constexpr(int(os.environ.get('MOE_EPI2_COAL', '1')))   # GEMM2 epilogue: re-layout for row-contiguous bf16x8 reductions
FAST_SILU = gl.constexpr(int(os.environ.get('MOE_FAST_SILU', '1')))   # tanh.approx-based SiLU in the GEMM1 epilogue
RAW = gl.constexpr(int(os.environ.get('MOE_RAW', '3')))
SPD = gl.constexpr(int(os.environ.get('MOE_SPD', '4')))   # scale prefetch distance (K steps) for the cp.async scale ring (RAW 3)
SRING_SLOTS = gl.constexpr(8)   # ring slots per set (must exceed SPD)
SRING_I64 = gl.constexpr((2 * 8 * 3 * 128 * 4 // 8) if RAW.value >= 4 else 1)   # int64 words of the cp.async scale ring (RAW >= 4 only)   # 1: promotion loop uses raw PTX mbarrier/TMEM ops (per-warp arrives, paired loads)
REGS1 = gl.constexpr(int(os.environ.get('MOE_REGS1', '224')))   # register budget of the promotion warps (per GEMM)
REGS2 = gl.constexpr(int(os.environ.get('MOE_REGS2', '240')))
SLW1 = gl.constexpr(int(os.environ.get('MOE_SLW1', '32')))      # TMEM load width (columns) per promotion slice
SLW2 = gl.constexpr(int(os.environ.get('MOE_SLW2', '32')))
CP_RPB = int(os.environ.get('MOE_CP_RPB', '8'))
CP_VEC = int(os.environ.get('MOE_CP_VEC', '4096'))
CP_NW = int(os.environ.get('MOE_CP_NW', '4'))
RT_TPB = int(os.environ.get('MOE_RT_TPB', '1'))
PREF = gl.constexpr(0)   # (old) separate prefetch warp: disabled
LPF = gl.constexpr(int(os.environ.get('MOE_LPF', '0')))   # loader-side L2 prefetch distance in K steps (0 = off)
OUT_RING = int(os.environ.get('MOE_OUT_RING', '16'))
FP8_MAX = 448.0


# ----------------------------------------------------------------------------------------------------------------------
# 1. routing
@triton.jit
def _routing_kernel(logits_ptr, bias_ptr, ids_ptr, w_ptr, counts_ptr, pair_le_ptr, pair_rank_ptr, T, local_off,
                    scaling, E: tl.constexpr, NG: tl.constexpr, TG: tl.constexpr, K: tl.constexpr, EL: tl.constexpr,
                    TPB: tl.constexpr):
    """DeepSeek-V3 no-aux routing for TPB tokens per program (rows -> warps, so reductions are warp shuffles)."""
    t = tl.program_id(0) * TPB + tl.arange(0, TPB)
    tm = t < T
    e = tl.arange(0, E)
    x = tl.load(logits_ptr + t[:, None] * E + e[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x))
    bias = tl.load(bias_ptr + e).to(tl.float32)
    swb = s + bias[None, :]
    GS: tl.constexpr = E // NG
    g = tl.reshape(swb, [TPB, NG, GS])
    gi = tl.arange(0, GS)[None, None, :]
    m1 = tl.max(g, axis=2)                                           # [TPB, NG]
    first = tl.min(tl.where(g == m1[:, :, None], gi, GS), axis=2)
    m2 = tl.max(tl.where(gi == first[:, :, None], float("-inf"), g), axis=2)
    gscore = m1 + m2                                                 # [TPB, NG]
    ga = tl.arange(0, NG)[None, :]
    gsel = gscore < float("-inf")                                    # all false
    for _ in tl.static_range(TG):
        cur = tl.where(gsel, float("-inf"), gscore)
        mx = tl.max(cur, axis=1)
        j = tl.min(tl.where(cur == mx[:, None], ga, NG), axis=1)
        gsel = gsel | (ga == j[:, None])
    emask = tl.reshape(tl.broadcast_to(gsel[:, :, None], [TPB, NG, GS]), [TPB, E])
    pruned = tl.where(emask, swb, float("-inf"))
    wsum = tl.zeros([TPB], dtype=tl.float32)
    for k in tl.static_range(K):
        mx = tl.max(pruned, axis=1)
        j = tl.min(tl.where(pruned == mx[:, None], e[None, :], E), axis=1)
        hit = e[None, :] == j[:, None]
        wk = tl.sum(tl.where(hit, s, 0.0), axis=1)
        wsum += wk
        tl.store(ids_ptr + t * K + k, j, mask=tm)
        tl.store(w_ptr + t * K + k, wk, mask=tm)
        pruned = tl.where(hit, float("-inf"), pruned)
        le = j - local_off
        loc = tm & (le >= 0) & (le < EL)
        rank = tl.atomic_add(counts_ptr + le * 16 + (t & 15), 1, mask=loc)   # 16 slots per expert
        tl.store(pair_le_ptr + t * K + k, tl.where(loc, le, -1), mask=tm)
        tl.store(pair_rank_ptr + t * K + k, rank, mask=loc)
    inv = scaling / (wsum + 1e-20)
    kk = tl.arange(0, K)
    wv = tl.load(w_ptr + t[:, None] * K + kk[None, :], mask=tm[:, None], other=0.0)
    tl.store(w_ptr + t[:, None] * K + kk[None, :], wv * inv[:, None], mask=tm[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# 0. staging: copy the four per-call input tensors into fixed buffers with ONE launch (byte copies, 16B per thread)
@triton.jit
def _stage_kernel(s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = tl.cdiv(n1, BLOCK)
    b2 = tl.cdiv(n2, BLOCK)
    off = tl.arange(0, BLOCK)
    if pid < b0:
        o = pid * BLOCK + off
        tl.store(d0 + o, tl.load(s0 + o, mask=o < n0), mask=o < n0)
    elif pid < b0 + b1:
        o = (pid - b0) * BLOCK + off
        tl.store(d1 + o, tl.load(s1 + o, mask=o < n1), mask=o < n1)
    elif pid < b0 + b1 + b2:
        o = (pid - b0 - b1) * BLOCK + off
        tl.store(d2 + o, tl.load(s2 + o, mask=o < n2), mask=o < n2)
    else:
        o = (pid - b0 - b1 - b2) * BLOCK + off
        tl.store(d3 + o, tl.load(s3 + o, mask=o < n3), mask=o < n3)


# ----------------------------------------------------------------------------------------------------------------------
# 2. offsets / tile tables (single program)
@triton.jit
def _offsets_kernel(counts_ptr, slot_base_ptr, cnt_ptr, offs_ptr, t1_ptr, t2_ptr, meta_ptr, EL: tl.constexpr,
                    NQ1: tl.constexpr, NT2: tl.constexpr):
    e = tl.arange(0, EL)
    sl = tl.arange(0, 16)
    cs = tl.load(counts_ptr + e[:, None] * 16 + sl[None, :])            # [EL, 16] per-slot counts
    cnt = tl.sum(cs, axis=1)
    tl.store(slot_base_ptr + e[:, None] * 16 + sl[None, :], tl.cumsum(cs, axis=1) - cs)   # exclusive within expert
    mt = (cnt + 127) // 128                             # m-tiles per expert
    rows = mt * 128
    excl = tl.cumsum(rows, axis=0) - rows               # exclusive prefix of padded rows
    tl.store(offs_ptr + e, excl)
    tl.store(offs_ptr + EL, tl.sum(rows, axis=0))
    u1 = mt * NQ1                                       # GEMM1 units per expert (128 rows x 128 j)
    u2 = mt * NT2                                       # GEMM2 tiles per expert (128 rows x 128 h)
    tl.store(t1_ptr + e, tl.cumsum(u1, axis=0) - u1)
    tl.store(t1_ptr + EL, tl.sum(u1, axis=0))
    tl.store(t2_ptr + e, tl.cumsum(u2, axis=0) - u2)
    tl.store(t2_ptr + EL, tl.sum(u2, axis=0))
    tl.store(meta_ptr + 0, tl.sum(rows, axis=0))
    tl.store(meta_ptr + 1, tl.sum(u1, axis=0))
    tl.store(meta_ptr + 2, tl.sum(u2, axis=0))
    tl.store(cnt_ptr + e, cnt)                          # per-expert totals (used by the gather kernel)


# ----------------------------------------------------------------------------------------------------------------------
# 2b. per-unit tables for the persistent GEMMs: info[u] = (expert, n-tile, padded row offset, 0)
@triton.jit
def _unit_table_kernel(t1_ptr, t2_ptr, offs_ptr, meta_ptr, info1_ptr, info2_ptr, EL: tl.constexpr, UB: tl.constexpr):
    e_idx = tl.arange(0, EL)
    n1 = tl.load(meta_ptr + 1)
    n2 = tl.load(meta_ptr + 2)
    for which in tl.static_range(2):
        if which == 0:
            tptr = t1_ptr
            n = n1
            iptr = info1_ptr
        else:
            tptr = t2_ptr
            n = n2
            iptr = info2_ptr
        starts_n = tl.load(tptr + 1 + e_idx)
        for u0 in range(tl.program_id(0) * UB, n, tl.num_programs(0) * UB):
            u = u0 + tl.arange(0, UB)
            e = tl.sum((u[:, None] >= starts_n[None, :]).to(tl.int32), axis=1)
            e = tl.minimum(e, EL - 1)
            local = u - tl.load(tptr + e)
            mt_e = tl.maximum(tl.load(offs_ptr + e + 1) - tl.load(offs_ptr + e), 128) // 128
            mt = local % mt_e
            nt = local // mt_e
            arow = tl.load(offs_ptr + e) + mt * 128
            m = u < n
            tl.store(iptr + u * 4 + 0, e, mask=m)
            tl.store(iptr + u * 4 + 1, nt, mask=m)
            tl.store(iptr + u * 4 + 2, arow, mask=m)


# ----------------------------------------------------------------------------------------------------------------------
# 3. gather: pair -> permuted row (fp8 activations, transposed scales, token id, routing weight); zero padding rows
@triton.jit
def _gather_kernel(pair_le_ptr, pair_rank_ptr, w_ptr, offs_ptr, slot_base_ptr, counts_ptr, row_tok_ptr, row_w_ptr,
                   T, EL: tl.constexpr, K: tl.constexpr, PB: tl.constexpr):
    """row -> (token, routing weight) for every permuted row; padding rows get token -1 / weight 0."""
    pid = tl.program_id(0)
    p = pid * PB + tl.arange(0, PB)
    pm = p < T * K
    le = tl.load(pair_le_ptr + p, mask=pm, other=-1)
    ok = le >= 0
    slot = (p // K) & 15
    r = tl.load(offs_ptr + le, mask=ok, other=0) + tl.load(slot_base_ptr + le * 16 + slot, mask=ok, other=0) \
        + tl.load(pair_rank_ptr + p, mask=ok, other=0)
    tl.store(row_tok_ptr + r, p // K, mask=ok)
    tl.store(row_w_ptr + r, tl.load(w_ptr + p, mask=ok, other=0.0), mask=ok)
    # padding rows: [offs[e] + cnt[e], offs[e+1])
    idx = p - T * K
    pad = (idx >= 0) & (idx < EL * 128)
    e = tl.where(pad, idx // 128, 0)
    rp = tl.load(offs_ptr + e, mask=pad, other=0) + tl.load(counts_ptr + e, mask=pad, other=0) + idx % 128
    pad = pad & (rp < tl.load(offs_ptr + e + 1, mask=pad, other=0))
    tl.store(row_tok_ptr + rp, -1, mask=pad)
    tl.store(row_w_ptr + rp, 0.0, mask=pad)


@triton.jit
def _copy_rows_kernel(row_tok_ptr, hs_ptr, a_ptr, hss_ptr, sa_ptr, meta_ptr, T, P_MAX, HD: tl.constexpr,
                      VEC: tl.constexpr, RPB: tl.constexpr, KB: tl.constexpr):
    """A_perm[r, :] = hidden_states[row_tok[r], :] and sa_perm[kb, r] = hidden_states_scale[kb, row_tok[r]]
    (zeros for padding rows).  RPB rows per program, 16B per thread for the activations."""
    n_rows = tl.load(meta_ptr)
    c = tl.arange(0, VEC)
    r0 = tl.program_id(0) * RPB
    for i in tl.static_range(RPB):
        r = r0 + i
        if r < n_rows:
            t = tl.load(row_tok_ptr + r)
            for h0 in tl.static_range(0, HD, VEC):
                v = tl.load(hs_ptr + t.to(tl.int64) * HD + h0 + c, mask=(h0 + c < HD) & (t >= 0), other=0.0)
                tl.store(a_ptr + r.to(tl.int64) * HD + h0 + c, v, mask=h0 + c < HD)
    # block scales: [KB, RPB] gather (scattered reads), contiguous RPB-wide stores per K block
    kb = tl.arange(0, 64)[:, None]
    rr = r0 + tl.arange(0, RPB)[None, :]
    tok = tl.load(row_tok_ptr + rr, mask=rr < n_rows, other=-1)
    sv = tl.load(hss_ptr + kb * T + tl.where(tok >= 0, tok, 0), mask=(kb < KB) & (tok >= 0), other=0.0)
    tl.store(sa_ptr + kb * P_MAX + rr, sv, mask=(kb < KB) & (rr < n_rows))


# ----------------------------------------------------------------------------------------------------------------------
# 5/6. grouped block-scaled GEMM core (Gluon, Blackwell): 128 x 256 tiles, two column-split promotion sets
@gluon.jit
def _lane_zero(x):
    # (x >> 31) == 0 for non-negative x, but opaque to range analysis: keeps broadcast loads in regular registers
    return gl.inline_asm_elementwise("shr.s32 $0, $1, 31;", "=r,r", [x], dtype=gl.int32, is_pure=True, pack=1)


@gluon.jit
def _tid(x):
    """%tid.x as a (physically per-thread) scalar."""
    return gl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _laneid(x):
    return gl.inline_asm_elementwise("mov.u32 $0, %laneid;", "=r,r", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _smem_base(x):
    """Shared-memory address of Triton's dynamic smem block (base of all allocations)."""
    return gl.inline_asm_elementwise("mov.u32 $0, global_smem;", "=r,r", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _lds32(addr):
    return gl.inline_asm_elementwise("ld.shared.b32 $0, [$1];", "=r,r", [addr], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _bar_wait_raw(addr, phase):
    """mbarrier parity wait by every lane of the calling warp (no partition-wide bar.sync), then a tcgen05 fence."""
    return gl.inline_asm_elementwise(
        "{\n.reg .pred p;\nWAITLOOP:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n@!p bra WAITLOOP;\n"
        "tcgen05.fence::after_thread_sync;\nmov.u32 $0, $1;\n}", "=r,r,r", [addr, phase], dtype=gl.int32,
        is_pure=False, pack=1)


@gluon.jit
def _bar_arrive_raw(addr, lane):
    """One arrive per warp (lane 0) after a tcgen05 fence: the barrier is initialised with count = number of warps."""
    return gl.inline_asm_elementwise(
        "{\n.reg .pred p;\nsetp.eq.u32 p, $2, 0;\ntcgen05.fence::before_thread_sync;\n"
        "@p mbarrier.arrive.shared::cta.b64 _, [$1];\nmov.u32 $0, $1;\n}", "=r,r,r", [addr, lane], dtype=gl.int32,
        is_pure=False, pack=1)


@gluon.jit
def _tmem_ld32(addr_t):
    """tcgen05.ld 32x32b.x32 without the implicit wait: addr_t is a [BM, 32] int32 tensor (TMEM reg layout) whose
    elements all hold the warp's TMEM address (lane quadrant | column).  Returns the 32 columns as int32 bits."""
    return gl.inline_asm_elementwise(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, "
        "$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, [$32];",
        "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
        "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r", [addr_t], dtype=gl.int32, is_pure=False,
        pack=32)


@gluon.jit
def _cp_async4(saddr, gptr):
    """cp.async of one fp32 per thread: global gptr -> shared saddr (both [BM] tensors, one element per thread)."""
    return gl.inline_asm_elementwise("cp.async.ca.shared.global [$1], [$2], 4;\n\tmov.u32 $0, $1;", "=r,r,l",
                                     [saddr, gptr], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _cp_commit(x):
    return gl.inline_asm_elementwise("cp.async.commit_group;\n\tmov.u32 $0, $1;", "=r,r", [x], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _cp_wait(x, ASM: gl.constexpr):
    return gl.inline_asm_elementwise(ASM, "=r,r", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _lds_f32(saddr):
    return gl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r", [saddr], dtype=gl.float32, is_pure=False,
                                     pack=1)


@gluon.jit
def _sts_f32(saddr, v):
    return gl.inline_asm_elementwise("st.shared.f32 [$1], $2;\n\tmov.u32 $0, $1;", "=r,r,f", [saddr, v],
                                     dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _ring_put(ring_t, slot, a, b, c):
    base = ring_t + slot * (3 * BM * 4)
    _sts_f32(base, a)
    _sts_f32(base + BM * 4, b)
    _sts_f32(base + 2 * BM * 4, c)


@gluon.jit
def _ring_issue(ring_t, slot, sa_ptr, SST, ridx, sb0_ptr, sb1_ptr, zero_rows, kk):
    """Prefetch (cp.async) the three scales of step kk into ring slot `slot` and commit the group."""
    base = ring_t + slot * (3 * BM * 4)
    _cp_async4(base, sa_ptr + kk * SST + ridx)
    _cp_async4(base + BM * 4, sb0_ptr + kk + zero_rows)
    _cp_async4(base + 2 * BM * 4, sb1_ptr + kk + zero_rows)
    _cp_commit(slot)


@gluon.jit
def _tanh(x):
    return gl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _wait_ld(x):
    return gl.inline_asm_elementwise("tcgen05.wait::ld.sync.aligned;\nmov.u32 $0, $1;", "=r,r", [x], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _load_info(info_ptr, u):
    e = gl.load(info_ptr + u * 4)
    nt = gl.load(info_ptr + u * 4 + 1)
    arow = gl.load(info_ptr + u * 4 + 2)
    return e, nt, arow


@gluon.jit
def _loader(a_desc, b_desc, a_base, b_base, a_smem, b_smem, ready, empty, info_ptr, n_units, pid, nprog,
            KT: gl.constexpr, EPI: gl.constexpr, STAGES: gl.constexpr):
    g = 0
    layA: gl.constexpr = gl.BlockedLayout([4], [32], [1], [0])
    layB: gl.constexpr = gl.BlockedLayout([8], [32], [1], [0])
    rA = gl.arange(0, 128, layout=layA)   # A rows of a tile (one 128B line each per K step)
    rB = gl.arange(0, 256, layout=layB)   # B rows of a tile
    e_n, nt_n, arow_n = _load_info(info_ptr, gl.minimum(pid, n_units - 1))
    for unit in range(pid, n_units, nprog):
        e = e_n
        nt = nt_n
        arow = arow_n
        e_n, nt_n, arow_n = _load_info(info_ptr, gl.minimum(unit + nprog, n_units - 1))
        for k in range(KT):
            s = g % STAGES
            ph = (g // STAGES) & 1
            mbarrier.wait(empty.index(s), ph ^ 1)
            mbarrier.expect(ready.index(s), BM * BK + BN * BK)
            tma.async_copy_global_to_shared(a_desc, [arow, k * BK], ready.index(s), a_smem.index(s))
            if EPI == 1:
                # B rows: [x1 j0..+64 | x2 j0..+64 | x1 j0+64..+128 | x2 j0+64..+128], j0 = nt*128
                b0 = e * (2 * I_C) + nt * 128
                tma.async_copy_global_to_shared(b_desc, [b0, k * BK], ready.index(s), b_smem.index(s).slice(0, 64))
                tma.async_copy_global_to_shared(b_desc, [b0 + I_C, k * BK], ready.index(s),
                                                b_smem.index(s).slice(64, 64))
                tma.async_copy_global_to_shared(b_desc, [b0 + 64, k * BK], ready.index(s),
                                                b_smem.index(s).slice(128, 64))
                tma.async_copy_global_to_shared(b_desc, [b0 + I_C + 64, k * BK], ready.index(s),
                                                b_smem.index(s).slice(192, 64))
            else:
                tma.async_copy_global_to_shared(b_desc, [e * H_C + nt * BN, k * BK], ready.index(s),
                                                b_smem.index(s))
            if LPF > 0:
                # touch the A / B lines of step g + LPF (possibly in this CTA's next unit) so the TMA loads hit L2
                nxt = k + LPF >= KT
                e_p = gl.where(nxt, e_n, e)
                nt_p = gl.where(nxt, nt_n, nt)
                arow_p = gl.where(nxt, arow_n, arow)
                kp = gl.where(nxt, k + LPF - KT, k + LPF) * BK
                if EPI == 1:
                    b0p = e_p * (2 * I_C) + nt_p * 128
                    brow = b0p + (rB % 64) + ((rB // 64) % 2) * I_C + (rB // 128) * 64
                    _l2_prefetch(b_base + brow.to(gl.int64) * H_C + kp)
                    _l2_prefetch(a_base + (arow_p + rA).to(gl.int64) * H_C + kp)
                else:
                    _l2_prefetch(b_base + (e_p * H_C + nt_p * BN + rB).to(gl.int64) * I_C + kp)
                    _l2_prefetch(a_base + (arow_p + rA).to(gl.int64) * I_C + kp)
            g += 1


@gluon.jit
def _l2_prefetch(ptrs):
    return gl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l", [ptrs], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _prefetcher(a_base, b_base, empty, info_ptr, n_units, pid, nprog, KT: gl.constexpr, EPI: gl.constexpr,
                STAGES: gl.constexpr):
    """Runs PREF K-steps ahead of the TMA loader (paced by the same `empty` barriers) and touches the A/B rows of
    that step so the TMA loads hit L2 instead of paying HBM latency."""
    lay: gl.constexpr = gl.BlockedLayout([8], [32], [1], [0])
    r = gl.arange(0, 256, layout=lay)
    g = 0
    e_n, nt_n, arow_n = _load_info(info_ptr, gl.minimum(pid, n_units - 1))
    for unit in range(pid, n_units, nprog):
        e = e_n
        nt = nt_n
        arow = arow_n
        e_n, nt_n, arow_n = _load_info(info_ptr, gl.minimum(unit + nprog, n_units - 1))
        if EPI == 1:
            b0 = e * (2 * I_C) + nt * 128
            brow = b0 + gl.where(r < 128, (r % 64) + (r // 64) * I_C, 64 + ((r - 128) % 64) + ((r - 128) // 64) * I_C)
            bptr = b_base + brow.to(gl.int64) * H_C
            aptr = a_base + (arow + (r % 128)).to(gl.int64) * H_C
        else:
            bptr = b_base + (e * H_C + nt * BN + r).to(gl.int64) * I_C
            aptr = a_base + (arow + (r % 128)).to(gl.int64) * I_C
        for k in range(KT):
            s = g % STAGES
            ph = (g // STAGES) & 1
            mbarrier.wait(empty.index(s), ph ^ 1)
            kp = gl.minimum(k + PREF, KT - 1) * BK
            if DBG != 5:
                _l2_prefetch(bptr + kp)
                _l2_prefetch(aptr + kp)
            g += 1


@gluon.jit
def _mma(a_smem, b_smem, ready, empty, acc_ready, acc_empty, acc_tmem, n_units, pid, nprog, KT: gl.constexpr,
         STAGES: gl.constexpr):
    n_my = gl.maximum((n_units - pid + nprog - 1) // nprog, 0)
    total = n_my * KT
    for g in range(total):
        s = g % STAGES
        ph = (g // STAGES) & 1
        buf = g % NBUF
        bph = (g // NBUF) & 1
        mbarrier.wait(ready.index(s), ph)
        mbarrier.wait(acc_empty.index(buf), bph ^ 1)
        if DBG == 7:   # timing experiment: no MMA work, pipeline runs at the speed of loader + promotion sets
            tcgen05_commit(empty.index(s))
            tcgen05_commit(acc_ready.index(buf))
        else:
            tcgen05_mma(a_smem.index(s), b_smem.index(s).permute((1, 0)), acc_tmem.index(buf), use_acc=False,
                        mbarriers=[empty.index(s), acc_ready.index(buf)])


@gluon.jit
def _zero_out(out_ptr, total, pid, nprog):
    """Zero the output (bf16) with 16-byte stores; one warp per CTA, overlapped with GEMM1."""
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [1], [0])
    V: gl.constexpr = 256
    per = ((total + nprog - 1) // nprog + V - 1) // V * V
    start = pid * per
    end = gl.minimum(start + per, total)
    offs = gl.arange(0, V, layout=layout)
    z = gl.zeros([V], gl.bfloat16, layout)
    for o in range(start, end, V):
        gl.store(out_ptr + o + offs, z, mask=o + offs < end)


@gluon.jit
def _scale_ptrs(e, nt, sbw_ptr, KT: gl.constexpr, EPI: gl.constexpr, SET: gl.constexpr):
    if EPI == 1:
        sb0_ptr = sbw_ptr + (e * 32 + nt) * KT       # gemm1 scale [E*32, 56]: x1 block nt (slices 0,1)
        sb1_ptr = sbw_ptr + (e * 32 + 16 + nt) * KT  # x2 block (slices 2,3)
    else:
        sb0_ptr = sbw_ptr + (e * 56 + nt * 2 + SET) * KT   # gemm2 scale [E*56, 16]: h block of this set
        sb1_ptr = sb0_ptr
    return sb0_ptr, sb1_ptr


@gluon.jit
def _row_index(row_tok_ptr, arow, rows, EPI: gl.constexpr):
    """Per-row index into the A-scale array [KB, P_MAX]: both GEMMs use the permuted row (sa_perm / cs)."""
    return arow + rows


@gluon.jit
def _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, k):
    return (gl.load(sa_ptr + k * SST + ridx), gl.load(sb0_ptr + k + zero_rows), gl.load(sb1_ptr + k + zero_rows))


@gluon.jit
def _preload(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, KT: gl.constexpr):
    """Scales of the first PF steps of a tile (issued one tile early)."""
    a0, b0, c0 = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, 0)
    a1, b1, c1 = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, gl.minimum(1, KT - 1))
    a2, b2, c2 = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, gl.minimum(2, KT - 1))
    a3, b3, c3 = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, gl.minimum(3, KT - 1))
    return a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3


@gluon.jit
def _pstep(t, rbar, ebar, bph, sa, sb, sc, acc0, acc1, acc2, acc3, taddr, ar, ae, lane,
           C0: gl.constexpr, SL: gl.constexpr, reg_layout: gl.constexpr):
    """One K block in TMEM buffer t (Gluon path) / at TMEM address taddr (RAW path): promote the 4 column slices.
    RAW: wait on acc_ready at smem address ar, two tcgen05.ld in flight per wait, one arrive per warp on ae."""
    s0 = gl.expand_dims(sa * sb, 1)
    s1 = gl.expand_dims(sa * sc, 1)
    if RAW >= 1:
        # RAW 1: raw waits/arrives + raw TMEM loads; 2: Gluon barriers + raw loads; >=3: raw barriers + Gluon loads
        if RAW == 2:
            mbarrier.wait(rbar, bph)
        else:
            _bar_wait_raw(ar, bph)
        if DBG == 1:
            acc0 = acc0 + s0
        elif RAW >= 3:
            p0 = t.slice(C0, SL).load(reg_layout)
            p1 = t.slice(C0 + SL, SL).load(reg_layout)
            acc0 = gl.fma(p0, s0, acc0)
            p2 = t.slice(C0 + 2 * SL, SL).load(reg_layout)
            acc1 = gl.fma(p1, s0, acc1)
            p3 = t.slice(C0 + 3 * SL, SL).load(reg_layout)
            acc2 = gl.fma(p2, s1, acc2)
            acc3 = gl.fma(p3, s1, acc3)
        else:
            at = gl.full([BM, SL], 0, gl.int32, reg_layout) + taddr
            p0 = _tmem_ld32(at).to(gl.float32, bitcast=True)
            p1 = _tmem_ld32(at + SL).to(gl.float32, bitcast=True)
            _wait_ld(taddr)
            acc0 = gl.fma(p0, s0, acc0)
            acc1 = gl.fma(p1, s0, acc1)
            p2 = _tmem_ld32(at + 2 * SL).to(gl.float32, bitcast=True)
            p3 = _tmem_ld32(at + 3 * SL).to(gl.float32, bitcast=True)
            _wait_ld(taddr)
            acc2 = gl.fma(p2, s1, acc2)
            acc3 = gl.fma(p3, s1, acc3)
        if RAW == 2:
            mbarrier.arrive(ebar, count=1)
        else:
            _bar_arrive_raw(ae, lane)
    else:
        mbarrier.wait(rbar, bph)
        if DBG == 1:
            acc0 = acc0 + s0
        elif SL == 64:
            p0 = t.slice(C0, SL).load(reg_layout)
            p1 = t.slice(C0 + SL, SL).load(reg_layout)
            acc0 = gl.fma(p0, s0, acc0)
            acc2 = gl.fma(p1, s1, acc2)
        else:
            p0 = t.slice(C0, SL).load(reg_layout)
            p1 = t.slice(C0 + SL, SL).load(reg_layout)
            acc0 = gl.fma(p0, s0, acc0)
            p2 = t.slice(C0 + 2 * SL, SL).load(reg_layout)
            acc1 = gl.fma(p1, s0, acc1)
            p3 = t.slice(C0 + 3 * SL, SL).load(reg_layout)
            acc2 = gl.fma(p2, s1, acc2)
            acc3 = gl.fma(p3, s1, acc3)
        mbarrier.arrive(ebar, count=1)
    return acc0, acc1, acc2, acc3


@gluon.jit
def _promote_tile(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, g0, acc_ready, acc_empty, acc_tmem, zero_rows,
                  tcol, ar0, ae0, lane, a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3,
                  SET: gl.constexpr, KT: gl.constexpr, SL: gl.constexpr, reg_layout: gl.constexpr):
    """Accumulate all K blocks of one tile for this set's 128 columns.  Returns 4 slices [BM, SL] (fp32).
    sb0 applies to slices 0,1 and sb1 to slices 2,3 (GEMM1: x1 / x2 column blocks; GEMM2: sb1 == sb0).
    Scales are prefetched four steps ahead (the first four are passed in, loaded during the previous tile)."""
    C0: gl.constexpr = SET * 128
    acc0 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc1 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc2 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc3 = gl.zeros([BM, SL], gl.float32, reg_layout)
    # 5-deep register prefetch: (a0..a3) hold steps k..k+3, (an) holds step k+4 (loaded one step ago).  The rotation
    # happens BEFORE the new load so that no register copy ever waits on a just-issued global load.
    an, bn, cn = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, gl.minimum(4, KT - 1))
    for k in range(KT):
        g = g0 + k
        buf = g % NBUF
        bph = (g // NBUF) & 1
        sa = a0
        sb = b0
        sc = c0
        a0 = a1
        b0 = b1
        c0 = c1
        a1 = a2
        b1 = b2
        c1 = c2
        a2 = a3
        b2 = b3
        c2 = c3
        a3 = an
        b3 = bn
        c3 = cn
        an, bn, cn = _ld3(sa_ptr, sb0_ptr, sb1_ptr, SST, ridx, zero_rows, gl.minimum(k + 5, KT - 1))
        acc0, acc1, acc2, acc3 = _pstep(acc_tmem.index(buf), acc_ready.index(buf), acc_empty.index(buf), bph,
                                        sa, sb, sc, acc0, acc1, acc2, acc3, tcol + buf * BN, ar0 + buf * 8,
                                        ae0 + buf * 8, lane, C0, SL, reg_layout)
    return acc0, acc1, acc2, acc3


@gluon.jit
def _promote_tile_ring(sa_ptr, sb0_ptr, sb1_ptr, sb0_n, sb1_n, SST, ridx, ridx_n, g0, zero_rows, ring_t, acc_tmem,
                       tcol, ar0, ae0, lane, ap, bp, cp, SET: gl.constexpr, KT: gl.constexpr, SL: gl.constexpr,
                       reg_layout: gl.constexpr):
    """RAW 3 promotion of one tile: per-step scales come from the cp.async smem ring (slot g % SRING_SLOTS), and the
    scales of step g + SPD (possibly in the next tile: sb*_n / ridx_n) are prefetched each step."""
    acc0 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc1 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc2 = gl.zeros([BM, SL], gl.float32, reg_layout)
    acc3 = gl.zeros([BM, SL], gl.float32, reg_layout)
    for k in range(KT):
        g = g0 + k
        buf = g % NBUF
        bph = (g // NBUF) & 1
        nxt = k + SPD >= KT
        kk = gl.where(nxt, k + SPD - KT, k + SPD)
        pa = gl.where(nxt, sa_ptr + kk * SST + ridx_n, sa_ptr + kk * SST + ridx)
        pb0 = gl.where(nxt, sb0_n + kk, sb0_ptr + kk) + zero_rows
        pb1 = gl.where(nxt, sb1_n + kk, sb1_ptr + kk) + zero_rows
        wslot = (g + SPD) % SRING_SLOTS
        wbase = ring_t + wslot * (3 * BM * 4)
        if RAW == 4:
            _cp_async4(wbase, pa)
            _cp_async4(wbase + BM * 4, pb0)
            _cp_async4(wbase + 2 * BM * 4, pb1)
            _cp_commit(wslot)
            _cp_wait(wslot, "cp.async.wait_group 4;\n\tmov.u32 $0, $1;")
        else:
            # store the scales loaded one step ago (step g + SPD - 1), then issue the loads for step g + SPD
            _ring_put(ring_t, (g + SPD - 1) % SRING_SLOTS, ap, bp, cp)
            ap = gl.load(pa)
            bp = gl.load(pb0)
            cp = gl.load(pb1)
        rbase = ring_t + (g % SRING_SLOTS) * (3 * BM * 4)
        sa = _lds_f32(rbase)
        sb = _lds_f32(rbase + BM * 4)
        sc = _lds_f32(rbase + 2 * BM * 4)
        acc0, acc1, acc2, acc3 = _pstep(acc_tmem.index(buf), 0, 0, bph, sa, sb, sc, acc0, acc1, acc2, acc3,
                                        tcol + buf * BN, ar0 + buf * 8, ae0 + buf * 8, lane, SET * 128, SL, reg_layout)
    return acc0, acc1, acc2, acc3, ap, bp, cp


@gluon.jit
def _store_bf16_atomic(ptrs, v, mask):
    gl.atomic_add(ptrs, v.to(gl.bfloat16), mask=mask, sem="relaxed")


@gluon.jit
def _epi_set(c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
             acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar, off_ae,
             off_ring, SET: gl.constexpr, KT: gl.constexpr, EPI: gl.constexpr, num_warps: gl.constexpr):
    if EPI == 1:
        SL: gl.constexpr = SLW1
    else:
        SL: gl.constexpr = SLW2
    sl_layout: gl.constexpr = TensorMemoryLayout(block=(BM, SL), col_stride=1)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [BM, SL], sl_layout, num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, reg_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, reg_layout)
    rows = gl.arange(0, BM, layout=row_layout)
    cols = gl.arange(0, SL, layout=col_layout)
    g = 0
    nt_i = 0   # tile counter (exchange barrier phases)
    # RAW path: smem addresses of the acc barriers and this warp's TMEM address (lane quadrant | set column base)
    sbase = _smem_base(pid)
    lane = _laneid(pid)
    wq = (_tid(pid) // 32) & 3
    tcol = off_tmem + ((wq * 32) << 16) + SET * 128   # off_tmem carries the TMEM base address (read before TMA traffic)
    ar0 = sbase + off_ar
    ae0 = sbase + off_ae
    last = n_units - 1
    e_c, nt_c, arow_c = _load_info(info_ptr, gl.minimum(pid, last))
    e_n, nt_n, arow_n = _load_info(info_ptr, gl.minimum(pid + nprog, last))
    sb0_c, sb1_c = _scale_ptrs(e_c, nt_c, sbw_ptr, KT, EPI, SET)
    ridx_c = _row_index(row_tok_ptr, arow_c, rows, EPI)
    # a per-lane zero derived from a LOADED value: keeps the broadcast scale loads out of uniform registers
    # (otherwise ptxas converts them right after the load and the whole loop waits on the load latency)
    zr_c = _lane_zero(ridx_c)
    if RAW >= 4:
        # per-thread column of this set's scale ring: [slot][value][thread]
        ring_t = gl.full([BM], 0, gl.int32, row_layout) + (sbase + off_ring + SET * (SRING_SLOTS * 3 * BM * 4)
                                                             + (_tid(pid) % 128) * 4)
        if RAW == 4:
            for kk in gl.static_range(SPD):   # prologue: scales of the first SPD steps of the first tile
                _ring_issue(ring_t, kk, sa_ptr, SST, ridx_c, sb0_c, sb1_c, zr_c, gl.minimum(kk, KT - 1))
        else:
            for kk in gl.static_range(SPD - 1):   # prologue: steps 0 .. SPD-2 stored, step SPD-1 kept in registers
                a_, b_, c_ = _ld3(sa_ptr, sb0_c, sb1_c, SST, ridx_c, zr_c, gl.minimum(kk, KT - 1))
                _ring_put(ring_t, kk, a_, b_, c_)
            ap, bp, cp = _ld3(sa_ptr, sb0_c, sb1_c, SST, ridx_c, zr_c, gl.minimum(SPD - 1, KT - 1))
    else:
        q = _preload(sa_ptr, sb0_c, sb1_c, SST, ridx_c, zr_c, KT)
    for unit in range(pid, n_units, nprog):
        e_nn, nt_nn, arow_nn = _load_info(info_ptr, gl.minimum(unit + 2 * nprog, last))
        sb0_n, sb1_n = _scale_ptrs(e_n, nt_n, sbw_ptr, KT, EPI, SET)
        ridx_n = _row_index(row_tok_ptr, arow_n, rows, EPI)
        zr_n = _lane_zero(ridx_n)
        if RAW < 4:
            qn = _preload(sa_ptr, sb0_n, sb1_n, SST, ridx_n, zr_n, KT)
        if EPI == 2:
            tok = gl.load(row_tok_ptr + arow_c + rows)
            w = gl.load(row_w_ptr + arow_c + rows)
        if RAW >= 4:
            if RAW == 4:
                ap = zr_c.to(gl.float32)
                bp = ap
                cp = ap
            acc0, acc1, acc2, acc3, ap, bp, cp = _promote_tile_ring(sa_ptr, sb0_c, sb1_c, sb0_n, sb1_n, SST, ridx_c,
                                                                    ridx_n, g, zr_c, ring_t, acc_tmem, tcol, ar0, ae0,
                                                                    lane, ap, bp, cp, SET, KT, SL, reg_layout)
        else:
            acc0, acc1, acc2, acc3 = _promote_tile(sa_ptr, sb0_c, sb1_c, SST, ridx_c, g, acc_ready, acc_empty,
                                                   acc_tmem, zr_c, tcol, ar0, ae0, lane, q[0], q[1], q[2], q[3], q[4],
                                                   q[5], q[6], q[7], q[8], q[9], q[10], q[11], SET, KT, SL, reg_layout)
        g += KT
        if EPI == 1:
            # acc0/acc1 = x1 (64 j), acc2/acc3 = x2 (64 j) of j-range [nt*128 + SET*64, +64) -> c = x1 * silu(x2)
            if FAST_SILU == 1:
                # silu(x) = x * sigmoid(x), sigmoid(x) = 0.5 + 0.5 * tanh(x / 2): one MUFU op instead of exp + divide
                # (tanh.approx rel. error ~5e-4, far below the fp8 quantisation of c)
                c0 = acc0 * acc2 * gl.fma(_tanh(acc2 * 0.5), 0.5, 0.5)
                c1 = acc1 * acc3 * gl.fma(_tanh(acc3 * 0.5), 0.5, 0.5)
            else:
                c0 = acc0 * (acc2 / (1.0 + gl.exp(-acc2)))
                c1 = acc1 * (acc3 / (1.0 + gl.exp(-acc3)))
            if SL == 64:
                amax = gl.max(gl.abs(c0), axis=1)
            else:
                amax = gl.maximum(gl.max(gl.abs(c0), axis=1), gl.max(gl.abs(c1), axis=1))
            # exchange the per-row amax with the other set (the fp8 scale block spans both sets' 64 j's)
            mbarrier.wait(x_done.index(SET), (nt_i & 1) ^ 1)          # other set finished reading my previous slot
            xch_smem.index(SET).store(amax)
            mbarrier.arrive(x_full.index(SET), count=1)
            mbarrier.wait(x_full.index(1 - SET), nt_i & 1)
            amax = gl.maximum(amax, xch_smem.index(1 - SET).load(row_layout))
            mbarrier.arrive(x_done.index(1 - SET), count=1)
            amax = gl.maximum(amax, 1e-30)
            inv = gl.expand_dims(FP8_MAX_C / amax, 1)
            cbase = c_ptr + gl.expand_dims((arow_c + rows).to(gl.int64) * I_C + nt_c * 128 + SET * 64, 1) \
                + gl.expand_dims(cols, 0)
            gl.store(cbase, (c0 * inv).to(gl.float8e4nv))
            if SL == 32:
                gl.store(cbase + SL, (c1 * inv).to(gl.float8e4nv))
            if SET == 0:
                gl.store(cs_ptr + nt_c * P_MAX + arow_c + rows, amax / FP8_MAX_C)
        elif EPI2_COAL == 1:
            # re-layout each [BM, 32] slice so that 4 consecutive lanes cover 64 contiguous bytes of one output row
            # (8 bf16 per lane): the bf16x8 reductions then touch 8 rows per warp instruction instead of 32 lines
            L2: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
            rows2 = gl.arange(0, BM, layout=gl.SliceLayout(1, L2))
            cols2 = gl.arange(0, SL, layout=gl.SliceLayout(0, L2))
            tok2 = gl.load(row_tok_ptr + arow_c + rows2)
            valid2 = tok2 >= 0
            obase2 = out_ptr + gl.expand_dims(gl.where(valid2, tok2, 0).to(gl.int64) * H_C + nt_c * BN + SET * 128, 1) \
                + gl.expand_dims(cols2, 0)
            vmask2 = gl.expand_dims(valid2, 1) & (gl.expand_dims(cols2, 0) >= 0)
            wv = gl.expand_dims(w, 1)
            _store_bf16_atomic(obase2, gl.convert_layout((acc0 * wv).to(gl.bfloat16), L2), vmask2)
            _store_bf16_atomic(obase2 + SL, gl.convert_layout((acc1 * wv).to(gl.bfloat16), L2), vmask2)
            _store_bf16_atomic(obase2 + 2 * SL, gl.convert_layout((acc2 * wv).to(gl.bfloat16), L2), vmask2)
            _store_bf16_atomic(obase2 + 3 * SL, gl.convert_layout((acc3 * wv).to(gl.bfloat16), L2), vmask2)
        else:
            valid = tok >= 0
            wv = gl.expand_dims(w, 1)
            obase = out_ptr + gl.expand_dims(gl.where(valid, tok, 0).to(gl.int64) * H_C + nt_c * BN + SET * 128, 1) \
                + gl.expand_dims(cols, 0)
            vmask = gl.expand_dims(valid, 1) & (gl.expand_dims(cols, 0) >= 0)
            if SL == 64:
                _store_bf16_atomic(obase, acc0 * wv, vmask)
                _store_bf16_atomic(obase + SL, acc2 * wv, vmask)
            else:
                _store_bf16_atomic(obase, acc0 * wv, vmask)
                _store_bf16_atomic(obase + SL, acc1 * wv, vmask)
                _store_bf16_atomic(obase + 2 * SL, acc2 * wv, vmask)
                _store_bf16_atomic(obase + 3 * SL, acc3 * wv, vmask)
        nt_i += 1
        e_c = e_n
        nt_c = nt_n
        arow_c = arow_n
        sb0_c = sb0_n
        sb1_c = sb1_n
        ridx_c = ridx_n
        zr_c = zr_n
        e_n = e_nn
        nt_n = nt_nn
        arow_n = arow_nn
        if RAW < 4:
            q = qn


@gluon.jit(do_not_specialize=["off_tmem", "off_ar", "off_ae", "off_ring"])
def _moe_gemm_kernel(a_desc, b_desc, a_base, b_base, sa_ptr, sbw_ptr, c_ptr, cs_ptr, out_ptr, row_tok_ptr,
                     row_w_ptr, info_ptr, meta_ptr, P_MAX, T, SST, out_numel, off_tmem, off_ar, off_ae, off_ring,
                     KT: gl.constexpr, EPI: gl.constexpr,
                     STAGES: gl.constexpr, num_warps: gl.constexpr):
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    n_units = gl.load(meta_ptr + EPI)   # meta[1] = gemm1 units, meta[2] = gemm2 tiles

    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BM, BK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, BK], b_desc.layout)
    xch_smem = gl.allocate_shared_memory(gl.float32, [2, BM], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [NBUF, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [NBUF, 1], mbarrier.MBarrierLayout())
    x_full = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    x_done = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    sring = gl.allocate_shared_memory(gl.int64, [SRING_I64, 1], mbarrier.MBarrierLayout())   # cp.async scale ring
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    for i in gl.static_range(NBUF):
        mbarrier.init(acc_ready.index(i), count=1)
        if RAW != 0 and RAW != 2:
            mbarrier.init(acc_empty.index(i), count=8)   # every promotion warp releases a buffer
        else:
            mbarrier.init(acc_empty.index(i), count=2)   # both promotion sets release a buffer
    for i in gl.static_range(2):
        mbarrier.init(x_full.index(i), count=1)
        mbarrier.init(x_done.index(i), count=1)
    mbarrier.init(sring.index(0), count=1)   # marker: its address (parsed from the PTX) locates the ring
    fence_async_shared()

    tmem_layout: gl.constexpr = TensorMemoryLayout(block=(BM, BN), col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [NBUF, BM, BN], tmem_layout)
    # TMEM base address: tcgen05.alloc wrote it to a smem slot (offset off_tmem, parsed from the PTX) that is reused by
    # the stage buffers once TMA traffic starts, so read it here, before the partitions start.
    off_tmem = _lds32(_smem_base(pid) + off_tmem)

    if EPI == 1 and PREF > 0:
        gl.warp_specialize([
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 0, KT, EPI, num_warps)),
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 1, KT, EPI, num_warps)),
            (_loader, (a_desc, b_desc, a_base, b_base, a_smem, b_smem, ready, empty, info_ptr, n_units, pid, nprog, KT,
                       EPI, STAGES)),
            (_mma, (a_smem, b_smem, ready, empty, acc_ready, acc_empty, acc_tmem, n_units, pid, nprog, KT, STAGES)),
            (_zero_out, (out_ptr, out_numel, pid, nprog)),
            (_prefetcher, (a_base, b_base, empty, info_ptr, n_units, pid, nprog, KT, EPI, STAGES)),
        ], [4, 1, 1, 1, 1], [REGS1, 24, 24, 24, 24])
    elif EPI == 1:
        gl.warp_specialize([
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 0, KT, EPI, num_warps)),
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 1, KT, EPI, num_warps)),
            (_loader, (a_desc, b_desc, a_base, b_base, a_smem, b_smem, ready, empty, info_ptr, n_units, pid, nprog, KT,
                       EPI, STAGES)),
            (_mma, (a_smem, b_smem, ready, empty, acc_ready, acc_empty, acc_tmem, n_units, pid, nprog, KT, STAGES)),
            (_zero_out, (out_ptr, out_numel, pid, nprog)),
        ], [4, 1, 1, 1], [REGS1, 24, 24, 24])
    elif PREF > 0:
        gl.warp_specialize([
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 0, KT, EPI, num_warps)),
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 1, KT, EPI, num_warps)),
            (_loader, (a_desc, b_desc, a_base, b_base, a_smem, b_smem, ready, empty, info_ptr, n_units, pid, nprog, KT,
                       EPI, STAGES)),
            (_mma, (a_smem, b_smem, ready, empty, acc_ready, acc_empty, acc_tmem, n_units, pid, nprog, KT, STAGES)),
            (_prefetcher, (a_base, b_base, empty, info_ptr, n_units, pid, nprog, KT, EPI, STAGES)),
        ], [4, 1, 1, 1], [REGS2, 24, 24, 24])
    else:
        gl.warp_specialize([
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 0, KT, EPI, num_warps)),
            (_epi_set, (c_ptr, cs_ptr, out_ptr, row_tok_ptr, row_w_ptr, sa_ptr, sbw_ptr, SST, P_MAX, info_ptr,
                        acc_ready, acc_empty, acc_tmem, xch_smem, x_full, x_done, n_units, pid, nprog, off_tmem, off_ar,
                        off_ae, off_ring, 1, KT, EPI, num_warps)),
            (_loader, (a_desc, b_desc, a_base, b_base, a_smem, b_smem, ready, empty, info_ptr, n_units, pid, nprog, KT,
                       EPI, STAGES)),
            (_mma, (a_smem, b_smem, ready, empty, acc_ready, acc_empty, acc_tmem, n_units, pid, nprog, KT, STAGES)),
        ], [4, 1, 1], [REGS2, 24, 24])


# ----------------------------------------------------------------------------------------------------------------------
# host side
class _Workspace:
    """Shape-only buffers (allocated once per (T, device)); no input-derived data survives between calls."""

    def __init__(self, T, device):
        self.T = T
        self.P_MAX = T * TOPK + E_LOCAL * (BLK - 1)
        dev = device
        self.ids = torch.empty((T, TOPK), dtype=torch.int32, device=dev)
        self.w = torch.empty((T, TOPK), dtype=torch.float32, device=dev)
        self.counts = torch.zeros(E_LOCAL * 16, dtype=torch.int32, device=dev)
        self.slot_base = torch.zeros(E_LOCAL * 16, dtype=torch.int32, device=dev)
        self.cnt = torch.zeros(E_LOCAL, dtype=torch.int32, device=dev)
        self.pair_le = torch.empty(T * TOPK, dtype=torch.int32, device=dev)
        self.pair_rank = torch.empty(T * TOPK, dtype=torch.int32, device=dev)
        self.offs = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
        self.t1 = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
        self.t2 = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device=dev)
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.max_u1 = (self.P_MAX // 128) * (I // 128)
        self.max_u2 = (self.P_MAX // 128) * (H // 256)
        self.info1 = torch.zeros(self.max_u1 * 4, dtype=torch.int32, device=dev)
        self.info2 = torch.zeros(self.max_u2 * 4, dtype=torch.int32, device=dev)
        self.a = torch.empty((self.P_MAX, H), dtype=torch.float8_e4m3fn, device=dev)
        self.sa = torch.zeros((KB1, self.P_MAX), dtype=torch.float32, device=dev)   # A-row-ordered block scales (GEMM1)
        self.out_ring = [torch.empty((T, H), dtype=torch.bfloat16, device=dev) for _ in range(N_GRAPH)]
        self.out_idx = 0
        self.st_logits = torch.empty((T, E_GLOBAL), dtype=torch.float32, device=dev)
        self.st_bias = torch.empty((E_GLOBAL,), dtype=torch.bfloat16, device=dev)
        self.st_hs = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=dev)
        self.st_hss = torch.empty((KB1, T), dtype=torch.float32, device=dev)
        self.graph_key = None
        self.graphs = []
        self.row_tok = torch.empty(self.P_MAX, dtype=torch.int32, device=dev)
        self.row_w = torch.empty(self.P_MAX, dtype=torch.float32, device=dev)
        self.c = torch.empty((self.P_MAX, I), dtype=torch.float8_e4m3fn, device=dev)
        self.cs = torch.empty((KB2, self.P_MAX), dtype=torch.float32, device=dev)
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count

        a_layout = gl.NVMMASharedLayout.get_default_for([128, 128], gl.float8e4nv)
        self.a_layout = a_layout
        self.b1_layout = gl.NVMMASharedLayout.get_default_for([64, 128], gl.float8e4nv)
        self.b2_layout = gl.NVMMASharedLayout.get_default_for([128, 128], gl.float8e4nv)


_WS = {}


def _workspace(T, device):
    key = (T, str(device))
    ws = _WS.get(key)
    if ws is None:
        ws = _Workspace(T, device)
        _WS[key] = ws
    return ws


USE_GRAPH = int(os.environ.get('MOE_GRAPH', '1'))
N_GRAPH = 4


def _launch_all(ws, out, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
                gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_off, scaling):
    """All device work of one call (captured into a CUDA graph when the buffer addresses are stable)."""
    T = ws.T
    ws.counts.zero_()
    _routing_kernel[(triton.cdiv(T, RT_TPB),)](routing_logits, routing_bias, ws.ids, ws.w, ws.counts, ws.pair_le,
                                               ws.pair_rank, T, local_off, scaling, E=E_GLOBAL, NG=N_GROUP,
                                               TG=TOPK_GROUP, K=TOPK, EL=E_LOCAL, TPB=RT_TPB, num_warps=RT_TPB)
    _offsets_kernel[(1,)](ws.counts, ws.slot_base, ws.cnt, ws.offs, ws.t1, ws.t2, ws.meta, EL=E_LOCAL, NQ1=I // 128,
                          NT2=H // 256, num_warps=1)
    _unit_table_kernel[(64,)](ws.t1, ws.t2, ws.offs, ws.meta, ws.info1, ws.info2, EL=E_LOCAL, UB=128, num_warps=4)
    _gather_kernel[(triton.cdiv(T * TOPK + E_LOCAL * 128, 512),)](ws.pair_le, ws.pair_rank, ws.w, ws.offs, ws.slot_base,
                                                                   ws.cnt, ws.row_tok, ws.row_w, T, EL=E_LOCAL, K=TOPK,
                                                                   PB=512, num_warps=4)
    _copy_rows_kernel[(triton.cdiv(ws.P_MAX, CP_RPB),)](ws.row_tok, hidden_states, ws.a, hidden_states_scale, ws.sa,
                                                        ws.meta, T, ws.P_MAX, HD=H, VEC=CP_VEC, RPB=CP_RPB, KB=KB1,
                                                        num_warps=CP_NW)
    a1_desc = TensorDescriptor.from_tensor(ws.a, [128, 128], ws.a_layout)
    b1_desc = TensorDescriptor.from_tensor(gemm1_weights.view(E_LOCAL * 2 * I, H), [64, 128], ws.b1_layout)
    grid = (ws.num_sms,)
    _launch_gemm(grid, (a1_desc, b1_desc, ws.a, gemm1_weights, ws.sa, gemm1_weights_scale, ws.c,
                        ws.cs, out, ws.row_tok, ws.row_w, ws.info1, ws.meta, ws.P_MAX, T, ws.P_MAX, out.numel()),
                 KB1, 1, STAGES1)
    a2_desc = TensorDescriptor.from_tensor(ws.c, [128, 128], ws.a_layout)
    b2_desc = TensorDescriptor.from_tensor(gemm2_weights.view(E_LOCAL * H, I), [256, 128], ws.b2_layout)
    _launch_gemm(grid, (a2_desc, b2_desc, ws.c, gemm2_weights, ws.cs, gemm2_weights_scale, ws.c, ws.cs, out,
                        ws.row_tok, ws.row_w, ws.info2, ws.meta, ws.P_MAX, T, ws.P_MAX, out.numel()),
                 KB2, 2, STAGES2)


_SMEM_OFFS = {}


def _resolve_smem_offset(ptx, reg):
    """Offset of smem address register `reg` relative to global_smem (follows add.s32 chains)."""
    off = 0
    for _ in range(8):
        if re.search(r"mov\.b32\s+%s, global_smem;" % re.escape(reg), ptx):
            return off
        m = re.search(r"add\.s32\s+%s, (%%r\d+), (-?\d+);" % re.escape(reg), ptx)
        if m is None:
            raise RuntimeError("cannot resolve smem offset of " + reg)
        reg, off = m.group(1), off + int(m.group(2))
    raise RuntimeError("smem offset chain too deep")


def _gemm_smem_offsets(ptx, stages):
    """(tmem base slot, acc_ready[0], acc_empty[0]) smem offsets, read from the compiled kernel's PTX.
    Barrier inits appear in program order: (ready[i], empty[i]) x S, (acc_ready[i], acc_empty[i]) x 2, (x_full, x_done) x 2."""
    inits = re.findall(r"mbarrier\.init\.shared::cta\.b64 \[(%r\d+)\], \d+;", ptx)
    if len(inits) != 2 * stages + 9:
        raise RuntimeError("unexpected number of mbarrier inits: %d" % len(inits))
    offs = [_resolve_smem_offset(ptx, r) for r in inits]
    ar, ae = offs[2 * stages], offs[2 * stages + 1]
    if offs[2 * stages + 2] != ar + 8 or offs[2 * stages + 3] != ae + 8:
        raise RuntimeError("acc barrier layout not contiguous: %s" % offs)
    m = re.search(r"tcgen05\.alloc\.cta_group::1\.sync\.aligned\.shared::cta\.b32 \[(%r\d+)\], \d+;", ptx)
    tm = _resolve_smem_offset(ptx, m.group(1))
    return tm, ar, ae, offs[-1]


def _launch_gemm(grid, args, KT, EPI, STAGES):
    key = (EPI, STAGES)
    if key not in _SMEM_OFFS:
        ck = _moe_gemm_kernel.warmup(*args, 0, 0, 0, 0, KT=KT, EPI=EPI, STAGES=STAGES, num_warps=4, grid=(1,))
        _SMEM_OFFS[key] = _gemm_smem_offsets(ck.asm["ptx"], STAGES)
    tm, ar, ae, rg = _SMEM_OFFS[key]
    _moe_gemm_kernel[grid](*args, tm, ar, ae, rg, KT=KT, EPI=EPI, STAGES=STAGES, num_warps=4)


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor):
    T = routing_logits.shape[0]
    dev = hidden_states.device
    ws = _workspace(T, dev)
    local_off = int(local_expert_offset)
    scaling = float(routed_scaling_factor)
    slot = ws.out_idx
    ws.out_idx = (ws.out_idx + 1) % N_GRAPH
    out = ws.out_ring[slot]
    if not USE_GRAPH:
        _launch_all(ws, out, routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
                    gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_off, scaling)
        return out
    # stage the per-call inputs into fixed buffers so the captured graph sees stable addresses
    srcs = (routing_logits, routing_bias, hidden_states, hidden_states_scale)
    dsts = (ws.st_logits, ws.st_bias, ws.st_hs, ws.st_hss)
    if all(x.is_contiguous() for x in srcs):
        bs = [x.numel() * x.element_size() for x in srcs]
        BLK = 16384
        nprog = sum(triton.cdiv(n, BLK) for n in bs)
        vs = [x.view(torch.uint8).view(-1) for x in srcs]
        vd = [x.view(torch.uint8).view(-1) for x in dsts]
        _stage_kernel[(nprog,)](vs[0], vd[0], bs[0], vs[1], vd[1], bs[1], vs[2], vd[2], bs[2], vs[3], vd[3], bs[3],
                                BLOCK=BLK, num_warps=8)
    else:
        for x, y in zip(srcs, dsts):
            y.copy_(x)
    key = (gemm1_weights.data_ptr(), gemm1_weights_scale.data_ptr(), gemm2_weights.data_ptr(),
           gemm2_weights_scale.data_ptr(), local_off, scaling)
    if ws.graph_key != key:
        args = (ws.st_logits, ws.st_bias, ws.st_hs, ws.st_hss, gemm1_weights, gemm1_weights_scale, gemm2_weights,
                gemm2_weights_scale, local_off, scaling)
        # eager warm-up (compiles everything), then capture one graph per output slot
        _launch_all(ws, out, *args)
        torch.cuda.synchronize(dev)
        ws.graphs = []
        for j in range(N_GRAPH):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _launch_all(ws, ws.out_ring[j], *args)
            ws.graphs.append(g)
        ws.graph_key = key
        return out
    ws.graphs[slot].replay()
    return out
