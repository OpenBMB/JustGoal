# From-scratch FlashAttention-style varlen causal backward for Blackwell (sm_100), written in Gluon.
# Layout (T, H, D) bf16, packed docs with cu_seqlens.
#
# Per CTA work item = (doc, kv-block of 128 rows, head). K, V resident in smem; loop over q-blocks of 64 rows
# (causal: only q-blocks at/after the kv block). Transposed formulation so that the softmax outputs land directly
# as MMA A-operands:
#   S^T = K Q^T, dP^T = V dO^T                     (TMEM, 128x64 fp32, 2 stages each)
#   P^T = exp2(S^T*scale*log2e - lse2[q]), dS^T = P^T (dP^T - delta[q])   -> smem bf16 (2 stages)
#   dV += P^T dO ; dK += dS^T Q                    (TMEM accumulators 128x128 fp32)
#   dQ^T = K^T dS^T (128 d x 64 q) -> reuses the dP^T TMEM slot -> epilogue warps -> coalesced red.global.add.f32
# dQ accumulated in fp32 (TPP, H, D) (T padded to a multiple of 64), then scaled/converted by a tiny postprocess.
# Warp roles: 8 softmax warps (default partition), 1 TMA loader warp, 1 MMA issuer warp, 4 epilogue warps.
import math
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, tma, mbarrier,
    tcgen05_mma, tcgen05_commit, fence_async_shared)

LOG2E = 1.4426950408889634


def _fit_exp2_poly():
    # degree-3 polynomial for 2^f on [0,1), fitted for relative error (max rel err ~1e-4)
    f = torch.linspace(0, 1, 100001, dtype=torch.float64)[:-1]
    y = 2.0 ** f
    A = torch.stack([torch.ones_like(f), f, f ** 2, f ** 3], 1) / y[:, None]
    c = torch.linalg.lstsq(A, torch.ones_like(f)[:, None]).solution[:, 0]
    for _ in range(30):
        p = ((c[3] * f + c[2]) * f + c[1]) * f + c[0]
        w = ((p / y - 1).abs() + 1e-9) ** 0.5
        c = torch.linalg.lstsq(A * w[:, None], (torch.ones_like(f) * w)[:, None]).solution[:, 0]
    p = ((c[3] * f + c[2]) * f + c[1]) * f + c[0]
    return [float(v) for v in c], float((p / y - 1).abs().max())


EXP2_COEFFS, EXP2_MAXERR = _fit_exp2_poly()


def _f32hex(x):
    import struct
    return "%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


EXP2_ASM = None  # built lazily (needs gl)
_SMEM_SCAN_CACHE = {}


# ----------------------------------------------------------------------------- helpers
@gluon.jit
def _named_sync(dummy, ID: gl.constexpr, N: gl.constexpr):
    # intra-partition barrier (all N threads of this partition)
    return gl.inline_asm_elementwise(f"bar.sync {ID}, {N}; mov.b32 $0, $1;", "=r,r", [dummy],
                                     dtype=gl.int32, is_pure=False, pack=1)


# 4x4 transpose inside each quad of lanes (lane%4 = row a of the quad, 4 packed consecutive columns).
# After it, lane (4g+a) holds column (c+a) for rows 4g..4g+3 -> 4 consecutive d values for one q.
_QUAD_T_ASM = gl.constexpr("""{
.reg .pred p0, p1;
.reg .u32 lid, b0, b1;
.reg .f32 s0, s1, r0, r1, y0, y1, y2, y3;
mov.u32 lid, %laneid;
and.b32 b0, lid, 1;
and.b32 b1, lid, 2;
setp.ne.u32 p0, b0, 0;
selp.f32 s0, $4, $5, p0;
selp.f32 s1, $6, $7, p0;
shfl.sync.bfly.b32 r0, s0, 1, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 r1, s1, 1, 0x1f, 0xffffffff;
selp.f32 y0, r0, $4, p0;
selp.f32 y1, $5, r0, p0;
selp.f32 y2, r1, $6, p0;
selp.f32 y3, $7, r1, p0;
setp.ne.u32 p1, b1, 0;
selp.f32 s0, y0, y2, p1;
selp.f32 s1, y1, y3, p1;
shfl.sync.bfly.b32 r0, s0, 2, 0x1f, 0xffffffff;
shfl.sync.bfly.b32 r1, s1, 2, 0x1f, 0xffffffff;
selp.f32 $0, r0, y0, p1;
selp.f32 $1, r1, y1, p1;
selp.f32 $2, y2, r0, p1;
selp.f32 $3, y3, r1, p1;
}""")


@gluon.jit
def _quad_transpose(x):
    return gl.inline_asm_elementwise(_QUAD_T_ASM, "=f,=f,=f,=f,f,f,f,f", [x], dtype=gl.float32, is_pure=False, pack=4)


@gluon.jit
def _red_add_v4(val, ptr_i64):
    # red.global.add.v4.f32 of 4 packed consecutive values at the (16B aligned) address of the first
    return gl.inline_asm_elementwise(
        "red.relaxed.gpu.global.add.v4.f32 [$8], {$4, $5, $6, $7}; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;",
        "=r,=r,=r,=r,f,f,f,f,l,l,l,l", [val, ptr_i64], dtype=gl.int32, is_pure=False, pack=4)


@gluon.jit
def _red_add_v2_f16x2(val, ptr_i64):
    # 4 packed fp32 (consecutive d) -> 2 x f16x2 -> red.global.add.noftz.v2.f16x2 (8B, at the address of the first)
    return gl.inline_asm_elementwise(
        "{ .reg .b32 h0, h1; cvt.rn.f16x2.f32 h0, $5, $4; cvt.rn.f16x2.f32 h1, $7, $6; "
        "red.relaxed.gpu.global.add.noftz.v2.f16x2 [$8], {h0, h1}; } mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;",
        "=r,=r,=r,=r,f,f,f,f,l,l,l,l", [val, ptr_i64], dtype=gl.int32, is_pure=False, pack=4)


_EXP2_MIX_ASM = gl.constexpr("""{
.reg .f32 xc, n, f, y;
.reg .s32 ni, yb;
ex2.approx.ftz.f32 $0, $2;
max.f32 xc, $3, 0fC2FC0000;
cvt.rmi.f32.f32 n, xc;
sub.f32 f, xc, n;
fma.rn.f32 y, f, 0f{C3}, 0f{C2};
fma.rn.f32 y, y, f, 0f{C1};
fma.rn.f32 y, y, f, 0f{C0};
cvt.rzi.s32.f32 ni, n;
shl.b32 ni, ni, 23;
mov.b32 yb, y;
add.s32 yb, yb, ni;
mov.b32 $1, yb;
}""")


@gluon.jit
def _exp2_mixed(x, ASM: gl.constexpr, SPLIT: gl.constexpr):
    # pairs of consecutive columns: even -> MUFU ex2, odd -> FMA-pipe polynomial (2^n * poly(f))
    if SPLIT:
        return gl.inline_asm_elementwise(ASM, "=f,=f,f,f", [x], dtype=gl.float32, is_pure=True, pack=2)
    else:
        return gl.exp2(x)


@gluon.jit
def _pack_f16x2(a, b):
    # (a, b) fp32 -> f16x2 word (a in the low half)
    return gl.inline_asm_elementwise("cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f", [a, b], dtype=gl.int32, is_pure=True, pack=1)


@gluon.jit
def _red_v2_f16x2_packed(pk, off, base):
    # pk: packed f16x2 words (pairs = 4 consecutive d), off: byte offsets (int32), base: int64 base pointer
    return gl.inline_asm_elementwise(
        "{ .reg .b64 a; mad.wide.s32 a, $4, 1, $6; red.relaxed.gpu.global.add.noftz.v2.f16x2 [a], {$2, $3}; } mov.b32 $0, 0; mov.b32 $1, 0;",
        "=r,=r,r,r,r,r,l,l", [pk, off, base], dtype=gl.int32, is_pure=False, pack=2)


@gluon.jit
def _dq_reds(dqh, base, half: gl.constexpr, HB: gl.constexpr, HD: gl.constexpr, ABL_NOATOMIC: gl.constexpr):
    # dqh: [D, HB] fp32 (already scaled) in the TMEM 32x32b register layout; base: [D] int64 per-thread pointers
    z = _quad_transpose(dqh)
    blocks = _split_cols(z, HB // 4)
    bl: gl.constexpr = blocks[0].type.layout
    rowp = gl.convert_layout(base, gl.SliceLayout(1, bl))
    colz = gl.arange(0, 4, layout=gl.SliceLayout(0, bl)) * 0
    p2 = rowp[:, None] + (colz.to(gl.int64))[None, :] + (half * HB * HD * 2)
    if not ABL_NOATOMIC:
        for m in gl.static_range(HB // 4):
            _red_add_v2_f16x2(blocks[m], p2 + (4 * m * HD * 2))


_PROBE_ASM = gl.constexpr("""{
.reg .u32 base, addr, v0, v1, found;
.reg .pred p1, p2;
mov.u32 base, global_smem;
mov.u32 found, 0xFFFFFFFF;
add.u32 addr, base, $1;
ld.shared.b32 v0, [addr];
ld.shared.b32 v1, [addr+4];
setp.eq.u32 p1, v0, $2;
setp.eq.u32 p2, v1, $3;
and.pred p1, p1, p2;
@p1 mov.u32 found, $1;
mov.u32 $0, found;
}""")

_BULK_RED_ASM = gl.constexpr("""{
.reg .u32 base, saddr;
.reg .pred p;
setp.ne.u32 p, $3, 0;
mov.u32 base, global_smem;
add.u32 saddr, base, $2;
@p cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$4], [saddr], 128;
@p cp.async.bulk.commit_group;
mov.u32 $0, $1;
}""")


@gluon.jit
def _smem_probe(stage, TAG: gl.constexpr, BAR_ID: gl.constexpr, SMEM_SCAN: gl.constexpr):
    # write a (group-specific) marker into the first 8 words of `stage`, then find its byte offset in global_smem
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    idx = gl.arange(0, 128, layout=lay)
    M0: gl.constexpr = 0x5A5A0000 + TAG * 0x1000
    M1: gl.constexpr = 0x3C3C0000 + TAG * 0x1000 + 0x101
    marker = gl.where(idx == 0, M0, gl.where(idx == 1, M1, 0))
    stage.slice(0, 128).store(marker)
    dummy = gl.zeros([1], gl.int32, layout=gl.BlockedLayout([1], [32], [4], [0]))
    dummy = _named_sync(dummy, BAR_ID, 128)
    found = gl.full([128], 0xFFFFFFFF, gl.int32, layout=lay)
    m0 = gl.full([128], M0, gl.int32, layout=lay)
    m1 = gl.full([128], M1, gl.int32, layout=lay)
    for k in gl.static_range((SMEM_SCAN + 128 * 1024 - 1) // (128 * 1024)):
        cand = (idx + k * 128) * 1024
        cand = gl.where(cand < SMEM_SCAN, cand, 0)
        f = gl.inline_asm_elementwise(_PROBE_ASM, "=r,r,r,r", [cand, m0, m1], dtype=gl.int32, is_pure=False, pack=1)
        found = gl.minimum(found.to(gl.uint32), f.to(gl.uint32)).to(gl.int32)
    off = gl.min(found.to(gl.uint32), axis=0).to(gl.int32)
    return off


@gluon.jit
def _sts_v4(words, addr):
    # words: int32 [.., 4k] (quads), addr: int32 smem byte offsets (from global_smem), pack=4
    return gl.inline_asm_elementwise(
        "{ .reg .u32 b, a; mov.u32 b, global_smem; add.u32 a, b, $8; st.shared.v4.b32 [a], {$4, $5, $6, $7}; } "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;",
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r", [words, addr], dtype=gl.int32, is_pure=False, pack=4)


@gluon.jit
def _sts_v2(words, addr):
    # words: int32 [.., 2k] (pairs), addr: int32 smem byte offsets (from global_smem), pack=2
    return gl.inline_asm_elementwise(
        "{ .reg .u32 b, a; mov.u32 b, global_smem; add.u32 a, b, $4; st.shared.v2.b32 [a], {$2, $3}; } mov.b32 $0, 0; mov.b32 $1, 0;",
        "=r,=r,r,r,r,r", [words, addr], dtype=gl.int32, is_pure=False, pack=2)


@gluon.jit
def _bulk_wait_read(dummy):
    return gl.inline_asm_elementwise("cp.async.bulk.wait_group.read 0; mov.b32 $0, $1;", "=r,r", [dummy], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _bulk_wait_all(dummy):
    return gl.inline_asm_elementwise("cp.async.bulk.wait_group 0; mov.b32 $0, $1;", "=r,r", [dummy], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _split_cols(x, SPLIT: gl.constexpr):
    # [R, C] -> SPLIT x [R, C // SPLIT] column blocks, keeping each thread's consecutive columns together
    split_count: gl.constexpr = SPLIT.bit_length() - 1
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            t = xs[j]
            next_xs += t.reshape(t.shape[0], 2, t.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs


@gluon.jit
def _trace(trace_ptr, part: gl.constexpr, it, ev: gl.constexpr, TRACE: gl.constexpr, NW: gl.constexpr):
    # record a timestamp for CTA 0 only: trace[part][it][ev]
    if TRACE:
        if gl.program_id(0) == 0:
            lay: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
            offs = gl.arange(0, 32 * NW, layout=lay)
            t = gl.inline_asm_elementwise("mov.u32 $0, %globaltimer_lo; add.u32 $0, $0, $1;", "=r,r", [offs * 0],
                                          dtype=gl.int32, is_pure=False, pack=1)
            gl.store(trace_ptr + ((part * 256 + it) * 8 + ev) + offs * 0, t, mask=offs == 0)


@gluon.jit
def _decode(work_ptr, cu_ptr, w, BM: gl.constexpr, BN: gl.constexpr):
    item = gl.load(work_ptr + w)
    h = item & 0xFF
    j = (item >> 8) & 0xFF
    d = item >> 16
    start = gl.load(cu_ptr + d)
    end = gl.load(cu_ptr + d + 1)
    L = end - start
    n_q = (L + BN - 1) // BN
    i0 = (j * BM) // BN
    n_iter = n_q - i0
    return h, j, d, start, L, i0, n_iter


# ----------------------------------------------------------------------------- loader (1 warp)
@gluon.jit
def _loader(q_desc, k_desc, v_desc, do_desc, q_bufs, do_bufs, k_buf, v_buf, lse_ring, lse2_ptr, delta_ptr,
            kv_ready, kv_empty, q_ready, q_empty, work_ptr, cu_ptr, n_work, T, trace_ptr,
            BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr, TRACE: gl.constexpr,
            ABL_NOLOAD: gl.constexpr):
    vlay: gl.constexpr = gl.BlockedLayout([2], [32], [1], [0])
    cols = gl.arange(0, BN, layout=vlay)
    it = 0
    wc = 0
    for w in range(gl.program_id(0), n_work, gl.num_programs(0)):
        h, j, d, start, L, i0, n_iter = _decode(work_ptr, cu_ptr, w, BM, BN)
        mbarrier.wait(kv_empty, (wc & 1) ^ 1)
        mbarrier.expect(kv_ready, 2 * k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [start + j * BM, h * D], kv_ready, k_buf)
        tma.async_copy_global_to_shared(v_desc, [start + j * BM, h * D], kv_ready, v_buf)
        # prefetch lse2/delta of the first q-block of this item
        q_i = i0 * BN + cols
        ok = q_i < L
        nlse = gl.load(lse2_ptr + h * T + start + q_i, mask=ok, other=float("inf"))
        ndel = gl.load(delta_ptr + h * T + start + q_i, mask=ok, other=0.0)
        for i in range(n_iter):
            s = it % NQS
            mbarrier.wait(q_empty.index(s), ((it // NQS) & 1) ^ 1)
            _trace(trace_ptr, 0, it, 0, TRACE, 1)
            if ABL_NOLOAD:
                mbarrier.arrive(q_ready.index(s), count=1)
            else:
                mbarrier.expect(q_ready.index(s), 2 * q_desc.block_type.nbytes)
                row = start + (i0 + i) * BN
                tma.async_copy_global_to_shared(q_desc, [row, h * D], q_ready.index(s), q_bufs.index(s))
                tma.async_copy_global_to_shared(do_desc, [row, h * D], q_ready.index(s), do_bufs.index(s))
            lse_ring.index(2 * s).store(nlse)
            lse_ring.index(2 * s + 1).store(ndel)
            mbarrier.arrive(q_ready.index(s), count=1)
            _trace(trace_ptr, 0, it, 1, TRACE, 1)
            # prefetch next q-block's lse2/delta
            q_i = (i0 + i + 1) * BN + cols
            ok = (q_i < L) & (i + 1 < n_iter)
            nlse = gl.load(lse2_ptr + h * T + start + q_i, mask=ok, other=float("inf"))
            ndel = gl.load(delta_ptr + h * T + start + q_i, mask=ok, other=0.0)
            it += 1
        wc += 1


# ----------------------------------------------------------------------------- MMA issuers (2 warps)
# TMEM plan (512 cols): dK 128 | dV 128 | S^T (64) | dQ^T (64) | dP^T (64) | P^T bf16 x2 (32 each)
# Front warp: S^T(it+2) (after the softmax loaded S^T(it+1): s_free) and dP^T(it+1) (after pds_ready(it)).
# Back warp:  dQ^T(it), dV(it), dK(it) after pds_ready(it) (dQ^T also waits for the epilogue drain of dQ^T(it-1)).
@gluon.jit
def _mma_front(q_bufs, do_bufs, k_buf, v_buf, s_t, dp_t,
               kv_ready, kv_empty, q_ready, s_ready, s_free, dp_ready, pds_ready,
               work_ptr, cu_ptr, n_work, trace_ptr,
               BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr, TRACE: gl.constexpr):
    it = 0
    wc = 0
    loaded = 0
    for w in range(gl.program_id(0), n_work, gl.num_programs(0)):
        h, j, d, start, L, i0, n_iter = _decode(work_ptr, cu_ptr, w, BM, BN)
        mbarrier.wait(kv_ready, wc & 1)
        # prologue: S^T(0), dP^T(0), S^T(1)
        s0 = it % NQS
        if loaded <= it:
            mbarrier.wait(q_ready.index(s0), (it // NQS) & 1)
            loaded = it + 1
        if it > 0:
            mbarrier.wait(s_free, (it - 1) & 1)                 # S^T(it-1) consumed
        tcgen05_mma(k_buf, q_bufs.index(s0).permute((1, 0)), s_t, use_acc=False)
        tcgen05_commit(s_ready)
        tcgen05_mma(v_buf, do_bufs.index(s0).permute((1, 0)), dp_t, use_acc=False)
        tcgen05_commit(dp_ready)
        if n_iter > 1:
            s1 = (it + 1) % NQS
            if loaded <= it + 1:
                mbarrier.wait(q_ready.index(s1), ((it + 1) // NQS) & 1)
                loaded = it + 2
            mbarrier.wait(s_free, it & 1)                       # S^T(it) consumed
            tcgen05_mma(k_buf, q_bufs.index(s1).permute((1, 0)), s_t, use_acc=False)
            tcgen05_commit(s_ready)
        for i in range(n_iter):
            _trace(trace_ptr, 1, it, 0, TRACE, 1)
            if i + 1 < n_iter:
                sn = (it + 1) % NQS
                if loaded <= it + 1:
                    mbarrier.wait(q_ready.index(sn), ((it + 1) // NQS) & 1)
                    loaded = it + 2
                mbarrier.wait(pds_ready.index(it % 2), (it // 2) & 1)      # softmax(it) done with dP^T(it)
                _trace(trace_ptr, 1, it, 1, TRACE, 1)
                tcgen05_mma(v_buf, do_bufs.index(sn).permute((1, 0)), dp_t, use_acc=False)
                tcgen05_commit(dp_ready)
            if i + 2 < n_iter:
                sn2 = (it + 2) % NQS
                if loaded <= it + 2:
                    mbarrier.wait(q_ready.index(sn2), ((it + 2) // NQS) & 1)
                    loaded = it + 3
                mbarrier.wait(s_free, (it + 1) & 1)             # S^T(it+1) consumed
                _trace(trace_ptr, 1, it, 2, TRACE, 1)
                tcgen05_mma(k_buf, q_bufs.index(sn2).permute((1, 0)), s_t, use_acc=False)
                tcgen05_commit(s_ready)
            if i + 1 == n_iter:
                tcgen05_commit(kv_empty)                         # (count 2: front + back)
            _trace(trace_ptr, 1, it, 3, TRACE, 1)
            it += 1
        wc += 1


@gluon.jit
def _mma_back(q_bufs, do_bufs, k_buf, dsT_bufs, dq_t, dk_t, dv_t, p_tm0, p_tm1,
              kv_ready, kv_empty, q_empty, pds_ready, pds_empty, dq_ready, dq_empty, dkdv_ready, dkdv_empty,
              work_ptr, cu_ptr, n_work, trace_ptr,
              BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, NQS: gl.constexpr, TRACE: gl.constexpr):
    it = 0
    wc = 0
    for w in range(gl.program_id(0), n_work, gl.num_programs(0)):
        h, j, d, start, L, i0, n_iter = _decode(work_ptr, cu_ptr, w, BM, BN)
        mbarrier.wait(kv_ready, wc & 1)
        mbarrier.wait(dkdv_empty, (wc & 1) ^ 1)
        for i in range(n_iter):
            si = it % NQS
            ts = it % 2
            _trace(trace_ptr, 4, it, 0, TRACE, 1)
            mbarrier.wait(pds_ready.index(ts), (it // 2) & 1)
            _trace(trace_ptr, 4, it, 1, TRACE, 1)
            if it > 0:
                mbarrier.wait(dq_empty.index(1 - ts), ((it - 1) // 2) & 1)   # dQ^T(it-1) drained
            _trace(trace_ptr, 4, it, 2, TRACE, 1)
            tcgen05_mma(k_buf.permute((1, 0)), dsT_bufs.index(ts), dq_t, use_acc=False)
            tcgen05_commit(dq_ready.index(ts))
            if ts == 0:
                tcgen05_mma(p_tm0, do_bufs.index(si), dv_t, use_acc=(i > 0))
            else:
                tcgen05_mma(p_tm1, do_bufs.index(si), dv_t, use_acc=(i > 0))
            tcgen05_mma(dsT_bufs.index(ts), q_bufs.index(si), dk_t, use_acc=(i > 0))
            tcgen05_commit(pds_empty.index(ts))
            tcgen05_commit(q_empty.index(si))
            if i + 1 == n_iter:
                tcgen05_commit(kv_empty)                         # (count 2: front + back)
                tcgen05_commit(dkdv_ready)
            _trace(trace_ptr, 4, it, 3, TRACE, 1)
            it += 1
        wc += 1


# ----------------------------------------------------------------------------- softmax partition (8 warps): columns [COL0, COL0+HB)
@gluon.jit
def _softmax(s_t, dp_t, p_tm0, p_tm1, dsT_bufs, lse_ring, s_ready, s_free, dp_ready, pds_ready, pds_empty,
             work_ptr, cu_ptr, n_work, scale_log2e, trace_ptr,
             COL0: gl.constexpr, BAR_ID: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr,
             NQS: gl.constexpr, num_warps: gl.constexpr, EXP_SPLIT: gl.constexpr, EXP_ASM: gl.constexpr,
             TRACE: gl.constexpr, ABL_NOSTS: gl.constexpr, ABL_NOSTTM: gl.constexpr):
    HB: gl.constexpr = BN // 2
    tl_h: gl.constexpr = TensorMemoryLayout(block=[BM, HB], col_stride=1)
    reg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, [BM, HB], tl_h, num_warps)
    reg_p: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, [BM, HB], tl_h, num_warps)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, reg_l))   # kv within block
    it = 0
    dummy = gl.full([1], 0, gl.int32, layout=gl.BlockedLayout([1], [32], [num_warps], [0]))
    for w in range(gl.program_id(0), n_work, gl.num_programs(0)):
        h, j, d, start, L, i0, n_iter = _decode(work_ptr, cu_ptr, w, BM, BN)
        kv_i = j * BM + rows
        kv_ok = kv_i < L
        full_kv = (j * BM + BM) <= L
        for i in range(n_iter):
            s = it % NQS
            ts = it % 2
            mbarrier.wait(s_ready, it & 1)
            _trace(trace_ptr, 2, it, 0, TRACE, num_warps)
            sT = s_t.slice(COL0, HB).load(reg_l)
            dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
            mbarrier.arrive(s_free, count=1)                     # (count 2: both halves)
            _trace(trace_ptr, 2, it, 1, TRACE, num_warps)
            lse2 = lse_ring.index(2 * s).slice(COL0, HB).load(gl.SliceLayout(0, reg_l))
            p = _exp2_mixed(sT * scale_log2e - lse2[None, :], EXP_ASM, EXP_SPLIT)
            if (i0 + i) * BN < j * BM + BM or not full_kv:
                q_i = (i0 + i) * BN + COL0 + gl.arange(0, HB, layout=gl.SliceLayout(0, reg_l))
                mask = (q_i[None, :] >= kv_i[:, None]) & kv_ok[:, None]
                p = gl.where(mask, p, 0.0)
            pb = gl.convert_layout(p.to(gl.bfloat16), reg_p)
            mbarrier.wait(pds_empty.index(ts), ((it // 2) & 1) ^ 1)   # dQ/dK of iteration it-2 done with dS^T buffer ts
            _trace(trace_ptr, 2, it, 2, TRACE, num_warps)
            if not ABL_NOSTTM:
                if ts == 0:
                    p_tm0.slice(COL0, HB).store(pb)
                else:
                    p_tm1.slice(COL0, HB).store(pb)
            mbarrier.wait(dp_ready, it & 1)
            _trace(trace_ptr, 2, it, 3, TRACE, num_warps)
            dp = dp_t.slice(COL0, HB).load(reg_l)
            _trace(trace_ptr, 2, it, 4, TRACE, num_warps)
            delta = lse_ring.index(2 * s + 1).slice(COL0, HB).load(gl.SliceLayout(0, reg_l))
            ds = p * (dp - delta[None, :])
            dsb = ds.to(gl.bfloat16)
            _trace(trace_ptr, 2, it, 5, TRACE, num_warps)
            if not ABL_NOSTS:
                dsT_bufs.index(ts).slice(COL0, HB, dim=1).store(dsb)
            fence_async_shared()
            _trace(trace_ptr, 2, it, 6, TRACE, num_warps)
            dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
            mbarrier.arrive(pds_ready.index(ts), count=1)        # (count 2: both halves)
            _trace(trace_ptr, 2, it, 7, TRACE, num_warps)
            it += 1


# ----------------------------------------------------------------------------- epilogue group (4 warps), parity PAR
# dQ^T(it) drain: TMEM -> regs (thread = d row, contiguous q) -> fp16 -> smem staging [128 d][128B] (lane-rotated
# chunk order: bank-conflict-light) -> one cp.reduce.async.bulk (128B, fp16 add) per d row into dqacc[h][d][t0:t0+64].
@gluon.jit
def _epilogue(dq_t, dk_t, dv_t, stage, dq_ready, dq_empty, dkdv_ready, dkdv_empty, dqacc_ptr, pad_ptr, dk_ptr, dv_ptr,
              work_ptr, cu_ptr, n_work, TPP, scale, trace_ptr,
              PAR: gl.constexpr, BAR_ID: gl.constexpr, SMEM_SCAN: gl.constexpr,
              H: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr, num_warps: gl.constexpr,
              ABL_NORED: gl.constexpr, ABL_NOEPI: gl.constexpr, TRACE: gl.constexpr):
    HB: gl.constexpr = BN // 2
    RS: gl.constexpr = BN * 2                                       # staging row stride (bytes) = 128
    tl_h: gl.constexpr = TensorMemoryLayout(block=[BM, HB], col_stride=1)
    tl_d: gl.constexpr = TensorMemoryLayout(block=[BM, D // 2], col_stride=1)
    reg_h: gl.constexpr = get_tmem_reg_layout(gl.float32, [BM, HB], tl_h, num_warps)
    reg_d: gl.constexpr = get_tmem_reg_layout(gl.float32, [BM, D // 2], tl_d, num_warps)
    dlay: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    dummy = gl.full([1], 0, gl.int32, layout=dlay)
    stage_off = _smem_probe(stage, PAR, BAR_ID, SMEM_SCAN)
    if gl.program_id(0) == 0:
        gl.store(trace_ptr + 16000 + PAR + gl.arange(0, 32 * num_warps, layout=dlay) * 0,
                 stage_off + gl.arange(0, 32 * num_warps, layout=dlay) * 0, mask=gl.arange(0, 32 * num_warps, layout=dlay) == 0)
    # packed-word layout (derived from a probe conversion): [D, HB//2] words, thread = d row
    probe = gl.zeros([BM, HB], gl.float32, layout=reg_h)
    pa, pb_ = probe.reshape(BM, HB // 2, 2).split()
    pkl: gl.constexpr = _pack_f16x2(pa, pb_).type.layout
    chunks = _split_cols(_pack_f16x2(pa, pb_), HB // 8)             # 4 chunk tensors [D, 4] (16B each)
    cl: gl.constexpr = chunks[0].type.layout
    drow = gl.arange(0, D, layout=gl.SliceLayout(1, cl))            # d row = lane + 32*warp
    r4 = drow % 4                                                   # lane-dependent chunk rotation
    row_base = stage_off + drow * RS
    c4 = gl.arange(0, 4, layout=gl.SliceLayout(0, cl)) * 0
    tid = gl.arange(0, 32 * num_warps, layout=dlay)
    krow = gl.arange(0, BM, layout=gl.SliceLayout(1, reg_d))        # kv within block
    dcol = gl.arange(0, D // 2, layout=gl.SliceLayout(0, reg_d))    # d (half)
    it = 0
    wc = 0
    for w in range(gl.program_id(0), n_work, gl.num_programs(0)):
        h, j, d, start, L, i0, n_iter = _decode(work_ptr, cu_ptr, w, BM, BN)
        pad0 = gl.load(pad_ptr + d)
        for i in range(n_iter):
            if (it % 2) == PAR:
                ph = (it // 2) & 1
                mbarrier.wait(dq_ready.index(PAR), ph)
                _trace(trace_ptr, 3, it, 0, TRACE, num_warps)
                if ABL_NOEPI:
                    mbarrier.arrive(dq_empty.index(PAR), count=1)
                else:
                    t0 = pad0 + (i0 + i) * BN                       # doc-padded column (multiple of 64 -> 128B aligned)
                    dummy = _bulk_wait_read(dummy)                  # staging free (my previous bulk reads done)
                    dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
                    for half in gl.static_range(2):
                        dqh = dq_t.slice(half * HB, HB).load(reg_h) * scale
                        if half == 1:
                            dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
                            mbarrier.arrive(dq_empty.index(PAR), count=1)
                            _trace(trace_ptr, 3, it, 1, TRACE, num_warps)
                        a, b = dqh.reshape(BM, HB // 2, 2).split()
                        ch = _split_cols(_pack_f16x2(a, b), HB // 8)   # 4 chunks of 4 words (q 8c..8c+7)
                        for jj in gl.static_range(4):
                            # at step jj, lane with rotation r stores chunk (jj + r) % 4 at its natural position
                            csel = (jj + r4) % 4
                            wsel = gl.where((csel == 0)[:, None], ch[0], gl.where((csel == 1)[:, None], ch[1],
                                    gl.where((csel == 2)[:, None], ch[2], ch[3])))
                            addr = (row_base + half * (HB * 2) + csel * 16)[:, None] + c4[None, :]
                            _sts_v4(wsel, addr)
                        _trace(trace_ptr, 3, it, 3 + half, TRACE, num_warps)
                    fence_async_shared()
                    _trace(trace_ptr, 3, it, 5, TRACE, num_warps)
                    dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
                    _trace(trace_ptr, 3, it, 6, TRACE, num_warps)
                    # 128 threads each issue one 128B bulk reduce: row d = tid -> dqacc[h][d][t0 : t0+64]
                    gptr = (dqacc_ptr + ((h * D + tid) * TPP + t0)).to(gl.int64)
                    saddr = stage_off + tid * RS
                    pred = (tid < D).to(gl.int32)
                    if not ABL_NORED:
                        gl.inline_asm_elementwise(_BULK_RED_ASM, "=r,r,r,r,l", [tid, saddr, pred, gptr],
                                                  dtype=gl.int32, is_pure=False, pack=1)
                    _trace(trace_ptr, 3, it, 2, TRACE, num_warps)
            it += 1
        if PAR == 0:
            mbarrier.wait(dkdv_ready, wc & 1)
            kv_i = j * BM + krow
            grow = start + kv_i
            obase = (grow[:, None] * H + h) * D + dcol[None, :]
            omask = (kv_i < L)[:, None]
            dk0 = dk_t.slice(0, D // 2).load(reg_d)
            gl.store(dk_ptr + obase, (dk0 * scale).to(gl.bfloat16), mask=omask)
            dk1 = dk_t.slice(D // 2, D // 2).load(reg_d)
            gl.store(dk_ptr + obase + D // 2, (dk1 * scale).to(gl.bfloat16), mask=omask)
            dv0 = dv_t.slice(0, D // 2).load(reg_d)
            gl.store(dv_ptr + obase, dv0.to(gl.bfloat16), mask=omask)
            dv1 = dv_t.slice(D // 2, D // 2).load(reg_d)
            dummy = _named_sync(dummy, BAR_ID, 32 * num_warps)
            mbarrier.arrive(dkdv_empty, count=1)
            gl.store(dv_ptr + obase + D // 2, dv1.to(gl.bfloat16), mask=omask)
        wc += 1
    dummy = _bulk_wait_all(dummy)


# ----------------------------------------------------------------------------- main kernel
@gluon.jit
def _bwd_kernel(q_desc, k_desc, v_desc, do_desc, lse2_ptr, delta_ptr, dqacc_ptr, pad_ptr, dk_ptr, dv_ptr,
                cu_ptr, work_ptr, n_work, T, TPP, scale_log2e, scale, trace_ptr,
                H: gl.constexpr, pds_layout: gl.constexpr, NQS: gl.constexpr, SMB_REGS: gl.constexpr, SMEM_SCAN: gl.constexpr,
                EPI_REGS: gl.constexpr, LOADER_REGS: gl.constexpr, MMA_REGS: gl.constexpr, num_warps: gl.constexpr,
                EXP_SPLIT: gl.constexpr,
                EXP_ASM: gl.constexpr,
                ABL_NOATOMIC: gl.constexpr, ABL_NOEPI: gl.constexpr, ABL_NOLOAD: gl.constexpr, TRACE: gl.constexpr,
                ABL_NOSTS: gl.constexpr, ABL_NOSTTM: gl.constexpr):
    BM: gl.constexpr = k_desc.block_type.shape[0]
    BN: gl.constexpr = q_desc.block_type.shape[0]
    D: gl.constexpr = q_desc.block_type.shape[1]
    q_bufs = gl.allocate_shared_memory(gl.bfloat16, [NQS, BN, D], q_desc.layout)
    do_bufs = gl.allocate_shared_memory(gl.bfloat16, [NQS, BN, D], do_desc.layout)
    k_buf = gl.allocate_shared_memory(gl.bfloat16, [BM, D], k_desc.layout)
    v_buf = gl.allocate_shared_memory(gl.bfloat16, [BM, D], v_desc.layout)
    dsT_bufs = gl.allocate_shared_memory(gl.bfloat16, [2, BM, BN], pds_layout)   # 64B swizzle: column halves are atom-aligned
    stage = gl.allocate_shared_memory(gl.int32, [2, D * BN * 2 // 4], gl.SwizzledSharedLayout(1, 1, 1, order=[0]))
    lse_ring = gl.allocate_shared_memory(gl.float32, [2 * NQS, BN], gl.SwizzledSharedLayout(1, 1, 1, order=[0]))

    kv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dkdv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dkdv_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    q_ready = gl.allocate_shared_memory(gl.int64, [NQS, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [NQS, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    s_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    dp_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    pds_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    pds_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    dq_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    dq_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(kv_ready, count=1)
    mbarrier.init(kv_empty, count=2)         # front + back MMA warps
    mbarrier.init(dkdv_ready, count=1)
    mbarrier.init(dkdv_empty, count=1)
    for i in gl.static_range(NQS):
        mbarrier.init(q_ready.index(i), count=2)     # TMA tx-arrive + loader's lse/delta arrive
        mbarrier.init(q_empty.index(i), count=1)
    mbarrier.init(s_ready, count=1)
    mbarrier.init(s_free, count=2)           # both softmax halves
    mbarrier.init(dp_ready, count=1)
    for i in gl.static_range(2):
        mbarrier.init(pds_empty.index(i), count=1)
        mbarrier.init(pds_ready.index(i), count=2)   # both softmax halves
        mbarrier.init(dq_ready.index(i), count=1)
        mbarrier.init(dq_empty.index(i), count=1)
    fence_async_shared()

    tl_s: gl.constexpr = TensorMemoryLayout(block=[BM, BN], col_stride=1)
    tl_d: gl.constexpr = TensorMemoryLayout(block=[BM, D], col_stride=1)
    s_t = allocate_tensor_memory(gl.float32, [BM, BN], tl_s)
    dq_t = allocate_tensor_memory(gl.float32, [BM, BN], tl_s)
    dp_t = allocate_tensor_memory(gl.float32, [BM, BN], tl_s)
    dk_t = allocate_tensor_memory(gl.float32, [BM, D], tl_d)
    dv_t = allocate_tensor_memory(gl.float32, [BM, D], tl_d)
    p_tm0 = allocate_tensor_memory(gl.bfloat16, [BM, BN], tl_s)
    p_tm1 = allocate_tensor_memory(gl.bfloat16, [BM, BN], tl_s)

    gl.warp_specialize([
        (_softmax, (s_t, dp_t, p_tm0, p_tm1, dsT_bufs, lse_ring, s_ready, s_free, dp_ready, pds_ready, pds_empty,
                    work_ptr, cu_ptr, n_work, scale_log2e, trace_ptr, 0, 9, BM, BN, D, NQS, num_warps, EXP_SPLIT,
                    EXP_ASM, TRACE, ABL_NOSTS, ABL_NOSTTM)),
        (_loader, (q_desc, k_desc, v_desc, do_desc, q_bufs, do_bufs, k_buf, v_buf, lse_ring, lse2_ptr, delta_ptr,
                   kv_ready, kv_empty, q_ready, q_empty, work_ptr, cu_ptr, n_work, T, trace_ptr, BM, BN, D, NQS, TRACE, ABL_NOLOAD)),
        (_mma_front, (q_bufs, do_bufs, k_buf, v_buf, s_t, dp_t,
                      kv_ready, kv_empty, q_ready, s_ready, s_free, dp_ready, pds_ready,
                      work_ptr, cu_ptr, n_work, trace_ptr, BM, BN, D, NQS, TRACE)),
        (_mma_back, (q_bufs, do_bufs, k_buf, dsT_bufs, dq_t, dk_t, dv_t, p_tm0, p_tm1,
                     kv_ready, kv_empty, q_empty, pds_ready, pds_empty, dq_ready, dq_empty, dkdv_ready, dkdv_empty,
                     work_ptr, cu_ptr, n_work, trace_ptr, BM, BN, D, NQS, TRACE)),
        (_softmax, (s_t, dp_t, p_tm0, p_tm1, dsT_bufs, lse_ring, s_ready, s_free, dp_ready, pds_ready, pds_empty,
                    work_ptr, cu_ptr, n_work, scale_log2e, trace_ptr, BN // 2, 11, BM, BN, D, NQS, num_warps, EXP_SPLIT,
                    EXP_ASM, TRACE, ABL_NOSTS, ABL_NOSTTM)),
        (_epilogue, (dq_t, dk_t, dv_t, stage.index(0), dq_ready, dq_empty, dkdv_ready, dkdv_empty, dqacc_ptr, pad_ptr, dk_ptr, dv_ptr,
                     work_ptr, cu_ptr, n_work, TPP, scale, trace_ptr, 0, 10, SMEM_SCAN, H, BM, BN, D, 4, ABL_NOATOMIC, ABL_NOEPI, TRACE)),
        (_epilogue, (dq_t, dk_t, dv_t, stage.index(1), dq_ready, dq_empty, dkdv_ready, dkdv_empty, dqacc_ptr, pad_ptr, dk_ptr, dv_ptr,
                     work_ptr, cu_ptr, n_work, TPP, scale, trace_ptr, 1, 12, SMEM_SCAN, H, BM, BN, D, 4, ABL_NOATOMIC, ABL_NOEPI, TRACE)),
    ], [1, 1, 1, num_warps, 4, 4], [LOADER_REGS, MMA_REGS, MMA_REGS, SMB_REGS, EPI_REGS, EPI_REGS])


# ----------------------------------------------------------------------------- pre/post (plain Triton)
@triton.jit
def _preprocess(o_ptr, do_ptr, lse_ptr, delta_ptr, lse2_ptr, dqacc_ptr, T, TPP, H: tl.constexpr, D: tl.constexpr,
                BT: tl.constexpr, LOG2E_: tl.constexpr):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = (t[:, None] * H + h) * D + d[None, :]
    m = (t < T)[:, None]
    # zero dqacc[h][d][tp] (layout [H][D][TPP], doc-padded columns)
    tl.store(dqacc_ptr + (h * D + d)[:, None] * TPP + t[None, :], tl.zeros([D, BT], tl.float16), mask=(t < TPP)[None, :])
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + off, mask=m, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + h * T + t, delta, mask=t < T)
    lse = tl.load(lse_ptr + h * T + t, mask=t < T, other=0.0)
    tl.store(lse2_ptr + h * T + t, lse * LOG2E_, mask=t < T)


@triton.jit
def _postprocess(dqacc_ptr, dq_ptr, blkdoc_ptr, pad_ptr, cu_ptr, TPP, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    # dq[t][h][d] = dqacc[h][d][tp]  (fp16, already scaled) -> bf16 ; tp = pad[doc] + (t - cu[doc])
    b = tl.program_id(0)
    h = tl.program_id(1)
    doc = tl.load(blkdoc_ptr + b)
    pad0 = tl.load(pad_ptr + doc)
    start = tl.load(cu_ptr + doc)
    end = tl.load(cu_ptr + doc + 1)
    tp = b * BT + tl.arange(0, BT)
    t = start + (tp - pad0)
    d = tl.arange(0, D)
    acc = tl.load(dqacc_ptr + (h * D + d)[:, None] * TPP + tp[None, :])                                  # [D, BT]
    val = tl.trans(acc).to(tl.float32).to(tl.bfloat16)                                                   # [BT, D]
    tl.store(dq_ptr + (t[:, None] * H + h) * D + d[None, :], val, mask=(t < end)[:, None])


# ----------------------------------------------------------------------------- host side
def make_plan(cu_seqlens_cpu, H, BM=128, BN=64, order="headmajor"):
    cu = [int(x) for x in cu_seqlens_cpu]
    items = []
    docs = []
    for d in range(len(cu) - 1):
        L = cu[d + 1] - cu[d]
        n_kv = (L + BM - 1) // BM
        n_q = (L + BN - 1) // BN
        docs.append((L, d, n_kv, n_q))
    if order == "cost":
        for (L, d, n_kv, n_q) in docs:
            for j in range(n_kv):
                cost = n_q - (j * BM) // BN
                for h in range(H):
                    items.append((cost, d, j, h))
        items.sort(key=lambda x: -x[0])
    else:
        # docs by size desc; within a doc head-major, kv-block ascending (cost descending)
        for (L, d, n_kv, n_q) in sorted(docs, key=lambda x: -x[0]):
            for h in range(H):
                for j in range(n_kv):
                    cost = n_q - (j * BM) // BN
                    items.append((cost, d, j, h))
    packed = [h | (j << 8) | (d << 16) for (_, d, j, h) in items]
    pad = [0]
    blkdoc = []
    for d in range(len(cu) - 1):
        n_q = (cu[d + 1] - cu[d] + BN - 1) // BN
        pad.append(pad[-1] + n_q * BN)
        blkdoc += [d] * n_q
    return dict(work=torch.tensor(packed, dtype=torch.int32), total_iters=sum(c for c, *_ in items),
                pad=torch.tensor(pad, dtype=torch.int32), blkdoc=torch.tensor(blkdoc, dtype=torch.int32), TPP=pad[-1])


def flash_bwd_varlen(q, k, v, o, do, lse, cu_seqlens, max_seqlen, plan=None, grid=None, NQS=3, EPI_REGS=64,
                     SMB_REGS=88, LOADER_REGS=24, MMA_REGS=48, exp_split=False, PDS_SW=64, SMEM_SCAN=None,
                     PRE_BT=64, PRE_WARPS=4, dqacc=None, out=None, abl=(), trace=None):
    global EXP2_ASM
    if EXP2_ASM is None:
        c = EXP2_COEFFS
        EXP2_ASM = gl.constexpr(_EXP2_MIX_ASM.value.replace("{C0}", _f32hex(c[0])).replace("{C1}", _f32hex(c[1]))
                                .replace("{C2}", _f32hex(c[2])).replace("{C3}", _f32hex(c[3])))
    T, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    BM, BN = 128, 64
    dev = q.device
    if plan is None:
        plan = make_plan(cu_seqlens.cpu().tolist(), H, BM, BN)
        plan = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in plan.items()}
    work, pad, blkdoc = plan["work"], plan["pad"], plan["blkdoc"]
    n_work = work.numel()
    scale = 1.0 / math.sqrt(D)
    TPP = plan["TPP"] + BN
    if dqacc is None:
        dqacc = torch.empty(H, D, TPP, device=dev, dtype=torch.float16)
    delta = torch.empty(H, T, device=dev, dtype=torch.float32)
    lse2 = torch.empty(H, T, device=dev, dtype=torch.float32)
    if out is None:
        dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    else:
        dq, dk, dv = out
    BT = 64
    _preprocess[(triton.cdiv(TPP, PRE_BT), H)](o, do, lse, delta, lse2, dqacc, T, TPP, H=H, D=D, BT=PRE_BT, LOG2E_=LOG2E,
                                               num_warps=PRE_WARPS)

    lay_q = gl.NVMMASharedLayout.get_default_for([BN, D], gl.bfloat16)
    lay_k = gl.NVMMASharedLayout.get_default_for([BM, D], gl.bfloat16)
    lay_p = gl.NVMMASharedLayout(swizzle_byte_width=PDS_SW, element_bitwidth=16, rank=2)
    q2, k2, v2, do2 = (x.view(T, H * D) for x in (q, k, v, do))
    q_desc = TensorDescriptor.from_tensor(q2, [BN, D], lay_q)
    do_desc = TensorDescriptor.from_tensor(do2, [BN, D], lay_q)
    k_desc = TensorDescriptor.from_tensor(k2, [BM, D], lay_k)
    v_desc = TensorDescriptor.from_tensor(v2, [BM, D], lay_k)
    if grid is None:
        grid = n_work
    tr = trace if trace is not None else torch.zeros(1, device=dev, dtype=torch.int32)
    if tr.numel() < 16384:
        tr = torch.zeros(16384, device=dev, dtype=torch.int32)
    global _SMEM_SCAN_CACHE
    if SMEM_SCAN is None:
        key = (NQS, EPI_REGS, SMB_REGS, LOADER_REGS, MMA_REGS, exp_split, PDS_SW, abl, trace is not None)
        if key not in _SMEM_SCAN_CACHE:
            # phase 1: compile with a tiny scan bound just to learn the real dynamic smem size
            kk = _bwd_kernel.warmup(q_desc, k_desc, v_desc, do_desc, lse2, delta, dqacc, pad, dk, dv, cu_seqlens, work, n_work,
                                    T, TPP, scale * LOG2E, scale, tr, H=H, pds_layout=lay_p, NQS=NQS, SMB_REGS=SMB_REGS,
                                    EPI_REGS=EPI_REGS, SMEM_SCAN=4096, LOADER_REGS=LOADER_REGS, MMA_REGS=MMA_REGS,
                                    num_warps=8, EXP_SPLIT=exp_split, EXP_ASM=EXP2_ASM,
                                    ABL_NOATOMIC=("noatomic" in abl), ABL_NOEPI=("noepi" in abl), ABL_NOLOAD=("noload" in abl),
                                    ABL_NOSTS=("nosts" in abl), ABL_NOSTTM=("nosttm" in abl), TRACE=(trace is not None), grid=(1,))
            _SMEM_SCAN_CACHE[key] = int(kk.metadata.shared) - 1024
        SMEM_SCAN = _SMEM_SCAN_CACHE[key]
    _bwd_kernel[(grid,)](q_desc, k_desc, v_desc, do_desc, lse2, delta, dqacc, pad, dk, dv, cu_seqlens, work, n_work,
                         T, TPP, scale * LOG2E, scale, tr, H=H, pds_layout=lay_p, NQS=NQS, SMB_REGS=SMB_REGS, EPI_REGS=EPI_REGS, SMEM_SCAN=SMEM_SCAN,
                         num_warps=8, EXP_SPLIT=exp_split, EXP_ASM=EXP2_ASM, LOADER_REGS=LOADER_REGS, MMA_REGS=MMA_REGS,
                         ABL_NOATOMIC=("noatomic" in abl), ABL_NOEPI=("noepi" in abl), ABL_NOLOAD=("noload" in abl),
                         ABL_NOSTS=("nosts" in abl), ABL_NOSTTM=("nosttm" in abl), TRACE=(trace is not None))
    _postprocess[(plan["TPP"] // BT, H)](dqacc, dq, blkdoc, pad, cu_seqlens, TPP, H=H, D=D, BT=BT, num_warps=4)
    return dq, dk, dv
