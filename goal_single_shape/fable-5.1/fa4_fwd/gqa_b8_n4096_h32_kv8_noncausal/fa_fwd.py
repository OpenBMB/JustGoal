# fa_fwd.py -- FlashAttention forward (non-causal, GQA, bf16, head_dim=128) for NVIDIA B200 (sm_100a), written from
# scratch in CuTe DSL (no reference kernel code was consulted).
#
# Design (per CTA; persistent, one CTA per SM, 148 CTAs):
#   * Work unit = one 128-row Q tile of one (batch, head); the CTA loops over KV tiles of 128 keys.
#   * Warp roles (12 warps): warp 0 = TMA producer for Q (double-buffered, prefetched one tile ahead) and K stages,
#     plus issuer of the TMA store of the previous tile's O; warp 2 = TMA producer for V stages; warp 1 = tcgen05.mma
#     issuer for S = Q K^T; warp 3 = tcgen05.mma issuer for O += P V; warps 4-11 = two softmax warpgroups.
#   * TMEM (512 columns): three 128-column S buffers (fp32), the bf16 P operand of the PV MMA aliases columns [64,128)
#     of its own S buffer (A operand read from TMEM), O accumulator in columns [384,512).
#   * The two softmax warpgroups process alternating KV tiles; each thread owns one full row (128 scores) of its tile:
#     exact tile max, exponentials with a *speculative* reference (last known running max, moved only when the tile
#     max exceeds it by 2^8 -> P stays bounded), packed bf16 P written to TMEM, running-max handoff to the next tile
#     through shared memory + mbarriers; in the rare case the reference changed, the stored P is rescaled in place and
#     the O accumulator is rescaled (after the previous PV completed).
#   * Epilogue of tile i (O drain, 1/l normalization, bf16 pack, swizzled smem staging, TMA store) is done by one
#     warpgroup while the other already starts tile i+1; partial row sums are exchanged through smem.
#   * Correctness: fp32 accumulation everywhere; validated against an fp32 naive reference (see final_eval.py).
# From-scratch FlashAttention forward (non-causal, GQA, bf16, hdim 128) for Blackwell SM100 in CuTe DSL.
# Architecture (per CTA = one 128-row Q tile of one (b, h)):
#   warp 0      : TMA producer (Q once, K/V tiles through 2 pipelines)
#   warp 1      : MMA issuer  (QK^T -> S in TMEM, P.V -> O in TMEM; order QK(j+2), PV(j))
#   warps 4..11 : two softmax warpgroups; WG w owns columns [64w, 64w+64) of S for all 128 rows
#   TMEM: S/P buffers x3 (cols 0..383, P bf16 aliases S), O (cols 384..511)
# From-scratch FlashAttention forward (non-causal, GQA, bf16, hdim 128) for Blackwell SM100 in CuTe DSL.
# Architecture (per CTA = one 128-row Q tile of one (b, h)):
#   warp 0      : TMA producer (Q once, K/V tiles through 2 pipelines)
#   warp 1      : MMA issuer  (QK^T -> S in TMEM, P.V -> O in TMEM; order QK(j+2), PV(j))
#   warps 4..11 : two softmax warpgroups; WG w owns columns [64w, 64w+64) of S for all 128 rows
#   TMEM: S/P buffers x3 (cols 0..383, P bf16 aliases S), O (cols 384..511)
# From-scratch FlashAttention forward (non-causal, GQA, bf16, hdim 128) for Blackwell SM100 in CuTe DSL.
# Architecture (per CTA = one 128-row Q tile of one (b, h)):
#   warp 0      : TMA producer (Q once, K/V tiles through 2 pipelines)
#   warp 1      : MMA issuer  (QK^T -> S in TMEM, P.V -> O in TMEM; order QK(j+2), PV(j))
#   warps 4..11 : two softmax warpgroups; WG w owns columns [64w, 64w+64) of S for all 128 rows
#   TMEM: S/P buffers x3 (cols 0..383, P bf16 aliases S), O (cols 384..511)
import math, torch
import cuda.bindings.driver as cuda
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir

LOG2E = 1.4426950408889634
import os as _os
EXP_NOSTORE = bool(int(_os.environ.get('FA_EXP_NOSTORE', '0')))
EXP_NOPACK = bool(int(_os.environ.get('FA_EXP_NOPACK', '0')))
EXP_NOMUFU = bool(int(_os.environ.get('FA_EXP_NOMUFU', '0')))
SINGLE_WG = bool(int(_os.environ.get('FA_SINGLE_WG', '0')))
PUBLISH_AFTER_Q = int(_os.environ.get('FA_PUBLISH_Q', '2'))
SCALAR_FFMA = bool(int(_os.environ.get('FA_SCALAR_FFMA', '0')))
EPI_WG1_ONLY = bool(int(_os.environ.get('FA_EPI_WG1', '0')))
TILE_GROUP = int(_os.environ.get('FA_TILE_GROUP', '1'))
SPECULATIVE = bool(int(_os.environ.get('FA_SPEC', '1')))   # consecutive KV tiles per warpgroup turn (1 = strict alternation)   # publish reference after this many exp quarters (0=before exp, 4=after all)

def pack_bf16x2(lo, hi):
    res = llvm.inline_asm(ir.IntegerType.get_signless(32), [cutlass.Float32(hi).ir_value(), cutlass.Float32(lo).ir_value()],
                          "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Uint32(res)

EXP2_ASM = """{
.reg .b64 X, T, R, P;
.reg .b32 t0, t1, i0, i1, p0, p1;
mov.b64 X, {$2, $3};
add.rn.f32x2 T, X, $4;
sub.rn.f32x2 R, T, $4;
sub.rn.f32x2 R, X, R;
fma.rn.f32x2 P, R, $5, $6;
fma.rn.f32x2 P, P, R, $7;
fma.rn.f32x2 P, P, R, $8;
mov.b64 {t0, t1}, T;
mov.b64 {p0, p1}, P;
shl.b32 i0, t0, 23;
shl.b32 i1, t1, 23;
add.s32 $0, p0, i0;
add.s32 $1, p1, i1;
}

v16p (final): identical to v16 plus register pinning after every tcgen05.wait::ld (pin_f32/pin_u32): to the
compiler the asynchronously loaded registers are plain SSA values, so uses could be scheduled above the wait;
an empty side-effecting asm tied to each register makes every use depend on something ordered after the wait.
"""

def _c64(a, b):
    import struct
    ai = struct.unpack("<I", struct.pack("<f", a))[0]; bi = struct.unpack("<I", struct.pack("<f", b))[0]
    return cutlass.Int64((bi << 32) | ai).ir_value()

def exp2_emul_pair(x0, x1):
    """2^x0, 2^x1 on the FMA pipe (cubic polynomial, round-to-nearest split); inputs must be >= -126 and <= 127."""
    f32 = ir.F32Type.get(); i32 = ir.IntegerType.get_signless(32)
    magic = _c64(12582912.0, 12582912.0)
    c3 = _c64(0.05550410866, 0.05550410866); c2 = _c64(0.2402265069, 0.2402265069)
    c1 = _c64(0.6931471806, 0.6931471806); one = _c64(1.0, 1.0)
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = llvm.inline_asm(st, [cutlass.Float32(x0).ir_value(), cutlass.Float32(x1).ir_value(), magic, c3, c2, c1, one],
                          EXP2_ASM, "=r,=r,f,f,l,l,l,l,l", has_side_effects=False, is_align_stack=False,
                          asm_dialect=llvm.AsmDialect.AD_ATT)
    r0 = llvm.extractvalue(i32, res, [0]); r1 = llvm.extractvalue(i32, res, [1])
    return cutlass.Float32(llvm.bitcast(f32, r0)), cutlass.Float32(llvm.bitcast(f32, r1))

def exp_half(x, emul_pairs, tPr):
    """x: TensorSSA of 64 fp32 (log2-domain, <= ~8). Computes p = 2^x: first `emul_pairs` pairs on the FMA pipe,
    the rest on MUFU; packs bf16 pairs into tPr (32 x u32) and returns the fp32 row-sum contribution."""
    acc = [cutlass.Float32(0.0) for _ in range(4)]
    for ii in range(32):
        if ii < emul_pairs:
            a = cute.arch.fmax(x[2 * ii], -126.0)
            bb = cute.arch.fmax(x[2 * ii + 1], -126.0)
            p_lo, p_hi = exp2_emul_pair(a, bb)
        else:
            p_lo = cute.arch.exp2(x[2 * ii])
            p_hi = cute.arch.exp2(x[2 * ii + 1])
        acc[ii % 4] = acc[ii % 4] + (p_lo + p_hi)
        tPr[ii] = pack_bf16x2(p_lo, p_hi)
    return (acc[0] + acc[1]) + (acc[2] + acc[3])


def fmax3(a, b, c):
    res = llvm.inline_asm(ir.F32Type.get(), [cutlass.Float32(a).ir_value(), cutlass.Float32(b).ir_value(), cutlass.Float32(c).ir_value()],
                          "max.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False, is_align_stack=False,
                          asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Float32(res)

def fma_f32(a, b, c):
    res = llvm.inline_asm(ir.F32Type.get(), [cutlass.Float32(a).ir_value(), cutlass.Float32(b).ir_value(), cutlass.Float32(c).ir_value()],
                          "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False, is_align_stack=False,
                          asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Float32(res)


def row_max(vals):
    """max over a python list of Float32 using a 3-input FMNMX tree."""
    cur = list(vals)
    while len(cur) > 1:
        nxt = []
        i = 0
        while i + 3 <= len(cur):
            nxt.append(fmax3(cur[i], cur[i + 1], cur[i + 2])); i += 3
        if len(cur) - i == 2:
            nxt.append(cute.arch.fmax(cur[i], cur[i + 1]))
        elif len(cur) - i == 1:
            nxt.append(cur[i])
        cur = nxt
    return cur[0]

def _bitcast_i32(x):
    return cutlass.Int32(llvm.bitcast(ir.IntegerType.get_signless(32), cutlass.Float32(x).ir_value()))

def _bitcast_f32(i):
    return cutlass.Float32(llvm.bitcast(ir.F32Type.get(), cutlass.Int32(i).ir_value()))

def exp2_emul_pair2(x0, x1):
    """2^x for a pair via packed FFMA/FADD ops + exponent injection (inputs clamped to >= -126 by caller)."""
    magic = cutlass.Float32(12582912.0)
    t0, t1 = cute.arch.add_packed_f32x2((x0, x1), (magic, magic))                       # round(x) in low mantissa bits
    r0, r1 = cute.arch.add_packed_f32x2((t0, t1), (-magic, -magic))                     # round(x) as float
    f0, f1 = cute.arch.fma_packed_f32x2((r0, r1), (cutlass.Float32(-1.0), cutlass.Float32(-1.0)), (x0, x1))   # f = x - round(x)
    c3 = cutlass.Float32(0.05550410866); c2 = cutlass.Float32(0.2402265069); c1 = cutlass.Float32(0.6931471806); one = cutlass.Float32(1.0)
    p0, p1 = cute.arch.fma_packed_f32x2((f0, f1), (c3, c3), (c2, c2))
    p0, p1 = cute.arch.fma_packed_f32x2((p0, p1), (f0, f1), (c1, c1))
    p0, p1 = cute.arch.fma_packed_f32x2((p0, p1), (f0, f1), (one, one))
    y0 = _bitcast_f32(_bitcast_i32(p0) + (_bitcast_i32(t0) << 23))
    y1 = _bitcast_f32(_bitcast_i32(p1) + (_bitcast_i32(t1) << 23))
    return y0, y1


def softmax_chunk(tS, base, c, neg_mc, emul_pairs, tPr, acc, npairs):
    """Elements [base, base+2*npairs) of rmem tensor tS: p = 2^(s*c + neg_mc); packs bf16 pairs into tPr[0:npairs]
    and accumulates row sums into acc (2 packed pairs). Pairs with index < emul_pairs use the FMA pipe."""
    for ii in range(npairs):
        e = base + 2 * ii
        if SCALAR_FFMA:
            x0 = fma_f32(tS[e], c, neg_mc)
            x1 = fma_f32(tS[e + 1], c, neg_mc)
        else:
            x0, x1 = cute.arch.fma_packed_f32x2((tS[e], tS[e + 1]), (c, c), (neg_mc, neg_mc))
        if ii < emul_pairs:
            p0, p1 = exp2_emul_pair2(cute.arch.fmax(x0, -126.0), cute.arch.fmax(x1, -126.0))
        elif EXP_NOMUFU:
            p0 = x0 * cutlass.Float32(1.0001)
            p1 = x1 * cutlass.Float32(1.0001)
        else:
            p0 = cute.arch.exp2(x0)
            p1 = cute.arch.exp2(x1)
        acc[ii % 2] = cute.arch.add_packed_f32x2(acc[ii % 2], (p0, p1))
        if EXP_NOPACK:
            tPr[ii] = _bitcast_i32(p0)
        else:
            tPr[ii] = pack_bf16x2(p0, p1)


def softmax_tile_chunks(tS0, tS1, c, neg_mc, emul_pairs, tPcs, acc, tP_all, row, st_atom_c, p_chunk_layout):
    """Full 128-column row in 8 chunks of 16: exp, pack, and store each chunk to TMEM immediately (low register pressure).
    tPcs: list of register tensors used round-robin so consecutive chunks do not serialize on a WAR dependency."""
    for ch in range(8):
        tSx = tS0 if ch < 4 else tS1
        tPc = tPcs[ch % len(tPcs)]
        softmax_chunk(tSx, (ch % 4) * 16, c, neg_mc, emul_pairs, tPc, acc, 8)
        tP_ch = cute.make_tensor(tP_all.iterator + ch * 8, p_chunk_layout)
        tiled_stc = tcgen05.make_tmem_copy(st_atom_c, tP_ch)
        cute.copy(tiled_stc, tPc, tiled_stc.get_slice(row).partition_D(tP_ch))


def load_quarter(tmem_base, qi, row, ld_atom_q, q_layout, dst):
    tSq = cute.make_tensor(tmem_base + qi * 32, q_layout)
    tiled = tcgen05.make_tmem_copy(ld_atom_q, tSq)
    cute.copy(tiled, tiled.get_slice(row).partition_S(tSq), dst)

def exp_quarter(tQ, c, neg_mc, emul_pairs, tPcs, acc, tP_all, qi, row, st_atom_c, p_chunk_layout):
    """32 scores in rmem tensor tQ -> exp, pack into 16 u32, one x16 TMEM store to P columns [16*qi, 16*qi+16)."""
    tPc = tPcs[qi % len(tPcs)]
    softmax_chunk(tQ, 0, c, neg_mc, emul_pairs, tPc, acc, 16)
    if not EXP_NOSTORE:
        tP_ch = cute.make_tensor(tP_all.iterator + qi * 16, p_chunk_layout)
        tiled_stc = tcgen05.make_tmem_copy(st_atom_c, tP_ch)
        cute.copy(tiled_stc, tPc, tiled_stc.get_slice(row).partition_D(tP_ch))


def exp_quarter_max(tQ, c, neg_mc, emul_pairs, tPcs, acc, tP_all, qi, row, st_atom_c, p_chunk_layout, m_run):
    """like exp_quarter, additionally returns max(m_run, max of the 32 scores)."""
    m_run = cute.arch.fmax(m_run, row_max([tQ[kk] for kk in range(32)]))
    exp_quarter(tQ, c, neg_mc, emul_pairs, tPcs, acc, tP_all, qi, row, st_atom_c, p_chunk_layout)
    return m_run

def exp_pass(s_base, row, ld_atom_q, q_layout, tQa, tQb, tQc, tQd, c, neg_mc, emul_pairs, tPcs, acc, tP_all, st_atom_c, p_chunk_layout, m_run):
    """Stream the 128 scores of this row through registers (two quarters in flight): exp/pack/store + running max."""
    load_quarter(s_base, 0, row, ld_atom_q, q_layout, tQa)
    load_quarter(s_base, 1, row, ld_atom_q, q_layout, tQb)
    cute.arch.fence_view_async_tmem_load()
    m_run = exp_quarter_max(tQa, c, neg_mc, emul_pairs, tPcs, acc, tP_all, 0, row, st_atom_c, p_chunk_layout, m_run)
    load_quarter(s_base, 2, row, ld_atom_q, q_layout, tQc)
    m_run = exp_quarter_max(tQb, c, neg_mc, emul_pairs, tPcs, acc, tP_all, 1, row, st_atom_c, p_chunk_layout, m_run)
    cute.arch.fence_view_async_tmem_load()
    load_quarter(s_base, 3, row, ld_atom_q, q_layout, tQd)
    m_run = exp_quarter_max(tQc, c, neg_mc, emul_pairs, tPcs, acc, tP_all, 2, row, st_atom_c, p_chunk_layout, m_run)
    cute.arch.fence_view_async_tmem_load()
    m_run = exp_quarter_max(tQd, c, neg_mc, emul_pairs, tPcs, acc, tP_all, 3, row, st_atom_c, p_chunk_layout, m_run)
    return m_run, acc


def rescale_packed_bf16(tPfix, f_p):
    """In-register rescale of 32 packed bf16x2 words by f_p (rare correction path)."""
    for kk in range(32):
        pw = tPfix[kk]
        lo = _bitcast_f32(cutlass.Int32(pw << 16)) * f_p
        hi = _bitcast_f32(cutlass.Int32(pw & cutlass.Uint32(0xFFFF0000))) * f_p
        tPfix[kk] = pack_bf16x2(lo, hi)


def epilogue_store_both(tmem_ptr, o_col, half_layout, ld_atom, row, tSr0, tSr1, inv_l, tOw4s, dst8s, o_free):
    """WG1 epilogue: both O halves loaded up front (one wait), O released immediately, then normalize (packed
    multiplies), pack to bf16 and write the swizzled smem staging tile. All loops static (plain python)."""
    for hh, dst in ((0, tSr0), (1, tSr1)):
        tO_half = cute.make_tensor(tmem_ptr + o_col + 64 * hh, half_layout)
        tiled_ld_o = tcgen05.make_tmem_copy(ld_atom, tO_half)
        cute.copy(tiled_ld_o, tiled_ld_o.get_slice(row).partition_S(tO_half), dst)
    cute.arch.fence_view_async_tmem_load()
    pin_f32(tSr0, 64); pin_f32(tSr1, 64)
    tcgen05_fence_before()
    cute.arch.mbarrier_arrive(o_free)
    for hh, tS in ((0, tSr0), (1, tSr1)):
        for cc in range(8):
            for e in range(4):
                a, b = cute.arch.mul_packed_f32x2((tS[8 * cc + 2 * e], tS[8 * cc + 2 * e + 1]), (inv_l, inv_l))
                tOw4s[cc][(0, e)] = pack_bf16x2(a, b)
            cute.autovec_copy(cute.recast_tensor(tOw4s[cc], cutlass.BFloat16), dst8s[hh][cc])


def tcgen05_fence_before():
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

def tcgen05_fence_after():
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


DBG_ITERS = 200000
DEBUG_WAITS = False   # bounded waits + timeout markers (debugging deadlocks only)
SLEEP_WAIT = bool(int(_os.environ.get('FA_SLEEP_WAIT', '0')))

def mbar_wait_sleep(ptr, parity):
    """mbarrier parity wait with a suspend hint and nanosleep backoff (avoids hammering the MIO pipe)."""
    llvm.inline_asm(None, [ptr.llvm_ptr, cutlass.Int32(parity).ir_value()],
                    "{\n.reg .pred p;\nWAIT_LOOP_${:uid}:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 20000;\n@!p nanosleep.u32 40;\n@!p bra WAIT_LOOP_${:uid};\n}",
                    "l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def mbar_wait_dbg(ptr: cute.Pointer, parity: cutlass.Int32, dbg: cute.Tensor, bidx: cutlass.Int32, code: cutlass.Constexpr):
    if cutlass.const_expr(DEBUG_WAITS):
        ok = cutlass.Boolean(False)
        for it in cutlass.range(DBG_ITERS, unroll=1):
            if not ok:
                ok = cute.arch.mbarrier_try_wait(ptr, parity)
        if not ok:
            dbg[(bidx, code)] = cutlass.Int32(1)
    elif cutlass.const_expr(SLEEP_WAIT):
        mbar_wait_sleep(ptr, parity)
    else:
        cute.arch.mbarrier_wait(ptr, parity)

def _opaque_f32(x):
    res = llvm.inline_asm(ir.F32Type.get(), [cutlass.Float32(x).ir_value()], "", "=f,0", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Float32(res)

def _opaque_u32(x):
    res = llvm.inline_asm(ir.IntegerType.get_signless(32), [cutlass.Uint32(x).ir_value()], "", "=r,0", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Uint32(res)

def pin_f32(t, n):
    """After tcgen05.wait::ld: tie every asynchronously loaded register to an empty side-effecting asm so that no use
    of it can be scheduled above the wait (to the compiler the loaded values are otherwise plain SSA values)."""
    for kk in range(n):
        t[kk] = _opaque_f32(t[kk])

def pin_u32(t, n):
    for kk in range(n):
        t[kk] = _opaque_u32(t[kk])

def clock32():
    res = llvm.inline_asm(ir.IntegerType.get_signless(32), [], "mov.u32 $0, %clock;", "=r", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Int32(res)

PROFILE_STAMPS = bool(int(_os.environ.get('FA_STAMPS', '0')))

def globaltimer_lo():
    res = llvm.inline_asm(ir.IntegerType.get_signless(32), [], "mov.u32 $0, %globaltimer_lo;", "=r", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Int32(res)

def smid():
    res = llvm.inline_asm(ir.IntegerType.get_signless(32), [], "mov.u32 $0, %smid;", "=r", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return cutlass.Int32(res)

@cute.jit
def stamp(dbg: cute.Tensor, bidx: cutlass.Int32, idx: cutlass.Int32, on: cutlass.Boolean):
    if on:
        dbg[(bidx, idx)] = clock32()

def mark(dbg, bidx, code):
    dbg[(bidx, code)] = cutlass.Int32(1)


class FlashFwdSm100:
    def __init__(self, kv_stages=2, emul_pairs=0):
        self.emul_pairs = emul_pairs
        self.emul_pairs_chunk = 2 * emul_pairs  # emulated pairs per 16-pair quarter (EMUL is per 8 pairs)
        self.M = 128; self.N = 128; self.D = 128
        self.kv_stages = kv_stages
        self.threads = 384
        self.num_s_bufs = 3
        self.tmem_cols = 512
        self.p_off = 64      # P (bf16 A operand, 64 cols) aliases columns [64,128) of its S buffer
        self.o_col = 384
        self.num_sms = 148
        self.use_setmaxnreg = True
        self.debug = True
        self.same_kv = False   # experiment: all CTAs read the K/V of (b=0, hkv=0)
        import os
        self.skip_o_store = bool(int(os.environ.get('FA_SKIP_STORE', '0')))
        self.skip_tmem_free = bool(int(os.environ.get('FA_SKIP_FREE', '0')))

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor,
                 scale_log2: cutlass.Float32, dbg: cute.Tensor, stream: cuda.CUstream):
        mma_qk = sm100_utils.make_trivial_tiled_mma(cutlass.BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                    cutlass.Float32, tcgen05.CtaGroup.ONE, (self.M, self.N))
        mma_qk_acc = sm100_utils.make_trivial_tiled_mma(cutlass.BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                        cutlass.Float32, tcgen05.CtaGroup.ONE, (self.M, self.N))
        mma_pv = sm100_utils.make_trivial_tiled_mma(cutlass.BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                    cutlass.Float32, tcgen05.CtaGroup.ONE, (self.M, self.D), tcgen05.OperandSource.TMEM)
        mma_pv_first = sm100_utils.make_trivial_tiled_mma(cutlass.BFloat16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                          cutlass.Float32, tcgen05.CtaGroup.ONE, (self.M, self.D), tcgen05.OperandSource.TMEM)
        tiler_qk = (self.M, self.N, self.D)   # (q rows, keys, dims)
        tiler_pv = (self.M, self.D, self.N)   # (q rows, dims, keys)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(mma_qk, tiler_qk, cutlass.BFloat16, 2)
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(mma_qk, tiler_qk, cutlass.BFloat16, self.kv_stages)
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(mma_pv, tiler_pv, cutlass.BFloat16, self.kv_stages)
        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, None, 0))
        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, None, 0))
        v_smem_layout = cute.slice_(v_smem_layout_staged, (None, None, None, 0))
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_qk.thr_id.shape,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_q, tma_q = cute.nvgpu.make_tiled_tma_atom_A(op, q, q_smem_layout, tiler_qk, mma_qk, cluster_layout_vmnk.shape)
        tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(op, k, k_smem_layout, tiler_qk, mma_qk, cluster_layout_vmnk.shape)
        tma_atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(op, v, v_smem_layout, tiler_pv, mma_pv, cluster_layout_vmnk.shape)
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(cutlass.BFloat16, utils.LayoutEnum.ROW_MAJOR, (self.M, self.D), 1)
        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0))
        tma_atom_o, tma_o = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), o, o_smem_layout, (self.M, self.D))
        q_bytes = cute.size_in_bytes(cutlass.BFloat16, q_smem_layout)
        k_bytes = cute.size_in_bytes(cutlass.BFloat16, k_smem_layout)
        v_bytes = cute.size_in_bytes(cutlass.BFloat16, v_smem_layout)
        n_q_tiles = cute.size(q, mode=[0]) // self.M
        H = cute.size(q, mode=[2]); B = cute.size(q, mode=[3])
        grid = [cutlass.min(n_q_tiles * H * B, self.num_sms), 1, 1]
        self.kernel(mma_qk, mma_qk_acc, mma_pv, mma_pv_first, tma_atom_q, tma_q, tma_atom_k, tma_k, tma_atom_v, tma_v, tma_atom_o, tma_o,
                    q_smem_layout_staged, k_smem_layout_staged, v_smem_layout_staged, o_smem_layout_staged, cluster_layout_vmnk,
                    q_bytes, k_bytes, v_bytes, scale_log2, dbg).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mma_qk: cute.TiledMma, mma_qk_acc: cute.TiledMma, mma_pv: cute.TiledMma, mma_pv_first: cute.TiledMma,
               tma_atom_q: cute.CopyAtom, mQ: cute.Tensor, tma_atom_k: cute.CopyAtom, mK: cute.Tensor,
               tma_atom_v: cute.CopyAtom, mV: cute.Tensor, tma_atom_o: cute.CopyAtom, mO: cute.Tensor,
               q_smem_layout_staged: cute.ComposedLayout, k_smem_layout_staged: cute.ComposedLayout,
               v_smem_layout_staged: cute.ComposedLayout, o_smem_layout_staged: cute.ComposedLayout, cluster_layout_vmnk: cute.Layout,
               q_bytes: cutlass.Constexpr, k_bytes: cutlass.Constexpr, v_bytes: cutlass.Constexpr,
               scale_log2: cutlass.Float32, dbg: cute.Tensor):
        M, N, D = self.M, self.N, self.D
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        n_q_tiles = cute.size(mQ, mode=[0]) // M
        H = cute.size(mQ, mode=[2])
        HKV = cute.size(mK, mode=[2])
        B = cute.size(mQ, mode=[3])
        n_kv_tiles = cute.size(mK, mode=[0]) // N
        total_tiles = n_q_tiles * H * B
        heads_per_kv = H // HKV
        # number of work tiles for this CTA
        n_my_tiles = (total_tiles - bidx + gdim - 1) // gdim

        if cutlass.const_expr(PROFILE_STAMPS):
            if tidx == 0:
                dbg[(bidx, 0)] = globaltimer_lo()
                dbg[(bidx, 2)] = smid()
                dbg[(bidx, 7)] = clock32()
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stages]
            v_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stages]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            q_empty: cute.struct.MemRange[cutlass.Int64, 2]
            s_full: cute.struct.MemRange[cutlass.Int64, 3]
            p_ready: cute.struct.MemRange[cutlass.Int64, 3]
            pv_done: cute.struct.MemRange[cutlass.Int64, 3]
            mref_ready: cute.struct.MemRange[cutlass.Int64, 3]
            o_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            mref: cute.struct.MemRange[cutlass.Float32, 3 * 128]
            lx: cute.struct.MemRange[cutlass.Float32, 2 * 128]
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = smem.allocate_tensor(cutlass.BFloat16, q_smem_layout_staged.outer, 128, q_smem_layout_staged.inner)
        sK = smem.allocate_tensor(cutlass.BFloat16, k_smem_layout_staged.outer, 128, k_smem_layout_staged.inner)
        sV = smem.allocate_tensor(cutlass.BFloat16, v_smem_layout_staged.outer, 128, v_smem_layout_staged.inner)
        sO = smem.allocate_tensor(cutlass.BFloat16, o_smem_layout_staged.outer, 128, o_smem_layout_staged.inner)

        q_full = storage.q_full.data_ptr()
        q_empty = storage.q_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_ready = storage.p_ready.data_ptr()
        pv_done = storage.pv_done.data_ptr()
        mref_ready = storage.mref_ready.data_ptr()
        o_free = storage.o_free
        mref = cute.make_tensor(storage.mref.data_ptr(), cute.make_layout(3 * 128))
        lx = cute.make_tensor(storage.lx.data_ptr(), cute.make_layout(2 * 128))

        prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        k_pipe = pipeline.PipelineTmaUmma.create(barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.kv_stages,
                                                 producer_group=prod_grp, consumer_group=cons_grp, tx_count=k_bytes,
                                                 cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        v_pipe = pipeline.PipelineTmaUmma.create(barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.kv_stages,
                                                 producer_group=prod_grp, consumer_group=cons_grp, tx_count=v_bytes,
                                                 cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(o_free, 128)
                for i in range(2):
                    cute.arch.mbarrier_init(q_full + i, 1)
                    cute.arch.mbarrier_init(q_empty + i, 1)
                for i in range(3):
                    cute.arch.mbarrier_init(pv_done + i, 1)
                    cute.arch.mbarrier_init(s_full + i, 1)
                    cute.arch.mbarrier_init(p_ready + i, 128)
                    cute.arch.mbarrier_init(mref_ready + i, 128)
            cute.arch.mbarrier_init_fence()
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier)
        cute.arch.sync_threads()
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()

        gQ = cute.local_tile(mQ, cute.slice_((M, N, D), (None, 0, None)), (None, None, None, None))
        gK = cute.local_tile(mK, cute.slice_((M, N, D), (0, None, None)), (None, None, None, None))
        gV = cute.local_tile(mV, cute.slice_((M, D, N), (0, None, None)), (None, None, None, None))
        thr_qk = mma_qk.get_slice(0)
        thr_pv = mma_pv.get_slice(0)
        tCgQ = thr_qk.partition_A(gQ)
        tCgK = thr_qk.partition_B(gK)
        tCgV = thr_pv.partition_B(gV)
        tQsQ, tQgQ = cpasync.tma_partition(tma_atom_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tCgQ, 0, 3))
        tKsK, tKgK = cpasync.tma_partition(tma_atom_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tCgK, 0, 3))
        tVsV, tVgV = cpasync.tma_partition(tma_atom_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tCgV, 0, 3))
        gO_t = cute.local_tile(mO, (M, D), (None, None, None, None))
        sO_2d = sO[(None, None, 0)]
        tOsO, tOgO = cpasync.tma_partition(tma_atom_o, 0, cute.make_layout(1), cute.group_modes(sO_2d, 0, 2), cute.group_modes(gO_t, 0, 2))
        tQ_frag = mma_qk.make_fragment_A(sQ)
        tK_frag = mma_qk.make_fragment_B(sK)
        tV_frag = mma_pv.make_fragment_B(sV)

        acc_layout = cute.make_layout(((M, N), 1, 1), stride=((65536, 1), 0, 0))
        p_a_layout = cute.make_layout(((M, 16), 1, 8), stride=((131072, 1), 0, 16))
        tO = cute.make_tensor(tmem_ptr + self.o_col, acc_layout)

        # ------------------------------------------------------------------ producer warps (0: Q+K, 2: V)
        if warp_idx == 0:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(40)
            k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
            # Q(0)
            w0 = bidx
            qt0 = w0 % n_q_tiles
            h0 = (w0 // n_q_tiles) % H
            b0 = w0 // (n_q_tiles * H)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
            cute.copy(tma_atom_q, tQgQ[(None, qt0, 0, h0, b0)], tQsQ[(None, 0)], tma_bar_ptr=q_full)
            for i in cutlass.range(n_my_tiles):
                w = bidx + i * gdim
                q_tile = w % n_q_tiles
                h = (w // n_q_tiles) % H
                b = w // (n_q_tiles * H)
                hkv = h // heads_per_kv
                b_kv = b
                if cutlass.const_expr(self.same_kv):
                    hkv = cutlass.Int32(0)
                    b_kv = cutlass.Int32(0)
                for j in cutlass.range(n_kv_tiles):
                    mbar_wait_dbg(k_pipe.sync_object_empty.get_barrier(k_prod.index), k_prod.phase, dbg, bidx, 1)
                    k_pipe.producer_acquire(k_prod, cutlass.Boolean(True))
                    cute.copy(tma_atom_k, tKgK[(None, j, 0, hkv, b_kv)], tKsK[(None, k_prod.index)], tma_bar_ptr=k_pipe.producer_get_barrier(k_prod))
                    k_prod.advance()
                    # prefetch next tile's Q early into the other buffer
                    if j == 2 and i + 1 < n_my_tiles:
                        wn = w + gdim
                        qtn = wn % n_q_tiles
                        hn = (wn // n_q_tiles) % H
                        bn = wn // (n_q_tiles * H)
                        qb = (i + 1) % 2
                        mbar_wait_dbg(q_empty + qb, ((i + 1) // 2 - 1) & 1, dbg, bidx, 26)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(q_full + qb, q_bytes)
                        cute.copy(tma_atom_q, tQgQ[(None, qtn, 0, hn, bn)], tQsQ[(None, qb)], tma_bar_ptr=q_full + qb)
            mark(dbg, bidx, 10)
        elif warp_idx == 2:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(40)
            v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
            for i in cutlass.range(n_my_tiles):
                w = bidx + i * gdim
                h = (w // n_q_tiles) % H
                b = w // (n_q_tiles * H)
                hkv = h // heads_per_kv
                b_kv = b
                if cutlass.const_expr(self.same_kv):
                    hkv = cutlass.Int32(0)
                    b_kv = cutlass.Int32(0)
                for j in cutlass.range(n_kv_tiles):
                    mbar_wait_dbg(v_pipe.sync_object_empty.get_barrier(v_prod.index), v_prod.phase, dbg, bidx, 2)
                    v_pipe.producer_acquire(v_prod, cutlass.Boolean(True))
                    cute.copy(tma_atom_v, tVgV[(None, 0, j, hkv, b_kv)], tVsV[(None, v_prod.index)], tma_bar_ptr=v_pipe.producer_get_barrier(v_prod))
                    v_prod.advance()
        # ------------------------------------------------------------------ MMA warps: 1 = QK stream, 3 = PV stream
        elif warp_idx == 1:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(40)
            mma_qk.set(tcgen05.Field.ACCUMULATE, False)
            mma_qk_acc.set(tcgen05.Field.ACCUMULATE, True)
            k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
            gq = cutlass.Int32(0)   # global QK counter
            for i in cutlass.range(n_my_tiles):
                qb = i % 2
                mbar_wait_dbg(q_full + qb, (i // 2) & 1, dbg, bidx, 3)
                for jj in cutlass.range(n_kv_tiles):
                    mm_on = cutlass.const_expr(PROFILE_STAMPS) and (lane == 0) and (jj < 34) and (i == 0)
                    stamp(dbg, bidx, 640 + jj * 8 + 0, mm_on)
                    stamp(dbg, bidx, 1000 + i, cutlass.const_expr(PROFILE_STAMPS) and (lane == 0) and (jj == 0) and (i < 8))
                    sb = gq % 3
                    mbar_wait_dbg(k_pipe.sync_object_full.get_barrier(k_cons.index), k_cons.phase, dbg, bidx, 4)
                    stamp(dbg, bidx, 640 + jj * 8 + 1, mm_on)
                    if gq >= 3:
                        mbar_wait_dbg(pv_done + sb, ((gq // 3) - 1) & 1, dbg, bidx, 5)   # PV(gq-3) has consumed P in this buffer
                    tcgen05_fence_after()
                    tS = cute.make_tensor(tmem_ptr + sb * N, acc_layout)
                    cute.gemm(mma_qk, tS, tQ_frag[(None, None, 0, qb)], tK_frag[(None, None, 0, k_cons.index)], tS)
                    for kb in range(1, 8):
                        cute.gemm(mma_qk_acc, tS, tQ_frag[(None, None, kb, qb)], tK_frag[(None, None, kb, k_cons.index)], tS)
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + sb)
                    if jj == n_kv_tiles - 1:
                        with cute.arch.elect_one():
                            tcgen05.commit(q_empty + qb)
                    k_pipe.consumer_release(k_cons)
                    k_cons.advance()
                    gq = gq + 1
                    stamp(dbg, bidx, 640 + jj * 8 + 2, mm_on)
            mark(dbg, bidx, 11)
        elif warp_idx == 3:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(40)
            mma_pv.set(tcgen05.Field.ACCUMULATE, True)
            mma_pv_first.set(tcgen05.Field.ACCUMULATE, False)
            v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
            gp = cutlass.Int32(0)   # global PV counter
            for i in cutlass.range(n_my_tiles):
                for j in cutlass.range(n_kv_tiles):
                    mm_on = cutlass.const_expr(PROFILE_STAMPS) and (lane == 0) and (j < 32) and (i == 0)
                    pv1_on = cutlass.const_expr(PROFILE_STAMPS) and (lane == 0) and (j < 4) and (i == 1)
                    pb = gp % 3
                    stamp(dbg, bidx, 920 + (j + 2) * 8 + 6, pv1_on)
                    mbar_wait_dbg(p_ready + pb, (gp // 3) & 1, dbg, bidx, 6)
                    stamp(dbg, bidx, 640 + (j + 2) * 8 + 3, mm_on)
                    stamp(dbg, bidx, 920 + (j + 2) * 8 + 3, pv1_on)
                    if j == 0 and i > 0:
                        mbar_wait_dbg(o_free, (i - 1) & 1, dbg, bidx, 27)
                        stamp(dbg, bidx, 1010 + i, cutlass.const_expr(PROFILE_STAMPS) and (lane == 0) and (i < 8))
                    tcgen05_fence_after()
                    mbar_wait_dbg(v_pipe.sync_object_full.get_barrier(v_cons.index), v_cons.phase, dbg, bidx, 7)
                    stamp(dbg, bidx, 640 + (j + 2) * 8 + 4, mm_on)
                    stamp(dbg, bidx, 920 + (j + 2) * 8 + 4, pv1_on)
                    tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + pb * N + self.p_off, dtype=cutlass.BFloat16), p_a_layout)
                    if j == 0:
                        cute.gemm(mma_pv_first, tO, tP[(None, None, 0)], tV_frag[(None, None, 0, v_cons.index)], tO)
                    else:
                        cute.gemm(mma_pv, tO, tP[(None, None, 0)], tV_frag[(None, None, 0, v_cons.index)], tO)
                    for kb in range(1, 8):
                        cute.gemm(mma_pv, tO, tP[(None, None, kb)], tV_frag[(None, None, kb, v_cons.index)], tO)
                    with cute.arch.elect_one():
                        tcgen05.commit(pv_done + pb)
                    v_pipe.consumer_release(v_cons)
                    v_cons.advance()
                    gp = gp + 1
                    stamp(dbg, bidx, 640 + (j + 2) * 8 + 5, mm_on)
                    stamp(dbg, bidx, 920 + (j + 2) * 8 + 5, pv1_on)
            if cutlass.const_expr(PROFILE_STAMPS):
                if lane == 0:
                    dbg[(bidx, 3)] = globaltimer_lo()
        elif warp_idx < 4:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_dealloc(40)
        # ------------------------------------------------------------------ softmax warpgroups
        else:
            if cutlass.const_expr(self.use_setmaxnreg):
                cute.arch.warpgroup_reg_alloc(232)
            wg = (warp_idx - 4) // 4
            row = (warp_idx % 4) * 32 + lane
            ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
            st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
            st_atom_u32 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Uint32)
            half_layout = cute.make_layout((M, 64), stride=(65536, 1))
            p_half_layout = cute.make_layout((M, 32), stride=(65536, 1))
            cS = cute.make_identity_tensor((M, 64))
            cP = cute.make_identity_tensor((M, 32))
            proto_ld = tcgen05.make_tmem_copy(ld_atom, cute.make_tensor(tmem_ptr, half_layout))
            proto_st = tcgen05.make_tmem_copy(st_atom_u32, cute.make_tensor(cute.recast_ptr(tmem_ptr, dtype=cutlass.Uint32), p_half_layout))
            tSr0 = cute.make_rmem_tensor(proto_ld.get_slice(row).partition_D(cS).shape, cutlass.Float32)
            ld_atom_q = ld_atom   # x32 = one 32-column quarter
            q_layout = cute.make_layout((M, 32), stride=(65536, 1))
            cQ = cute.make_identity_tensor((M, 32))
            proto_ldq = tcgen05.make_tmem_copy(ld_atom_q, cute.make_tensor(tmem_ptr, q_layout))
            tQa = cute.make_rmem_tensor(proto_ldq.get_slice(row).partition_D(cQ).shape, cutlass.Float32)
            tQb = cute.make_rmem_tensor(proto_ldq.get_slice(row).partition_D(cQ).shape, cutlass.Float32)
            tQc = cute.make_rmem_tensor(proto_ldq.get_slice(row).partition_D(cQ).shape, cutlass.Float32)
            tQd = cute.make_rmem_tensor(proto_ldq.get_slice(row).partition_D(cQ).shape, cutlass.Float32)
            st_atom_c = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Uint32)
            p_chunk_layout = cute.make_layout((M, 16), stride=(65536, 1))
            p_all_layout = cute.make_layout((M, 64), stride=(65536, 1))
            cPc = cute.make_identity_tensor((M, 16))
            proto_stc = tcgen05.make_tmem_copy(st_atom_c, cute.make_tensor(cute.recast_ptr(tmem_ptr, dtype=cutlass.Uint32), p_chunk_layout))
            tPcs = [cute.make_rmem_tensor(proto_stc.get_slice(row).partition_S(cPc).shape, cutlass.Uint32) for _ in range(2)]
            ld_atom_u32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Uint32)
            tPfix = cute.make_rmem_tensor(proto_st.get_slice(row).partition_S(cP).shape, cutlass.Uint32)
            neg_inf = cutlass.Float32(float("-inf"))
            rescale_thresh = cutlass.Float32(8.0)
            tSr1 = cute.make_rmem_tensor(proto_ld.get_slice(row).partition_D(cS).shape, cutlass.Float32)
            tOw4s = [cute.make_rmem_tensor((1, 4), cutlass.Uint32) for _ in range(8)]
            dst8s = [[cute.local_tile(sO_2d, (1, 8), (row, 8 * hh + cc)) for cc in range(8)] for hh in range(2)]
            l_p = cutlass.Float32(0.0)
            m_known_p = cutlass.Float32(float("-inf"))
            g_base_p = cutlass.Int32(0)
            q_tile_p = cutlass.Int32(0)
            h_p = cutlass.Int32(0)
            b_p = cutlass.Int32(0)
            i_p = cutlass.Int32(0)
            # tiles handled by this WG: t with (t // TILE_GROUP) % 2 == wg
            n_groups = (n_kv_tiles + TILE_GROUP - 1) // TILE_GROUP
            n_my_groups = (n_groups - wg + 1) // 2
            n_my = n_my_groups * TILE_GROUP
            if cutlass.const_expr(SINGLE_WG):
                n_my = n_kv_tiles * (1 - wg)
            for i in cutlass.range(n_my_tiles):
                w = bidx + i * gdim
                q_tile = w % n_q_tiles
                h = (w // n_q_tiles) % H
                b = w // (n_q_tiles * H)
                g_base = i * n_kv_tiles
                m_known = cutlass.Float32(float("-inf"))
                l = cutlass.Float32(0.0)
                for it in cutlass.range(n_my):
                    t = 2 * it + wg
                    g = g_base + t
                    buf = g % 3
                    par = (g // 3) & 1
                    st_on = cutlass.const_expr(PROFILE_STAMPS) and (row == 0) and (t < 32) and (i == 0)
                    s1_on = cutlass.const_expr(PROFILE_STAMPS) and (row == 0) and (t < 4) and (i == 1)
                    stamp(dbg, bidx, 64 + (t * 8 + 0) * 2 + wg, st_on)
                    stamp(dbg, bidx, 1020 + t * 4 + 0, s1_on)
                    mbar_wait_dbg(s_full + buf, par, dbg, bidx, 8 + wg)
                    stamp(dbg, bidx, 64 + (t * 8 + 1) * 2 + wg, st_on)
                    stamp(dbg, bidx, 1020 + t * 4 + 1, s1_on)
                    tcgen05_fence_after()
                    s_base = tmem_ptr + buf * N
                    tP_all = cute.make_tensor(cute.recast_ptr(s_base + self.p_off, dtype=cutlass.Uint32), p_all_layout)
                    # ---- pass 1: load the whole row (4 quarters in flight), exact tile max
                    load_quarter(s_base, 0, row, ld_atom_q, q_layout, tQa)
                    load_quarter(s_base, 1, row, ld_atom_q, q_layout, tQb)
                    load_quarter(s_base, 2, row, ld_atom_q, q_layout, tQc)
                    load_quarter(s_base, 3, row, ld_atom_q, q_layout, tQd)
                    cute.arch.fence_view_async_tmem_load()
                    pin_f32(tQa, 32); pin_f32(tQb, 32); pin_f32(tQc, 32); pin_f32(tQd, 32)
                    m_tile = row_max([tQa[kk] for kk in range(32)] + [tQb[kk] for kk in range(32)] + [tQc[kk] for kk in range(32)] + [tQd[kk] for kk in range(32)])
                    stamp(dbg, bidx, 64 + (t * 8 + 2) * 2 + wg, st_on)
                    # ---- speculative reference: last known, unless this tile clearly moves it (then P <= 1 anyway)
                    m_spec = m_known
                    if (m_tile - m_known) * scale_log2 > rescale_thresh:
                        m_spec = m_tile
                    if t < 2:
                        # first tiles of a row: the previous tile's reference is needed before choosing (eager)
                        if t > 0:
                            mbar_wait_dbg(mref_ready + ((g - 1) % 3), ((g - 1) // 3) & 1, dbg, bidx, 22 + wg)
                            mp0 = mref[((g - 1) % 3) * 128 + row]
                            m_spec = mp0
                            if (m_tile - mp0) * scale_log2 > rescale_thresh:
                                m_spec = m_tile
                        else:
                            m_spec = m_tile
                    neg_mc = -(m_spec * scale_log2)
                    acc = [(cutlass.Float32(0.0), cutlass.Float32(0.0)), (cutlass.Float32(0.0), cutlass.Float32(0.0))]
                    # ---- exp pass from registers; P chunks land in columns [64,128) => process quarters 2,3 first
                    exp_quarter(tQc, scale_log2, neg_mc, self.emul_pairs_chunk, tPcs, acc, tP_all, 2, row, st_atom_c, p_chunk_layout)
                    exp_quarter(tQd, scale_log2, neg_mc, self.emul_pairs_chunk, tPcs, acc, tP_all, 3, row, st_atom_c, p_chunk_layout)
                    exp_quarter(tQa, scale_log2, neg_mc, self.emul_pairs_chunk, tPcs, acc, tP_all, 0, row, st_atom_c, p_chunk_layout)
                    exp_quarter(tQb, scale_log2, neg_mc, self.emul_pairs_chunk, tPcs, acc, tP_all, 1, row, st_atom_c, p_chunk_layout)
                    a00 = acc[0][0]; a01 = acc[0][1]; a10 = acc[1][0]; a11 = acc[1][1]
                    stamp(dbg, bidx, 64 + (t * 8 + 3) * 2 + wg, st_on)
                    # ---- now reconcile with the previous tile's reference
                    m_prev = neg_inf
                    if t > 0:
                        mbar_wait_dbg(mref_ready + ((g - 1) % 3), ((g - 1) // 3) & 1, dbg, bidx, 22 + wg)
                        m_prev = mref[((g - 1) % 3) * 128 + row]
                    if m_prev > m_known:
                        l = l * cute.arch.exp2((m_known - m_prev) * scale_log2)
                    need = (m_tile - m_prev) * scale_log2 > rescale_thresh
                    m_new = m_prev
                    if need:
                        m_new = m_tile
                    mref[(g % 3) * 128 + row] = m_new
                    cute.arch.mbarrier_arrive(mref_ready + (g % 3))
                    any_need = cute.arch.vote_any_sync(need)
                    alpha = cutlass.Float32(1.0)
                    if need:
                        alpha = cute.arch.exp2((m_prev - m_new) * scale_log2)
                    l = l * alpha
                    m_known = m_new
                    # P was computed with m_spec; if the final reference differs, rescale the stored P (rare)
                    fix = m_spec != m_new
                    any_fix = cute.arch.vote_any_sync(fix)
                    if any_fix:
                        cute.arch.fence_view_async_tmem_store()
                        f_p = cutlass.Float32(1.0)
                        if fix:
                            f_p = cute.arch.exp2((m_spec - m_new) * scale_log2)
                        a00 = a00 * f_p; a01 = a01 * f_p; a10 = a10 * f_p; a11 = a11 * f_p
                        for hh in range(2):
                            tPh = cute.make_tensor(cute.recast_ptr(s_base + self.p_off + 32 * hh, dtype=cutlass.Uint32), p_half_layout)
                            tiled_ldp = tcgen05.make_tmem_copy(ld_atom_u32, tPh)
                            cute.copy(tiled_ldp, tiled_ldp.get_slice(row).partition_S(tPh), tPfix)
                            cute.arch.fence_view_async_tmem_load()
                            pin_u32(tPfix, 32)
                            rescale_packed_bf16(tPfix, f_p)
                            tiled_stp = tcgen05.make_tmem_copy(st_atom_u32, tPh)
                            cute.copy(tiled_stp, tPfix, tiled_stp.get_slice(row).partition_D(tPh))
                    l = l + ((a00 + a01) + (a10 + a11))
                    stamp(dbg, bidx, 64 + (t * 8 + 4) * 2 + wg, st_on)
                    # ---- O rescale (warp-uniform; only when the reference moved), after PV(t-1) completed
                    if any_need:
                        if t > 0:
                            mbar_wait_dbg(pv_done + ((g - 1) % 3), ((g - 1) // 3) & 1, dbg, bidx, 16 + wg)
                            tcgen05_fence_after()
                            for hh in range(2):
                                tO_h = cute.make_tensor(tmem_ptr + self.o_col + 64 * hh, half_layout)
                                tiled_ld_o = tcgen05.make_tmem_copy(ld_atom, tO_h)
                                cute.copy(tiled_ld_o, tiled_ld_o.get_slice(row).partition_S(tO_h), tSr0)
                                cute.arch.fence_view_async_tmem_load()
                                pin_f32(tSr0, 64)
                                tSr0.store(tSr0.load() * alpha)
                                tiled_st_o = tcgen05.make_tmem_copy(st_atom, tO_h)
                                cute.copy(tiled_st_o, tSr0, tiled_st_o.get_slice(row).partition_D(tO_h))
                            cute.arch.fence_view_async_tmem_store()
                    stamp(dbg, bidx, 64 + (t * 8 + 5) * 2 + wg, st_on)
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence_before()
                    cute.arch.mbarrier_arrive(p_ready + buf)
                    stamp(dbg, bidx, 64 + (t * 8 + 6) * 2 + wg, st_on)
                    stamp(dbg, bidx, 1020 + t * 4 + 2, s1_on)
                # ---- tile end: WG0 hands its partial sum to WG1 and moves on; WG1 drains O and stores the tile
                if wg == 0:
                    lx[row] = l
                    lx[128 + row] = m_known
                    cute.arch.barrier_arrive(barrier_id=4, number_of_threads=256)
                else:
                    cute.arch.barrier(barrier_id=4, number_of_threads=256)
                    l0 = lx[row]
                    m0 = lx[128 + row]
                    l_tot = l + l0 * cute.arch.exp2((m0 - m_known) * scale_log2)
                    inv_l = cutlass.Float32(1.0) / l_tot
                    g_last = g_base + n_kv_tiles - 1
                    mbar_wait_dbg(pv_done + (g_last % 3), (g_last // 3) & 1, dbg, bidx, 18 + wg)
                    tcgen05_fence_after()
                    if warp_idx == 8:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)   # previous tile's O store finished reading sO
                    cute.arch.barrier(barrier_id=3, number_of_threads=128)
                    epilogue_store_both(tmem_ptr, self.o_col, half_layout, ld_atom, row, tSr0, tSr1, inv_l, tOw4s, dst8s, o_free)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=3, number_of_threads=128)
                    if cutlass.const_expr(not self.skip_o_store):
                        if warp_idx == 8:
                            cute.copy(tma_atom_o, tOsO[(None,)], tOgO[(None, q_tile, 0, h, b)])
                            cute.arch.cp_async_bulk_commit_group()
            if warp_idx == 8:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            mark(dbg, bidx, 12 + wg)
        cute.arch.sync_threads()
        if cutlass.const_expr(PROFILE_STAMPS):
            ts1 = globaltimer_lo()
            ts6 = globaltimer_lo()
        if cutlass.const_expr(not self.skip_tmem_free):
            tmem.free(tmem_ptr)
        if cutlass.const_expr(PROFILE_STAMPS):
            ts2 = globaltimer_lo()
            if tidx == 0:
                dbg[(bidx, 8)] = clock32()
                dbg[(bidx, 1)] = ts1
                dbg[(bidx, 6)] = ts6
                dbg[(bidx, 4)] = ts2


# ---------------------------------------------------------------------------------------------- host side
_compiled = {}

def _mk(kv_stages, emul_pairs, same_kv):
    kk = FlashFwdSm100(kv_stages, emul_pairs); kk.same_kv = same_kv; return kk

def flash_fwd(q, k, v, kv_stages=2, emul_pairs=0, same_kv=False):
    """q: (B,S,H,D) bf16, k/v: (B,S,HKV,D) bf16 -> o (B,S,H,D) bf16. Non-causal, softmax scale 1/sqrt(D)."""
    B, S, H, D = q.shape
    assert D == 128 and S % 128 == 0 and (S // 128) % 2 == 0 and H % k.shape[2] == 0, "kernel specialized for D=128, S multiple of 256"
    o = torch.empty_like(q)
    qt = from_dlpack(q.permute(1, 3, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)   # (S, D, H, B)
    kt = from_dlpack(k.permute(1, 3, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)   # (S, D, HKV, B)
    vt = from_dlpack(v.permute(3, 1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0)   # (D, S, HKV, B)
    ot = from_dlpack(o.permute(1, 3, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    scale_log2 = cutlass.Float32((1.0 / math.sqrt(D)) * LOG2E)
    n_ctas = (S // 128) * H * B
    if getattr(flash_fwd, "_dbg_key", None) != (n_ctas, q.device):
        flash_fwd._dbg = torch.zeros(n_ctas, 1152, dtype=torch.int32, device=q.device)
        flash_fwd._dbg_key = (n_ctas, q.device)
    dbg = flash_fwd._dbg
    dbg_t = from_dlpack(dbg, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (kv_stages, emul_pairs, same_kv)
    if key not in _compiled:
        import time; t0 = time.time()
        _compiled[key] = cute.compile(_mk(kv_stages, emul_pairs, same_kv), qt, kt, vt, ot, scale_log2, dbg_t, stream)
        print(f'[compile {time.time()-t0:.1f}s]', flush=True)
    _compiled[key](qt, kt, vt, ot, scale_log2, dbg_t, stream)
    flash_fwd.last_dbg = dbg
    return o
