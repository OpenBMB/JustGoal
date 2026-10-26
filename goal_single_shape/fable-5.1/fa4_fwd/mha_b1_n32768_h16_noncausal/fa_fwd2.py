"""2-CTA (cluster of 2 SMs, cta_group::2 MMAs) variant of the from-scratch attention forward kernel.

Layout (B, S, H, D), B=1, H_q == H_kv.  Written without reference to any existing attention kernel source.

Design (per CTA = 256 query rows of one head, 384 threads):
  warp 0      : TMA producer (Q once, then K/V tiles through 2-stage smem pipelines)
  warp 1      : tcgen05 MMA issuer  (S = Q K^T -> TMEM, O += P V -> TMEM, l += P 1 -> TMEM)
  warps 4-7   : softmax warpgroup 0 (query rows   0..127 of the CTA tile)
  warps 8-11  : softmax warpgroup 1 (query rows 128..255 of the CTA tile)
  KV tile = 112 rows. S tile (128 x 112 fp32) lives in TMEM; P (bf16) is written in place over S.
  Row sums are produced by an extra N=8 MMA against a ones tile (no CUDA-core adds).
  Online softmax with lazy rescaling: O/l are rescaled only when the running max grows by > 2^8.
  exp2: a fraction of elements use MUFU ex2, the rest a degree-3 polynomial on the FMA pipe.
  MMA issue order per kv tile j:  S0_j | PV1_{j-1} | S1_j | PV0_j   (WG ping-pong; each WG gets a full
  MMA half-period to run softmax on its S tile while the tensor core works on the other WG).
"""
import math
import os as _os
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm, nvvm, arith
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda

# ----------------------------------------------------------------------------- config
D = 128
TILE_M = 128            # query rows per softmax warpgroup
NUM_WG = 2              # softmax warpgroups per CTA
CTA_M = TILE_M * NUM_WG  # 256
TILE_N = int(_os.environ.get("FA_TILE_N", "128"))   # kv rows per tile
ROWSUM_MMA = 0
LAZY_MAX = int(_os.environ.get("FA_LAZY_MAX", "1"))
CHUNKED = int(_os.environ.get("FA_CHUNKED", "1"))
EARLY_PST = int(_os.environ.get("FA_EARLY_PST", "0"))     # store each P half to TMEM as soon as it is complete         # process S in 32-column chunks with streaming TMEM loads (lazy path)       # 1: exp with previous running max, sum-guarded exact-max recompute path
GUARD = 2.0 ** float(_os.environ.get("FA_GUARD_LOG2", "120"))   # row-sum guard of the lazy path (2^120 in production; lower it to exercise the rescale path)
PINGPONG = int(_os.environ.get("FA_PINGPONG", "0"))       # strict alternation of the two warpgroups' exp phases (named barriers)
POLY_DEG = int(_os.environ.get("FA_POLY_DEG", "3"))       # exp2 emulation polynomial degree (2 or 3)
EMU_CLAMP = int(_os.environ.get("FA_EMU_CLAMP", "1"))     # clamp x >= -120 in the emulated path (needed for robustness)
LD_REP = int(_os.environ.get("FA_LD_REP", "32"))          # TMEM load repetition for S (16 or 32)
PACK_ALU = int(_os.environ.get("FA_PACK_ALU", "0"))       # 1: fp32->bf16x2 via integer add + prmt (ALU pipe) instead of cvt (F2FP)
CLUSTER_M = 2                    # 2-CTA MMA: instruction M = 256 spans the CTA pair
MMA_M = TILE_M * CLUSTER_M       # 256
L_N = 16                         # row-sum MMA N (cta_group::2 needs N >= 16)
K_STAGES = 2
V_STAGES = 2
assert K_STAGES == V_STAGES == 2
NUM_KB_QK = D // 16
NUM_KB_PV = TILE_N // 16
THREADS = 128 * (1 + NUM_WG)
EMU_PAIRS = int(_os.environ.get("FA_EMU_PAIRS", "2"))       # of every 8 column-pairs, this many use the polynomial exp2
SOFTMAX_MODE = int(_os.environ.get("FA_SOFTMAX_MODE", "2"))  # 0: no softmax work (timing only), 1: TMEM ld/st only, 2: full
PRODUCER_REGS = int(_os.environ.get("FA_PRODUCER_REGS", "40"))
NO_TMA_KV = int(_os.environ.get("FA_NO_TMA_KV", "0"))   # timing experiment: skip K/V loads (garbage results)
NO_LMMA = int(_os.environ.get("FA_NO_LMMA", "0")) or (not ROWSUM_MMA)   # skip row-sum MMAs
PROF = int(_os.environ.get("FA_PROF", "0"))             # in-kernel clock64 instrumentation -> prof buffer
SOFTMAX_REGS = (65536 // THREADS // 8 * 8 * THREADS - 128 * PRODUCER_REGS) // 256 // 8 * 8
RESCALE_THRESH_LOG2 = 8.0
MAGIC = 12582912.0      # 1.5 * 2^23
C0, C1, C2, C3 = 0.999928280583151, 0.693260926275332, 0.24260790199985344, 0.055170250079494526
SCHR_MAGIC = 384.0                       # 1.5 * 2^(23-15): 15 fractional bits of x kept in the mantissa
SCHR_TCLAMP = SCHR_MAGIC - 126.0
SCHR_CONST = ((127 << 23) - (0x43C00000 << 8)) & 0xFFFFFFFF
SCHR_CONST = SCHR_CONST - (1 << 32) if SCHR_CONST >= (1 << 31) else SCHR_CONST
SD2 = (0.23374654249497664, -0.6945535230677414, 1.4574618104430557)          # 2^(m-1)/m on [1,2), degree 2 (max rel 3.5e-3)
SD3 = (-0.10646725460536235, 0.7077268630716722, -1.377909088190583, 1.7760058455002197)   # degree 3 (max rel 6.8e-4)
NEG_INF = float("-inf")
# TMEM column map (512 columns): [S0 0..111][l0 112..119] [S1 128..239][l1 240..247] [O0 256..383] [O1 384..511]
S_COL0, L_COL0, O_COL0 = 0, 112, 256
assert ROWSUM_MMA == 0 or TILE_N <= 112
WG_COL_STRIDE_S, WG_COL_STRIDE_O = 128, 128
NUM_TMEM_COLS = 512
Q_BYTES = CTA_M * D * 2
K_BYTES = TILE_N * D * 2         # both CTAs' halves (transaction count on the leader barrier)
V_BYTES = TILE_N * D * 2

# ----------------------------------------------------------------------------- small device helpers


@dsl_user_op
def fmax3(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip),
             cutlass.Float32(c).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fmax2(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def ex2(a, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;", "=f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round-to-nearest convert two fp32 to bf16 and pack: lo -> bits[15:0], hi -> bits[31:16]."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def exp2_f16x2(x0, x1, *, loc=None, ip=None):
    """packed fp16 2^x for two fp32 inputs: returns bits with 2^x0 in [15:0] and 2^x1 in [31:16]."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Float32(x0).ir_value(loc=loc, ip=ip), cutlass.Float32(x1).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 h;\ncvt.rn.f16x2.f32 h, $2, $1;\nex2.approx.f16x2 $0, h;\n}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def exp2_f16x2_to_bf16x2(x0, x1, *, loc=None, ip=None):
    """2^x for two fp32 inputs via packed fp16 MUFU, converted to packed bf16 (x0 -> bits[15:0])."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Float32(x0).ir_value(loc=loc, ip=ip), cutlass.Float32(x1).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 h;\n.reg .b16 l, u;\n.reg .f32 f0, f1;\ncvt.rn.f16x2.f32 h, $2, $1;\nex2.approx.f16x2 h, h;\n"
            "mov.b32 {l, u}, h;\ncvt.f32.f16 f0, l;\ncvt.f32.f16 f1, u;\ncvt.rn.bf16x2.f32 $0, f1, f0;\n}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_f16x2(lo, hi, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def f32_bits(a, *, loc=None, ip=None):
    return cutlass.Int32(arith.bitcast(T.i32(), cutlass.Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def bits_f32(a, *, loc=None, ip=None):
    return cutlass.Float32(arith.bitcast(T.f32(), cutlass.Int32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def umma_ss(d_tmem, a_lo, a_hi, a_off: int, b_lo, b_hi, b_off: int, idesc: int, acc, *, loc=None, ip=None):
    """tcgen05.mma (A, B smem descriptors = base + immediate offset). Caller must be a single elected thread."""
    ops = [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (d_tmem, a_lo, a_hi, b_lo, b_hi, acc)]
    llvm.inline_asm(
        None, ops,
        "{\n.reg .pred p;\n.reg .b64 da, db;\n.reg .b32 al, bl, id;\n"
        f"add.u32 al, $1, {a_off};\nadd.u32 bl, $3, {b_off};\nmov.b32 id, {idesc};\n"
        "mov.b64 da, {al, $2};\nmov.b64 db, {bl, $4};\nsetp.ne.b32 p, $5, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, id, p;\n}",
        "r,r,r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def umma_ts(d_tmem, a_tmem, a_off: int, b_lo, b_hi, b_off: int, idesc: int, acc, *, loc=None, ip=None):
    """tcgen05.mma (A from TMEM = base + imm, B smem descriptor = base + imm). Caller must be a single elected thread."""
    ops = [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (d_tmem, a_tmem, b_lo, b_hi, acc)]
    llvm.inline_asm(
        None, ops,
        "{\n.reg .pred p;\n.reg .b64 db;\n.reg .b32 at, bl, id;\n"
        f"add.u32 at, $1, {a_off};\nadd.u32 bl, $2, {b_off};\nmov.b32 id, {idesc};\n"
        "mov.b64 db, {bl, $3};\nsetp.ne.b32 p, $4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [$0], [at], db, id, p;\n}",
        "r,r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def desc_offsets(smem_tensor, nkb, nstage):
    """trace-time: 16-byte-unit descriptor start-address offsets of (k-block, stage) slices of a smem operand tile."""
    lay = smem_tensor.layout
    return [[cute.crd2idx((0, 0, kb, st), lay) * 2 // 16 for st in range(nstage)] for kb in range(nkb)]


IDESC_QK = (1 << 4) | (1 << 7) | (1 << 10) | ((TILE_N >> 3) << 17) | ((TILE_M >> 4) << 24)                 # D=f32, A=B=bf16, K-major both
IDESC_PV = (1 << 4) | (0 << 7) | (1 << 10) | (1 << 16) | ((D >> 3) << 17) | ((TILE_M >> 4) << 24)           # A f16, B bf16 MN-major, N=128
IDESC_L = (1 << 4) | (0 << 7) | (1 << 10) | (1 << 16) | ((8 >> 3) << 17) | ((TILE_M >> 4) << 24)            # A f16, B bf16 MN-major, N=8
ASM_MMA = int(_os.environ.get("FA_ASM_MMA", "0"))
P_F16 = int(_os.environ.get("FA_P_F16", "0"))   # P in fp16 via ex2.approx.f16x2 (A operand f16, B bf16)
P_DTYPE = cutlass.Float16 if P_F16 else cutlass.BFloat16
EXP_MODE = int(_os.environ.get("FA_EXP_MODE", "0"))   # 0: fp32 MUFU / poly mix (EMU_PAIRS); 1: packed fp16 MUFU for all pairs


@dsl_user_op
def clock64(*, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


def fma2(a, b, c):
    """packed fp32x2 fma on pairs (tuples of two Float32)."""
    return cute.arch.fma_packed_f32x2(a, b, c)


def add2(a, b):
    return cute.arch.add_packed_f32x2(a, b)


Q0, Q1, Q2 = 1.0004421303389959, 0.7034229839741121, 0.23841580061481452   # degree-2 minimax on [-0.5, 0.5]


@dsl_user_op
def pack_bf16x2_alu(lo, hi, *, loc=None, ip=None):
    """fp32 pair -> packed bf16x2 by round-half-up on the ALU pipe: (bits + 0x8000) >> 16, lo -> bits[15:0]."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Float32(lo).ir_value(loc=loc, ip=ip), cutlass.Float32(hi).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\nadd.u32 a, a, 0x8000;\nadd.u32 b, b, 0x8000;\nprmt.b32 $0, a, b, 0x7632;\n}",
            "=r,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def exp2_emu_pair(x0, x1):
    """2^x for a pair via Cody-Waite style range reduction + polynomial + exponent bit injection."""
    if EMU_CLAMP:
        x0 = fmax2(x0, -120.0)
        x1 = fmax2(x1, -120.0)
    t0, t1 = add2((x0, x1), (MAGIC, MAGIC))          # MAGIC + round(x)
    n0, n1 = add2((t0, t1), (-MAGIC, -MAGIC))        # round(x)
    f0, f1 = fma2((n0, n1), (-1.0, -1.0), (x0, x1))  # x - round(x) in [-0.5, 0.5]
    if POLY_DEG == 3:
        p0, p1 = fma2((f0, f1), (C3, C3), (C2, C2))
        p0, p1 = fma2((f0, f1), (p0, p1), (C1, C1))
        p0, p1 = fma2((f0, f1), (p0, p1), (C0, C0))
    else:
        p0, p1 = fma2((f0, f1), (Q2, Q2), (Q1, Q1))
        p0, p1 = fma2((f0, f1), (p0, p1), (Q0, Q0))
    r0 = bits_f32(f32_bits(p0) + (f32_bits(t0) << 23))
    r1 = bits_f32(f32_bits(p1) + (f32_bits(t1) << 23))
    return r0, r1


def row_max(vals):
    """max over a python list of Float32 SSA values using 3-input max, as a balanced tree."""
    vals = list(vals)
    while len(vals) > 1:
        nxt = []
        i = 0
        while i + 2 < len(vals):
            nxt.append(fmax3(vals[i], vals[i + 1], vals[i + 2]))
            i += 3
        if i + 1 < len(vals):
            nxt.append(fmax2(vals[i], vals[i + 1]))
        elif i < len(vals):
            nxt.append(vals[i])
        vals = nxt
    return vals[0]


# ----------------------------------------------------------------------------- MMA issue helpers (warp 1)
@cute.jit
def issue_qk(mma_first, mma_acc, tS, tQ, tK, w: cutlass.Constexpr, kstage: cutlass.Constexpr):
    cute.gemm(mma_first, tS, tQ[(None, None, 0, w)], tK[(None, None, 0, kstage)], tS)
    for kb in cutlass.range_constexpr(1, NUM_KB_QK):
        cute.gemm(mma_acc, tS, tQ[(None, None, kb, w)], tK[(None, None, kb, kstage)], tS)


@cute.jit
def issue_pv(mma_pv_first, mma_pv_acc, mma_l_first, mma_l_acc, tO, tL, tPpv, tPl, tV, tOnes,
             vstage: cutlass.Constexpr, accumulate):
    if accumulate:
        cute.gemm(mma_pv_acc, tO, tPpv[(None, None, 0)], tV[(None, None, 0, vstage)], tO)
        if cutlass.const_expr(not NO_LMMA):
            cute.gemm(mma_l_acc, tL, tPl[(None, None, 0)], tOnes[(None, None, 0, 0)], tL)
    else:
        cute.gemm(mma_pv_first, tO, tPpv[(None, None, 0)], tV[(None, None, 0, vstage)], tO)
        if cutlass.const_expr(not NO_LMMA):
            cute.gemm(mma_l_first, tL, tPl[(None, None, 0)], tOnes[(None, None, 0, 0)], tL)
    for kb in cutlass.range_constexpr(1, NUM_KB_PV):
        cute.gemm(mma_pv_acc, tO, tPpv[(None, None, kb)], tV[(None, None, kb, vstage)], tO)
        if cutlass.const_expr(not NO_LMMA):
            cute.gemm(mma_l_acc, tL, tPl[(None, None, kb)], tOnes[(None, None, kb, 0)], tL)


@cute.jit
def mma_tile(j, stage: cutlass.Constexpr,
             mma_qk, mma_qk_acc, mma_pv, mma_pv_acc, mma_l, mma_l_acc,
             tQ, tK, tV, tOnes, tS0, tS1, tO0, tO1, tL0, tL1, tPpv0, tPpv1, tPl0, tPl1,
             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base,
             t_kv, t_p):
    """One kv tile of MMA issue: S0_j | PV1_{j-1} | S1_j | PV0_j.  K/V stage indices are static."""
    kphase = (j // K_STAGES) & 1
    if cutlass.const_expr(PROF):
        c0 = clock64()
    cute.arch.mbarrier_wait(k_full_base + stage, kphase)
    if cutlass.const_expr(PROF):
        t_kv = t_kv + (clock64() - c0)
    tc_fence_after()
    issue_qk(mma_qk, mma_qk_acc, tS0, tQ, tK, 0, stage)
    with cute.arch.elect_one():
        tcgen05.commit(s_full_base + 0, 0b11, tcgen05.CtaGroup.TWO)
    if j > 0:
        jp = j - 1
        if cutlass.const_expr(PROF):
            c1 = clock64()
        cute.arch.mbarrier_wait(p_ready_base + 1, jp & 1)
        if cutlass.const_expr(PROF):
            t_p = t_p + (clock64() - c1)
        tc_fence_after()
        issue_pv(mma_pv, mma_pv_acc, mma_l, mma_l_acc, tO1, tL1, tPpv1, tPl1, tV, tOnes, 1 - stage, cutlass.Boolean(jp > 0))
        with cute.arch.elect_one():
            tcgen05.commit(pv_done_base + 2 + (1 - stage), 0b11, tcgen05.CtaGroup.TWO)
            tcgen05.commit(v_empty_base + (1 - stage), 0b11, tcgen05.CtaGroup.TWO)
    issue_qk(mma_qk, mma_qk_acc, tS1, tQ, tK, 1, stage)
    with cute.arch.elect_one():
        tcgen05.commit(s_full_base + 1, 0b11, tcgen05.CtaGroup.TWO)
        tcgen05.commit(k_empty_base + stage, 0b11, tcgen05.CtaGroup.TWO)
    if cutlass.const_expr(PROF):
        c2 = clock64()
    cute.arch.mbarrier_wait(v_full_base + stage, kphase)
    if cutlass.const_expr(PROF):
        c3 = clock64()
        t_kv = t_kv + (c3 - c2)
    cute.arch.mbarrier_wait(p_ready_base + 0, j & 1)
    if cutlass.const_expr(PROF):
        t_p = t_p + (clock64() - c3)
    tc_fence_after()
    issue_pv(mma_pv, mma_pv_acc, mma_l, mma_l_acc, tO0, tL0, tPpv0, tPl0, tV, tOnes, stage, cutlass.Boolean(j > 0))
    with cute.arch.elect_one():
        tcgen05.commit(pv_done_base + 0 + stage, 0b11, tcgen05.CtaGroup.TWO)
    return t_kv, t_p


@cute.jit
def mma_tile_asm(j, stage: cutlass.Constexpr, offs: cutlass.Constexpr, tmem_base, q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, o_lo, o_hi,
                 k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p):
    """One kv tile of MMA issue with inline-PTX MMAs: S0_j | PV1_{j-1} | S1_j | PV0_j. Stages static."""
    OQ, OK, OV, OO = offs
    kphase = (j // K_STAGES) & 1
    s0 = tmem_base + S_COL0
    s1 = tmem_base + (S_COL0 + WG_COL_STRIDE_S)
    o0 = tmem_base + O_COL0
    o1 = tmem_base + (O_COL0 + WG_COL_STRIDE_O)
    l0 = tmem_base + L_COL0
    l1 = tmem_base + (L_COL0 + WG_COL_STRIDE_S)
    if cutlass.const_expr(PROF):
        c0 = clock64()
    cute.arch.mbarrier_wait(k_full_base + stage, kphase)
    if cutlass.const_expr(PROF):
        t_kv = t_kv + (clock64() - c0)
    tc_fence_after()
    # S0_j
    with cute.arch.elect_one():
        for kb in cutlass.range_constexpr(NUM_KB_QK):
            umma_ss(s0, q_lo, q_hi, OQ[kb][0], k_lo, k_hi, OK[kb][stage], IDESC_QK, 1 if kb > 0 else 0)
        tcgen05.commit(s_full_base + 0, 0b11, tcgen05.CtaGroup.TWO)
    # PV1_{j-1}
    if j > 0:
        jp = j - 1
        accp = cutlass.min(jp, 1)
        if cutlass.const_expr(PROF):
            c1 = clock64()
        cute.arch.mbarrier_wait(p_ready_base + 1, jp & 1)
        if cutlass.const_expr(PROF):
            t_p = t_p + (clock64() - c1)
        tc_fence_after()
        with cute.arch.elect_one():
            for kb in cutlass.range_constexpr(NUM_KB_PV):
                umma_ts(o1, s1, kb * 8, v_lo, v_hi, OV[kb][1 - stage], IDESC_PV, accp if kb == 0 else 1)
                if cutlass.const_expr(not NO_LMMA):
                    umma_ts(l1, s1, kb * 8, o_lo, o_hi, OO[kb][0], IDESC_L, accp if kb == 0 else 1)
            tcgen05.commit(pv_done_base + 2 + (1 - stage), 0b11, tcgen05.CtaGroup.TWO)
            tcgen05.commit(v_empty_base + (1 - stage), 0b11, tcgen05.CtaGroup.TWO)
    # S1_j
    with cute.arch.elect_one():
        for kb in cutlass.range_constexpr(NUM_KB_QK):
            umma_ss(s1, q_lo, q_hi, OQ[kb][1], k_lo, k_hi, OK[kb][stage], IDESC_QK, 1 if kb > 0 else 0)
        tcgen05.commit(s_full_base + 1, 0b11, tcgen05.CtaGroup.TWO)
        tcgen05.commit(k_empty_base + stage, 0b11, tcgen05.CtaGroup.TWO)
    # PV0_j
    acc0 = cutlass.min(j, 1)
    if cutlass.const_expr(PROF):
        c2 = clock64()
    cute.arch.mbarrier_wait(v_full_base + stage, kphase)
    if cutlass.const_expr(PROF):
        c3 = clock64()
        t_kv = t_kv + (c3 - c2)
    cute.arch.mbarrier_wait(p_ready_base + 0, j & 1)
    if cutlass.const_expr(PROF):
        t_p = t_p + (clock64() - c3)
    tc_fence_after()
    with cute.arch.elect_one():
        for kb in cutlass.range_constexpr(NUM_KB_PV):
            umma_ts(o0, s0, kb * 8, v_lo, v_hi, OV[kb][stage], IDESC_PV, acc0 if kb == 0 else 1)
            if cutlass.const_expr(not NO_LMMA):
                umma_ts(l0, s0, kb * 8, o_lo, o_hi, OO[kb][0], IDESC_L, acc0 if kb == 0 else 1)
        tcgen05.commit(pv_done_base + 0 + stage, 0b11, tcgen05.CtaGroup.TWO)
    return t_kv, t_p


@cute.jit
def pv1_tail_asm(jp, stage: cutlass.Constexpr, offs: cutlass.Constexpr, tmem_base, v_lo, v_hi, o_lo, o_hi):
    OQ, OK, OV, OO = offs
    s1 = tmem_base + (S_COL0 + WG_COL_STRIDE_S)
    o1 = tmem_base + (O_COL0 + WG_COL_STRIDE_O)
    l1 = tmem_base + (L_COL0 + WG_COL_STRIDE_S)
    accp = cutlass.min(jp, 1)
    with cute.arch.elect_one():
        for kb in cutlass.range_constexpr(NUM_KB_PV):
            umma_ts(o1, s1, kb * 8, v_lo, v_hi, OV[kb][stage], IDESC_PV, accp if kb == 0 else 1)
            if cutlass.const_expr(not NO_LMMA):
                umma_ts(l1, s1, kb * 8, o_lo, o_hi, OO[kb][0], IDESC_L, accp if kb == 0 else 1)


# ---------------------------------------------------------------------------- softmax helpers (warps 4..11)
_trace_printed = [False]


def exp_chunk(sc, neg_mc, scale_log2, rP, off, acc, cbase):
    """one 32-column chunk: sc = list of 32 Float32 (columns cbase..cbase+31); packed bf16x2 into rP[off + c]; acc: 4 packed partial sums.
    EXP_MODE 2: emulated pairs use t = x + MAGIC (15 fractional bits), one shift-add exponent injection and a short
    polynomial correction in the mantissa (scale FMA folded into t)."""
    if EXP_MODE == 2:
        bM = neg_mc + SCHR_MAGIC
    for c in range(16):
        cg = cbase // 2 + c                      # global pair index (selects MUFU vs emulated)
        if EXP_MODE == 2 and (cg % 8) < EMU_PAIRS:
            t0, t1 = fma2((sc[2 * c], sc[2 * c + 1]), (scale_log2, scale_log2), (bM, bM))
            if EMU_CLAMP:
                t0 = fmax2(t0, SCHR_TCLAMP)
                t1 = fmax2(t1, SCHR_TCLAMP)
            r0b = (f32_bits(t0) << 8) + SCHR_CONST
            r1b = (f32_bits(t1) << 8) + SCHR_CONST
            r0 = bits_f32(r0b)
            r1 = bits_f32(r1b)
            m0 = bits_f32((r0b & 0x007FFFFF) | 0x3F800000)
            m1 = bits_f32((r1b & 0x007FFFFF) | 0x3F800000)
            if POLY_DEG == 3:
                q0, q1 = fma2((m0, m1), (SD3[0], SD3[0]), (SD3[1], SD3[1]))
                q0, q1 = fma2((m0, m1), (q0, q1), (SD3[2], SD3[2]))
                q0, q1 = fma2((m0, m1), (q0, q1), (SD3[3], SD3[3]))
            else:
                q0, q1 = fma2((m0, m1), (SD2[0], SD2[0]), (SD2[1], SD2[1]))
                q0, q1 = fma2((m0, m1), (q0, q1), (SD2[2], SD2[2]))
            p0, p1 = fma2((r0, r1), (q0, q1), (0.0, 0.0))
        else:
            x0, x1 = fma2((sc[2 * c], sc[2 * c + 1]), (scale_log2, scale_log2), (neg_mc, neg_mc))
            if (cg % 8) < EMU_PAIRS:
                p0, p1 = exp2_emu_pair(x0, x1)
            else:
                p0 = ex2(x0)
                p1 = ex2(x1)
        if PACK_ALU:
            rP[off + c] = pack_bf16x2_alu(p0, p1)
        else:
            rP[off + c] = pack_bf16x2(p0, p1)
        acc[c % 4] = (p0, p1) if acc[c % 4] is None else add2(acc[c % 4], (p0, p1))


def exp_pass(s, neg_mc, scale_log2, rP):
    """p = 2^(s*c - m*c) for one row (python list of Float32), packed bf16x2 into rP; returns the fp32 row sum."""
    if not _trace_printed[0]:
        _trace_printed[0] = True
        print(f"[trace] exp_pass: PACK_ALU={PACK_ALU} EMU_PAIRS={EMU_PAIRS} POLY_DEG={POLY_DEG} EMU_CLAMP={EMU_CLAMP} LAZY_MAX={LAZY_MAX} PINGPONG={PINGPONG}", flush=True)
    acc = [(cutlass.Float32(0.0), cutlass.Float32(0.0)) for _ in range(4)]
    for c in range(TILE_N // 2):
        x0, x1 = fma2((s[2 * c], s[2 * c + 1]), (scale_log2, scale_log2), (neg_mc, neg_mc))
        if (c % 8) < EMU_PAIRS:
            p0, p1 = exp2_emu_pair(x0, x1)
        else:
            p0 = ex2(x0)
            p1 = ex2(x1)
        if PACK_ALU:
            rP[c] = pack_bf16x2_alu(p0, p1)
        else:
            rP[c] = pack_bf16x2(p0, p1)
        acc[c % 4] = add2(acc[c % 4], (p0, p1))
    a01 = add2(acc[0], acc[1])
    a23 = add2(acc[2], acc[3])
    a = add2(a01, a23)
    return a[0] + a[1]


@cute.jit
def rescale_o(alpha, tiled_ld_o, tTR_tO, rO, tiled_st_o, tRT_tO):
    """O (TMEM, 128 columns) *= alpha for this thread's row."""
    for ci in cutlass.range_constexpr(D // 32):
        cute.copy(tiled_ld_o, tTR_tO[(None, None, ci)], rO)
        cute.arch.fence_view_async_tmem_load()
        for e in cutlass.range_constexpr(32):
            rO[e] = rO[e] * alpha
        cute.copy(tiled_st_o, rO, tRT_tO[(None, None, ci)])
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def softmax_tile(j, first: cutlass.Constexpr, masked: cutlass.Constexpr, valid_last: cutlass.Constexpr, m_run, l_run,
                 wg, s_full_base, p_ready_base, pv_done_base, prof, is_trace,
                 tiled_ld_s, tTR_tS, rS, rSc0, rSc1, rSc2, rSc3, tiled_st_p, rP, rPh0, rPh1, tTR_tP,
                 tiled_ld_o, tTR_tO, rO, tiled_st_o, tRT_tO,
                 scale_log2, thresh_raw, t_wait, t_work, n_resc, t_load, t_store):
    """Softmax for one (128 x TILE_N) S tile of this warpgroup; returns updated (m_run, l_run) and profile counters."""
    rSc = [rSc0, rSc1, rSc2, rSc3]
    rPh = [rPh0, rPh1]
    p_full_store = cutlass.Boolean(False)
    if cutlass.const_expr(PROF):
        c0 = clock64()
    cute.arch.mbarrier_wait(s_full_base + wg, j & 1)
    if cutlass.const_expr(PROF):
        c1 = clock64()
        t_wait = t_wait + (c1 - c0)
        if is_trace:
          if j == 100:
              prof[20] = c0
              prof[21] = c1
    tc_fence_after()
    if cutlass.const_expr(SOFTMAX_MODE >= 1):
        if cutlass.const_expr(CHUNKED and not first and LAZY_MAX):
            cute.copy(tiled_ld_s, tTR_tS[(None, None, 0)], rSc[0])
        else:
            cute.copy(tiled_ld_s, tTR_tS, rS)
        cute.arch.fence_view_async_tmem_load()
    if cutlass.const_expr(PROF):
        c1b = clock64()
        t_load = t_load + (c1b - c1)
        if is_trace:
          if j == 100:
              prof[22] = c1b + cutlass.Int64(f32_bits(rS[0]) & 0)
    if cutlass.const_expr(PINGPONG):
        # strict alternation of the two warpgroups' exp phases (WG0 starts first on tile 0)
        if cutlass.const_expr(not first):
            cute.arch.barrier(barrier_id=2 + wg, number_of_threads=256)
        else:
            if wg == 1:
                cute.arch.barrier(barrier_id=3, number_of_threads=256)
    if cutlass.const_expr(SOFTMAX_MODE == 2):
        if cutlass.const_expr(masked):
            for i in cutlass.range_constexpr(valid_last, TILE_N):
                rS[i] = cutlass.Float32(NEG_INF)
        if cutlass.const_expr(not (CHUNKED and not first and LAZY_MAX)):
            s = [rS[i] for i in range(TILE_N)]
        if cutlass.const_expr(first or not LAZY_MAX):
            # ---- exact-max path
            m_tile = row_max(s)
            if cutlass.const_expr(first):
                m_run = m_tile
            else:
                need = m_tile > (m_run + thresh_raw)
                if cute.arch.vote_any_sync(need):
                    if cutlass.const_expr(PROF):
                        n_resc = n_resc + 1
                    m_new = fmax2(m_run, m_tile)
                    alpha = ex2((m_run - m_new) * scale_log2)
                    m_run = m_new
                    l_run = l_run * alpha
                    cute.arch.mbarrier_wait(pv_done_base + 2 * wg + ((j - 1) & 1), ((j - 1) >> 1) & 1)
                    tc_fence_after()
                    rescale_o(alpha, tiled_ld_o, tTR_tO, rO, tiled_st_o, tRT_tO)
            neg_mc = -(m_run * scale_log2)
            l_tile = exp_pass(s, neg_mc, scale_log2, rP)
            l_run = l_run + l_tile
        else:
            # ---- lazy path: exponentiate with the previous running max; the row sum guards against overflow.
            # S is consumed in 32-column chunks with the next chunk's TMEM load in flight (bounded register liveness).
            neg_mc = -(m_run * scale_log2)
            acc = [(cutlass.Float32(0.0), cutlass.Float32(0.0)) for _ in range(4)]
            if cutlass.const_expr(CHUNKED):
                for ck in cutlass.range_constexpr(4):
                    if cutlass.const_expr(ck + 1 < 4):
                        cute.copy(tiled_ld_s, tTR_tS[(None, None, ck + 1)], rSc[ck + 1])
                    sc = [rSc[ck][i] for i in range(32)]
                    exp_chunk(sc, neg_mc, scale_log2, rPh[ck // 2], (ck % 2) * 16, acc, ck * 32)
                    if cutlass.const_expr(EARLY_PST and ck == 1):
                        cute.copy(tiled_st_p, rPh[0], tTR_tP[(None, None, 0)])
                    if cutlass.const_expr(ck + 1 < 4):
                        cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(EARLY_PST):
                    cute.copy(tiled_st_p, rPh[1], tTR_tP[(None, None, 1)])
                a01 = add2(acc[0], acc[1])
                a23 = add2(acc[2], acc[3])
                a = add2(a01, a23)
                l_tile = a[0] + a[1]
            else:
                l_tile = exp_pass(s, neg_mc, scale_log2, rP)
            need2 = l_tile > GUARD
            if cute.arch.vote_any_sync(need2):
                p_full_store = cutlass.Boolean(True)
                if cutlass.const_expr(PROF):
                    n_resc = n_resc + 1
                if cutlass.const_expr(CHUNKED and EARLY_PST):
                    # P (packed) already overwrote TMEM columns 0..63 of this S tile: chunks 0,1 are still in registers
                    cute.copy(tiled_ld_s, tTR_tS[(None, None, 2)], rSc[2])
                    cute.copy(tiled_ld_s, tTR_tS[(None, None, 3)], rSc[3])
                    cute.arch.fence_view_async_tmem_load()
                    s2 = [rSc[0][i] for i in range(32)] + [rSc[1][i] for i in range(32)] + [rSc[2][i] for i in range(32)] + [rSc[3][i] for i in range(32)]
                else:
                    cute.copy(tiled_ld_s, tTR_tS, rS)
                    cute.arch.fence_view_async_tmem_load()
                    s2 = [rS[i] for i in range(TILE_N)]
                m_new2 = fmax2(m_run, row_max(s2))
                alpha2 = ex2((m_run - m_new2) * scale_log2)
                m_run = m_new2
                l_run = l_run * alpha2
                cute.arch.mbarrier_wait(pv_done_base + 2 * wg + ((j - 1) & 1), ((j - 1) >> 1) & 1)
                tc_fence_after()
                rescale_o(alpha2, tiled_ld_o, tTR_tO, rO, tiled_st_o, tRT_tO)
                neg_mc = -(m_run * scale_log2)
                l_tile = exp_pass(s2, neg_mc, scale_log2, rP)
            l_run = l_run + l_tile
    if cutlass.const_expr(PROF):
        if is_trace:
          if j == 100:
              prof[23] = clock64() + cutlass.Int64(rP[TILE_N // 2 - 1] & 0)
    if cutlass.const_expr(PINGPONG):
        cute.arch.barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
    if cutlass.const_expr(PROF):
        c2 = clock64()
    if cutlass.const_expr(SOFTMAX_MODE >= 1):
        if cutlass.const_expr(not (CHUNKED and not first and LAZY_MAX)):
            cute.copy(tiled_st_p, rP, tTR_tP)
        else:
            if p_full_store:
                cute.copy(tiled_st_p, rP, tTR_tP)
            else:
                if cutlass.const_expr(not EARLY_PST):
                    cute.copy(tiled_st_p, rPh0, tTR_tP[(None, None, 0)])
                    cute.copy(tiled_st_p, rPh1, tTR_tP[(None, None, 1)])
        cute.arch.fence_view_async_tmem_store()
    tc_fence_before()
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(p_ready_base + wg, 0)
    if cutlass.const_expr(PROF):
        c3 = clock64()
        t_work = t_work + (c3 - c1)
        t_store = t_store + (c3 - c2)
        if is_trace:
          if j == 100:
              prof[24] = c2
              prof[25] = c3
    return m_run, l_run, t_wait, t_work, n_resc, t_load, t_store


# ----------------------------------------------------------------------------- kernel
class FlashFwd:
    def __init__(self, S: int, H: int, scale: float):
        assert S % CTA_M == 0
        self.S, self.H = S, H
        self.scale_log2 = scale * math.log2(math.e)
        self.num_kv_tiles = (S + TILE_N - 1) // TILE_N
        self.valid_last = S - (self.num_kv_tiles - 1) * TILE_N
        self.num_q_tiles = S // CTA_M

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor, prof_addr: cutlass.Int64, stream: cuda.CUstream):
        S, H = self.S, self.H
        # gmem views. Q/K/O as (S, D, H) [M/N, K, L]; V as (D, S, H) [N, K, L] (MN-major B operand)
        mQ = cute.make_tensor(q.iterator, cute.make_layout((S, D, H), stride=(H * D, 1, D)))
        mK = cute.make_tensor(k.iterator, cute.make_layout((S, D, H), stride=(H * D, 1, D)))
        mV = cute.make_tensor(v.iterator, cute.make_layout((D, S, H), stride=(1, H * D, D)))
        mO = cute.make_tensor(o.iterator, cute.make_layout((S, D, H), stride=(H * D, 1, D)))

        mma_qk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (MMA_M, TILE_N, 16), tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        op_pv = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (MMA_M, D, 16), tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.TMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
        op_l = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (MMA_M, L_N, 16), tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.TMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
        if cutlass.const_expr(P_F16):
            object.__setattr__(op_pv, "a_dtype", cutlass.Float16)   # P (A operand) in fp16, V (B operand) in bf16
            object.__setattr__(op_l, "a_dtype", cutlass.Float16)
        mma_pv = cute.make_tiled_mma(op_pv)
        mma_l = cute.make_tiled_mma(op_l)

        tiler_qk = (MMA_M, TILE_N, D)
        tiler_pv = (MMA_M, D, TILE_N)
        tiler_l = (MMA_M, L_N, TILE_N)
        cta_layout_vmnk = cute.tiled_divide(cute.make_layout((CLUSTER_M, 1, 1)), (mma_qk.thr_id,))
        q_smem_layout = sm100_utils.make_smem_layout_a(mma_qk, tiler_qk, cutlass.BFloat16, NUM_WG)
        k_smem_layout = sm100_utils.make_smem_layout_b(mma_qk, tiler_qk, cutlass.BFloat16, K_STAGES)
        v_smem_layout = sm100_utils.make_smem_layout_b(mma_pv, tiler_pv, cutlass.BFloat16, V_STAGES)
        ones_smem_layout = sm100_utils.make_smem_layout_b(mma_l, tiler_l, cutlass.BFloat16, 1)
        p_smem_layout = sm100_utils.make_smem_layout_a(mma_pv, tiler_pv, cutlass.BFloat16, 1)

        op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        tma_q, tma_tQ = cute.nvgpu.make_tiled_tma_atom_A(op, mQ, cute.select(q_smem_layout, mode=[0, 1, 2]), tiler_qk, mma_qk, cta_layout_vmnk.shape)
        tma_k, tma_tK = cute.nvgpu.make_tiled_tma_atom_B(op, mK, cute.select(k_smem_layout, mode=[0, 1, 2]), tiler_qk, mma_qk, cta_layout_vmnk.shape)
        tma_v, tma_tV = cute.nvgpu.make_tiled_tma_atom_B(op, mV, cute.select(v_smem_layout, mode=[0, 1, 2]), tiler_pv, mma_pv, cta_layout_vmnk.shape)

        self.kernel(mma_qk, mma_pv, mma_l, tma_q, tma_tQ, tma_k, tma_tK, tma_v, tma_tV, mO,
                    q_smem_layout, k_smem_layout, v_smem_layout, ones_smem_layout, p_smem_layout, cta_layout_vmnk, prof_addr).launch(
            grid=[self.num_q_tiles, H, 1], block=[THREADS, 1, 1], cluster=[CLUSTER_M, 1, 1], min_blocks_per_mp=1, stream=stream)

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(self, mma_qk: cute.TiledMma, mma_pv: cute.TiledMma, mma_l: cute.TiledMma,
               tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor,
               q_smem_layout: cute.ComposedLayout, k_smem_layout: cute.ComposedLayout,
               v_smem_layout: cute.ComposedLayout, ones_smem_layout: cute.ComposedLayout,
               p_smem_layout: cute.ComposedLayout, cta_layout_vmnk: cute.Layout, prof_addr: cutlass.Int64):
        NT = self.num_kv_tiles
        prof = cute.make_tensor(cute.make_ptr(cutlass.Int64, prof_addr, cute.AddressSpace.gmem), cute.make_layout(32))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, head, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
        mma_v = cta_coord_vmnk[0]                      # 0: leader (issues MMAs), 1: peer
        cluster_idx = bidx // CLUSTER_M                # cluster covers Q rows [512*cluster_idx, +512)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, K_STAGES * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, V_STAGES * 2]
            q_full: cutlass.Int64
            s_full: cute.struct.MemRange[cutlass.Int64, NUM_WG]
            p_ready: cute.struct.MemRange[cutlass.Int64, NUM_WG]
            pv_done: cute.struct.MemRange[cutlass.Int64, 2 * NUM_WG]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=q_smem_layout.outer, byte_alignment=1024, swizzle=q_smem_layout.inner)
        sK = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=k_smem_layout.outer, byte_alignment=1024, swizzle=k_smem_layout.inner)
        sV = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=v_smem_layout.outer, byte_alignment=1024, swizzle=v_smem_layout.inner)
        sOnes = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=ones_smem_layout.outer, byte_alignment=1024, swizzle=ones_smem_layout.inner)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)

        # pipelines: TMA producer (1 thread) -> UMMA consumer (commit)
        k_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=K_STAGES, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), tx_count=(0 if NO_TMA_KV else K_BYTES),
            barrier_storage=storage.k_mbar.data_ptr(), cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        v_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=V_STAGES, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), tx_count=(0 if NO_TMA_KV else V_BYTES),
            barrier_storage=storage.v_mbar.data_ptr(), cta_layout_vmnk=cta_layout_vmnk, defer_sync=True)
        k_full_base = storage.k_mbar.data_ptr()
        k_empty_base = storage.k_mbar.data_ptr() + K_STAGES
        v_full_base = storage.v_mbar.data_ptr()
        v_empty_base = storage.v_mbar.data_ptr() + V_STAGES
        q_full = storage.q_full
        s_full_base = storage.s_full.data_ptr()
        p_ready_base = storage.p_ready.data_ptr()
        pv_done_base = storage.pv_done.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_full, 1)
                for w in cutlass.range_constexpr(NUM_WG):
                    cute.arch.mbarrier_init(s_full_base + w, 1)
                    cute.arch.mbarrier_init(p_ready_base + w, 4 * CLUSTER_M)
                    cute.arch.mbarrier_init(pv_done_base + 2 * w, 1)
                    cute.arch.mbarrier_init(pv_done_base + 2 * w + 1, 1)
        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ones tile for the row-sum MMA (every element = bf16 1.0)
        ones_words = cute.size_in_bytes(cutlass.BFloat16, ones_smem_layout) // 4
        sOnes_i32 = cute.make_tensor(cute.recast_ptr(sOnes.iterator, dtype=cutlass.Int32), cute.make_layout(ones_words))
        for i in cutlass.range(tidx, ones_words, THREADS):
            sOnes_i32[i] = cutlass.Int32(0x3F803F80)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # TMEM
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier,
                                   is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar)
        tmem.allocate(NUM_TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

        # ---- fragments / partitions shared by roles
        thr_qk = mma_qk.get_slice(mma_v)
        thr_pv = mma_pv.get_slice(mma_v)
        mask_a = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2)
        mask_b = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode=1)
        is_leader = mma_v == 0
        gQ = cute.local_tile(mQ, cute.slice_(tiler_qk_c(), (None, 0, None)), (None, None, None))     # (bM, bK, RestM, RestK, RestL)
        gK = cute.local_tile(mK, cute.slice_(tiler_qk_c(), (0, None, None)), (None, None, None))     # (bN, bK, RestN, RestK, RestL)
        gV = cute.local_tile(mV, cute.slice_(tiler_pv_c(), (0, None, None)), (None, None, None))     # (bD, bN, RestD, RestN, RestL)
        gO = cute.local_tile(mO, (TILE_M, D), (None, None, None))                                  # (128, bD, RestM, RestD, RestL)
        tCgQ = thr_qk.partition_A(gQ)
        tCgK = thr_qk.partition_B(gK)
        tCgV = thr_pv.partition_B(gV)
        tQsQ, tQgQ = cpasync.tma_partition(tma_q, cta_coord_vmnk[2], cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])), cute.group_modes(sQ, 0, 3), cute.group_modes(tCgQ, 0, 3))
        tKsK, tKgK = cpasync.tma_partition(tma_k, cta_coord_vmnk[1], cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])), cute.group_modes(sK, 0, 3), cute.group_modes(tCgK, 0, 3))
        tVsV, tVgV = cpasync.tma_partition(tma_v, cta_coord_vmnk[1], cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])), cute.group_modes(sV, 0, 3), cute.group_modes(tCgV, 0, 3))

        tQ = mma_qk.make_fragment_A(sQ)          # (MMA, MMA_M, MMA_K, NUM_WG)
        tK = mma_qk.make_fragment_B(sK)          # (MMA, MMA_N, MMA_K, K_STAGES)
        tV = mma_pv.make_fragment_B(sV)          # (MMA, MMA_N, MMA_K, V_STAGES)
        tOnes = mma_l.make_fragment_B(sOnes)     # (MMA, MMA_N, MMA_K, 1)
        s_acc_layout = mma_qk.make_fragment_C(mma_qk.partition_shape_C((MMA_M, TILE_N))).layout
        o_acc_layout = mma_pv.make_fragment_C(mma_pv.partition_shape_C((MMA_M, D))).layout
        l_acc_layout = mma_l.make_fragment_C(mma_l.partition_shape_C((MMA_M, L_N))).layout
        p_frag_layout_pv = mma_pv.make_fragment_A(cute.select(p_smem_layout, mode=[0, 1, 2]).outer).layout
        p_frag_layout_l = mma_l.make_fragment_A(cute.select(p_smem_layout, mode=[0, 1, 2]).outer).layout
        tS0 = cute.make_tensor(tmem_ptr + S_COL0, s_acc_layout)
        tS1 = cute.make_tensor(tmem_ptr + (S_COL0 + WG_COL_STRIDE_S), s_acc_layout)
        tO0 = cute.make_tensor(tmem_ptr + O_COL0, o_acc_layout)
        tO1 = cute.make_tensor(tmem_ptr + (O_COL0 + WG_COL_STRIDE_O), o_acc_layout)
        tL0 = cute.make_tensor(tmem_ptr + L_COL0, l_acc_layout)
        tL1 = cute.make_tensor(tmem_ptr + (L_COL0 + WG_COL_STRIDE_S), l_acc_layout)
        p_dtype = P_DTYPE
        tPpv0 = cute.make_tensor(cute.recast_ptr(tmem_ptr + S_COL0, dtype=p_dtype), p_frag_layout_pv)
        tPpv1 = cute.make_tensor(cute.recast_ptr(tmem_ptr + (S_COL0 + WG_COL_STRIDE_S), dtype=p_dtype), p_frag_layout_pv)
        tPl0 = cute.make_tensor(cute.recast_ptr(tmem_ptr + S_COL0, dtype=p_dtype), p_frag_layout_l)
        tPl1 = cute.make_tensor(cute.recast_ptr(tmem_ptr + (S_COL0 + WG_COL_STRIDE_S), dtype=p_dtype), p_frag_layout_l)

        # register reallocation (warpgroup-collective): producer/MMA warpgroup shrinks, softmax warpgroups grow.
        # Role code is nested inside the branches so that ptxas sees the register budget dominating each role.
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(PRODUCER_REGS)
            # ================================================================== warp 0: TMA producer
            if warp_idx == 0:
                k_prod = k_pipe.make_producer()
                v_prod = v_pipe.make_producer()
                if is_leader:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(q_full, Q_BYTES * CLUSTER_M)
                for w in cutlass.range_constexpr(NUM_WG):
                    cute.copy(tma_q, tQgQ[(None, cluster_idx * NUM_WG + w, 0, head)], tQsQ[(None, w)], tma_bar_ptr=q_full, mcast_mask=mask_a)
                for j in cutlass.range(NT):
                    kh = k_prod.acquire_and_advance()
                    if cutlass.const_expr(not NO_TMA_KV):
                        cute.copy(tma_k, tKgK[(None, j, 0, head)], tKsK[(None, kh.index)], tma_bar_ptr=kh.barrier, mcast_mask=mask_b)
                    vh = v_prod.acquire_and_advance()
                    if cutlass.const_expr(not NO_TMA_KV):
                        cute.copy(tma_v, tVgV[(None, 0, j, head)], tVsV[(None, vh.index)], tma_bar_ptr=vh.barrier, mcast_mask=mask_b)
                k_prod.tail()
                v_prod.tail()

            # ================================================================== warp 1: MMA issuer
            if warp_idx == 1 and is_leader and cutlass.const_expr(ASM_MMA):
                offs = (desc_offsets(sQ, NUM_KB_QK, NUM_WG), desc_offsets(sK, NUM_KB_QK, K_STAGES),
                        desc_offsets(sV, NUM_KB_PV, V_STAGES), desc_offsets(sOnes, NUM_KB_PV, 1))
                q_d = tcgen05.smem_descriptor_to_int(tQ[(None, None, 0, 0)].iterator)
                k_d = tcgen05.smem_descriptor_to_int(tK[(None, None, 0, 0)].iterator)
                v_d = tcgen05.smem_descriptor_to_int(tV[(None, None, 0, 0)].iterator)
                o_d = tcgen05.smem_descriptor_to_int(tOnes[(None, None, 0, 0)].iterator)
                q_lo, q_hi = cutlass.Int32(q_d & 0xFFFFFFFF), cutlass.Int32(q_d >> 32)
                k_lo, k_hi = cutlass.Int32(k_d & 0xFFFFFFFF), cutlass.Int32(k_d >> 32)
                v_lo, v_hi = cutlass.Int32(v_d & 0xFFFFFFFF), cutlass.Int32(v_d >> 32)
                o_lo, o_hi = cutlass.Int32(o_d & 0xFFFFFFFF), cutlass.Int32(o_d >> 32)
                tmem_base = cutlass.Int32(tmem_ptr.toint())
                cute.arch.mbarrier_wait(q_full, 0)
                tc_fence_after()
                n_pairs = NT // 2
                t_kv = cutlass.Int64(0)
                t_p = cutlass.Int64(0)
                t_start = clock64()
                for jj in cutlass.range(n_pairs):
                    j = jj * 2
                    t_kv, t_p = mma_tile_asm(j, 0, offs, tmem_base, q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, o_lo, o_hi,
                                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                    t_kv, t_p = mma_tile_asm(j + 1, 1, offs, tmem_base, q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, o_lo, o_hi,
                                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                if cutlass.const_expr(NT % 2 == 1):
                    t_kv, t_p = mma_tile_asm(cutlass.Int32(NT - 1), (NT - 1) % 2, offs, tmem_base, q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, o_lo, o_hi,
                                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                jp = NT - 1
                cute.arch.mbarrier_wait(p_ready_base + 1, jp & 1)
                tc_fence_after()
                pv1_tail_asm(cutlass.Int32(jp), jp % V_STAGES, offs, tmem_base, v_lo, v_hi, o_lo, o_hi)
                with cute.arch.elect_one():
                    tcgen05.commit(pv_done_base + 2 + ((NT - 1) % 2), 0b11, tcgen05.CtaGroup.TWO)
                    tcgen05.commit(v_empty_base + (jp % V_STAGES), 0b11, tcgen05.CtaGroup.TWO)
                if cutlass.const_expr(PROF):
                    t_total = clock64() - t_start
                    with cute.arch.elect_one():
                        cute.arch.atomic_add(prof.iterator + 0, t_total)
                        cute.arch.atomic_add(prof.iterator + 1, t_kv)
                        cute.arch.atomic_add(prof.iterator + 2, t_p)

            if warp_idx == 1 and is_leader and cutlass.const_expr(not ASM_MMA):
                mma_qk_acc = mma_qk.with_()
                mma_qk_acc.set(tcgen05.Field.ACCUMULATE, True)
                mma_pv_acc = mma_pv.with_()
                mma_pv_acc.set(tcgen05.Field.ACCUMULATE, True)
                mma_l_acc = mma_l.with_()
                mma_l_acc.set(tcgen05.Field.ACCUMULATE, True)
                cute.arch.mbarrier_wait(q_full, 0)
                tc_fence_after()
                n_pairs = NT // 2
                t_kv = cutlass.Int64(0)
                t_p = cutlass.Int64(0)
                t_start = clock64()
                for jj in cutlass.range(n_pairs):
                    j = jj * 2
                    t_kv, t_p = mma_tile(j, 0, mma_qk, mma_qk_acc, mma_pv, mma_pv_acc, mma_l, mma_l_acc,
                             tQ, tK, tV, tOnes, tS0, tS1, tO0, tO1, tL0, tL1, tPpv0, tPpv1, tPl0, tPl1,
                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                    t_kv, t_p = mma_tile(j + 1, 1, mma_qk, mma_qk_acc, mma_pv, mma_pv_acc, mma_l, mma_l_acc,
                             tQ, tK, tV, tOnes, tS0, tS1, tO0, tO1, tL0, tL1, tPpv0, tPpv1, tPl0, tPl1,
                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                if cutlass.const_expr(NT % 2 == 1):
                    t_kv, t_p = mma_tile(cutlass.Int32(NT - 1), (NT - 1) % 2, mma_qk, mma_qk_acc, mma_pv, mma_pv_acc, mma_l, mma_l_acc,
                             tQ, tK, tV, tOnes, tS0, tS1, tO0, tO1, tL0, tL1, tPpv0, tPpv1, tPl0, tPl1,
                             k_full_base, k_empty_base, v_full_base, v_empty_base, s_full_base, p_ready_base, pv_done_base, t_kv, t_p)
                # tail: PV1_{NT-1}
                jp = NT - 1
                cute.arch.mbarrier_wait(p_ready_base + 1, jp & 1)
                tc_fence_after()
                issue_pv(mma_pv, mma_pv_acc, mma_l, mma_l_acc, tO1, tL1, tPpv1, tPl1, tV, tOnes, jp % V_STAGES, cutlass.Boolean(jp > 0))
                with cute.arch.elect_one():
                    tcgen05.commit(pv_done_base + 2 + ((NT - 1) % 2), 0b11, tcgen05.CtaGroup.TWO)
                    tcgen05.commit(v_empty_base + (jp % V_STAGES), 0b11, tcgen05.CtaGroup.TWO)
                if cutlass.const_expr(PROF):
                    t_total = clock64() - t_start
                    with cute.arch.elect_one():
                        cute.arch.atomic_add(prof.iterator + 0, t_total)
                        cute.arch.atomic_add(prof.iterator + 1, t_kv)
                        cute.arch.atomic_add(prof.iterator + 2, t_p)
        else:
            cute.arch.setmaxregister_increase(SOFTMAX_REGS)
            # ================================================================== warps 4..11: softmax warpgroups
            if cutlass.const_expr(True):
                wg = (warp_idx - 4) // 4
                tidx_wg = tidx - 128 * (wg + 1)
                tS = cute.make_tensor(tmem_ptr + (S_COL0 + wg * WG_COL_STRIDE_S), s_acc_layout)[((None, None), 0, 0)]   # (128, TILE_N) fp32
                tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + (S_COL0 + wg * WG_COL_STRIDE_S), dtype=cutlass.Int32),
                                      cute.make_layout((TILE_M, TILE_N // 2), stride=(65536, 1)))   # (128, 56) packed bf16x2
                tO = cute.make_tensor(tmem_ptr + (O_COL0 + wg * WG_COL_STRIDE_O), o_acc_layout)[((None, None), 0, 0)]   # (128, 128) fp32
                tL = cute.make_tensor(tmem_ptr + (L_COL0 + wg * WG_COL_STRIDE_S), l_acc_layout)[((None, None), 0, 0)]   # (128, 8) fp32

                ld_s_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32 if LD_REP == 32 else tcgen05.Repetition.x16), cutlass.Float32)
                tiled_ld_s = tcgen05.make_tmem_copy(ld_s_atom, tS)
                thr_ld_s = tiled_ld_s.get_slice(tidx_wg)
                tTR_tS = thr_ld_s.partition_S(tS)                                           # (V, 1, 7) tmem
                tTR_cS = thr_ld_s.partition_D(cute.make_identity_tensor(tS.shape))          # (V', 1, 4)
                rS = cute.make_rmem_tensor(tTR_cS.shape, cutlass.Float32)
                rSc0 = cute.make_rmem_tensor(tTR_cS[(None, None, 0)].shape, cutlass.Float32)
                rSc1 = cute.make_rmem_tensor(tTR_cS[(None, None, 0)].shape, cutlass.Float32)
                rSc2 = cute.make_rmem_tensor(tTR_cS[(None, None, 0)].shape, cutlass.Float32)
                rSc3 = cute.make_rmem_tensor(tTR_cS[(None, None, 0)].shape, cutlass.Float32)

                st_p_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Int32)
                tiled_st_p = tcgen05.make_tmem_copy(st_p_atom, tP)
                thr_st_p = tiled_st_p.get_slice(tidx_wg)
                tTR_tP = thr_st_p.partition_D(tP)                                           # (V, 1, 7) tmem
                tTR_cP = thr_st_p.partition_S(cute.make_identity_tensor(tP.shape))          # (V', 1, 2)
                rP = cute.make_rmem_tensor(tTR_cP.shape, cutlass.Int32)
                rPh0 = cute.make_rmem_tensor(tTR_cP[(None, None, 0)].shape, cutlass.Int32)
                rPh1 = cute.make_rmem_tensor(tTR_cP[(None, None, 0)].shape, cutlass.Int32)

                ld_o_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
                tiled_ld_o = tcgen05.make_tmem_copy(ld_o_atom, tO)
                thr_ld_o = tiled_ld_o.get_slice(tidx_wg)
                tTR_tO = thr_ld_o.partition_S(tO)                                           # (V, 1, 4) tmem
                tTR_cO = thr_ld_o.partition_D(cute.make_identity_tensor(tO.shape))          # (V', 1, 4)
                rO = cute.make_rmem_tensor(tTR_cO[(None, None, 0)].shape, cutlass.Float32)
                st_o_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
                tiled_st_o = tcgen05.make_tmem_copy(st_o_atom, tO)
                thr_st_o = tiled_st_o.get_slice(tidx_wg)
                tRT_tO = thr_st_o.partition_D(tO)                                           # (V, 1, 4) tmem

                ld_l_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
                tiled_ld_l = tcgen05.make_tmem_copy(ld_l_atom, tL)
                thr_ld_l = tiled_ld_l.get_slice(tidx_wg)
                tTR_tL = thr_ld_l.partition_S(tL)
                rL = cute.make_rmem_tensor(thr_ld_l.partition_D(cute.make_identity_tensor(tL.shape)).shape, cutlass.Float32)
                st_l_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
                tiled_st_l = tcgen05.make_tmem_copy(st_l_atom, tL)
                thr_st_l = tiled_st_l.get_slice(tidx_wg)
                tRT_tL = thr_st_l.partition_D(tL)

                scale_log2 = cutlass.Float32(self.scale_log2)
                thresh_raw = cutlass.Float32(RESCALE_THRESH_LOG2 / self.scale_log2)
                is_trace = (bidx * 4096 + head * 256 + (warp_idx - 4) * 32 + cute.arch.lane_idx()) == 0
                m_run = cutlass.Float32(NEG_INF)
                l_run = cutlass.Float32(0.0)
                t_wait = cutlass.Int64(0)
                t_work = cutlass.Int64(0)
                n_resc = cutlass.Int64(0)
                t_load = cutlass.Int64(0)
                t_store = cutlass.Int64(0)
                args = (wg, s_full_base, p_ready_base, pv_done_base, prof, is_trace,
                        tiled_ld_s, tTR_tS, rS, rSc0, rSc1, rSc2, rSc3, tiled_st_p, rP, rPh0, rPh1, tTR_tP,
                        tiled_ld_o, tTR_tO, rO, tiled_st_o, tRT_tO, scale_log2, thresh_raw)
                m_run, l_run, t_wait, t_work, n_resc, t_load, t_store = softmax_tile(
                    cutlass.Int32(0), True, False, self.valid_last, m_run, l_run, *args, t_wait, t_work, n_resc, t_load, t_store)
                for j in cutlass.range(1, NT - 1):
                    m_run, l_run, t_wait, t_work, n_resc, t_load, t_store = softmax_tile(
                        j, False, False, self.valid_last, m_run, l_run, *args, t_wait, t_work, n_resc, t_load, t_store)
                m_run, l_run, t_wait, t_work, n_resc, t_load, t_store = softmax_tile(
                    cutlass.Int32(NT - 1), False, self.valid_last < TILE_N, self.valid_last, m_run, l_run, *args, t_wait, t_work, n_resc, t_load, t_store)
                if cutlass.const_expr(PROF):
                    if cute.arch.lane_idx() == 0:
                        cute.arch.atomic_add(prof.iterator + 16 + warp_idx, t_work)
                    if tidx_wg == 0:
                        cute.arch.atomic_add(prof.iterator + 4 + wg * 4, t_wait)
                        cute.arch.atomic_add(prof.iterator + 5 + wg * 4, t_work)
                        cute.arch.atomic_add(prof.iterator + 6 + wg * 4, n_resc)
                        cute.arch.atomic_add(prof.iterator + 12 + wg * 2, t_load)
                        cute.arch.atomic_add(prof.iterator + 13 + wg * 2, t_store)

                # ---- epilogue: O / l -> bf16 -> gmem
                cute.arch.mbarrier_wait(pv_done_base + 2 * wg + ((NT - 1) & 1), ((NT - 1) >> 1) & 1)
                tc_fence_after()
                inv_l = cutlass.Float32(1.0) / l_run
                gO_tile = gO[(None, None, cluster_idx * (NUM_WG * CLUSTER_M) + wg * CLUSTER_M + mma_v, 0, head)]   # (128,128) bf16 gmem
                tTR_gO = thr_ld_o.partition_D(gO_tile)                           # (V', 1, 4) gmem
                rO16 = cute.make_rmem_tensor(tTR_gO[(None, None, 0)].shape, cutlass.BFloat16)
                for ci in cutlass.range_constexpr(D // 32):
                    cute.copy(tiled_ld_o, tTR_tO[(None, None, ci)], rO)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(32):
                        rO[e] = rO[e] * inv_l
                    rO16.store(rO.load().to(cutlass.BFloat16))
                    cute.autovec_copy(rO16, tTR_gO[(None, None, ci)])

        # ================================================================== teardown
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


def tiler_qk_c():
    return (MMA_M, TILE_N, D)


def tiler_pv_c():
    return (MMA_M, D, TILE_N)


# ----------------------------------------------------------------------------- python wrapper
_compiled = {}
prof_buf = None


def attention_fwd(q, k, v, scale=None):
    """q,k,v: (1, S, H, D) bf16 cuda tensors, contiguous. returns (1, S, H, D) bf16."""
    B, S, H, Dd = q.shape
    assert B == 1 and Dd == D and k.shape == q.shape and v.shape == q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    key = (S, H, scale)
    tq, tk, tv, to = (from_dlpack(t, assumed_align=16) for t in (q, k, v, o))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    global prof_buf
    if prof_buf is None:
        prof_buf = torch.zeros(32, dtype=torch.int64, device="cuda")
    prof_addr = cutlass.Int64(prof_buf.data_ptr())
    if key not in _compiled:
        import time as _t
        t0 = _t.time()
        fa = FlashFwd(S, H, scale)
        _compiled[key] = cute.compile(fa, tq, tk, tv, to, prof_addr, stream)
        print(f"[fa_fwd] compiled S={S} H={H} in {_t.time()-t0:.1f}s", flush=True)
    _compiled[key](tq, tk, tv, to, prof_addr, stream)
    return o
