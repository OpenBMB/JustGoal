"""Preprocess kernels for the CuTe backward: delta = rowsum(dO*O) and lse*log2(e).

The packed-layout variant also *repacks* the two per-row scalars so that every sequence
starts on a multiple of BM tokens and the tail of each sequence is filled with a huge lse.
That buys two things in the main kernel:

* TMA needs the global base address of every box to be 16-byte aligned, and lse/delta are
  TMA'd along the token axis.  A sequence starting at an arbitrary token would break that.
* A query row past the end of its sequence then reads lse = +huge, so exp2 gives exactly
  zero with no per-element predicate -- the per-element query-limit mask was costing the
  packed path 1.6x against the dense one.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
HUGE = 1.0e30
_L2E = tl.constexpr(LOG2E)
_HUGE = tl.constexpr(HUGE)


@triton.jit
def _bshd_pre(O, DO, LSE, DELTA, LSE2, DQ, R, N, H: tl.constexpr,
              D: tl.constexpr, BLK: tl.constexpr, ZDQ: tl.constexpr):
    """Dense (batch, seq, heads, dim).  delta/lse2 come out as [B, H, N].

    Also zeroes dQ when ZDQ, because the main kernel reduces into it.  Fusing that into
    this pass rather than calling dq.zero_() separately turns two memory-bound kernels into
    one: this one already streams O and dO (2 x 134 MB on mha) and the zero-fill is 134 MB
    of pure writes, which the same pass can retire alongside the reads.
    """
    pid = tl.program_id(0)
    r = pid * BLK + tl.arange(0, BLK)
    keep = r < R
    p = r[:, None] * D + tl.arange(0, D)[None, :]
    o = tl.load(O + p, mask=keep[:, None], other=0.0).to(tl.float32)
    d = tl.load(DO + p, mask=keep[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * d, axis=1)
    if ZDQ:
        tl.store(DQ + p, tl.zeros((BLK, D), dtype=DQ.dtype.element_ty), mask=keep[:, None])

    nh = N * H
    b = r // nh
    rem = r % nh
    n = rem // H
    h = rem % H
    idx = b * nh + h * N + n
    tl.store(DELTA + idx, delta, mask=keep)
    tl.store(LSE2 + idx, tl.load(LSE + idx, mask=keep, other=0.0) * _L2E, mask=keep)


@triton.jit
def _cu_pad(CU, CUP, BLK, NSEQ, NBLK, BM: tl.constexpr, BN: tl.constexpr,
            ORD: tl.constexpr):
    """Serial prefix sum of round_up(len_s, BM); one program, NSEQ is small.

    Also builds the packed layout's **key-block map**.  The main kernel's grid used to be
    ceil(max_seqlen / BN) key blocks for *every* sequence -- 13 312 CTAs for the 4 160 that
    have work on the contest's varlen case -- and SMEM allows only one CTA per SM, so each
    empty CTA occupies a whole SM for a launch and an exit.  Comparing occupied SM-cycles
    with the wall clock prices that at 3.8 % of the workload (mha, which has no empty CTAs,
    accounts for 99.7 % of its kernel time in CTA lifetimes; varlen for 96.2 %).

    BLK[i] = s * 65536 + j for the i-th (sequence, key block) pair that has work, so the
    grid can be one flat axis of ceil(T / BN) + NSEQ entries instead.  Unused entries get a
    key block far past the end of sequence 0, which makes the kernel's own `total > 0` test
    reject them without a second predicate.
    """
    acc = 0
    tl.store(CUP, 0)
    for s in range(NSEQ):
        lo = tl.load(CU + s)
        hi = tl.load(CU + s + 1)
        acc += ((hi - lo + BM - 1) // BM) * BM
        tl.store(CUP + s + 1, acc)

    # ORD: emit the map longest sequence first.  The CTAs of one sequence are contiguous in
    # the grid, so the launch order is the schedule, and the longest sequence owns the
    # longest CTA (its first key block runs ceil(len/BM) tile steps).  Starting it last puts
    # that CTA on the critical path; dev/p3acct.py's greedy simulation measures the
    # difference at 1.5 % of varlen.  Ranked by length descending, ties by index, with the
    # ranks parked past the end of the map.  NSEQ is small, so O(NSEQ^2) scalar work in one
    # program costs nothing.
    if ORD:
        for s in range(NSEQ):
            n = tl.load(CU + s + 1) - tl.load(CU + s)
            r = 0
            for s2 in range(NSEQ):
                n2 = tl.load(CU + s2 + 1) - tl.load(CU + s2)
                r += tl.where((n2 > n) | ((n2 == n) & (s2 < s)), 1, 0)
            tl.store(BLK + NBLK + s, r)

    nb = 0
    for it in range(NSEQ):
        sel = it
        if ORD:
            sel = 0
            for s in range(NSEQ):
                sel = tl.where(tl.load(BLK + NBLK + s) == it, s, sel)
        lo = tl.load(CU + sel)
        hi = tl.load(CU + sel + 1)
        nkv = (hi - lo + BN - 1) // BN
        for j in range(nkv):
            tl.store(BLK + nb + j, sel * 65536 + j)
        nb += nkv
    for i in range(nb, NBLK):
        tl.store(BLK + i, 65535)


@triton.jit
def _fix_tail(DK, DV, CU, T, TB, H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    """Copy a sequence's redirected tail block back over its own last rows.

    D22 sends the last key block of each packed sequence to scratch rows `[TB + s*BN, ...)`
    rather than to the sequence's own rows, because only `qlen - (nkv-1)*BN` of its BN rows
    belong to the sequence.  This copies that valid prefix into place, so every row of
    `[0, T)` ends up written exactly once and neither dkp nor dvp needs zeroing.
    """
    r = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    lo = tl.load(CU + s)
    hi = tl.load(CU + s + 1)
    n = hi - lo
    nkv = (n + BN - 1) // BN
    nvalid = n - (nkv - 1) * BN
    # A sequence whose length is a multiple of BN has no out-of-range tail row, so its last
    # block stored in place and its scratch rows were never written: copying them back would
    # clobber correct data with whatever the buffer held.  (Measured as 82 of 67 M elements
    # wrong, because the clobbered rows are a sequence's *last* key rows, which few queries
    # attend to and whose dK is therefore small enough to pass atol by accident.)
    if (n % BN != 0) and (r < nvalid):
        src = (TB + s * BN + r) * (H * D) + h * D + tl.arange(0, D)
        dst = (lo + (nkv - 1) * BN + r) * (H * D) + h * D + tl.arange(0, D)
        tl.store(DK + dst, tl.load(DK + src))
        tl.store(DV + dst, tl.load(DV + src))


@triton.jit
def _varlen_pre(O, DO, LSE, DELTA, LSE2, CU, CUP, DQ, T, TPAD, H: tl.constexpr,
                D: tl.constexpr, BM: tl.constexpr, BLK: tl.constexpr,
                ZDQ: tl.constexpr):
    """Packed (tokens, heads, dim).  delta/lse2 come out as [H, TPAD], sequence s based at
    CUP[s], with the [len_s, round_up(len_s, BM)) tail filled with a huge lse."""
    blk = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    lo = tl.load(CU + s)
    hi = tl.load(CU + s + 1)
    n = hi - lo
    npad = ((n + BM - 1) // BM) * BM
    pb = tl.load(CUP + s)

    t = blk * BLK + tl.arange(0, BLK)
    inpad = t < npad
    valid = t < n
    p = (lo + t)[:, None] * (H * D) + h * D + tl.arange(0, D)[None, :]
    o = tl.load(O + p, mask=valid[:, None], other=0.0).to(tl.float32)
    d = tl.load(DO + p, mask=valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * d, axis=1)
    l2 = tl.load(LSE + h * T + lo + t, mask=valid, other=0.0) * _L2E

    dst = h * TPAD + pb + t
    tl.store(DELTA + dst, tl.where(valid, delta, 0.0), mask=inpad)
    tl.store(LSE2 + dst, tl.where(valid, l2, _HUGE), mask=inpad)
    if ZDQ:
        # dQ is reduced into, so it has to start at zero, and the kernel's dQ boxes cover
        # exactly [lo, lo + round_up(n, BM)) -- this program's `inpad` range.  Folding the
        # fill in here rather than calling dqp.zero_() lets 134 MB of writes ride along with
        # the O / dO reads this pass already does, as ZFUSE does on the dense path.  Where
        # one sequence's padded range laps into the next sequence's rows both passes write
        # zero, so the overlap is benign.
        dq = (lo + t)[:, None] * (H * D) + h * D + tl.arange(0, D)[None, :]
        tl.store(DQ + dq, tl.zeros((BLK, D), dtype=DQ.dtype.element_ty), mask=inpad[:, None])


def pre_bshd(out, dout, lse, delta, lse2, B, N, H, D, blk=256, dq=None):
    R = B * N * H
    _bshd_pre[(triton.cdiv(R, blk),)](out, dout, lse, delta, lse2,
                                      dq if dq is not None else out, R, N,
                                      H=H, D=D, BLK=blk, ZDQ=dq is not None)


def pre_varlen(out, dout, lse, delta, lse2, cu, cup, T, Tpad, nseq, max_seqlen, H, D, BM,
           blk=256, bmap=None, nblk=0, BN=128, dq=None, vord=0):
    _cu_pad[(1,)](cu, cup, bmap if bmap is not None else cup, nseq, nblk, BM=BM, BN=BN,
                  ORD=vord)
    _varlen_pre[(triton.cdiv(max_seqlen, blk) + BM // blk, nseq, H)](
        out, dout, lse, delta, lse2, cu, cup, dq if dq is not None else out, T, Tpad,
        H=H, D=D, BM=BM, BLK=blk, ZDQ=dq is not None)


def fix_tail(dkp, dvp, cu, T, nseq, H, D, BN):
    _fix_tail[(BN, nseq, H)](dkp, dvp, cu, T, T + BN, H=H, D=D, BN=BN)


"""D17: the A operands this kernel produces itself (`P^T`, `dS^T`) live in TMEM, not SMEM.

`dev/phase.py`'s CTA-lifetime probe prices the three roles against the 2650-cycle step
period on mha: the MMA warp is busy 2140 cycles (81%), the softmax 1081 (41%) and the dQ
epilogue 832 (31%).  The MMA warp is the only saturated role, and its 2140 cycles sit 1.27x
above the 1690-cycle floor that 212 992 B of SMEM operand reads per step impose at
126 B/cycle/SM.  So the operand bytes are the binding resource.

Two of the five products take an A operand that is produced inside the kernel:

    dV += P^T  dO      A = P^T  (BN x BM) bf16, K-major
    dK += dS^T Q       A = dS^T (BN x BM) bf16, K-major

`tcgen05.mma` can read A from TMEM (`OperandSource.TMEM`), whose one restriction -- A must
be K-major -- both already satisfy, and a (128, 64) bf16 A operand occupies **32 TMEM
columns**.  Those columns are free: they are the first half of the `S^T` accumulator the
softmax has just finished reading.  So this removes 32 768 B of the 212 992 (-15.4%) without
touching the `2*(2*BM) + 2*HD <= 512` TMEM bound that closed D11, D15 and D16.

    stage region, 128 columns, one step
    [0,64)   S^T  (fp32 acc)   --softmax reads-->   [0,32) P^T | [32,64) dS^T  (bf16 A)
    [64,128) dP^T (fp32 acc)   --softmax reads-->   (dead)
    ... dV reads P^T, dK reads dS^T, then dQ overwrites all 128 columns

The softmax's two warpgroups split the *query* columns, and a warpgroup's `P^T` write lands
on TMEM columns holding `S^T` query columns owned by the *other* warpgroup, so one named
barrier separates the `tcgen05.ld` phase from the `tcgen05.st` phase.  `dS^T` must also stay
in SMEM because `dQ` reads it MN-major, which TMEM A cannot do -- hence `TA_DS` is a separate
knob from `TA_PT`: it adds a store to remove a read.

Mechanism validated standalone first in `dev/probe_tmema.py` (max_abs 5.7e-06).

D12's docstring follows.

D12: dQ reduced with the TMA unit (`cp.reduce.async.bulk.tensor`), not with per-thread
atomics.  Otherwise identical to dev/cute_ws.py (0.810x), whose docstring follows.

The in-kernel accounting of that kernel makes the dQ epilogue the binding role:

    mma:loop_total 2628      sm:loop_total 2728      dq:loop_total 3005
                                                       wait_dq_full 1554
                                                       epilogue     1451

and D9.6 attributed 1317 of those 1451 cycles to the 16-byte global atomics themselves
(DQX=1 drops them and runs mha at 1.515 ms against 1.955).  The atomic *volume* is fixed at
N^2 D / (2 BN) and BN is already at the CtaGroup.ONE cap of 128, so the only way down is to
stop issuing the reduction from the warps.  `cpasync.CopyReduceBulkTensorTileS2GOp` hands
the whole BM x HD tile to the TMA unit as one bulk reduce-add: 128 threads do a coalesced
SMEM store and one thread issues a single descriptor-driven operation, which the TMA engine
then drains asynchronously while the warp returns to the next tile step.

This is not the SMEM round trip that D9.6 rejected -- that one wrote SMEM only to *re-read*
it into registers for wider atomics (2160 cycles against 1332).  Here nothing reads it back.

SMEM pays for the staging tile: DQS x BM x HD x 2 B = 32 KB at BM=64, found by dropping the
Q/dO pipeline from NSTAGE=3 to 2 (which P3.0 measured as a wash on mha).  Buffer reuse needs
no mbarrier -- the dQ warpgroup is the only writer, so a thread-local
`cp_async_bulk_wait_group(DQS-1, read=True)` is the whole handshake.
"""

_WS_DOCSTRING = """D9.2: warp-specialised fused causal attention backward for B200 (CuTe DSL / tcgen05).

Same math as dev/cute_bwd.py (see its docstring); the difference is the schedule.  The
non-specialised version runs at 25.8% tensor-pipe activity with one active warp per
scheduler and 42.7% long_scoreboard, because within one (q, kv) tile step everything is
serial: MMA(S^T,dP^T) -> softmax -> MMA(dV,dK,dQ) -> dQ epilogue.

Here three roles run concurrently:

  warp  8        TMA + MMA: issues the staged Q/dO loads and all five MMAs
  warps 0-7      two softmax warpgroups (one per column half of S^T) plus the dQ epilogue

and the MMA warp is software-pipelined one tile step ahead:

    ... MMA13(it+1) ; wait softmax(it) ; MMA245(it) ; MMA13(it+2) ; ...

so softmax(it) overlaps the tensor-core execution of MMA13(it+1), and the dQ epilogue of
step it-1 overlaps MMA245(it-1).  TMEM holds two (S^T, dP^T) stages for the ping-pong:

    [0,128)  dK          loop-carried
    [128,256) dV         loop-carried
    [256,384) stage 0 : S^T 64 | dP^T 64, then dQ (128) once the softmax has read them
    [384,512) stage 1 : same

The softmax warpgroups must therefore run dq_epilogue(it-1) *before* softmax(it): the MMA
warp needs stage (it+1)%2 released before it can issue MMA13(it+1), and that release is the
dQ read-out of step it-1.
"""

import math
import os
import sys

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode as MM
import cutlass.utils.blackwell_helpers as bh
from cutlass.cute.experimental.utils import make_tmem_layout_a
from cutlass.cute.runtime import from_dlpack

IO = cutlass.BFloat16
ACC = cutlass.Float32
HD = 128
BM = 64            # query tile
BN = 128           # key/value tile
NWG = 2            # softmax warpgroups
SUB = BM // NWG    # softmax column subtile = one warpgroup's share
NSMX = 128 * NWG   # softmax threads
DQ_WG = int(os.environ.get("DQ_WG", "1"))      # warpgroups dedicated to the dQ epilogue
# GQA query-head group split across CTAs: without it a GQA CTA owns a key head and loops
# all 8 query heads, leaving a grid of 512 CTAs with a 32:1 spread in per-CTA work.  4 is
# still the knee after D9.6, re-measured on gqa as medians of three interleaved runs
# (single runs on this harness carry +-3%, enough to invert the gs=2 / gs=4 order):
#
#     gs=1  2.047 ms      gs=2  1.820 ms      gs=4  1.740 ms      gs=8  1.803 ms
#
# Each split adds one bf16 atomic accumulation step to dK/dV, so accuracy moves the other
# way: gs=2 keeps dk at a 4.5x margin to atol, gs=4 halves it to 2.2x (= dv's own margin,
# which bf16 output quantisation sets at |dv| up to 13.9), and gs=8 would halve it again
# for no speed.  gs=4 is the largest split that stays at the margin dv already has.
GSPLIT = int(os.environ.get("GSPLIT", "4"))
PROF = int(os.environ.get("PROF", "0"))   # in-kernel clock64 phase accounting
DQX = int(os.environ.get("DQX", "0"))     # 1 = drop the dQ atomics (attribution only)
# Accumulate dQ in bf16 with atom.global.add.bf16x2 straight into the bf16 output: halves
# the dQ atomic L2 traffic and drops the FP32 dQaccum buffer and its convert pass.  Audited
# in dev/acc.py: 0 failing elements, max_abs unchanged (atol/max_abs 9.0x on mha, 4.5x on
# gqa); only the bulk of the error distribution moves, roughly doubling at the median.
DQ_BF16 = int(os.environ.get("DQ_BF16", "1"))
VLX = int(os.environ.get("VLX", "0"))   # bisect knobs for the packed-layout overhead
NTHREADS = NSMX + 128 * DQ_WG + 32
DQ_WARP0 = NSMX // 32
MMA_WARP = (NSMX + 128 * DQ_WG) // 32
NSTAGE = int(os.environ.get("NSTAGE", "3"))   # Q/dO/lse/delta pipeline depth
DQS = int(os.environ.get("DQS", "1"))         # dQ TMA-reduce staging buffers
# The staging tile is 16 384 B at BM=64 and the SMEM budget has 320 B spare, so it is
# paid for by giving sPt one stage instead of two (also 16 384 B).  sPt is read by
# exactly one MMA, dV(it), so a single buffer only needs softmax(it+1) to hold off
# until dV(it) has retired -- one mbarrier the MMA warp commits between dV and dK.
PT1 = int(os.environ.get("PT1", "1"))
PTS = 1 if PT1 else 2      # sPt stages
PTS1 = 0 if PT1 else 1     # stage index multiplier
TRX = int(os.environ.get("TRX", "0"))         # 1 = skip the TMA reduce (attribution)
ZFUSE = int(os.environ.get("ZFUSE", "1"))     # fold the dQ zero-fill into the preprocess
LSX = int(os.environ.get("LSX", "0"))         # 1 = drop the lse/delta SMEM reads (attribution)
KVTMA = int(os.environ.get("KVTMA", "1"))     # dK/dV out through the TMA unit
KVX = int(os.environ.get("KVX", "0"))         # 1 = skip the dK/dV epilogue (attribution)
KVW = int(os.environ.get("KVW", "0"))         # 1 = wait only for the TMA read, not the write
PRE = NSTAGE - 1
# dQ is the one tensor reduced through global atomics, and the width of those atomics is
# set by how many adjacent head-dim columns of one query row a thread's TMEM fragment
# holds.  The dK/dV epilogue's 16x256b pattern is only pair-contiguous on an M=64
# accumulator -- ((2,2,8),1):((1@1,8@0,8@1),0) -- so dQ went out as sixteen 4-byte atomics
# per 64-column subtile, and those 64 narrow atomics per tile step cost 1317 of that
# epilogue's 1332 cycles (DQX=1, dropping them entirely, ran mha at 1.515 ms against
# 1.955 ms).  16x32bx2 with 32 repetitions is the only pattern that is row-contiguous:
# ((32,1),1,1):((1@1,0),0,0), 32 adjacent columns of a single row, so the same reduction
# is four 16-byte atomics.  dev/probe_dqfrag.py enumerates every op and repetition with
# its fragment layout.  Width is worth far more than instruction count here: at the same
# fragment, mha runs 3.140 / 2.198 / 1.773 ms with 4 / 8 / 16-byte atomics.
DQ_LD = int(os.environ.get("DQ_LD", "1"))
# D20: give the packed layout a flat, compact grid.  Provisioning
# ceil(max_seqlen / BN) key blocks for every sequence launches 13 312 CTAs for the 4 160 that
# have work on the contest's varlen case, and SMEM allows one CTA per SM, so every empty CTA
# occupies a whole SM for a launch and an exit.  Comparing occupied SM-cycles against the
# wall clock prices that at 3.8 % of the workload: mha, which has no empty CTAs, accounts for
# 99.7 % of its kernel time in CTA lifetimes, varlen for 96.2 %.  The preprocess builds a
# (sequence, key block) map instead, so the grid is one axis of ceil(T / BN) + nseq.
VGRID = int(os.environ.get("VGRID", "1"))
# D22: the packed layout stores dK/dV instead of reduce-adding them, with each
# sequence's tail block redirected to scratch rows past the end and copied back by a
# small fix-up pass.  Retires the dkp/dvp zero-fill (269 MB) and the reduce-add's
# read-modify-write (266 MB), about 4.4 % of varlen.
VTAIL = int(os.environ.get("VTAIL", "1"))
# D22b: emit the key-block map longest sequence first, so the longest CTA is not
# left on the critical path.  dev/p3acct.py simulates the difference at 1.5 %.
VORD = int(os.environ.get("VORD", "0"))
# D17: route the kernel's own A operands through TMEM instead of SMEM.
TA_PT = int(os.environ.get("TA_PT", "1"))   # P^T  -> TMEM, feeds dV; retires sPt entirely
TA_DS = int(os.environ.get("TA_DS", "1"))   # dS^T -> TMEM, feeds dK; sSt stays for dQ
TA = TA_PT or TA_DS
DQ_VEC = int(os.environ.get("DQ_VEC", "8" if DQ_LD else "2"))
LOG2E = 1.4426950408889634
QPK = BN // BM
# lse/delta are TMA'd along the token axis.  dev/pre.py repacks them so that every
# sequence starts on a multiple of BM tokens, which keeps the box's global base address
# 16-byte aligned, and fills each sequence's tail with a huge lse so that query rows past
# the end need no per-element predicate.
LSB = BM
LPAD = ((LSB * 4 + 127) // 128) * 32   # TMA destinations must be 128-byte aligned
BIG = float(1 << 30)   # power of two, so the masking arithmetic below is exact
KVPEN = BIG * 65536.0  # enough to mask a whole key row at once


def _off(t, crd, qb, varlen):
    """Slice a (rows, dim, head, batch) tensor and, for the packed layout, shift the row
    origin to this sequence's first token so the tile index stays sequence-local."""
    x = t[crd]
    if cutlass.const_expr(varlen and not (VLX & 1)):
        x = cute.domain_offset((qb,) + (0,) * (cute.rank(x) - 1), x)
    return x


def make_mma(m, n, k, am, bm, src=None):
    op = tcgen05.MmaF16BF16Op(
        IO, ACC, (m, n, 16),
        tcgen05.CtaGroup.ONE, src or tcgen05.OperandSource.SMEM, am, bm,
    )
    return cute.make_tiled_mma(op)


@cute.struct
class Shared:
    qdo_full: cute.struct.MemRange[cutlass.Int64, NSTAGE]
    qdo_empty: cute.struct.MemRange[cutlass.Int64, NSTAGE]
    s_full: cute.struct.MemRange[cutlass.Int64, 2]
    sm_ready: cute.struct.MemRange[cutlass.Int64, 2]
    dq_full: cute.struct.MemRange[cutlass.Int64, 2]
    tm_free: cute.struct.MemRange[cutlass.Int64, 2]
    dv_done: cute.struct.MemRange[cutlass.Int64, 1]
    bar_kv: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_hold: cutlass.Int32


def _issue(x, atom_q, tQ, atom_o, tO, atom_l, tL, atom_d, tD, sQ, sO, sLse, sDel,
           full_base, qdo_bytes, one, niter, i0, hq0, bsel, qb, qba, varlen):
    """Stage Q / dO / lse / delta for flat step x into stage x % NSTAGE."""
    g = x // niter
    i = i0 + x - g * niter
    hq = hq0 + g
    s = x % NSTAGE
    gQ = cute.flat_divide(_off(tQ, (None, None, hq, bsel), qb, varlen), (BM, HD))
    gO = cute.flat_divide(_off(tO, (None, None, hq, bsel), qb, varlen), (BM, HD))
    gL = cute.flat_divide(cute.domain_offset((qba + i * BM,), tL[(None, hq, bsel)]), (LSB,))
    gD = cute.flat_divide(cute.domain_offset((qba + i * BM,), tD[(None, hq, bsel)]), (LSB,))
    qs, qg = cpasync.tma_partition(atom_q, 0, one, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
    os_, og = cpasync.tma_partition(atom_o, 0, one, cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2))
    ls, lg = cpasync.tma_partition(atom_l, 0, one, cute.group_modes(sLse, 0, 1), cute.group_modes(gL, 0, 1))
    ds, dg = cpasync.tma_partition(atom_d, 0, one, cute.group_modes(sDel, 0, 1), cute.group_modes(gD, 0, 1))
    bar = full_base + s
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bar, qdo_bytes)
    cute.copy(atom_q, qg[(None, i, 0)], qs[(None, s)], tma_bar_ptr=bar)
    cute.copy(atom_o, og[(None, i, 0)], os_[(None, s)], tma_bar_ptr=bar)
    cute.copy(atom_l, lg[(None, 0)], ls[(None, s)], tma_bar_ptr=bar)
    cute.copy(atom_d, dg[(None, 0)], ds[(None, s)], tma_bar_ptr=bar)


@cute.kernel
def kern(
    mma_s: cute.TiledMma, mma_kv: cute.TiledMma, mma_dq: cute.TiledMma,
    mma_kvt: cute.TiledMma,
    atom_q: cute.CopyAtom, tQ: cute.Tensor,
    atom_k: cute.CopyAtom, tK: cute.Tensor,
    atom_v: cute.CopyAtom, tV: cute.Tensor,
    atom_o: cute.CopyAtom, tO: cute.Tensor,
    atom_l: cute.CopyAtom, tL: cute.Tensor,
    atom_d: cute.CopyAtom, tD: cute.Tensor,
    atom_r: cute.CopyAtom, tDQ: cute.Tensor,
    atom_dk: cute.CopyAtom, tDK: cute.Tensor,
    atom_dv: cute.CopyAtom, tDV: cute.Tensor,
    mdK: cute.Tensor, mdV: cute.Tensor,
    mCu: cute.Tensor, mCup: cute.Tensor, mBlk: cute.Tensor, mDbg: cute.Tensor,
    layK: cute.ComposedLayout, layQ: cute.ComposedLayout, layPt: cute.ComposedLayout,
    layPt1: cute.ComposedLayout, layDQ: cute.ComposedLayout,
    laK_s: cute.ComposedLayout, laQ_s: cute.ComposedLayout,
    laPt_kv: cute.ComposedLayout, laPt_kv1: cute.ComposedLayout,
    lbQ_kv: cute.ComposedLayout,
    laPt_dq: cute.ComposedLayout, lbK_dq: cute.ComposedLayout,
    qk_scale: cutlass.Float32, sm_scale: cutlass.Float32,
    nq: cutlass.Int32, npad: cutlass.Int32,
    group: cutlass.Constexpr,
    varlen: cutlass.Constexpr,
    gs: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    j, hy, b = cute.arch.block_idx()
    if cutlass.const_expr(varlen and VGRID):
        # One flat key-block axis; the map packs the sequence index above the key block, and
        # a spare slot carries key block 65535 of sequence 0, which `total > 0` rejects below
        # without needing a predicate of its own.
        m = cutlass.Int32(mBlk[j])
        b = m // 65536
        j = m % 65536
    # D17 probe: the whole CTA's lifetime, against which the per-step loop totals are
    # compared.  sum(lifetime) / 148 is the occupied SM-cycles the grid actually costs, so
    # the residue over sum(loop_total) is the per-CTA prologue / epilogue / launch tax.
    if cutlass.const_expr(PROF):
        t_cta0 = cute.arch.clock64()
    # GQA otherwise has only nkv_heads * batch * key_blocks CTAs, which for 2 key heads is
    # 512 of them with a 32:1 spread in per-CTA work.  gs > 1 splits the query-head group
    # across gs CTAs -- gs = group gives one query head per CTA, i.e. MHA's grid -- and
    # reduces dK/dV across the split with atomics.  The split costs accuracy: dK/dV then
    # accumulate through gs bf16 atomic adds.
    if cutlass.const_expr(gs > 1):
        hkv = hy // gs
        GRP = group // gs
        hq0 = hkv * group + (hy - hkv * gs) * GRP
    else:
        hq0 = hy * group
        hkv = hy
        GRP = group
    KV_ATOMIC = (varlen and not (VLX & 4)) or gs > 1

    # ---------------- shared memory ----------------
    # Allocation order matters: the big 1024-byte-aligned tiles come first so that their
    # alignment padding is free, then the TMA-written scalars (which need 128-byte-aligned
    # stage strides), then the barrier struct.  The budget is 232448 bytes and this uses
    # 231808 of it.
    smem = utils.SmemAllocator()
    bK = smem.allocate_tensor(IO, layK.outer, byte_alignment=1024).iterator
    bV = smem.allocate_tensor(IO, layK.outer, byte_alignment=1024).iterator
    bQ = smem.allocate_tensor(IO, layQ.outer, byte_alignment=1024).iterator
    bO = smem.allocate_tensor(IO, layQ.outer, byte_alignment=1024).iterator
    bPt = (smem.allocate_tensor(IO, layPt1.outer, byte_alignment=1024).iterator
           if cutlass.const_expr(not TA_PT) else None)
    bSt = smem.allocate_tensor(IO, layPt.outer, byte_alignment=1024).iterator
    bDQ = smem.allocate_tensor(IO, layDQ.outer, byte_alignment=1024).iterator
    sLse = smem.allocate_tensor(ACC, cute.make_layout((LSB, NSTAGE), stride=(1, LPAD)),
                                byte_alignment=128)
    sDel = smem.allocate_tensor(ACC, cute.make_layout((LSB, NSTAGE), stride=(1, LPAD)),
                                byte_alignment=128)
    st = smem.allocate(Shared)

    def mkv(base, lay):
        return cute.make_tensor(cute.recast_ptr(base, lay.inner, IO), lay.outer)

    sK = mkv(bK, layK)
    sV = mkv(bV, layK)
    sQ = mkv(bQ, layQ)
    sO = mkv(bO, layQ)
    sPt = mkv(bPt, layPt1) if cutlass.const_expr(not TA_PT) else None
    sSt = mkv(bSt, layPt)
    sDQ = mkv(bDQ, layDQ)    # (BM, HD, DQS) staging for the TMA reduce

    frK = mma_s.make_fragment_A(mkv(bK, laK_s))
    frV = mma_s.make_fragment_A(mkv(bV, laK_s))
    frQs = mma_s.make_fragment_B(mkv(bQ, laQ_s))
    frOs = mma_s.make_fragment_B(mkv(bO, laQ_s))
    frPt = (mma_kv.make_fragment_A(mkv(bPt, laPt_kv1))
            if cutlass.const_expr(not TA_PT) else None)
    frSt = mma_kv.make_fragment_A(mkv(bSt, laPt_kv))
    mma_b = mma_kvt if cutlass.const_expr(TA) else mma_kv
    frQkv = mma_b.make_fragment_B(mkv(bQ, lbQ_kv))
    frOkv = mma_b.make_fragment_B(mkv(bO, lbQ_kv))
    frStdq = mma_dq.make_fragment_A(mkv(bSt, laPt_dq))
    frKdq = mma_dq.make_fragment_B(mkv(bK, lbK_dq))

    # Packed layout: this CTA's (b) is a sequence index, and the row origin and length come
    # from cu_seqlens.  Dense layout: one sequence per batch entry, origin 0, length N.
    if cutlass.const_expr(varlen):
        qb = cutlass.Int32(mCu[b])
        qlen = cutlass.Int32(mCu[b + 1]) - qb
        nqs = (qlen + (BM - 1)) // BM
        bsel = cutlass.Int32(0)
        qba = cutlass.Int32(mCup[b])      # lse/delta base, a multiple of BM
        r0 = j * BN
    else:
        qb = cutlass.Int32(0)
        qlen = nq * BM
        nqs = nq
        bsel = b
        qba = cutlass.Int32(0)
        r0 = j * BN

    i0 = r0 // BM
    niter = nqs - i0
    total = GRP * niter
    one = cute.make_layout(1)
    qdo_bytes = 2 * BM * HD * 2 + 2 * LSB * 4
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem_alloc = utils.TmemAllocator(st.tmem_hold, barrier_for_retrieve=tmem_bar)

    # A packed-layout CTA whose key block starts past the end of its sequence has no work
    # at all -- it does not even allocate TMEM, which matters because the grid is
    # over-provisioned to ceil(max_seqlen / BN) key blocks per sequence.
    if total > 0:
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                for sx in cutlass.range_constexpr(NSTAGE):
                    cute.arch.mbarrier_init(st.qdo_full.data_ptr() + sx, 1)
                    cute.arch.mbarrier_init(st.qdo_empty.data_ptr() + sx, 1)
                for sx in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(st.s_full.data_ptr() + sx, 1)
                    cute.arch.mbarrier_init(st.sm_ready.data_ptr() + sx, NWG * 4)
                    cute.arch.mbarrier_init(st.dq_full.data_ptr() + sx, 1)
                    cute.arch.mbarrier_init(st.tm_free.data_ptr() + sx, 4 * DQ_WG)
                cute.arch.mbarrier_init(st.dv_done.data_ptr(), 1)
                cute.arch.mbarrier_init(st.bar_kv.data_ptr(), 1)
                cute.arch.mbarrier_init_fence()

        pool = tmem_alloc.reserve(512)
        if cutlass.const_expr(PROF):
            if tidx == 0:
                cute.arch.atomic_add(mDbg.iterator + 15, cute.arch.clock64() - t_cta0)
        lay_s = mma_s.make_fragment_C(mma_s.partition_shape_C((BN, BM))).layout
        lay_kv = mma_kv.make_fragment_C(mma_kv.partition_shape_C((BN, HD))).layout
        lay_dq = mma_dq.make_fragment_C(mma_dq.partition_shape_C((BM, HD))).layout
        # 2 TMEM stages of 128 columns each, so the stage index can be a runtime value.
        st2 = cute.make_layout(2, stride=128)
        tdK = pool.allocate_tensor(lay_kv, ACC)
        tdV = pool.allocate_tensor(lay_kv, ACC)
        pS0 = pool.allocate(lay_s, ACC)      # stage 0: S^T at +0,  dP^T at +64
        pP0 = pool.allocate(lay_s, ACC)
        pool.allocate(lay_s, ACC)            # stage 1 occupies the next 128 columns
        pool.allocate(lay_s, ACC)
        tSt = cute.make_tensor(pS0, cute.append(lay_s, st2))
        tPt = cute.make_tensor(pP0, cute.append(lay_s, st2))
        # dQ aliases its stage's 128 columns, reused once the softmax has read S^T/dP^T.
        tdQ = cute.make_tensor(pS0, cute.append(lay_dq, st2))
        # D17: the bf16 A operands live in the S^T half of the same stage.  TMEM pointer
        # arithmetic is in 32-bit columns, so the bases are the pool's own column offsets:
        # dK (128) + dV (128) puts stage 0's S^T at column 256, and A_PT / A_DS at +0 / +32.
        # In the A layout's own units one column is two bf16, hence the stage stride of 256.
        lay_a1 = cute.select(make_tmem_layout_a(mma_kvt, (BN, BM), 1), mode=[0, 1, 2])
        lay_a2 = cute.append(lay_a1, cute.make_layout(2, stride=2 * 128))
        pPtA = cute.recast_ptr(pool.base_ptr + 2 * HD, dtype=IO)
        pStA = cute.recast_ptr(pool.base_ptr + 2 * HD + 32, dtype=IO)
        tPtA = cute.make_tensor(pPtA, lay_a2)      # MMA profile: ((BN,16), 1, BM/16, 2)
        tStA = cute.make_tensor(pStA, lay_a2)
        # The MMA profile splits K into k-blocks of 16, which is wrong for the softmax's
        # store; it needs the plain (BN, BM) view of the same columns.  lay_a1's K strides
        # are (1, 16) over (16, BM/16), i.e. contiguous, so the flat layout is exact.  The
        # lane stride is lay_s's doubled, since a bf16 element is half as wide
        # (dev/probe_tmema.py prints both: 1<<16 for the fp32 accumulator, 1<<17 here).
        lay_af = cute.append(cute.make_layout(((BN, BM), 1, 1), stride=((1 << 17, 1), 0, 0)),
                             cute.make_layout(2, stride=2 * 128))
        tPtAf = cute.make_tensor(pPtA, lay_af)
        tStAf = cute.make_tensor(pStA, lay_af)

        # ---------------- K/V residency (MMA warp) ----------------
        if warp_idx == MMA_WARP:
            gK = cute.flat_divide(_off(tK, (None, None, hkv, bsel), qb, varlen), (BN, HD))
            gV = cute.flat_divide(_off(tV, (None, None, hkv, bsel), qb, varlen), (BN, HD))
            tKs, tKg = cpasync.tma_partition(atom_k, 0, one, cute.group_modes(sK, 0, 2),
                                             cute.group_modes(gK, 0, 2))
            tVs, tVg = cpasync.tma_partition(atom_v, 0, one, cute.group_modes(sV, 0, 2),
                                             cute.group_modes(gV, 0, 2))
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(st.bar_kv.data_ptr(), 2 * BN * HD * 2)
            cute.copy(atom_k, tKg[(None, j, 0)], tKs[None], tma_bar_ptr=st.bar_kv.data_ptr())
            cute.copy(atom_v, tVg[(None, j, 0)], tVs[None], tma_bar_ptr=st.bar_kv.data_ptr())

            IA = (atom_q, tQ, atom_o, tO, atom_l, tL, atom_d, tD, sQ, sO, sLse, sDel,
                  st.qdo_full.data_ptr(), qdo_bytes, one, niter, i0, hq0, bsel, qb, qba,
                  varlen)
            # Fill NSTAGE-1 stages; the spare one is what step 0 writes into.
            for p in cutlass.range_constexpr(PRE):
                if p < total:
                    _issue(p, *IA)
            cute.arch.mbarrier_wait(st.bar_kv.data_ptr(), 0)

            # prologue: MMA13(0)
            cute.arch.mbarrier_wait(st.qdo_full.data_ptr(), 0)
            mma_s.set(tcgen05.Field.ACCUMULATE, False)
            a0 = tSt[(None, None, None, 0)]
            p0 = tPt[(None, None, None, 0)]
            cute.gemm(mma_s, a0, frK[(None, None, None, 0)], frQs[(None, None, None, 0)], a0)
            mma_s.set(tcgen05.Field.ACCUMULATE, False)
            cute.gemm(mma_s, p0, frV[(None, None, None, 0)], frOs[(None, None, None, 0)], p0)
            with cute.arch.elect_one():
                tcgen05.commit(st.s_full.data_ptr())

            if cutlass.const_expr(PROF):
                with cute.arch.elect_one():
                    cute.arch.atomic_add(mDbg.iterator + 12,
                                         cute.arch.clock64() - t_cta0)
            w_sm = cutlass.Int64(0)
            w_dq = cutlass.Int64(0)
            w_qf = cutlass.Int64(0)
            w_qe = cutlass.Int64(0)
            t_lp = cutlass.Int64(0)
            for it in cutlass.range(total, unroll=1):
                nx = it + 1
                if cutlass.const_expr(PROF):
                    c0 = cute.arch.clock64()
                # TMA for step it+NSTAGE-1 goes into stage (it-1)%NSTAGE, which the dQ
                # warpgroup releases in its own step it-1.
                if it + PRE < total:
                    if it >= 1:
                        if cutlass.const_expr(PROF):
                            ca = cute.arch.clock64()
                        cute.arch.mbarrier_wait(st.qdo_empty.data_ptr() + (it - 1) % NSTAGE,
                                                ((it - 1) // NSTAGE) % 2)
                        if cutlass.const_expr(PROF):
                            w_qe = w_qe + (cute.arch.clock64() - ca)
                    _issue(it + PRE, *IA)
                # MMA13(it+1) into the other TMEM stage
                if nx < total:
                    ns = nx % 2
                    if it >= 1:
                        if cutlass.const_expr(PROF):
                            ca = cute.arch.clock64()
                        cute.arch.mbarrier_wait(st.tm_free.data_ptr() + ns, ((it - 1) // 2) % 2)
                        if cutlass.const_expr(PROF):
                            w_dq = w_dq + (cute.arch.clock64() - ca)
                    if cutlass.const_expr(PROF):
                        ca = cute.arch.clock64()
                    cute.arch.mbarrier_wait(st.qdo_full.data_ptr() + nx % NSTAGE, (nx // NSTAGE) % 2)
                    if cutlass.const_expr(PROF):
                        w_qf = w_qf + (cute.arch.clock64() - ca)
                    aS = tSt[(None, None, None, ns)]
                    aP = tPt[(None, None, None, ns)]
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(mma_s, aS, frK[(None, None, None, 0)],
                              frQs[(None, None, None, nx % NSTAGE)], aS)
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(mma_s, aP, frV[(None, None, None, 0)],
                              frOs[(None, None, None, nx % NSTAGE)], aP)
                    with cute.arch.elect_one():
                        tcgen05.commit(st.s_full.data_ptr() + ns)
                # MMA245(it) once the softmax has produced P^T / dS^T
                ts = it % 2
                if cutlass.const_expr(PROF):
                    ca = cute.arch.clock64()
                cute.arch.mbarrier_wait(st.sm_ready.data_ptr() + ts, (it // 2) % 2)
                if cutlass.const_expr(PROF):
                    w_sm = w_sm + (cute.arch.clock64() - ca)
                acc = it != 0
                if cutlass.const_expr(TA_PT):
                    mma_kvt.set(tcgen05.Field.ACCUMULATE, acc)
                    cute.gemm(mma_kvt, tdV, tPtA[(None, None, None, ts)],
                              frOkv[(None, None, None, it % NSTAGE)], tdV)
                else:
                    mma_kv.set(tcgen05.Field.ACCUMULATE, acc)
                    cute.gemm(mma_kv, tdV, frPt[(None, None, None, PTS1 * ts)],
                              frOkv[(None, None, None, it % NSTAGE)], tdV)
                    if cutlass.const_expr(PT1):
                        # sPt single-buffered: tell softmax(it+1) when dV(it) has retired.
                        with cute.arch.elect_one():
                            tcgen05.commit(st.dv_done.data_ptr())
                if cutlass.const_expr(TA_DS):
                    mma_kvt.set(tcgen05.Field.ACCUMULATE, acc)
                    cute.gemm(mma_kvt, tdK, tStA[(None, None, None, ts)],
                              frQkv[(None, None, None, it % NSTAGE)], tdK)
                else:
                    mma_kv.set(tcgen05.Field.ACCUMULATE, acc)
                    cute.gemm(mma_kv, tdK, frSt[(None, None, None, ts)],
                              frQkv[(None, None, None, it % NSTAGE)], tdK)
                aQ = tdQ[(None, None, None, ts)]
                mma_dq.set(tcgen05.Field.ACCUMULATE, False)
                cute.gemm(mma_dq, aQ, frStdq[(None, None, None, ts)],
                          frKdq[(None, None, None, 0)], aQ)
                with cute.arch.elect_one():
                    tcgen05.commit(st.dq_full.data_ptr() + ts)
                    # Release this step's Q/dO stage from the MMA warp itself.  Routing it
                    # through the dQ warpgroup instead cost 950 cycles per tile step (31% of
                    # the MMA warp's loop) because the stage was then held until the dQ
                    # epilogue had run, rather than until the MMAs reading it had retired.
                    tcgen05.commit(st.qdo_empty.data_ptr() + it % NSTAGE)
                if cutlass.const_expr(PROF):
                    t_lp = t_lp + (cute.arch.clock64() - c0)
            if cutlass.const_expr(PROF):
                with cute.arch.elect_one():
                    cute.arch.atomic_add(mDbg.iterator + 0, w_sm)
                    cute.arch.atomic_add(mDbg.iterator + 1, w_dq)
                    cute.arch.atomic_add(mDbg.iterator + 2, w_qf)
                    cute.arch.atomic_add(mDbg.iterator + 3, w_qe)
                    cute.arch.atomic_add(mDbg.iterator + 4, t_lp)

        # ---------------- dQ epilogue warpgroup(s) ----------------
        elif warp_idx >= DQ_WARP0:
            lane = (tidx - NSMX) % 128
            at_dq = cute.make_copy_atom(
                tcgen05.Ld16x32bx2Op(tcgen05.Repetition(32)) if cutlass.const_expr(DQ_LD)
                else tcgen05.Ld16x256bOp(tcgen05.Repetition(8)), ACC)
            eQ = cute.flat_divide(tdQ[((None, None), 0, 0, None)], (BM, 64))
            t2rq = tcgen05.make_tmem_copy(at_dq, eQ[(None, None, 0, 0, 0)])
            thrq = t2rq.get_slice(lane)
            pQ = thrq.partition_S(eQ)
            cQ = thrq.partition_D(cute.flat_divide(cute.make_identity_tensor((BM, HD)), (BM, 64)))
            rshape = cQ[(None, None, None, 0, 0)].shape
            NREGQ = cute.size(cute.make_layout(rshape))
            NSBQ = HD // 64
            # Same "issue every TMEM load before consuming any of them" shape as D9.6: the
            # second tcgen05.ld's latency is otherwise exposed in full.
            rQs = [cute.make_rmem_tensor(rshape, ACC) for _ in range(NSBQ)]
            rQbs = [cute.make_rmem_tensor(cute.make_layout(NREGQ), IO) for _ in range(NSBQ)]
            # The dQ staging tile is written through the same thread->(row, col) map the TMEM
            # fragment already has, so the store is a plain vectorised SMEM write and needs
            # no transpose.
            sQd = cute.flat_divide(sDQ, (BM, 64))
            pSd = thrq.partition_D(sQd)
            dq_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128 * DQ_WG)
            one_dq = cute.make_layout(1)

            if cutlass.const_expr(PROF):
                if lane == 0:
                    cute.arch.atomic_add(mDbg.iterator + 17,
                                         cute.arch.clock64() - t_cta0)
            w_df = cutlass.Int64(0)
            t_ep = cutlass.Int64(0)
            t_lp = cutlass.Int64(0)
            for it in cutlass.range(total, unroll=1):
                ts = it % 2
                ds = it % DQS
                if cutlass.const_expr(PROF):
                    c0 = cute.arch.clock64()
                cute.arch.mbarrier_wait(st.dq_full.data_ptr() + ts, (it // 2) % 2)
                if cutlass.const_expr(PROF):
                    cb = cute.arch.clock64()
                    w_df = w_df + (cb - c0)
                for sb in cutlass.range_constexpr(NSBQ):
                    cute.copy(t2rq, pQ[(None, None, None, 0, sb, ts)], rQs[sb])
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                # TMEM is free the moment the fragment is in registers -- releasing it here
                # rather than after the reduction is what the atomic version could not do.
                if lane % 32 == 0:
                    cute.arch.mbarrier_arrive(st.tm_free.data_ptr() + ts)
                # The staging buffer is reusable once the TMA unit has *read* the copy
                # issued DQS steps ago.  Only warp 0 of this warpgroup issues the reduce, so
                # only warp 0 holds bulk-async groups and only it can wait on them; the
                # named barrier is what publishes "buffer free" to the other three warps.
                if lane < 32:
                    if cutlass.const_expr(DQS > 1):
                        cute.arch.cp_async_bulk_wait_group(DQS - 1, read=True)
                    else:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                dq_bar.arrive_and_wait()
                for sb in cutlass.range_constexpr(NSBQ):
                    rQbs[sb].store(cute.group_modes(
                        rQs[sb], 0, cute.rank(rQs[sb])).load().to(IO))
                    cute.autovec_copy(rQbs[sb], pSd[(None, None, None, 0, sb, ds)])
                cute.arch.fence_view_async_shared()
                dq_bar.arrive_and_wait()
                if cutlass.const_expr(not DQX):
                    g = it // niter
                    gdQ = cute.flat_divide(
                        _off(tDQ, (None, None, hq0 + g, bsel), qb, varlen), (BM, HD))
                    rs, rg = cpasync.tma_partition(
                        atom_r, 0, one_dq,
                        cute.group_modes(sDQ, 0, 2),
                        cute.group_modes(gdQ, 0, 2))
                    # TMA copies are warp-level in the DSL (it elects internally), so this
                    # must be executed by a whole warp, not by a single thread.
                    if lane < 32:
                        if cutlass.const_expr(not TRX):
                            cute.copy(atom_r, rs[(None, ds)],
                                      rg[(None, i0 + it - g * niter, 0)])
                        cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(PROF):
                    cc = cute.arch.clock64()
                    t_ep = t_ep + (cc - cb)
                    t_lp = t_lp + (cc - c0)
            # The reduction must land before the kernel's completion is observed.
            if cutlass.const_expr(PROF):
                t_fw = cute.arch.clock64()
            if lane < 32:
                cute.arch.cp_async_bulk_wait_group(0, read=False)
            if cutlass.const_expr(PROF):
                if lane == 0:
                    cute.arch.atomic_add(mDbg.iterator + 16,
                                         cute.arch.clock64() - t_fw)
            if cutlass.const_expr(PROF):
                if lane == 0:
                    cute.arch.atomic_add(mDbg.iterator + 8, w_df)
                    cute.arch.atomic_add(mDbg.iterator + 9, t_ep)
                    cute.arch.atomic_add(mDbg.iterator + 10, t_lp)

        # ---------------- softmax warpgroups (+ dK/dV epilogue) ----------------
        else:
            wg = tidx // 128
            lane = tidx % 128
            # S^T / dP^T reads: warpgroup wg owns columns [wg*SUB, wg*SUB+SUB)
            t2r_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(SUB)), ACC)
            eS = cute.flat_divide(tSt[((None, None), 0, 0, None)], (BN, SUB))
            eP = cute.flat_divide(tPt[((None, None), 0, 0, None)], (BN, SUB))
            t2r = tcgen05.make_tmem_copy(t2r_atom, eS[(None, None, 0, 0, 0)])
            thr = t2r.get_slice(lane)
            pS = thr.partition_S(eS)
            pP = thr.partition_S(eP)
            pC = thr.partition_D(cute.flat_divide(cute.make_identity_tensor((BN, BM)), (BN, SUB)))
            rshape = pC[(None, None, None, 0, 0)].shape
            rS = cute.make_rmem_tensor(rshape, ACC)
            rP = cute.make_rmem_tensor(rshape, ACC)
            rO = cute.make_rmem_tensor(rshape, IO)
            dPt = (thr.partition_D(cute.flat_divide(sPt, (BN, SUB)))
                   if cutlass.const_expr(not TA_PT) else None)
            dSt = thr.partition_D(cute.flat_divide(sSt, (BN, SUB)))
            # D17: the bf16 A operands go straight into TMEM.  Same lane -> key-row map as
            # the fp32 accumulator read above (a lane owns a key row either way), and a
            # warpgroup's SUB query columns are SUB/2 of the A operand's 32-bit columns.
            if cutlass.const_expr(TA):
                at_st = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(SUB // 2)), IO)
                ePtA = cute.flat_divide(tPtAf[((None, None), 0, 0, None)], (BN, SUB))
                eStA = cute.flat_divide(tStAf[((None, None), 0, 0, None)], (BN, SUB))
                r2t = tcgen05.make_tmem_copy(at_st, ePtA[(None, None, 0, 0, 0)])
                thr_st = r2t.get_slice(lane)
                aPt = thr_st.partition_D(ePtA)
                aSt = thr_st.partition_D(eStA)
                # One rendezvous per step: a warpgroup's A write lands on TMEM columns that
                # hold S^T query columns owned by the *other* warpgroup, so every lane must
                # be past its tcgen05.ld before any lane stores.
                ta_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=NSMX)
            rSf = cute.group_modes(rS, 0, cute.rank(rS))
            rPf = cute.group_modes(rP, 0, cute.rank(rP))
            rOf = cute.group_modes(rO, 0, cute.rank(rO))
            pCf = cute.group_modes(pC, 0, 3)
            NREG = cute.size(rSf)

            # The causal predicate is (n - m) <= dlt with n,m fixed per register, so pre-scale
            # (n - m) by BIG once and the inner loop needs only sub + fmax, no int->float.
            if cutlass.const_expr(os.environ.get("DBG_LAY")):
                print(f"DBG pC  = {pC.layout}")
                print(f"DBG rS  = {rS.layout}")
                print(f"DBG dPt = {dPt.layout}")
            # pC comes out ((32,1),1,1,1,2):((1@1,0),0,0,0,32@1): the key column index is exactly
            # wg*SUB + e, and the key row is thread-local and constant across e.  So the causal
            # predicate n - m <= dlt needs no per-register table at all: pre-scale (n - wg*SUB)
            # by a power of two and the inner loop is one sub against an immediate plus one fmax.
            mbase = wg * SUB
            nrow = pCf[(0, 0, wg)][0]
            fnm = (ACC(nrow) - ACC(mbase)) * ACC(BIG)
            if cutlass.const_expr(varlen and not (VLX & 2)):
                # A key row past the end of this sequence masks its whole row of P^T at
                # once, so its dK/dV come out exactly zero.
                fnm = fnm + cute.arch.fmax(
                    ACC(0.0), ACC(r0 + nrow + 1) - ACC(qlen)) * ACC(KVPEN)

            w_sf = cutlass.Int64(0)
            t_cp = cutlass.Int64(0)
            t_lp = cutlass.Int64(0)
            for it in cutlass.range(total, unroll=1):
                ts = it % 2
                s = it % NSTAGE
                if cutlass.const_expr(PROF):
                    c0 = cute.arch.clock64()
                if cutlass.const_expr(PT1 and not TA_PT):
                    if it >= 1:
                        cute.arch.mbarrier_wait(st.dv_done.data_ptr(), (it - 1) % 2)
                cute.arch.mbarrier_wait(st.s_full.data_ptr() + ts, (it // 2) % 2)
                if cutlass.const_expr(PROF):
                    cb = cute.arch.clock64()
                    w_sf = w_sf + (cb - c0)
                iq = i0 + it - (it // niter) * niter
                fb = fnm - ACC(iq * BM - r0) * ACC(BIG)
                cute.copy(t2r, pS[(None, None, None, 0, wg, ts)], rS)
                cute.copy(t2r, pP[(None, None, None, 0, wg, ts)], rP)
                # lse and delta are indexed by the query column, which is wg*SUB + e --
                # warp-uniform and per-register, not per-thread.  That is structural: TMEM's
                # 128 lanes are the accumulator's M rows, so a thread owns a *key row* and a
                # register owns a query column, whichever tcgen05.ld pattern is used.  Hence
                # 32 broadcast reads of each scalar per thread per step.  Reading them LVEC
                # at a time through an explicit vector copy changes nothing -- ptxas already
                # emits the same 16 instructions per warp per step either way
                # (smsp__sass_inst_executed_op_shared_ld.sum is bit-identical at LVEC
                # 1 / 2 / 8), see docs/phase2-log.md.
                for e in cutlass.range_constexpr(NREG):
                    pen = cute.arch.fmax(ACC(0.0), fb - ACC(e * BIG))
                    lsv = sLse[(mbase + e, s)] if cutlass.const_expr(not LSX) else ACC(0.0)
                    dlv = sDel[(mbase + e, s)] if cutlass.const_expr(not LSX) else ACC(0.0)
                    sv = rSf[e] * qk_scale - lsv - pen
                    p = cute.arch.exp2(sv)
                    rOf[e] = p.to(IO)
                    rPf[e] = p * (rPf[e] - dlv) * sm_scale
                if cutlass.const_expr(TA):
                    cute.arch.fence_view_async_tmem_load()
                    ta_bar.arrive_and_wait()
                if cutlass.const_expr(TA_PT):
                    cute.copy(r2t, rO, aPt[(None, None, None, 0, wg, ts)])
                else:
                    cute.autovec_copy(rO, dPt[(None, None, None, 0, wg, PTS1 * ts)])
                rOf.store(rPf.load().to(IO))
                if cutlass.const_expr(TA_DS):
                    cute.copy(r2t, rO, aSt[(None, None, None, 0, wg, ts)])
                cute.autovec_copy(rO, dSt[(None, None, None, 0, wg, ts)])
                if cutlass.const_expr(TA):
                    cute.arch.fence_view_async_tmem_store()
                else:
                    cute.arch.fence_view_async_tmem_load()
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane % 32 == 0:
                    cute.arch.mbarrier_arrive(st.sm_ready.data_ptr() + ts)
                if cutlass.const_expr(PROF):
                    cc = cute.arch.clock64()
                    t_cp = t_cp + (cc - cb)
                    t_lp = t_lp + (cc - c0)
            if cutlass.const_expr(PROF):
                if lane == 0 and wg == 0:
                    cute.arch.atomic_add(mDbg.iterator + 5, w_sf)
                    cute.arch.atomic_add(mDbg.iterator + 6, t_cp)
                    cute.arch.atomic_add(mDbg.iterator + 7, t_lp)

            # dK / dV are final once the last MMA245 has retired.
            if cutlass.const_expr(PROF):
                t_kv0 = cute.arch.clock64()
            cute.arch.mbarrier_wait(st.dq_full.data_ptr() + (total - 1) % 2,
                                    ((total - 1) // 2) % 2)
            at_m128 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), ACC)
            eKV = cute.flat_divide(tdK[((None, None), 0, 0)], (BN, 32))
            tc = tcgen05.make_tmem_copy(at_m128, eKV[(None, None, 0, 0)])
            th = tc.get_slice(lane)
            ck = th.partition_D(cute.flat_divide(cute.make_identity_tensor((BN, HD)), (BN, 32)))
            rr = cute.make_rmem_tensor(ck[(None, None, None, 0, 0)].shape, ACC)
            ro = cute.make_rmem_tensor(ck[(None, None, None, 0, 0)].shape, IO)
            NSB = (HD // 32) // NWG
            kv_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=NSMX)

            # dK/dV for a key row past the end of the sequence is exactly zero, but these are
            # plain stores: writing them would clobber the next sequence's rows.  Predicating
            # the store is not an option -- the predicate is per key row, hence divergent, and
            # the compiler then predicates each *element*, turning one 64-byte store into 32
            # 2-byte ones (measured: global_st 4.2e6 vs 5.2e5, tensor pipe 27% vs 46%).
            # Instead redirect the whole store of an out-of-range row to a scratch row past
            # the end of the padded output, which keeps every store vectorised.
            # In the packed layout a key row past the end of a sequence has dK/dV exactly
            # zero, but a plain store of it would clobber the next sequence's rows.  A
            # per-row predicate is not an option: it is divergent inside a warp, so the
            # compiler predicates each *element* and one 64-byte store becomes 32 2-byte
            # ones (measured: global_st 4.2e6 vs 5.2e5, tensor pipe 27% vs 46%).  Adding
            # instead of storing makes the out-of-range rows harmless at the same width --
            # atom.global.add on vector<8 x bf16> is 16 bytes, like the store it replaces.
            rof = cute.group_modes(ro, 0, cute.rank(ro))
            rov = cute.zipped_divide(rof, (8,))


            def store_kv(tAcc, mOut):
                gl = cute.flat_divide(
                    cute.flat_divide(_off(mOut, (None, None, hkv, bsel), qb, varlen),
                                     (BN, HD))[(None, None, j, 0)], (BN, 32))
                e = cute.flat_divide(tAcc[((None, None), 0, 0)], (BN, 32))
                ps = th.partition_S(e)
                pd = th.partition_D(gl)
                for sb in cutlass.range_constexpr(NSB):
                    col = wg * NSB + sb
                    cute.copy(tc, ps[(None, None, None, 0, col)], rr)
                    ro.store(rr.load().to(IO))
                    d = pd[(None, None, None, 0, col)]
                    if cutlass.const_expr(KV_ATOMIC):
                        df = cute.group_modes(d, 0, cute.rank(d))
                        for grp in cutlass.range_constexpr(cute.size(rof) // 8):
                            cute.arch.atomic_add(
                                df.iterator + cute.crd2idx(8 * grp, df.layout),
                                rov[(None, grp)].load().ir_value())
                    else:
                        cute.autovec_copy(ro, d)

            def stage_kv(tAcc, sBuf):
                """Stage one dK/dV accumulator into an SMEM tile for the TMA to drain."""
                e = cute.flat_divide(tAcc[((None, None), 0, 0)], (BN, 32))
                ps = th.partition_S(e)
                pd = th.partition_D(cute.flat_divide(sBuf, (BN, 32)))
                for sb in cutlass.range_constexpr(NSB):
                    col = wg * NSB + sb
                    cute.copy(tc, ps[(None, None, None, 0, col)], rr)
                    ro.store(rr.load().to(IO))
                    cute.autovec_copy(ro, pd[(None, None, None, 0, col)])

            if cutlass.const_expr(KVTMA):
                # The per-thread fragment is 32 adjacent head-dim columns of one key row =
                # 64 B, so a warp's 16-byte store instruction covers 32 *different* rows and
                # fills only 16 of every 32-byte sector -- NCU: "on average only 16.0 of the
                # 32 sectors", Est. Speedup 28.75%.  Hand the whole BN x HD tile to the TMA
                # unit instead, which needs no staging memory of its own: sK and sV are
                # already (BN, HD) bf16 tiles in exactly the layout the descriptor wants, and
                # they are dead once the last MMA has retired.  For the packed layout and for
                # the GQA query-head split the same path reduce-adds instead of storing, which
                # also retires the vector<8 x bf16> atomics those cases were paying.
                stage_kv(tdK, sK)
                stage_kv(tdV, sV)
                cute.arch.fence_view_async_shared()
                kv_bar.arrive_and_wait()
                if tidx < 32 and cutlass.const_expr(not KVX):
                    # The last key block of a packed sequence covers BN rows of which only
                    # qlen - (nkv-1)*BN belong to it; the rest are the next sequence's.  Aim
                    # its box at this sequence's scratch rows instead (`_off`'s origin is
                    # chosen so that block j lands there) and let the fix-up pass copy the
                    # valid prefix back.  Every other block stores in place.
                    qb_kv = qb
                    if cutlass.const_expr(varlen and VTAIL):
                        if r0 + BN > qlen:
                            qb_kv = npad + BN + b * BN - j * BN
                    gdK = cute.flat_divide(
                        _off(tDK, (None, None, hkv, bsel), qb_kv, varlen), (BN, HD))
                    gdV = cute.flat_divide(
                        _off(tDV, (None, None, hkv, bsel), qb_kv, varlen), (BN, HD))
                    ks, kg = cpasync.tma_partition(
                        atom_dk, 0, one, cute.group_modes(sK, 0, 2),
                        cute.group_modes(gdK, 0, 2))
                    vs, vg = cpasync.tma_partition(
                        atom_dv, 0, one, cute.group_modes(sV, 0, 2),
                        cute.group_modes(gdV, 0, 2))
                    cute.copy(atom_dk, ks[None], kg[(None, j, 0)])
                    cute.copy(atom_dv, vs[None], vg[(None, j, 0)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=bool(KVW))
            else:
                store_kv(tdK, mdK)
                store_kv(tdV, mdV)
            if cutlass.const_expr(PROF):
                if tidx == 0:
                    cute.arch.atomic_add(mDbg.iterator + 13,
                                         cute.arch.clock64() - t_kv0)

        tmem_bar.arrive_and_wait()
        if warp_idx == 0:
            tmem_alloc.relinquish_alloc_permit()
        tmem_alloc.free(pool.base_ptr)
        if cutlass.const_expr(PROF):
            if tidx == 0:
                cute.arch.atomic_add(mDbg.iterator + 11, cute.arch.clock64() - t_cta0)
                cute.arch.atomic_add(mDbg.iterator + 14, cutlass.Int64(1))


def _host_body(mQ, mK, mV, mO, mLse, mDel, mdQa, mdK, mdV, mCu, mCup, mBlk, mDbg,
               qk_scale, sm_scale, nq, nkv, npad, nhkv, nb, group, varlen, gs):
    mma_s = make_mma(BN, BM, HD, MM.K, MM.K)
    mma_kv = make_mma(BN, HD, BM, MM.K, MM.MN)
    mma_dq = make_mma(BM, HD, BN, MM.MN, MM.MN)
    # Same (M, N, K) and the same B major mode, so the accumulator and B fragment layouts
    # are identical to mma_kv's; only where A is fetched from differs.
    mma_kvt = make_mma(BN, HD, BM, MM.K, MM.MN, tcgen05.OperandSource.TMEM)

    layK = cute.select(bh.make_smem_layout(MM.K, (BN, HD), IO, 1), mode=[0, 1])
    layQ = bh.make_smem_layout(MM.K, (BM, HD), IO, NSTAGE)
    layPt = bh.make_smem_layout(MM.K, (BN, BM), IO, 2)
    layPt1 = bh.make_smem_layout(MM.K, (BN, BM), IO, PTS)
    layDQ = bh.make_smem_layout(MM.K, (BM, HD), IO, DQS)

    laK_s = bh.make_smem_layout_a(mma_s, (BN, BM, HD), IO, 1)
    laQ_s = bh.make_smem_layout_b(mma_s, (BN, BM, HD), IO, NSTAGE)
    laPt_kv = bh.make_smem_layout_a(mma_kv, (BN, HD, BM), IO, 2)
    laPt_kv1 = bh.make_smem_layout_a(mma_kv, (BN, HD, BM), IO, PTS)
    lbQ_kv = bh.make_smem_layout_b(mma_kvt if TA else mma_kv, (BN, HD, BM), IO, NSTAGE)
    laPt_dq = bh.make_smem_layout_a(mma_dq, (BM, HD, BN), IO, 2)
    lbK_dq = bh.make_smem_layout_b(mma_dq, (BM, HD, BN), IO, 1)

    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    layQ1 = cute.select(layQ, mode=[0, 1])
    atom_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, layQ1, (BM, HD))
    atom_k, tK = cpasync.make_tiled_tma_atom(op, mK, layK, (BN, HD))
    atom_v, tV = cpasync.make_tiled_tma_atom(op, mV, layK, (BN, HD))
    atom_o, tO = cpasync.make_tiled_tma_atom(op, mO, layQ1, (BM, HD))
    lay1 = cute.make_layout(LSB)
    atom_l, tL = cpasync.make_tiled_tma_atom(op, mLse, lay1, (LSB,))
    atom_d, tD = cpasync.make_tiled_tma_atom(op, mDel, lay1, (LSB,))
    op_red = cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionKind.ADD)
    atom_r, tDQ = cpasync.make_tiled_tma_atom(
        op_red, mdQa, cute.select(layDQ, mode=[0, 1]), (BM, HD))
    # dK/dV are CTA-private in the dense MHA case (one CTA owns a key block of one key head)
    # and shared otherwise -- across sequences in the packed layout, and across the GQA
    # query-head split -- so the dense case can use a plain bulk store and the rest need the
    # reduce.  Both reuse sK / sV as their staging tile, hence layK as the SMEM layout.
    kv_shared = (varlen and not VTAIL) or gs > 1
    op_kv = (cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionKind.ADD) if kv_shared
             else cpasync.CopyBulkTensorTileS2GOp())
    atom_dk, tDK = cpasync.make_tiled_tma_atom(op_kv, mdK, layK, (BN, HD))
    atom_dv, tDV = cpasync.make_tiled_tma_atom(op_kv, mdV, layK, (BN, HD))

    kern(mma_s, mma_kv, mma_dq, mma_kvt,
         atom_q, tQ, atom_k, tK, atom_v, tV, atom_o, tO, atom_l, tL, atom_d, tD,
         atom_r, tDQ, atom_dk, tDK, atom_dv, tDV, mdK, mdV, mCu, mCup, mBlk, mDbg,
         layK, layQ, layPt, layPt1, layDQ,
         laK_s, laQ_s, laPt_kv, laPt_kv1, lbQ_kv, laPt_dq, lbK_dq,
         qk_scale, sm_scale, nq, npad, group, varlen, gs,
         ).launch(grid=(nkv, nhkv * gs, 1 if (varlen and VGRID) else nb),
                  block=(NTHREADS, 1, 1))


_HOSTS = {}
_WS = {}
_WRAP = {}
_COMPILED = {}
_LAST = {}
BSHD = (1, 3, 2, 0)
SCAL = (2, 1, 0)


def get_host(nhkv, nb, group, varlen, gs=1):
    key = (nhkv, nb, group, varlen, gs)
    f = _HOSTS.get(key)
    if f is None:
        @cute.jit
        def f(mQ, mK, mV, mO, mLse, mDel, mdQa, mdK, mdV, mCu, mCup, mBlk, mDbg,
              qk_scale: cutlass.Float32, sm_scale: cutlass.Float32,
              nq: cutlass.Int32, nkv: cutlass.Int32, npad: cutlass.Int32):
            _host_body(mQ, mK, mV, mO, mLse, mDel, mdQa, mdK, mdV, mCu, mCup, mBlk, mDbg,
                       qk_scale, sm_scale, nq, nkv, npad, nhkv, nb, group, varlen, gs)
        _HOSTS[key] = f
    return f


def _tw(x, perm):
    key = (x.data_ptr(), tuple(x.shape), perm)
    ent = _WRAP.get(key)
    if ent is None:
        view = x.permute(*perm)
        ent = (view, from_dlpack(view, assumed_align=16))
        _WRAP[key] = ent
    return ent[1]


def _ws(key, shape, device, dtype=torch.float32):
    buf = _WS.get(key)
    if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WS[key] = buf
    return buf


def _iws(key, shape, device):
    return _ws(key, shape, device, torch.int32)


NPROF = 20


def _dbg(device):
    buf = _WS.get("dbg")
    if buf is None:
        buf = torch.zeros(NPROF, dtype=torch.int64, device=device)
        _WS["dbg"] = buf
    return buf


def prof_report(iters):
    """Per-phase cycle accounting from the last PROF run, per CTA per step."""
    names = ["mma:wait_sm_ready", "mma:wait_dq_free", "mma:wait_qdo_full",
             "mma:wait_qdo_empty", "mma:loop_total", "sm:wait_s_full", "sm:compute",
             "sm:loop_total", "dq:wait_dq_full", "dq:epilogue", "dq:loop_total",
             "cta:lifetime", "mma:prologue", "sm:kvepilogue", "cta:count",
             "cta:tmem_alloc", "dq:final_wait", "dq:prologue", "spare"]
    v = _WS["dbg"].tolist()
    for n, x in zip(names, v):
        print(f"  {n:22s} {x / max(iters, 1):12.1f} cycles/step")


def attn_bwd_bshd(q, k, v, out, dout, lse, sm_scale, dq, dk, dv):

    B, N, Hq, _ = q.shape
    Hkv = k.shape[2]
    group = Hq // Hkv
    assert N % BN == 0 and N % BM == 0

    delta = _ws(("delta", B, N, Hq), (B, Hq, N), q.device)
    lse2 = _ws(("lse2", B, N, Hq), (B, Hq, N), q.device)
    if DQ_BF16:
        dqa = dq          # accumulate straight into the bf16 output, no convert pass
    else:
        dqa = _ws(("dqa", B, N, Hq), (B, N, Hq, HD), q.device)

    gs = min(group, GSPLIT)
    # dQ is reduced into, so it has to start at zero; the preprocess pass does that fill
    # alongside its own streaming reads rather than paying for a separate 134 MB memset.
    pre_bshd(out, dout, lse, delta, lse2, B, N, Hq, HD, dq=dqa if ZFUSE else None)
    if not ZFUSE:
        dqa.zero_()
    if gs > 1:
        dk.zero_()          # dK/dV are reduced across the query-head split with atomics
        dv.zero_()

    dummy = _iws(("cu0",), (1,), q.device)
    args = (_tw(q, BSHD), _tw(k, BSHD), _tw(v, BSHD), _tw(dout, BSHD),
            _tw(lse2, SCAL), _tw(delta, SCAL),
            _tw(dqa, BSHD), _tw(dk, BSHD), _tw(dv, BSHD),
            _tw(dummy, (0,)), _tw(dummy, (0,)), _tw(dummy, (0,)),
            _tw(_dbg(q.device), (0,)),
            cutlass.Float32(sm_scale * LOG2E), cutlass.Float32(sm_scale),
            cutlass.Int32(N // BM), cutlass.Int32(N // BN), cutlass.Int32(0))
    key = ("bshd", B, N, Hq, Hkv, gs)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(get_host(Hkv, B, group, False, gs), *args)
        _COMPILED[key] = fn
    _LAST["call"] = (fn, args)
    fn(*args)
    if not DQ_BF16:
        dq.copy_(dqa)
    return dq, dk, dv


VLEN = (0, 2, 1)   # (T,H,D) -> (T,D,H,1) for the TMA, matching the dense (N,D,H,B) form
VSCAL = (1, 0)     # (H,T)   -> (T,H,1)


def _tw4(x, perm):
    """Like _tw, but append a length-1 batch mode so the packed layout can share the
    dense kernel's (rows, dim, head, batch) indexing."""
    key = (x.data_ptr(), tuple(x.shape), perm, 4)
    ent = _WRAP.get(key)
    if ent is None:
        view = x.permute(*perm).unsqueeze(-1)
        ent = (view, from_dlpack(view, assumed_align=16))
        _WRAP[key] = ent
    return ent[1]


def attn_bwd_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale,
                    dq=None, dk=None, dv=None):

    T, H, _ = q.shape
    nseq = cu_seqlens.shape[0] - 1
    max_seqlen = int(max_seqlen)

    # Repacked per-row scalars: every sequence starts on a multiple of BM tokens, so the
    # upper bound on the padded token count is T + BM*nseq (rounded for the TMA stride).
    Tp = (T + BM * nseq + 7) & ~7
    delta = _ws(("vdelta", T, H, nseq), (H, Tp), q.device)
    lse2 = _ws(("vlse2", T, H, nseq), (H, Tp), q.device)
    cup = _iws(("vcup", nseq), (nseq + 1,), q.device)
    # ceil(T / BN) + nseq bounds the number of (sequence, key block) pairs with work, and is
    # computable on the host -- cu_seqlens lives on the device, so the exact count is not.
    nblk = (T + BN - 1) // BN + nseq
    bmap = _iws(("vbmap", T, nseq), (nblk + nseq,), q.device)
    # dQ is accumulated with atomics over tiles of BM rows.  The last query tile of the last
    # sequence can reach up to BM-1 rows past the end of the packed tensor; those rows carry
    # exactly zero (their P^T is exactly zero), so the only requirement is that the address
    # be legal.  Padding the buffer by BM rows is cheaper than predicating every atomic.
    dqp = _ws(("vdq", T, H), (T + BM, H, HD), q.device, torch.bfloat16)
    # dK/dV are accumulated (not stored) so that out-of-range key rows add exactly zero;
    # that needs zeroed buffers and BN scratch rows for the last sequence's tail.
    # + BN of scratch for the dense-style stores, then one BN-row tail region per
    # sequence for VTAIL's redirected last block.
    ktail = BN * (nseq + 1) if VTAIL else BN
    dkp = _ws(("vdk", T, H, ktail), (T + ktail, H, HD), q.device, torch.bfloat16)
    dvp = _ws(("vdv", T, H, ktail), (T + ktail, H, HD), q.device, torch.bfloat16)

    pre_varlen(out, dout, lse, delta, lse2, cu_seqlens, cup, T, Tp, nseq, max_seqlen,
               H, HD, BM, bmap=bmap, nblk=nblk, BN=BN,
               dq=dqp if VTAIL else None, vord=VORD)
    if not VTAIL:
        dqp.zero_()
        dkp.zero_()
        dvp.zero_()

    args = (_tw4(q, VLEN), _tw4(k, VLEN), _tw4(v, VLEN), _tw4(dout, VLEN),
            _tw4(lse2, VSCAL), _tw4(delta, VSCAL),
            _tw4(dqp, VLEN), _tw4(dkp, VLEN), _tw4(dvp, VLEN),
            _tw(cu_seqlens, (0,)), _tw(cup, (0,)), _tw(bmap, (0,)),
            _tw(_dbg(q.device), (0,)),
            cutlass.Float32(sm_scale * LOG2E), cutlass.Float32(sm_scale),
            cutlass.Int32(0),
            cutlass.Int32(nblk if VGRID else (max_seqlen + BN - 1) // BN),
            cutlass.Int32(T))
    key = ("varlen", T, H, nseq, max_seqlen, nblk)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(get_host(H, nseq, 1, True, 1), *args)
        _COMPILED[key] = fn
    _LAST["call"] = (fn, args)
    fn(*args)
    if VTAIL:
        fix_tail(dkp, dvp, cu_seqlens, T, nseq, H, HD, BN)
    rq, rk, rv = dqp[:T], dkp[:T], dvp[:T]
    if dq is not None:
        dq.copy_(rq)
        dk.copy_(rk)
        dv.copy_(rv)
        return dq, dk, dv
    return rq, rk, rv




# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def run_bshd(q, k, v, out, dout, lse, sm_scale, dq, dk, dv):
    """attention_bwd_causal_h16_d128, layout (batch, seq, heads, dim).

    Destination passing style: dq/dk/dv are pre-allocated by the harness.
    """
    return attn_bwd_bshd(q, k, v, out, dout, lse, sm_scale, dq, dk, dv)


def run_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    """attention_bwd_varlen_causal_h16_d128, layout (tokens, heads, dim).

    Value returning, *not* destination passing: for this definition the evaluator's
    allocate_outputs() cannot resolve the `num_seqs` axis from the input shapes -- it is
    only implied by the `len_cu_seqlens == num_seqs + 1` constraint -- so a DPS solution
    raises before the kernel is ever called.
    """
    return attn_bwd_varlen(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale)
