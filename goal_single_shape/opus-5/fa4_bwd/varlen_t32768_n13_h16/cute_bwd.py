"""CuTe DSL tcgen05 varlen causal MHA backward for Blackwell (sm100).

Transposed-S formulation, five MMAs per (n-tile, m-tile) pair:
    sT  = K  @ Q^T          (BN, BM)   acc slot A
    dpT = V  @ dO^T         (BN, BM)   acc slot B
    pT  = exp2(sT*qk_scale - lse[m]*log2e)  masked  -> sP  (bf16, SMEM)
    dsT = pT * (dpT - delta[m])                     -> sDS (bf16, SMEM)
    dV += pT  @ dO          (BN, D)    persistent acc
    dK += dsT @ Q           (BN, D)    persistent acc
    dQ  = ds  @ K           (BM, D)    acc slot A (reused), TMA-reduced out

Every SMEM tile is [128,128] bf16 with a 128B swizzle and D (or BM) contiguous.
A K-major view and the transposed MN-major view of such a tile are byte-identical
-- K gives (128,(64,2)):(64,(1,8192)), MN gives ((64,2),128):((1,8192),64), both
under S<3,4,3> -- so all five MMAs read the same physical tiles, no transposes.
"""

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack

from tri_bwd import _pre_kernel, _sched_kernel

LOG2E = 1.4426950408889634


class Cfg:
    BM = 128
    BN = 128
    # Swept: (epi_n, threads) = (8, 512) is the best of {8,16,32}x{128..1024}.
    # 1024 threads spills and costs ~30%; wider epi tiles raise register
    # pressure in the softmax loop faster than they cut copy overhead.
    epi_n = 8
    threads = 512
    # Debug bisect knobs, all no-ops at their defaults.
    #   iters: cap the m-loop trip count (0 = no cap)
    #   skip:  bit0 MMA1/3, bit1 softmax, bit2 MMA2/4/5, bit3 dQ epilogue,
    #          bit4 m-loop, bit5 dK/dV epilogue, bit6 K/V load,
    #          bit7 dQ TMA store only (keeps the TMEM->SMEM staging)
    #   softmax cost-attribution probes (all produce wrong results):
    #          bit11 drop the TMEM loads, bit12 drop exp2, bit13 drop the ds
    #          math, bit14 fold ds into p so only one SMEM store is issued,
    #          bit16 disable the TMEM double buffer,
    #          bit18 drop the lse/delta bulk copy (sLSED uninitialised),
    #          bit29 drop the dS r2s store, keeping the ds chain alive by
    #          folding ds*1e-30 into P -- bit256 was meant to price the
    #          stores but dead-codes the whole softmax instead (DESIGN 38.3),
    #          bit19 drop the in-loop Q/dO TMA and its waits
    #          (the prologue load is kept, so every m-tile reuses tile n_idx)
    #   per-sync attribution probes (round 9, all produce wrong results):
    #          bit20 drop sync #1 (loop head, after the bar_m1 wait),
    #          bit21 drop sync #2 (after the softmax, before GEMM2/4/5),
    #          bit22 drop sync #3 (after the dQ staging, before the TMA store),
    #          bit23 drop both fence_view_async_tmem_load,
    #          bit24 drop both fence_proxy("async.shared"),
    #          bit26 drop the dQ-store drain (cp_async_bulk_wait_group)
    #   structure knob: bit27 syncs #2/#3 as arrive-only named barriers for
    #          the 15 warps that do not issue anything (correct, not faster)
    iters = 0
    skip = 0
    # Structure knob: invert the loop nest.  A work item owns one m-tile of
    # Q and walks the n-tiles, so dQ becomes the resident accumulator and
    # dK/dV are produced once per iteration and reduced into global memory.
    # That is what frees a TMEM slot for the next tile's GEMM1 to be issued
    # from inside the softmax window -- the one placement 25.x showed does
    # not merely relocate the issuer's back-pressure.  See DESIGN 30.4.
    mout = 0
    dqb = 4
    # occ=1 is a TIMING PROBE, NOT A KERNEL: it produces wrong results on
    # purpose.  Every input tile is aliased onto one 64KB block and dK/dP are
    # aliased onto dV/S, which puts the CTA inside 114.8KB of SMEM and 256
    # TMEM columns -- the two gates 47.1 measured -- so a second CTA can be
    # co-resident.  The instruction stream, every UMMA shape and every TMA is
    # unchanged, so it carries no shape premium and what it measures is an
    # upper bound on the 2 CTA/SM route.  48.3 shows the kernel's own hiding
    # rate is bracketed between ~10% and ~90% by everything measured so far
    # and that no probe can resolve it, because the rate is bimodal in
    # whether a shared per-SM unit saturates and the kernel drives six of
    # them at once.  This reads the answer off the kernel itself.
    occ = 0
    # Structure knob: split the softmax into a p-pass and a ds-pass so that
    # GEMM2 can be issued between them (see the comment in sfx).
    tpass = 0
    # Persistent grid: number of CTAs that walk the whole work list with a
    # stride.  0 keeps one CTA per work item (the original launch); -1 means
    # one CTA per SM, read from the device rather than written down.  A larger
    # grid is pointless -- TMEM and SMEM both cap residency at one CTA/SM --
    # and measurably worse (pers = 2*SM costs +450us, two waves against that
    # cap).
    pers = -1
    # Work-index mapping inside the persistent loop:
    #   0  head-major static stride (pair = w % npair_max)
    #   1  pair-major boustrophedon -- globally descending cost, near-perfect
    #      balance, but it loses the L2 reuse of the head-major order
    #   2  head-major with a global atomic ticket (dynamic balance)
    # The ticket is the only mapping that pays for the persistent grid: the
    # work items differ in length by 32x (the shortest doc is 257 tokens, the
    # longest 8192), so a static stride leaves the tail of the wave idle.
    pmaj = 2
    # Number of segments the softmax is cut into so that GEMM2/4's leading K
    # blocks can be issued between them.  0 keeps the single-segment softmax.
    kseg = 0
    # Softmax TMEM prefetch depth: sub-tile j+dbd-1 is loaded while j is being
    # reduced.  2 is a plain double buffer.
    dbd = 2
    # Issue GEMM1 for tile i+1 at the TAIL of tile i instead of at the head of
    # tile i+1.  dQ parks on TMEM slot B, so slot A (S) is dead the moment the
    # softmax retires it and the issue can move off the critical path: warp 0
    # leaves the MMA region with 32 UMMAs queued while the other warps run the
    # dQ epilogue.  Measured: 1886us vs 1751/1797 baseline, i.e. 110us WORSE,
    # and 1805 vs 1723/1724 under bit19 -- so it is not the shallower Q
    # prefetch (bit19 removes that wait entirely) but the overlap itself that
    # costs.  GEMM1 writes TMEM slot A exactly while the dQ epilogue LDTMs
    # slot B, and the two contend for the TMEM port: the MMA pipe and the CTA
    # cannot both touch TMEM at once, which is also why dropping the MMA waits
    # (bit30) buys only 34us and why epilogue overlap depth is non-monotonic
    # (0 -> 1855, 1 -> 1764, 2 -> 2173).  Left in as a knob because it is the
    # cleanest probe of that contention.
    #
    # That TMEM-port reading is WRONG -- probe_tmem_bw measured UMMA writes and
    # LDTM reads at full overlap (41.2 / 756.1 / 753.6us for read-only /
    # write-only / both, where serial would be 797.4).  What actually pins the
    # two together is the tcgen05 pipe's in-order retirement, and g1f=1 does
    # not change the order: GEMM1 still retires after GEMM5/2/4, so bar_m1 --
    # which the next softmax waits on -- still transitively drains all 40
    # UMMAs.  g1f=2 issues GEMM1 FIRST instead, which is the version the
    # ordering argument actually calls for.  See DESIGN 19.
    g1f = 0
    # Move every TMA issue off warp 0 and onto warp 1, and issue the Q load
    # from the same place as the dO load.  warp 0 is the only warp that issues
    # UMMAs, the issue is back-pressured by the MMA pipe, and warp 0 is also a
    # participant in every sync_threads -- so anything else parked on it lands
    # straight on the critical path.  warp 1 is sitting on bar_m5 at that point
    # anyway.  Q has to move to the dO site rather than stay at the loop head:
    # its arrive_and_expect_tx must not overtake warp 0's wait on bar_tma, or
    # the two loads fold into one phase and the wait covers both.  Measured:
    # 1785/1810 against 1785/1752 interleaved, i.e. nothing outside the noise.
    # A TMA issue is a handful of instructions and costs well under 50ns, so
    # warp 0 is not held up by carrying them -- which, together with g1f above,
    # rules out "work parked on warp 0" as the explanation for either result.
    # Incompatible with g1f: that one waits on Q(i+1) in [F], which precedes
    # the issue site here, so both on at once deadlocks.
    tw = 0
    # Warp-specialise the UMMA issuer.  The per-iteration ledger is
    #   575ns issue GEMM1/3 | 768ns softmax | 862ns issue GEMM5/2/4 |
    #   346ns dQ epilogue | 810ns everything else = 3361ns, against a
    # measured 3360ns -- fully serial, because warp 0 both issues every UMMA
    # (where it is back-pressured by the tcgen05 pipe for the whole execution
    # time of what it queues; g1f=2 proved this by costing exactly the 8
    # instructions it inserted) and participates in every sync_threads.
    # With ws the CTA runs 17 warps: warp 16 issues and nothing else, so its
    # 862ns of GEMM5/2/4 issue runs underneath the other 16 warps' dQ
    # epilogue instead of in front of it.  The 575ns of GEMM1/3 stays exposed
    # -- the softmax genuinely depends on it -- and so does the softmax
    # itself, since GEMM5/2/4 cannot be issued before the dS it reads exists.
    # Worker warps keep tidx 0..511, so every partition index is unchanged.
    # The dK/dV epilogue is NOT specialised: it holds a sync_threads, which is
    # a whole-CTA barrier that warp 16 has to reach.  It runs once per n-tile
    # (97us total), and warp 16 simply shadows warp 0 there -- tidx%128 already
    # maps 512..543 onto warp 0's lanes and every write it makes is the same
    # value to the same address.
    ws = 0
    # Hoist every S/dP sub-tile into registers before any arithmetic, fence,
    # and issue the NEXT tile's GEMM1 into the slot the hoist just emptied --
    # from inside the softmax, which is the only window where a UMMA overlaps
    # CUDA-core work instead of displacing it.
    hst = 0
    # Break the GEMM -> softmax -> GEMM serial chain.  SMEM (227KB) pins the
    # kernel at ONE CTA per SM, so the tensor pipe and the CUDA cores can only
    # overlap if a single CTA's own dependences allow it, and they do not:
    # S and dP are both live across the fused softmax, so all four TMEM slots
    # are occupied and nothing can be issued into them.  tpass=2 makes them
    # live in DISJOINT halves -- the ds-pass reads bf16 P back out of sP and
    # never touches slot A again -- which frees slot A at the pass boundary.
    # ovl issues the next tile's GEMM1 (into slot A) and this tile's GEMM2
    # (which reads only sP, just written) there, so 16 of the 40 UMMAs execute
    # underneath the ds-pass instead of displacing it.
    ovl = 0
    # g1b splits the barrier ovl waits on in two, so that BOTH softmax passes
    # get a UMMA group to run under instead of only the ds-pass: the p-pass
    # waits on bar_g1 (GEMM1' + GEMM2 only) and the ds-pass on bar_m1 (which
    # covers GEMM3).  Without it the p-pass waits on bar_m1 too and drains
    # GEMM5/4/3 first, which is why ovl could never hide more than 16 UMMAs.
    g1b = 0
    # spc: warpgroup 3 stops doing softmax and becomes the sole UMMA
    # issuer, inside the same 512-thread launch.  ws did the same with a
    # 17th warp and lost, but 544 threads drop ptxas from 128 to 96
    # registers per thread; staying at 512 keeps the register budget and
    # costs only the 12/16 of the softmax width the issuer used to carry.
    spc = 0
    # preg: hand P from the p-pass to the ds-pass in registers instead of
    # reading it back out of sP.  sP is still written -- GEMM2 reads it
    # out of shared -- but the round trip the ds-pass paid per element is
    # gone.  Only affordable under spc: a warpgroup then owns n_sub+1 = 6
    # sub-tiles, so P costs 24 registers; the 64 that spilled when this
    # was tried before were for all 16.
    preg = 0
    # inc: issue GEMM2/4 from inside the softmax, one K block at a time, as
    # soon as the sub-tiles feeding it are stored.  With the default
    # contiguous deal a warpgroup owns sub-tiles [4*wg, 4*wg+4) = rows
    # [32*wg, 32*wg+32) of BM, which is exactly K blocks 2*wg and 2*wg+1 of
    # both GEMMs, so the rendezvous is warpgroup-local: no CTA-wide sync, no
    # round-robin sub-tile order, and no tmem_load fence (the early GEMMs
    # write dV/dK, never the slots the softmax is reading).  Those three are
    # what made kseg cost 40us per issue point on top of an 86us reorder;
    # what is left is one fence_proxy and one bar.sync over 128 threads.
    # K blocks commute -- tcgen05 retires UMMAs in issue order and they all
    # accumulate -- so the four warpgroups may issue theirs independently.
    inc = 0

    # unr: LLVM unroll factor for the m-loop.  The point is not the loop
    # control but `ph`: with unroll=1 the Q double-buffer slot is a runtime
    # value, so the 16 UMMAs of GEMM1 and GEMM4 that read `st=ph` recompute
    # their SMEM descriptor base every iteration, and the mbarrier waits take
    # a runtime phase.  Unrolling by two lets SCEV prove the parity and folds
    # all of it to constants, at the price of doubling the body in I-cache.
    unr = 0

    # rlo: registers the spc issuer warpgroup keeps (setmaxnreg decrease).
    # 0 disables the redistribution and leaves the uniform allocation.
    rlo = 0

    # nkh: TIMING PROBE ONLY -- results are wrong.  Divide every GEMM's K
    # loop by nkh, so the kernel issues 40/nkh UMMAs per iteration while
    # every mbarrier, commit, fence and softmax stays byte-identical.  The
    # skip bits that drop whole GEMMs (bit0/bit2) break the CUDA context;
    # this prices the tensor pipe without touching the control flow.  If the
    # time scales with the UMMA count, the UMMAs are serial with the rest of
    # the iteration; if it does not, they already overlap.
    nkh = 0

    # dqs: TIMING PROBE ONLY -- results are wrong.  Issue the dQ epilogue's
    # bulk tensor store as a plain store instead of a store-reduce.  The
    # byte volume and the instruction count are identical; only the L2
    # read-modify-write goes away.  bit7 prices the whole store at 215us --
    # this splits that into the atomic and the raw traffic.
    dqs = 0

    # Softmax share of the warpgroup that owns the MMA issuer.  Warp 0 posts
    # all 40 UMMAs of an iteration and sits in the tcgen05 queue while it
    # does, so with the even 4/4/4/4 split of the 16 sub-tiles it reaches the
    # closing barrier last and the other three warpgroups wait on it.  wgs=1
    # gives it one sub-tile and the other three five each; 0 keeps the even
    # split.  Only values leaving (16 - wgs) divisible by three are legal.
    wgs = 0

    # CTA cluster size along x.  The kernel uses no cluster-scoped
    # instruction, so this only changes placement: the CTAs of a cluster are
    # guaranteed to land on the same GPC, hence to share L2 slices.  Adjacent
    # work items under the pair-major schedule walk the same K/V tile or the
    # same dQ rows, so co-locating them is the one L2 lever left that does not
    # need the mixed cta_group MMA the toolchain refuses (3.3, 22.3).
    clu = 0
    # Fold the two softmax scale factors into the operands, so that the
    # per-element multiplies in the ds math leave the m-loop.
    #   bit0: V *= sm_scale in SMEM once per work item.  dP = dO @ V^T carries
    #         the factor into the ds math and dV = P^T @ dO never reads V, so
    #         `dp_e * sc` disappears with no correction anywhere downstream.
    #   bit1: K *= qk_scale in SMEM, and lse gains log2(qk_scale) so that
    #         p comes out as P/qk and ds as dS/qk.  dQ = ds @ K' is then exact
    #         with no epilogue scaling, and dV/dK each take one *qk in their
    #         epilogue -- paid once per work item instead of 32 FFMA per lane
    #         per m-tile (the m-loop averages 18.6 tiles per work item).
    fsc = 0


class BwdKernel:
    def __init__(self, bm, bn, d, epi_n, threads, iters=0, skip=0,
                 tpass=0, pers=0, pmaj=0, kseg=0, dbd=2, g1f=0, tw=0,
                 ws=0, hst=0, ovl=0, spc=0, preg=0, inc=0, unr=0, rlo=0,
                 nkh=0, dqs=0, wgs=0, clu=0, fsc=0, mout=0, dqb=1, g1b=0,
                 occ=0):
        self.occ = occ
        self.bm, self.bn, self.d = bm, bn, d
        # dK/dV stage through sP/sDS, which are (BN, BM) tiles.
        assert bm == d, "dK/dV staging needs BM == D"
        self.epi_n = epi_n
        # The extra warp is the issuer; nwg stays at the worker count so the
        # softmax and epilogue keep their existing 4-way warpgroup split.
        self.threads = threads + 32 if ws else threads
        self.iters = iters
        self.skip = skip
        self.tpass = tpass
        self.pers = pers
        self.pmaj = pmaj
        self.kseg = kseg
        self.dbd = dbd
        self.g1f = g1f
        self.tw = tw
        self.ws = ws
        self.hst = hst
        self.ovl = ovl
        self.g1b = g1b
        self.spc = spc
        self.preg = preg
        self.inc = inc
        self.unr = unr
        self.rlo = rlo
        self.nkh = nkh
        self.dqs = dqs
        self.wgs = wgs
        self.clu = clu
        self.fsc = fsc
        self.mout = mout
        # Landing-pad depth for the dQ epilogue's tcgen05.ld stream.  At
        # depth 1 every sub-tile loads into the same registers, so the
        # unrolled loop carries a WAR chain and each load's TMEM latency
        # is fully exposed -- ncu attributes 59% of all stall cycles to
        # long_scoreboard.  Deeper rings let the loads fly concurrently.
        self.dqb = dqb
        # m-outer rewires the loop bounds, both epilogues and the operand that
        # is preloaded.  None of the scheduling knobs were written against that
        # shape, so they are refused rather than silently mis-compiled.
        assert not mout or not (ws or hst or ovl or spc or kseg or inc or wgs
                                or g1f or tw or clu or fsc or nkh or rlo
                                or preg or dqs), \
            "mout=1 is not composable with the n-outer scheduling knobs"
        # ws+kseg is the combination that matters: kseg alone hands the segment
        # issue to warp 0, which is then back-pressured out of its own softmax
        # share, and ws alone only frees that one share.  Together the issuer is
        # warp 16 and the segments overlap the softmax for real.
        self.ab_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE

    # ------------------------------------------------------------------ setup
    def _setup(self):
        ab, acc = self.ab_dtype, self.acc_dtype
        K, MN = OperandMajorMode.K, OperandMajorMode.MN

        def mk(am, bmm, mn):
            return sm100_utils.make_trivial_tiled_mma(
                ab, ab, am, bmm, acc, self.cta_group, mn
            )

        # (1) sT = K@Q^T and (3) dpT = V@dO^T : M=BN, N=BM, K=D
        self.mma_s = mk(K, K, (self.bn, self.bm))
        # (2) dV += pT@dO and (4) dK += dsT@Q : M=BN, N=D, K=BM
        self.mma_kv = mk(K, MN, (self.bn, self.d))
        # One atom per early-issue segment.  set() replaces the atom's SSA
        # value in place, so reusing a single atom across two `warp_idx == 0`
        # regions makes the second region's set() reference a value defined in
        # the first, which does not dominate it (ICE: "operand #0 does not
        # dominate this use").  Fragments are tiler-bound, not atom-bound, so
        # fP/fDS/fDOmn/fQmn are shared.
        self.mma_kv_seg = [mk(K, MN, (self.bn, self.d))
                           for _ in range(max(self.kseg - 1, 0))]
        # GEMM1/3 are issued from up to three disjoint `warp_idx == 0`
        # regions (prologue, loop head, loop tail) and each needs its own atom
        # for the same reason mma_kv_seg does.
        self.mma_s_a = mk(K, K, (self.bn, self.bm))
        self.mma_s_f = mk(K, K, (self.bn, self.bm))
        self.mma_s_h = mk(K, K, (self.bn, self.bm))
        # (5) dQ = ds@K : M=BM, N=D, K=BN
        self.mma_dq = mk(MN, MN, (self.bm, self.d))

        tiler_s = (self.bn, self.bm, self.d)
        tiler_kv = (self.bn, self.d, self.bm)
        tiler_dq = (self.bm, self.d, self.bn)

        self.lay_a_k = sm100_utils.make_smem_layout_a(self.mma_s, tiler_s, ab, 1)
        self.lay_b_k = sm100_utils.make_smem_layout_b(self.mma_s, tiler_s, ab, 1)
        # Q is double buffered so that the load for tile i+1 can be issued at
        # the TOP of iteration i, which is what lets GEMM1(i+1) be issued ahead
        # of GEMM5/2/4(i).  dO stays single buffered: GEMM3(i+1) cannot move up
        # anyway, because its TMEM slot is the one GEMM5 writes dQ into.
        self.lay_b_k_q = sm100_utils.make_smem_layout_b(
            self.mma_s, tiler_s, ab, 2)
        self.lay_b_mn_q = sm100_utils.make_smem_layout_b(
            self.mma_kv, tiler_kv, ab, 2)
        self.lay_md_q = sm100_utils.make_smem_layout(K, (self.bm, self.d), ab, 2)
        self.lay_a_p = sm100_utils.make_smem_layout_a(self.mma_kv, tiler_kv, ab, 1)
        self.lay_b_mn = sm100_utils.make_smem_layout_b(self.mma_kv, tiler_kv, ab, 1)
        self.lay_a_mn = sm100_utils.make_smem_layout_a(self.mma_dq, tiler_dq, ab, 1)
        self.lay_b_mn2 = sm100_utils.make_smem_layout_b(self.mma_dq, tiler_dq, ab, 1)
        # mout inverts which operand needs the extra stage.  K is read by
        # GEMM1 at the head and by GEMM5 at the tail of the same iteration,
        # so its refill cannot land before bar_m5 unless it has a second
        # slot; Q, loaded once per work item, gives up the one it had.
        self.lay_a_k2 = sm100_utils.make_smem_layout_a(self.mma_s, tiler_s, ab, 2)
        self.lay_b_mn2_2 = sm100_utils.make_smem_layout_b(
            self.mma_dq, tiler_dq, ab, 2)
        self.lay_nd2 = sm100_utils.make_smem_layout(K, (self.bn, self.d), ab, 2)

        # Plain 2D views of the MMA operand blocks, one per distinct shape.
        # They coincide only when BM == BN == D, so each TMA descriptor picks
        # the one matching its own box.
        def lay2d(shape):
            return cute.slice_(
                sm100_utils.make_smem_layout(K, shape, ab, 1), (None, None, 0))

        self.lay_nd = lay2d((self.bn, self.d))   # K, V, and the dK/dV stores
        self.lay_md = lay2d((self.bm, self.d))   # Q, dO, and the dQ staging
        self.lay_nm = lay2d((self.bn, self.bm))  # P, dS

        self.epi_tile = (self.bn, self.epi_n)
        self.epi_tile_dq = (self.bm, self.epi_n)
        # Each 128-lane warpgroup owns one slice of the epilogue tiles.  The
        # softmax walks the BM columns of S/dP, the dK/dV/dQ epilogues walk the
        # D columns of their accumulators; equal only when BM == D.
        # spc carves the issuer warpgroup out of the threads already
        # launched instead of adding a warp, so the softmax runs on 12.
        self.nwg = ((self.threads - 32 if self.ws
                     else self.threads - 128 if self.spc
                     else self.threads) // 128)
        self.n_sub = (self.bm // self.epi_n) // self.nwg
        # nwg need not divide the sub-tile count.  When it does not, the
        # sub-tiles are dealt round-robin and wrapped with a modulo rather
        # than given a tail branch, exactly as spc has always done over 3
        # warpgroups: the last turn of the high warpgroups redoes a sub-tile
        # a low warpgroup already did.  That is idempotent -- same TMEM in,
        # same SMEM out, same value written -- and free, because the low
        # warpgroups take that turn anyway.  It is what lets threads=768
        # (nwg=6) run at all: 16 M sub-tiles over 6 warpgroups.
        self.ilv = self.spc or (
            not self.wgs and (self.bm // self.epi_n) % self.nwg != 0)
        self.n_turn = self.n_sub + 1 if self.ilv else self.n_sub
        # One atom per issue point, same dominance rule as mma_kv_seg.
        self.mma_s_o = mk(K, K, (self.bn, self.bm)) if self.ovl else None
        self.mma_kv_o = (mk(K, MN, (self.bn, self.d))
                         if self.ovl else None)
        # Separate atoms so the in-softmax issue never mutates the one [F]
        # owns; ACCUMULATE is pinned True once outside the m-loop and these
        # are then only read, which keeps them out of the dominance rule.
        self.mma_inc = tuple(mk(K, MN, (self.bn, self.d))
                             for _ in range(6 if self.inc else 0))
        # The hoist needs every sub-tile resident before its fence, so the
        # staging buffer must span all of them; it also issues from a single
        # point outside sfx, which kseg/tpass/ws would each duplicate.
        assert not self.hst or (not self.kseg and not self.tpass
                                and not self.ws), \
            "hst needs kseg/tpass/ws off"
        # The ds-pass must not read slot A back, or GEMM1' would race it.
        # ovl only pays under spc: the issue has to come from a warp that
        # owes the softmax nothing, or the barrier closing the pass waits
        # on the tcgen05 back-pressure anyway -- which is exactly what the
        # worker-issued forms measured (+183us one-shot, +250us split into
        # one two-UMMA burst per ds-pass sub-tile).
        # ovl=1 issues from a dedicated warpgroup (spc).
        assert not self.ovl or (self.tpass == 2
                                and not self.kseg and not self.ws
                                and not self.hst), \
            "ovl needs tpass=2, with kseg/ws/hst off"
        assert self.ovl != 1 or self.spc, "ovl=1 needs spc"
        assert not self.g1b or self.ovl == 1, "g1b needs ovl=1"
        # 16 sub-tiles over 3 warpgroups is dealt round-robin and wrapped
        # with a modulo rather than given a tail branch: warpgroups 1 and
        # 2 redo sub-tiles 0 and 1 on their sixth turn.  That is
        # idempotent -- same TMEM in, same SMEM out, same value written --
        # and free, because warpgroup 0 takes six turns either way.
        # threads=640 adds the issuer warpgroup instead of carving it out of
        # the 512, so the softmax keeps all 16 warps.  That costs registers
        # -- 65536/640 = 102 uniform -- which is what `rlo` buys back: the
        # issuer owns almost none and hands the rest to the workers.
        assert not self.spc or (self.nwg in (3, 4)
                                and self.threads == (512 if self.nwg == 3
                                                     else 640)
                                and not self.ws and not self.kseg
                                and not self.hst), \
            "spc needs threads=512 (nwg 3) or 640 (nwg 4), ws/kseg/hst off"
        # Asymmetric register allocation.  The issuer warpgroup holds `rlo`
        # and the four worker warpgroups split what is left, rounded down to
        # the 8-register granule; with rlo=24 that is 120 each, against the
        # 128 a 512-thread launch gets and the 102 a uniform 640 would.
        assert not self.rlo or (self.spc and self.nwg == 4), \
            "rlo needs spc with threads=640"
        self.rhi = ((65536 - 128 * self.rlo) // 512) // 8 * 8 if self.rlo else 0
        assert not self.preg or (self.tpass == 2 and self.spc), \
            "preg needs tpass=2 and spc"
        # The deal has to be the contiguous one and each warpgroup has to own
        # exactly two K blocks, which pins n_sub to 4.
        assert not self.inc or (self.tpass == 0 and self.n_sub == 4
                                and not self.spc and not self.kseg
                                and not self.ws and not self.hst
                                and not self.ovl), \
            "inc needs tpass=0 and n_sub=4 with spc/kseg/ws/hst/ovl off"
        # The skew reindexes the sub-tiles by hand, so every other scheme that
        # owns that mapping has to be off, and the remainder has to divide
        # evenly over the warpgroups that are not the issuer's.
        assert not self.wgs or (self.nwg >= 4 and not self.spc
                                and not self.kseg and not self.ws
                                and not self.hst and not self.ovl
                                and not self.inc
                                and (self.bm // self.epi_n - self.wgs)
                                % (self.nwg - 1) == 0), \
            "wgs needs nwg>=4, (16-wgs)%(nwg-1)==0, spc/kseg/ws/hst/ovl/inc off"
        # Sub-tiles per non-issuer warpgroup under the wgs skew.  Unlike the
        # round-robin deal this one is exact -- warpgroup 0 takes wgs of them
        # and the rest take nhi each -- so it covers the 16 sub-tiles with no
        # duplicated turn, and each warpgroup's indices stay one base plus
        # constant offsets, which is what keeps the register working set where
        # the contiguous deal had it.
        self.nhi = ((self.bm // self.epi_n - self.wgs) // (self.nwg - 1)
                    if self.wgs else 0)
        # The epilogues keep every warpgroup: by the time they run the
        # issuer has no UMMA left to queue, so it is a worker again.
        # With threads=640 the issuer warpgroup is an extra one, not a carved
        # out one, so the D columns still divide over four warpgroups; folding
        # it in would make n_sub_d 3, which ilv_d would then have to wrap.
        self.nwg_d = (self.threads // 128
                      if (self.spc and self.nwg == 3) else self.nwg)
        self.n_sub_d = (self.d // self.epi_n) // self.nwg_d
        # Same deal on the D side, and it is safe for the same reason plus
        # one more: every epilogue stages its sub-tiles into SMEM and then
        # has a single warp issue one TMA for the whole tile, so a duplicated
        # sub-tile restages the same bytes and the store-reduce still fires
        # exactly once.  A duplicate would otherwise ADD dQ twice.
        self.ilv_d = (self.d // self.epi_n) % self.nwg_d != 0
        self.n_turn_d = self.n_sub_d + 1 if self.ilv_d else self.n_sub_d
        # A ring deeper than the turn count would make the batch loop
        # iterate zero times and silently drop the whole dQ epilogue.
        self.dqb = min(self.dqb, self.n_turn_d)
        assert self.n_turn_d % self.dqb == 0, "dqb must divide n_turn_d"
        # Both sub-tile counts are floor divisions, so an nwg that does not
        # divide them silently drops the tail sub-tiles and the kernel returns
        # NaN instead of failing.  threads=384 (nwg=3) and threads=640 (nwg=5)
        # both do that, and both were benchmarked as if they were valid before
        # this guard existed: 384 looked 40us *faster* precisely because it
        # skipped a sixteenth of the softmax.
        cov_m = (self.wgs + self.nhi * (self.nwg - 1) if self.wgs
                 else self.nwg * self.n_turn)
        assert (cov_m >= self.bm // self.epi_n
                and self.nwg_d * self.n_turn_d >= self.d // self.epi_n), \
            "threads=%d (nwg=%d) does not cover the %d M / %d D sub-tiles" % (
                self.threads, self.nwg, self.bm // self.epi_n,
                self.d // self.epi_n)
        self.nk_s = self.d // cute.size(self.mma_s.shape_mnk, mode=[2])
        self.nk_kv = self.bm // cute.size(self.mma_kv.shape_mnk, mode=[2])
        self.nk_dq = self.bn // cute.size(self.mma_dq.shape_mnk, mode=[2])
        if cutlass.const_expr(self.nkh):
            # Timing probe: fewer UMMAs, same everything else.  See Cfg.nkh.
            self.nk_s //= self.nkh
            self.nk_kv //= self.nkh
            self.nk_dq //= self.nkh
        self.bytes_md = self.bm * self.d * ab.width // 8   # one Q / dO tile
        self.bytes_nd = self.bn * self.d * ab.width // 8   # one K / V tile
        # lse and delta for one m-tile.  These ride a plain cp.async.bulk
        # rather than a tensor map: the (2, BM) tensor-map box is accepted in
        # isolation but raises an illegal instruction inside this kernel, and a
        # bulk copy has no box or swizzle to get wrong.  What it does demand is
        # a 16B-aligned source, and cu_seqlens puts document starts on arbitrary
        # float boundaries -- so LSED gives each row a four-float slot holding
        # (lse*log2e, delta*sm_scale, _, _).  A tile then starts at element
        # 4*(start+m0), which is 16B-aligned for every start, so one copy per
        # tile suffices and, crucially, the softmax still reads it at a
        # compile-time offset.  Rounding the start down instead would have made
        # every broadcast read of lse/delta runtime-addressed, which measured
        # 64us across the m-loop -- far more than the two spare floats cost.
        self.bytes_ls = 4 * self.bm * 4

    # ------------------------------------------------------------------- host
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mDO: cute.Tensor,
        mLSED: cute.Tensor,
        mDQ: cute.Tensor,
        mDK: cute.Tensor,
        mDV: cute.Tensor,
        mSched: cute.Tensor,
        sm_scale: cutlass.Float32,
        npair_max: cutlass.Int32,
        seqlen_total: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup()
        op = cpasync.CopyBulkTensorTileG2SOp()
        box_md = (self.bm, self.d)
        box_nd = (self.bn, self.d)
        tma_q, gQ = cpasync.make_tiled_tma_atom(op, mQ, self.lay_md, box_md)
        tma_k, gK = cpasync.make_tiled_tma_atom(op, mK, self.lay_nd, box_nd)
        tma_v, gV = cpasync.make_tiled_tma_atom(op, mV, self.lay_nd, box_nd)
        tma_do, gDO = cpasync.make_tiled_tma_atom(op, mDO, self.lay_md, box_md)
        red = (cpasync.CopyBulkTensorTileS2GOp() if self.dqs
               else cpasync.CopyReduceBulkTensorTileS2GOp(
                   cute.ReductionKind.ADD))
        tma_dq, gDQ = cpasync.make_tiled_tma_atom(red, mDQ, self.lay_md, box_md)
        # dK/dV: every (n-tile, head) is owned by exactly one CTA, so a plain
        # store is enough and the host buffers need no memset.  The one hazard
        # is the last n-tile of a document, whose padding rows land on the next
        # document -- that tile takes a predicated per-element global store
        # instead (its total volume is nbatch*H*BN*D*2*2 ~ 13MB).
        # Under mout an n-tile is touched by every m-tile at or below it, so
        # the store must accumulate.  A reduce of the zero padding rows is a
        # no-op, which is also what retires the predicated per-element tail
        # path the plain store needs.
        st = (cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD)
              if self.mout else cpasync.CopyBulkTensorTileS2GOp())
        tma_dk, gDKt = cpasync.make_tiled_tma_atom(st, mDK, self.lay_nd, box_nd)
        tma_dv, gDVt = cpasync.make_tiled_tma_atom(st, mDV, self.lay_nd, box_nd)

        self.kernel(
            tma_q, gQ, tma_k, gK, tma_v, gV, tma_do, gDO, mLSED,
            tma_dq, gDQ, tma_dk, gDKt, tma_dv, gDVt, mDK, mDV,
            mSched, sm_scale, npair_max, seqlen_total, grid_x,
        # minctasm=1 tells ptxas that one resident CTA per SM is all this
        # kernel ever needs -- it asks for 227KB of SMEM, so a second CTA
        # could not be co-resident under any register budget.  Without it
        # ptxas sizes the register file for a hypothetical second CTA, and
        # the sizing is discontinuous in the block size: measured by ncu,
        # 512 threads got 94 registers (exactly what the kernel wants, zero
        # spills) but 544 threads got 56, and the resulting spills cost
        # +824us -- which is what made the first ws=1 measurement look like
        # a structural loss rather than a compiler artefact.
        ).launch(grid=[self.pers if self.pers else grid_x, 1, 1],
                 block=[self.threads, 1, 1], stream=stream,
                 cluster=([self.clu, 1, 1] if self.clu else None),
                 min_blocks_per_mp=2 if self.occ else 1)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        tma_q: cute.CopyAtom, gQ: cute.Tensor,
        tma_k: cute.CopyAtom, gK: cute.Tensor,
        tma_v: cute.CopyAtom, gV: cute.Tensor,
        tma_do: cute.CopyAtom, gDO: cute.Tensor,
        mLSED: cute.Tensor,
        tma_dq: cute.CopyAtom, gDQ: cute.Tensor,
        tma_dk: cute.CopyAtom, gDK: cute.Tensor,
        tma_dv: cute.CopyAtom, gDV: cute.Tensor,
        mDKr: cute.Tensor, mDVr: cute.Tensor,
        mSched: cute.Tensor,
        sm_scale: cutlass.Float32,
        npair_max: cutlass.Int32,
        seqlen_total: cutlass.Int32,
        nwork: cutlass.Int32,
    ):
        # Rebuild the MMA atoms and SMEM layouts inside the kernel region: MLIR
        # region isolation forbids referencing values traced on the host.
        self._setup()
        BM, BN, D = self.bm, self.bn, self.d
        ab, acc = self.ab_dtype, self.acc_dtype

        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 128
        wg = tidx // 128
        if cutlass.const_expr(self.ws or (self.spc and self.nwg == 4)):
            # The issuer warps shadow warpgroup 0 in the dK/dV epilogue: it
            # reads the same TMEM and writes the same values, so the duplicate
            # is idempotent and needs no predicate.  The dQ epilogue excludes
            # them outright via nw_epi, because its store is not idempotent.
            wg = wg % self.nwg
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        nw_work = cutlass.const_expr(self.nwg * 4)
        # Only the softmax is restricted to the worker warpgroups; the
        # epilogues run on every warp the launch has.  Under ws the 17th
        # warp shadows warp 0, so it must stay out of the dQ epilogue.
        nw_epi = cutlass.const_expr(
            nw_work if (self.ws or (self.spc and self.nwg == 4))
            else self.threads // 32)
        mma_warp = cutlass.const_expr(
            nw_work if (self.ws or self.spc) else 0)
        if cutlass.const_expr(self.rlo):
            if warp_idx < nw_work:
                cute.arch.setmaxregister_increase(self.rhi)
            else:
                cute.arch.setmaxregister_decrease(self.rlo)
        # spc gives the TMA its own warp out of the issuer warpgroup: the
        # issuer is back-pressured by the tcgen05 queue for most of the
        # tile and a TMA queued behind it would land just as late.
        tma_warp = cutlass.const_expr(
            1 if self.tw else mma_warp + 1 if self.spc else mma_warp)
        bidx, _, _ = cute.arch.block_idx()
        wstep = self.pers if self.pers else nwork
        # pmaj=2 hands work out dynamically, so a CTA may need more rounds
        # than its even share; the spare rounds are two barriers each.
        nround = (nwork + wstep - 1) // wstep + (4 if self.pmaj == 2 else 0)
        nhead = nwork // npair_max

        @cute.struct
        class SharedStorage:
            bar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: cutlass.Int32
            # Broadcast slot for the pmaj=2 work ticket.  It lives in the
            # struct rather than in its own allocation so that sLSED, whose
            # 128B TMA alignment would otherwise be paid twice, starts at the
            # first 128B boundary with nothing wasted behind it.
            work: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        bar_kv = storage.bar.data_ptr()
        bar_tma = bar_kv + 1
        bar_m1 = bar_kv + 2
        bar_m2 = bar_kv + 3
        bar_do = bar_kv + 4
        bar_m5 = bar_kv + 5
        # mout gives V its own barrier: K and V are refilled at the same
        # point but K lands in the far stage of a double buffer while V
        # overwrites the only slot it has, so their waits are one iteration
        # apart and cannot share a phase.
        bar_v = bar_kv + 6
        # g1b: the p-pass needs slot A and nothing else, but bar_m1 is
        # committed after GEMM3 at the head of the iteration, so waiting on it
        # transitively drains GEMM5/4 of the preceding tile and GEMM3 of this
        # one -- 24 of the 40 UMMAs -- before the p-pass may start.  That is
        # why ovl only ever hid the ds-pass.  bar_g1 is committed inside the
        # ovl block, right after GEMM1' (slot A) and GEMM2 (the last reader of
        # sP, which the p-pass is about to overwrite), so the p-pass waits on
        # exactly its own two dependences and runs underneath GEMM5/4/3.
        bar_g1 = bar_kv + 7
        # Landing pad for lse/delta: BM four-float slots, lse at slot offset 0
        # and delta at 1.  Single buffered: its last reader is the softmax, and
        # the load for the next tile goes out after GEMM2, well past that point.
        sLSED = smem.allocate_tensor(acc, cute.make_layout(4 * self.bm), 128)
        sWork = cute.make_tensor(storage.work.ptr, cute.make_layout(1))
        # Broadcast views: same (BN, BM) shape as S^T, row stride 0.  Partitioned
        # with the TMEM-load thread layout they land element-for-element on top of
        # rS/rDP, so the per-column lse/delta arrive in one vector load each.
        # The pair is already fetched by a single LDS.64: lse and delta are
        # adjacent floats in one slot and the backend merges the two reads.
        # Writing that merge by hand (an autovec_copy into a two-float
        # fragment) emits a bit-identical 39,759,696 shared-load instructions,
        # so the 32 loads per warp-iteration here are one per element and
        # already minimal for this layout.
        bcast = cute.make_layout((BN, BM), stride=(0, 4))
        sLSE_b = cute.make_tensor(sLSED.iterator, bcast)
        sDelta_b = cute.make_tensor(sLSED.iterator + 1, bcast)

        # [k0, nk) so a GEMM can be split along K into two halves that are
        # issued at different points in the iteration.  Splitting along K costs
        # no extra UMMAs: the instruction K is 16 either way.
        def do_gemm(mma, accum, fa, fb, k0, nk, init_acc, st=0, sta=0):
            mma.set(tcgen05.Field.ACCUMULATE, init_acc)
            for kb in cutlass.range_constexpr(k0, nk):
                cute.gemm(mma, accum, fa[(None, None, kb, sta)],
                          fb[(None, None, kb, st)], accum)
                mma.set(tcgen05.Field.ACCUMULATE, True)

        def gemm_k1(mma, accum, fa, fb, kb, st=0):
            # One K block at an index only known at run time: the contiguous
            # deal makes it a function of the warpgroup, so the constexpr
            # range do_gemm walks cannot express it.
            mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(mma, accum, fa[(None, None, kb, 0)],
                      fb[(None, None, kb, st)], accum)

        def alloc(lay):
            return smem.allocate_tensor(ab, lay.outer, 128, swizzle=lay.inner)

        def view(buf, lay):
            return cute.make_tensor(buf.iterator, lay.outer)

        if cutlass.const_expr(self.occ):
            # One 64KB block carries all four input tiles.  lay_b_k_q is the
            # largest of them (two stages of BM x D), so every other view fits
            # inside it and every stage index stays in bounds.
            raw_in = smem.allocate(2 * self.bm * self.d * ab.width // 8, 1024)

            def vin(lay):
                return cute.make_tensor(
                    cute.recast_ptr(raw_in, lay.inner, ab), lay.outer)

            sQ = vin(self.lay_b_k_q)
            sK = vin(self.lay_a_k)
            sV = vin(self.lay_a_k)
            sDO = vin(self.lay_b_k)
        else:
            sK = alloc(self.lay_a_k2 if self.mout else self.lay_a_k)
            sV = alloc(self.lay_a_k)
            sQ = alloc(self.lay_b_k if self.mout else self.lay_b_k_q)
            sDO = alloc(self.lay_b_k)
        # sP and sDS take one 32KB block each (192KB total).  sDQ aliases sP
        # and the dQ staging therefore waits for GEMM2, the last reader of sP.
        # The alias is NOT driven by SMEM capacity: a dead-SMEM probe priced
        # the carveout at +2us for 16KB and +3us for 32KB, so a third block
        # (224KB, under the 227KB cap) would be nearly free.  What it would buy
        # -- releasing the epilogue two MMAs early instead of one -- is what
        # actually costs: see the LDTM/UMMA contention note at the GEMM group.
        half = self.bn * self.bm * ab.width // 8
        raw = smem.allocate(half if self.occ else 2 * half, 1024)
        sP = cute.make_tensor(
            cute.recast_ptr(raw, self.lay_a_p.inner, ab), self.lay_a_p.outer)
        sDS = cute.make_tensor(
            cute.recast_ptr(raw if self.occ else raw + half,
                            self.lay_a_p.inner, ab), self.lay_a_p.outer)
        sDQ = cute.make_tensor(
            cute.recast_ptr(raw, self.lay_md.inner, ab),
            self.lay_md.outer)

        sK_mn = view(sK, self.lay_b_mn2_2 if self.mout else self.lay_b_mn2)
        sQ_mn = view(sQ, self.lay_b_mn if self.mout else self.lay_b_mn_q)
        sDO_mn = view(sDO, self.lay_b_mn)    # B of GEMM2 (N=D, K=BM), MN-major
        sDS_mn = view(sDS, self.lay_a_mn)    # A of GEMM5 (M=BM, K=BN), MN-major
        sK_2d = view(sK, self.lay_nd2 if self.mout else self.lay_nd)
        sV_2d = view(sV, self.lay_nd)
        sQ_2d = view(sQ, self.lay_md if self.mout else self.lay_md_q)
        sDO_2d = view(sDO, self.lay_md)
        sP_2d = view(sP, self.lay_nm)
        sDS_2d = view(sDS, self.lay_nm)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for b in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(bar_kv + b, 1)
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_do)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # TMEM: dV | dK | slot A (sT, later dQ) | slot B (dpT)
        tmem_alloc = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.threads
            ),
            is_two_cta=False,
        )
        assert BM == D, "TMEM slot A/B are sized BM columns; BM must == D"
        pool = tmem_alloc.reserve(256 if cutlass.const_expr(self.occ) else 512)
        if cutlass.const_expr(self.occ):
            # 45.1: the permit, not the columns, is what keeps the second CTA
            # off the SM.  CUTLASS releases it at exit; release it here.
            tmem_alloc.relinquish_alloc_permit()
        lay_kv = self.mma_kv.make_fragment_C(
            self.mma_kv.partition_shape_C((BN, D))).layout
        lay_s = self.mma_s.make_fragment_C(
            self.mma_s.partition_shape_C((BN, BM))).layout
        lay_dq = self.mma_dq.make_fragment_C(
            self.mma_dq.partition_shape_C((BM, D))).layout
        if cutlass.const_expr(self.mout):
            # m-outer: dQ is the resident accumulator and dV/dK are produced
            # fresh every iteration, so they alias the slots S and dP vacate.
            # The softmax reads both S and dP before either GEMM2 or GEMM4 is
            # issued, and the tmem_load fence between them retires those
            # reads, so the WAR edge is already covered by the existing fence.
            # The fourth slot is spare here; phase B parks S(j+1) in it.
            ptr_dq_m = pool.allocate(D, acc)
            ptr_a = pool.allocate(BM, acc)
            ptr_dv = ptr_a
            ptr_b = pool.allocate(BM, acc)
            ptr_dk = ptr_b
            pool.allocate(D, acc)
        elif cutlass.const_expr(self.occ):
            # Two slots instead of four: dK lands on dV and dP on S.  Results
            # are wrong; the UMMA stream is identical.
            ptr_dv = pool.allocate(D, acc)
            ptr_dk = ptr_dv
            ptr_a = pool.allocate(BM, acc)
        else:
            ptr_dv = pool.allocate(D, acc)
            ptr_dk = pool.allocate(D, acc)
            ptr_a = pool.allocate(BM, acc)
        tDV = cute.make_tensor(ptr_dv, lay_kv)
        tDK = cute.make_tensor(ptr_dk, lay_kv)
        tS = cute.make_tensor(ptr_a, lay_s)
        # dQ aliases dP, not S.  Both are dead once the softmax has read
        # them, but parking dQ on slot B frees slot A the moment the
        # softmax finishes, which is what would let the NEXT tile's GEMM1
        # issue while this tile's dQ epilogue still drains slot B.  With dQ
        # on slot A the chain G1 -> softmax -> G5 -> dQ-epi -> G1' is one
        # unbreakable serial dependency per iteration.
        if cutlass.const_expr(self.occ):
            ptr_b = ptr_a
        elif cutlass.const_expr(not self.mout):
            ptr_b = pool.allocate(BM, acc)
        tDP = cute.make_tensor(ptr_b, lay_s)
        tDQ = cute.make_tensor(
            ptr_dq_m if cutlass.const_expr(self.mout) else ptr_b, lay_dq)

        fK = self.mma_s.make_fragment_A(sK)
        fV = self.mma_s.make_fragment_A(sV)
        fQ = self.mma_s.make_fragment_B(sQ)
        fDO = self.mma_s.make_fragment_B(sDO)
        fP = self.mma_kv.make_fragment_A(sP)
        fDS = self.mma_kv.make_fragment_A(sDS)
        fDOmn = self.mma_kv.make_fragment_B(sDO_mn)
        fQmn = self.mma_kv.make_fragment_B(sQ_mn)
        fDSmn = self.mma_dq.make_fragment_A(sDS_mn)
        fKmn = self.mma_dq.make_fragment_B(sK_mn)

        epi = self.epi_tile
        atom_t2r = sm100_utils.get_tmem_load_op(
            (BN, BM, D), utils.LayoutEnum.ROW_MAJOR, ab, acc, epi, False)
        tc_t2r = tcgen05.make_tmem_copy(
            atom_t2r, cute.flat_divide(tS[((None, None), 0, 0)], epi)[
                (None, None, 0, 0)])
        th_t2r = tc_t2r.get_slice(lane)
        atom_r2s = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, ab, acc, tc_t2r)
        tc_r2s = cute.make_tiled_copy_D(atom_r2s, tc_t2r)
        th_r2s = tc_r2s.get_slice(lane)

        # dQ is the one accumulator with BM rows instead of BN, so it needs
        # its own TMEM->reg->SMEM pair (and its own fragment width, which
        # scales with the row count per warp).
        epi_dq = self.epi_tile_dq
        atom_t2r_dq = sm100_utils.get_tmem_load_op(
            (BM, BN, D), utils.LayoutEnum.ROW_MAJOR, ab, acc, epi_dq, False)
        tc_t2r_dq = tcgen05.make_tmem_copy(
            atom_t2r_dq,
            cute.flat_divide(tDQ[((None, None), 0, 0)], epi_dq)[
                (None, None, 0, 0)])
        th_t2r_dq = tc_t2r_dq.get_slice(lane)
        atom_r2s_dq = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, ab, acc, tc_t2r_dq)
        tc_r2s_dq = cute.make_tiled_copy_D(atom_r2s_dq, tc_t2r_dq)
        th_r2s_dq = tc_r2s_dq.get_slice(lane)

        def part_t(t):
            return cute.group_modes(
                th_t2r.partition_S(
                    cute.flat_divide(t[((None, None), 0, 0)], epi)), 3, 5)

        tt_S = part_t(tS)
        tt_DP = part_t(tDP)
        tt_DV = part_t(tDV)
        tt_DK = part_t(tDK)
        tt_DQ = cute.group_modes(
            th_t2r_dq.partition_S(
                cute.flat_divide(tDQ[((None, None), 0, 0)], epi_dq)), 3, 5)

        cid = cute.make_identity_tensor((BN, BM))
        tt_c = cute.group_modes(
            th_t2r.partition_D(cute.flat_divide(cid, epi)), 3, 5)
        frg = tt_c[(None, None, None, 0)].shape
        nfrg = cute.size(frg)
        # Same row coordinates, but tiled over D columns for the dK/dV tail.
        tt_cd = cute.group_modes(
            th_t2r.partition_D(
                cute.flat_divide(cute.make_identity_tensor((BN, D)), epi)),
            3, 5)

        tt_L = cute.group_modes(
            th_t2r.partition_D(cute.flat_divide(sLSE_b, epi)), 3, 5)
        tt_D = cute.group_modes(
            th_t2r.partition_D(cute.flat_divide(sDelta_b, epi)), 3, 5)

        rS = cute.make_rmem_tensor(frg, acc)
        rDP = cute.make_rmem_tensor(frg, acc)
        # Second landing pad for the softmax TMEM double buffer: tile j+1 is
        # loaded while tile j is still being reduced, so the tcgen05.ld
        # latency is covered by real work instead of a long-scoreboard stall.
        # Declared only when the buffer is live: in this DSL a declaration
        # alone pins the live range, so an unconditional pair cannot be
        # switched off by a knob.
        nd = cutlass.const_expr(1 if (self.skip & 65536) else self.dbd)
        # Built in one shot: the DSL tracks a list that grows inside a Python
        # loop as a value whose type changes per iteration and rejects it.
        # S and dP are staged separately.  Only slot A (S) has to be fully
        # drained before the next tile's GEMM1 may overwrite it, so under hst
        # the S ring spans every sub-tile while dP keeps the ordinary depth-nd
        # double buffer.  Hoisting BOTH costs 64 live vectors and spills
        # outright (+1066us measured).
        ns_s = cutlass.const_expr(self.n_sub if self.hst else nd)
        bufs_s = tuple(rS if k == 0 else cute.make_rmem_tensor(frg, acc)
                       for k in range(ns_s))
        bufs_d = tuple(rDP if k == 0 else cute.make_rmem_tensor(frg, acc)
                       for k in range(nd))
        # One landing pad per sub-tile this warpgroup owns.
        bufs_p = tuple(cute.make_rmem_tensor(frg, ab)
                       for _ in range(self.n_sub + 1 if self.preg else 0))
        rP = cute.make_rmem_tensor(frg, ab)
        rDS = cute.make_rmem_tensor(frg, ab)
        rt_P = tc_r2s.retile(rP)
        rt_DS = tc_r2s.retile(rDS)
        ts_P = cute.group_modes(
            th_r2s.partition_D(cute.flat_divide(sP_2d, epi)), 3, 5)
        ts_DS = cute.group_modes(
            th_r2s.partition_D(cute.flat_divide(sDS_2d, epi)), 3, 5)
        # dK/dV stage through sP/sDS, not through the dead K/V blocks.  Both
        # are (BN, BM), which spans a full (BN, D) tile only while BM == D --
        # hence the assert in __init__.  The point is the drain: staging in
        # K/V forced the store to be drained before the next work item's K/V
        # load could refill those blocks, and that drain sat between two
        # sync_threads with the whole CTA behind it.  In sP/sDS the next
        # reader is instead the next tile's softmax r2s, which the m-loop
        # already drains for sDQ (= sP) at the head of every iteration, so the
        # store now has GEMM1's whole execution to retire under.

        frg_dq = cute.group_modes(
            th_t2r_dq.partition_D(
                cute.flat_divide(cute.make_identity_tensor((BM, D)),
                                 epi_dq)), 3, 5)[
            (None, None, None, 0)].shape
        nfrg_dq = cute.size(frg_dq)
        rDQ = cute.make_rmem_tensor(frg_dq, acc)
        bufs_dq = tuple(
            rDQ if k == 0 else cute.make_rmem_tensor(frg_dq, acc)
            for k in range(self.dqb))
        rDQb = cute.make_rmem_tensor(frg_dq, ab)
        rt_DQ = tc_r2s_dq.retile(rDQb)
        ts_DQ = cute.group_modes(
            th_r2s_dq.partition_D(cute.flat_divide(sDQ, epi_dq)), 3, 5)

        # Named barriers 1/2: the bit27 variant of syncs #2 and #3.
        nbar_mma = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads)
        nbar_dq = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads)
        # Named barrier 3 pairs a worker segment boundary with the issuer warp.
        # Under ws the issuer sits outside the `warp_idx < nw_work` block, so a
        # plain bar.sync at the boundary would be reached by 16 of the 17 warps
        # and deadlock; both sides arrive on this one instead.
        nbar_seg = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads)
        # Named barrier 4: the hoist rendezvous.  Slot A is dead only once all
        # 16 warps have retired their LDTMs, but only warp 0 has to know, so
        # workers arrive and walk on into the arithmetic.
        nbar_hst = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads)

        # kseg walks the sub-tiles round-robin across warpgroups (wg, wg+nwg,
        # ...) instead of giving each warpgroup a contiguous block, so that
        # after round r the columns [0, 32*(r+1)) of sP/sDS are all written --
        # a contiguous prefix of GEMM2/4's K, which is what lets their leading
        # K blocks issue early.  The default order has no such prefix: all four
        # warpgroups finish their blocks at the same time.
        jst = cutlass.const_expr(
            self.nwg if (self.kseg or self.ilv) else 1)
        j0 = (wg if cutlass.const_expr(self.kseg or self.ilv)
              else wg * self.n_sub)
        jst_d = cutlass.const_expr(self.nwg_d if self.ilv_d else 1)
        j0_d = (wg if cutlass.const_expr(self.ilv_d)
                else wg * self.n_sub_d)
        nhi = cutlass.const_expr(self.nhi)
        # Modulus for the round-robin deal; 0 means index straight through.
        jmk = cutlass.const_expr(self.bm // self.epi_n if self.ilv else 0)
        jmk_d = cutlass.const_expr(self.d // self.epi_n if self.ilv_d else 0)

        def jix(t, m):
            return t % m if cutlass.const_expr(m) else t

        # The D-side deal, bound by default argument because the DSL forbids
        # a staged branch from closing over an enclosing-scope name.
        def jdx(t, base=j0_d, s=jst_d, m=jmk_d, jx=jix):
            return jx(base + t * s, m)
        # Probe bit11 keeps the segment boundaries (sync + fence) but issues
        # no UMMA inside them, so [F] must issue the full K range again.  That
        # isolates the cost of the boundary itself from the cost of warp 0
        # being back-pressured by the tcgen05 queue while it issues.
        kdone = cutlass.const_expr(
            0 if (self.skip & 2048) else
            (self.nk_kv // self.kseg) * (self.kseg - 1) if self.kseg else 0)
        qk_scale = sm_scale * LOG2E

        def take_ticket(sW=sWork, ctr=mSched.iterator + npair_max * 4):
            # Unconditional: a ticket past the end of the work list is simply
            # skipped, and the handful of spare rounds is not worth a guard.
            with cute.arch.elect_one():
                sW[0] = cute.arch.atomic_add(
                    ctr, cutlass.Int32(1), sem="relaxed", scope="gpu")

        # Persistent walk over the work list.  The TMEM allocation, the
        # mbarrier init and the TMA descriptor prefetch above are paid once
        # per CTA instead of once per (n-tile, head); with pers == 0 the
        # stride equals the work count, so every CTA runs exactly one item
        # and the launch is the original non-persistent one.
        for r in cutlass.range(nround, unroll=1):
            # Boustrophedon: reverse the CTA order on odd rounds.  Combined
            # with the pair-major index below -- tile cost is non-increasing
            # in `pair`, so pair-major walks the work list in globally
            # descending cost -- this pairs each CTA's expensive tile with a
            # cheap one.  A plain stride leaves a 13% tail (592 vs 524 average
            # m-tiles per CTA); this leaves 0.7%.
            if cutlass.const_expr(self.pmaj == 2):
                # One global ticket per work item.  Work is handed out in
                # ascending order, so the CTAs in flight always cover a narrow
                # window of the head-major list and keep its L2 reuse, while
                # the hand-out itself absorbs the 13% tail a static stride
                # leaves.  Tickets past the end of the list find slen == 0 and
                # exhausted, so the spare rounds below are simply skipped.
                cute.arch.sync_threads()
                if warp_idx == 0:
                    take_ticket()
                cute.arch.sync_threads()
                w = sWork[0]
            elif cutlass.const_expr(self.pmaj == 1):
                w = r * wstep + (bidx if r % 2 == 0 else wstep - 1 - bidx)
            else:
                w = r * wstep + bidx
            ok = w < nwork
            w = w if ok else cutlass.Int32(0)
            if cutlass.const_expr(self.pmaj == 1):
                pair = w // nhead
                head = w - pair * nhead
            else:
                pair = w % npair_max
                head = w // npair_max
            slen = mSched[pair * 4 + 1]
            slen = slen if ok else cutlass.Int32(0)
            if slen > 0:
                start = mSched[pair * 4 + 0]
                # Slot 2 is the index of the tile this work item OWNS: an
                # n-tile normally, an m-tile under mout.  The loop variable
                # below walks the other axis in both cases, so the pair
                # (o_idx, o0) is the resident one and (n_idx, n0) / (i, m0)
                # are derived from it.
                o_idx = mSched[pair * 4 + 2]
                n_idx = o_idx if cutlass.const_expr(not self.mout) \
                    else cutlass.Int32(0)
                n0 = n_idx * BN
                hd = head * D

                def tma_src(g, rows):
                    gt = cute.local_tile(
                        cute.domain_offset((start, hd), g), (rows, D), (None, 0)
                    )
                    return cute.group_modes(gt, 0, 2)

                one = cute.make_layout(1)
                # dK/dV stage through the K/V blocks: those are the only (BN, D)
                # blocks in SMEM once BM < BN, and both are dead by the epilogue.
                tKs, tKg = cpasync.tma_partition(
                    tma_k, 0, one, cute.group_modes(sK_2d, 0, 2), tma_src(gK, BN))
                tVs, tVg = cpasync.tma_partition(
                    tma_v, 0, one, cute.group_modes(sV_2d, 0, 2), tma_src(gV, BN))
                tQs, tQg = cpasync.tma_partition(
                    tma_q, 0, one, cute.group_modes(sQ_2d, 0, 2), tma_src(gQ, BM))
                tDOs, tDOg = cpasync.tma_partition(
                    tma_do, 0, one, cute.group_modes(sDO_2d, 0, 2), tma_src(gDO, BM))
                # lse/delta: one bulk copy of BM four-float slots.
                lay_ls1 = cute.make_layout(4 * BM)
                atom_ls = cute.make_copy_atom(
                    cpasync.CopyBulkG2SOp(), acc, num_bits_per_copy=4 * BM * 32)

                def load_ls(m, atom=atom_ls, mL=mLSED, lay=lay_ls1, s=sLSED,
                            row=head * mLSED.shape[1], bar=bar_do):
                    """Stage one m-tile of lse/delta. Caller must hold elect_one:
                    unlike a TMA atom, the bulk atom elects no lane itself."""
                    cute.copy(atom,
                              cute.make_tensor(mL.iterator + row + 4 * m, lay),
                              s, mbar_ptr=bar)
                tDQs, tDQg = cpasync.tma_partition(
                    tma_dq, 0, one, cute.group_modes(sDQ, 0, 2), tma_src(gDQ, BM))
                tDKs, tDKg = cpasync.tma_partition(
                    tma_dk, 0, one, cute.group_modes(sP_2d, 0, 2), tma_src(gDK, BN))
                tDVs, tDVg = cpasync.tma_partition(
                    tma_dv, 0, one, cute.group_modes(sDS_2d, 0, 2), tma_src(gDV, BN))

                # NOTE: cute.copy on a TMA atom performs its own single-thread
                # election; nesting it inside elect_one() deadlocks the CTA.
                def load_kv(j, st=0, bar=bar_kv, bv=bar_v, tk=tma_k,
                            tv=tma_v, kg=tKg, ks=tKs, vg=tVg, vs=tVs,
                            nb=self.bytes_nd, mo=self.mout):
                    """Stage K/V tile j.  Caller must be the mma warp.

                    Under mout K goes to stage st of a double buffer and V
                    to its single slot, so the two waits fall an iteration
                    apart and each needs its own barrier.

                    Everything is bound as a default argument: the DSL forbids
                    closing over enclosing-scope names from inside a staged
                    branch, the same rule the softmax's sfx follows."""
                    if cutlass.const_expr(mo):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(bar, nb)
                        cute.copy(tk, kg[(None, j)], ks[(None, st)],
                                  tma_bar_ptr=bar)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(bv, nb)
                        cute.copy(tv, vg[(None, j)], vs, tma_bar_ptr=bv)
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(bar, 2 * nb)
                        cute.copy(tk, kg[(None, j)], ks, tma_bar_ptr=bar)
                        cute.copy(tv, vg[(None, j)], vs, tma_bar_ptr=bar)

                if cutlass.const_expr(not (self.skip & 64)):
                    # Under mout K/V are the inner operands, so the load moves
                    # into the loop and the prologue issues only tile 0.
                    if warp_idx == mma_warp:
                        load_kv(cutlass.Int32(0) if cutlass.const_expr(self.mout)
                                else n_idx)

                # Global (BN, D) windows on dK/dV for the predicated tail store.
                def part_g(m):
                    gt = cute.local_tile(
                        cute.domain_offset((start + n0, hd), m), (BN, D), (0, 0))
                    return cute.group_modes(
                        th_t2r.partition_D(cute.flat_divide(gt, epi)), 3, 5)

                tg_DK = part_g(mDKr)
                tg_DV = part_g(mDVr)

                if cutlass.const_expr(self.mout):
                    # m-outer: the work item owns m-tile o_idx and walks the
                    # n-tiles it can see.  Causality lets m-tile o reach every
                    # n-tile whose first row is at or below o's last row.
                    m0_o = o_idx * BM
                    i0 = cutlass.Int32(0)
                    nblk_n = (slen + BN - 1) // BN
                    cap_n = (m0_o + BM - 1) // BN + 1
                    nblk_m = nblk_n if nblk_n < cap_n else cap_n
                else:
                    m0_o = cutlass.Int32(0)
                    nblk_m = (slen + BM - 1) // BM
                    # First m-tile this n-tile can contribute to.  Causality
                    # needs m_global >= n_global, so the m-loop starts at the
                    # tile holding row n0 -- n_idx only when BM == BN.
                    i0 = n0 // BM
                if cutlass.const_expr(self.iters):
                    cap = i0 + self.iters
                    nblk_m = nblk_m if nblk_m < cap else cap
                # dO is prefetched one iteration ahead: the load for tile i+1 is
                # issued right after GEMM2, its last reader, retires, so it
                # overlaps the dQ epilogue instead of stalling the next MMA.  Q is
                # double buffered and refilled a full iteration ahead instead --
                # see the top of the m-loop.
                # Under mout Q/dO/lse belong to the owned m-tile: one load per
                # work item, and no in-loop refill at all.
                mtile0 = o_idx if cutlass.const_expr(self.mout) else i0
                if warp_idx == mma_warp:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_tma, self.bytes_md)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_do, self.bytes_md + self.bytes_ls
                            if not (self.skip & 262144) else self.bytes_md)
                        if cutlass.const_expr(not (self.skip & 262144)):
                            load_ls(start + mtile0 * BM)
                    cute.copy(tma_q, tQg[(None, mtile0)],
                              tQs if cutlass.const_expr(self.mout)
                              else tQs[(None, 0)], tma_bar_ptr=bar_tma)
                    cute.copy(tma_do, tDOg[(None, mtile0)], tDOs,
                              tma_bar_ptr=bar_do)
                # K/V and Q/dO are two independent 64 KB loads, so K/V is waited
                # on only after Q/dO has been issued and the two fly together.
                # Waiting first serialised them, and the first GEMM1 below needs
                # both anyway.
                if cutlass.const_expr(not (self.skip & 64)
                                      and not self.mout):
                    cute.arch.mbarrier_wait(bar_kv, 0)
                if cutlass.const_expr(self.fsc & 1):
                    # One pass over the 32 KB V tile, once per work item, buys
                    # back 32 FFMA per lane on every one of the ~18.6 m-tiles
                    # that follow.  Lane indexes d and the warpgroup strides
                    # over n, so each warp touches 32 contiguous bf16 -- the
                    # access is coalesced and conflict-free.
                    for e in cutlass.range_constexpr(self.bn * 128
                                                     // self.threads):
                        vn = e * (self.threads // 128) + wg
                        sV_2d[(vn, lane)] = (
                            sV_2d[(vn, lane)].to(acc) * sm_scale).to(ab)
                    cute.arch.sync_threads()
                if warp_idx == mma_warp:
                    if cutlass.const_expr((self.g1f or self.hst or self.ovl)
                                          and not (self.skip & 1)):
                        # The first tile has no predecessor to ride on, so its
                        # GEMM1 is issued here; every later one goes out at the
                        # tail of the preceding iteration.
                        cute.arch.mbarrier_wait(bar_tma, 0)
                        do_gemm(self.mma_s, tS, fK, fQ, 0, self.nk_s,
                                False, st=0)
                        if cutlass.const_expr(self.g1b):
                            with cute.arch.elect_one():
                                tcgen05.commit(bar_g1)

                for i in cutlass.range(i0, nblk_m if not (self.skip & 16) else i0,
                                       unroll=self.unr if self.unr else 1):
                    if cutlass.const_expr(self.mout):
                        # The owned m-tile is fixed; the loop walks n.
                        m0 = m0_o
                        n0 = i * BN
                    else:
                        m0 = i * BM
                    ph = (i - i0) % 2

                    if cutlass.const_expr(not (self.skip & 1)):
                        if warp_idx == mma_warp:
                            # GEMM1 needs only K and Q, so it is issued as soon
                            # as Q lands; the dO tile then finishes arriving under
                            # a UMMA that is already running.  Splitting the two
                            # waits this way is worth 18us.  Under g1f this tile's
                            # GEMM1 was already issued one iteration ago and only
                            # GEMM3 is left here.
                            if cutlass.const_expr(self.mout):
                                # K/V are the inner operands now, so the wait
                                # at the head is on them.  Q/dO/lse landed in
                                # the prologue and are not refilled, so their
                                # barriers are consumed once, on the first
                                # iteration, and never flip phase again.
                                if cutlass.const_expr(not (self.skip & 64)):
                                    cute.arch.mbarrier_wait(bar_kv, ph)
                                if i == i0:
                                    if cutlass.const_expr(
                                            not (self.skip & 524288)):
                                        cute.arch.mbarrier_wait(bar_tma, 0)
                                do_gemm(self.mma_s_a, tS, fK, fQ, 0, self.nk_s,
                                        False, st=0, sta=ph)
                            elif cutlass.const_expr(not (self.g1f or self.hst
                                                         or self.ovl)):
                                if cutlass.const_expr(not (self.skip & 524288)):
                                    cute.arch.mbarrier_wait(bar_tma, ph)
                                do_gemm(self.mma_s_a, tS, fK, fQ, 0, self.nk_s,
                                        False, st=ph)
                            # Q for the tile after this one goes into the other
                            # slot right here: GEMM4 of the preceding tile was
                            # its last reader and has retired, so the slot is
                            # already free and the load has a whole iteration
                            # to land.  Under tw the issue moves to the dO site.
                            if cutlass.const_expr(not (self.skip & 524288)
                                                  and not self.tw
                                                  and not self.mout):
                                if i + 1 < nblk_m:
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            bar_tma, self.bytes_md)
                                    cute.copy(tma_q, tQg[(None, i + 1)],
                                              tQs[(None, 1 - ph)],
                                              tma_bar_ptr=bar_tma)
                            if cutlass.const_expr(not (self.skip & 524288)):
                                if cutlass.const_expr(self.mout):
                                    if i == i0:
                                        cute.arch.mbarrier_wait(bar_do, 0)
                                else:
                                    cute.arch.mbarrier_wait(bar_do, ph)
                            if cutlass.const_expr(self.mout
                                                  and not (self.skip & 64)):
                                cute.arch.mbarrier_wait(bar_v, ph)
                            do_gemm(self.mma_s_a, tDP, fV, fDO, 0, self.nk_s,
                                    False)
                            with cute.arch.elect_one():
                                tcgen05.commit(bar_m1)

                    if cutlass.const_expr(not (self.skip & 1)):
                        # Probe bit28: do not drain GEMM1/3 before issuing the
                        # second MMA group, so all five UMMAs form ONE queued
                        # group per iteration instead of two groups separated by
                        # a full pipeline drain.  Only valid together with bit1
                        # (softmax off), which is what stops tS/tDP from being
                        # read while GEMM1/3 are still writing them.  It prices
                        # the per-iteration cost of the group boundary itself.
                        if cutlass.const_expr(not (self.skip & 268435456)):
                            cute.arch.mbarrier_wait(
                                bar_g1 if cutlass.const_expr(self.g1b)
                                else bar_m1, ph)
                        if cutlass.const_expr(self.mout
                                              and not (self.skip & 64)):
                            # bar_m1 has retired GEMM3, V's only reader, so
                            # its single slot is free; K goes to the other
                            # half of its double buffer, which GEMM5 below
                            # is not reading.  Issued here the pair has the
                            # whole softmax plus GEMM5/2/4 to land in --
                            # from the tail of the iteration, where it used
                            # to sit, it had only the epilogue and cost
                            # 519us of exposed TMA.
                            if warp_idx == mma_warp:
                                if i + 1 < nblk_m:
                                    load_kv(i + 1, 1 - ph)
                    # The previous iteration's dQ store still reads sDQ (= sP),
                    # which the softmax below overwrites.  Drained here, a whole
                    # MMA group after it was issued, and published by the barrier
                    # the softmax needs anyway.
                    if cutlass.const_expr(not (self.skip & 67108864)):
                        if warp_idx == tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                    # Replacing this bar.sync with a one-way mbarrier (the tma
                    # warp arrives after the drain, everyone waits) so that each
                    # warp resumes on its own poll instead of leaving in
                    # lockstep measured +14us, and doing the same to syncs #2/#3
                    # via bit27 on top of it measured +80us.  The 208us that
                    # dropping all five waits is worth is therefore NOT a convoy
                    # cost -- it is the real dependency latency.  See 27.2.
                    if cutlass.const_expr(not (self.skip & 1048576)):
                        cute.arch.sync_threads()

                    # The softmax is a single fused pass over S and dP.  Splitting
                    # it -- release GEMM1 early, run the p-pass while GEMM3 is still
                    # in flight, issue GEMM2 in the gap, then run the ds-pass --
                    # measured bit-for-bit the same device time (2598us both ways):
                    # the TMEM read traffic and UMMA do not make progress
                    # concurrently on this part, so there is nothing to win here.
                    # n-outer accumulates dV/dK across the m-loop and writes
                    # dQ fresh each iteration; m-outer does the mirror image.
                    acc_on = i > i0
                    acc_dq = cutlass.Boolean(i > i0) \
                        if cutlass.const_expr(self.mout) else False
                    acc_kv = False if cutlass.const_expr(self.mout) else acc_on
                    if warp_idx < nw_work:
                        if cutlass.const_expr(not (self.skip & 2)):
                            # Only the diagonal tile needs the causal compare and only
                            # the last m-tile (or a partial n-tile) needs the column-
                            # bound compare, so the body is specialised into three
                            # uniform branches.  Row validity is folded into the bounds
                            # -- an out-of-document row gets an empty keep interval --
                            # so it costs one select per tile, not one per element.
                            # sm_scale lives in sDelta and in the dP scaling, so ds is a
                            # single FFMA plus a multiply.
                            okn = (n0 + lane) < slen
                            dgl = (n0 + lane) if okn else cutlass.Int32(1 << 30)
                            cll = slen if okn else cutlass.Int32(0)

                            # Everything is bound as a default argument: the DSL forbids
                            # closing over enclosing-scope names from inside a staged
                            # branch, and these three calls all sit under one.
                            def sfx(causal, coltail, j0, jja, jjb,
                                    ab=ab, acc=acc, skp=self.skip,
                                    nfrg=nfrg, m0=m0,
                                    qk=qk_scale,
                                    sc=(None if self.fsc & 1 else sm_scale),
                                    dgl=dgl, cll=cll,
                                    tc_t2r=tc_t2r, tt_S=tt_S, tt_DP=tt_DP,
                                    tt_L=tt_L, tt_D=tt_D, tt_c=tt_c,
                                    rS=rS, rDP=rDP,
                                    rP=rP, rDS=rDS, rt_P=rt_P, rt_DS=rt_DS,
                                    tc_r2s=tc_r2s,
                                    ts_P=ts_P, ts_DS=ts_DS, nd=nd,
                                    bufs_p=bufs_p, preg=self.preg,
                                    bufs_s=bufs_s, bufs_d=bufs_d, ns_s=ns_s,
                                    tp=self.tpass, jst=jst, hst=self.hst,
                                    jx=jix, jmk=jmk, psel=0):
                                # TMEM double buffer: tile j+1 is loaded while tile j is
                                # still being reduced, so the tcgen05.ld latency is paid
                                # under real work instead of stalling the first FFMA.
                                # A full hoist (one landing pad per tile) was measured
                                # and is slower: 8 extra live vectors push the register
                                # count past what the 512-thread launch can afford.
                                #
                                # What matters here is tcgen05.ld *latency*, not its
                                # instruction count, and this double buffer is what
                                # hides it.  Two attempts to trade count for structure
                                # both lost badly:
                                #   - Splitting the reduction into a p-half (S, lse) and
                                #     a ds-half (dP, delta) drops the peak live vectors
                                #     from 80 to 48 and makes epi_n=16 fit without
                                #     spilling, which halves the TMEM-pipe instructions
                                #     (15.41M -> 7.71M).  It still costs 381us: each
                                #     sub-tile now exposes two serial load latencies
                                #     instead of one prefetched pair.  long_scoreboard
                                #     even falls (9.28 -> 8.00) while the kernel gets
                                #     slower -- that ratio is per *issued instruction*,
                                #     so it is diluted by instruction count and must not
                                #     be optimised directly.
                                #   - Hoisting that split to two loops over all
                                #     sub-tiles keeps p live across the first loop and
                                #     spills outright (0 -> 4.03M local loads, 422M ->
                                #     513M instructions, +487us).
                                # epi_n=16 on its own (single pass) spills for the same
                                # register reason and costs 73us.  epi_n=8 stands.
                                # Indexed by jj, not by the position within the
                                # segment: a kseg segment can start on an odd jja,
                                # and the prefetch below has to land in the same
                                # buffer the loop body will read first.
                                # Extending the double buffer to the lse/delta vectors
                                # as well was measured at 3us (noise) for 16 more live
                                # registers, so only S/dP are staged.
                                dbuf = cutlass.const_expr(nd > 1)
                                # tpass splits the reduction into a p-pass (S -> sP) and
                                # a ds-pass (S, dP -> sDS).  p is not carried between
                                # them: the ds-pass re-reads S from TMEM and
                                # re-evaluates exp2.  That costs one more TMEM read per
                                # sub-tile -- the TMEM pipe sits at 0.84% of peak -- plus
                                # the 69us the whole kernel spends in exp2, and it drops
                                # the peak live vector count instead of raising it.
                                # Keeping p live across the pass boundary is what made
                                # the earlier two-loop attempt spill outright (+487us).
                                # The point of the split is that GEMM2 only needs sP, so
                                # it can be issued at the boundary and then executes
                                # under the ds-pass.
                                # psel=1 runs the p-pass alone, psel=2 the
                                # ds-pass alone, so the caller can put work in
                                # between; psel=0 keeps both back to back.
                                ps0 = cutlass.const_expr(1 if psel == 2 else 0)
                                ps1 = cutlass.const_expr(
                                    1 if psel == 1 else (2 if tp else 1))
                                for ps in cutlass.range_constexpr(ps0, ps1):
                                    do_p = cutlass.const_expr(ps == 0 or not tp)
                                    do_d = cutlass.const_expr(ps == 1 or not tp)
                                    # tpass=1 re-reads S from TMEM in the ds-pass;
                                    # tpass=2 reads the bf16 P back out of sP instead,
                                    # which is the layout the r2s store just wrote, so
                                    # every thread loads exactly its own elements and no
                                    # barrier is needed.
                                    ld_s = cutlass.const_expr(do_p or tp == 1)
                                    # bit25: TIMING PROBE ONLY -- results are wrong.
                                    # Drops the 16 tt_DP tcgen05.ld per tile and
                                    # feeds the dS chain from rS instead, leaving
                                    # exp2, the ds math, the r2s and the STS
                                    # untouched.  Halving the LDTM count in situ
                                    # measures its exposure directly; extrapolating
                                    # from probe_ldtm mixes an instruction-rate
                                    # regime with this loop's latency-bound one.
                                    nldp = cutlass.const_expr(skp & 33554432)
                                    ld_d = cutlass.const_expr(do_d and not nldp)
                                    if cutlass.const_expr(dbuf):
                                        for k in cutlass.range_constexpr(nd - 1):
                                            jf = jja + k
                                            if cutlass.const_expr(jf < jjb):
                                                if cutlass.const_expr(ld_s
                                                                      and not hst):
                                                    cute.copy(
                                                        tc_t2r,
                                                        tt_S[(None, None, None,
                                                              jx(j0 + jf * jst,
                                                                 jmk))],
                                                        bufs_s[jf % ns_s])
                                                if cutlass.const_expr(ld_d):
                                                    cute.copy(
                                                        tc_t2r,
                                                        tt_DP[(None, None, None,
                                                               jx(j0 + jf * jst,
                                                                  jmk))],
                                                        bufs_d[jf % nd])
                                    for jj in cutlass.range_constexpr(jja, jjb):
                                        j = jx(j0 + jj * jst, jmk)
                                        if cutlass.const_expr(dbuf):
                                            rS = bufs_s[jj % ns_s]
                                            rDP = bufs_d[jj % nd]
                                            jn = jj + nd - 1
                                            if cutlass.const_expr(jn < jjb):
                                                if cutlass.const_expr(ld_s
                                                                      and not hst):
                                                    cute.copy(
                                                        tc_t2r,
                                                        tt_S[(None, None, None,
                                                              jx(j0 + jn * jst,
                                                                 jmk))],
                                                        bufs_s[jn % ns_s])
                                                if cutlass.const_expr(ld_d):
                                                    cute.copy(
                                                        tc_t2r,
                                                        tt_DP[(None, None, None,
                                                               jx(j0 + jn * jst,
                                                                  jmk))],
                                                        bufs_d[jn % nd])
                                        else:
                                            if cutlass.const_expr(ld_s):
                                                cute.copy(
                                                    tc_t2r,
                                                    tt_S[(None, None, None, j)], rS)
                                            if cutlass.const_expr(ld_d):
                                                cute.copy(
                                                    tc_t2r,
                                                    tt_DP[(None, None, None, j)],
                                                    rDP)
                                        # All 128 lanes of a warpgroup read the same lse
                                        # and delta values, so these stay as SMEM views
                                        # and are read per element: staging them in
                                        # fragments pinned 2*nfrg registers across the
                                        # whole element loop, which is what kept the
                                        # wider epilogue tiles from fitting.
                                        rPb = (bufs_p[jj]
                                               if cutlass.const_expr(preg)
                                               else rP)
                                        if cutlass.const_expr(not ld_s
                                                              and not preg):
                                            cute.autovec_copy(
                                                ts_P[(None, None, None, j)], rt_P)
                                        # bit17: TIMING PROBE ONLY -- results are wrong.
                                        # Replaces the per-element lse/delta LDS with
                                        # one warp-uniform runtime float, pricing the
                                        # whole lse/delta path at 7us.  This is NOT
                                        # where the stalls are: shared-memory stalls
                                        # land on short_scoreboard, which ncu puts at
                                        # 1.05 of 13.85, while long_scoreboard (8.81,
                                        # 64%%) is tcgen05.ld latency exposure.  See
                                        # DESIGN 36.2; an earlier comment here blamed
                                        # these LDS for long_scoreboard and was wrong.
                                        # bit15: TIMING PROBE ONLY -- results are wrong.
                                        # Drops both per-element scale
                                        # multiplies (`* qk` on S and `* sc` on
                                        # dP), 64 FFMA per lane per iteration.
                                        # Both factors can be folded into an
                                        # operand for real -- qk into K and sc
                                        # into V -- so this prices the whole
                                        # fsc direction before paying for the
                                        # fold.  The fsc=1 SMEM fold measured
                                        # +175us because the fold itself landed
                                        # on the critical path; a pre-kernel
                                        # fold would cost ~34us of DRAM instead,
                                        # which is only worth paying if the
                                        # multiplies are worth more than that.
                                        nsc = cutlass.const_expr(skp & 32768)
                                        nlse = cutlass.const_expr(skp & 131072)
                                        # bit29: TIMING PROBE ONLY -- results are
                                        # wrong.  Drops the dS half of the r2s
                                        # stores.  bit256 was supposed to price the
                                        # stores but dead-codes the whole softmax
                                        # instead -- its xu count falls to the
                                        # skip=2 value, 241536 -- because rP and rDS
                                        # have no other consumer.  This one keeps
                                        # the ds chain alive by folding a negligible
                                        # multiple of it into P, so only the store
                                        # and its cvt disappear.
                                        nstd = cutlass.const_expr(skp & 536870912)
                                        tl = tt_L[(None, None, None, j)]
                                        td = tt_D[(None, None, None, j)]
                                        fz = cutlass.Float32(m0) * 1.0e-9
                                        cj = tt_c[(None, None, None, j)]
                                        for e in cutlass.range_constexpr(nfrg):
                                            if cutlass.const_expr(not ld_s):
                                                p = rPb[e].to(acc)
                                            elif cutlass.const_expr(skp & 4096):
                                                p = (rS[e] if cutlass.const_expr(nsc) else rS[e] * qk) - (fz if cutlass.const_expr(nlse) else tl[e])
                                            else:
                                                p = cute.arch.exp2((rS[e] if cutlass.const_expr(nsc) else rS[e] * qk) - (fz if cutlass.const_expr(nlse) else tl[e]))
                                            if cutlass.const_expr(ld_s
                                                                  and (causal
                                                                       or coltail)):
                                                ml = m0 + cj[e][1]
                                                if cutlass.const_expr(causal
                                                                      and coltail):
                                                    keep = (ml >= dgl) & (ml < cll)
                                                elif cutlass.const_expr(causal):
                                                    keep = ml >= dgl
                                                else:
                                                    keep = ml < cll
                                                p = p if keep else cutlass.Float32(0.0)
                                            # Staging the fragment in fp32 and narrowing
                                            # it with one vector truncf emits exactly
                                            # the same 143,982,368 fma-pipe instructions
                                            # -- the scalar form is already vectorised
                                            # into cvt.rn.bf16x2.f32 by the backend.
                                            if cutlass.const_expr(do_p):
                                                rP[e] = p.to(ab)
                                                if cutlass.const_expr(preg):
                                                    rPb[e] = rP[e]
                                            if cutlass.const_expr(ld_d
                                                                  and not do_d):
                                                rDS[e] = rDP[e].to(ab)
                                            if cutlass.const_expr(do_d):
                                                dp_e = (rS[e] if cutlass.const_expr(nldp) else rDP[e])
                                                if cutlass.const_expr(skp & 8192):
                                                    ds = dp_e
                                                else:
                                                    dpv = (dp_e if cutlass.const_expr(sc is None or nsc) else dp_e * sc)
                                                    ds = p * (dpv - (fz if cutlass.const_expr(nlse) else td[e]))
                                                rDS[e] = ds.to(ab)
                                                if cutlass.const_expr(nstd
                                                                      and do_p):
                                                    rP[e] = (
                                                        p + ds * 1.0e-30).to(ab)
                                        # Probe 256 drops the register->SMEM stores of P
                                        # and dS.  GEMM2/4/5 then read whatever is in
                                        # sP/sDS, which is harmless for timing: UMMA is
                                        # data-independent and bf16 garbage cannot
                                        # trigger a denormal slowdown, and the softmax
                                        # never reads these buffers back.
                                        if cutlass.const_expr(not (skp & 256)):
                                            if cutlass.const_expr(do_p):
                                                cute.copy(tc_r2s, rt_P,
                                                          ts_P[(None, None, None, j)])
                                            if cutlass.const_expr((do_d or ld_d)
                                                                  and not nstd):
                                                cute.copy(tc_r2s, rt_DS,
                                                          ts_DS[(None, None, None, j)])

                            def run_sfx(j0, jja, jjb, psel=0, sfx=sfx, m0=m0,
                                        slen=slen, n0=n0, BM=BM, BN=BN,
                                        skp=self.skip):
                                # A tile straddles the diagonal iff its first row is
                                # still inside the n-range; with BM < BN more than one
                                # m-tile does.
                                # Probe 512 forces every tile down the unmasked branch,
                                # pricing the causal/column compares and their selects.
                                # The ds-pass reads P back out of sP, which the
                                # p-pass already masked, so ds = p*(...) is
                                # already zero where the mask bites and no
                                # predicate is needed.  Collapsing it to ONE
                                # branch is also what keeps each ovl issue atom
                                # to a single control-flow region.
                                if cutlass.const_expr(psel == 2):
                                    sfx(False, False, j0, jja, jjb, psel=psel)
                                elif cutlass.const_expr(skp & 512):
                                    sfx(False, False, j0, jja, jjb, psel=psel)
                                elif m0 < n0 + BN:
                                    sfx(True, True, j0, jja, jjb, psel=psel)
                                elif (m0 + BM > slen) or (n0 + BN > slen):
                                    sfx(False, True, j0, jja, jjb, psel=psel)
                                else:
                                    sfx(False, False, j0, jja, jjb, psel=psel)

                            # Splitting the iteration along BM so that later
                            # halves' GEMM1/3 run under the first half's softmax was
                            # measured at +317us: a tcgen05 UMMA costs the same whether
                            # its N is 64 or 128 (halving N doubled the GEMM1/3 time,
                            # +324us with GEMM2/4/5 off, against their whole 307us
                            # budget), so subdividing a GEMM buys latency hiding at
                            # exactly twice its price.  UMMA time tracks instruction
                            # COUNT, not FLOPs.
                            # Pull every sub-tile of S and dP into registers
                            # first.  After the fence slot A is dead, and the
                            # next tile's GEMM1 -- which writes slot A and
                            # reads only Q and K out of SMEM -- can execute
                            # while this tile's exp2 chain runs.  This is the
                            # only such window: the tcgen05 pipe retires in
                            # issue order and back-pressures the issuing warp,
                            # so moving a UMMA between two points that both sit
                            # OUTSIDE the softmax just relocates the same stall
                            # (g1f=1 and g1f=2, +110us and +168us).  The
                            # rendezvous is one-directional so that the 15
                            # non-issuing warps never wait on the issue.
                            if cutlass.const_expr(self.hst):
                                for k in cutlass.range_constexpr(self.n_sub):
                                    cute.copy(tc_t2r,
                                              tt_S[(None, None, None,
                                                    j0 + k * jst)], bufs_s[k])
                                cute.arch.fence_view_async_tmem_load()
                                if warp_idx == mma_warp:
                                    nbar_hst.arrive_and_wait()
                                    if cutlass.const_expr(not (self.skip & 1)):
                                        if i + 1 < nblk_m:
                                            cute.arch.mbarrier_wait(bar_tma,
                                                                    1 - ph)
                                            do_gemm(self.mma_s_h, tS, fK, fQ,
                                                    0, self.nk_s, False,
                                                    st=1 - ph)
                                else:
                                    nbar_hst.arrive_unaligned()

                            if cutlass.const_expr(self.kseg):
                                # A K-split costs no extra UMMAs -- the instruction
                                # K is 16 either way -- so unlike the nhalf split
                                # above this overlap does not pay for itself twice.
                                per = cutlass.const_expr(self.n_sub // self.kseg)
                                kpr = cutlass.const_expr(self.nk_kv // self.kseg)
                                for r in cutlass.range_constexpr(self.kseg):
                                    run_sfx(j0, r * per, (r + 1) * per)
                                    if cutlass.const_expr(r + 1 < self.kseg):
                                        # Only the SMEM proxy fence belongs
                                        # here.  A segment issues GEMM2/GEMM4,
                                        # which write tDV/tDK and never touch
                                        # tS/tDP, so there is no WAR hazard
                                        # against the softmax's LDTMs -- the
                                        # one at [E] guards GEMM5 overwriting
                                        # slot B, which is a different edge.
                                        # tcgen05.wait::ld here would drain
                                        # every outstanding LDTM at every
                                        # segment boundary and expose the very
                                        # latency the double buffer hides.
                                        cute.arch.fence_proxy("async.shared",
                                                              space="cta")
                                        if cutlass.const_expr(self.ws):
                                            # Signal, do not rendezvous.  The
                                            # issuer is warp 16, outside this
                                            # block, and it is back-pressured
                                            # for the whole segment it issues;
                                            # an arrive_and_wait here would
                                            # block all 16 workers behind that
                                            # back-pressure and destroy the very
                                            # overlap the segment exists to
                                            # create.  Workers arrive and walk
                                            # straight into the next segment's
                                            # softmax; only warp 16 waits.
                                            # Unaligned: 16 of the 17 warps
                                            # reach this instruction.
                                            nbar_seg.arrive_unaligned()
                                        else:
                                            # bit10 prices the barrier alone,
                                            # bit11 the issue alone.
                                            if cutlass.const_expr(
                                                    not (self.skip & 1024)):
                                                cute.arch.sync_threads()
                                            if cutlass.const_expr(
                                                    not (self.skip & 2048)):
                                                if warp_idx == 0:
                                                    do_gemm(
                                                        self.mma_kv_seg[r], tDV,
                                                        fP, fDOmn,
                                                        r * kpr, (r + 1) * kpr,
                                                        acc_on if r == 0
                                                        else True)
                                                    do_gemm(
                                                        self.mma_kv_seg[r], tDK,
                                                        fDS, fQmn,
                                                        r * kpr, (r + 1) * kpr,
                                                        acc_on if r == 0
                                                        else True, st=ph)
                            elif cutlass.const_expr(self.ovl):
                                run_sfx(j0, 0, self.n_turn, 1)
                                # Slot A is dead only once every worker has
                                # retired its LDTMs and sP is complete only
                                # once every worker has stored it, so the
                                # issuer has to see both.  The rendezvous is
                                # one-directional: workers signal and walk
                                # straight into the ds-pass, and only the
                                # issuer waits on the tcgen05 queue.
                                cute.arch.fence_view_async_tmem_load()
                                cute.arch.fence_proxy("async.shared",
                                                      space="cta")
                                nbar_seg.arrive_unaligned()
                                # GEMM3 is the ds-pass's only new dependence,
                                # and it ran under the p-pass just now.
                                if cutlass.const_expr(
                                        self.g1b and not (self.skip & 1)):
                                    cute.arch.mbarrier_wait(bar_m1, ph)
                                run_sfx(j0, 0, self.n_turn, 2)
                            elif cutlass.const_expr(self.spc):
                                # One extra turn covers the sub-tile that
                                # 16 / 3 leaves over.
                                run_sfx(j0, 0, self.n_turn)
                            elif cutlass.const_expr(self.inc):
                                # Two sub-tiles complete one K block of GEMM2
                                # and GEMM4, so issue it here and let it run
                                # under the rest of the softmax.  Only the
                                # warpgroup that wrote those columns takes
                                # part: the fence makes its own stores visible
                                # to the async proxy and the barrier is its own
                                # 128 threads, so nothing crosses warpgroups.
                                # No tmem_load fence -- these GEMMs write
                                # dV/dK and never the slots being read.
                                for r in cutlass.range_constexpr(2):
                                    run_sfx(j0, r * 2, r * 2 + 2)
                                    # Probe 1024 keeps the split but drops the
                                    # rendezvous; probe 2048 keeps both and
                                    # moves the issue back to [F], where the
                                    # UMMA count and the result are unchanged.
                                    # The two together price the split, the
                                    # rendezvous and the placement separately.
                                    if cutlass.const_expr(
                                            not (self.skip & 1024)):
                                        cute.arch.fence_proxy("async.shared",
                                                              space="cta")
                                        cute.arch.barrier(
                                            barrier_id=8 + wg,
                                            number_of_threads=128)
                                    # No guard for the first m-tile: [F]
                                    # re-issues the whole K range there with
                                    # ACCUMULATE off, which overwrites anything
                                    # these put in.  tcgen05 retires UMMAs in
                                    # issue order, so that clearing block is
                                    # the only ordering constraint there is.
                                    if cutlass.const_expr(self.skip & 2048):
                                        pass
                                    elif warp_idx % 4 == 0:
                                        kb = 2 * wg + r
                                        gemm_k1(self.mma_inc[2 * r], tDV,
                                                fP, fDOmn, kb)
                                        gemm_k1(self.mma_inc[2 * r + 1],
                                                tDK, fDS, fQmn, kb, st=ph)
                            elif cutlass.const_expr(self.wgs):
                                # The issuer's own softmax share now runs
                                # inside the tcgen05 queue wait instead of
                                # after it.  The two arms together cover
                                # sub-tiles 0..15 exactly once, so the result
                                # is unchanged; only the arrival order at the
                                # closing barrier moves.
                                if wg == 0:
                                    run_sfx(cutlass.Int32(0), 0, self.wgs)
                                else:
                                    run_sfx(self.wgs + (wg - 1) * nhi, 0, nhi)
                            else:
                                run_sfx(j0, 0, self.n_turn)

                    # ovl's issue lives out here for the same reason the ws
                    # segment issue does: the issuer warpgroup owes the
                    # softmax nothing, so it can sit in the tcgen05 queue
                    # for the whole ds-pass while the 12 workers run it.
                    # GEMM1' writes slot A, which the p-pass has finished
                    # reading, and GEMM2 reads sP, which the p-pass has
                    # finished writing; the ds-pass touches neither.
                    if cutlass.const_expr(self.ovl == 1
                                          and not (self.skip & 2)):
                        if warp_idx == mma_warp:
                            nbar_seg.arrive_and_wait()
                            if cutlass.const_expr(not (self.skip & 1)):
                                if i + 1 < nblk_m:
                                    cute.arch.mbarrier_wait(bar_tma,
                                                            1 - ph)
                                    do_gemm(self.mma_s_o, tS, fK, fQ,
                                            0, self.nk_s, False, st=1 - ph)
                            if cutlass.const_expr(not (self.skip & 4)):
                                do_gemm(self.mma_kv_o, tDV, fP, fDOmn,
                                        0, self.nk_kv, acc_on)
                            # GEMM2 is the last reader of sP, which the next
                            # p-pass overwrites, so it has to sit inside this
                            # barrier alongside GEMM1'.
                            if cutlass.const_expr(self.g1b):
                                with cute.arch.elect_one():
                                    tcgen05.commit(bar_g1)
                        elif warp_idx >= nw_work:
                            nbar_seg.arrive_unaligned()

                    # The segment issues for ws live out here, outside the
                    # worker block, so warp 16 runs them while warps 0-15 are
                    # already inside the NEXT segment's softmax.  That is the
                    # only arrangement that breaks the GEMM1/3 -> softmax ->
                    # GEMM5/2/4 serial chain; see the note at the ws+kseg
                    # assert that used to be in __init__.
                    if cutlass.const_expr(self.ws and self.kseg
                                          and not (self.skip & 2)):
                        if warp_idx == mma_warp:
                            kpr = cutlass.const_expr(self.nk_kv // self.kseg)
                            for r in cutlass.range_constexpr(self.kseg - 1):
                                nbar_seg.arrive_and_wait()
                                do_gemm(self.mma_kv_seg[r], tDV, fP, fDOmn,
                                        r * kpr, (r + 1) * kpr,
                                        acc_on if r == 0 else True)
                                do_gemm(self.mma_kv_seg[r], tDK, fDS, fQmn,
                                        r * kpr, (r + 1) * kpr,
                                        acc_on if r == 0 else True, st=ph)
                    if cutlass.const_expr(not (self.skip & 8388608)):
                        cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(not (self.skip & 16777216)):
                        cute.arch.fence_proxy("async.shared", space="cta")
                    # Knob bit27 turns syncs #2 and #3 into named barriers on
                    # which only warp 0 (the sole UMMA / TMA issuer) waits and the
                    # other 15 warps just arrive -- safe, because the runners-ahead
                    # are re-joined by the bar_m2 mbarrier and by sync #1, but
                    # measured at 2165 vs 2159us, i.e. no better than bar.sync 0.
                    # The three CTA barriers are not instruction-bound; what they
                    # cost is the real dependency wait.
                    if cutlass.const_expr(not (self.skip & 2097152)):
                        if cutlass.const_expr(not (self.skip & 134217728)):
                            cute.arch.sync_threads()
                        elif warp_idx == mma_warp:
                            nbar_mma.arrive_and_wait()
                        else:
                            nbar_mma.arrive()

                    # GEMM5 and GEMM2 go first behind bar_m5 so the dQ epilogue
                    # starts one MMA early -- GEMM4 runs underneath it -- while sDQ
                    # can still alias sP, because GEMM2 is the last reader of sP and
                    # has retired by then.
                    #
                    # Overlapping the epilogue with MORE MMAs is a loss: the LDTM
                    # reads and UMMA contend for the same TMEM ports, so the
                    # kernel time is non-monotonic in the overlap depth.  Measured:
                    #   0 MMAs under the epilogue (wait for GEMM2/4/5)  1855us
                    #   1 MMA   (wait for GEMM5+GEMM2, GEMM4 below)     1764us  <-- here
                    #   2 MMAs  (wait for GEMM5 only, needs its own sDQ block)
                    #                                                  2173us
                    # An overlap depth of 1 is the optimum; do not widen it.
                    # Reordering the three MMAs behind one barrier each, so that
                    # every buffer is released by its own last reader, is also a
                    # loss: GEMM4 first (hurrying the Q refill) costs 20us and
                    # GEMM2 first (hurrying the dO refill) costs 30us.  Both TMAs
                    # are already fully hidden -- see also probe bit19, which makes
                    # the kernel SLOWER by removing them -- so the extra commit and
                    # wait buy nothing and the epilogue's position is all that
                    # matters.
                    if cutlass.const_expr(not (self.skip & 4)):
                        if warp_idx == mma_warp:
                            # g1f=2 queues the next tile's GEMM1 AHEAD of
                            # GEMM5/2/4 instead of behind them.  The tcgen05
                            # pipe retires in issue order, so the position is
                            # the whole point: bar_m1, which the next softmax
                            # waits on, covers GEMM1+GEMM3, and with GEMM1
                            # queued last (g1f=1, or stock at the head of the
                            # next iteration) that wait transitively drains
                            # GEMM5/2/4 as well -- the softmax ends up waiting
                            # on all 40 UMMAs and nothing overlaps.  Issued
                            # first, GEMM1 retires early and GEMM2/4, whose dK
                            # and dV are not read until the end of the n-tile,
                            # drain underneath the next softmax instead.
                            # Slot A is free either way: the softmax read it in
                            # [D] and the tmem_load fence above has retired.
                            if cutlass.const_expr(self.g1f == 2
                                                  and not (self.skip & 1)):
                                if i + 1 < nblk_m:
                                    if cutlass.const_expr(
                                            not (self.skip & 524288)):
                                        cute.arch.mbarrier_wait(bar_tma, 1 - ph)
                                    do_gemm(self.mma_s_f, tS, fK, fQ,
                                            0, self.nk_s, False, st=1 - ph)
                            do_gemm(self.mma_dq, tDQ, fDSmn, fKmn,
                                    0, self.nk_dq, acc_dq,
                                    st=ph if cutlass.const_expr(self.mout)
                                    else 0)
                            if cutlass.const_expr(self.inc
                                                  and (self.skip & 2048)):
                                do_gemm(self.mma_inc[4], tDV, fP, fDOmn,
                                        0, self.nk_kv, acc_on)
                            elif cutlass.const_expr(self.inc):
                                if i == i0:
                                    do_gemm(self.mma_inc[4], tDV, fP, fDOmn,
                                            0, self.nk_kv, False)
                            elif cutlass.const_expr(not self.ovl):
                                do_gemm(self.mma_kv, tDV, fP, fDOmn,
                                        kdone, self.nk_kv,
                                        True if kdone else acc_kv)
                            with cute.arch.elect_one():
                                tcgen05.commit(bar_m5)
                            if cutlass.const_expr(self.inc
                                                  and (self.skip & 2048)):
                                do_gemm(self.mma_inc[5], tDK, fDS, fQmn,
                                        0, self.nk_kv, acc_on, st=ph)
                            elif cutlass.const_expr(self.inc):
                                if i == i0:
                                    do_gemm(self.mma_inc[5], tDK, fDS, fQmn,
                                            0, self.nk_kv, False, st=ph)
                            else:
                                do_gemm(self.mma_kv, tDK, fDS, fQmn,
                                        kdone, self.nk_kv,
                                        True if kdone else acc_kv,
                                        st=0 if cutlass.const_expr(self.mout)
                                        else ph)
                            with cute.arch.elect_one():
                                tcgen05.commit(bar_m2)
                            if cutlass.const_expr(self.g1f == 1
                                                  and not (self.skip & 1)):
                                # Slot A died at the fence above, so the next
                                # tile's GEMM1 can be queued right behind
                                # GEMM5/2/4.  bar_m1 is committed after GEMM3
                                # at the head of that tile and drains this one
                                # too, so the softmax there still sees both.
                                if i + 1 < nblk_m:
                                    if cutlass.const_expr(
                                            not (self.skip & 524288)):
                                        cute.arch.mbarrier_wait(bar_tma, 1 - ph)
                                    do_gemm(self.mma_s_f, tS, fK, fQ,
                                            0, self.nk_s, False, st=1 - ph)
                        # Probe bit30: skip both MMA-completion waits, so the loop
                        # only issues UMMAs and never drains the tcgen05 pipeline.
                        # Wrong results; prices the per-iteration drain latency,
                        # i.e. the ceiling on what cross-iteration pipelining could
                        # recover.  Pair with bit1/bit3/bit19 so nothing races on
                        # the buffers the waits protect.
                        if cutlass.const_expr(not (self.skip & 1073741824)):
                            cute.arch.mbarrier_wait(bar_m5, ph)

                    # dO for the next tile: GEMM2 is its last reader and retired
                    # with bar_m5, so the dO TMA goes out one MMA ahead of the Q
                    # TMA and lands under the dQ epilogue.
                    if warp_idx == tma_warp and cutlass.const_expr(
                            not (self.skip & 524288) and not self.mout):
                        if i + 1 < nblk_m:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bar_do, self.bytes_md + self.bytes_ls
                                    if not (self.skip & 262144)
                                    else self.bytes_md)
                                # The softmax read sLSED before GEMM2 was
                                # issued, so overwriting the single buffer here
                                # is safe.
                                if cutlass.const_expr(not (self.skip & 262144)):
                                    load_ls(start + (i + 1) * BM)
                            cute.copy(tma_do, tDOg[(None, i + 1)], tDOs,
                                      tma_bar_ptr=bar_do)
                            # Q rides along: bar_m5 has already retired GEMM4
                            # of the previous tile, the last reader of the slot
                            # this lands in.
                            if cutlass.const_expr(self.tw):
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        bar_tma, self.bytes_md)
                                cute.copy(tma_q, tQg[(None, i + 1)],
                                          tQs[(None, 1 - ph)],
                                          tma_bar_ptr=bar_tma)

                    # dQ: stage the bf16 tile in its own SMEM block and let one TMA
                    # store-reduce accumulate it into dq.  Rows past the document
                    # end hold zeros, rows past T are clamped by the TMA
                    # descriptor, so no predication is needed.
                    if cutlass.const_expr(not (self.skip & 8) and not self.mout):
                        # Under ws this is the whole point: warp 16 is still
                        # queueing GEMM5/2/4 while these 16 warps drain dQ.
                        if warp_idx < nw_epi:
                            for ib in cutlass.range_constexpr(
                                    self.n_turn_d // self.dqb):
                                jb = cutlass.const_expr(ib * self.dqb)
                                for k in cutlass.range_constexpr(self.dqb):
                                    cute.copy(tc_t2r_dq,
                                              tt_DQ[(None, None, None,
                                                     jdx(jb + k))],
                                              bufs_dq[k])
                                for k in cutlass.range_constexpr(self.dqb):
                                    for e in cutlass.range_constexpr(nfrg_dq):
                                        rDQb[e] = bufs_dq[k][e].to(ab)
                                    cute.copy(tc_r2s_dq, rt_DQ,
                                              ts_DQ[(None, None, None,
                                                     jdx(jb + k))])

                    if cutlass.const_expr(not (self.skip & (4 | 1073741824))):
                        cute.arch.mbarrier_wait(bar_m2, ph)

                    if cutlass.const_expr(self.mout):
                        # m-outer: this is where dK/dV live.  bar_m2 above has
                        # drained GEMM2/GEMM4 (which wrote them) and GEMM5
                        # (the last reader of sK), so the TMEM slots are ready
                        # and K/V can be refilled for the next n-tile.
                        if cutlass.const_expr(not (self.skip & 32)):
                            if warp_idx < nw_epi:
                                for jj in cutlass.range_constexpr(self.n_turn_d):
                                    j = jdx(jj)
                                    cute.copy(tc_t2r,
                                              tt_DK[(None, None, None, j)], rS)
                                    cute.copy(tc_t2r,
                                              tt_DV[(None, None, None, j)], rDP)
                                    for e in cutlass.range_constexpr(nfrg):
                                        rP[e] = rS[e].to(ab)
                                        rDS[e] = rDP[e].to(ab)
                                    cute.copy(tc_r2s, rt_P,
                                              ts_P[(None, None, None, j)])
                                    cute.copy(tc_r2s, rt_DS,
                                              ts_DS[(None, None, None, j)])
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.sync_threads()
                            if warp_idx == tma_warp:
                                # Reduce, not store: an n-tile is written by
                                # every m-tile at or below it.  The padding
                                # rows carry zeros, so no predicated tail.
                                cute.copy(tma_dk, tDKs, tDKg[(None, i)])
                                cute.copy(tma_dv, tDVs, tDVg[(None, i)])
                                cute.arch.cp_async_bulk_commit_group()
                        else:
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.sync_threads()
                    elif cutlass.const_expr(not (self.skip & 8)):
                        if cutlass.const_expr(not (self.skip & 8388608)):
                            cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(not (self.skip & 16777216)):
                            cute.arch.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(not (self.skip & 4194304)):
                            if cutlass.const_expr(not (self.skip & 134217728)):
                                cute.arch.sync_threads()
                            elif warp_idx == mma_warp:
                                nbar_dq.arrive_and_wait()
                            else:
                                nbar_dq.arrive()
                        if warp_idx == tma_warp:
                            if cutlass.const_expr(not (self.skip & 128)):
                                cute.copy(tma_dq, tDQs, tDQg[(None, i)])
                                cute.arch.cp_async_bulk_commit_group()
                    elif cutlass.const_expr(not (self.skip & 4194304)):
                        cute.arch.sync_threads()

                    if cutlass.const_expr(self.g1f == 3
                                          and not (self.skip & 1)):
                        # g1f=3: the next tile's GEMM1 goes out at the very tail
                        # of this iteration, AFTER the dQ epilogue.  g1f=1 puts
                        # it in the same spot but one step earlier, before the
                        # epilogue's LDTMs, so GEMM1 contends with them for the
                        # TMEM ports (+110us); g1f=2 puts it ahead of GEMM5/2/4
                        # and only relocates the head stall (+168us).  Here the
                        # only thing still in flight is the dQ bulk store and
                        # the loop bookkeeping, so GEMM1 runs for free and the
                        # head of the next iteration is down to GEMM3 alone.
                        # Measured +58us, which completes the picture: all four
                        # placements of GEMM1 lose (stock, g1f=1/2/3).  The
                        # issuing warp is back-pressured by the tcgen05 pipe,
                        # and warp 0 is also one of the sixteen softmax warps,
                        # so any issue point OUTSIDE the softmax stalls the
                        # other fifteen warps at the next barrier by exactly
                        # the same amount.  The only cure is to stop making a
                        # softmax warp do the issuing -- see the note at [E].
                        if warp_idx == mma_warp:
                            if i + 1 < nblk_m:
                                if cutlass.const_expr(
                                        not (self.skip & 524288)):
                                    cute.arch.mbarrier_wait(bar_tma, 1 - ph)
                                do_gemm(self.mma_s_f, tS, fK, fQ,
                                        0, self.nk_s, False, st=1 - ph)

                if warp_idx == tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_threads()

                # dK/dV: TMEM -> reg -> bf16 SMEM (reusing sP/sDS) -> one plain TMA
                # store each.  A tile that runs past the document end cannot use the
                # TMA (its padding rows belong to the next document), so it falls
                # back to a per-element predicated global store straight from the
                # registers -- only nbatch*H tiles take that path.
                if cutlass.const_expr(self.mout):
                    # m-outer: dK/dV were reduced away inside the loop; what is
                    # left here is the one dQ tile this work item owns.  It is
                    # written exactly once, so the store-reduce degenerates to
                    # a store and the 141us that 77488 per-iteration reduces
                    # cost becomes 4160 of them.
                    if cutlass.const_expr(not (self.skip & 8)):
                        if warp_idx < nw_epi:
                            for ib in cutlass.range_constexpr(
                                    self.n_turn_d // self.dqb):
                                jb = cutlass.const_expr(ib * self.dqb)
                                for k in cutlass.range_constexpr(self.dqb):
                                    cute.copy(tc_t2r_dq,
                                              tt_DQ[(None, None, None,
                                                     jdx(jb + k))],
                                              bufs_dq[k])
                                for k in cutlass.range_constexpr(self.dqb):
                                    for e in cutlass.range_constexpr(nfrg_dq):
                                        rDQb[e] = bufs_dq[k][e].to(ab)
                                    cute.copy(tc_r2s_dq, rt_DQ,
                                              ts_DQ[(None, None, None,
                                                     jdx(jb + k))])
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_threads()
                        if warp_idx == tma_warp:
                            cute.copy(tma_dq, tDQs, tDQg[(None, o_idx)])
                            cute.arch.cp_async_bulk_commit_group()
                elif cutlass.const_expr(not (self.skip & 32)):
                    if (n0 + BN) <= slen:
                        # Measured: the whole tail is worth 30us against a run
                        # with it removed, and double buffering the TMEM reads
                        # here moved nothing, so it stays a plain loop.
                        for jj in cutlass.range_constexpr(self.n_turn_d):
                            j = jdx(jj)
                            cute.copy(tc_t2r, tt_DK[(None, None, None, j)], rS)
                            cute.copy(tc_t2r, tt_DV[(None, None, None, j)], rDP)
                            for e in cutlass.range_constexpr(nfrg):
                                rP[e] = rS[e].to(ab)
                                rDS[e] = rDP[e].to(ab)
                            cute.copy(tc_r2s, rt_P,
                                      ts_P[(None, None, None, j)])
                            cute.copy(tc_r2s, rt_DS,
                                      ts_DS[(None, None, None, j)])
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_threads()
                        # Same warp that drains the group in the m-loop:
                        # cp.async.bulk groups are per-thread.
                        if warp_idx == tma_warp:
                            cute.copy(tma_dk, tDKs, tDKg[(None, n_idx)])
                            cute.copy(tma_dv, tDVs, tDVg[(None, n_idx)])
                            cute.arch.cp_async_bulk_commit_group()
                    else:
                        for jj in cutlass.range_constexpr(self.n_turn_d):
                            j = jdx(jj)
                            cute.copy(tc_t2r, tt_DK[(None, None, None, j)], rS)
                            cute.copy(tc_t2r, tt_DV[(None, None, None, j)], rDP)
                            cj = tt_cd[(None, None, None, j)]
                            gk = tg_DK[(None, None, None, j)]
                            gv = tg_DV[(None, None, None, j)]
                            for e in cutlass.range_constexpr(nfrg):
                                if (n0 + cj[e][0]) < slen:
                                    gk[e] = rS[e].to(ab)
                                    gv[e] = rDP[e].to(ab)
                        cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_threads()

                # Reset the barrier phases for the next work item: the
                # m-loop parity is (i - i0) % 2 and the trip count varies
                # per tile, so the phases would not line up otherwise.
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        for b in cutlass.range_constexpr(8):
                            cute.arch.mbarrier_init(bar_kv + b, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()

        # No m-loop follows the last work item, so its dK/dV store is drained
        # here instead; SMEM dies with the CTA.
        if warp_idx == tma_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        tmem_alloc.relinquish_alloc_permit()
        tmem_alloc.free(pool.base_ptr)



_cache = {}


def triton_bwd(q, k, v, o, do, lse, cu_seqlens, sm_scale, cfg=Cfg):
    T, H, D = q.shape
    nbatch = cu_seqlens.numel() - 1
    BM, BN = cfg.BM, cfg.BN
    # Every (n-tile, head) of dK/dV is written by exactly one CTA, so no memset
    # -- except under mout, where every m-tile touching an n-tile contributes
    # and the store becomes a TMA reduce, which needs a zero start.
    dkv = (torch.zeros if cfg.mout else torch.empty)(
        2, T, H * D, device=q.device, dtype=q.dtype)
    dk, dv = dkv[0], dkv[1]
    # Four floats per (head, token): (lse*log2e, delta*sm_scale, _, _).  The
    # padding buys 16B-aligned tile starts for every cu_seqlens, which is what
    # lets the main kernel stage both with one bulk copy.  One tile of slack
    # past the end: the last copy still pulls a full tile.
    pad = 4 * BM
    lsed = torch.empty(H * T * 4 + pad, device=q.device, dtype=torch.float32)
    lsed[-pad:].zero_()
    # dQ accumulates in bf16 straight through the TMA store-reduce: no fp32
    # round trip, half the reduce traffic, and no post pass.
    dq = torch.empty(T, H, D, device=q.device, dtype=q.dtype)
    # One work item per tile of the outer axis: n-tiles normally, m-tiles
    # under mout.  BM == BN today, so the bound is the same either way.
    _bo = BM if cfg.mout else BN
    npair_max = (T + _bo - 1) // _bo + nbatch
    # zeros, not empty: _sched_kernel only writes the valid slots, the tail must
    # read back as slen == 0 so those CTAs exit immediately.
    # Four slots past the work list hold the pmaj=2 atomic ticket counter.
    sched = torch.zeros(npair_max * 4 + 4, device=q.device, dtype=torch.int32)
    # Swept: (BT, num_warps) = (64, 8) for the delta+zero pass, (64, 4) for the
    # fp32->bf16 pass.  The default (128, 4) costs the pre pass ~26us extra.
    _pre_kernel[((T + 63) // 64, H)](o, do, lse, lsed, dq, T, sm_scale,
                                     H * D, D, D, 64,
                                     num_warps=8, num_stages=1)
    blk = 1
    while blk < npair_max:
        blk *= 2
    _sched_kernel[(1,)](cu_seqlens, sched, nbatch, npair_max, BN, BM, blk,
                        cfg.mout, num_warps=4)

    grid_x = npair_max * H
    args = (
        *[from_dlpack(x.view(T, H * D), assumed_align=16) for x in (q, k, v, do)],
        from_dlpack(lsed[:H * T * 4].view(H, T * 4), assumed_align=16),
        from_dlpack(dq.view(T, H * D), assumed_align=16),
        from_dlpack(dk, assumed_align=16),
        from_dlpack(dv, assumed_align=16),
        from_dlpack(sched, assumed_align=16),
        cutlass.Float32(sm_scale), cutlass.Int32(npair_max),
        cutlass.Int32(T), cutlass.Int32(grid_x),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    # cfg.pers < 0 asks for one CTA per SM.  The count is a device property;
    # hard-coding 148 would silently mis-size the grid on any other part.
    pers = (torch.cuda.get_device_properties(q.device).multi_processor_count
            if cfg.pers < 0 else cfg.pers)
    key = (T, H, D, BM, BN, cfg.epi_n, cfg.threads, cfg.iters,
           cfg.skip, cfg.tpass, pers, cfg.pmaj, cfg.kseg, cfg.dbd,
           cfg.g1f, cfg.tw, cfg.ws, cfg.hst, cfg.ovl, cfg.spc, cfg.preg, cfg.inc, cfg.unr, cfg.rlo,
           cfg.nkh, cfg.dqs, cfg.wgs, cfg.clu, cfg.fsc, cfg.mout, cfg.dqb,
           cfg.g1b)
    fn = _cache.get(key)
    if fn is None:
        fn = cute.compile(
            BwdKernel(BM, BN, D, cfg.epi_n, cfg.threads, cfg.iters,
                      cfg.skip, cfg.tpass, pers, cfg.pmaj,
                      cfg.kseg, cfg.dbd, cfg.g1f, cfg.tw,
                      cfg.ws, cfg.hst, cfg.ovl, cfg.spc, cfg.preg, cfg.inc, cfg.unr, cfg.rlo,
                      cfg.nkh, cfg.dqs, cfg.wgs, cfg.clu, cfg.fsc,
                      cfg.mout, cfg.dqb, cfg.g1b, cfg.occ), *args)
        _cache[key] = fn
    fn(*args)
    return dq, dk.view(T, H, D), dv.view(T, H, D)
