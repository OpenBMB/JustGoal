"""Chunked Gated-DeltaNet forward for B200, written in Triton.

Derivation (own, from the recurrence; see gdn_naive.py for the scalar form).
Inside a chunk of C tokens, with S0 the state at chunk entry, G the inclusive
cumulative log-decay and beta the delta-rule gate:

    u_j  = beta_j * (v_j - alpha_j * S_{j-1} k_j)
    (I + M) U = diag(beta) V - diag(beta*exp(G)) K S0^T ,
    M[j,l] = beta_j * exp(G_j - G_l) * (k_j . k_l)   for l < j   (strict lower)
    T      = (I + M)^{-1}                            (unit lower triangular)
    W      = T diag(beta*exp(G)) K ,  Uv = T diag(beta) V ,  U = Uv - W S0^T
    S_next = exp(G_C) S0 + (diag(exp(G_C - G)) U)^T K
    O      = Wattn Uv + (diag(exp(G)) Q*scale - Wattn W) S0^T = Obase + Qeff S0^T
    Wattn[i,j] = exp(G_i - G_j) (q_i . k_j) * scale  for j <= i

`T` comes exactly from the nilpotency of `M`:
    (I+M)^-1 = (I-M)(I+M^2)(I+M^4)...(I+M^{C/2})    -> 2*log2(C) matmuls.

Two kernels:

* `_wy`   - one CTA per (chunk, v-head), fully parallel.  Emits W, Uv, Obase,
  Qeff and exp(G_C).  The row factor exp(G_C - G) is folded into the stored W
  and Uv so the sequential pass needs neither that vector nor an exp2.
* `_scan` - the only sequential part: three matmuls per chunk.  Each row of the
  state along d_v evolves independently of the others, so d_v is sliced across
  SPLIT CTAs to shorten the critical path of the longest sequence.  The state
  is held transposed as [d_k, d_v_slice] so it is always the right-hand dot
  operand and no register tile is transposed through shared memory in the loop.

All exponentials are evaluated in the log2 domain (ex2.approx hardware path).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wy(Q, K, V, AIN, BIN, ALOG, DTB, CU,
        WQ, UV, OUT, GCS,
        N: tl.constexpr, HQK: tl.constexpr, HV: tl.constexpr,
        C: tl.constexpr, D: tl.constexpr, GQ: tl.constexpr,
        SCALE: tl.constexpr, NSQ: tl.constexpr):
    hv = tl.program_id(0)
    cid = tl.program_id(1)
    hq = hv // GQ

    base = 0
    ss = 0
    ee = 0
    ci = 0
    ok = 0
    for i in range(N):
        s = tl.load(CU + i).to(tl.int32)
        e = tl.load(CU + i + 1).to(tl.int32)
        nc = (e - s + C - 1) // C
        hit = (cid >= base) & (cid < base + nc)
        ss = tl.where(hit, s, ss)
        ee = tl.where(hit, e, ee)
        ci = tl.where(hit, cid - base, ci)
        ok = tl.where(hit, 1, ok)
        base = base + nc
    if ok == 0:
        return

    rows = tl.arange(0, C)
    dcol = tl.arange(0, D)
    tt = ss + ci * C + rows
    good = tt < ee
    tts = tl.where(good, tt, 0)

    kmat = tl.load(K + (tts[:, None] * HQK + hq) * D + dcol[None, :],
                   mask=good[:, None], other=0.0)
    av = tl.load(AIN + tts * HV + hv, mask=good, other=0.0).to(tl.float32)
    bv = tl.load(BIN + tts * HV + hv, mask=good, other=0.0).to(tl.float32)
    alog = tl.load(ALOG + hv).to(tl.float32)
    dtb = tl.load(DTB + hv).to(tl.float32)

    x = av + dtb
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    gl = tl.where(good, -tl.exp(alog) * sp * 1.4426950408889634, 0.0)
    G = tl.cumsum(gl, axis=0)
    GC = tl.sum(gl, axis=0)
    beta = tl.where(good, 1.0 / (1.0 + tl.exp(-bv)), 0.0)

    lo = rows[:, None] > rows[None, :]
    M = tl.where(lo, beta[:, None] *
                 tl.exp2(tl.where(lo, G[:, None] - G[None, :], -1.0e30)) *
                 tl.dot(kmat, tl.trans(kmat)), 0.0)

    # T = (I+M)^-1 by nilpotency doubling.  NSQ = log2(C) is exact; a smaller
    # NSQ truncates after M^(2^NSQ - 1), leaving a residual bounded by
    # ||M||^(2^NSQ) which check.py validates against the tolerance.
    T = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0) - M
    Pb = M
    for _ in tl.static_range(NSQ):
        P = tl.dot(Pb, Pb, input_precision="tf32x3")
        Pb = P
        T = T + tl.dot(T, Pb, input_precision="tf32x3")

    Tb = T
    woff = (cid * HV + hv) * 2 * C * D + rows[:, None] * D + dcol[None, :]
    off = (cid * HV + hv) * C * D + rows[:, None] * D + dcol[None, :]
    # keep both tiles in bf16: an fp32 copy alive across the attention block
    # costs 2x32 registers per thread and drops the CTA/SM occupancy to one
    Wm = tl.dot(Tb, ((beta * tl.exp2(G))[:, None] *
                     kmat.to(tl.float32)), input_precision="tf32x3")
    vmat = tl.load(V + (tts[:, None] * HV + hv) * D + dcol[None, :],
                   mask=good[:, None], other=0.0)
    Uv = tl.dot(Tb, beta[:, None] * vmat.to(tl.float32), input_precision="tf32x3")

    qmat = tl.load(Q + (tts[:, None] * HQK + hq) * D + dcol[None, :],
                   mask=good[:, None], other=0.0)
    le = rows[:, None] >= rows[None, :]
    Wat = (tl.where(le, tl.dot(qmat, tl.trans(kmat)) * SCALE, 0.0) *
           tl.exp2(tl.where(le, G[:, None] - G[None, :], -1.0e30)))
    tl.store(OUT + (tts[:, None] * HV + hv) * D + dcol[None, :],
             tl.dot(Wat, Uv, input_precision="tf32x3").to(kmat.dtype), mask=good[:, None])
    Qe = (tl.exp2(G) * SCALE)[:, None] * qmat.to(tl.float32) \
        - tl.dot(Wat, Wm, input_precision="tf32x3")
    tl.store(WQ + woff + C * D, Qe)

    # fold diag(exp(G_C - G)) into the two tiles the sequential pass consumes
    e = tl.exp2(GC - G)[:, None]
    tl.store(WQ + woff, e * Wm)
    tl.store(UV + off, e * Uv)
    tl.store(GCS + cid * HV + hv, tl.exp2(GC))


@triton.jit
def _scan(K, WQ, UV, OUT, GCS, STATE, NEWSTATE, CU,
          N: tl.constexpr, HQK: tl.constexpr, HV: tl.constexpr,
          C: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
          GQ: tl.constexpr, SPLIT: tl.constexpr, MAXC: tl.constexpr):
    pid = tl.program_id(0)
    sp = pid % SPLIT
    hv = (pid // SPLIT) % HV
    n = pid // (SPLIT * HV)
    hq = hv // GQ

    base = 0
    for i in range(N):
        s = tl.load(CU + i).to(tl.int32)
        e = tl.load(CU + i + 1).to(tl.int32)
        base = base + tl.where(i < n, (e - s + C - 1) // C, 0)
    ss = tl.load(CU + n).to(tl.int32)
    ee = tl.load(CU + n + 1).to(tl.int32)
    nc = (ee - ss + C - 1) // C

    rr = sp * R + tl.arange(0, R)
    dcol = tl.arange(0, D)
    rows = tl.arange(0, C)
    ci = tl.arange(0, MAXC)
    # every per-chunk decay scalar for this sequence, hoisted out of the loop
    gv = tl.load(GCS + (base + ci) * HV + hv, mask=ci < nc, other=1.0)

    sbase = ((n * HV + hv) * D + rr[:, None]) * D + dcol[None, :]
    St = tl.trans(tl.load(STATE + sbase))

    for j in range(nc):
        cid = base + j
        wo = (cid * HV + hv) * 2 * C * D + rows[:, None] * D + dcol[None, :]
        Wm = tl.load(WQ + wo)
        Qe = tl.load(WQ + wo + C * D)
        Uvm = tl.load(UV + (cid * HV + hv) * C * D + rows[:, None] * D + rr[None, :])
        tt = ss + j * C + rows
        gd = tt < ee
        tts = tl.where(gd, tt, 0)
        km = tl.load(K + (tts[:, None] * HQK + hq) * D + dcol[None, :],
                     mask=gd[:, None], other=0.0)
        gc = tl.sum(tl.where(ci == j, gv, 0.0))

        Sb = St.to(Wm.dtype)
        optr = OUT + (tts[:, None] * HV + hv) * D + rr[None, :]
        ob = tl.load(optr, mask=gd[:, None], other=0.0)
        tl.store(optr, (ob.to(tl.float32) + tl.dot(Qe, Sb, input_precision="tf32x3")).to(Wm.dtype),
                 mask=gd[:, None])
        Ue = (Uvm.to(tl.float32) - tl.dot(Wm, Sb, input_precision="tf32x3")).to(Wm.dtype)
        St = St * gc + tl.dot(tl.trans(km).to(tl.float32), Ue, input_precision="tf32x3")

    tl.store(NEWSTATE + sbase, tl.trans(St))


class GDN:
    """Pre-allocates workspaces so the timed region contains kernels only."""

    def __init__(self, T, N, HQK, HV, D, C=64, split=8, dtype=torch.bfloat16,
                 scale=None, nsq=None, nw_wy=8, nw_scan=4, st_wy=1, st_scan=3):
        self.T, self.N, self.HQK, self.HV, self.D = T, N, HQK, HV, D
        self.C, self.split = C, split
        self.nch = (T + C - 1) // C + N
        self.maxc = 1 << ((T + C - 1) // C).bit_length()
        # Complete the nilpotent inverse; truncation can destabilize long recurrences.
        self.nsq = max(C.bit_length() - 2, 0 if nsq is None else nsq)
        self.scale = D ** -0.5 if scale is None else scale
        self.nw = (nw_wy, nw_scan)
        self.st = (st_wy, st_scan)
        dev = "cuda"
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dtype)
        self.WQ = torch.zeros(self.nch, HV, 2 * C, D, device=dev, dtype=torch.float32)     # [W ; Qeff] per (chunk, head)
        self.UV = torch.zeros(self.nch, HV, C, D, device=dev, dtype=torch.float32)
        self.OUT = z(T, HV, D)
        self.GCS = torch.zeros(self.nch, HV, device=dev, dtype=torch.float32)
        self.NEWSTATE = torch.zeros(N, HV, D, D, device=dev, dtype=torch.float32)
        self.fused = True
        self.h = {}

    def wy(self, q, k, v, a, b, A_log, dt_bias, state, cu):
        self.h['wy'] = _wy[(self.HV, self.nch)](
            q, k, v, a, b, A_log, dt_bias, cu,
            self.WQ, self.UV, self.OUT, self.GCS,
            N=self.N, HQK=self.HQK, HV=self.HV, C=self.C, D=self.D,
            GQ=self.HV // self.HQK, SCALE=self.scale, NSQ=self.nsq,
            num_warps=self.nw[0], num_stages=self.st[0])

    def scan(self, q, k, v, a, b, A_log, dt_bias, state, cu):
        self.h['scan'] = _scan[(self.N * self.HV * self.split,)](
            k, self.WQ, self.UV, self.OUT, self.GCS,
            state, self.NEWSTATE, cu,
            N=self.N, HQK=self.HQK, HV=self.HV, C=self.C, D=self.D,
            R=self.D // self.split, GQ=self.HV // self.HQK, SPLIT=self.split,
            MAXC=self.maxc, num_warps=self.nw[1], num_stages=self.st[1])

    def attn(self, *a):
        return

    def __call__(self, *args):
        self.wy(*args)
        self.scan(*args)
        return self.OUT, self.NEWSTATE

    def stats(self):
        return {k: (h.n_regs, h.n_spills, h.metadata.shared)
                for k, h in self.h.items()}
