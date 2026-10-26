"""Chunked Gated Delta Net prefill (k-last state): Triton kernels for the parallel phases.

Math (row-vector convention, S in R^{K x V}, chunk of C=128 tokens, gamma_i = prod_{j<=i} g_j):
  A[i,j] = beta_i * (gamma_i/gamma_j) * k_i.k_j   (j < i),  T = (I + A)^{-1}
  W = T diag(beta*gamma) K,  U' = T diag(beta) V,  K' = diag(gamma_C/gamma_i) K
  per-chunk affine map of the state:  S_C = M S_0 + B,  M = gamma_C I - K'^T W,  B = K'^T U'
  output:  O = scale * [ diag(gamma) Q S_0 + P (U' - W S_0) ],  P = tril(Q K^T * gamma_i/gamma_j)

Kernel `gdn_intra_kernel` (one program per chunk x v-head): W, U' (bf16), M (fp32, SW128
K-major smem image for the tcgen05 scan kernel), B (bf16, [k/8][v][8] lane layout).
Kernel `gdn_output_kernel` (one program per chunk x v-head): O from the chunk's start state.
The sequential state scan lives in scan_cuda.py.
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

CHUNK = 128
HEAD = 128
NUM_QK_HEADS = 4
NUM_V_HEADS = 8


@triton.jit
def _dot3(a, b):
    """fp32 x fp32 -> fp32 matmul with ~16-bit mantissa accuracy using 3 bf16 tensor-core dots."""
    ah = a.to(tl.bfloat16)
    al = (a - ah.to(tl.float32)).to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(ah, bh) + (tl.dot(ah, bl) + tl.dot(al, bh))


@triton.jit
def _locate_chunk(cu_ptr, c, num_seqs, C: tl.constexpr, NS: tl.constexpr):
    """Map global chunk index c -> (seq, seq_start, seq_end, local_chunk). seq=-1 if c is out of range."""
    sidx = tl.arange(0, NS)
    smask = sidx < num_seqs
    st = tl.load(cu_ptr + sidx, mask=smask, other=0)
    en = tl.load(cu_ptr + sidx + 1, mask=smask, other=0)
    n = tl.where(smask, (en - st + C - 1) // C, 0)
    base = tl.cumsum(n) - n
    hit = (c >= base) & (c < base + n) & smask
    nhit = tl.sum(hit.to(tl.int32))
    seq = tl.sum(tl.where(hit, sidx, 0)) + tl.where(nhit > 0, 0, -1)
    seq_start = tl.sum(tl.where(hit, st, 0))
    seq_end = tl.sum(tl.where(hit, en, 0))
    local = c - tl.sum(tl.where(hit, base, 0))
    return seq, seq_start, seq_end, local


@triton.jit
def _gates(a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, t, valid, h, HV: tl.constexpr):
    """Per-token log-gate (masked to 0 outside the sequence) and beta."""
    a_raw = tl.load(a_ptr + t * HV + h, mask=valid, other=0.0).to(tl.float32)
    b_raw = tl.load(b_ptr + t * HV + h, mask=valid, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h).to(tl.float32)
    x = a_raw + dt_bias
    sp = tl.where(x > 20.0, x, libdevice.log1p(tl.exp(x)))
    glog = tl.where(valid, -tl.exp(A_log) * sp, 0.0)
    beta = 1.0 / (1.0 + tl.exp(-b_raw))
    return glog, beta


@triton.jit
def _dot3b(a, b):
    """batched version of _dot3 for [B, M, K] x [B, K, N]."""
    ah = a.to(tl.bfloat16)
    al = (a - ah.to(tl.float32)).to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(ah, bh) + (tl.dot(ah, bl) + tl.dot(al, bh))


@triton.jit
def _merge_level_scratch(t_ptr, a_ptr, S: tl.constexpr, C: tl.constexpr):
    """Block-triangular inverse level on scratch memory: T holds the inverse of the block-diagonal
    (S x S) part of L = I + A; write T21 = -T22 A21 T11 for every diagonal 2S-block so that T becomes
    the inverse of the block-diagonal (2S x 2S) part. Tiles are addressed as [NB, S, S] 3D loads."""
    NB: tl.constexpr = C // (2 * S)
    bidx = tl.arange(0, NB)[:, None, None]
    ii = tl.arange(0, S)[None, :, None]
    jj = tl.arange(0, S)[None, None, :]
    r0 = bidx * (2 * S)
    T11 = tl.load(t_ptr + (r0 + ii) * C + r0 + jj)
    T22 = tl.load(t_ptr + (r0 + S + ii) * C + r0 + S + jj)
    A21 = tl.load(a_ptr + (r0 + S + ii) * C + r0 + jj)
    T21 = -_dot3b(_dot3b(T22, A21), T11)
    tl.store(t_ptr + (r0 + S + ii) * C + r0 + jj, T21)


@triton.jit
def gdn_intra_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, cu_ptr,
    w_ptr, u_ptr, m_ptr, bt_ptr, ascr_ptr, tscr_ptr,
    num_seqs,
    C: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, NS: tl.constexpr, ABL: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    seq, seq_start, seq_end, local = _locate_chunk(cu_ptr, c, num_seqs, C, NS)
    if seq < 0:
        return
    hq = h // (HV // HQ)
    rows = tl.arange(0, C)
    cols = tl.arange(0, C)
    d = tl.arange(0, D)
    t = seq_start + local * C + rows
    valid = t < seq_end
    a_scr = ascr_ptr + (c * HV + h) * (C * C)
    t_scr = tscr_ptr + (c * HV + h) * (C * C)

    k = tl.load(k_ptr + (t[:, None] * HQ + hq) * D + d[None, :], mask=valid[:, None], other=0.0)
    glog, beta = _gates(a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, t, valid, h, HV)
    gc = tl.cumsum(glog)  # chunk-local inclusive cumsum of log gates
    gC = tl.sum(tl.where(rows == C - 1, gc, 0.0))
    lower = rows[:, None] > cols[None, :]

    # A = strictly-lower(beta_i * exp(g_i - g_j) * k_i.k_j) -> scratch
    kk = tl.dot(k, tl.trans(k))
    A = tl.where(lower, beta[:, None] * tl.exp(tl.where(lower, gc[:, None] - gc[None, :], 0.0)) * kk, 0.0)
    tl.store(a_scr + rows[:, None] * C + cols[None, :], A)
    tl.store(t_scr + rows[:, None] * C + cols[None, :], tl.zeros([C, C], dtype=tl.float32))
    tl.debug_barrier()
    if ABL >= 1:
        tl.store(t_scr + rows[:, None] * C + cols[None, :], tl.where(rows[:, None] == cols[None, :], 1.0, 0.0))
        tl.debug_barrier()
    if ABL == 0 or ABL >= 3:

        # T16: forward substitution on the eight diagonal 16x16 blocks, as a [8,16,16] batch loaded from scratch
        NB16: tl.constexpr = C // 16
        b3 = tl.arange(0, NB16)[:, None, None]
        r16 = tl.arange(0, 16)[None, :, None]
        c16 = tl.arange(0, 16)[None, None, :]
        Ab = tl.load(a_scr + (b3 * 16 + r16) * C + b3 * 16 + c16)  # [NB16, 16, 16]
        Tb = tl.where(r16 == c16, 1.0, 0.0) + tl.zeros([NB16, 16, 16], dtype=tl.float32)
        for i in tl.static_range(1, 16):
            Ai = tl.sum(tl.where(r16 == i, Ab, 0.0), axis=1)  # [NB16, 16] = row i of each block (zero for j >= i)
            contrib = tl.sum(Ai[:, :, None] * Tb, axis=1)  # [NB16, 16]
            newrow = tl.where(c16 == i, 1.0, 0.0) - contrib[:, None, :]
            Tb = tl.where(r16 == i, newrow, Tb)
        if ABL != 4:
            tl.store(t_scr + (b3 * 16 + r16) * C + b3 * 16 + c16, Tb)
        tl.debug_barrier()
        if ABL != 3:
            _merge_level_scratch(t_scr, a_scr, 16, C)
            tl.debug_barrier()
            _merge_level_scratch(t_scr, a_scr, 32, C)
            tl.debug_barrier()
            _merge_level_scratch(t_scr, a_scr, 64, C)
            tl.debug_barrier()

    T16 = tl.load(t_scr + rows[:, None] * C + cols[None, :]).to(tl.bfloat16)

    eg = tl.exp(gc)
    offs = (t[:, None] * HV + h) * D + d[None, :]
    W = tl.dot(T16, (k.to(tl.float32) * (beta * eg)[:, None]).to(tl.bfloat16))  # [C, K]
    tl.store(w_ptr + offs, W.to(tl.bfloat16), mask=valid[:, None])
    kp = (k.to(tl.float32) * tl.exp(gC - gc)[:, None]).to(tl.bfloat16)  # [C, K]
    kpT = tl.trans(kp)  # [K, C]
    Wh = W.to(tl.bfloat16)
    Wl = (W - Wh.to(tl.float32)).to(tl.bfloat16)
    KW = tl.dot(kpT, Wh) + tl.dot(kpT, Wl)  # [K, K], indices (k_out, k_in)
    dk = tl.arange(0, D)
    ko = dk[:, None]
    ki = dk[None, :]
    M = tl.where(ko == ki, tl.exp(gC), 0.0) - KW
    # swizzled SW128 K-major image: byte = (kin//32)*16384 + kout*128 + (((kin%32)//4) ^ (kout%8))*16 + (kin%4)*4
    m_off = (ki // 32) * 4096 + ko * 32 + (((ki % 32) // 4) ^ (ko % 8)) * 4 + (ki % 4)
    if ABL <= 1:
        tl.store(m_ptr + (c * HV + h) * (D * D) + m_off, M)
    else:
        tl.store(m_ptr + (c * HV + h) * (D * D) + ko * D + ki, M)

    v = tl.load(v_ptr + (t[:, None] * HV + h) * D + d[None, :], mask=valid[:, None], other=0.0)
    U = tl.dot(T16, (v.to(tl.float32) * beta[:, None]).to(tl.bfloat16))  # [C, V]
    tl.store(u_ptr + offs, U.to(tl.bfloat16), mask=valid[:, None])
    B = tl.dot(kpT, U.to(tl.bfloat16))  # [K, V]
    b_off = ((ko // 8) * D + ki) * 8 + (ko % 8)  # here ko = k index, ki = v index
    if ABL <= 1:
        tl.store(bt_ptr + (c * HV + h) * (D * D) + b_off, B.to(tl.bfloat16))
    else:
        tl.store(bt_ptr + (c * HV + h) * (D * D) + ko * D + ki, B.to(tl.bfloat16))


@triton.jit
def gdn_output_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, cu_ptr, state_ptr, sc_ptr,
    w_ptr, u_ptr, out_ptr, num_seqs, scale,
    C: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, NS: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    seq, seq_start, seq_end, local = _locate_chunk(cu_ptr, c, num_seqs, C, NS)
    if seq < 0:
        return
    hq = h // (HV // HQ)
    rows = tl.arange(0, C)
    cols = tl.arange(0, C)
    d = tl.arange(0, D)
    dk = tl.arange(0, D)
    t = seq_start + local * C + rows
    valid = t < seq_end

    # chunk start state S [K, V]
    if local == 0:
        # initial state is k-last [V][K] fp32 -> transposed read
        S = tl.load(state_ptr + ((seq * HV + h) * D + d[None, :]) * D + dk[:, None]).to(tl.bfloat16)
    else:
        S = tl.load(sc_ptr + (c * HV + h) * (D * D) + ((dk[:, None] // 8) * D + d[None, :]) * 8 + (dk[:, None] % 8))

    q = tl.load(q_ptr + (t[:, None] * HQ + hq) * D + d[None, :], mask=valid[:, None], other=0.0)
    k = tl.load(k_ptr + (t[:, None] * HQ + hq) * D + d[None, :], mask=valid[:, None], other=0.0)
    glog, _ = _gates(a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, t, valid, h, HV)
    gc = tl.cumsum(glog)
    gdiff = gc[:, None] - gc[None, :]
    causal = rows[:, None] >= cols[None, :]
    decay = tl.exp(tl.where(causal, gdiff, 0.0))
    qk = tl.dot(q, tl.trans(k))
    P = tl.where(causal, decay * qk, 0.0) * scale

    offs = (t[:, None] * HV + h) * D + d[None, :]
    W = tl.load(w_ptr + offs, mask=valid[:, None], other=0.0)
    U = tl.load(u_ptr + offs, mask=valid[:, None], other=0.0)
    WS = tl.dot(W, S)  # [C, V]
    Unew = (U.to(tl.float32) - WS).to(tl.bfloat16)
    qg = (q.to(tl.float32) * (scale * tl.exp(gc))[:, None]).to(tl.bfloat16)
    O = tl.dot(qg, S) + tl.dot(P.to(tl.bfloat16), Unew)
    tl.store(out_ptr + offs, O.to(tl.bfloat16), mask=valid[:, None])


def intra_launch(q, k, v, a, b, A_log, dt_bias, cu_seqlens, W, U, m_tiles, b_tiles, a_scr, t_scr, max_chunks, NS):
    num_seqs = cu_seqlens.shape[0] - 1
    gdn_intra_kernel[(max_chunks, NUM_V_HEADS)](
        q, k, v, a, b, A_log, dt_bias, cu_seqlens, W, U, m_tiles, b_tiles, a_scr, t_scr, num_seqs,
        C=CHUNK, D=HEAD, HQ=NUM_QK_HEADS, HV=NUM_V_HEADS, NS=NS, ABL=int(os.environ.get("GDN_ABL", "0")), num_warps=int(os.environ.get("GDN_NW", "8")),
    )


def output_launch(q, k, a, b, A_log, dt_bias, cu_seqlens, state, state_chunks, W, U, output, scale, max_chunks, NS):
    num_seqs = cu_seqlens.shape[0] - 1
    gdn_output_kernel[(max_chunks, NUM_V_HEADS)](
        q, k, a, b, A_log, dt_bias, cu_seqlens, state, state_chunks, W, U, output, num_seqs, scale,
        C=CHUNK, D=HEAD, HQ=NUM_QK_HEADS, HV=NUM_V_HEADS, NS=NS, num_warps=8,
    )
