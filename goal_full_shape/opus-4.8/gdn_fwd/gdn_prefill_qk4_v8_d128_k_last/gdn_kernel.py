import math
import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _mm(a, b, MM_BF16: tl.constexpr, PREC: tl.constexpr):
    # bf16 tensor-core matmul (fp32 accumulate) when MM_BF16, else fp32 with input_precision.
    if MM_BF16:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision=PREC)


@triton.jit
def gdn_fused_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, Alog_ptr, a_ptr, dtb_ptr, b_ptr,
    cu_ptr, out_ptr, nstate_ptr,
    scale,
    HQ: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
    C: tl.constexpr, BV: tl.constexpr,
    HAS_STATE: tl.constexpr,
    PREC: tl.constexpr,
    MM_BF16: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_vb = tl.program_id(2)
    hq = pid_h // (HV // HQ)
    v0 = pid_vb * BV

    s0 = tl.load(cu_ptr + pid_s).to(tl.int32)
    s1 = tl.load(cu_ptr + pid_s + 1).to(tl.int32)
    L = s1 - s0

    rk = tl.arange(0, D)          # K dim (128)
    rv = tl.arange(0, BV)         # V slice
    rc = tl.arange(0, C)          # chunk tokens

    if L <= 0:
        # Empty sequence: reference leaves new_state = 0 for this seq. Match it.
        ns_base0 = nstate_ptr + (pid_s * HV + pid_h) * D * D
        ns_ptrs0 = ns_base0 + (v0 + rv)[:, None] * D + rk[None, :]
        tl.store(ns_ptrs0, tl.zeros((BV, D), dtype=tl.float32))
        return

    a_exp = tl.exp(tl.load(Alog_ptr + pid_h))
    dtb = tl.load(dtb_ptr + pid_h)

    # state S[K, BV]; input state layout [N,HV,V,K] -> state[s,h,v,k]; S[k,v]=state[...,v,k]
    if HAS_STATE:
        st_base = state_ptr + (pid_s * HV + pid_h) * D * D
        # load tile [BV(v), D(k)] then transpose to [D(k),BV(v)]
        st_ptrs = st_base + (v0 + rv)[:, None] * D + rk[None, :]
        S = tl.load(st_ptrs).to(tl.float32)      # [BV, D]
        S = tl.trans(S)                          # [D, BV]
    else:
        S = tl.zeros((D, BV), dtype=tl.float32)

    nchunks = (L + C - 1) // C
    for c in range(0, nchunks):
        t0 = s0 + c * C
        cl = tl.minimum(C, s1 - t0)
        mrow = rc < cl                           # valid tokens in chunk

        # load q,k [C,D] (head hq), v [C,BV] (head pid_h)
        tptr = t0 + rc
        q_ptrs = q_ptr + (tptr[:, None] * HQ + hq) * D + rk[None, :]
        k_ptrs = k_ptr + (tptr[:, None] * HQ + hq) * D + rk[None, :]
        v_ptrs = v_ptr + (tptr[:, None] * HV + pid_h) * D + (v0 + rv)[None, :]
        q = tl.load(q_ptrs, mask=mrow[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_ptrs, mask=mrow[:, None], other=0.0).to(tl.float32)
        vv = tl.load(v_ptrs, mask=mrow[:, None], other=0.0).to(tl.float32)

        a_ = tl.load(a_ptr + tptr * HV + pid_h, mask=mrow, other=0.0).to(tl.float32)
        b_ = tl.load(b_ptr + tptr * HV + pid_h, mask=mrow, other=0.0).to(tl.float32)
        x = a_ + dtb
        g_log = -a_exp * _softplus(x)            # [C], <=0
        g_log = tl.where(mrow, g_log, 0.0)
        beta = tl.where(mrow, 1.0 / (1.0 + tl.exp(-b_)), 0.0)   # [C]
        # zero masked keys so they don't affect A
        k = tl.where(mrow[:, None], k, 0.0)

        Lc = tl.cumsum(g_log, axis=0)            # inclusive cumulative log-gate [C]
        L_last = tl.sum(g_log, axis=0)           # total (masked rows are 0)
        expL = tl.exp(Lc)                        # [C]

        # A[i,l] = beta_i (k_i . k_l) exp(Lc_i - Lc_l), strictly lower (l<i)
        KK = _mm(k, tl.trans(k), MM_BF16, PREC)      # [C,C]
        decay = tl.exp(Lc[:, None] - Lc[None, :])              # exp(Lc_i - Lc_l)
        low = rc[:, None] > rc[None, :]
        A = tl.where(low, beta[:, None] * KK * decay, 0.0)     # [C,C]

        # T = (I+A)^{-1} via stable forward substitution (row by row).
        T = tl.where(rc[:, None] == rc[None, :], 1.0, 0.0)   # identity
        for i in tl.static_range(1, C):
            sel = rc == i
            Ai = tl.sum(tl.where(sel[:, None], A, 0.0), axis=0)   # row i of A [C]
            upd = tl.sum(Ai[:, None] * T, axis=0)                 # Ai @ T  [C]
            rowvals = tl.where(sel, 1.0, 0.0) - upd
            T = tl.where(sel[:, None], rowvals[None, :], T)

        # U0 = T @ (beta*V) ; Wk = T @ (beta*expL*k)
        bV = beta[:, None] * vv                  # [C,BV]
        KBG = (beta * expL)[:, None] * k         # [C,D]
        U0 = _mm(T, bV, MM_BF16, PREC)        # [C,BV]
        Wk = _mm(T, KBG, MM_BF16, PREC)       # [C,D]

        # U = U0 - Wk @ S_in
        U = U0 - _mm(Wk, S, MM_BF16, PREC)    # [C,BV]

        # Output: O = scale*( (expL*q) @ S_in + tril(q@k^T * exp(Lc_t-Lc_i)) @ U )
        QG = expL[:, None] * q                   # [C,D]
        Oi = _mm(QG, S, MM_BF16, PREC)        # [C,BV]
        QK = _mm(q, tl.trans(k), MM_BF16, PREC)   # [C,C]
        tri = rc[:, None] >= rc[None, :]
        P2 = tl.where(tri, QK * decay, 0.0)
        Ointra = _mm(P2, U, MM_BF16, PREC)    # [C,BV]
        O = scale * (Oi + Ointra)

        o_ptrs = out_ptr + (tptr[:, None] * HV + pid_h) * D + (v0 + rv)[None, :]
        tl.store(o_ptrs, O.to(out_ptr.dtype.element_ty), mask=mrow[:, None])

        # state update: S = expL_last*S + KD^T@U ; KD_i = exp(L_last-Lc_i)*k_i
        KD = tl.exp(L_last - Lc)[:, None] * k    # [C,D]
        S = tl.exp(L_last) * S + _mm(tl.trans(KD), U, MM_BF16, PREC)   # [D,BV]

    # write new_state [N,HV,V,K]: nstate[s,h,v,k] = S[k,v]
    ns_base = nstate_ptr + (pid_s * HV + pid_h) * D * D
    ns_ptrs = ns_base + (v0 + rv)[:, None] * D + rk[None, :]
    tl.store(ns_ptrs, tl.trans(S))               # [BV, D]


def gdn_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
            C=16, BV=None, ieee=False, num_warps=None, bf16=True):
    T, HQ, D = q.shape
    HV = v.shape[1]
    N = cu_seqlens.numel() - 1
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    # Tuned empirically (tf32): BV=32 (4 V-blocks) maximizes occupancy and is fastest
    # on 99/100 workloads; num_warps=4 throughout (warps=2 is latency-bound/catastrophic).
    if BV is None:
        BV = 32
    if num_warps is None:
        num_warps = 4
    dev = q.device
    out = torch.empty((T, HV, D), dtype=torch.bfloat16, device=dev)
    nstate = torch.empty((N, HV, D, D), dtype=torch.float32, device=dev)
    cu = cu_seqlens.to(torch.int64)
    grid = (N, HV, D // BV)
    gdn_fused_kernel[grid](
        q, k, v, state if state is not None else q, A_log, a, dt_bias, b,
        cu, out, nstate,
        float(scale),
        HQ, HV, D, C, BV,
        state is not None, ("ieee" if ieee else "tf32"), bf16,
        num_warps=num_warps, num_stages=1,
    )
    return out, nstate
