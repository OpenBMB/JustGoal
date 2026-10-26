"""Range-safe KDA: 16-token chunks, full triangular inverse, FP32 accumulation."""
import torch
import triton
import triton.language as tl

f16 = tl.float16


@triton.jit
def _kda_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, Alog_ptr, dtb_ptr, state_ptr,
    out_ptr, fstate_ptr, L, lower_bound, scale,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    BV: tl.constexpr, NCHUNK: tl.constexpr, NDBL: tl.constexpr, DT: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    offs_c = tl.arange(0, C)
    offs_k = tl.arange(0, K)
    offs_v0 = tl.arange(0, BV)
    offs_v1 = tl.arange(0, BV) + BV

    A_h = tl.exp(tl.load(Alog_ptr + h).to(tl.float32))
    dtb = tl.load(dtb_ptr + h * K + offs_k).to(tl.float32)
    sb = state_ptr + n * (H * V * K) + h * (V * K)
    M0 = tl.load(sb + offs_v0[:, None] * K + offs_k[None, :]).to(DT)
    M1 = tl.load(sb + offs_v1[:, None] * K + offs_k[None, :]).to(DT)

    eye = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)
    ltri = offs_c[:, None] >= offs_c[None, :]
    sltri = offs_c[:, None] > offs_c[None, :]

    base_n = n * L
    for ci in range(NCHUNK):
        rows = base_n + ci * C + offs_c
        q = tl.load(q_ptr + rows[:, None] * (H * K) + h * K + offs_k[None, :]).to(tl.float32)
        k = tl.load(k_ptr + rows[:, None] * (H * K) + h * K + offs_k[None, :]).to(tl.float32)
        g = tl.load(g_ptr + rows[:, None] * (H * K) + h * K + offs_k[None, :]).to(tl.float32)
        beta = tl.sigmoid(tl.load(beta_ptr + rows * H + h).to(tl.float32))
        q = q * tl.rsqrt(tl.sum(q * q, 1)[:, None] + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, 1)[:, None] + 1e-6)
        la = lower_bound * tl.sigmoid(A_h * (g + dtb[None, :]))
        G = tl.cumsum(la, axis=0)
        Aexp = tl.exp(G)
        qd = (q * Aexp).to(DT)
        kd = (k * tl.exp(-G)).to(DT)
        kf = (k * Aexp).to(DT)
        A_last = tl.exp(tl.sum(la, axis=0))
        khd = (kd.to(tl.float32) * A_last[None, :]).to(DT)

        W = tl.dot(kf, tl.trans(kd))                         # fp32 (kd huge)
        Nmat = -(beta[:, None] * W) * sltri
        T = eye + Nmat
        Ni = Nmat
        for _ in tl.range(NDBL):
            Ni = tl.dot(Ni.to(DT), Ni.to(DT), out_dtype=tl.float32).to(tl.float32)
            T = tl.dot(T.to(DT), (eye + Ni).to(DT), out_dtype=tl.float32).to(tl.float32)
        Td = T.to(DT)
        P = ((tl.dot(qd, tl.trans(kd)) * ltri)).to(DT)       # matmul fp32, mask, cast bf16
        Alk = A_last[None, :]

        for vt in tl.range(2):
            offs_v = vt * BV + tl.arange(0, BV)
            Mv = tl.where(vt == 0, M0, M1)
            Mt = tl.trans(Mv)                                 # already bf16
            vv = tl.load(v_ptr + rows[:, None] * (H * V) + h * V + offs_v[None, :]).to(tl.float32)
            R = (beta[:, None] * (vv - tl.dot(kf, Mt).to(tl.float32))).to(DT)
            U = tl.dot(Td, R, out_dtype=tl.float32)
            o = (tl.dot(qd, Mt, out_dtype=tl.float32).to(tl.float32)
                 + tl.dot(P, U.to(DT), out_dtype=tl.float32).to(tl.float32))
            tl.store(out_ptr + rows[:, None] * (H * V) + h * V + offs_v[None, :],
                     o.to(out_ptr.dtype.element_ty))
            Mv = (Mv.to(tl.float32) * Alk + tl.dot(tl.trans(U.to(DT)), khd, out_dtype=tl.float32).to(tl.float32)).to(DT)
            M0 = tl.where(vt == 0, Mv, M0)
            M1 = tl.where(vt == 1, Mv, M1)

    fb = fstate_ptr + n * (H * V * K) + h * (V * K)
    tl.store(fb + offs_v0[:, None] * K + offs_k[None, :], M0.to(fstate_ptr.dtype.element_ty))
    tl.store(fb + offs_v1[:, None] * K + offs_k[None, :], M1.to(fstate_ptr.dtype.element_ty))


def triton_kda(inp, lower_bound=-1.0, scale=None, C=16, BV=64, ndbl=3, dt='bf16',
               num_warps=8, num_stages=1):
    q, k, v, g = inp["q"], inp["k"], inp["v"], inp["g"]
    beta, A_log, dt_bias = inp["beta"], inp["A_log"], inp["dt_bias"]
    state = inp["state"]
    N, L, H, K, V, T = inp["N"], inp["L"], inp["H"], inp["K"], inp["V"], inp["T"]
    if scale is None:
        scale = 1.0 / (K ** 0.5)
    out = torch.empty(1, T, H, V, dtype=torch.bfloat16, device=q.device)
    fstate = torch.empty_like(state)
    dtb = dt_bias.contiguous()
    NCHUNK = L // C
    DT = {'bf16': tl.bfloat16, 'fp32': tl.float32, 'fp16': tl.float16}[dt]
    _kda_fwd_kernel[(N * H,)](
        q, k, v, g, beta, A_log, dtb, state, out, fstate, L, lower_bound, scale,
        H=H, K=K, V=V, C=C, BV=BV, NCHUNK=NCHUNK, NDBL=ndbl, DT=DT,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, fstate


if __name__ == "__main__":
    from common import make_inputs, LOWER_BOUND
    from naive_kda import naive_kda
    from bench import bench
    torch.cuda.set_device(0)
    inp = make_inputs(seed=0)
    o_n, s_n = naive_kda(inp, lower_bound=LOWER_BOUND)
    cfg = dict(C=16, BV=64, ndbl=3, num_warps=8, num_stages=1)
    o_t, s_t = triton_kda(inp, lower_bound=LOWER_BOUND, **cfg)
    om = (o_t.float() - o_n).abs().max().item()
    sm = (s_t.float() - s_n).abs().max().item()
    ok = 'PASS' if om < 4.73e-3 and sm < 6.55e-2 else 'FAIL'
    t = bench(lambda: triton_kda(inp, lower_bound=LOWER_BOUND, **cfg), "triton_kda_best")
    print(f">> BEST {cfg} {ok} out={om:.2e} st={sm:.2e} TIME={t:.1f}us  (ref 407.32us, target <339.4us)")
