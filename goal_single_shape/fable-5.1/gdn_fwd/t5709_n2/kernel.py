"""Gated Delta Net prefill (chunked, k-last state) for B200.

Pipeline (all on the current stream, CUDA-graph capturable):
  meta  (CUDA)   chunk bookkeeping from cu_seqlens on device (64-token blocks and 128-token chunks)
  prep  (CUDA)   per 64-block: K K^T (mma.sync), gates, cumulative log-decay, T=(I+A)^-1 by fp32 elimination
  intra (CUDA, tcgen05) per 128-chunk item: 2x2 block inverse, W, U, M' = -K^T Wd, B = K^T Ud (hi/lo), Qt, Ot
  seq   (CUDA)   state recurrence h <- g h + M' h + B over 128-chunks; writes bf16 h_c per chunk (Hs tile image)
  out   (CUDA, tcgen05) o = Qt h_c + Ot
  Default: intra + seq + out items run in ONE persistent kernel (work queue, flags); Qt/Ot/Hs are stored as 32 KB
  swizzled tile images so the out items fetch them with TMA bulk copies. Triton intra/out kernels are dev-only fallbacks.
  Chunks are ordered short-sequences-first (their seq items never wait for late intra items); Hs readiness is published
  by an idle warp; the seq stage barriers are initialised once per CTA (phases continue across seq items).
"""
import os, ctypes, subprocess, glob
import torch
import triton
import triton.language as tl

CB = 64      # elimination block
CH = 128     # state chunk
VW = 16
USE_CUDA_INTRA = os.environ.get('GDN_CUDA_INTRA', '1') == '1'
USE_SEQ3 = os.environ.get('GDN_SEQ3', '1') == '1'
USE_CUDA_OUT = os.environ.get('GDN_CUDA_OUT', '1') == '1'
FUSE_SEQ = os.environ.get('GDN_FUSE_SEQ', '1') == '1'
FUSE_OUT = os.environ.get('GDN_FUSE_OUT', '1') == '1'
SEQ_OVER_ITEM = float(os.environ.get('GDN_SEQ_OVER_ITEM', '0'))   # (seq item time) / (intra item time), schedule tuning
_NSM = {}


def _num_sms(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _NSM:
        _NSM[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NSM[idx]
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def _find_nvcc():
    cands = [os.environ.get("NVCC", ""), "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc", "/usr/local/cuda-13/bin/nvcc"]
    cands += glob.glob("/usr/local/cuda*/bin/nvcc")
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME:
            cands.append(os.path.join(CUDA_HOME, "bin", "nvcc"))
    except Exception:
        pass
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return "nvcc"


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    src = os.path.join(_HERE, "gdn_cuda.cu")
    so = os.path.join(_HERE, "libgdn_cuda.so")
    need = (not os.path.exists(so)) or os.path.getmtime(so) < os.path.getmtime(src)
    if not need:
        try:
            _LIB = ctypes.CDLL(so)
        except OSError:
            need = True
    if need:
        tmp = so + f".{os.getpid()}.tmp"
        cmd = [_find_nvcc(), "-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, src]
        subprocess.check_call(cmd)
        os.replace(tmp, so)
        _LIB = ctypes.CDLL(so)
    L = _LIB
    vp, i = ctypes.c_void_p, ctypes.c_int
    L.gdn_meta.argtypes = [vp, i, i, i, vp, vp, vp, vp, vp, vp, vp, i, vp, i, i, vp]; L.gdn_meta.restype = i
    L.gdn_prep.argtypes = [vp, vp, vp, vp, vp, vp, i, i, i, vp, vp, vp, vp]; L.gdn_prep.restype = i
    L.gdn_seq3.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, i, i, vp]; L.gdn_seq3.restype = i
    L.gdn_out.argtypes = [vp, vp, vp, vp, vp, i, i, vp]; L.gdn_out.restype = i
    L.gdn_intra.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, i, i, i, ctypes.c_float, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i, i, vp, vp, i, vp, i, vp]; L.gdn_intra.restype = i
    return L


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


@triton.jit
def intra_kernel(T_ptr, q_ptr, k_ptr, v_ptr, gam_ptr, bet_ptr, cinfo_ptr, cstart64_ptr, cstart128_ptr,
                 Mp_ptr, B_ptr, Qt_ptr, Ot_ptr, gchunk_ptr, scale,
                 HQK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                 CB: tl.constexpr, VW: tl.constexpr, SPLIT_U: tl.constexpr = True, STAGE: tl.constexpr = 3):
    c = tl.program_id(0)
    hv = tl.program_id(1)
    length = tl.load(cinfo_ptr + c * 4 + 2)
    if length <= 0:
        return
    n = tl.load(cinfo_ptr + c * 4 + 0)
    tok0 = tl.load(cinfo_ptr + c * 4 + 1)
    c64 = tl.load(cstart64_ptr + n) + 2 * (c - tl.load(cstart128_ptr + n))
    has2 = length > CB
    hq = hv // (HV // HQK)
    CH: tl.constexpr = 2 * CB
    t = tl.arange(0, CH)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)
    valid = t < length
    lo = t < CB                                   # first 64-block
    K = tl.load(k_ptr + (tok0 + t)[:, None] * (HQK * DK) + hq * DK + dk[None, :], mask=valid[:, None], other=0.0)
    ch1 = c64 * HV + hv
    ch2 = ch1 + HV
    chsel = tl.where(lo, ch1, ch2)
    tt = t % CB
    G = tl.load(gam_ptr + chsel * CB + tt, mask=lo | has2, other=0.0)
    beta = tl.load(bet_ptr + chsel * CB + tt, mask=lo | has2, other=0.0)
    gl1 = tl.sum(tl.where(t == CB - 1, G, 0.0), 0)
    G = tl.where(lo, G, G + gl1)                  # cumulative over the 128-chunk
    gl = tl.sum(tl.where(t == CH - 1, G, 0.0), 0)
    ch = c * HV + hv
    tl.store(gchunk_ptr + ch, tl.exp(gl))
    # block-diagonal T - I (bf16): blocks (0,0) from ch1, (1,1) from ch2
    s2 = t[None, :]
    t2 = t[:, None]
    same_blk = (t2 < CB) == (s2 < CB)
    Tdiag = tl.load(T_ptr + chsel[:, None] * (CB * CB) + (t2 % CB) * CB + (s2 % CB),
                    mask=same_blk & ((t2 < CB) | has2), other=0.0)
    Kt = tl.trans(K)
    Skk = tl.dot(K, Kt)                                                       # [CH, CH]
    dG = t2 - s2
    dGf = G[:, None] - G[None, :]
    lower = t2 > s2
    A21 = tl.where((t2 >= CB) & (s2 < CB), beta[:, None] * tl.exp(tl.where(lower, dGf, 0.0)) * Skk, 0.0).to(tl.bfloat16)
    eye = (t2 == s2).to(tl.bfloat16)
    T1i = tl.where((t2 < CB) & (s2 < CB), Tdiag + eye, 0.0)
    T2i = tl.where((t2 >= CB) & (s2 >= CB), Tdiag + eye, 0.0)
    X = tl.dot(A21, T1i)
    T21 = -tl.dot(T2i, X.to(tl.bfloat16))                                    # nonzero only in block (1,0)
    Toff = Tdiag.to(tl.float32) + T21                                         # T - I, strictly lower
    eG = tl.exp(G)
    bg = beta * eG
    W = bg[:, None] * K.to(tl.float32) + tl.dot((Toff * bg[None, :]).to(tl.bfloat16), K)
    dL = tl.exp(gl - G)
    Mp = -tl.dot(Kt, (dL[:, None] * W).to(tl.bfloat16))
    Mp8 = tl.reshape(Mp.to(tl.bfloat16), [4, 2, 2, 8, 8, 2, 4, 2])
    Mpf = tl.reshape(tl.permute(Mp8, [0, 1, 4, 3, 6, 5, 2, 7]), [DK * DK])
    tl.store(Mp_ptr + ch * DK * DK + tl.arange(0, DK * DK), Mpf)
    if STAGE == 1:
        return
    Q = tl.load(q_ptr + (tok0 + t)[:, None] * (HQK * DK) + hq * DK + dk[None, :], mask=valid[:, None], other=0.0)
    Sqk = tl.dot(Q, Kt)
    causal = t2 >= s2
    A = tl.where(causal, scale * tl.exp(tl.where(causal, dGf, 0.0)) * Sqk, 0.0).to(tl.bfloat16)
    Qt = scale * eG[:, None] * Q.to(tl.float32) - tl.dot(A, W.to(tl.bfloat16))
    tl.store(Qt_ptr + ch * CH * DK + t[:, None] * DK + dk[None, :], Qt.to(tl.bfloat16))
    if STAGE == 2:
        return
    V = tl.load(v_ptr + (tok0 + t)[:, None] * (HV * DV) + hv * DV + dv[None, :], mask=valid[:, None], other=0.0)
    U = beta[:, None] * V.to(tl.float32) + tl.dot((Toff * beta[None, :]).to(tl.bfloat16), V)
    Ot = tl.dot(A, U.to(tl.bfloat16))
    tl.store(Ot_ptr + ch * CH * DV + t[:, None] * DV + dv[None, :], Ot.to(tl.bfloat16))
    Ud = dL[:, None] * U
    uh = Ud.to(tl.bfloat16)
    if SPLIT_U:
        ul = (Ud - uh.to(tl.float32)).to(tl.bfloat16)
        B = tl.dot(Kt, ul, tl.dot(Kt, uh))
    else:
        B = tl.dot(Kt, uh)
    ii = tl.arange(0, DK)[:, None]
    B8 = tl.reshape(B.to(tl.bfloat16), [4, 2, 2, 8, 8, 2, 4, 2])
    Bf = tl.reshape(tl.permute(B8, [4, 0, 3, 6, 1, 5, 2, 7]), [DK * DV])
    tl.store(B_ptr + ch * DK * DV + tl.arange(0, DK * DV), Bf)


@triton.jit
def out_kernel(Qt_ptr, Ot_ptr, Hs_ptr, cinfo_ptr, out_ptr,
               HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, CH: tl.constexpr):
    c = tl.program_id(0)
    hv = tl.program_id(1)
    length = tl.load(cinfo_ptr + c * 4 + 2)
    if length <= 0:
        return
    tok0 = tl.load(cinfo_ptr + c * 4 + 1)
    rows = tl.arange(0, CH)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)
    ch = c * HV + hv
    Qt = tl.load(Qt_ptr + ch * CH * DK + rows[:, None] * DK + dk[None, :])
    H = tl.load(Hs_ptr + ch * DK * DV + dk[:, None] * DV + dv[None, :])
    Ot = tl.load(Ot_ptr + ch * CH * DV + rows[:, None] * DV + dv[None, :])
    o = tl.dot(Qt, H) + Ot.to(tl.float32)
    valid = rows < length
    tl.store(out_ptr + (tok0 + rows)[:, None] * (HV * DV) + hv * DV + dv[None, :], o.to(tl.bfloat16), mask=valid[:, None])


def gdn_prefill_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    L = _load_lib()
    T, HQK, DK = q.shape
    HV, DV = v.shape[1], v.shape[2]
    N = state.shape[0]
    assert DK == 128 and DV == 128 and HV == 2 * HQK
    # the CUDA kernels store Qt/Ot/Hs as swizzled 32 KB tile images; the Triton dev-only kernels use row-major -> no mixing
    assert USE_CUDA_INTRA == USE_CUDA_OUT, "CUDA and Triton intra/out kernels use different Qt/Ot/Hs layouts"
    dev = q.device
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); a = a.contiguous(); b = b.contiguous()
    state = state.contiguous()
    A_log = A_log.contiguous().float(); dt_bias = dt_bias.contiguous().float()
    cu = cu_seqlens.contiguous()
    if cu.dtype != torch.int64:
        cu = cu.to(torch.int64)
    C64 = T // CB + N
    assert (T + CH - 1) // CH <= 1008, "a sequence may have at most 1008 chunks of 128 tokens (per-item gate buffer, IT_GVALS_CAP)"
    C128 = T // CH + N
    stream = torch.cuda.current_stream(dev).cuda_stream
    sp = ctypes.c_void_p(stream)

    i32 = torch.int32; bf = torch.bfloat16; f32 = torch.float32
    cstart64 = torch.empty(N + 1, dtype=i32, device=dev)
    cstart128 = torch.empty(2 * (N + 1), dtype=i32, device=dev)     # [cbeg(n) .. total | cend(n)]
    cinfo64 = torch.empty(C64 * 4, dtype=i32, device=dev)
    cinfo128 = torch.empty(C128 * 4, dtype=i32, device=dev)
    gam = torch.empty(C64 * HV * CB, dtype=f32, device=dev)
    bet = torch.empty(C64 * HV * CB, dtype=f32, device=dev)
    Tm = torch.empty(C64 * HV * CB * CB, dtype=bf, device=dev)
    gchunk = torch.empty(C128 * HV, dtype=f32, device=dev)
    Mp = torch.empty(C128 * HV * DK * DK, dtype=bf, device=dev)
    Bp = torch.empty(C128 * HV * DK * DV, dtype=bf, device=dev)
    Qt = torch.empty(C128 * HV * CH * DK, dtype=bf, device=dev)
    Ot = torch.empty(C128 * HV * CH * DV, dtype=bf, device=dev)
    Hs = torch.empty(C128 * HV * DK * DV, dtype=bf, device=dev)
    out = torch.empty(T, HV, DV, dtype=bf, device=dev)
    new_state = torch.empty(N, HV, DV, DK, dtype=f32, device=dev)

    counters = torch.empty(16, dtype=i32, device=dev)
    ready = torch.empty(C128, dtype=i32, device=dev)
    ready2 = torch.empty(C128 * HV, dtype=i32, device=dev)
    NVS = DV // VW
    nseq = NVS * HV * N
    nitems = C128 * HV
    nsm = _num_sms(dev)
    # queue split: the early seq group (up to nsm//2 items) is dequeued once the remaining intra work fits the other SMs
    max_early = min(nseq, nsm // 2)
    Ksplit = max(0, min(nitems, nitems - int((nsm - max_early) * SEQ_OVER_ITEM))) if (FUSE_SEQ and USE_CUDA_INTRA) else nitems
    seq_order = torch.empty(max(nseq, 1), dtype=i32, device=dev)
    C64meta = 0 if USE_CUDA_INTRA else C64      # the fused kernel does not use the 64-block tables
    rc = L.gdn_meta(_p(cu), N, C64meta, C128, _p(cstart64), _p(cstart128), _p(cinfo64), _p(cinfo128), _p(counters), _p(ready), _p(ready2), HV,
                    _p(seq_order), Ksplit, max_early, sp)
    assert rc == 0, f"meta launch failed {rc}"
    if USE_CUDA_INTRA:
        rc = L.gdn_intra(_p(q), _p(k), _p(v), _p(a), _p(b), _p(A_log), _p(dt_bias), _p(cinfo128), C128, HQK, HV, float(scale), _p(counters), _p(ready),
                         _p(Mp), _p(Bp), _p(Qt), _p(Ot), _p(gchunk), _p(state), _p(new_state), _p(Hs), _p(cstart128), N, 1 if FUSE_SEQ else 0,
                         _p(ready2), _p(out), 1 if (FUSE_SEQ and FUSE_OUT) else 0, _p(seq_order), Ksplit, sp)
        assert rc == 0, f"intra launch failed {rc}"
    else:
        rc = L.gdn_prep(_p(k), _p(a), _p(b), _p(A_log), _p(dt_bias), _p(cinfo64), C64, HQK, HV, _p(gam), _p(bet), _p(Tm), sp)
        assert rc == 0, f"prep launch failed {rc}"
        intra_kernel[(C128, HV)](Tm, q, k, v, gam, bet, cinfo128, cstart64, cstart128, Mp, Bp, Qt, Ot, gchunk, float(scale),
                                 HQK=HQK, HV=HV, DK=DK, DV=DV, CB=CB, VW=VW, num_warps=8)
    if os.environ.get('GDN_COLD_L2'):
        torch.empty(256 << 20, dtype=torch.uint8, device=dev).fill_(1)
    if FUSE_SEQ and USE_CUDA_INTRA:
        pass
    elif USE_SEQ3:
        rc = L.gdn_seq3(_p(Mp), _p(Bp), _p(gchunk), _p(state), _p(new_state), _p(Hs), _p(cstart128), _p(ready), HV, N, sp)
    else:
        raise RuntimeError("seq v2 removed")
    if FUSE_SEQ and FUSE_OUT and USE_CUDA_INTRA:
        pass
    elif USE_CUDA_OUT:
        rc = L.gdn_out(_p(Qt), _p(Ot), _p(Hs), _p(cinfo128), _p(out), C128, HV, sp)
        assert rc == 0, f"out launch failed {rc}"
    else:
        out_kernel[(C128, HV)](Qt, Ot, Hs, cinfo128, out, HV=HV, DK=DK, DV=DV, CH=CH, num_warps=8)
    if os.environ.get('GDN_STASH'):
        globals()['_STASH'] = dict(Mp=Mp, Bp=Bp, Qt=Qt, Ot=Ot, gchunk=gchunk, Hs=Hs, ready=ready, ready2=ready2, cinfo128=cinfo128)
    return out, new_state
