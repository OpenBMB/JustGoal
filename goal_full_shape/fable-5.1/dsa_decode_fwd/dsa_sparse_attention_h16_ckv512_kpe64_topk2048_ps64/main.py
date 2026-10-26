"""DSA sparse MLA decode (dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64) — custom CUDA kernel.

The kernel lives in dsa_decode.cu (same directory). It is compiled once with nvcc into a shared
library (hardware/shape-independent one-time preparation) and launched through ctypes.
"""
import ctypes, hashlib, os, subprocess, sys, tempfile, math
import torch

_LIB = None
_WS = {}      # device -> (capacity_T, workspace tensor); shape-only, data-independent, zero-initialised once
_NUM_SMS = {}
DEBUG_BUF = None
FORCE_NB = int(os.environ.get("DSA_FORCE_NB", "0"))  # 0 = automatic unit size; 1/2 force 32/64-row units (experiments only)  # optional int64 CUDA tensor for per-phase timestamps (DSA_DEBUG builds)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "dsa_decode.cu")


def _find_nvcc():
    for cand in (os.environ.get("NVCC"), "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"):
        if cand and os.path.exists(cand):
            return cand
    from shutil import which
    p = which("nvcc")
    if p:
        return p
    raise RuntimeError("nvcc not found")


def _build():
    global _LIB
    if _LIB is not None:
        return _LIB
    with open(_SRC, "rb") as f:
        src = f.read()
    tag = hashlib.sha256(src + os.environ.get("DSA_NVCC_FLAGS", "").encode()).hexdigest()[:16]
    cache_dir = os.path.join(tempfile.gettempdir(), "dsa_decode_build")
    os.makedirs(cache_dir, exist_ok=True)
    so = os.path.join(cache_dir, f"dsa_decode_{tag}.so")
    if not os.path.exists(so):
        nvcc = _find_nvcc()
        tmp = so + f".{os.getpid()}.tmp"
        cmd = [nvcc, "-O3", "-std=c++17", "-arch=sm_100a", "-shared", "-Xcompiler", "-fPIC",
               "-o", tmp, _SRC] + os.environ.get("DSA_NVCC_FLAGS", "").split()
        subprocess.check_call(cmd)
        os.replace(tmp, so)
    _LIB = ctypes.CDLL(so)
    _LIB.dsa_decode_launch.restype = ctypes.c_int
    _LIB.dsa_decode_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                              ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                              ctypes.c_void_p]
    _LIB.dsa_decode_workspace_bytes.restype = ctypes.c_size_t
    _LIB.dsa_decode_workspace_bytes.argtypes = [ctypes.c_int]
    return _LIB


def _fallback(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Plain torch implementation for shapes outside the fast path (never used for topk<=4096)."""
    T, H, Dc = q_nope.shape
    P, ps, _ = ckv_cache.shape
    ckv = ckv_cache.reshape(P * ps, Dc).float()
    kpe = kpe_cache.reshape(P * ps, -1).float()
    out = torch.zeros(T, H, Dc, dtype=torch.float32, device=q_nope.device)
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device=q_nope.device)
    for t in range(T):
        idx = sparse_indices[t].long()
        idx = idx[(idx >= 0) & (idx < P * ps)]
        if idx.numel() == 0:
            continue
        kc, kp = ckv[idx], kpe[idx]
        s = (q_nope[t].float() @ kc.T + q_pe[t].float() @ kp.T) * float(sm_scale)
        lse[t] = torch.logsumexp(s, -1) / math.log(2.0)
        out[t] = torch.softmax(s, -1) @ kc
    return out.to(torch.bfloat16), lse


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, H, Dc = q_nope.shape
    P, ps, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    assert H == 16 and Dc == 512 and q_pe.shape[-1] == 64 and ps == 64
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    if topk > 2048:
        return _fallback(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    lib = _build()
    dev = q_nope.device
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _NUM_SMS:
        _NUM_SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    cap, wsbuf, state = _WS.get(key, (0, None, None))
    if cap < T:
        cap = max(T, 16)
        wsbuf = torch.zeros(lib.dsa_decode_workspace_bytes(cap), dtype=torch.uint8, device=dev)
        state = (ctypes.c_uint * 2)(0, 0)   # host-side epoch / ticket base for this workspace
        _WS[key] = (cap, wsbuf, state)
    q_nope = q_nope.contiguous(); q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous(); kpe_cache = kpe_cache.contiguous()
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    sparse_indices = sparse_indices.contiguous()
    out = torch.empty(T, H, Dc, dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty(T, H, dtype=torch.float32, device=q_nope.device)
    stream = torch.cuda.current_stream(q_nope.device).cuda_stream
    rc = lib.dsa_decode_launch(q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(), kpe_cache.data_ptr(),
                               sparse_indices.data_ptr(), out.data_ptr(), lse.data_ptr(), T, topk, P * ps,
                               float(sm_scale), stream, wsbuf.data_ptr(), cap, _NUM_SMS[key], FORCE_NB,
                               ctypes.addressof(state), DEBUG_BUF.data_ptr() if DEBUG_BUF is not None else None)
    if rc != 0:
        raise RuntimeError(f"dsa_decode_launch failed with code {rc}")
    return out, lse
