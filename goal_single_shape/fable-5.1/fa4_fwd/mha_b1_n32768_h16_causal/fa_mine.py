import torch, ctypes, os, math
_lib = None
def _load():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfa_fwd.so"))
        _lib.fa_fwd_launch.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        _lib.fa_fwd_launch.restype = ctypes.c_int
    return _lib
def fa_fwd(q, k, v, causal=True, n_pairs=0, trace=None):
    assert causal and q.dtype == torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    B, S, H, D = q.shape; Hkv = k.shape[2]; assert D == 128
    out = torch.empty_like(q)
    r = _load().fa_fwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), B, S, H, Hkv, 1.0 / math.sqrt(D), n_pairs, trace.data_ptr() if trace is not None else 0)
    if r < 0: raise RuntimeError(f"fa_fwd_launch failed {r}")
    return out
