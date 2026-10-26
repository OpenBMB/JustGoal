import torch, ctypes, os
_lib = None
def load(path=None):
    global _lib
    path = path or os.environ.get("FA_LIB", "./libfa.so")
    _lib = ctypes.CDLL(path)
    _lib.fa_fwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    _lib.fa_fwd.restype = ctypes.c_int
    return _lib
def fa_fwd(q, k, v, scale=None, out=None, grid=0):
    if _lib is None: load()
    B, S, HQ, D = q.shape; HKV = k.shape[2]
    assert D == 128 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    if out is None: out = torch.empty_like(q)
    scale = scale if scale is not None else D ** -0.5
    rc = _lib.fa_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), B, S, HQ, HKV, scale,
                     torch.cuda.current_stream().cuda_stream, grid)
    if rc != 0: raise RuntimeError(f"fa_fwd rc={rc}")
    return out
