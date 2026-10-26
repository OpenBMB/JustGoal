import ctypes, os, torch
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.environ.get("FA_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel", "libfa.so")))
        _lib.fa_fwd.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        _lib.fa_fwd.restype = ctypes.c_int
    return _lib

def fa_fwd(q, k, v, softmax_scale=None, out=None):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    assert D == 128 and q.dtype == torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    if softmax_scale is None: softmax_scale = D ** -0.5
    if out is None: out = torch.empty_like(q)
    rc = lib().fa_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), B, S, HQ, HKV,
                      float(softmax_scale), torch.cuda.current_stream().cuda_stream)
    if rc != 0: raise RuntimeError(f"fa_fwd failed rc={rc}")
    return out
