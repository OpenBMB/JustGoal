import ctypes, torch, os
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfa.so"))
        _lib.fa_fwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _lib.fa_fwd.restype = ctypes.c_int
    return _lib

def my_attn(q, k, v, scale, grid=0, out=None, dbg=None, clk=None):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    assert q.dtype == torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and D == 128
    if out is None: out = torch.empty_like(q)
    rc = lib().fa_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), B, S, HQ, HKV, float(scale), grid,
                      torch.cuda.current_stream().cuda_stream, dbg.data_ptr() if dbg is not None else None, clk.data_ptr() if clk is not None else None)
    if rc != 0: raise RuntimeError(f"fa_fwd rc={rc}")
    return out
