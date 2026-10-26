import ctypes, math, os, torch
_lib = None
def lib():
    global _lib
    if _lib is None:
        path = os.environ.get("FA_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfa.so"))
        _lib = ctypes.CDLL(path)
        _lib.fa_fwd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        _lib.fa_fwd.restype = ctypes.c_int
        _lib.fa_fwd_max_grid.restype = ctypes.c_int
    return _lib
_grid = None
def max_grid():
    global _grid
    if _grid is None:
        _grid = lib().fa_fwd_max_grid()
    return _grid
def fa_fwd(q, k, v, grid=None, dbg=None):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    assert D == 128 and q.dtype == torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    o = torch.empty_like(q)
    g = grid or max_grid()
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib().fa_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), B, S, Hq, Hkv, 1.0 / math.sqrt(D), g, stream, dbg.data_ptr() if dbg is not None else None)
    if rc != 0:
        raise RuntimeError("fa_fwd failed")
    return o
