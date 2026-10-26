import torch, ctypes, os
_lib = None
def _get():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL('/root/kda/libkda.so')
        _lib.kda_fwd_launch.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int, ctypes.c_int,
                                        ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _lib.kda_smem_bytes.restype = ctypes.c_int
    return _lib

def kda_fwd(q,k,v,g,beta,A_log,dt_bias,state,cu_seqlens,lower_bound=-4.0,scale=None,out=None,stop_at=0,dbg=None):
    lib=_get()
    B,T,H,D = q.shape
    N = cu_seqlens.numel()-1
    if scale is None: scale = D ** -0.5
    if out is None: out = torch.empty_like(q)
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    lib.kda_fwd_launch(p(q),p(k),p(v),p(g),p(beta),p(A_log),p(dt_bias),p(state),p(out),
                       p(cu_seqlens),H,N,ctypes.c_float(lower_bound),ctypes.c_float(scale),
                       ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), stop_at,
                       ctypes.c_void_p(0 if dbg is None else dbg.data_ptr()))
    return out
def smem_bytes(): return _get().kda_smem_bytes()
