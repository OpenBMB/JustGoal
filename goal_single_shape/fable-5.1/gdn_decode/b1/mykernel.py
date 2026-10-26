import ctypes, os, torch
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.environ.get("GDN_LIB", os.path.expanduser("~/gdn/libgdn.so")))
        _lib.gdn_launch.restype = ctypes.c_int
        _lib.gdn_launch.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*8 + [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return _lib

def run(variant, q, k, v, state, A_log, a, dt_bias, b, scale, out, new_state):
    st = torch.cuda.current_stream().cuda_stream
    rc = lib().gdn_launch(variant, q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr(), A_log.data_ptr(),
                          a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(), scale, out.data_ptr(), new_state.data_ptr(), st)
    if rc != 0:
        raise RuntimeError(f"gdn_launch rc={rc}")
    return out, new_state
