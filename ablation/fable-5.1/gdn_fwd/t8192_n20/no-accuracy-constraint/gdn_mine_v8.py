"""ctypes wrapper for gdn_fwd_v4.so (persistent megakernel)."""
import ctypes, os, torch

_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.environ.get("GDN_SO", "/root/gdn/gdn_fwd_v8.so"))
        _lib.gdn_fwd.restype = ctypes.c_int
        _lib.gdn_fwd.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 11 + [ctypes.c_int, ctypes.c_void_p]
        _lib.gdn_ws_sizes.restype = ctypes.c_int
        _lib.gdn_ws_sizes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
        _lib.gdn_info.restype = ctypes.c_int
        _lib.gdn_info.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return _lib

_ws = {}
def workspace(T, N, dev):
    key = (T, N, str(dev))
    if key not in _ws:
        nc = (T + 63) // 64 + N
        sz = (ctypes.c_longlong * 3)()
        lib().gdn_ws_sizes(T, N, sz)
        _ws[key] = dict(
            W=torch.empty(nc * 8 * 64 * 128, dtype=torch.bfloat16, device=dev),
            U=torch.empty(nc * 8 * 64 * 128, dtype=torch.bfloat16, device=dev),
            cl=torch.empty(nc * 8 * 64, dtype=torch.float32, device=dev),
            S0=torch.empty(nc * 8 * 128 * 128, dtype=torch.bfloat16, device=dev),
            Dl=torch.empty(nc * 8 * 64 * 128, dtype=torch.bfloat16, device=dev),
            G=torch.empty(nc * 8 * 64 * 64, dtype=torch.bfloat16, device=dev),
            tbl=torch.zeros(sz[0] // 4, dtype=torch.int32, device=dev),
            flags=torch.zeros(sz[1] // 4, dtype=torch.int32, device=dev),
            prog=torch.zeros(sz[2] // 8, dtype=torch.int64, device=dev),
            epoch=[0],
        )
    return _ws[key]

def info():
    o = (ctypes.c_int * 5)()
    lib().gdn_info(o)
    return dict(smem=o[0], blocks=o[1], chain=o[2], prep=o[3], out=o[4])

@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, N = q.shape[0], cu_seqlens.shape[0] - 1
    if scale is None or scale == 0.0:
        scale = 128 ** -0.5
    out = torch.empty((T, 8, 128), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((N, 8, 128, 128), dtype=torch.float32, device=q.device)
    ws = workspace(T, N, q.device)
    ws["epoch"][0] += 1
    ep = ws["epoch"][0]
    if ep % 1024 == 0:  # queue counter slot wraps; reset all flags (rare)
        ws["flags"].zero_(); ws["prog"].zero_()
        ws["epoch"][0] += 1; ep += 1
    st = torch.cuda.current_stream().cuda_stream
    rc = lib().gdn_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr(), A_log.data_ptr(), a.data_ptr(),
                       dt_bias.data_ptr(), b.data_ptr(), cu_seqlens.data_ptr(), T, N, float(scale), out.data_ptr(),
                       new_state.data_ptr(), ws["W"].data_ptr(), ws["U"].data_ptr(), ws["cl"].data_ptr(),
                       ws["S0"].data_ptr(), ws["Dl"].data_ptr(), ws["G"].data_ptr(), ws["tbl"].data_ptr(), ws["flags"].data_ptr(),
                       ws["prog"].data_ptr(), ep, st)
    if rc != 0:
        raise RuntimeError(f"gdn_fwd cuda error {rc}")
    return out, new_state
