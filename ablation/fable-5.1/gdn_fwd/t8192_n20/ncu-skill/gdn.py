"""ctypes wrapper for libgdn.so. run() computes everything (gates -> output, new_state) inside the timed call."""
import ctypes, os, torch
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgdn.so"))
        _lib.gdn_workspace_bytes.restype = ctypes.c_size_t
        _lib.gdn_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        _lib.gdn_forward.restype = ctypes.c_int
        _lib.gdn_forward.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return _lib

_ws_cache = {}

def get_workspace(T, N, device):
    """Workspace buffer (allocated once per size; reused across calls, holds no cross-call results)."""
    nbytes = lib().gdn_workspace_bytes(T, N)
    key = (nbytes, str(device))
    if key not in _ws_cache:
        _ws_cache[key] = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    return _ws_cache[key]

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, workspace=None, return_ws=False):
    T = q.shape[0]; N = cu_seqlens.numel() - 1
    assert q.dtype == torch.bfloat16 and cu_seqlens.dtype == torch.int64 and cu_seqlens.is_cuda
    for t in (q, k, v, a, b, A_log, dt_bias, cu_seqlens): assert t.is_contiguous()
    if state is not None: assert state.is_contiguous() and state.dtype == torch.float32
    out = torch.empty((T, 8, 128), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((N, 8, 128, 128), dtype=torch.float32, device=q.device)
    L = lib()
    nbytes = L.gdn_workspace_bytes(T, N)
    if workspace is None or workspace.numel() < nbytes:
        workspace = get_workspace(T, N, q.device)
    if scale is None or scale == 0.0: scale = 128 ** -0.5
    err = L.gdn_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr() if state is not None else None,
                        A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(), cu_seqlens.data_ptr(),
                        float(scale), T, N, out.data_ptr(), new_state.data_ptr(), workspace.data_ptr(),
                        torch.cuda.current_stream().cuda_stream)
    if err != 0: raise RuntimeError("gdn_forward cuda error %d" % err)
    if return_ws: return out, new_state, workspace
    return out, new_state

def unswizzle_tiles(ws, T, N, nchunks):
    """Per-chunk-head fp32 tensors: Wn,Qg,Kp [nc,8,64,128], U [nc,8,64,128] (cols=v), Aq [nc,8,64,64], gC [nc,8]."""
    maxChunks = T // 64 + N; BLK = 3 * 16384 + 8192 + 64 + 4 * 4096
    blk = ws[:maxChunks * 8 * BLK].view(maxChunks * 8, BLK)[:nchunks * 8]
    dev = ws.device
    row = torch.arange(64, device=dev)[:, None]; col = torch.arange(128, device=dev)[None, :]
    phys128 = ((col >> 3) ^ (row & 7)) * 8 + (col & 7)
    res = {}
    for i, name in enumerate(["Wn", "Qg", "Kp"]):
        t = blk[:, i * 16384:(i + 1) * 16384].contiguous().view(torch.bfloat16).view(nchunks, 8, 64, 128)
        res[name] = t.gather(3, phys128.expand(nchunks, 8, 64, 128)).float()
    col64 = torch.arange(64, device=dev)[None, :]
    phys64 = ((col64 >> 3) ^ (row & 7)) * 8 + (col64 & 7)
    aq = blk[:, 49152:49152 + 8192].contiguous().view(torch.bfloat16).view(nchunks, 8, 64, 64)
    res["Aq"] = aq.gather(3, phys64.expand(nchunks, 8, 64, 64)).float()
    res["gC"] = blk[:, 57344:57348].contiguous().view(torch.float32).view(nchunks, 8).clone()
    col32 = torch.arange(32, device=dev)[None, :]
    phys32 = ((col32 >> 3) ^ ((row >> 1) & 3)) * 8 + (col32 & 7)
    u = blk[:, 57408:57408 + 16384].contiguous().view(torch.bfloat16).view(nchunks, 8, 4, 64, 32)
    u = u.gather(4, phys32.expand(nchunks, 8, 4, 64, 32)).float()
    res["U"] = u.permute(0, 1, 3, 2, 4).reshape(nchunks, 8, 64, 128)
    return res
