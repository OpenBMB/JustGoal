"""DSA sparse decode attention for B200 — host side.

run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output bf16 [T,16,512], lse fp32 [T,16], log2)

Device code: dsa_kernel.cu (written from scratch).  The prebuilt cubin is loaded through the CUDA driver API with
ctypes and launched on the current torch stream.  Partials are epoch-tagged; the epoch is a per-call counter.
"""
import ctypes
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_KEYS_PER_CTA = int(os.environ.get("DSA_KEYS_PER_CTA", "32"))
_VARIANT = os.environ.get("DSA_VARIANT", "")
_LOG2E = 1.4426950408889634
_MAXBLK = 512

_cuda = None
_state = {}


class _MapTab(ctypes.Structure):
    _fields_ = [("e", (ctypes.c_uint16 * 8) * _MAXBLK)]


_NONE = 0xFFFF


def _check(err, what):
    if err != 0:
        s = ctypes.c_char_p()
        _cuda.cuGetErrorString(err, ctypes.byref(s))
        raise RuntimeError(f"{what} failed: CUDA error {err} ({s.value.decode() if s.value else '?'})")


def _init(device):
    global _cuda
    key = ("mod", device.index if device.index is not None else torch.cuda.current_device())
    if key in _state:
        return _state[key]
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _cuda.cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.c_void_p, ctypes.c_char_p]
        _cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        _cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        _cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        _cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                                                                       ctypes.c_int, ctypes.c_size_t]
    torch.cuda.init()
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make sure torch's primary context exists & is current
        ctx = ctypes.c_void_p()
        _check(_cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            _check(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), key[1]), "cuDevicePrimaryCtxRetain")
            _check(_cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        with open(os.path.join(_HERE, f"dsa_s{_KEYS_PER_CTA}{_VARIANT}.cubin"), "rb") as f:
            image = f.read()
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(image, len(image))
        _check(_cuda.cuModuleLoadData(ctypes.byref(mod), buf), "cuModuleLoadData")
        fn = ctypes.c_void_p()
        _check(_cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"dsa_fwd"), "cuModuleGetFunction")
        gptr = ctypes.c_void_p()
        gsz = ctypes.c_size_t()
        _check(_cuda.cuModuleGetGlobal_v2(ctypes.byref(gptr), ctypes.byref(gsz), mod, b"dsa_cfg"), "cuModuleGetGlobal")
        cfg = (ctypes.c_int * 8)()
        _check(_cuda.cuMemcpyDtoH_v2(cfg, gptr, 32), "cuMemcpyDtoH")
        S, NT, NCHUNK, SMEM, COMB, CTRS = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5]
        _check(_cuda.cuFuncSetAttribute(fn, 8, SMEM), "cuFuncSetAttribute(smem)")  # MAX_DYNAMIC_SHARED_SIZE_BYTES
        nb = ctypes.c_int()
        _check(_cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(nb), fn, NT, SMEM), "cuOccupancy")
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        st = dict(mod=mod, fn=fn, S=S, NT=NT, NCHUNK=NCHUNK, SMEM=SMEM, COMB=COMB, CTRS=CTRS, ws={}, maps={},
                  max_resident=nb.value * n_sm, n_sm=n_sm, epoch=0)
        sym = ctypes.c_void_p(); ssz = ctypes.c_size_t()
        if _cuda.cuModuleGetGlobal_v2(ctypes.byref(sym), ctypes.byref(ssz), mod, b"c_tab") == 0:
            st["ctab_sym"] = sym.value; st["ctab_last"] = None
        if os.environ.get("DSA_TIMING"):
            tb = torch.zeros(4096 * 16, dtype=torch.int64, device=device)
            st["tbuf"] = tb
            sym = ctypes.c_void_p(); ssz = ctypes.c_size_t()
            _check(_cuda.cuModuleGetGlobal_v2(ctypes.byref(sym), ctypes.byref(ssz), mod, b"g_tbuf"), "cuModuleGetGlobal(g_tbuf)")
            ptr = ctypes.c_uint64(tb.data_ptr())
            _check(_cuda.cuMemcpyHtoD_v2(sym, ctypes.byref(ptr), 8), "cuMemcpyHtoD(g_tbuf)")
        _state[key] = st
        return st


def _workspace(st, T, device):
    ws = st["ws"].get(T)
    if ws is None:
        nblk = T * st["NCHUNK"]
        # epoch-tagged partials (tags start at 0 = never valid): o as {bf16x2, tag} per dim pair, m/l as {m, l, tag, 0}
        o_part = torch.zeros(nblk * (16 * 1088 // 2), dtype=torch.bfloat16, device=device)  # >= 16 KB + 128 B per slot for every layout
        ml_part = torch.zeros(max(T * 2 * 16 * st["NCHUNK"], nblk * 32, nblk * 16 * 4), dtype=torch.float32, device=device)  # [t][2][16][NCHUNK] or [slot][32]
        bar = torch.zeros(max(2 * T * st["CTRS"], 4096 + 64 * 32), dtype=torch.int32, device=device)  # 2 lines per request (parity) (+ probe chain)
        if os.environ.get("DSA_RTPROBE"):
            for i in range(64):
                bar[4096 + 32 * i] = (i + 1) % 64
        ws = (o_part, ml_part, bar)
        st["ws"][T] = ws
    return ws


def _identity_map(grid):
    tab = _MapTab()
    for b in range(_MAXBLK):
        tab.e[b][0] = b if b < grid else 0
        for i in range(1, 7):
            tab.e[b][i] = _NONE
        tab.e[b][7] = 0  # plain
    return tab


def _schedule_map(smid, T, nchunk, nsm):
    """Static primary chunks + takeover lists, from the probed block->SM placement only (data-independent).
    SM j (first-appearance order) slot k (k-th block on that SM in block order) gets chunk q = k*NSM + j in
    chunk-major order (c = q // T, t = (q % T + rot(c)) % T): low chunk indices (the likely-active ones under tail
    padding) are spread one per SM.  Slot-0 blocks are holders; the holder of SM j lists the slot-1 chunk of SM j+1,
    the slot-2 chunk of SM j+2 and the slot-3 chunk of SM j+3 as takeover candidates (each (SM, slot>=1) chunk has
    exactly one holder).  At run time a holder whose own chunk is idle takes the first candidate that is active and
    shares its SM with an active slot-0 chunk; the candidate's own block sees the same data and stands down."""
    grid = len(smid)
    first = {}
    for b, s_ in enumerate(smid):
        first.setdefault(s_, b)
    sms = sorted(first, key=lambda s_: first[s_])
    NSM = len(sms)
    jof = {s_: j for j, s_ in enumerate(sms)}
    slots = {}          # (j, k) -> block
    blk_jk = {}
    seen = {}
    for b, s_ in enumerate(smid):
        j = jof[s_]
        k = seen.get(j, 0)
        seen[j] = k + 1
        slots[(j, k)] = b
        blk_jk[b] = (j, k)
    nchunks = T * nchunk
    # chunk for (j, k): q = k*NSM + j (must be < nchunks); extra blocks (if any) get chunks beyond the band layout
    chunk_of = {}
    used = set()
    for (j, k), b in sorted(slots.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        q = k * NSM + j
        if q < nchunks:
            c = q // T
            shift = ((c * T) // NSM) * (T // 4 if T >= 4 else 1)
            t = (q % T + shift) % T
            ch = t * nchunk + c
        else:
            ch = None
        chunk_of[(j, k)] = ch
    # fix up: any chunk ids not covered (q >= nchunks mapping) -> assign remaining chunks to blocks with ch None
    assigned = set(v for v in chunk_of.values() if v is not None)
    if len(assigned) != len([v for v in chunk_of.values() if v is not None]):
        raise RuntimeError("schedule map is not injective")
    remaining = [ch for ch in range(nchunks) if ch not in assigned]
    for key in chunk_of:
        if chunk_of[key] is None:
            chunk_of[key] = remaining.pop()
    assert not remaining and len(chunk_of) == grid
    tab = _MapTab()
    for b in range(_MAXBLK):
        for i in range(8):
            tab.e[b][i] = _NONE
        if b >= grid:
            tab.e[b][0] = 0; tab.e[b][7] = 0
    def chunk(j, k):
        return chunk_of.get((j % NSM, k), None)
    for b in range(grid):
        j, k = blk_jk[b]
        own = chunk_of[(j, k)]
        tab.e[b][0] = own
        if k == 0:
            tab.e[b][7] = 1
            for m in (1, 2, 3):
                cand = chunk(j + m, m)
                s0 = chunk(j + m, 0)
                tab.e[b][2 * m - 1] = cand if (cand is not None and s0 is not None) else _NONE
                tab.e[b][2 * m] = s0 if (cand is not None and s0 is not None) else _NONE
        elif k <= 3:
            tab.e[b][7] = k + 1
            tab.e[b][1] = chunk(j, 0) if chunk(j, 0) is not None else _NONE
            h = (j - k) % NSM
            tab.e[b][2] = chunk(h, 0) if chunk(h, 0) is not None else _NONE
            for m in range(1, k):
                cand = chunk(h + m, m)
                s0 = chunk(h + m, 0)
                ok = cand is not None and s0 is not None
                tab.e[b][3 + 2 * (m - 1)] = cand if ok else _NONE
                tab.e[b][4 + 2 * (m - 1)] = s0 if ok else _NONE
            if tab.e[b][1] == _NONE or tab.e[b][2] == _NONE:
                tab.e[b][7] = 0  # plain
        else:
            tab.e[b][7] = 0  # slots beyond 3: plain
    return tab


def _launch(st, args_common, grid, tab, probe_ptr, stream):
    st["epoch"] = (st["epoch"] + 1) & 0xFFFFFFFF
    if st["epoch"] == 0:
        st["epoch"] = 1
    if st.get("ctab_sym") is not None and st.get("ctab_last") is not tab:
        _check(_cuda.cuMemcpyHtoD_v2(ctypes.c_void_p(st["ctab_sym"]), ctypes.byref(tab), ctypes.sizeof(tab)), "cuMemcpyHtoD(c_tab)")
        st["ctab_last"] = tab
    args = args_common + [tab, ctypes.c_void_p(probe_ptr), ctypes.c_uint(st["epoch"])]
    params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    extra = 1 if os.environ.get("DSA_RTPROBE") else 0
    _check(_cuda.cuLaunchKernel(st["fn"], grid + extra, 1, 1, st["NT"], 1, 1, st["SMEM"], stream, params, None),
           "cuLaunchKernel")


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, Hh, DC = q_nope.shape
    assert Hh == 16 and DC == 512 and q_pe.shape[-1] == 64
    assert ckv_cache.shape[1] == 64 and sparse_indices.shape == (T, 2048)
    device = q_nope.device
    st = _init(device)
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    sparse_indices = sparse_indices.contiguous()
    if torch.is_tensor(sm_scale):
        sm_scale = float(sm_scale.item())
    output = torch.empty((T, 16, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((T, 16), dtype=torch.float32, device=device)
    o_part, ml_part, bar = _workspace(st, T, device)

    args = [
        ctypes.c_void_p(q_nope.data_ptr()), ctypes.c_void_p(q_pe.data_ptr()),
        ctypes.c_void_p(ckv_cache.data_ptr()), ctypes.c_void_p(kpe_cache.data_ptr()),
        ctypes.c_void_p(sparse_indices.data_ptr()), ctypes.c_float(sm_scale * _LOG2E),
        ctypes.c_void_p(o_part.data_ptr()), ctypes.c_void_p(ml_part.data_ptr()), ctypes.c_void_p(bar.data_ptr()),
        ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(lse.data_ptr()), ctypes.c_int(T),
    ]
    stream = torch.cuda.current_stream(device).cuda_stream
    grid = T * st["NCHUNK"]
    assert grid >= T * st["COMB"] and grid <= st["max_resident"], (grid, st["max_resident"])
    extra = 1 if os.environ.get("DSA_RTPROBE") else 0
    tab = st["maps"].get(grid)
    if tab is None:
        if os.environ.get("DSA_PLACEMENT", "1") == "1" and grid <= _MAXBLK:
            # probe the block -> SM placement once per grid size (identity table; results are valid), then derive
            # the static schedule table (depends only on the placement, never on input data)
            probe = torch.zeros(_MAXBLK, dtype=torch.int32, device=device)
            _launch(st, args, grid, _identity_map(grid), probe.data_ptr(), stream)
            torch.cuda.current_stream(device).synchronize()
            smid = probe[:grid].tolist()
            st["smid"] = smid
            tab = _schedule_map(smid, T, st["NCHUNK"], st["n_sm"]) if len(set(smid)) > 1 else _identity_map(grid)
            st["maps"][grid] = tab
            return output, lse
        tab = st["maps"][grid] = _identity_map(grid)
    _launch(st, args, grid, tab, 0, stream)
    return output, lse
