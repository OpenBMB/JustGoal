import ctypes, math, os, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_CU = os.path.join(_HERE, "gdn_kernel.cu")
_CU2 = os.path.join(_HERE, "gdn_split.cu")
_NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
_CACHE = os.environ.get("GDN_BUILD_DIR", "/tmp/gdn_build")
_libs = {}


def _build(chunk, nw=8, extra=()):
    key = (chunk, nw, extra)
    if key in _libs:
        return _libs[key]
    st = os.stat(_CU)
    st2 = os.stat(_CU2)
    st3 = os.stat(os.path.join(_HERE, "gdn_common.cuh"))
    tag = "%d_%d_%d_%d_%d_%d" % (int(st.st_mtime), st.st_size, int(st2.st_mtime), st2.st_size,
                                 int(st3.st_mtime), st3.st_size)
    os.makedirs(_CACHE, exist_ok=True)
    so = os.path.join(_CACHE, "gdn_%d_%d_%s%s.so" % (chunk, nw, tag, "".join(extra).replace("-","").replace("=","")))
    if not os.path.exists(so):
        tmp = so + ".tmp%d.so" % os.getpid()
        cmd = [_NVCC, "-O3", "-std=c++17", "-arch=sm_100a", "-DCHUNK=%d" % chunk,
               "-DNWARPS=%d" % nw,
               "--use_fast_math", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared"] + \
              list(extra) + ["-o", tmp, _CU, _CU2]
        subprocess.run(cmd, check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.gdn_launch.restype = None
    lib.gdn_launch.argtypes = [ctypes.c_void_p] * 11 + [
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p]
    lib.gdn_launch_split.restype = None
    lib.gdn_launch_split.argtypes = [ctypes.c_void_p] * 12 + [
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p]
    lib.gdn_launch_coop.restype = ctypes.c_int
    lib.gdn_launch_coop.argtypes = lib.gdn_launch_split.argtypes
    lib.gdn_ws_bytes.restype = ctypes.c_size_t
    lib.gdn_ws_bytes.argtypes = [ctypes.c_int]
    _libs[key] = lib
    return lib


def run_split(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk=32, vsplit=None,
              nw=8, stages=99, extra=()):
    """Two-kernel path.  `stages` cuts the kernels short for bisection
    measurements (99 = everything); `extra` adds nvcc flags for experiments."""
    import torch
    T, HQ, D = q.shape
    HV = v.shape[1]
    N = cu_seqlens.numel() - 1
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    nchub = (T + chunk - 1) // chunk + N
    if vsplit is None:
        vsplit = 1
        while N * HV * vsplit < 300 and vsplit < 4:
            vsplit *= 2
    lib = _build(chunk, nw, extra)
    o = torch.empty((T, HV, D), dtype=torch.bfloat16, device=q.device)
    ns = torch.empty((N, HV, D, D), dtype=torch.float32, device=q.device)
    wsb = lib.gdn_ws_bytes(int(nchub))
    ws = torch.empty(int(wsb), dtype=torch.uint8, device=q.device)
    stream = torch.cuda.current_stream().cuda_stream
    lib.gdn_launch_split(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                         state.data_ptr() if state is not None else None,
                         A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(),
                         cu_seqlens.data_ptr(), o.data_ptr(), ns.data_ptr(), ws.data_ptr(),
                         ctypes.c_float(scale), int(T), int(N), int(nchub), int(vsplit),
                         int(stages), ctypes.c_void_p(stream))
    return o, ns


def run_coop(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk=32, vsplit=None,
             nw=8, stages=99, extra=()):
    """Both split kernels in one cooperative launch, which pays the ~5 us
    per-launch floor (and the ~2 us gap between launches) once instead of twice.
    It needs the whole grid co-resident; if that is not possible for this shape
    the launcher says so and we fall back to the two-kernel path."""
    import torch
    T, HQ, D = q.shape
    HV = v.shape[1]
    N = cu_seqlens.numel() - 1
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    nchub = (T + chunk - 1) // chunk + N
    if vsplit is None:
        vsplit = 8
    lib = _build(chunk, nw, extra)
    o = torch.empty((T, HV, D), dtype=torch.bfloat16, device=q.device)
    ns = torch.empty((N, HV, D, D), dtype=torch.float32, device=q.device)
    ws = torch.empty(int(lib.gdn_ws_bytes(int(nchub))), dtype=torch.uint8, device=q.device)
    stream = torch.cuda.current_stream().cuda_stream
    args = (q.data_ptr(), k.data_ptr(), v.data_ptr(),
            state.data_ptr() if state is not None else None,
            A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(),
            cu_seqlens.data_ptr(), o.data_ptr(), ns.data_ptr(), ws.data_ptr(),
            ctypes.c_float(scale), int(T), int(N), int(nchub), int(vsplit), int(stages),
            ctypes.c_void_p(stream))
    if not lib.gdn_launch_coop(*args):
        lib.gdn_launch_split(*args)
    return o, ns


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk=32, vsplit=None, nw=8,
        stages=99, extra=()):
    """Fully fused single-kernel path (see the docstrings in gdn_kernel.cu)."""
    import torch
    T, HQ, D = q.shape
    HV = v.shape[1]
    N = cu_seqlens.numel() - 1
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)
    if vsplit is None:
        vsplit = 1
        while N * HV * vsplit < 100 and vsplit < 4:
            vsplit *= 2
    lib = _build(chunk, nw, extra)
    o = torch.empty((T, HV, D), dtype=torch.bfloat16, device=q.device)
    ns = torch.empty((N, HV, D, D), dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream().cuda_stream
    lib.gdn_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                   state.data_ptr() if state is not None else None,
                   A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(),
                   cu_seqlens.data_ptr(), o.data_ptr(), ns.data_ptr(),
                   ctypes.c_float(scale), int(T), int(N), int(vsplit), int(stages),
                   ctypes.c_void_p(stream))
    return o, ns


# ---------------------------------------------------------------- dispatch --
# Shape-based dispatch, from measured sweeps over all the workload shapes
# (never data-dependent, never cached between calls).
#
#   * Up to ~80 tokens per sequence the fused kernel wins: its serial chain is
#     only one to three chunks and it pays no intermediate HBM traffic.  A small
#     average can hide a long sequence, and the fused kernel's cost follows the
#     longest one, so the total is capped as well.
#   * Past that the split pair wins.  Both kernels want 8 warps -- the parallel
#     UT transform is 19% faster with 8 than with 4 and the serial scan is
#     indifferent -- so the only question left is the V split.  Spending it on
#     V slices buys more blocks but re-reads W/Qg/Kd once per slice; with few
#     sequences there are SMs going begging, so slice wide (vsplit 8), and with
#     many the blocks already fill the machine, so keep the traffic down
#     (vsplit 2).
def _c(x):
    return x if x.is_contiguous() else x.contiguous()


_FUSED_MAX_PER = 80       # tokens per sequence
_WIDE_MAX_SEQS = 13       # sequences below which we slice V more aggressively
_TINY_MAX_T = 16          # total tokens below which a 16-token chunk is cheaper
_FUSED_MAX_T = 200        # ... but with several sequences a small average can
                          # hide a long one, and the fused kernel's cost follows
                          # the longest sequence, so cap the total as well
_COOP_MAX_T = 1000        # below this the two-launch floor is a big share of the
                          # time and the cooperative single launch wins; above it
                          # the fused kernel's larger shared footprint costs the
                          # UT phase a third of its resident blocks and it loses


def _plan(T, N):
    """(kind, vsplit, nwarps, chunk) from the shapes alone.  Note that only
    ceil(T/N) is known here, not the longest sequence -- reading cu_seqlens
    would mean a device sync inside the timed region -- so the boundaries are
    in terms of the average, with the total capped as well where a long
    sequence hiding behind a small average would hurt."""
    per = (T + N - 1) // max(N, 1)        # tokens per sequence
    if T <= _TINY_MAX_T:                  # one short chunk, whatever the split
        return "fused", 8, 8, 16
    if per <= _FUSED_MAX_PER and T <= _FUSED_MAX_T:
        return "fused", 8, 8, 32
    if N <= _WIDE_MAX_SEQS:               # few sequences: slice V wide for blocks
        return ("coop" if T <= _COOP_MAX_T else "split"), 8, 8, 32
    return "split", 2, 8, 32              # enough blocks already; keep the
                                          # workspace re-reads down instead


def dispatch(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    q, k, v = _c(q), _c(k), _c(v)
    A_log, a, dt_bias, b = _c(A_log), _c(a), _c(dt_bias), _c(b)
    cu_seqlens = _c(cu_seqlens)
    if state is not None:
        state = _c(state)
    T = int(q.shape[0])
    N = int(cu_seqlens.numel()) - 1
    if T == 0 or N <= 0:
        import torch
        D, HV = q.shape[2], v.shape[1]
        return (torch.empty((T, HV, D), dtype=torch.bfloat16, device=q.device),
                torch.zeros((max(N, 0), HV, D, D), dtype=torch.float32, device=q.device))
    kind, vs, nw, ch = _plan(T, N)
    f = run if kind == "fused" else (run_coop if kind == "coop" else run_split)
    return f(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, chunk=ch, vsplit=vs, nw=nw)
