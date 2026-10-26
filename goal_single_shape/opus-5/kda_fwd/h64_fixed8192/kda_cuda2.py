"""ctypes wrapper for the two-kernel KDA forward."""
import ctypes, os, subprocess, torch
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.environ.get("KDA_SRC","kda_k2.cu")); SO = os.path.join(HERE, "libkda2.so")
_cur_so = SO   # dlopen caches by path, so every distinct build needs its own file
NVCC = "/usr/local/cuda/bin/nvcc"
_lib = None; _ws = {}; _built = False

def build(force=False, verbose=True, extra=()):
    global _cur_so, _built
    _built = True
    # A distinct flag set must land in a distinct .so: dlopen() keys its cache on
    # the path, so recompiling in place silently returns the previously loaded one.
    import hashlib
    key = os.path.basename(SRC) + "|" + "|".join(extra)
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    _cur_so = os.path.join(HERE, "libkda2_%s.so" % tag)
    if not force and os.path.exists(_cur_so) and os.path.getmtime(_cur_so) > os.path.getmtime(SRC):
        return
    cmd = [NVCC, "-O3", "-gencode", "arch=compute_100a,code=sm_100a", "-shared", "-Xcompiler", "-fPIC",
           "--use_fast_math", "-Xptxas", "-v", *extra, "-o", _cur_so, SRC]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        for ln in r.stderr.splitlines():
            if any(w in ln for w in ("register", "spill", "smem", "error", "Error")):
                print("[nvcc]", ln)
    if r.returncode != 0:
        print(r.stderr); raise RuntimeError("nvcc failed")

def _load():
    global _lib
    if _lib is None:
        if not _built: build()
        _lib = ctypes.CDLL(_cur_so)
        _lib.kda2_launch.restype = ctypes.c_int
        _lib.kda2_launch.argtypes = [ctypes.c_void_p]*16 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        _lib.kda2_smem_prep.restype = ctypes.c_size_t
        _lib.kda2_smem_scan.restype = ctypes.c_size_t
    return _lib

def workspace(T, H, D=128, C=32, dev="cuda"):
    key = (T, H, D, C)
    if key not in _ws:
        NC = T // C
        _ws[key] = dict(
            KQ=torch.empty(NC, H, 2*C, D, device=dev, dtype=torch.bfloat16),
            Kh=torch.empty(NC, H, C, D, device=dev, dtype=torch.bfloat16),
            Tm=torch.empty(NC, H, C, C, device=dev, dtype=torch.bfloat16),
            Ah=torch.empty(NC, H, C, C, device=dev, dtype=torch.bfloat16),
            Ec=torch.empty(NC, H, D, device=dev, dtype=torch.float32),
            Prod=torch.zeros(NC * H, device=dev, dtype=torch.int32))
    return _ws[key]

def kda_fwd(q, k, v, g, beta, A_log, dt_bias, scale, state, out=None, ht=None):
    lib = _load()
    B, T, H, D = q.shape
    DV = v.shape[-1]
    w = workspace(T, H, D)
    if out is None: out = torch.empty(B, T, H, DV, device=q.device, dtype=torch.bfloat16)
    if ht is None: ht = torch.empty_like(state)
    a = [x.data_ptr() for x in (q, k, v, g, beta, A_log, dt_bias, state, out, ht,
                                w["KQ"], w["Kh"], w["Tm"], w["Ah"], w["Ec"], w["Prod"])]
    rc = lib.kda2_launch(*a, B, T, H, ctypes.c_float(scale),
                         ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0: raise RuntimeError(f"kda2_launch rc={rc}")
    return out, ht
