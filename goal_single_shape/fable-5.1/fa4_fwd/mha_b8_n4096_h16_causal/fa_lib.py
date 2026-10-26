import torch, ctypes, subprocess, os, math
_lib = None
def build(src="fa_fwd.cu", so="libfa_fwd.so", extra=""):
    global _lib
    cmd = f"nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -lineinfo -Xptxas -v {extra} -o {so} {src} -lcuda"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise SystemExit("build failed")
    for line in r.stderr.splitlines():
        if "registers" in line or "spill" in line: print("[ptxas]", line.strip())
    _lib = ctypes.CDLL(os.path.abspath(so))
    _lib.fa_fwd.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*5 + [ctypes.c_void_p]
    return _lib
_ws = None
def fa_fwd(q, k, v, G=8, grid=0):
    global _ws
    B, S, H, D = q.shape
    if _ws is None: _ws = torch.zeros(2, dtype=torch.int32, device=q.device)
    o = torch.empty_like(q)
    rc = _lib.fa_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), _ws.data_ptr(), B, S, H, G, grid, torch.cuda.current_stream().cuda_stream)
    assert rc == 0
    return o
