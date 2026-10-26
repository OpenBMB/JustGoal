import ctypes, os, subprocess, torch
from common import *

SRC = os.environ.get("MLA_SRC", "/workspace/mla_fresh/mla_decode.cu")
def build(tmax=64, cl=8, extra=""):
    so = f"/workspace/mla_fresh/build/libmla_{SRC.split('/')[-1][:-3]}_T{tmax}_C{cl}{extra.replace(' ','_').replace('-','')}.so"
    os.makedirs(os.path.dirname(so), exist_ok=True)
    src = SRC
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        cmd = (f"/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v --shared -Xcompiler -fPIC "
               f"-DT_MAX={tmax} -DCL={cl} {extra} -o {so} {src}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr); raise SystemExit("nvcc failed")
        print("\n".join(l for l in r.stderr.splitlines() if "registers" in l or "spill" in l))
    return ctypes.CDLL(so)

class Mine:
    def __init__(self, tmax=64, cl=8, extra=""):
        self.lib = build(tmax, cl, extra)
        self.lib.mla_decode_launch.restype = ctypes.c_int
        self.lib.mla_decode_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_float, ctypes.c_int, ctypes.c_int] + \
            [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mla_decode_max_part.restype = ctypes.c_int
        self.bufs = None
        self.prof = torch.zeros(4096 * 32, dtype=torch.int64, device='cuda')

    def prepare(self, B, n_idx, device="cuda"):
        mp = self.lib.mla_decode_max_part(n_idx) + B  # chunk-count upper bound = N/T + B
        if self.bufs is None or self.bufs[0] < B or self.bufs[1] < mp:
            # generous scratch that fits both the cluster (v1) and the split-KV (v3) indexing schemes
            self.bufs = (B, mp,
                         torch.empty(B * mp * H * DC, dtype=torch.float32, device=device),
                         torch.empty(B * mp * 16 * 32, dtype=torch.float32, device=device),
                         torch.zeros(B * 16, dtype=torch.int32, device=device))
        return mp

    def __call__(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, out=None, lse=None):
        B = kv_indptr.numel() - 1
        mp = self.prepare(B, kv_indices.numel())
        if out is None: out = torch.empty(B, H, DC, dtype=torch.bfloat16, device=q_nope.device)
        if lse is None: lse = torch.empty(B, H, dtype=torch.float32, device=q_nope.device)
        _, _, po, pml, cnt = self.bufs
        err = self.lib.mla_decode_launch(q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(), kpe_cache.data_ptr(),
                                         kv_indptr.data_ptr(), kv_indices.data_ptr(), float(sm_scale), B, kv_indices.numel(),
                                         out.data_ptr(), lse.data_ptr(), po.data_ptr(), pml.data_ptr(), cnt.data_ptr(), mp,
                                         torch.cuda.current_stream().cuda_stream, self.prof.data_ptr())
        assert err == 0, f"launch error {err}"
        return out, lse
