"""DSA sparse decode attention for B200 (sm_100a). Hand-written CUDA kernel (dsa.cu) compiled with nvcc
to a cubin at import time and launched through the CUDA driver API. run() is the entry point."""
import os, sys, subprocess, hashlib, ctypes, math
import torch
from cuda.bindings import driver as cu

_DIR = os.path.dirname(os.path.abspath(__file__))
CS = 80
CL = 2
H, DC, DP, TOPK = 16, 512, 64, 2048
CPT = (TOPK + CS - 1) // CS
KSTRIDE, NWARPS = 328, 4
NT = CS // 8
NCLUSTERS_CAP = int(os.environ.get("DSA_NCLUSTERS_CAP", "100000"))
SMEM_BYTES = (CS * KSTRIDE * 2 + 10240 + NT * 32 * 8 + 16 * CS * 4 + 32 * 4 + CS * 4 + 8)
SMEM_BYTES = (SMEM_BYTES + 127) // 128 * 128
NTHREADS = NWARPS * 32
LOG2E = 1.4426950408889634


def _check(res):
    err = res[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    return res[1] if len(res) == 2 else res[1:]


def _find_nvcc():
    for p in [os.environ.get("NVCC"), "/usr/local/cuda/bin/nvcc",
              os.path.join(os.environ.get("CUDA_HOME", "/nonexistent"), "bin", "nvcc")]:
        if p and os.path.exists(p):
            return p
    from shutil import which
    p = which("nvcc")
    if p:
        return p
    raise RuntimeError("nvcc not found")


def _build_cubin():
    src_path = os.path.join(_DIR, "dsa.cu")
    with open(src_path, "rb") as f:
        src = f.read()
    flags = ["-arch=sm_100a", "-O3", "-cubin", "-std=c++17", "-Xptxas", "-v"] + (["-DDEBUG_TS"] if os.environ.get("DSA_TS") else []) + os.environ.get("DSA_FLAGS", "").split()
    key = hashlib.sha256(src + " ".join(flags).encode()).hexdigest()[:16]
    cache_dir = os.path.join(_DIR, "_build")
    os.makedirs(cache_dir, exist_ok=True)
    cubin = os.path.join(cache_dir, f"dsa_{key}.cubin")
    if not os.path.exists(cubin):
        nvcc = _find_nvcc()
        cmd = [nvcc] + flags + ["-o", cubin + ".tmp", src_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed:\n" + r.stdout + r.stderr)
        if os.environ.get("DSA_VERBOSE"):
            print(r.stderr, file=sys.stderr)
        os.replace(cubin + ".tmp", cubin)
    with open(cubin, "rb") as f:
        return f.read()


class _Kernel:
    def __init__(self):
        torch.cuda.init()
        dev = torch.cuda.current_device()
        torch.zeros(1, device="cuda")  # make sure the primary context exists / is current
        _check(cu.cuInit(0))
        cudev = _check(cu.cuDeviceGet(dev))
        ctx = _check(cu.cuDevicePrimaryCtxRetain(cudev))
        _check(cu.cuCtxSetCurrent(ctx))
        image = _build_cubin()
        self.module = _check(cu.cuModuleLoadData(image))
        self.func = _check(cu.cuModuleGetFunction(self.module, b"dsa_kernel"))
        _check(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES))
        if CL > 8:
            _check(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
        self.nclusters = self._query_clusters()
        self.max_T = 0
        self._alloc(16)

    def _query_clusters(self):
        # hardware-only: how many CL-wide clusters can be co-resident with this kernel's resources
        n = None
        try:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = CL, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = NTHREADS, 1, 1
            cfg.sharedMemBytes = SMEM_BYTES
            attr = cu.CUlaunchAttribute()
            attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = CL, 1, 1
            cfg.attrs = [attr]; cfg.numAttrs = 1
            n = int(_check(cu.cuOccupancyMaxActiveClusters(self.func, cfg)))
        except Exception as e:  # pragma: no cover
            print("cuOccupancyMaxActiveClusters failed:", e, file=sys.stderr)
        if not n or n <= 0:
            n = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count // CL
        env = os.environ.get("DSA_NCLUSTERS")
        if env:
            n = int(env)
        if n > NCLUSTERS_CAP:
            n = NCLUSTERS_CAP
        if os.environ.get("DSA_VERBOSE"):
            print(f"nclusters={n} smem={SMEM_BYTES}", file=sys.stderr)
        return n

    def _alloc(self, T):
        # workspace depends only on shapes (num_tokens), never on input data
        self.part = torch.zeros(T * CPT * H * DC, dtype=torch.int32, device="cuda")   # tagged words: bf16 << 16 | gen16
        self.pml = torch.zeros(T * CPT * H * 2, dtype=torch.int32, device="cuda")      # tagged fp32 (low 8 bits = gen8)
        self.word = torch.zeros((T + 1) * 32, dtype=torch.int32, device="cuda")   # per-request completion counters + task claim counter (monotonic)
        self.gen = 0
        self.last_T = -1
        self.max_T = T
        self.ts = torch.zeros(self.nclusters * CL * 32, dtype=torch.int64, device="cuda") if os.environ.get("DSA_TS") else None

    def launch(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
        T = q_nope.shape[0]
        if T > self.max_T:
            self._alloc(T)
        if T != self.last_T:              # counters must restart when the request count changes (rare, untimed)
            self.word.zero_(); self.part.zero_(); self.pml.zero_(); self.gen = 0; self.last_T = T
        self.gen += 1
        if self.gen > 0xffff:             # 16-bit tags wrap: clear all tagged workspace (once per 65535 calls)
            self.word.zero_(); self.part.zero_(); self.pml.zero_(); self.gen = 1
        stream = torch.cuda.current_stream().cuda_stream
        args = (q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(), kpe_cache.data_ptr(), sparse_indices.data_ptr(),
                self.part.data_ptr(), self.pml.data_ptr(), self.word.data_ptr(),
                out.data_ptr(), lse.data_ptr(), float(sm_scale) * LOG2E, T, self.nclusters, self.gen,
                self.ts.data_ptr() if self.ts is not None else 0)
        types = (ctypes.c_void_p,) * 10 + (ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)
        _check(cu.cuLaunchKernel(self.func, self.nclusters * CL, 1, 1, NTHREADS, 1, 1, SMEM_BYTES, cu.CUstream(stream), (args, types), 0))


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _Kernel()
    return _kernel


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, h, dc = q_nope.shape
    assert h == H and dc == DC and q_pe.shape[-1] == DP and sparse_indices.shape[-1] == TOPK
    assert ckv_cache.shape[1] == 64 and ckv_cache.shape[2] == DC and kpe_cache.shape[2] == DP
    assert q_nope.dtype == torch.bfloat16 and ckv_cache.dtype == torch.bfloat16
    q_nope = q_nope.contiguous(); q_pe = q_pe.contiguous()
    sparse_indices = sparse_indices.contiguous()
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    if torch.is_tensor(sm_scale):
        sm_scale = float(sm_scale.item())
    out = torch.empty((T, H, DC), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((T, H), dtype=torch.float32, device=q_nope.device)
    k = _get_kernel()
    k.launch(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse)
    return out, lse
