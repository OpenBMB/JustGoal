"""Loader + callable wrapper for the hand-written CUDA DSA FWD kernel."""

import os

# Must be set before cpp_extension builds: otherwise torch appends its default
# arch list (down to sm_75), where bf16 mma.m16n8k16 does not exist - the build
# fails and, worse, compiling every arch takes tens of minutes.
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_MOD = None


def get_module(verbose=True):
    global _MOD
    if _MOD is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _MOD = load(
            name="dsa_cuda_ext",
            sources=[os.path.join(here, "dsa_kernel.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-Xptxas=-v"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=verbose,
        )
    return _MOD


class DsaCuda:
    """Callable with pre-allocated scratch so the timed path is launch-only.

    block_n: KV index slots handled by one partial-kernel CTA.
    num_warps: warps per partial-kernel CTA.
    """

    def __init__(self, w, block_n=32, num_warps=8, head_groups=2, tiles_per_cta=1, fused=True,
                 verbose=False):
        self.mod = get_module(verbose)
        self.block_n = int(block_n)
        self.num_warps = int(num_warps)
        self.head_groups = int(head_groups)
        self.tpp = int(tiles_per_cta)
        self.fused = bool(fused)
        self.q_nope = w["q_nope"]
        self.q_pe = w["q_pe"]
        T, H, D = self.q_nope.shape
        DPE = self.q_pe.shape[-1]
        self.ckv = w["ckv_cache"].reshape(-1, D)
        self.kpe = w["kpe_cache"].reshape(-1, DPE)
        self.indices = w["sparse_indices"]
        self.sm_scale = w["sm_scale"]
        topk = self.indices.shape[1]
        self.nsplit = (topk + self.block_n - 1) // self.block_n
        dev = self.q_nope.device
        self.o_part = torch.zeros((T, self.nsplit, H, D), device=dev, dtype=torch.bfloat16)
        # [T, H, nsplit]: the combine reads one head's splits contiguously.
        self.lse = torch.full((T, H, self.nsplit), float("-inf"), device=dev, dtype=torch.float32)
        self.out = torch.empty((T, H, D), device=dev, dtype=torch.bfloat16)
        # Fused (atomic) path: double-buffered numerator/denominator so nothing
        # needs zeroing between launches, plus two monotonic counters. A CTA
        # derives its generation from the ticket it draws, which makes the whole
        # scheme CUDA-graph safe.
        self.num = torch.zeros((T, H, D), device=dev, dtype=torch.float32)
        self.den = torch.zeros((2, T, H), device=dev, dtype=torch.float32)
        # 32-int stride: one counter per 128-byte line, so the two requests'
        # arrival counters do not share a line.
        self.tick = torch.zeros((T * 32,), device=dev, dtype=torch.int32)
        self.done = torch.zeros((T * 32,), device=dev, dtype=torch.int32)

    def pointers(self):
        return {k: hex(getattr(self, k).data_ptr())
                for k in ("num", "den", "tick", "done", "out")}

    def __repr__(self):
        return (f"DsaCuda(bn={self.block_n}, warps={self.num_warps}, hg={self.head_groups}, "
                f"tpp={self.tpp}, grid_splits={self.nsplit // self.tpp}, fused={self.fused})")

    def smem_bytes(self):
        return self.mod.dsa_partial_smem(self.block_n)

    def probe(self, mode):
        """Cost decomposition: 0 = launch only, 1 = + index read, 2 = + KV staging."""
        sink = torch.zeros(self.nsplit, device=self.out.device, dtype=torch.float32)

        def run():
            self.mod.dsa_probe(self.ckv, self.kpe, self.indices, sink, 32, mode)
        return run

    def __call__(self, stop=0):
        self.mod.dsa_fwd(self.q_nope, self.q_pe, self.ckv, self.kpe, self.indices,
                         self.o_part, self.lse, self.num, self.den, self.out,
                         self.tick, self.done, self.sm_scale,
                         self.block_n, self.num_warps, self.head_groups, self.tpp, self.fused,
                         stop)
        if self.fused:
            final_lse = self.lse.flatten()[:self.out.shape[0] * self.out.shape[1]].view(self.out.shape[:2])
        else:
            final_lse = torch.logsumexp(self.lse, dim=-1) * 1.4426950408889634
        return self.out, final_lse

    def staged(self, stop):
        """Callable that stops after stage `stop` (see the kernel's `stop` switch)."""
        def run():
            self.__call__(stop)
        return run


class DsaCudaCluster:
    """Cluster variant: one thread-block cluster per request, cross-CTA reduction
    over distributed shared memory behind a hardware barrier."""

    def __init__(self, w, block_n=32, verbose=False):
        self.mod = get_module(verbose)
        self.block_n = int(block_n)
        self.q_nope = w["q_nope"]
        self.q_pe = w["q_pe"]
        T, H, D = self.q_nope.shape
        DPE = self.q_pe.shape[-1]
        self.ckv = w["ckv_cache"].reshape(-1, D)
        self.kpe = w["kpe_cache"].reshape(-1, DPE)
        self.indices = w["sparse_indices"]
        self.sm_scale = w["sm_scale"]
        self.out = torch.empty((T, H, D), device=self.q_nope.device, dtype=torch.bfloat16)

    def __repr__(self):
        return f"DsaCudaCluster(bn={self.block_n}, cluster=16)"

    def __call__(self):
        self.mod.dsa_fwd_cluster(self.q_nope, self.q_pe, self.ckv, self.kpe, self.indices,
                                 self.out, self.sm_scale, self.block_n)
        return self.out
