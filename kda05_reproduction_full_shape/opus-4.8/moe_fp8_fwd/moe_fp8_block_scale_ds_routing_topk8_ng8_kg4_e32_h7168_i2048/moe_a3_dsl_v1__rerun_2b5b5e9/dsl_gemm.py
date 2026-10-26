"""Thin wrapper that drives our own copy of the CuTe DSL contiguous grouped
blockwise-FP8 GEMM (`cute_grouped_gemm.BlockwiseContiguousGroupedGemmKernel`) with
MoE tensors in our native orientations, zero-copy where possible.

The kernel is hand-written from cutlass.cute tcgen05/TMA/pipeline primitives
(sanctioned low-level building blocks); we build our own mainloop/epilogue/scheduler
and never use a device-level CollectiveBuilder GEMM template. Compile-once per (N,K);
M (valid_m) is fully dynamic so a single compiled kernel serves every seq_len
(verified: dsl_spike/a3_dynm).

Tensor contract (see docs/draft.md A2/A3):
  a   : [vm, K]        fp8_e4m3  (K contiguous)              -> cute [vm,K,1]
  b   : [N, K, L]      fp8_e4m3  (K contiguous, strided view of [L,N,K].permute(1,2,0))
  sfa : [K//128, vm]   fp32      (vm contiguous) passed as .t() -> cute [vm,K//128,1] (M-major)
  sfb : [N//128,K//128,L] fp32   (strided view of [L,N//128,K//128].permute(1,2,0))
  gidx: [vm]           int32     group (local-expert) id per row
  c   : [vm, N]        bf16      output                      -> cute [vm,N,1]
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

try:
    from .cute_grouped_gemm import BlockwiseContiguousGroupedGemmKernel
    from .cute_masked_gemm import BlockwiseMaskedGroupedGemmKernel
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cute_grouped_gemm import BlockwiseContiguousGroupedGemmKernel
    from cute_masked_gemm import BlockwiseMaskedGroupedGemmKernel


def _as_cute(t, elem):
    if elem is cutlass.Float8E4M3FN:
        ct = from_dlpack(t.view(torch.uint8), assumed_align=16)
        ct.element_type = elem
    else:
        ct = from_dlpack(t, assumed_align=16)
    ld = next((i for i, s in enumerate(t.stride()) if s == 1), None)
    return ct.mark_layout_dynamic(leading_dim=ld)


class GroupedGemm:
    def __init__(self, mma_tiler=(128, 128), cluster=(1, 1), use_2cta=False):
        self.mma_tiler = tuple(mma_tiler)
        self.cluster = tuple(cluster)
        self.use_2cta = use_2cta
        self._compiled = {}           # (N, K) -> compiled kernel
        self._max_clusters = None

    def _max_active_clusters(self):
        if self._max_clusters is None:
            hw = cutlass.utils.HardwareInfo()
            self._max_clusters = hw.get_max_active_clusters(self.cluster[0] * self.cluster[1])
        return self._max_clusters

    def run(self, a, b_nkl, sfa_kbm, sfb_nbkbl, gidx, c):
        """a[vm,K] fp8, b_nkl[N,K,L] fp8 view, sfa_kbm[K//128,vm] fp32,
        sfb_nbkbl[N//128,K//128,L] fp32 view, gidx[vm] int32, c[vm,N] bf16 (written)."""
        N = b_nkl.shape[0]
        K = b_nkl.shape[1]
        key = (N, K)

        a_c = _as_cute(a.unsqueeze(-1), cutlass.Float8E4M3FN)
        b_c = _as_cute(b_nkl, cutlass.Float8E4M3FN)
        c_c = _as_cute(c.unsqueeze(-1), cutlass.BFloat16)
        sfa_c = _as_cute(sfa_kbm.transpose(0, 1).unsqueeze(-1), cutlass.Float32)  # [vm,K//128,1] M-major
        sfb_c = _as_cute(sfb_nbkbl, cutlass.Float32)
        gidx_c = from_dlpack(gidx, assumed_align=16).mark_layout_dynamic()

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = self._compiled.get(key)
        if compiled is None:
            gemm = BlockwiseContiguousGroupedGemmKernel(
                cutlass.Float32, self.use_2cta, self.mma_tiler, self.cluster
            )
            compiled = cute.compile(
                gemm, a_c, b_c, c_c, sfa_c, sfb_c, gidx_c,
                self._max_active_clusters(), stream,
            )
            self._compiled[key] = compiled
        compiled(a_c, b_c, c_c, sfa_c, sfb_c, gidx_c, stream)
        return c


class MaskedGroupedGemm:
    """Masked grouped blockwise-FP8 GEMM: A/C laid out as [max_m, ., L] per-expert
    blocks of max_m rows, masked_m[L] valid rows each. The persistent scheduler skips
    tiles where cur_m >= masked_m[l], so inactive experts cost ~0 and GEMM work scales
    with the number of ACTIVE experts (unlike the contiguous path which reads all L).
    Static grid (grid = ceil(max_m/128) x N/128 x L) -> no D2H sync -> graph-capturable.

    Our physical buffers (expert l's rows at flat index l*max_m + m) map directly:
      a_flat[L*max_m, K] fp8   -> [max_m, K, L] via .view(L,max_m,K).permute(1,2,0)
      c_flat[L*max_m, N] (out) -> [max_m, N, L] via .view(L,max_m,N).permute(1,2,0)
      sfa_pool[K//128, L*max_m] -> [max_m, K//128, L] via .view(KB,L,max_m).permute(2,0,1)
      b_nkl[N,K,L], sfb[N//128,K//128,L]  (same strided weight views as contiguous)
      masked_m[L] int32 = per-expert valid row count (= expert_count)."""

    def __init__(self, mma_tiler=(128, 128), cluster=(1, 1), use_2cta=False):
        self.mma_tiler = tuple(mma_tiler)
        self.cluster = tuple(cluster)
        self.use_2cta = use_2cta
        self._compiled = {}
        self._max_clusters = None

    def _max_active_clusters(self):
        if self._max_clusters is None:
            hw = cutlass.utils.HardwareInfo()
            self._max_clusters = hw.get_max_active_clusters(self.cluster[0] * self.cluster[1])
        return self._max_clusters

    def run(self, a_flat, b_nkl, sfa_pool, sfb_nbkbl, masked_m, c_flat, L, max_m):
        N = b_nkl.shape[0]
        K = b_nkl.shape[1]
        KB = sfa_pool.shape[0]
        key = (N, K)

        a3 = a_flat.view(L, max_m, K).permute(1, 2, 0)          # [max_m, K, L], K-major
        c3 = c_flat.view(L, max_m, N).permute(1, 2, 0)          # [max_m, N, L], N-major
        sfa3 = sfa_pool.view(KB, L, max_m).permute(2, 0, 1)     # [max_m, KB, L], M-major
        a_c = _as_cute(a3, cutlass.Float8E4M3FN)
        b_c = _as_cute(b_nkl, cutlass.Float8E4M3FN)
        c_c = _as_cute(c3, cutlass.BFloat16)
        sfa_c = _as_cute(sfa3, cutlass.Float32)
        sfb_c = _as_cute(sfb_nbkbl, cutlass.Float32)
        mm_c = from_dlpack(masked_m, assumed_align=16).mark_layout_dynamic()

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = self._compiled.get(key)
        if compiled is None:
            gemm = BlockwiseMaskedGroupedGemmKernel(
                cutlass.Float32, self.use_2cta, self.mma_tiler, self.cluster
            )
            compiled = cute.compile(
                gemm, a_c, b_c, c_c, sfa_c, sfb_c, mm_c,
                self._max_active_clusters(), stream,
            )
            self._compiled[key] = compiled
        compiled(a_c, b_c, c_c, sfa_c, sfb_c, mm_c, stream)
        return c_flat
