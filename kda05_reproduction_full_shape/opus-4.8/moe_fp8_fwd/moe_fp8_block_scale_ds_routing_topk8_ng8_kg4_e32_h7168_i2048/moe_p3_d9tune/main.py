"""MoE FP8 block-scale solution.

Phase 3: shape-specialized dispatch.
- tiny/mid seq (T <= GRAPH_MAX): CUDA-graph-capture the VENDOR FULLY-FUSED kernel
  (flashinfer.fused_moe.trtllm_fp8_block_scale_moe -- routing + GEMM1 + SwiGLU + GEMM2
  + weighted combine, all internal). Captured once per (T, input ptrs) and replayed:
  one graph launch, no host sync, and the kernel internally fuses SwiGLU + avoids the
  intermediate [R,4096] HBM roundtrip that our earlier decomposed graph paid. De-risk
  (scripts/derisk_graph_baseline.py) showed the graphed fused kernel is 1.15-1.85x
  faster per replay than our 8-kernel decomposition on every dev workload, with ZERO
  correctness risk (it is the reference kernel). Replaces the decomposed graph path
  (previous decomposed implementation preserved in git at commit da8beef).
- large seq (T > GRAPH_MAX): eager D4 path (custom compaction + 2 flashinfer grouped
  GEMMs + fused bf16 scatter-accumulate finalize); already beats baseline on 11948/14107
  via the finalize fusion (removes ~1GB out_f32+cast roundtrip). At that size host gap
  ~0, so graphing gives nothing; keep the proven eager path.
"""
import os
import torch
from torch.utils.cpp_extension import load
import flashinfer.gemm as fg
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from flashinfer.autotuner import autotune

_CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None
GRAPH_MAX = int(os.environ.get("MOE_GRAPH_MAX", "2048"))
H, I, G1OUT, EL, HB, IB = 7168, 2048, 4096, 32, 56, 16
# routing hyper-params (constant for this definition)
NUM_EXPERTS_GLOBAL, TOP_K, N_GROUP, TOPK_GROUP, INTERMEDIATE_SIZE = 256, 8, 8, 4, 2048


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="moe_fp8_phase3_graph",
            sources=[os.path.join(_THIS_DIR, "moe_kernels.cu")],
            extra_include_paths=[os.path.join(_CUDA_HOME, "include")],
            extra_cuda_cflags=["-O3"],
            verbose=bool(int(os.environ.get("MOE_VERBOSE", "0"))),
        )
    return _ext


def _as_int(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def _as_float(x):
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def _grp(a, b, a_scale, b_scale, m_indptr, out=None, mma_sm=1):
    return fg.group_gemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, m_indptr,
        scale_granularity_mnk=(1, 128, 128), scale_major_mode="K", mma_sm=mma_sm,
        out=out, out_dtype=torch.bfloat16, backend="trtllm")


_graphs = {}  # key -> (graph, out_buffer)
_tuned = set()  # eager-path GEMM shapes already autotuned (flashinfer caches best config)


def _autotune_eager(a_fp8, w13, a_scale, s13, c_fp8, c_scale, w2, s2, offsets):
    """One-time flashinfer autotune profiling for the eager large-T grouped GEMMs.
    De-risk (scripts/derisk_largegemm.py) showed trtllm+autotune cuts GEMM2 ~15% /
    full ~+3.4% (14107) / +1.9% (11948) vs the default mma_sm=1 config, with ZERO
    numerical change (same kernel, better tactic). Runs during the harness warmup
    runs only; subsequent calls auto-use the cached best config (even outside the
    autotune context), so the timed iterations pay nothing extra."""
    with autotune(True):
        for _ in range(4):
            _g1 = _grp(a_fp8, w13, a_scale, s13, offsets)
            _grp(c_fp8, w2, c_scale, s2, offsets)
    torch.cuda.synchronize()


def _fused(logits, bias, hid, hscale, w13, s13, w2, s2, offset, lne, rsf):
    """Vendor fully-fused FP8 block-scale MoE (the reference kernel)."""
    return trtllm_fp8_block_scale_moe(
        logits.to(torch.float32).contiguous(),
        bias.contiguous() if bias is not None else None,
        hid.contiguous(),
        hscale.to(torch.float32).contiguous(),
        w13.contiguous(),
        s13.to(torch.float32).contiguous(),
        w2.contiguous(),
        s2.to(torch.float32).contiguous(),
        NUM_EXPERTS_GLOBAL, TOP_K, N_GROUP, TOPK_GROUP, INTERMEDIATE_SIZE,
        offset, lne, rsf,
        routing_method_type=2, use_shuffled_weight=False)


def _build_fused_graph(logits, bias, hid, hscale, w13, s13, w2, s2, offset, lne, rsf):
    def body():
        return _fused(logits, bias, hid, hscale, w13, s13, w2, s2, offset, lne, rsf)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):  # warm up autotuner + workspace allocation before capture
            out = body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = body()
    return g, out


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    ext = _get_ext()
    offset = _as_int(local_expert_offset)
    rsf = _as_float(routed_scaling_factor)
    T = routing_logits.size(0)
    dev = hidden_states.device
    lne = gemm1_weights.size(0)

    if T == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=dev)

    if T <= GRAPH_MAX:
        key = (T, routing_logits.data_ptr(),
               routing_bias.data_ptr() if routing_bias is not None else 0,
               hidden_states.data_ptr(), hidden_states_scale.data_ptr(),
               gemm1_weights.data_ptr(), gemm1_weights_scale.data_ptr(),
               gemm2_weights.data_ptr(), gemm2_weights_scale.data_ptr(), offset, rsf)
        ent = _graphs.get(key)
        if ent is None:
            ent = _build_fused_graph(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                offset, lne, rsf)
            _graphs[key] = ent
        g, out = ent
        g.replay()
        return out.clone()

    # eager large-seq path (custom compaction + grouped GEMMs + fused finalize)
    logits = routing_logits.to(torch.float32).contiguous()
    bias = routing_bias.to(torch.bfloat16).contiguous()
    hid = hidden_states.contiguous()
    hscale = hidden_states_scale.to(torch.float32).contiguous()
    w13 = gemm1_weights.contiguous()
    s13 = gemm1_weights_scale.to(torch.float32).contiguous()
    w2 = gemm2_weights.contiguous()
    s2 = gemm2_weights_scale.to(torch.float32).contiguous()

    a_fp8, a_scale, perm_token, perm_w, offsets, tok_rows, tok_w, tok_n = ext.route_gather(
        logits, bias, hid, hscale, offset, rsf)
    if a_fp8.size(0) == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
    g1 = _grp(a_fp8, w13, a_scale, s13, offsets)
    c_fp8, c_scale = ext.swiglu_quant(g1)
    sig = (a_fp8.size(0),)  # R uniquely keys the grouped-GEMM problem shape
    if sig not in _tuned:
        _autotune_eager(a_fp8, w13, a_scale, s13, c_fp8, c_scale, w2, s2, offsets)
        _tuned.add(sig)
    o = _grp(c_fp8, w2, c_scale, s2, offsets)
    return ext.finalize(o, tok_rows, tok_w, tok_n, T)
