// tvm-ffi entry point for the B200 FP8 block-scale MoE kernel.
// All device code and the launch sequence live in moe_device.cuh so that the
// profiling harness under profile/ can compile the same code with -lineinfo.

#include <mutex>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "moe_device.cuh"

namespace moe_b200 {

struct Pool {
  void* base = nullptr;
  size_t bytes = 0;
  std::mutex mu;
};
static Pool g_pool;

static Work acquire(int T, int ksplit1, int ksplit2) {
  const int cap = T;
  const int pairs_cap = T * TOPK;

  size_t o = 0;
  const size_t o_counts = o; o = align_up(o + sizeof(int) * E_LOCAL, 256);
  const size_t o_cnt2 = o;   o = align_up(o + sizeof(int) * E_LOCAL, 256);
  const size_t o_offs = o;   o = align_up(o + sizeof(int) * (E_LOCAL + 1), 256);
  const size_t o_g1s = o;    o = align_up(o + sizeof(int) * (E_LOCAL + 1), 256);
  const size_t o_g2s = o;    o = align_up(o + sizeof(int) * (E_LOCAL + 1), 256);
  const size_t o_meta = o;   o = align_up(o + sizeof(int) * 8, 256);
  const size_t head = o;     // everything above must start zeroed
  const size_t o_elist = o;  o = align_up(o + sizeof(int) * (size_t)E_LOCAL * cap, 256);
  const size_t o_ewt = o;    o = align_up(o + sizeof(float) * (size_t)E_LOCAL * cap, 256);
  const size_t o_cs = o;     o = align_up(o + sizeof(float) * (size_t)IB * pairs_cap, 256);
  const size_t o_cq = o;     o = align_up(o + (size_t)pairs_cap * I_SZ, 256);
  const size_t o_g1p = o;
  o = align_up(o + sizeof(__nv_bfloat16) * (size_t)ksplit1 * pairs_cap * G1N, 256);
  const size_t need = o;

  std::lock_guard<std::mutex> lk(g_pool.mu);
  if (g_pool.bytes < need) {
    if (g_pool.base) cudaFree(g_pool.base);
    cudaError_t err = cudaMalloc(&g_pool.base, need);
    if (err != cudaSuccess) {
      g_pool.base = nullptr;
      g_pool.bytes = 0;
      TVM_FFI_THROW(RuntimeError) << "cudaMalloc(" << need
                                  << ") failed: " << cudaGetErrorString(err);
    }
    g_pool.bytes = need;
    // counts/meta start at zero; every launch afterwards leaves them at zero.
    cudaMemset(g_pool.base, 0, head);
  }

  char* p = static_cast<char*>(g_pool.base);
  Work w;
  w.counts = reinterpret_cast<int*>(p + o_counts);
  w.cnt2 = reinterpret_cast<int*>(p + o_cnt2);
  w.offs = reinterpret_cast<int*>(p + o_offs);
  w.g1_start = reinterpret_cast<int*>(p + o_g1s);
  w.g2_start = reinterpret_cast<int*>(p + o_g2s);
  w.meta = reinterpret_cast<int*>(p + o_meta);
  w.elist = reinterpret_cast<int*>(p + o_elist);
  w.ewt = reinterpret_cast<float*>(p + o_ewt);
  w.cs = reinterpret_cast<float*>(p + o_cs);
  w.cq = reinterpret_cast<__nv_fp8_e4m3*>(p + o_cq);
  w.g1p = reinterpret_cast<__nv_bfloat16*>(p + o_g1p);
  w.cap = cap;
  w.pairs_cap = pairs_cap;
  w.ksplit1 = ksplit1;
  w.ksplit2 = ksplit2;
  return w;
}

void MoeRun(tvm::ffi::TensorView routing_logits, tvm::ffi::TensorView routing_bias,
            tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView hidden_states_scale,
            tvm::ffi::TensorView gemm1_weights, tvm::ffi::TensorView gemm1_weights_scale,
            tvm::ffi::TensorView gemm2_weights, tvm::ffi::TensorView gemm2_weights_scale,
            int64_t local_expert_offset, double routed_scaling_factor,
            tvm::ffi::TensorView output) {
  const int T = static_cast<int>(routing_logits.size(0));
  if (routing_logits.size(1) != E_GLOBAL)
    TVM_FFI_THROW(ValueError) << "num_experts must be " << E_GLOBAL;
  if (gemm1_weights.size(0) != E_LOCAL)
    TVM_FFI_THROW(ValueError) << "num_local_experts must be " << E_LOCAL;
  if (T <= 0) return;

  DLDevice dev = hidden_states.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Tiling picked by sweeping (M_TILE, CTAs/SM, ksplit) on this B200; see
  // profile/moe_v2_tune. Small seq_len is weight-bandwidth bound and wants the
  // smallest tile that still amortises the weight read (M_TILE=8) plus a K split
  // to fill the machine when only a few experts are live. Large seq_len instead
  // wants a large M_TILE to avoid re-reading the weights, but registers cap us
  // at 32 (see docs/draft.md Phase 2 item 1).
  const bool small = (T <= 128);
  const int ks1 = small ? 4 : 1;
  const int ks2 = small ? 2 : 1;

  Work w = acquire(T, ks1, ks2);

  const void* lg = routing_logits.data_ptr();
  const void* bs = routing_bias.data_ptr();
  const void* hs = hidden_states.data_ptr();
  const void* hss = hidden_states_scale.data_ptr();
  const void* g1w = gemm1_weights.data_ptr();
  const void* g1s = gemm1_weights_scale.data_ptr();
  const void* g2w = gemm2_weights.data_ptr();
  const void* g2s = gemm2_weights_scale.data_ptr();
  void* op = output.data_ptr();
  const int off = static_cast<int>(local_expert_offset);
  const float rsf = static_cast<float>(routed_scaling_factor);

  if (small)
    launch<8, 8>(T, w, stream, lg, bs, hs, hss, g1w, g1s, g2w, g2s, op, off, rsf);
  else
    launch<32, 4>(T, w, stream, lg, bs, hs, hss, g1w, g1s, g2w, g2s, op, off, rsf);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_run, MoeRun);

}  // namespace moe_b200
