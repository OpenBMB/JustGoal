#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void launch_route(const float *logits, const void *bias, float *tokw, int *cnt, int *ptok,
                  float *pw, void *out, int *tokr, int *flag, int *base, int *mt, void *pt,
                  int64_t p2,
                  int64_t p3, int64_t p4, int64_t p5, int64_t p6, int64_t p7, int T, int E,
                  int EL, int ELP, int OFF, float RSF, int PMAX, int H, int ZOUT,
                  cudaStream_t stream);

void route(torch::Tensor logits, torch::Tensor bias, torch::Tensor tokw, torch::Tensor cnt,
           torch::Tensor ptok, torch::Tensor pw, torch::Tensor out, torch::Tensor tokr,
           torch::Tensor flag,
           torch::Tensor base, torch::Tensor mt, torch::Tensor pt, int64_t p2, int64_t p3,
           int64_t p4, int64_t p5, int64_t p6, int64_t p7, int64_t EL, int64_t ELP,
           int64_t OFF, double RSF, int64_t PMAX, int64_t ZOUT) {
  launch_route(logits.data_ptr<float>(), bias.data_ptr(), tokw.data_ptr<float>(),
               cnt.data_ptr<int>(), ptok.data_ptr<int>(), pw.data_ptr<float>(), out.data_ptr(),
               tokr.data_ptr<int>(), flag.data_ptr<int>(), base.data_ptr<int>(), mt.data_ptr<int>(), pt.data_ptr(),
               p2, p3, p4, p5, p6, p7, (int)logits.size(0), (int)logits.size(1), (int)EL,
               (int)ELP, (int)OFF, (float)RSF, (int)PMAX, (int)out.size(1), (int)ZOUT,
               at::cuda::getCurrentCUDAStream());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("route", &route); }
