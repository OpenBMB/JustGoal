// Host-side helper to build CUtensorMap via the driver entry point (no -lcuda needed).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode_fn() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qres;
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qres);
    if (e != cudaSuccess || p == nullptr) {
      fprintf(stderr, "failed to get cuTensorMapEncodeTiled: %d\n", (int)e);
      exit(1);
    }
    fn = (PFN_encodeTiled)p;
  }
  return fn;
}

// rank-N bf16 tensor map, 128B swizzle. dims[0] is the contiguous dim (elements), strides in bytes for dims 1..N-1.
static int make_tmap_bf16(CUtensorMap* tm, int rank, const cuuint64_t* dims, const cuuint64_t* strides_bytes,
                          const cuuint32_t* box) {
  cuuint32_t estr[5] = {1, 1, 1, 1, 1};
  CUresult r = get_encode_fn()(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)nullptr /*set below*/, dims,
                               strides_bytes, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}
static int make_tmap_bf16_ptr(CUtensorMap* tm, void* gptr, int rank, const cuuint64_t* dims,
                              const cuuint64_t* strides_bytes, const cuuint32_t* box) {
  cuuint32_t estr[5] = {1, 1, 1, 1, 1};
  CUresult r = get_encode_fn()(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gptr, dims, strides_bytes, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}
