// Grouped FP8 block-scale GEMMs for MoE on SM100 (B200).
// Warp-specialized, continuous (never-draining) TMA pipeline over a flat (tile,k) stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;
using e4m3 = cutlass::float_e4m3_t;
#define DEVI __device__ __forceinline__
#define SLEEPNS 8

DEVI uint32_t su32(void const* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su32(b)), "r"(c));
}
DEVI void mbar_expect(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(b)), "r"(n));
}
DEVI void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su32(b)));
}
DEVI void mbar_wait(uint64_t* b, uint32_t ph) {
  uint32_t a = su32(b);
  asm volatile("{\n.reg .pred p;\nLW%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra LW%=;\n}\n" :: "r"(a), "r"(ph));
}
DEVI void mbar_wait_s(uint64_t* b, uint32_t ph) {
  uint32_t a = su32(b);
  asm volatile("{\n.reg .pred p;\nLS%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@p bra DS%=;\nnanosleep.u32 %2;\nbra LS%=;\nDS%=:\n}\n"
               :: "r"(a), "r"(ph), "n"(SLEEPNS));
}
DEVI void tma2(void const* m, uint64_t* b, void* d, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];"
               :: "r"(su32(d)), "l"(m), "r"(su32(b)), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma3(void const* m, uint64_t* b, void* d, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(su32(d)), "l"(m), "r"(su32(b)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma3_mc(void const* m, uint32_t bar, void* d, int c0, int c1, int c2, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               ".multicast::cluster [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(su32(d)), "l"(m), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "h"(mask) : "memory");
}
DEVI void umma_mma(uint64_t da, uint64_t db, uint32_t tc, uint32_t sc, uint32_t idesc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}"
               :: "r"(tc), "l"(da), "l"(db), "r"(idesc), "r"(sc),
                  "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}
DEVI void umma_commit(uint64_t* b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(su32(b)));
}
DEVI void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEVI uint32_t ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(addr), "r"(rank)); return o;
}
DEVI void mbar_arrive_remote(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(addr) : "memory");
}
DEVI void tma2_mc(void const* m, uint32_t bar, void* d, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(su32(d)), "l"(m), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
DEVI void my_cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
DEVI void cbar(int n) { asm volatile("bar.sync 1, %0;" :: "r"(n) : "memory"); }

DEVI void tmem_ld32(uint32_t addr, float* d) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=f"(d[0]),
        "=f"(d[1]),
        "=f"(d[2]),
        "=f"(d[3]),
        "=f"(d[4]),
        "=f"(d[5]),
        "=f"(d[6]),
        "=f"(d[7]),
        "=f"(d[8]),
        "=f"(d[9]),
        "=f"(d[10]),
        "=f"(d[11]),
        "=f"(d[12]),
        "=f"(d[13]),
        "=f"(d[14]),
        "=f"(d[15]),
        "=f"(d[16]),
        "=f"(d[17]),
        "=f"(d[18]),
        "=f"(d[19]),
        "=f"(d[20]),
        "=f"(d[21]),
        "=f"(d[22]),
        "=f"(d[23]),
        "=f"(d[24]),
        "=f"(d[25]),
        "=f"(d[26]),
        "=f"(d[27]),
        "=f"(d[28]),
        "=f"(d[29]),
        "=f"(d[30]),
        "=f"(d[31])
      : "r"(addr));
}
DEVI void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }


static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 128;
static constexpr int TILE_BYTES = BM * BK;
static constexpr int EPAD = 136;
static constexpr int NBUF = 4;

using SmemAtom   = decltype(UMMA::Layout_K_SW128_Atom<e4m3>{});
using SmemLayout = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BM>, Int<BK>>{}));
using MmaTraitsT = MMA_Traits<SM100_MMA_F8F6F4_SS, e4m3, e4m3, float,
                              cute::C<BM>, cute::C<BN>,
                              cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                              cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                              cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                              cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
using MmaAtom = MMA_Atom<MmaTraitsT>;

template <int SA, int SB, int NKMAX, int ER>
struct SharedStorage {
  alignas(1024) uint8_t sA[SA][2 * TILE_BYTES];
  alignas(1024) uint8_t sB[SB][2 * TILE_BYTES];
  alignas(1024) __nv_bfloat16 sEpi[ER][EPAD];
  alignas(1024) __nv_bfloat16 sScale[NKMAX][BM];
  alignas(16) uint64_t dA[SA][2][4];
  alignas(16) uint64_t dB[SB][2][4];
  alignas(8) uint64_t fullA[SA];
  alignas(8) uint64_t fullB[SB];
  alignas(8) uint64_t emptyA[SA];
  alignas(8) uint64_t mmaBar[8];
  alignas(8) uint64_t freeBar[NBUF];
  alignas(16) uint32_t tmemPtr[4];
  alignas(4) uint32_t idesc_s;
};

template <int MODE, int NK, int NTN, int NCOL, int BROWS, int SA, int SB, int NCONS, int CSZ, int ER, int VAR>
__global__ __launch_bounds__(64 + NCONS, 1)
void moe_gemm_kernel(const __grid_constant__ CUtensorMap tmapA,
                     const __grid_constant__ CUtensorMap tmapB,
                     const float* __restrict__ Asc, const float* __restrict__ Wsc,
                     __nv_bfloat16* __restrict__ Out,
                     const int* __restrict__ tile_e, const int* __restrict__ tile_base,
                     const int* __restrict__ tile_j0, const int* __restrict__ cnt,
                     const float* __restrict__ row_w, const int* __restrict__ ntiles_p,
                     int MR, int MAXC, int NPROG) {
  extern __shared__ char smem_raw[];
  SharedStorage<SA, SB, NK, ER>& ss = *reinterpret_cast<SharedStorage<SA, SB, NK, ER>*>(smem_raw);
  constexpr int NACC = BN * 128 / NCONS;

  const int tid = threadIdx.x;
  const bool prod = tid >= NCONS;
  const bool wmma = (tid >= NCONS) && (tid < NCONS + 32);
  const bool wtma = tid >= NCONS + 32;
  const int row = tid & 127;
  const int part = (tid >> 7) & ((NCONS / 128) - 1);
  const int rowbase = (tid & 127) & ~31;
  cudaGridDependencySynchronize();
  const int bid = blockIdx.x;
  const int rank = bid % CSZ;
  const int cl = bid / CSZ;
  const int NCLUST = NPROG / CSZ;
  const int total = (*ntiles_p) * NTN;
  const int totalC = total / CSZ;
  const int ntile_cta = cl < totalC ? (totalC - cl + NCLUST - 1) / NCLUST : 0;
  const int TOTQ = ntile_cta * NK;
  const uint16_t MCMASK = (uint16_t)((1u << CSZ) - 1u);

  if (wmma) cute::TMEM::Allocator1Sm().allocate(512, ss.tmemPtr);
  if (tid == NCONS) {
    TiledMMA tiled_mma = make_tiled_mma(MmaAtom{});
    auto thr_mma = tiled_mma.get_slice(0);
    CUTE_UNROLL
    for (int st = 0; st < SA; ++st) {
      CUTE_UNROLL
      for (int h = 0; h < 2; ++h) {
      Tensor t = make_tensor(make_smem_ptr((e4m3*)(ss.sA[st] + h * TILE_BYTES)), SmemLayout{});
      Tensor f = thr_mma.make_fragment_A(thr_mma.partition_A(t));
      CUTE_UNROLL
      for (int kk = 0; kk < 4; ++kk) ss.dA[st][h][kk] = (uint64_t)f(_, _, kk)[0];
      }
    }
    CUTE_UNROLL
    for (int st = 0; st < SB; ++st) {
      CUTE_UNROLL
      for (int h = 0; h < 2; ++h) {
      Tensor t = make_tensor(make_smem_ptr((e4m3*)(ss.sB[st] + h * TILE_BYTES)), SmemLayout{});
      Tensor f = thr_mma.make_fragment_B(thr_mma.partition_B(t));
      CUTE_UNROLL
      for (int kk = 0; kk < 4; ++kk) ss.dB[st][h][kk] = (uint64_t)f(_, _, kk)[0];
      }
    }
    ss.idesc_s = (uint32_t)(UMMA::make_runtime_instr_desc(tiled_mma.idesc_) >> 32);
    CUTE_UNROLL
    for (int i = 0; i < SA; ++i) { mbar_init(&ss.fullA[i], 1); mbar_init(&ss.emptyA[i], CSZ); }
    CUTE_UNROLL
    for (int i = 0; i < SB; ++i) mbar_init(&ss.fullB[i], 1);
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) mbar_init(&ss.mmaBar[i], 1);
    for (int i = 0; i < NBUF; ++i) mbar_init(&ss.freeBar[i], NCONS / 32);
    fence_async_shared();
  }
  __syncthreads();
  my_cluster_sync();
  const uint32_t tmem_base = ss.tmemPtr[0];
  const uint32_t tld = tmem_base + ((((tid >> 5) & 3) * 32u) << 16) + part * NACC;
  const uint32_t idesc = ss.idesc_s;

  if (wmma) {
    // ------------------------------------------------ MMA issue warp
    if (tid == NCONS) {
      uint32_t phA = 0, phB = 0, phF = 0;
      const int NGm = TOTQ >> 1;
      for (int g = 0; g < NGm; ++g) {
        const int sa = g % SA, sb = g % SB, gb = g & 1, mb = g & 7;
        mbar_wait(&ss.fullA[sa], (phA >> sa) & 1u); phA ^= (1u << sa);
        (void)sb; (void)phB;
        if (VAR != 4 && g >= 2) { mbar_wait(&ss.freeBar[gb], (phF >> gb) & 1u); phF ^= (1u << gb); }
        if (VAR >= 2) { mbar_arrive(&ss.mmaBar[mb]); }
        else {
          CUTE_UNROLL
          for (int h = 0; h < 2; ++h) {
            const uint32_t tc = tmem_base + ((2 * g + h) & (NBUF - 1)) * BN;
            CUTE_UNROLL
            for (int kk = 0; kk < 4; ++kk)
              umma_mma(ss.dA[sa][h][kk], ss.dB[sb][h][kk], tc, kk == 0 ? 0u : 1u, idesc);
          }
          umma_commit(&ss.mmaBar[mb]);
        }
      }
    }
  } else if (wtma) {
    // ------------------------------------------------ TMA issue warp
    if (tid == NCONS + 32) {
      uint32_t phM = 0, phE = 0;
      const uint32_t e0_rank0 = mapa(su32(&ss.emptyA[0]), 0);
      const int NG = TOTQ >> 1;
      CUTE_UNROLL
      for (int gs = 0; gs < SA; ++gs) if (gs < NG) {
        if (VAR == 3) mbar_expect(&ss.fullA[gs], 2 * TILE_BYTES);
        else mbar_expect(&ss.fullA[gs], 4 * TILE_BYTES);
        mbar_arrive_remote(e0_rank0 + gs * 8);
      }
      CUTE_UNROLL
      for (int gs = 0; gs < SA; ++gs) if (gs < NG) {
        if (rank == 0 && VAR != 3) {
          mbar_wait(&ss.emptyA[gs], (phE >> gs) & 1u);
          const int q0 = gs * 2, ti = q0 / NK, k = q0 - ti * NK;
          const int bse = tile_base[((cl + ti * NCLUST) * CSZ) / NTN];
          tma3_mc(&tmapA, su32(&ss.fullA[gs]), ss.sA[gs], 0, bse, k, MCMASK);
        }
      }
      phE = (1u << SA) - 1u;
      CUTE_UNROLL
      for (int gs = 0; gs < SB; ++gs) if (gs < NG) {
        const int q0 = gs * 2, ti = q0 / NK, k = q0 - ti * NK;
        const int w = (cl + ti * NCLUST) * CSZ + rank;
        const int tq = w / NTN, nq = w - tq * NTN;
        const int gb = tile_e[tq] * BROWS + nq * BN;
        tma3(&tmapB, &ss.fullA[gs], ss.sB[gs], 0, gb, k);
      }
      int tiA = -1, baseA = 0, tiB = -1, growBc = 0;
      for (int g = 0; g + 1 < NG; ++g) {
        const int pb = g & 7;
        mbar_wait(&ss.mmaBar[pb], (phM >> pb) & 1u); phM ^= (1u << pb);
        const int na = g + SA;
        const int ps = g % SA;
        if (na < NG) {
          if (VAR == 3) mbar_expect(&ss.fullA[ps], 2 * TILE_BYTES);
          else mbar_expect(&ss.fullA[ps], 4 * TILE_BYTES);
          mbar_arrive_remote(e0_rank0 + ps * 8);
          if (rank == 0 && VAR != 3) {
            mbar_wait(&ss.emptyA[ps], (phE >> ps) & 1u); phE ^= (1u << ps);
            const int q0 = na * 2, ti = q0 / NK, k = q0 - ti * NK;
            if (ti != tiA) { tiA = ti; baseA = tile_base[((cl + ti * NCLUST) * CSZ) / NTN]; }
            tma3_mc(&tmapA, su32(&ss.fullA[ps]), ss.sA[ps], 0, baseA, k, MCMASK);
          }
        }
        const int nb = g + SB;
        if (nb < NG) {
          const int q0 = nb * 2, ti = q0 / NK, k = q0 - ti * NK;
          if (ti != tiB) {
            tiB = ti;
            const int w = (cl + ti * NCLUST) * CSZ + rank;
            const int tq = w / NTN, nq = w - tq * NTN;
            growBc = tile_e[tq] * BROWS + nq * BN;
          }
          const int pz = g % SB;
          tma3(&tmapB, &ss.fullA[pz], ss.sB[pz], 0, growBc, k);
        }
      }
    }
  } else if (VAR == 4) {
    // consumers disabled (timing probe)
  } else {
    // ------------------------------------------------ consumers
    float cacc[NACC];
    uint32_t phM = 0;
    int e = 0, base = 0, nq = 0, tq = 0, ccnt = 0;
    const int NGc = TOTQ >> 1;
    for (int g = 0; g < NGc; ++g) {
      const int q0 = g << 1;
      const int ti = q0 / NK, k0 = q0 - ti * NK;
      if (k0 == 0) {
        const int w = (cl + ti * NCLUST) * CSZ + rank;
        tq = w / NTN; nq = w - tq * NTN;
        e = tile_e[tq]; base = tile_base[tq]; ccnt = cnt[e];
        cbar(NCONS);
        const float* wp = Wsc + (size_t)e * (NCOL / 128) * NK + (size_t)nq * NK;
        const int nrows = min(BM, ccnt - tile_j0[tq]);
        for (int idx = tid; idx < NK * BM; idx += NCONS) {
          const int pp = idx >> 7, rr = idx & 127;
          if (rr < nrows)
            ss.sScale[pp][rr] = __float2bfloat16(Asc[(size_t)pp * MR + base + rr] * wp[pp]);
        }
        CUTE_UNROLL
        for (int i = 0; i < NACC; ++i) cacc[i] = 0.f;
      }
      const int gb = g & 1, mb = g & 7;
      if (tid == 0) mbar_wait_s(&ss.mmaBar[mb], (phM >> mb) & 1u);
      cbar(NCONS);
      phM ^= (1u << mb);
      if (VAR == 0 && rowbase < ccnt) {
        CUTE_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          const int bf = (q0 + hh) & (NBUF - 1);
          const float sc = __bfloat162float(ss.sScale[k0 + hh][row]);
          const uint32_t ta = tld + bf * BN;
          float r[NACC];
          CUTE_UNROLL
          for (int ci = 0; ci < NACC / 32; ++ci) tmem_ld32(ta + ci * 32, r + ci * 32);
          tmem_wait_ld();
          CUTE_UNROLL
          for (int j = 0; j < NACC; ++j) cacc[j] += r[j] * sc;
        }
      }
      __syncwarp();
      if ((tid & 31) == 0) mbar_arrive(&ss.freeBar[gb]);
      if (k0 + 2 == NK) {
        const int j0 = tile_j0[tq], n0 = nq * BN;
        float ow = 1.f;
        if (MODE == 1) ow = row_w[(size_t)e * MAXC + j0 + row];
        const int c = ccnt;
        const int r0 = tid / 16, c0 = (tid % 16) * 8;
        CUTE_UNROLL
        for (int ch = 0; ch < BM / ER; ++ch) {
          if (j0 + ch * ER >= c) break;
          if (ch > 0) cbar(NCONS);
          if (row >= ch * ER && row < ch * ER + ER) {
            CUTE_UNROLL
            for (int i = 0; i < NACC; ++i)
              ss.sEpi[row - ch * ER][part * NACC + i] = __float2bfloat16(cacc[i] * ow);
          }
          cbar(NCONS);
          CUTE_UNROLL
          for (int rr = r0; rr < ER; rr += NCONS / 16) {
            const int gr = ch * ER + rr;
            if (j0 + gr < c) {
              uint4 v = *reinterpret_cast<const uint4*>(&ss.sEpi[rr][c0]);
              *reinterpret_cast<uint4*>(&Out[(size_t)(base + gr) * NCOL + n0 + c0]) = v;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (wmma) { cute::TMEM::Allocator1Sm().free(tmem_base, 512);
              cute::TMEM::Allocator1Sm().release_allocation_lock(); }
  my_cluster_sync();
}

static CUtensorMap make_tmap(void* ptr, int64_t rows, int64_t cols) {
  CUtensorMap tm{};
  uint64_t gd[3] = {128, (uint64_t)rows, (uint64_t)(cols / 128)};
  uint64_t gs[2] = {(uint64_t)cols, 128};
  uint32_t bd[3] = {128, BM, 2};
  uint32_t es[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, gd, gs, bd, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "tensormap fail ", (int)r);
  return tm;
}

template <int MODE, int NK, int NTN, int NCOL, int BROWS, int SA, int SB, int NCONS, int CSZ, int ER, int VAR>
static void launch(int64_t A, int64_t B, int64_t Asc, int64_t Wsc, int64_t Out,
                   int64_t tile_e, int64_t tile_base, int64_t tile_j0, int64_t cnt,
                   int64_t row_w, int64_t ntiles, int MR, int MAXC, int NPROG, int NEXP) {
  constexpr int KDIM = NK * 128;
  auto tmapA = make_tmap((void*)A, MR, KDIM);
  auto tmapB = make_tmap((void*)B, (int64_t)NEXP * BROWS, KDIM);
  auto kern = moe_gemm_kernel<MODE, NK, NTN, NCOL, BROWS, SA, SB, NCONS, CSZ, ER, VAR>;
  constexpr int smem = (int)sizeof(SharedStorage<SA, SB, NK, ER>);
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NPROG, 1, 1);
  cfg.blockDim = dim3(64 + NCONS, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CSZ;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, kern, tmapA, tmapB, (const float*)Asc, (const float*)Wsc,
      (__nv_bfloat16*)Out, (const int*)tile_e, (const int*)tile_base,
      (const int*)tile_j0, (const int*)cnt, (const float*)row_w,
      (const int*)ntiles, MR, MAXC, NPROG);
}

#define G1CASE(SA, SB, NC, CS, ER) if (CFG == ((SA) * 4096 + (SB) * 256 + ((NC) / 256) * 16 + (CS))) { \
  if (VAR==2) launch<0,56,32,4096,4096,SA,SB,NC,CS,ER,2>(Ap,W13,Asp,W13s,G1,tile_e,tile_base,tile_j0,cnt,none,ntiles,MR,0,NPROG,NEXP); \
  else if (VAR==3) launch<0,56,32,4096,4096,SA,SB,NC,CS,ER,3>(Ap,W13,Asp,W13s,G1,tile_e,tile_base,tile_j0,cnt,none,ntiles,MR,0,NPROG,NEXP); \
  else if (VAR==4) launch<0,56,32,4096,4096,SA,SB,NC,CS,ER,4>(Ap,W13,Asp,W13s,G1,tile_e,tile_base,tile_j0,cnt,none,ntiles,MR,0,NPROG,NEXP); \
  else launch<0,56,32,4096,4096,SA,SB,NC,CS,ER,0>(Ap,W13,Asp,W13s,G1,tile_e,tile_base,tile_j0,cnt,none,ntiles,MR,0,NPROG,NEXP); return; }
#define G2CASE(SA, SB, NC, CS, ER) if (CFG == ((SA) * 4096 + (SB) * 256 + ((NC) / 256) * 16 + (CS))) { \
  if (VAR==2) launch<1,16,56,7168,7168,SA,SB,NC,CS,ER,2>(Cq,W2,Cs,W2s,P,tile_e,tile_base,tile_j0,cnt,row_w,ntiles,MR,MAXC,NPROG,NEXP); \
  else if (VAR==3) launch<1,16,56,7168,7168,SA,SB,NC,CS,ER,3>(Cq,W2,Cs,W2s,P,tile_e,tile_base,tile_j0,cnt,row_w,ntiles,MR,MAXC,NPROG,NEXP); \
  else if (VAR==4) launch<1,16,56,7168,7168,SA,SB,NC,CS,ER,4>(Cq,W2,Cs,W2s,P,tile_e,tile_base,tile_j0,cnt,row_w,ntiles,MR,MAXC,NPROG,NEXP); \
  else launch<1,16,56,7168,7168,SA,SB,NC,CS,ER,0>(Cq,W2,Cs,W2s,P,tile_e,tile_base,tile_j0,cnt,row_w,ntiles,MR,MAXC,NPROG,NEXP); return; }

void gemm1(int64_t Ap, int64_t W13, int64_t Asp, int64_t W13s, int64_t G1,
           int64_t tile_e, int64_t tile_base, int64_t tile_j0, int64_t cnt,
           int64_t ntiles, int64_t MR, int64_t NPROG, int64_t CFG, int64_t VAR) {
  const int64_t none = 0;
  const int NEXP = 32;
  G1CASE(3,3,512,2,64) G1CASE(3,3,512,1,64) G1CASE(3,3,256,2,64)
  TORCH_CHECK(false, "bad cfg ", CFG);
}
void gemm2(int64_t Cq, int64_t W2, int64_t Cs, int64_t W2s, int64_t P,
           int64_t tile_e, int64_t tile_base, int64_t tile_j0, int64_t cnt,
           int64_t row_w, int64_t ntiles, int64_t MR, int64_t MAXC, int64_t NPROG,
           int64_t CFG, int64_t VAR) {
  const int NEXP = 32;
  G2CASE(3,3,256,2,64) G2CASE(3,3,256,1,64) G2CASE(3,3,512,2,64)
  TORCH_CHECK(false, "bad cfg ", CFG);
}


template <class K, class... Args>
static void launch_pdl(K kern, dim3 grid, dim3 blk, size_t smem, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = blk; cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = a; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// ============================================================ front-end kernels
#define FULLM 0xffffffffu
DEVI float wmax(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULLM, v, o));
  return v;
}
DEVI int wmin_i(int v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v = min(v, __shfl_xor_sync(FULLM, v, o));
  return v;
}

__global__ __launch_bounds__(128, 1)
void routing_kernel(const float* __restrict__ logits, const __nv_bfloat16* __restrict__ bias,
                    int* __restrict__ cnt, int* __restrict__ slot_tok, float* __restrict__ slot_w,
                    int* __restrict__ tok_e, int* __restrict__ tok_j, int* __restrict__ tok_n,
                    int* __restrict__ done, int* __restrict__ padoffs, int* __restrict__ tile_e,
                    int* __restrict__ tile_base, int* __restrict__ tile_j0, int* __restrict__ ntiles,
                    int LOFF, float RSF, int T, int MAXC) {
  __shared__ int is_last;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int l = threadIdx.x & 31;
  if (warp >= T) return;
  const float NEG = -3.0e38f;

  float sg[8], swb[8];
  const float* lp = logits + (size_t)warp * 256;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float x = lp[i * 32 + l];
    float sv = 1.0f / (1.0f + __expf(-x));
    sg[i] = sv;
    swb[i] = sv + __bfloat162float(bias[i * 32 + l]);
  }
  float gsc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float m1 = wmax(swb[i]);
    unsigned bm = __ballot_sync(FULLM, swb[i] == m1);
    int first = __ffs(bm) - 1;
    float m2 = wmax(l == first ? NEG : swb[i]);
    gsc[i] = m1 + m2;
  }
  float cur[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int r = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      if (gsc[j] > gsc[i] || (gsc[j] == gsc[i] && j < i)) ++r;
    cur[i] = (r < 4) ? swb[i] : NEG;
  }
  bool sel[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) sel[i] = false;
  float sumw = 0.f;
  #pragma unroll
  for (int rd = 0; rd < 8; ++rd) {
    float best = NEG; int bi = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) if (cur[i] > best) { best = cur[i]; bi = i; }
    float gm = wmax(best);
    int myexp = (best == gm) ? (bi * 32 + l) : (1 << 30);
    int win = wmin_i(myexp);
    int wi = win >> 5, wl = win & 31;
    float sv = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) if (i == wi) sv = sg[i];
    sumw += __shfl_sync(FULLM, sv, wl);
    #pragma unroll
    for (int i = 0; i < 8; ++i) if (i == wi && l == wl) { cur[i] = NEG; sel[i] = true; }
  }
  const float inv = RSF / (sumw + 1e-20f);
  int nbefore = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int ge = i * 32 + l;
    const int e = ge - LOFF;
    const bool mine = sel[i] && e >= 0 && e < 32;
    unsigned b = __ballot_sync(FULLM, mine);
    int pos = nbefore + __popc(b & ((1u << l) - 1u));
    if (mine) {
      int j = atomicAdd(cnt + e, 1);
      slot_tok[(size_t)e * MAXC + j] = warp;
      slot_w[(size_t)e * MAXC + j] = sg[i] * inv;
      tok_e[warp * 8 + pos] = e;
      tok_j[warp * 8 + pos] = j;
    }
    nbefore += __popc(b);
  }
  if (l == 0) tok_n[warp] = nbefore;
  __syncthreads();
  if (threadIdx.x == 0) is_last = (atomicAdd(done, 1) == gridDim.x - 1) ? 1 : 0;
  __syncthreads();
  if (!is_last) return;
  if (threadIdx.x == 0) *done = 0;
  if (threadIdx.x >= 32) return;
  const int ll = threadIdx.x;
  const int cc = cnt[ll];
  const int nt = (cc + 128 - 1) / 128;
  int ps = cc, ts = nt;
  #pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int a = __shfl_up_sync(FULLM, ps, o), b = __shfl_up_sync(FULLM, ts, o);
    if (ll >= o) { ps += a; ts += b; }
  }
  const int poff = ps - cc;
  const int toff = ts - nt;
  padoffs[ll] = poff;
  for (int kq = 0; kq < nt; ++kq) {
    tile_e[toff + kq] = ll;
    tile_base[toff + kq] = poff + kq * 128;
    tile_j0[toff + kq] = kq * 128;
  }
  if (ll == 31) *ntiles = ts;
}

__global__ __launch_bounds__(32, 1)
void meta_kernel(const int* __restrict__ cnt, int* __restrict__ padoffs, int* __restrict__ tile_e,
                 int* __restrict__ tile_base, int* __restrict__ tile_j0, int* __restrict__ ntiles,
                 int BMc) {
  const int l = threadIdx.x;
  const int c = cnt[l];
  const int nt = (c + BMc - 1) / BMc;
  int ps = nt * BMc, ts = nt;
  #pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int a = __shfl_up_sync(FULLM, ps, o), b = __shfl_up_sync(FULLM, ts, o);
    if (l >= o) { ps += a; ts += b; }
  }
  const int poff = ps - nt * BMc;
  const int toff = ts - nt;
  padoffs[l] = poff;
  for (int k = 0; k < nt; ++k) {
    tile_e[toff + k] = l;
    tile_base[toff + k] = poff + k * BMc;
    tile_j0[toff + k] = k * BMc;
  }
  if (l == 31) *ntiles = ts;
}

__global__ __launch_bounds__(256, 1)
void permute_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As,
                    uint8_t* __restrict__ Ap, float* __restrict__ Asp,
                    const int* __restrict__ cnt, const int* __restrict__ slot_tok,
                    const int* __restrict__ tile_e, const int* __restrict__ tile_base,
                    const int* __restrict__ tile_j0, const int* __restrict__ ntiles_p,
                    int T, int MR, int MAXC, int HDIM) {
  cudaGridDependencySynchronize();
  const int tq = blockIdx.x >> 2;
  if (tq >= *ntiles_p) return;
  const int rsub = (blockIdx.x & 3) * 32;
  const int e = tile_e[tq], base = tile_base[tq], j0 = tile_j0[tq], c = cnt[e];
  if (j0 + rsub >= c) return;
  const int hb = blockIdx.y;              // 14 blocks of 512 bytes
  const int tid = threadIdx.x;
  const int r0 = rsub + (tid >> 5), c0 = (tid & 31) * 16;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int rr = r0 + i * 8;
    if (j0 + rr < c) {
      const int tok = slot_tok[(size_t)e * MAXC + j0 + rr];
      const uint4 v = *reinterpret_cast<const uint4*>(A + (size_t)tok * HDIM + hb * 512 + c0);
      *reinterpret_cast<uint4*>(Ap + (size_t)(base + rr) * HDIM + hb * 512 + c0) = v;
    }
  }
  if (tid < 32 && j0 + rsub + tid < c) {
    const int tok = slot_tok[(size_t)e * MAXC + j0 + rsub + tid];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int kb = hb * 4 + q;
      Asp[(size_t)kb * MR + base + rsub + tid] = As[(size_t)kb * T + tok];
    }
  }
}

void prep(int64_t logits, int64_t bias, int64_t hs, int64_t hss,
          int64_t Ap, int64_t Asp, int64_t cnt, int64_t slot_tok, int64_t slot_w,
          int64_t tok_e, int64_t tok_j, int64_t tok_n, int64_t padoffs,
          int64_t tile_e, int64_t tile_base, int64_t tile_j0, int64_t ntiles,
          int64_t T, int64_t MR, int64_t MAXC, int64_t MAXTILES,
          int64_t LOFF, double RSF) {
  auto st = at::cuda::getCurrentCUDAStream();
  int nb = (int)((T + 3) / 4);
  launch_pdl(routing_kernel, dim3(nb,1,1), dim3(128,1,1), 0, (const float*)logits, (const __nv_bfloat16*)bias,
      (int*)cnt, (int*)slot_tok, (float*)slot_w, (int*)tok_e, (int*)tok_j, (int*)tok_n,
      (int*)ntiles + 1, (int*)padoffs, (int*)tile_e, (int*)tile_base, (int*)tile_j0,
      (int*)ntiles, (int)LOFF, (float)RSF, (int)T, (int)MAXC);
  dim3 g((unsigned)MAXTILES * 4, 14, 1);
  launch_pdl(permute_kernel, g, dim3(256,1,1), 0, (const uint8_t*)hs, (const float*)hss,
      (uint8_t*)Ap, (float*)Asp, (const int*)cnt, (const int*)slot_tok,
      (const int*)tile_e, (const int*)tile_base, (const int*)tile_j0,
      (const int*)ntiles, (int)T, (int)MR, (int)MAXC, 7168);
}


__global__ __launch_bounds__(128, 1)
void swiglu_kernel(const __nv_bfloat16* __restrict__ G1, uint8_t* __restrict__ Cq,
                   float* __restrict__ Cs, const int* __restrict__ cnt,
                   const int* __restrict__ tile_e, const int* __restrict__ tile_base,
                   const int* __restrict__ tile_j0, const int* __restrict__ ntiles_p, int MR) {
  cudaGridDependencySynchronize();
  const int tq = blockIdx.x >> 2;
  if (tq >= *ntiles_p) return;
  const int rsub = (blockIdx.x & 3) * 32;
  const int jb = blockIdx.y;
  const int e = tile_e[tq], base = tile_base[tq], j0 = tile_j0[tq], c = cnt[e];
  if (j0 + rsub >= c) return;
  const int row = rsub + (threadIdx.x >> 2);
  const int sub = (threadIdx.x & 3) * 32;
  const bool ok = (j0 + row) < c;
  const int rw = ok ? row : 0;
  const size_t go = (size_t)(base + rw) * 4096 + jb * 128 + sub;
  float v[32];
  float am = 0.f;
  #pragma unroll
  for (int i = 0; i < 32; i += 8) {
    float4 a = *reinterpret_cast<const float4*>(G1 + go + i);
    float4 b = *reinterpret_cast<const float4*>(G1 + go + 2048 + i);
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      float2 f1 = __bfloat1622float2(pa[t]);
      float2 f2 = __bfloat1622float2(pb[t]);
      float c0v = (f2.x / (1.0f + __expf(-f2.x))) * f1.x;
      float c1v = (f2.y / (1.0f + __expf(-f2.y))) * f1.y;
      v[i + 2 * t] = c0v;
      v[i + 2 * t + 1] = c1v;
      am = fmaxf(am, fmaxf(fabsf(c0v), fabsf(c1v)));
    }
  }
  am = fmaxf(am, __shfl_xor_sync(FULLM, am, 1));
  am = fmaxf(am, __shfl_xor_sync(FULLM, am, 2));
  const float sc = fmaxf(am, 1e-30f) * (1.0f / 448.0f);
  const float inv = 1.0f / sc;
  uint16_t q[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float2 t2 = make_float2(v[2 * i] * inv, v[2 * i + 1] * inv);
    __nv_fp8x2_e4m3 f2(t2);
    q[i] = *reinterpret_cast<uint16_t*>(&f2);
  }
  if (ok) {
    *reinterpret_cast<uint4*>(Cq + (size_t)(base + row) * 2048 + jb * 128 + sub) =
        *reinterpret_cast<const uint4*>(q);
    *reinterpret_cast<uint4*>(Cq + (size_t)(base + row) * 2048 + jb * 128 + sub + 16) =
        *reinterpret_cast<const uint4*>(q + 8);
    if ((threadIdx.x & 3) == 0) Cs[(size_t)jb * MR + base + row] = sc;
  }
}

__global__ __launch_bounds__(256, 1)
void combine_kernel(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ Out,
                    const int* __restrict__ tok_e, const int* __restrict__ tok_j,
                    const int* __restrict__ tok_n, const int* __restrict__ padoffs,
                    int* __restrict__ cnt, int T, int BNC) {
  cudaGridDependencySynchronize();
  __shared__ int rows_s[8];
  const int t = blockIdx.x;
  const int c0 = blockIdx.y * BNC + (threadIdx.x << 3);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 32) cnt[threadIdx.x] = 0;
  const int n = tok_n[t];
  if (threadIdx.x < 8) {
    int r = 0;
    if (threadIdx.x < n) r = padoffs[tok_e[t * 8 + threadIdx.x]] + tok_j[t * 8 + threadIdx.x];
    rows_s[threadIdx.x] = r;
  }
  __syncthreads();
  float acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int i = 0; i < n; ++i) {
    float4 d = *reinterpret_cast<const float4*>(P + (size_t)rows_s[i] * 7168 + c0);
    const __nv_bfloat16* pd = reinterpret_cast<const __nv_bfloat16*>(&d);
    #pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(pd[k]);
  }
  __nv_bfloat16 o[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
  *reinterpret_cast<float4*>(Out + (size_t)t * 7168 + c0) = *reinterpret_cast<const float4*>(o);
}

void post(int64_t G1, int64_t Cq, int64_t Cs, int64_t P, int64_t Out, int64_t cnt,
          int64_t tok_e, int64_t tok_j, int64_t tok_n, int64_t padoffs, int64_t tile_e,
          int64_t tile_base, int64_t tile_j0, int64_t ntiles, int64_t T, int64_t MR,
          int64_t MAXTILES, int64_t stage) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (stage == 0) {
    dim3 g((unsigned)MAXTILES * 4, 16, 1);
    launch_pdl(swiglu_kernel, g, dim3(128,1,1), 0, (const __nv_bfloat16*)G1, (uint8_t*)Cq, (float*)Cs,
        (const int*)cnt, (const int*)tile_e, (const int*)tile_base, (const int*)tile_j0,
        (const int*)ntiles, (int)MR);
  } else {
    const int BNC = 1024;
    dim3 g((unsigned)T, 7168 / BNC, 1);
    launch_pdl(combine_kernel, g, dim3(BNC/8,1,1), 0, (const __nv_bfloat16*)P, (__nv_bfloat16*)Out,
        (const int*)tok_e, (const int*)tok_j, (const int*)tok_n, (const int*)padoffs,
        (int*)cnt, (int)T, BNC);
  }
}

static void set_l2_window(void* ptr, size_t bytes) {
  static size_t maxw = 0;
  if (maxw == 0) {
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, 0);
    maxw = pr.accessPolicyMaxWindowSize;
    size_t cap = pr.persistingL2CacheMaxSize;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, cap);
  }
  cudaStreamAttrValue a{};
  a.accessPolicyWindow.base_ptr = ptr;
  a.accessPolicyWindow.num_bytes = bytes < maxw ? bytes : maxw;
  a.accessPolicyWindow.hitRatio = 1.0f;
  a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(at::cuda::getCurrentCUDAStream(), cudaStreamAttributeAccessPolicyWindow, &a);
}

void moe_all(int64_t logits, int64_t bias, int64_t hs, int64_t hss,
             int64_t W13, int64_t W13s, int64_t W2, int64_t W2s, int64_t Out,
             int64_t Ap, int64_t Asp, int64_t cnt, int64_t slot_tok, int64_t slot_w,
             int64_t tok_e, int64_t tok_j, int64_t tok_n, int64_t padoffs,
             int64_t tile_e, int64_t tile_base, int64_t tile_j0, int64_t ntiles,
             int64_t G1, int64_t Cq, int64_t Cs, int64_t Pp,
             int64_t T, int64_t MR, int64_t MAXC, int64_t MAXTILES,
             int64_t LOFF, double RSF, int64_t NPROG, int64_t CFG1, int64_t CFG2, int64_t L2W) {
  prep(logits, bias, hs, hss, Ap, Asp, cnt, slot_tok, slot_w, tok_e, tok_j, tok_n,
       padoffs, tile_e, tile_base, tile_j0, ntiles, T, MR, MAXC, MAXTILES, LOFF, RSF);
  if (L2W) set_l2_window((void*)Ap, (size_t)MR * 7168);
  gemm1(Ap, W13, Asp, W13s, G1, tile_e, tile_base, tile_j0, cnt, ntiles, MR, NPROG, CFG1, 0);
  post(G1, Cq, Cs, Pp, 0, cnt, tok_e, tok_j, tok_n, padoffs, tile_e, tile_base, tile_j0,
       ntiles, T, MR, MAXTILES, 0);
  if (L2W) set_l2_window((void*)Cq, (size_t)MR * 2048);
  gemm2(Cq, W2, Cs, W2s, Pp, tile_e, tile_base, tile_j0, cnt, slot_w, ntiles, MR, MAXC,
        NPROG, CFG2, 0);
  post(G1, Cq, Cs, Pp, Out, cnt, tok_e, tok_j, tok_n, padoffs, tile_e, tile_base, tile_j0,
       ntiles, T, MR, MAXTILES, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_all", &moe_all);
  m.def("prep", &prep);
  m.def("post", &post);
  m.def("gemm1", &gemm1);
  m.def("gemm2", &gemm2);
}
