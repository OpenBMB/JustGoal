import torch
from torch.utils.cpp_extension import load_inline
_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float* o){
    int4 v = *reinterpret_cast<const int4*>(p);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for(int i=0;i<8;i++) o[i]=__bfloat162float(h[i]);
}
template<int H,int D,int DP,int TOPK,int NS>
__global__ void dsa_kernel(
    const __nv_bfloat16* __restrict__ q_nope,const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv,const __nv_bfloat16* __restrict__ kpe,
    const int* __restrict__ sidx,__nv_bfloat16* __restrict__ out,float* __restrict__ lse,
    float sm_scale,int T)
{
    const int t=blockIdx.x,head=blockIdx.y;
    const int w=threadIdx.x>>5,lane=threadIdx.x&31;
    const int DPL=D/32,DPPL=DP/32;
    __shared__ float sAcc[NS][16][32]; __shared__ float sL[NS];
    const int* myidx=sidx+t*TOPK;
    float qn[16],qp[2];
    ld8(q_nope+(size_t)(t*H+head)*D+lane*DPL, qn);
    ld8(q_nope+(size_t)(t*H+head)*D+lane*DPL+8, qn+8);
    const __nv_bfloat16* qp_ptr=q_pe+(size_t)(t*H+head)*DP+lane*DPPL;
    qp[0]=__bfloat162float(qp_ptr[0]); qp[1]=__bfloat162float(qp_ptr[1]);
    float acc[16];
    #pragma unroll
    for(int i=0;i<DPL;i++) acc[i]=0.f;
    float l=0.f;
    const int C=4;
    for(int base=w; base<TOPK; base+=NS*C){
        int ids[C]; float kcb[C][16]; float kpb[C][2];
        #pragma unroll
        for(int j=0;j<C;j++){
            int kk=base+j*NS; int id=(kk<TOPK)?myidx[kk]:-1; ids[j]=id;
            if(id>=0){
                const __nv_bfloat16* kc=ckv+(size_t)id*D+lane*DPL;
                ld8(kc,kcb[j]); ld8(kc+8,kcb[j]+8);
                const __nv_bfloat16* kp=kpe+(size_t)id*DP+lane*DPPL;
                kpb[j][0]=__bfloat162float(kp[0]); kpb[j][1]=__bfloat162float(kp[1]);
            } else {
                #pragma unroll
                for(int i=0;i<DPL;i++) kcb[j][i]=0.f;
                kpb[j][0]=0.f; kpb[j][1]=0.f;
            }
        }
        float partv[C];
        #pragma unroll
        for(int j=0;j<C;j++){
            float p=0.0f;
            #pragma unroll
            for(int i=0;i<DPL;i++) p+=qn[i]*kcb[j][i];
            #pragma unroll
            for(int i=0;i<DPPL;i++) p+=qp[i]*kpb[j][i];
            partv[j]=p;
        }
        #pragma unroll
        for(int off=16;off>0;off>>=1){
            #pragma unroll
            for(int j=0;j<C;j++) partv[j]+=__shfl_xor_sync(0xffffffff,partv[j],off);
        }
        // no-max softmax: pp = exp(logit*sm_scale)
        float pp[C]; int nvalidc=0;
        #pragma unroll
        for(int j=0;j<C;j++){ pp[j]=(ids[j]>=0)?__expf(partv[j]*sm_scale):0.0f; l+=pp[j]; if(ids[j]>=0)nvalidc++; }
        #pragma unroll
        for(int i=0;i<DPL;i++){ float a=acc[i];
            #pragma unroll
            for(int j=0;j<C;j++) a+=pp[j]*kcb[j][i];
            acc[i]=a; }
        if(nvalidc<C) break;
    }
    #pragma unroll
    for(int i=0;i<DPL;i++) sAcc[w][i][lane]=acc[i];
    if(lane==0) sL[w]=l;
    __syncthreads();
    if(w==0){
        float L=0.f;
        #pragma unroll
        for(int j=0;j<NS;j++) L+=sL[j];
        float o[16];
        #pragma unroll
        for(int i=0;i<DPL;i++){ float a=0.f;
            #pragma unroll
            for(int j=0;j<NS;j++) a+=sAcc[j][i][lane];
            o[i]=a; }
        __nv_bfloat16* o_ptr=out+(size_t)(t*H+head)*D+lane*DPL;
        if(L>0.f){ float linv=1.0f/L;
            #pragma unroll
            for(int i=0;i<DPL;i++) o_ptr[i]=__float2bfloat16(o[i]*linv);
            if(lane==0) lse[t*H+head]=__log2f(L);
        } else {
            #pragma unroll
            for(int i=0;i<DPL;i++) o_ptr[i]=__float2bfloat16(0.f);
            if(lane==0) lse[t*H+head]=-INFINITY;
        }
    }
}
void launch(torch::Tensor q_nope,torch::Tensor q_pe,torch::Tensor ckv,torch::Tensor kpe,
    torch::Tensor sidx,torch::Tensor out,torch::Tensor lse,double sm_scale){
    int T=q_nope.size(0); const int NS=8;
    dim3 grid(T,16); dim3 block(32*NS);
    dsa_kernel<16,512,64,2048,NS><<<grid,block>>>(
        (const __nv_bfloat16*)q_nope.data_ptr(),(const __nv_bfloat16*)q_pe.data_ptr(),
        (const __nv_bfloat16*)ckv.data_ptr(),(const __nv_bfloat16*)kpe.data_ptr(),
        sidx.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),lse.data_ptr<float>(),(float)sm_scale,T);
}
'''
_CPP="void launch(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double);"
_mod=load_inline(name="dsa_nomax",cpp_sources=_CPP,cuda_sources=_CUDA,functions=["launch"],verbose=False,extra_cuda_cflags=["-O3","--use_fast_math"])
def run(q_nope,q_pe,ckv_cache,kpe_cache,sparse_indices,sm_scale):
    T,H,D=q_nope.shape; DP=q_pe.shape[-1]
    ckv=ckv_cache.view(-1,D); kpe=kpe_cache.view(-1,DP)
    out=torch.empty((T,H,D),dtype=torch.bfloat16,device=q_nope.device)
    lse=torch.empty((T,H),dtype=torch.float32,device=q_nope.device)
    _mod.launch(q_nope,q_pe,ckv,kpe,sparse_indices.to(torch.int32),out,lse,float(sm_scale))
    return out,lse
