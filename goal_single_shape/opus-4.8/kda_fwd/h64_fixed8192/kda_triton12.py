"""v12: v9 but pass AN is column-split over D (grid (G,H,D//BM)) for higher occupancy.
Mp[:,block] and Ex[:,block] updates are independent per column-block."""
import torch, math
import triton, triton.language as tl
from kda_triton8 import _pass1, _passB, _passC

@triton.jit
def _passAN2(uv_ptr,uk_ptr,khsc_ptr,atot_ptr, mgrp_ptr,ngrp_ptr,
             T,H, C:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,L:tl.constexpr,IP:tl.constexpr):
    g=tl.program_id(0); h=tl.program_id(1); pb=tl.program_id(2)
    rows=tl.arange(0,C); dd=tl.arange(0,D); cb=pb*BM+tl.arange(0,BM)
    st=H*D
    Mp=(dd[:,None]==cb[None,:]).to(tl.float32)   # [D,BM] identity columns
    Ex=tl.zeros((D,BM),dtype=tl.float32)         # [D,BM]
    for l in range(L):
        c=g*L+l
        t=c*C+rows
        off=t[:,None]*st + h*D + dd[None,:]
        voff=t[:,None]*st + h*D + cb[None,:]
        Uk=tl.load(uk_ptr+off); khsc=tl.load(khsc_ptr+off)
        Uv=tl.load(uv_ptr+voff).to(tl.float32)
        atot=tl.load(atot_ptr+(c*H+h)*D+dd).to(tl.float32)
        khscT=tl.trans(khsc)
        W=Uv-tl.dot(Uk,Ex.to(tl.bfloat16))                 # [C,BM]
        Ex=atot[:,None]*Ex+tl.dot(khscT,W.to(tl.bfloat16))
        t1=tl.dot(Uk,Mp.to(tl.bfloat16))                   # [C,BM]
        Mp=atot[:,None]*Mp-tl.dot(khscT,t1.to(tl.bfloat16))
    moff=(g*H+h)*D*D + dd[:,None]*D + cb[None,:]
    tl.store(mgrp_ptr+moff, Mp.to(mgrp_ptr.dtype.element_ty))
    tl.store(ngrp_ptr+moff, Ex.to(ngrp_ptr.dtype.element_ty))

def kda_forward12(q,k,v,g_log,beta,scale,C=32,G=4,BV=128,BM=64,ns=1,ip="tf32",
                  nw1=4,nwAN=4,nwB=4,nwC=4,st1=1,stAN=3,stC=1,store_dt=torch.bfloat16):
    # Tuned B200 best (2026-09-18): 1280us device time, err=1.46e-3<tol, 2.72x the 470us ref.
    _,T,H,D=q.shape
    NC=T//C; assert NC%G==0; L=NC//G
    uv=torch.empty_like(v); uk=torch.empty_like(k); o=torch.empty_like(v)
    qbar=torch.empty(1,T,H,D,device=q.device,dtype=store_dt)
    khsc=torch.empty(1,T,H,D,device=q.device,dtype=store_dt)
    atot=torch.empty(NC,H,D,device=q.device,dtype=torch.float32)
    rtril=torch.empty(NC,H,C,C,device=q.device,dtype=store_dt)
    mgrp=torch.empty(G,H,D,D,device=q.device,dtype=store_dt)
    ngrp=torch.empty(G,H,D,D,device=q.device,dtype=torch.float32)
    eg=torch.empty(G,H,D,D,device=q.device,dtype=torch.float32)
    final_state=torch.empty((1,H,D,D),device=q.device,dtype=torch.float32)
    _pass1[(NC,H)](q,k,v,g_log,beta,uv,uk,qbar,khsc,atot,rtril,T,H,scale,
                   C=C,D=D,NS=ns,IP=ip,num_warps=nw1,num_stages=st1)
    _passAN2[(G,H,D//BM)](uv,uk,khsc,atot,mgrp,ngrp,T,H,C=C,D=D,BM=BM,L=L,IP=ip,num_warps=nwAN,num_stages=stAN)
    _passB[(H,)](mgrp,ngrp,eg,H,D=D,G=G,IP=ip,num_warps=nwB,num_stages=1)
    _passC[(G,H,D//BV)](qbar,khsc,atot,uv,uk,rtril,eg,o,T,H,C=C,D=D,BV=BV,L=L,G=G,IP=ip,
                        final_state_ptr=final_state,num_warps=nwC,num_stages=stC)
    return o, final_state

if __name__=="__main__":
    from kda_ref import build_inputs
    import kda_triton9 as K9
    dev="cuda"
    g=torch.load("/root/kda_fwd/golden.pt"); of=g['out_fp32'].to(dev); tol=g['tol']
    d=build_inputs(N=1,L=8192,seed=12345)
    q,k,v,gl,beta,scale=d['q'],d['k'],d['v'],d['g_log'],d['beta'],d['scale']
    _,T,H,D=q.shape
    o,_=kda_forward12(q,k,v,gl,beta,scale)
    err=(o.float()-of).abs().max().item()
    print(f"correctness: err={err:.3e} tol={tol:.3e} PASS={err<tol}")
    def t(fn,it=200):
        for _ in range(15): fn()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it
    # isolate pAN2 vs pAN
    C,G=32,8; NC=T//C; L=NC//G
    uv=torch.empty_like(v); uk=torch.empty_like(k)
    khsc=torch.empty(1,T,H,D,device='cuda',dtype=torch.bfloat16)
    atot=torch.empty(NC,H,D,device='cuda',dtype=torch.float32)
    mgrp=torch.empty(G,H,D,D,device='cuda',dtype=torch.bfloat16)
    ngrp=torch.empty(G,H,D,D,device='cuda',dtype=torch.float32)
    pAN0=lambda:K9._passAN[(G,H)](uv,uk,khsc,atot,mgrp,ngrp,T,H,C=C,D=D,L=L,IP="tf32",num_warps=4,num_stages=1)
    print(f"pAN(v9, fused): {t(pAN0):.4f}")
    for BM in [32,64]:
        for nw in [2,4]:
            f=lambda BM=BM,nw=nw:_passAN2[(G,H,D//BM)](uv,uk,khsc,atot,mgrp,ngrp,T,H,C=C,D=D,BM=BM,L=L,IP="tf32",num_warps=nw,num_stages=1)
            try: print(f"pAN2 BM={BM} nw={nw}: {t(f):.4f}")
            except Exception as ex: print(f"pAN2 BM={BM} nw={nw}: FAIL {str(ex)[:40]}")
    def bench(**kw):
        for _ in range(10): kda_forward12(q,k,v,gl,beta,scale,**kw)
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200): kda_forward12(q,k,v,gl,beta,scale,**kw)
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/200
    print("=== full v12 ; target<=0.392 ref=0.470 ===")
    for G in [8,16]:
        for BM in [32,64]:
            ms=bench(C=32,G=G,BV=128,BM=BM); oo,_=kda_forward12(q,k,v,gl,beta,scale,C=32,G=G,BV=128,BM=BM)
            er=(oo.float()-of).abs().max().item()
            print(f"C=32 G={G} BV=128 BM={BM}: {ms:.4f} PASS={er<tol}")
