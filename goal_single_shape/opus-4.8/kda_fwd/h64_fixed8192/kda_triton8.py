"""KDA forward, 2-level parallel scan with bf16 MMA (fp32 accumulate) for throughput.
 Structure identical to v7; matmul operands are bf16 (2x tensor-core throughput vs tf32).
 Newton inverse keeps tf32 (precision-sensitive). Everything else bf16."""
import torch, math
import triton, triton.language as tl

@triton.jit
def _pass1(q_ptr,k_ptr,v_ptr,g_ptr,b_ptr,
           uv_ptr,uk_ptr,qbar_ptr,khsc_ptr,atot_ptr,rtril_ptr,
           T,H,scale, C:tl.constexpr,D:tl.constexpr,NS:tl.constexpr,IP:tl.constexpr):
    c=tl.program_id(0); h=tl.program_id(1)
    rows=tl.arange(0,C); dd=tl.arange(0,D)
    ri=rows[:,None]; ci=rows[None,:]
    strict_lower=ri>ci; tril_incl=ri>=ci
    eye=(ri==ci).to(tl.float32)
    st=H*D
    t=c*C+rows
    off=t[:,None]*st + h*D + dd[None,:]
    q=tl.load(q_ptr+off).to(tl.float32); k=tl.load(k_ptr+off).to(tl.float32)
    v=tl.load(v_ptr+off); g=tl.load(g_ptr+off).to(tl.float32)
    qn=q*tl.rsqrt(tl.sum(q*q,axis=1))[:,None]*scale
    kn=k*tl.rsqrt(tl.sum(k*k,axis=1))[:,None]
    b=tl.load(b_ptr+(t*H+h)).to(tl.float32)
    gc=tl.cumsum(g,axis=0); tot=tl.sum(g,axis=0)
    egc=tl.exp(gc)
    s=0.5*tot[None,:]
    kbar=kn*egc; qbar=qn*egc; khsc=kn*tl.exp(tot[None,:]-gc)
    kbar_s=(kn*tl.exp(gc-s)).to(tl.bfloat16); qbar_s=(qn*tl.exp(gc-s)).to(tl.bfloat16)
    khat_s=(kn*tl.exp(s-gc)).to(tl.bfloat16)
    # Newton (tf32) on Att = I + tril(beta*kbar@khat^T,-1)
    P=tl.dot(kbar_s,tl.trans(khat_s),input_precision="tf32")
    Att=eye+tl.where(strict_lower,P*b[:,None],0.0)
    X=eye
    for _ in range(NS):
        X=tl.dot(X,2.0*eye-tl.dot(Att,X,input_precision="tf32"),input_precision="tf32")
    Xb=X.to(tl.bfloat16)
    Uv=tl.dot(Xb,(b[:,None]*v.to(tl.float32)).to(tl.bfloat16))
    Uk=tl.dot(Xb,(b[:,None]*kbar).to(tl.bfloat16))
    R=tl.dot(qbar_s,tl.trans(khat_s),input_precision="tf32")
    Rt=tl.where(tril_incl,R,0.0)
    tl.store(uv_ptr+off,Uv.to(uv_ptr.dtype.element_ty))
    tl.store(uk_ptr+off,Uk.to(uk_ptr.dtype.element_ty))
    tl.store(qbar_ptr+off,qbar.to(qbar_ptr.dtype.element_ty))
    tl.store(khsc_ptr+off,khsc.to(khsc_ptr.dtype.element_ty))
    tl.store(atot_ptr+(c*H+h)*D+dd, tl.exp(tot).to(atot_ptr.dtype.element_ty))
    tl.store(rtril_ptr+((c*H+h)*C+ri)*C+ci, Rt.to(rtril_ptr.dtype.element_ty))

@triton.jit
def _passA(uk_ptr,khsc_ptr,atot_ptr, mgrp_ptr,
           T,H, C:tl.constexpr,D:tl.constexpr,L:tl.constexpr,NC:tl.constexpr,IP:tl.constexpr):
    g=tl.program_id(0); h=tl.program_id(1)
    rows=tl.arange(0,C); dd=tl.arange(0,D)
    di=dd[:,None]; dj=dd[None,:]
    eye=(di==dj).to(tl.float32)
    st=H*D
    Mp=eye
    for l in range(L):
        c=g*L+l
        t=c*C+rows
        off=t[:,None]*st + h*D + dd[None,:]
        Uk=tl.load(uk_ptr+off); khsc=tl.load(khsc_ptr+off)
        atot=tl.load(atot_ptr+(c*H+h)*D+dd).to(tl.float32)
        Mcorr=tl.dot(tl.trans(khsc),Uk)                      # bf16 MMA
        Mp=atot[:,None]*Mp - tl.dot(Mcorr.to(tl.bfloat16),Mp.to(tl.bfloat16))
    moff=(g*H+h)*D*D + di*D + dj
    tl.store(mgrp_ptr+moff, Mp.to(mgrp_ptr.dtype.element_ty))

@triton.jit
def _passN(uv_ptr,uk_ptr,khsc_ptr,atot_ptr, ngrp_ptr,
           T,H, C:tl.constexpr,D:tl.constexpr,BV:tl.constexpr,L:tl.constexpr,IP:tl.constexpr):
    g=tl.program_id(0); h=tl.program_id(1); pv=tl.program_id(2)
    rows=tl.arange(0,C); dd=tl.arange(0,D); vd=pv*BV+tl.arange(0,BV)
    st=H*D
    E=tl.zeros((D,BV),dtype=tl.float32)
    for l in range(L):
        c=g*L+l
        t=c*C+rows
        off=t[:,None]*st + h*D + dd[None,:]
        voff=t[:,None]*st + h*D + vd[None,:]
        Uv=tl.load(uv_ptr+voff).to(tl.float32); Uk=tl.load(uk_ptr+off)
        khsc=tl.load(khsc_ptr+off)
        atot=tl.load(atot_ptr+(c*H+h)*D+dd).to(tl.float32)
        W=Uv-tl.dot(Uk,E.to(tl.bfloat16))
        E=atot[:,None]*E+tl.dot(tl.trans(khsc),W.to(tl.bfloat16))
    eoff=(g*H+h)*D*D + dd[:,None]*D + vd[None,:]
    tl.store(ngrp_ptr+eoff, E.to(ngrp_ptr.dtype.element_ty))

@triton.jit
def _passB(mgrp_ptr,ngrp_ptr, eg_ptr,
           H, D:tl.constexpr,G:tl.constexpr,IP:tl.constexpr):
    h=tl.program_id(0)
    dd=tl.arange(0,D); di=dd[:,None]; dj=dd[None,:]
    E=tl.zeros((D,D),dtype=tl.float32)
    for g in range(G):
        moff=(g*H+h)*D*D + di*D + dj
        tl.store(eg_ptr+moff, E.to(eg_ptr.dtype.element_ty))
        Mg=tl.load(mgrp_ptr+moff); Ng=tl.load(ngrp_ptr+moff).to(tl.float32)
        E=tl.dot(Mg,E.to(tl.bfloat16))+Ng

@triton.jit
def _passC(qbar_ptr,khsc_ptr,atot_ptr,uv_ptr,uk_ptr,rtril_ptr,eg_ptr,o_ptr,
           T,H, C:tl.constexpr,D:tl.constexpr,BV:tl.constexpr,L:tl.constexpr,
           G:tl.constexpr,IP:tl.constexpr,final_state_ptr=None):
    g=tl.program_id(0); h=tl.program_id(1); pv=tl.program_id(2)
    rows=tl.arange(0,C); dd=tl.arange(0,D); vd=pv*BV+tl.arange(0,BV)
    ri=rows[:,None]; ci=rows[None,:]
    st=H*D
    eoff=(g*H+h)*D*D + dd[:,None]*D + vd[None,:]
    E=tl.load(eg_ptr+eoff).to(tl.float32)
    for l in range(L):
        c=g*L+l
        t=c*C+rows
        off=t[:,None]*st + h*D + dd[None,:]
        voff=t[:,None]*st + h*D + vd[None,:]
        qbar=tl.load(qbar_ptr+off); khsc=tl.load(khsc_ptr+off)
        Uv=tl.load(uv_ptr+voff).to(tl.float32); Uk=tl.load(uk_ptr+off)
        atot=tl.load(atot_ptr+(c*H+h)*D+dd).to(tl.float32)
        Rt=tl.load(rtril_ptr+((c*H+h)*C+ri)*C+ci)
        Eb=E.to(tl.bfloat16)
        W=Uv-tl.dot(Uk,Eb)
        o=tl.dot(qbar,Eb)+tl.dot(Rt,W.to(tl.bfloat16))
        tl.store(o_ptr+voff,o.to(o_ptr.dtype.element_ty))
        E=atot[:,None]*E+tl.dot(tl.trans(khsc),W.to(tl.bfloat16))
    if final_state_ptr is not None:
        if g == G - 1:
            # E is [K,V]; the output contract stores state as [N,H,V,K].
            final_off = h * D * D + vd[None, :] * D + dd[:, None]
            tl.store(final_state_ptr + final_off, E)


def kda_forward8(q,k,v,g_log,beta,scale,C=32,G=16,BV=64,ns=5,ip="tf32",
                 nw1=2,nwA=4,nwB=4,nwC=4,st1=1,stA=1,stN=1,stC=1,store_dt=torch.bfloat16):
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
    _pass1[(NC,H)](q,k,v,g_log,beta,uv,uk,qbar,khsc,atot,rtril,T,H,scale,
                   C=C,D=D,NS=ns,IP=ip,num_warps=nw1,num_stages=st1)
    _passA[(G,H)](uk,khsc,atot,mgrp,T,H,C=C,D=D,L=L,NC=NC,IP=ip,num_warps=nwA,num_stages=stA)
    _passN[(G,H,D//BV)](uv,uk,khsc,atot,ngrp,T,H,C=C,D=D,BV=BV,L=L,IP=ip,num_warps=nwC,num_stages=stN)
    _passB[(H,)](mgrp,ngrp,eg,H,D=D,G=G,IP=ip,num_warps=nwB,num_stages=1)
    _passC[(G,H,D//BV)](qbar,khsc,atot,uv,uk,rtril,eg,o,T,H,C=C,D=D,BV=BV,L=L,G=G,IP=ip,
                        num_warps=nwC,num_stages=stC)
    return o

if __name__=="__main__":
    from kda_ref import build_inputs
    dev="cuda"
    g=torch.load("/root/kda_fwd/golden.pt"); of=g['out_fp32'].to(dev); tol=g['tol']
    d=build_inputs(N=1,L=8192,seed=12345)
    q,k,v,gl,beta,scale=d['q'],d['k'],d['v'],d['g_log'],d['beta'],d['scale']
    for C,G,ns in [(32,16,5),(16,16,4),(32,8,5)]:
        try:
            o=kda_forward8(q,k,v,gl,beta,scale,C=C,G=G,BV=64,ns=ns)
            err=(o.float()-of).abs().max().item()
            print(f"correctness C={C} G={G}: err={err:.3e} tol={tol:.3e} PASS={err<tol}")
        except Exception as ex:
            print(f"correctness C={C} G={G}: FAIL {str(ex)[:90]}")
    def bench(**kw):
        for _ in range(10): kda_forward8(q,k,v,gl,beta,scale,**kw)
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200): kda_forward8(q,k,v,gl,beta,scale,**kw)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)/200
    print("target <=0.392ms  ref=0.470ms")
    for C,ns in [(32,5),(16,4)]:
        for G in [8,16,32]:
            for BV in [64,128]:
                try:
                    ms=bench(C=C,G=G,BV=BV,ns=ns)
                    oo=kda_forward8(q,k,v,gl,beta,scale,C=C,G=G,BV=BV,ns=ns)
                    er=(oo.float()-of).abs().max().item()
                    print(f"C={C} G={G} BV={BV}: {ms:.4f} ms PASS={er<tol}")
                except Exception as ex:
                    print(f"C={C} G={G} BV={BV}: FAIL {str(ex)[:45]}")
