import ctypes, torch, math, sys, os
from common import *
from timing import bench
lib = ctypes.CDLL(os.path.abspath("libmla_final_c2.so"))
lib.mla_prefill_launch.restype = ctypes.c_int
lib.mla_prefill_launch.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

def mla_prefill(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, nclusters=0, dbg=None, dbg_qt=0):
    total_q = q_nope.shape[0]; B = qo_indptr.shape[0] - 1
    out = torch.empty(total_q, H, D_CKV, dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty(total_q, H, dtype=torch.float32, device=q_nope.device)
    err = lib.mla_prefill_launch(q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(), kpe_cache.data_ptr(),
                                 qo_indptr.data_ptr(), kv_indptr.data_ptr(), kv_indices.data_ptr(), out.data_ptr(), lse.data_ptr(),
                                 float(sm_scale), B, total_q, ckv_cache.shape[0], nclusters, torch.cuda.current_stream().cuda_stream, dbg.data_ptr() if dbg is not None else None, dbg_qt)
    assert err == 0, f"launch error {err}"
    return out, lse

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "small":
        # small synthetic case: user-given lens
        qlens = [int(x) for x in sys.argv[2].split(",")]; kvlens = [int(x) for x in sys.argv[3].split(",")]
        import common; common.Q_LENS[:] = qlens; common.KV_LENS[:] = kvlens
        g = torch.Generator(device="cuda"); g.manual_seed(1)
        tq = sum(qlens); npg = max(sum(kvlens) * 2, 1024)
        inp = dict(q_nope=torch.randn(tq, H, D_CKV, device="cuda", generator=g).to(torch.bfloat16),
                   q_pe=torch.randn(tq, H, D_KPE, device="cuda", generator=g).to(torch.bfloat16),
                   ckv_cache=torch.randn(npg, 1, D_CKV, device="cuda", generator=g).to(torch.bfloat16),
                   kpe_cache=torch.randn(npg, 1, D_KPE, device="cuda", generator=g).to(torch.bfloat16),
                   qo_indptr=torch.tensor([0]+list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device="cuda"),
                   kv_indptr=torch.tensor([0]+list(torch.tensor(kvlens).cumsum(0)), dtype=torch.int32, device="cuda"),
                   kv_indices=torch.randperm(npg, device="cuda", generator=g)[:sum(kvlens)].to(torch.int32).contiguous(),
                   sm_scale=SM_SCALE)
        from ref_fp32 import ref_fp32
        out_ref, lse_ref = ref_fp32(**inp)
        print("launching", flush=True); out, lse = mla_prefill(**inp); torch.cuda.synchronize(); print("kernel done", flush=True)
        compare("mine-vs-fp32", out, lse, out_ref, lse_ref)
        d = (out.float() - out_ref.float()).abs()
        print("err by hd block (64):", [f"{v:.3f}" for v in d.amax(dim=(0,1)).view(8,64).amax(dim=1).tolist()])
        print("err by position:", [f"{v:.2f}" for v in d.amax(dim=(1,2)).tolist()[:32]])
        print("err by head (pos0):", [f"{v:.2f}" for v in d[0].amax(dim=1).tolist()])
        # compare against partial-only hypotheses: out vs ref computed with only first/second kv half? skip
        print("ref sample", out_ref[0,0,:4].tolist(), "mine", out[0,0,:4].tolist(), "ratio", (out[0,0,:8].float()/out_ref[0,0,:8].float()).tolist())
        sys.exit(0)
    inp = make_inputs(0)
    ref = torch.load("ref_fp32_seed0.pt"); out_ref, lse_ref = ref["out"].cuda(), ref["lse"].cuda()
    out, lse = mla_prefill(**inp); torch.cuda.synchronize()
    compare("mine-vs-fp32", out, lse, out_ref, lse_ref)
    if mode == "bench":
        ev, kern, top = bench(lambda: mla_prefill(**inp))
        print(f"MINE: event {ev*1000:.1f} us/iter, kernel-sum {kern*1000:.1f} us/iter (all kernels incl. gather prologue), {top}")
