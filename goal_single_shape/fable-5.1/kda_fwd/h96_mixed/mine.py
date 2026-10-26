import os, sys, time, torch, numpy as np
from torch.utils.cpp_extension import load
from common import make_case

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.environ.get("KDA_SRC", "kda_fwd.cu")
name = "kda_fwd_" + os.path.splitext(os.path.basename(src))[0]
ext = load(name=name, sources=[os.path.join(HERE, src)],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--ptxas-options=-v", "-lineinfo"],
           extra_cflags=["-O3"], verbose=("-v" in sys.argv), build_directory=os.path.join(HERE, "build_" + name) if os.path.isdir(os.path.join(HERE, "build_" + name)) or not os.makedirs(os.path.join(HERE, "build_" + name), exist_ok=True) else None)


def seq_order_for(cu):
    lens = (cu[1:] - cu[:-1]).tolist()
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    return torch.tensor(order, dtype=torch.int32, device=cu.device)


def run_mine(case, state, out=None):
    if out is None:
        out = torch.empty_like(case["q"])
    order = seq_order_for(case["cu_seqlens"])
    ext.kda_fwd(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"], case["dt_bias"],
                state, case["cu_seqlens"], order, out, case["scale"], case["lower_bound"])
    return out


if __name__ == "__main__":
    case = make_case()
    print("smem bytes", ext.smem_bytes())
    state = case["initial_state"].clone()
    out = run_mine(case, state)
    torch.cuda.synchronize()
    torch.save({"out": out.cpu(), "final_state": state.cpu()}, "mine.pt")
    print("saved mine.pt; nan?", torch.isnan(out.float()).any().item(), torch.isnan(state.float()).any().item())
    if "--time" in sys.argv:
        from flashinfer.testing import bench_gpu_time
        rot = 64
        pool = case["initial_state"].unsqueeze(0).expand(rot, *case["initial_state"].shape).clone()
        outb = torch.empty_like(case["q"])
        cur = [0]
        def run():
            i = cur[0] % rot; cur[0] += 1
            run_mine(case, pool[i], outb)
        ms = bench_gpu_time(run, enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False, dry_run_iters=10, repeat_iters=50)
        ms = [float(x) for x in ms]
        print(f"MINE cold-L2: median {np.median(ms):.4f} ms  min {min(ms):.4f}  mean {np.mean(ms):.4f}  n={len(ms)}")
