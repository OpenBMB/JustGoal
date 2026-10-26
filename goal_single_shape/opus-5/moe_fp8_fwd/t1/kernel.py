import os, ctypes
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUBIN = os.path.join(_HERE, "moe.cubin")

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


class _Engine:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = _Engine()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        torch.empty(1, device="cuda")
        cu = ctypes.CDLL("libcuda.so.1")
        self.cu = cu
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        cu.cuLaunchCooperativeKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

        with open(_CUBIN, "rb") as f:
            data = f.read()
        self._data = ctypes.create_string_buffer(data, len(data))
        mod = ctypes.c_void_p()
        self._chk(cu.cuModuleLoadData(ctypes.byref(mod), self._data))
        self.mod = mod
        fn = ctypes.c_void_p()
        self._chk(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_fused"))
        self.fn = fn
        self.nt = 512
        self.smem = 16384
        self._chk(cu.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))
        nb = ctypes.c_int()
        self._chk(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(nb), fn, self.nt, ctypes.c_size_t(self.smem)))
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        per_sm = max(1, nb.value)
        self.nctas = per_sm * sms
        self.gbuf = torch.empty(8 * 4096, dtype=torch.float32, device="cuda")
        self.obuf = torch.empty(7168, dtype=torch.float32, device="cuda")
        self.ctr = torch.zeros(320, dtype=torch.int32, device="cuda")
        self._cnctas = ctypes.c_int(self.nctas)

    def _chk(self, r):
        if r != 0:
            raise RuntimeError("cuda driver error %d" % r)

    def launch(self, args, stream):
        n = len(args)
        arr = (ctypes.c_void_p * n)()
        keep = []
        for i, a in enumerate(args):
            keep.append(a)
            arr[i] = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
        self._chk(self.cu.cuLaunchCooperativeKernel(
            self.fn, self.nctas, 1, 1, self.nt, 1, 1, self.smem,
            ctypes.c_void_p(stream), arr))


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    e = _Engine.get()
    out = torch.empty((1, 7168), dtype=torch.bfloat16, device=hidden_states.device)
    eoff = int(local_expert_offset.item() if torch.is_tensor(local_expert_offset) else local_expert_offset)
    rsf = float(routed_scaling_factor.item() if torch.is_tensor(routed_scaling_factor) else routed_scaling_factor)
    args = [
        _p(routing_logits), _p(routing_bias), _p(hidden_states), _p(hidden_states_scale),
        _p(gemm1_weights), _p(gemm1_weights_scale), _p(gemm2_weights), _p(gemm2_weights_scale),
        ctypes.c_int(eoff), ctypes.c_float(rsf),
        _p(e.gbuf), _p(e.obuf), _p(out), _p(e.ctr), e._cnctas,
    ]
    e.launch(args, torch.cuda.current_stream().cuda_stream)
    return out
