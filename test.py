import torch
import cublas_hgemm
# import gemm_cublas_hgemm
# import gemm_cublas_gemmex
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

m = 4096
k = 2048
n = 8192

a = torch.rand(m, k, device="cuda", dtype=torch.float16)
w = torch.rand(k, n, device="cuda", dtype=torch.float16)

out = torch.empty((a.shape[0], n), dtype=torch.float16, device="cuda")

cublas_hgemm.gemm_cublas_hgemm(a, w, out)
res_pytorch = torch.matmul(a, w)

print(((res_pytorch - out).abs() / (res_pytorch.abs() + 1e-12)))

assert torch.allclose(out, res_pytorch, atol=1e-2, rtol=1e-2)

out = torch.empty((a.shape[0], n), dtype=torch.float16, device="cuda")

cublas_hgemm.gemm_cublas_gemmex(a, w, out)
res_pytorch = torch.matmul(a, w)

print(((res_pytorch - out).abs() / (res_pytorch.abs() + 1e-12)))

assert torch.allclose(out, res_pytorch, atol=1e-2, rtol=1e-2)

def run_hgemm(a, w, n):
    out = torch.empty((a.shape[0], n), dtype=torch.float16, device="cuda")
    cublas_hgemm.gemm_cublas_hgemm(a, w, out)
    return 0

def run_gemmex(a, w, n):
    out = torch.empty((a.shape[0], n), dtype=torch.float16, device="cuda")
    cublas_hgemm.gemm_cublas_gemmex(a, w, out)
    return 0

def run_torch(a, w):
    out = torch.empty((a.shape[0], n), dtype=torch.float16, device="cuda")
    torch.matmul(a, w, out=out)
    return 0

torch.cuda.cudart().cudaProfilerStart()

num_runs = 100
with torch.no_grad():
    # torch.cuda.nvtx.range_push(f"hgemm bench")
    # warmup
    run_hgemm(a, w, n)

    times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        run_hgemm(a, w, n)

        end_event.record()

        torch.cuda.synchronize()

        tps = (start_event.elapsed_time(end_event))  # in ms
        times.append(tps)
    
    # torch.cuda.nvtx.range_pop()

    print(f"hgemm took: {np.mean(times):.4f} ms")

    # torch.cuda.nvtx.range_push(f"torch bench")

    # warmup
    run_torch(a, w)

    times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        run_gemmex(a, w, n)

        end_event.record()

        torch.cuda.synchronize()

        tps = (start_event.elapsed_time(end_event))  # in ms
        times.append(tps)
    
    # torch.cuda.nvtx.range_pop()

    print(f"gemmex took: {np.mean(times):.4f} ms")

    # warmup
    run_torch(a, w)

    times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        run_torch(a, w)

        end_event.record()

        torch.cuda.synchronize()

        tps = (start_event.elapsed_time(end_event))  # in ms
        times.append(tps)
    
    # torch.cuda.nvtx.range_pop()

    print(f"torch took: {np.mean(times):.4f} ms")

torch.cuda.cudart().cudaProfilerStop()