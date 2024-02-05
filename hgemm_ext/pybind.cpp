#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

#include "hgemm.cuh"

#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)

void gemm_cublas_hgemm
(
    torch::Tensor a,
    torch::Tensor w,
    torch::Tensor c
)
{
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 1, w, 0, 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    cublas_hgemm
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        (const half*) w.data_ptr(),
        (half*) c.data_ptr(),
        c.size(0), // m
        c.size(1), // n
        a.size(1) // k
    );
}

void gemm_cublas_gemmex
(
    torch::Tensor a,
    torch::Tensor w,
    torch::Tensor c
)
{
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 1, w, 0, 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    cublas_gemmex
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        (const half*) w.data_ptr(),
        (half*) c.data_ptr(),
        c.size(0), // m
        c.size(1), // n
        a.size(1) // k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_cublas_hgemm", &gemm_cublas_hgemm, "gemm_cublas_hgemm");
    m.def("gemm_cublas_gemmex", &gemm_cublas_gemmex, "gemm_cublas_gemmex");
}
