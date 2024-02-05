#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>

static const char *_cudaGetErrorEnum(cublasStatus_t error);

void cublas_hgemm
(
    cublasHandle_t cublas_handle,
    const half* inp,
    const half* w,
    half* out,
    int size_m,
    int size_n,
    int size_k
);

void cublas_gemmex
(
    cublasHandle_t cublas_handle,
    const half* inp,
    const half* w,
    half* out,
    int size_m,
    int size_n,
    int size_k
);