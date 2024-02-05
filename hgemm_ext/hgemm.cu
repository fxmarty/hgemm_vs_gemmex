#include "hgemm.cuh"

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void cublas_hgemm
(
    cublasHandle_t cublas_handle,
    const half* inp,
    const half* w,
    half* out,
    int size_m,
    int size_n,
    int size_k
) {
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    cublasStatus_t return_status;

    // PyTorch is row-major while cublas works with column major.
    // Hence Y = A * W in row-major becomes Y^T = W^T * A^T in column major.
    // Hence, m=size_n, n=size_m, k=size_k, and the ordering of `w` and `inp`.
    return_status = cublasHgemm(cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha,
        w, size_n,
        inp, size_k,
        &beta,
        out, size_n
    );

    if (return_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(_cudaGetErrorEnum(return_status));
    }
}
 
void cublas_gemmex
(
    cublasHandle_t cublas_handle,
    const half* inp,
    const half* w,
    half* out,
    int size_m,
    int size_n,
    int size_k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t return_status;

    // PyTorch is row-major while cublas works with column major.
    // Hence Y = A * W in row-major becomes Y^T = W^T * A^T in column major.
    // Hence, m=size_n, n=size_m, k=size_k, and the ordering of `w` and `inp`.
    return_status = cublasGemmEx(cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha,
        w, CUDA_R_16F, size_n,
        inp, CUDA_R_16F, size_k,
        &beta,
        out, CUDA_R_16F, size_n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    if (return_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(_cudaGetErrorEnum(return_status));
    }
}