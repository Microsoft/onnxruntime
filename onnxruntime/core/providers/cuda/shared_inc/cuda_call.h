// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_pch.h"
namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW, typename SUCCTYPE = ERRTYPE>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg,
    const char* file, const int line);

#define CUDA_CALL(expr) (::onnxruntime::CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUBLAS_CALL(expr) (::onnxruntime::CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CURAND_CALL(expr) (::onnxruntime::CudaCall<curandStatus_t, false>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CUDNN_CALL(expr) (::onnxruntime::CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CUDNN_CALL2(expr, m) (::onnxruntime::CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m, __FILE__, __LINE__))

#define CUFFT_CALL(expr) (::onnxruntime::CudaCall<cufftResult, false>((expr), #expr, "CUFFT", CUFFT_SUCCESS, "", __FILE__, __LINE__))

#define CUDA_CALL_THROW(expr) (::onnxruntime::CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUBLAS_CALL_THROW(expr) (::onnxruntime::CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CURAND_CALL_THROW(expr) (::onnxruntime::CudaCall<curandStatus_t, true>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS, "", __FILE__, __LINE__))

// the cudnn configuration call that doesn't need set stream
#define CUDNN_CALL_THROW(expr) (::onnxruntime::CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CUFFT_CALL_THROW(expr) (::onnxruntime::CudaCall<cufftResult, true>((expr), #expr, "CUFFT", CUFFT_SUCCESS, "", __FILE__, __LINE__))

#ifdef ORT_USE_NCCL
#define NCCL_CALL(expr) (::onnxruntime::CudaCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess, "", __FILE__, __LINE__))
#define NCCL_CALL_THROW(expr) (::onnxruntime::CudaCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess, "", __FILE__, __LINE__))
#endif

}  // namespace onnxruntime
