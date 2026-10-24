// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/kernel_registration_validation.h"

#include "core/framework/allocator_utils.h"
#include "core/framework/memcpy.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/int4.h"
#include "core/mlas/inc/mlas.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#endif

#if defined(ENABLE_TRAINING_OPS)
#include "orttraining/training_ops/cpu/cpu_training_kernels.h"
#endif

#include "core/framework/compute_capability.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

#if defined(MLAS_HALF_GEMM_ACCELERATION_POSSIBLE)
#define ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
#endif

namespace onnxruntime {

// The MemcpyFromHost and MemcpyToHost kernels registered for the CPU EP are generic memcpy kernels.
// Other EPs may provide their own memcpy kernels.
// For a memcpy between host (CPU) and device of some other EP:
// - If the EP provides the corresponding memcpy kernel, it will be used.
// - Otherwise, one of these generic memcpy kernels will be used.

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()),
    Memcpy);

CPUExecutionProvider::CPUExecutionProvider(const CPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCpuExecutionProvider}, info_{info} {}

std::vector<AllocatorPtr> CPUExecutionProvider::CreatePreferredAllocators() {
  const bool create_arena = DoesCpuAllocatorSupportArenaUsage() ? info_.create_arena : false;
  AllocatorCreationInfo device_info_cpu{[](int) { return std::make_unique<CPUAllocator>(); },
                                        DEFAULT_CPU_ALLOCATOR_DEVICE_ID, create_arena};

  return std::vector<AllocatorPtr>{CreateAllocator(device_info_cpu)};
}

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 10, Clip);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, Elu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 15, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Relu);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, Selu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Sigmoid);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, Softplus);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, Softsign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Tanh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, PRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomNormal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomUniform);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomNormalLike);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomUniformLike);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Multinomial);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 11, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7, float, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7, double, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, float, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, double, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7, float, Min);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7, float, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, Max);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, Not);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, And);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Or);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Xor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7, float, Mean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, float, Mean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, float, Sin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, double, Sin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, float, Cos);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, double, Cos);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Tan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Asin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Acos);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Atan);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, double, Gemm);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Hardmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8, double, MatMul);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9, float, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9, double, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, double,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, InstanceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, float, LpNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, double, LpNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, LRN);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, MaxPool);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, MLFloat16, MaxPool);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, LpPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, GlobalLpPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, GlobalAveragePool);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, GlobalAveragePool);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, GlobalMaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MaxRoiPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, GRU);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, LSTM);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, RNN);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Gather);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 4, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, SpaceToDepth);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Tile);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, int8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, float, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, double, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, int8_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, int16_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, int32_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, int64_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, uint8_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, uint16_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, uint32_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, uint64_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, bool, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, MLFloat16, Expand);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12, string, Expand);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 8, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Loop);

// Opset 9
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 19, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, MeanVarianceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int16_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int16_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Less);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, EyeLike);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, double, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, IsNaN);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, Sign);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, Shrink);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                      int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, float_int64_t_int64_t,
                                                      OneHot);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int64_t_string_int64_t,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, float_string_int64_t,
                                                      OneHot);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, float_float_float,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int64_t_int32_t_float,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int64_t_float_int64_t,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int32_t_float_int32_t,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int32_t_float_float,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int64_t_float_float,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, int64_t_float_int32_t,
                                                      OneHot);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, MaxUnpool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Sinh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Cosh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Asinh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Acosh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Atanh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, Scatter);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, TfIdfVectorizer);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, string, Where);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, float, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, double, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, int8_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, uint8_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, uint32_t, Where);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, double, Gemm);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, double, MatMul);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int32_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, int64_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 13, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 13, double,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9, int8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9, uint8_t, Upsample);

// Opset 10
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, StringNormalizer);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, float, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, double, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, Mod);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, int8_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 21, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, uint8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, int8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, int32_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, uint8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, int8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 20, uint8_t,
                                                      QLinearMatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 20, int8_t,
                                                      QLinearMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, uint8_t, MatMulInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, ConvInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, uint8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 11, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 19, IsInf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 15, float, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 15, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 27, ReverseSequence);

// opset 11
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13, float, CumSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13, double, CumSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13, int32_t, CumSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13, int64_t, CumSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int16_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, float, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, double, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, uint8_t, DynamicQuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t,
                                                      ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t,
                                                      ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int64_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Hardmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, Softmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 15, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Gather);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, Det);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, ScatterElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, NonMaxSuppression);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 18, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MaxUnpool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 17, LpPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, Conv);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 18, MLFloat16,
                                                      AveragePool);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceLength);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceAt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceEmpty);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceInsert);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceErase);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceConstruct);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, ConcatFromSequence);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, SplitToSequence);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, ScatterND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, double, Gemm);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, GatherElements);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, uint8_t, BitShift);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, uint16_t, BitShift);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, uint32_t, BitShift);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, uint64_t, BitShift);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, GatherND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 26, Range);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, Unique);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, float, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, double, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, int64_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, int32_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, int8_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, int16_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23, uint8_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, float_int64_t_int64_t, OneHot);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_string_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, float_string_int64_t, OneHot);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, float_float_float, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_int32_t_float, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int32_t_float_int32_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int32_t_float_float, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_float_float, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, int64_t_float_int32_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, int8_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Resize);

// opset 12
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Clip);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Min);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 21, MaxPool);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 21, MLFloat16, MaxPool);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, uint8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, GatherND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 27, Einsum);

// REVIEW(codemzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
// However these types work on GPU implementation.
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, MLFloat16_MLFloat16, Dropout);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, MLFloat16_float, Dropout);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, MLFloat16_double, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, float_float, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, float_double,
                                                      Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, double_float,
                                                      Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, double_double,
                                                      Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 27, Celu);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, float,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, double,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int8_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int16_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int32_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int64_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint8_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint16_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint32_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint64_t,
                                                      GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, double, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int8_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int16_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint8_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint16_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15, uint64_t, LessOrEqual);

// opset 13
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Erf);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, Cast);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int32_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, bool, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Expand);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, string, Expand);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Gemm);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, MatMul);
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Min);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Mean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Sigmoid);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Sign);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, Size);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Sum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, LRN);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MeanVarianceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, float_float, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, float_double, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, double_float, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, double_double, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int16_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint16_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Neg);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27, Mod);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Mod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, double, Relu);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Log);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 14, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, DepthToSpace);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20, Transpose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Tile);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, GatherElements);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, ScatterND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, ScatterElements);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19, float, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19, double, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19, MLFloat16, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19, BFloat16, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, bool, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, GatherND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t,
                                                      ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t,
                                                      ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t,
                                                      ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, uint8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, uint8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, double,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int64_t,
                                                      ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, int8_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Hardmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Softmax);

// Opset 14
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Relu);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, Trilu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Add);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Add);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Sub);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Mul);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Div);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Div);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 18, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 15, Identity);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 14, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 14, double,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, GRU);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, LSTM);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, RNN);

// Opset 15
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 18, Shape);

#if !defined(DISABLE_OPTIONAL_TYPE)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 17, OptionalHasElement);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 17, OptionalGetElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, Optional);
#endif

// Opset 16
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, If);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 19, float, GridSample);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 17, ScatterElements);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 17, ScatterND);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, string, Where);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t, Where);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, LeakyRelu);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, PRelu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, Scan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int16_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint16_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int16_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint16_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint64_t, LessOrEqual);

// Opset 17
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, 19, DFT);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, BlackmanWindow);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, HammingWindow);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, HannWindow);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, MelWeightMatrix);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, STFT);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, float, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, double, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization);
// Opset 18
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, int8_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, uint8_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 21, LpPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, Col2Im);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, ScatterND);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, ScatterElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, Split);
#if !defined(DISABLE_OPTIONAL_TYPE)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, OptionalHasElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, OptionalGetElement);

#endif

// Opset 19
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, AveragePool);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, MLFloat16, AveragePool);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, int32_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, uint8_t,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, int8_t,
                                                      DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FNUZ,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2,
                                                      DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2FNUZ,
                                                      DequantizeLinear);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int16_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint16_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, float, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, double, Equal);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, string, Equal);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, uint8_t,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, int8_t,
                                                      QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FNUZ,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2,
                                                      QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2FNUZ,
                                                      QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int8_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Shape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, float, DeformConv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, double, DeformConv);

// Opset 20
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 20, ConstantOfShape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, bool, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, bool, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, DFT);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 21, float, GridSample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 21, double, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, AffineGrid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, AffineGrid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, MLFloat16, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, BFloat16, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, Gelu);
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, MLFloat16, Gelu);
#endif
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E4M3FN, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E4M3FNUZ, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E5M2, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E5M2FNUZ, IsNaN);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, IsInf);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, StringConcat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, RegexFullMatch);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, StringSplit);
#endif

// Opset 21
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Int4x2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FNUZ, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2FNUZ, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Int4x2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FNUZ, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2FNUZ, QuantizeLinear);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, uint8_t, QLinearMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, int8_t, QLinearMatMul);

// Opset 22
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Acos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Cos);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Tan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Asin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Atan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Sinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Cosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Asinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Acosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Atanh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, DeformConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, DeformConv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Det);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float_float, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float_double, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double_float, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double_double, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, GridSample);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Elu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, EyeLike);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GlobalAveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GlobalMaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GRU);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, LSTM);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, RNN);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, HardSigmoid);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, InstanceNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, LpPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MaxUnpool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16, Round);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Sin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Softsign);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, ThresholdedRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, LpNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, LpNormalization);

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16, AveragePool);
#endif

// Opset 23
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, float, Attention);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, MLFloat16, Attention);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Int4x2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, UInt4x2, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FN, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FNUZ, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2FNUZ, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Int4x2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, UInt4x2, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FN, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FNUZ, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2FNUZ, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, float, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, double, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, MLFloat16, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, float, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, MLFloat16, RotaryEmbedding);

// Opset 24
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, float, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, MLFloat16, Attention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, TensorScatter);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Int4x2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, UInt4x2, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FN, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FNUZ, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2FNUZ, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Int4x2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, UInt4x2, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FN, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FNUZ, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2FNUZ, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Size);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, SplitToSequence);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, float, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, double, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int64_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int32_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int8_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int16_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, uint8_t, TopK);

// Opset 25
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Cast);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int16_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int4x2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt4x2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int2x4, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt2x4, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FNUZ, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2FNUZ, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int16_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int4x2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt4x2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int2x4, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt2x4, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FNUZ, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2FNUZ, QuantizeLinear);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Loop);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Reshape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Transpose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Unsqueeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Scan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Size);

// Opset 26
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, BitCast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, float, CumProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, double, CumProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, int32_t, CumProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, int64_t, CumProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, uint32_t, CumProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, uint64_t, CumProd);

// Opset 27
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 27, Range);

// Opset 28
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Cast);
#define DECLARE_QDQ_28(T, Op) class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, T, Op);
DECLARE_QDQ_28(int8_t, DequantizeLinear)
DECLARE_QDQ_28(uint8_t, DequantizeLinear)
DECLARE_QDQ_28(int16_t, DequantizeLinear)
DECLARE_QDQ_28(uint16_t, DequantizeLinear)
DECLARE_QDQ_28(int32_t, DequantizeLinear)
DECLARE_QDQ_28(Int4x2, DequantizeLinear)
DECLARE_QDQ_28(UInt4x2, DequantizeLinear)
DECLARE_QDQ_28(Int2x4, DequantizeLinear)
DECLARE_QDQ_28(UInt2x4, DequantizeLinear)
DECLARE_QDQ_28(Float6E2M3, DequantizeLinear)
DECLARE_QDQ_28(Float6E3M2, DequantizeLinear)
DECLARE_QDQ_28(int8_t, QuantizeLinear)
DECLARE_QDQ_28(uint8_t, QuantizeLinear)
DECLARE_QDQ_28(int16_t, QuantizeLinear)
DECLARE_QDQ_28(uint16_t, QuantizeLinear)
DECLARE_QDQ_28(Int4x2, QuantizeLinear)
DECLARE_QDQ_28(UInt4x2, QuantizeLinear)
DECLARE_QDQ_28(Int2x4, QuantizeLinear)
DECLARE_QDQ_28(UInt2x4, QuantizeLinear)
DECLARE_QDQ_28(Float6E2M3, QuantizeLinear)
DECLARE_QDQ_28(Float6E3M2, QuantizeLinear)
#if !defined(DISABLE_FLOAT8_TYPES)
DECLARE_QDQ_28(Float8E4M3FN, DequantizeLinear)
DECLARE_QDQ_28(Float8E4M3FNUZ, DequantizeLinear)
DECLARE_QDQ_28(Float8E5M2, DequantizeLinear)
DECLARE_QDQ_28(Float8E5M2FNUZ, DequantizeLinear)
DECLARE_QDQ_28(Float8E4M3FN, QuantizeLinear)
DECLARE_QDQ_28(Float8E4M3FNUZ, QuantizeLinear)
DECLARE_QDQ_28(Float8E5M2, QuantizeLinear)
DECLARE_QDQ_28(Float8E5M2FNUZ, QuantizeLinear)
#endif
#undef DECLARE_QDQ_28
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int8_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int16_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint8_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint16_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint32_t, BitShift);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint64_t, BitShift);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Compress);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, float_int64_t_int64_t, OneHot);
#if !defined(DISABLE_STRING_TYPE)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_string_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, float_string_int64_t, OneHot);
#endif
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, float_float_float, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_int32_t_float, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t_float_int32_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t_float_float, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_float_float, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_float_int32_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t_BFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t_BFloat16_int32_t, OneHot);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, ReverseSequence);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Unique);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Einsum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, float, Celu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, double, Celu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, MLFloat16, Celu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, BFloat16, Celu);

// !!PLEASE READ BELOW!! Following that, add new entries above this comment

/*  *** IMPORTANT! ***
 NEVER update a versioned entry to change the start or end version. These MUST be treated as immutable.
   i.e. if the macro has 'VERSIONED' in it, do not modify that entry

 When updating a declaration to add a new version of an operator there are 2 simple steps:

   1. There should be a non-versioned entry for that latest version. Update this to be versioned.
      Note that the end version is inclusive, so the end value will be one less than the operator's new opset version.
   2. Add a new non-versioned entry for the new opset.

 e.g. Say opset 13 is being added, and we need to update Add. The most recent change to Add was in opset 7 so it
      should have an un-versioned registration in the opset 7 section like this:

     class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, float, Add);

   Step 1 is to change that to add 'VERSIONED_' to the macro and add an end version of 12 as the new opset is 13:
     class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12, float, Add);

   Step 2 is to create a new un-versioned entry in the opset 13 sections:
     class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Add);

 The process is the same for TYPED and untyped kernels - just repeat for each type when updating the typed entries.

 The changes below in the registrations using BuildKernelCreateInfo are essentially the same. Update existing
 registration to use the VERSIONED_ macro, add end version, add new un-versioned entry in the section for the new
 opset.

 To double-check what versions an operator should have registrations for see
 https://github.com/onnx/onnx/blob/main/docs/Operators.md
*****/

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterOnnxOperatorKernels(KernelRegistry& kernel_registry) {
  static constexpr BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 10, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 15,
                                                                      LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomNormal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomUniform)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomNormalLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, RandomUniformLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Multinomial)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 12,
                                                                            uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 11, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7,
                                                                            float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7,
                                                                            double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7,
                                                                            float, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7,
                                                                            float, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, Not)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Xor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10,
                                                                            bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10,
                                                                            int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10,
                                                                            int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10,
                                                                            float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 10,
                                                                            double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 7,
                                                                            float, Mean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            float, Mean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, double, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, float, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, double, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Tan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Asin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Acos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 21, Atan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                      Hardmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8,
                                                                            float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8,
                                                                            double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9,
                                                                            float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9,
                                                                            double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                      ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 21,
                                                                      InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, float,
                                                                            LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, double,
                                                                            LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9,
                                                                      AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11,
                                                                      MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, LpPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, GlobalLpPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MaxRoiPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            uint8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            uint8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int32_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                            int64_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 13, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12,
                                                                      Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 4, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 5, 12,
                                                                      Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 9, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12,
                                                                      SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                      DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                      Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 12,
                                                                      Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
                                                                      Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            int8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            float, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            double, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            int8_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            int16_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            int32_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            int64_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            uint8_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            uint16_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            uint32_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            uint64_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            bool, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            MLFloat16, Expand)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 12,
                                                                            string, Expand)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 8, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Loop)>,

      // Opset 9
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                      Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 19,
                                                                      ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                      MeanVarianceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int16_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint16_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, EyeLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            double, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            MLFloat16, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            float_int64_t_int64_t, OneHot)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_string_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            float_string_int64_t, OneHot)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            float_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_int32_t_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int32_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            int64_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                      MaxUnpool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Sinh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Cosh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Asinh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Acosh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 21, Atanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                      Scatter)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, TfIdfVectorizer)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            uint8_t, NonZero)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            string, Where)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            double, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            int8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            uint8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15,
                                                                            uint32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                      Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int32_t, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            int64_t, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 13,
                                                                            float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 13,
                                                                            double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 15, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9,
                                                                            float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9,
                                                                            int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9,
                                                                            int8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 9,
                                                                            uint8_t, Upsample)>,

  // Opset 10
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, StringNormalizer)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                      AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            int8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                            uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 21, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12,
                                                                            uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12,
                                                                            int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12,
                                                                            int32_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12,
                                                                            uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 12,
                                                                            int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 20,
                                                                            uint8_t, QLinearMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 20,
                                                                            int8_t, QLinearMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, uint8_t,
                                                                  MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, int8_t,
                                                                  MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, ConvInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, uint8_t,
                                                                  QLinearConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, int8_t,
                                                                  QLinearConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 11,
                                                                      Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 10,
                                                                      NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 19, IsInf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 15,
                                                                            float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 15,
                                                                            double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 10, 27, ReverseSequence)>,
      // opset 11
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11, Clip)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13,
                                                                            float, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13,
                                                                            double, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13,
                                                                            int32_t, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 13,
                                                                            int64_t, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int16_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint16_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, float, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, double, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MLFloat16,
                                                                            Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, uint8_t,
                                                                  DynamicQuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Hardmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 15, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, Det)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 18,
                                                                      AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MaxUnpool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 17,
                                                                      LpPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceLength)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceAt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceEmpty)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceInsert)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceErase)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, SequenceConstruct)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, ConcatFromSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                      SplitToSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                      GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            uint8_t, BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            uint16_t, BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            uint32_t, BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            uint64_t, BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                      GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 26,
                                                                      Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            int64_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            int32_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            int8_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            int16_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 23,
                                                                            uint8_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            float_int64_t_int64_t, OneHot)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_string_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            float_string_int64_t, OneHot)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            float_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_int32_t_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int32_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 27,
                                                                            int64_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 11,
                                                                            int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int32_t, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            int64_t, ReduceSumSquare)>,

      // OpSet 12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Clip)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Min)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Max)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 21, MaxPool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            uint8_t, ReduceMax)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                      GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 27, Einsum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, bool, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16,
                                                                  Expand)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, string,
                                                                  Expand)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Erf)>,
      // REVIEW(codemzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
      // However these types work on GPU implementation.
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12,
      // MLFloat16_MLFloat16, Dropout)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12,
      // MLFloat16_float, Dropout)>, BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider,
      // kOnnxDomain, 12, MLFloat16_double, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            float_float, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            float_double, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            double_float, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 12,
                                                                            double_double, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 27, Celu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            float, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            double, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int16_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint16_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            float, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            double, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int16_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint16_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 15,
                                                                            uint64_t, LessOrEqual)>,

      // opset 13
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Mean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int32_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20,
                                                                      Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                            MeanVarianceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, float_float,
                                                                            Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, float_double,
                                                                            Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, double_float,
                                                                            Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 21, double_double,
                                                                            Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t,
                                                                  ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t,
                                                                  ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t,
                                                                  ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t,
                                                                  ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                      Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int16_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint16_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 18,
                                                                            double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t,
                                                                  Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  int8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  int16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  uint8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  uint16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13,
                                                                  uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 14, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20,
                                                                      Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20,
                                                                      Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 20,
                                                                      Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27,
                                                                      DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 27,
                                                                      SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15,
                                                                      ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15,
                                                                      ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                      Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19,
                                                                            float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19,
                                                                            double, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 19,
                                                                            MLFloat16, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, uint8_t,
                                                                  NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int64_t, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            int8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 17,
                                                                            uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 15, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, Hardmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double,
                                                                  Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float,
                                                                  Softmax)>,

      // OpSet 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double,
                                                                  CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t,
                                                                  CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t,
                                                                  CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, Trilu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int16_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint16_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 18,
                                                                      Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 15,
                                                                      Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 14,
                                                                            float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 14,
                                                                            double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, 21, RNN)>,

      // Opset 15
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, float,
                                                                  BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, double,
                                                                  BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 18, Shape)>,

#if !defined(DISABLE_OPTIONAL_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 17,
                                                                      OptionalHasElement)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, 17,
                                                                      OptionalGetElement)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 15, Optional)>,
#endif

      // Opset 16
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18,
                                                                      Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float,
                                                                  RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double,
                                                                  RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 19,
                                                                            float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 17,
                                                                      ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 17,
                                                                      ScatterND)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, string, Where)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t,
                                                                  Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t,
                                                                  Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t,
                                                                  Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t,
                                                                  Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t,
                                                                  Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, 18, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int16_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint16_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint64_t,
                                                                  GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, float,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, double,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int8_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int16_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int32_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, int64_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint8_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint16_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint32_t,
                                                                  LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, uint64_t,
                                                                  LessOrEqual)>,

      // Opset 17
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, BlackmanWindow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, 19, DFT)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, HammingWindow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, HannWindow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, MelWeightMatrix)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, STFT)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, float,
                                                                  LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, double,
                                                                  LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, MLFloat16,
                                                                  LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 17, BFloat16,
                                                                  LayerNormalization)>,

      // Opset 18
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18,
                                                                            float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18,
                                                                            int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18,
                                                                            int8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18,
                                                                            uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, float,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, double,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int32_t,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int64_t,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int8_t,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, uint8_t,
                                                                            ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, float,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, double,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int32_t,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int64_t,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, int8_t,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 19, uint8_t,
                                                                            ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, float,
                                                                  ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, double,
                                                                  ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 21, LpPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, Col2Im)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t,
                                                                  BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t,
                                                                  BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t,
                                                                  BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int8_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int16_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int32_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, int64_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint8_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint16_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint32_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, uint64_t,
                                                                  BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, 18, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, Split)>,
#if !defined(DISABLE_OPTIONAL_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, OptionalHasElement)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 18, OptionalGetElement)>,
#endif

      // Opset 19
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            int32_t, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E4M3FN, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E4M3FNUZ, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E5M2, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E5M2FNUZ, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int8_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int16_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int32_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int64_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint8_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint16_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint32_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint64_t,
                                                                  Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, double, Equal)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, string, Equal)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                      Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            int8_t, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E4M3FN, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E4M3FNUZ, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E5M2, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                            Float8E5M2FNUZ, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20,
                                                                      Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int32_t,
                                                                  Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, int8_t,
                                                                  Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, uint8_t,
                                                                  Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, float, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, double, DeformConv)>,

      // Opset 20
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 20,
                                                                      ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, bool, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, bool, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, DFT)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 21, float,
                                                                            GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, 21, double,
                                                                            GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float,
                                                                  AffineGrid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double,
                                                                  AffineGrid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, double, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, MLFloat16,
                                                                  IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, BFloat16,
                                                                  IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, float,
                                                                  Gelu)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E4M3FN,
                                                                  IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E4M3FNUZ,
                                                                  IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E5M2,
                                                                  IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, Float8E5M2FNUZ,
                                                                  IsNaN)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, IsInf)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, StringConcat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, RegexFullMatch)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, StringSplit)>,
#endif

      // Opset 21
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int32_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Int4x2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, UInt4x2,
                                                                            DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FNUZ,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2FNUZ,
                                                                            DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, uint16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, int16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Int4x2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, UInt4x2,
                                                                            QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FNUZ,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2FNUZ,
                                                                            QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, uint8_t,
                                                                  QLinearMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 21, int8_t,
                                                                  QLinearMatMul)>,

      // Opset 22
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Tan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Asin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Acos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Atan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Sinh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Cosh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Asinh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Acosh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Atanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Det)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, EyeLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float_float,
                                                                  Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float_double,
                                                                  Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double_float,
                                                                  Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double_double,
                                                                  Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float,
                                                                  LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double,
                                                                  LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, LpPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MaxUnpool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16,
                                                                  Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, double, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, AveragePool)>,

      // Opset 23
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23,
                                                                            float, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23,
                                                                            MLFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int32_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Int4x2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, UInt4x2,
                                                                            DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FN,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FNUZ,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2FNUZ,
                                                                            DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, uint16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, int16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Int4x2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, UInt4x2,
                                                                            QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FN,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E4M3FNUZ,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Float8E5M2FNUZ,
                                                                            QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, 23, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, float,
                                                                  RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, double,
                                                                  RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, MLFloat16,
                                                                  RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, float,
                                                                  RotaryEmbedding)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 23, MLFloat16,
                                                                  RotaryEmbedding)>,

      // opset 24
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, float, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, MLFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, TensorScatter)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int32_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Int4x2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, UInt4x2,
                                                                            DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FN,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FNUZ,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2FNUZ,
                                                                            DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, uint16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, int16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Int4x2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, UInt4x2,
                                                                            QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FN,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E4M3FNUZ,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Float8E5M2FNUZ,
                                                                            QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, 24, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int64_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int32_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int8_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, int16_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, uint8_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 24, SplitToSequence)>,

      // opset 25
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int8_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int16_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int32_t,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int4x2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt4x2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int2x4,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt2x4,
                                                                            DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FNUZ,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2,
                                                                            DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2FNUZ,
                                                                            DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int8_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, uint16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, int16_t,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int4x2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt4x2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Int2x4,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, UInt2x4,
                                                                            QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FNUZ,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2,
                                                                            QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2FNUZ,
                                                                            QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 25, Unsqueeze)>,

      // opset 26
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, BitCast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, float, CumProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, double, CumProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, int32_t, CumProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, int64_t, CumProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, uint32_t, CumProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 26, uint64_t, CumProd)>,

      // opset 27
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 27, Range)>,

      // opset 28
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int16_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint16_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Int4x2, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, UInt4x2, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Int2x4, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, UInt2x4, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float6E2M3, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float6E3M2, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int16_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint16_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Int4x2, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, UInt4x2, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Int2x4, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, UInt2x4, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float6E2M3, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float6E3M2, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E4M3FN, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E4M3FNUZ, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E5M2, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E5M2FNUZ, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E4M3FN, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E4M3FNUZ, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E5M2, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Float8E5M2FNUZ, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int8_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int16_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int32_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, int64_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint8_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint16_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint32_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, uint64_t,
                                                                  BitShift)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  float_int64_t_int64_t, OneHot)>,
#if !defined(DISABLE_STRING_TYPE)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_string_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  float_string_int64_t, OneHot)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  float_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_int32_t_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int32_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_float_float, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_BFloat16_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28,
                                                                  int32_t_BFloat16_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, ReverseSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, Einsum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, float, Celu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, double, Celu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, MLFloat16, Celu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 28, BFloat16, Celu)>,
  };
  static_assert(registration_internal::IsKernelRegistrationTableValid(
      function_table, BuildKernelCreateInfo<void>));

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
Status RegisterFp16GemmMatMulKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 8,
                                                                            MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 10,
                                                                            MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 12,
                                                                            MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 8,
                                                                            MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 9, 12,
                                                                            MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16,
                                                                  MatMul)>,
  };

  if constexpr (std::size(function_table) > 1) {
    // Kernel registration is process-wide and based on the default backend
    // selector config (nullptr => use_kleidiai=true). A session can still
    // disable KleidiAI via kOrtSessionOptionsMlasDisableKleidiAi, in which case
    // MatMul/Gemm may run their non-accelerated fp16 fallback for these kernels.
    if (MlasHalfGemmAccelerationSupported(nullptr)) {
      for (auto& function_table_entry : function_table) {
        KernelCreateInfo info = function_table_entry();
        if (info.kernel_def != nullptr) {
          ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
        }
      }
    }
  }

  return Status::OK();
}
#endif

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
Status RegisterFp16Kernels(KernelRegistry& kernel_registry) {
  static constexpr BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 21, MLFloat16,
                                                                            GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16,
                                                                  GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 21, MLFloat16,
                                                                            Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16,
                                                                  Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, 18,
                                                                            MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 19, 21, MLFloat16,
                                                                            AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22, MLFloat16,
                                                                  AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, 11,
                                                                            MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 12, 21, MLFloat16,
                                                                            MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 22,
                                                                  MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12,
                                                                            MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, 13,
                                                                            MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16,
                                                                  Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 15,
                                                                            MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 16, MLFloat16,
                                                                  LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 14, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 20, MLFloat16,
                                                                  Gelu)>};
  static_assert(registration_internal::IsKernelRegistrationTableValid(
      function_table, BuildKernelCreateInfo<void>));

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
#endif

// Forward declarations of ml op kernels
#ifndef DISABLE_ML_OPS
namespace ml {
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float, ArrayFeatureExtractor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double, ArrayFeatureExtractor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int32_t, ArrayFeatureExtractor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t, ArrayFeatureExtractor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string, ArrayFeatureExtractor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Binarizer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, CastMap);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, CategoryMapper);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_int64_t, DictVectorizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_float, DictVectorizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_double, DictVectorizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_string, DictVectorizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_float, DictVectorizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_double, DictVectorizer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, FeatureVectorizer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Imputer);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 1, LabelEncoder);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, LinearClassifier);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, LinearRegressor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Normalizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t, OneHotEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float, OneHotEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double, OneHotEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string, OneHotEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float, Scaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double, Scaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t, Scaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int32_t, Scaler);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, SVMClassifier);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, SVMRegressor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, float,
                                                      TreeEnsembleClassifier);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, double,
                                                      TreeEnsembleClassifier);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, int64_t,
                                                      TreeEnsembleClassifier);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, int32_t,
                                                      TreeEnsembleClassifier);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, float,
                                                      TreeEnsembleRegressor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2, double,
                                                      TreeEnsembleRegressor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, ZipMap);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, float_string,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, string_float,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, int64_float,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, float_int64,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, int64_string,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, string_int64,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, int64_int64,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, string_string,
                                                      LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3, float_float,
                                                      LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, float, TreeEnsembleClassifier);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, double, TreeEnsembleClassifier);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, int64_t, TreeEnsembleClassifier);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, int32_t, TreeEnsembleClassifier);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, float, TreeEnsembleRegressor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, double, TreeEnsembleRegressor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 5, float, TreeEnsemble);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 5, double, TreeEnsemble);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_string, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_float, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_float, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_string, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_string, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_float, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_int16, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_string, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_double, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_double, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_double, LabelEncoder);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterOnnxMLOperatorKernels(KernelRegistry& kernel_registry) {
  static constexpr BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float,
                                                                  ArrayFeatureExtractor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double,
                                                                  ArrayFeatureExtractor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int32_t,
                                                                  ArrayFeatureExtractor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t,
                                                                  ArrayFeatureExtractor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string,
                                                                  ArrayFeatureExtractor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Binarizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, CastMap)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, CategoryMapper)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_int64_t,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_float,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string_double,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_string,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_float,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t_double,
                                                                  DictVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, FeatureVectorizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Imputer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 1,
                                                                      LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, LinearClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, LinearRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, Normalizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t,
                                                                  OneHotEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float,
                                                                  OneHotEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double,
                                                                  OneHotEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, string,
                                                                  OneHotEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, float, Scaler)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, double, Scaler)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int64_t,
                                                                  Scaler)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, int32_t,
                                                                  Scaler)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, SVMClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, SVMRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            float, TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            double, TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            int64_t, TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            int32_t, TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            float, TreeEnsembleRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, 2,
                                                                            double, TreeEnsembleRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 1, ZipMap)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            float_string, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            string_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            int64_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            float_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            int64_string, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            string_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            int64_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            string_string, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 2, 3,
                                                                            float_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, float,
                                                                  TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, double,
                                                                  TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, int64_t,
                                                                  TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, int32_t,
                                                                  TreeEnsembleClassifier)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, float,
                                                                  TreeEnsembleRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 3, double,
                                                                  TreeEnsembleRegressor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 5, float,
                                                                  TreeEnsemble)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 5, double,
                                                                  TreeEnsemble)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_string,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_float,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_float,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_int64,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_string,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_int64,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_int64,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_string,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, float_float,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_int16,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_string,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, string_double,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, int64_double,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_int64,
                                                                  LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMLDomain, 4, double_double,
                                                                  LabelEncoder)>,
  };
  static_assert(registration_internal::IsKernelRegistrationTableValid(
      function_table, BuildKernelCreateInfo<void>));

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
}  // namespace ml
#endif

Status RegisterCPUKernels(KernelRegistry& kernel_registry) {
  ORT_RETURN_IF_ERROR(RegisterOnnxOperatorKernels(kernel_registry));
#ifdef ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED
  ORT_RETURN_IF_ERROR(RegisterFp16GemmMatMulKernels(kernel_registry));
#endif
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
  if (MlasFp16AccelerationSupported()) {
    ORT_RETURN_IF_ERROR(RegisterFp16Kernels(kernel_registry));
  }
#endif
#ifndef DISABLE_ML_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::ml::RegisterOnnxMLOperatorKernels(kernel_registry));
#endif
#ifndef DISABLE_CONTRIB_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::RegisterCpuContribKernels(kernel_registry));
#endif
#if defined(ENABLE_TRAINING_OPS)
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::RegisterCpuTrainingKernels(kernel_registry));
#endif
  return Status::OK();
}

KernelRegistryAndStatus GetCpuKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterCPUKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> CPUExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetCpuKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

std::unique_ptr<IDataTransfer> CPUExecutionProvider::GetDataTransfer() const {
  return std::make_unique<CPUDataTransfer>();
}

}  // namespace onnxruntime
