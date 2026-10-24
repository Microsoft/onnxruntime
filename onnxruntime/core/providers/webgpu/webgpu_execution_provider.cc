// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_execution_provider.h"
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
#include "core/providers/webgpu/d3d12_external_data_loader.h"
#endif

#include <mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <gsl/gsl>

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#endif

#include "allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/run_options.h"
#include "core/graph/function_utils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/common/parse_string.h"

#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/external_data_loader.h"
#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/tensor/cast.h"
#include "core/providers/webgpu/tensor/concat.h"
#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/tensor/gather.h"
#include "core/providers/webgpu/tensor/grid_sample.h"
#include "core/providers/webgpu/tensor/reshape.h"
#include "core/providers/webgpu/generator/range.h"
#include "core/providers/webgpu/tensor/tile.h"
#include "core/providers/webgpu/tensor/trilu.h"
#include "core/providers/webgpu/tensor/unsqueeze.h"
#include "core/providers/webgpu/math/binary_elementwise_ops.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/tensor/where.h"
#include "core/providers/webgpu/reduction/reduction_ops.h"

#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/runtime_compatibility.h"
#endif

namespace onnxruntime {

#if defined(ORT_USE_EP_API_ADAPTERS)
onnxruntime::ep::adapter::Logger& WebGpuExecutionProvider::GetEpLogger() const {
  return *ep_logger_;
}

void WebGpuExecutionProvider::SetEpLogger(const OrtLogger* logger) {
  ep_logger_ = std::make_unique<onnxruntime::ep::adapter::Logger>(logger);
}
#endif

namespace webgpu {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    const auto& ep = *static_cast<const WebGpuExecutionProvider*>(Info().GetExecutionProvider());
    DataTransfer transfer(ep.BufferManager(), ep.Recording(), ep.GetDeviceId());
    return transfer.CopyTensor(*X, *Y);
  }
};

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, type, Op)>

static const BuildKernelCreateInfoFn build_kernel_create_info_function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

    // element-wise operators
    // unary - math
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Abs),
    KERNEL_CREATE_INFO(13, Abs),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Neg),
    KERNEL_CREATE_INFO(13, Neg),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Floor),
    KERNEL_CREATE_INFO(13, Floor),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Ceil),
    KERNEL_CREATE_INFO(13, Ceil),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Reciprocal),
    KERNEL_CREATE_INFO(13, Reciprocal),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Sqrt),
    KERNEL_CREATE_INFO(13, Sqrt),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Exp),
    KERNEL_CREATE_INFO(13, Exp),
    KERNEL_CREATE_INFO_VERSIONED(9, 12, Erf),
    KERNEL_CREATE_INFO(13, Erf),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Sigmoid),
    KERNEL_CREATE_INFO(13, Sigmoid),
    KERNEL_CREATE_INFO(6, HardSigmoid),
    KERNEL_CREATE_INFO(14, HardSwish),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Log),
    KERNEL_CREATE_INFO(13, Log),

    KERNEL_CREATE_INFO(7, Sin),
    KERNEL_CREATE_INFO(7, Cos),
    KERNEL_CREATE_INFO(7, Tan),
    KERNEL_CREATE_INFO(7, Asin),
    KERNEL_CREATE_INFO(7, Acos),
    KERNEL_CREATE_INFO(7, Atan),
    KERNEL_CREATE_INFO(9, Sinh),
    KERNEL_CREATE_INFO(9, Cosh),
    KERNEL_CREATE_INFO(9, Asinh),
    KERNEL_CREATE_INFO(9, Acosh),
    KERNEL_CREATE_INFO(9, Atanh),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Tanh),
    KERNEL_CREATE_INFO(13, Tanh),
    KERNEL_CREATE_INFO(1, Not),

    // // activations
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, int32_t, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, int32_t, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, uint32_t, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, uint32_t, Clip)>,
    // int64 Clip is registered conditionally in RegisterKernels (gated on enable_int64).
    KERNEL_CREATE_INFO(6, Elu),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Relu),
    KERNEL_CREATE_INFO_VERSIONED(13, 13, Relu),
    KERNEL_CREATE_INFO(14, Relu),
    KERNEL_CREATE_INFO_VERSIONED(6, 15, LeakyRelu),
    KERNEL_CREATE_INFO(16, LeakyRelu),
    KERNEL_CREATE_INFO(10, ThresholdedRelu),
    KERNEL_CREATE_INFO(20, Gelu),
    KERNEL_CREATE_INFO_VERSIONED(1, 21, Softplus),
    KERNEL_CREATE_INFO(22, Softplus),

    // // binary - math
    // All binary elementwise ops (Add, Sub, Mul, Div, Max, Min, Equal, Greater, Less,
    // GreaterOrEqual, LessOrEqual, Pow, PRelu, And) are registered via RegisterKernels
    // through RegisterBinaryElementwiseKernels, with conditional int64 support.

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, 18, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, Shape)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 15, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 18, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, 24, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 25, Identity)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceMean)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMin)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceProd)>,

    // ReduceSum: registered via RegisterKernels with conditional int64 support

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL1)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL2)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSum)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceSumSquare)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSumExp)>,

    // Where: registered via RegisterKernels with conditional int64 support

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, Transpose)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain,
                                                                          13, 27, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 28, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider,
                                                                          kMSInternalNHWCDomain, 13, 27, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider,
                                                                kMSInternalNHWCDomain, 28, DepthToSpace)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 21, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 21, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 22, Conv)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, ConvTranspose)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 9, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 7, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 8, 9, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 9, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 18, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 21, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 9, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 18, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 19, 21, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 22, AveragePool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalAveragePool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalMaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalMaxPool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, MatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MatMul)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ArgMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ArgMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ArgMax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ArgMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ArgMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ArgMin)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Softmax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 1, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, Split)>,

    // Gather: registered via RegisterKernels with conditional int64 support

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, GatherElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherElements)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, GatherND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, GatherND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherND)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, 17, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 18, 18, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 19, Resize)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 9, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Slice)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Trilu)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 17, LayerNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, RMSNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, RotaryEmbedding)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 21, LpNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, LpNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 5, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 5, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 21, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 6, 21, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 22, InstanceNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 27, Einsum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 28, Einsum)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Pad)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, If)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 13, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 14, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 8, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 9, 13, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 14, 14, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 15, BatchNormalization)>,

    KERNEL_CREATE_INFO_VERSIONED(10, 12, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(13, 18, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(19, 20, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(21, 22, DequantizeLinear),
    KERNEL_CREATE_INFO(23, DequantizeLinear),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, CumSum)>,

    KERNEL_CREATE_INFO_VERSIONED(17, 19, DFT),
    KERNEL_CREATE_INFO(20, DFT),

    KERNEL_CREATE_INFO_VERSIONED(1, 9, TopK),
    KERNEL_CREATE_INFO_VERSIONED(10, 10, TopK),
    KERNEL_CREATE_INFO_VERSIONED(11, 23, TopK),
    KERNEL_CREATE_INFO(24, TopK),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 15, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 17, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ScatterND)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 15, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 17, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ScatterElements)>,

    KERNEL_CREATE_INFO_VERSIONED(7, 13, LSTM),
    KERNEL_CREATE_INFO(14, LSTM),

    KERNEL_CREATE_INFO_VERSIONED(7, 13, GRU),
    KERNEL_CREATE_INFO(14, GRU),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 19, GridSample)>,
};

std::unique_ptr<KernelRegistry> RegisterKernels(bool enable_graph_capture, bool enable_int64) {
  auto kernel_registry = std::make_unique<KernelRegistry>();

  for (auto& function_table_entry : build_kernel_create_info_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  // Register Cast kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(6, 8, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(9, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(13, 18, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(19, 20, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(21, 22, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(23, 23, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(24, 24, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastVersionedKernelInfo(25, 27, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo(28, enable_int64)));

  // Register int64 Clip kernels with conditional int64 support
  RegisterClipInt64Kernels(*kernel_registry, enable_int64);

  // Register Range kernels with conditional int64 support
  RegisterRangeKernels(*kernel_registry, enable_int64);

  // Register Unsqueeze kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(1, 10, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(11, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(13, 20, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(21, 22, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(23, 23, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo(24, 24, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeKernelInfo(25, enable_int64)));

  // Register Expand kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateExpandVersionedKernelInfo(8, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateExpandKernelInfo(13, enable_int64)));

  // Register all binary elementwise kernels (Add, Sub, Mul, Div, Max, Min, Equal, Greater,
  // Less, GreaterOrEqual, LessOrEqual, Pow, PRelu, And) with conditional int64 support.
  RegisterBinaryElementwiseKernels(*kernel_registry, enable_int64);

  // Register Reshape kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(5, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(13, 13, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(14, 18, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(19, 20, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(21, 22, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeVersionedKernelInfo(23, 24, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReshapeKernelInfo(25, enable_int64)));

  // Register Concat kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateConcatVersionedKernelInfo(1, 3, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateConcatVersionedKernelInfo(4, 10, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateConcatVersionedKernelInfo(11, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateConcatKernelInfo(13, enable_int64)));

  // Register Gather kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateGatherVersionedKernelInfo(1, 10, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateGatherVersionedKernelInfo(11, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateGatherKernelInfo(13, enable_int64)));

  // Register Where kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateWhereVersionedKernelInfo(9, 15, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateWhereKernelInfo(16, enable_int64)));

  // Register Tile kernels with conditional int64 support.
  // Tile is a pure data-movement op; int64 is safe because element values are never
  // interpreted or used in shader arithmetic.
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateTileVersionedKernelInfo(6, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateTileKernelInfo(13, enable_int64)));

  // Register ReduceSum kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReduceSumVersionedKernelInfo(1, 10, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReduceSumVersionedKernelInfo(11, 12, enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateReduceSumKernelInfo(13, enable_int64)));

#ifndef DISABLE_CONTRIB_OPS
  Status status = ::onnxruntime::contrib::webgpu::RegisterWebGpuContribKernels(*kernel_registry, enable_graph_capture);
  ORT_ENFORCE(status.IsOK(), "Failed to register WebGPU contrib kernels: " + status.ErrorMessage());
#endif

  return kernel_registry;
}

#if defined(ORT_USE_EP_API_ADAPTERS)

namespace {
std::mutex g_kernel_registry_mutex;
std::shared_ptr<KernelRegistry> g_kernel_registry;
std::shared_ptr<KernelRegistry> g_graph_capture_kernel_registry;
std::shared_ptr<KernelRegistry> g_int64_kernel_registry;
}  // namespace

void CleanupKernelRegistries() {
  std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
  g_kernel_registry.reset();
  g_graph_capture_kernel_registry.reset();
  g_int64_kernel_registry.reset();
}
#endif

std::shared_ptr<KernelRegistry> GetKernelRegistry(bool enable_graph_capture, bool enable_int64) {
  // kernel registry variables are defined differently based on build configuration
  //
  // - When building as a static library, use static local variable. This is because
  //   we don't have a reliable way to explicitly destroy the kernel registry after
  //   use.
  //
  // - When building as a shared library, use global variables. The cleanup will be performed
  //   when `ReleaseEpFactory` is called.
  //
  // Graph capture mode needs a separate kernel registry because contrib kernel registration
  // differs based on enable_graph_capture, and enable_int64 is always true when
  // enable_graph_capture is true.
  if (enable_graph_capture) {
#if !defined(ORT_USE_EP_API_ADAPTERS)
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(true, true);
    return registry;
#else
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_graph_capture_kernel_registry == nullptr) {
      g_graph_capture_kernel_registry = RegisterKernels(true, true);
    }
    return g_graph_capture_kernel_registry;
#endif
  } else if (enable_int64) {
#if defined(ORT_USE_EP_API_ADAPTERS)
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_int64_kernel_registry == nullptr) {
      g_int64_kernel_registry = RegisterKernels(false, true);
    }
    return g_int64_kernel_registry;
#else
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(false, true);
    return registry;
#endif
  } else {
#if defined(ORT_USE_EP_API_ADAPTERS)
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_kernel_registry == nullptr) {
      g_kernel_registry = RegisterKernels(false, false);
    }
    return g_kernel_registry;
#else
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(false, false);
    return registry;
#endif
  }
}

}  // namespace webgpu

using namespace webgpu;

WebGpuExecutionProvider::WebGpuExecutionProvider(int context_id,
                                                 WebGpuContext& context,
                                                 WebGpuExecutionProviderConfig&& config)
    : IExecutionProvider{kWebGpuExecutionProvider, WebGpuDevice(context_id)},
      context_id_{context_id},
      context_{context},
      preferred_data_layout_{config.data_layout},
      force_cpu_node_names_{std::move(config.force_cpu_node_names)},
      enable_graph_capture_{config.enable_graph_capture},
      // enable_int64_ is always true when enable_graph_capture_ is true
      enable_int64_{config.enable_graph_capture || config.enable_int64},
      multi_rotary_cache_concat_offset_{config.multi_rotary_cache_concat_offset},
      kv_cache_quantization_bits_{config.kv_cache_quantization_bits},
      enable_matmul_fp32_accumulation_{config.enable_matmul_fp32_accumulation},
      forced_matmul_algorithm_{config.forced_matmul_algorithm},
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
      weight_load_acceleration_mode_{config.weight_load_acceleration_mode},
      enable_uma_weight_loading_{config.enable_uma_weight_loading},
#endif
      recording_{std::make_unique<webgpu::CommandRecordingState>()},
      prepack_allocator_{CreateWebGpuAllocator(
          context_id,
          /*device_free=*/!context.HasDevice(),
          [this]() -> const webgpu::BufferManager& { return InitializerBufferManager(); },
          [this]() -> webgpu::CommandRecordingState& { return Recording(); }, false)} {
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  if (webgpu::IsWeightLoadAccelerationEnabled(
          config.weight_load_acceleration_mode) &&
      context_.HasDevice()) {
    accelerated_initializer_allocator_ =
        CreateD3D12AcceleratedWebGpuAllocator(
            context_,
            context_id_,
            [this]() -> webgpu::CommandRecordingState& { return Recording(); },
            accelerated_initializer_state_);
  }
#else
  if (webgpu::IsWeightLoadAccelerationRequired(
          config.weight_load_acceleration_mode)) {
    ORT_THROW(
        "The requested weightLoadAcceleration mode requires a supported "
        "disk-to-GPU weight loading implementation.");
  }
  if (webgpu::IsWeightLoadAccelerationEnabled(
          config.weight_load_acceleration_mode)) {
    LOGS_DEFAULT(WARNING)
        << "Accelerated weight loading is unavailable in this build; using "
           "the ordinary WebGPU initializer loading path.";
  }
#endif
  if (enable_graph_capture_ && config.session_buffer_pool_generations > 0) {
    session_buffer_pool_ = std::make_unique<webgpu::SessionBufferPool>(
        config.session_buffer_pool_generations);
  }
  if (config.enable_pix_capture) {
#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
    // set pix frame generator
    pix_frame_generator_ = context_.CreatePIXFrameGenerator();
#else
    ORT_THROW("Support PIX capture requires extra build flags (--enable_pix_capture)");
#endif  // ENABLE_PIX_FOR_WEBGPU_EP
  }

  WebGpuContextFactory::RetainContext(context_id_);
}

std::vector<AllocatorPtr> WebGpuExecutionProvider::CreatePreferredAllocators() {
  const bool device_free = !context_.HasDevice();
  return {
  // allocator for initializers
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
      context_.HasDevice() &&
              accelerated_initializer_allocator_ != nullptr
          ? accelerated_initializer_allocator_
          :
#endif
          CreateWebGpuAllocator(
              context_id_,
              device_free,
              [this]() -> const webgpu::BufferManager& { return InitializerBufferManager(); },
              [this]() -> webgpu::CommandRecordingState& { return Recording(); }, true),
      // default allocator
      CreateWebGpuAllocator(
          context_id_,
          device_free,
          [this]() -> const webgpu::BufferManager& { return BufferManager(); },
          [this]() -> webgpu::CommandRecordingState& { return Recording(); },
          false,
          [this]() { return !IsRunActive(); }),
  };
}

#if !defined(ORT_USE_EP_API_ADAPTERS)
std::vector<std::unique_ptr<ComputeCapability>> WebGpuExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& /* graph_optimizer_registry */,
    IResourceAccountant* /* resource_accountant */) const {
  InlinedVector<NodeIndex> candidates;
  // `tenative_candidates` is a subset of `candidates`.
  InlinedVector<NodeIndex> tenative_candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      // If the node was added by layout transformer, do not move it to CPU
      if (node.GetExecutionProviderType() == kWebGpuExecutionProvider) {
        candidates.push_back(node.Index());
      }
      continue;
    }

    const KernelCreateInfo* webgpu_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has a webgpu kernel for this node
    if (webgpu_kernel_def == nullptr) {
      LOGS(*GetLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                               << node.OpType() << " node name: " << node.Name();
      continue;
    }

    // TODO: currently this lookup is O(N). If the list becomes large we should optimize this.
    if (std::find(force_cpu_node_names_.cbegin(),
                  force_cpu_node_names_.cend(),
                  node.Name()) != force_cpu_node_names_.cend()) {
      LOGS(*GetLogger(), INFO) << "Force CPU execution for node: " << node.Name();
      continue;
    }

    //
    // The following code checks if the node is really supported by webgpu EP
    //

#define FALLBACK_TO_CPU_IF_EXIST_INPUT(idx)           \
  if (inputs.size() > idx && inputs[idx]->Exists()) { \
    continue;                                         \
  }

#define FALLBACK_TO_CPU_IF_EXIST_OUTPUT(idx)            \
  if (outputs.size() > idx && outputs[idx]->Exists()) { \
    continue;                                           \
  }

    // Check for Attention
    if (node.OpType() == "Attention" && node.Domain() == kMSDomain) {
      const auto& inputs = node.InputDefs();
      const auto& outputs = node.OutputDefs();

      // Current implementation does not support mask_index(input[3]), past(input[4]) and past_seq_len(input[6])
      FALLBACK_TO_CPU_IF_EXIST_INPUT(3);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(4);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(6);

      // Current implementation does not support present(output[1])
      FALLBACK_TO_CPU_IF_EXIST_OUTPUT(1);

      // If attribute past_present_share_buffer is set, fallback to CPU
      const auto& past_present_share_buffer = node.GetAttributes().find("past_present_share_buffer");
      if (past_present_share_buffer != node.GetAttributes().end() &&
          past_present_share_buffer->second.i() != 0) {
        continue;
      }
    }

    // Check for MatMulBnb4
    if (node.OpType() == "MatMulBnb4" && node.Domain() == kMSDomain) {
      // Current implementation only supports the forward case (transB=1). transB defaults to 1.
      const auto& trans_b = node.GetAttributes().find("transB");
      if (trans_b != node.GetAttributes().end() && trans_b->second.i() == 0) {
        continue;
      }
    }

    candidates.push_back(node.Index());
    tenative_candidates.push_back(node.Index());
  }

  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, tenative_candidates, *GetLogger());
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    if (cpu_nodes.contains(node_index)) {
      continue;
    }

    auto sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.emplace_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

#endif  // !defined(ORT_USE_EP_API_ADAPTERS)

std::unique_ptr<onnxruntime::IDataTransfer> WebGpuExecutionProvider::GetDataTransfer() const {
  return std::make_unique<webgpu::DataTransfer>(BufferManager(), Recording(), context_id_);
}

#if defined(__wasm__)
std::unique_ptr<onnxruntime::IExternalDataLoader> WebGpuExecutionProvider::GetExternalDataLoader() const {
  return std::make_unique<webgpu::ExternalDataLoader>();
}
#elif defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
std::unique_ptr<onnxruntime::IExternalDataLoader> WebGpuExecutionProvider::GetExternalDataLoader() const {
  if (accelerated_initializer_state_ == nullptr) {
    return nullptr;
  }

  return std::make_unique<webgpu::D3D12AcceleratedExternalDataLoader>(
      context_, context_id_, accelerated_initializer_state_,
      weight_load_acceleration_mode_, enable_uma_weight_loading_);
}
#endif

std::optional<bool> WebGpuExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                                          std::string_view node_op_type,
                                                                          DataLayout target_data_layout) const {
  // NHWC for Resize operator is not implemented on kWebGpuExecutionProvider
  if (node_domain == kOnnxDomain && node_op_type == "Resize") {
    return target_data_layout != DataLayout::NHWC;
  }

  // GridSample is NCHW-only (opset 16 spec requires NCHW input)
  if (node_domain == kOnnxDomain && node_op_type == "GridSample") {
    return target_data_layout != DataLayout::NHWC;
  }

  // WebGPU perfer NCHW for InstanceNormalization due to a better performance
  if (node_domain == kOnnxDomain && node_op_type == "InstanceNormalization") {
    return target_data_layout != DataLayout::NHWC;
  }

  return std::nullopt;
}

WebGpuExecutionProvider::~WebGpuExecutionProvider() {
  // Release all captured graphs (both fully captured and partially captured) and their associated resources.
  // Use captured_graphs_ keys to also cover partially captured graphs that have GPU command handles
  // but never completed capture (i.e., CaptureBegin was called but CaptureEnd was not).
  std::vector<int> graph_ids;
  graph_ids.reserve(captured_graphs_.size());
  for (const auto& [id, _] : captured_graphs_) {
    graph_ids.push_back(id);
  }
  for (int id : graph_ids) {
    auto status = ReleaseCapturedGraph(id);
    if (!status.IsOK()) {
      LOGS(*GetLogger(), WARNING) << "Failed to release captured graph " << id << ": " << status.ErrorMessage();
    }
  }
  // Release any per-graph buffer managers for graphs that had buffer managers created
  // but no entries in captured_graphs_ (edge case cleanup)
  per_graph_buffer_mgrs_.clear();

  // Release pooled buffers before the WebGpuContext is released.
  if (session_buffer_pool_) {
    session_buffer_pool_->Clear();
  }

  prepack_allocator_.reset();
  session_buffer_pool_.reset();
  if (context_.ActiveSerializedRecording() == recording_.get()) {
    context_.SetActiveSerializedRecording(nullptr);
  }
  if (context_.Device()) {
    // A failed Run may leave an unsubmitted recording in the context-shared pools.
    context_.BufferManager().DiscardPendingBuffers(*recording_);
    context_.InitializerBufferManager().DiscardPendingBuffers(*recording_);
  }
  recording_.reset();
#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  pix_frame_generator_.reset();
#endif

  WebGpuContextFactory::ReleaseContext(context_id_);
}

std::unique_ptr<profiling::EpProfiler> WebGpuExecutionProvider::GetProfiler() {
  auto profiler = std::make_unique<WebGpuProfiler>(context_);
  // Only set session_profiler_ on the first call (session-level profiler).
  // Subsequent calls from run-level profiling create temporary profilers that
  // should not overwrite it, as those profilers have shorter lifetimes.
  if (session_profiler_ == nullptr) {
    session_profiler_ = profiler.get();
  }
  return profiler;
}

Status WebGpuExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  const bool serialized = webgpu::ep::UseSerializedExecutionMode();
  if (serialized) {
    context_.SetActiveSerializedRecording(recording_.get());
  }
  bool started = false;
  auto release_on_error = gsl::finally([&] {
    if (serialized && !started) {
      graph_buffer_mgr_active_ = false;
      context_.SetActiveSerializedRecording(nullptr);
    }
  });
#endif
  if (context_.ValidationMode() >= ValidationMode::Basic) {
    context_.PushErrorScope();
  }

  // Start profiling if session-level or run-level profiling is enabled. The CPU time
  // base used to align GPU timestamps is pushed into the context by
  // WebGpuProfiler::StartProfiling, so no timepoint is threaded through here.
  if ((session_profiler_ && session_profiler_->Enabled()) || run_options.enable_profiling) {
    context_.StartProfiling();
  }

  if (IsGraphCaptureEnabled()) {
    auto graph_annotation_str = run_options.config_options.GetConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation);
    int graph_annotation_id = 0;
    if (graph_annotation_str.has_value()) {
      ORT_ENFORCE(onnxruntime::TryParseStringWithClassicLocale<int>(*graph_annotation_str, graph_annotation_id),
                  "Failed to parse the graph annotation id: ",
                  *graph_annotation_str);
    }

    current_graph_annotation_id_ = graph_annotation_id;

    // Create a per-graph buffer manager on first encounter of each annotation ID
    if (graph_annotation_id != -1) {
      auto [it, inserted] = per_graph_buffer_mgrs_.try_emplace(graph_annotation_id, nullptr);
      if (inserted) {
        it->second = webgpu::BufferManagerFactory::Create(
            context_,
            webgpu::BufferCacheMode::Graph,
            webgpu::BufferCacheMode::GraphSimple,
            webgpu::BufferCacheMode::Disabled,
            webgpu::BufferCacheMode::Disabled);
        if (session_buffer_pool_) {
          session_buffer_pool_->SeedInto(*it->second);
        }
      }
      graph_buffer_mgr_active_ = true;

      if (IsGraphCaptureAllowed() && !IsGraphCaptured(graph_annotation_id)) {
        auto& commands = captured_graphs_[graph_annotation_id];
        context_.CaptureBegin(&commands, *it->second, *recording_);
      }
    }
  }

  run_active_.store(true);
#if defined(ORT_USE_EP_API_ADAPTERS)
  started = true;
#endif
  return Status::OK();
}

Status WebGpuExecutionProvider::OnRunEnd(bool /* sync_stream */, const onnxruntime::RunOptions& run_options) {
  auto end_run = gsl::finally([&] {
    graph_buffer_mgr_active_ = false;
    run_active_.store(false);
#if defined(ORT_USE_EP_API_ADAPTERS)
    if (webgpu::ep::UseSerializedExecutionMode()) {
      context_.SetActiveSerializedRecording(nullptr);
    }
#endif
  });
  // When capturing, flushing creates the replay-ready dispatch entries before
  // CaptureEnd() detaches their external storage.
  Status flush_status = context_.Flush(BufferManager(), *recording_);

  if (!flush_status.IsOK()) {
    if (IsGraphCaptureEnabled()) {
      context_.CaptureEnd(*recording_);
      auto commands_it = captured_graphs_.find(current_graph_annotation_id_);
      if (commands_it != captured_graphs_.end()) {
        context_.ReleaseGraphResources(commands_it->second);
        commands_it->second.clear();
      }
    }
    graph_buffer_mgr_active_ = false;
    if (context_.ValidationMode() >= ValidationMode::Basic) {
      static_cast<void>(context_.PopErrorScope());
    }
    run_active_.store(false);
    return flush_status;
  }

  if (IsGraphCaptureEnabled() && !IsGraphCaptured(current_graph_annotation_id_)) {
    if (current_graph_annotation_id_ != -1 && IsGraphCaptureAllowed()) {
      context_.CaptureEnd(*recording_);
      captured_graph_ids_.insert(current_graph_annotation_id_);
      ORT_RETURN_IF_ERROR(ReplayGraph(current_graph_annotation_id_));
    } else {
      IncrementRegularRunCountBeforeGraphCapture();
    }
  }

  if (session_profiler_ && session_profiler_->Enabled()) {
    // Session-level profiling: collect into profiler's own events storage.
    context_.CollectProfilingData(session_profiler_->GpuEvents());
  } else if (run_options.enable_profiling) {
    // Run-level profiling: collect into shared events vector.
    context_.CollectProfilingData();
  }

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  if (pix_frame_generator_) {
    pix_frame_generator_->GeneratePIXFrame();
  }
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

  // Reset buffer manager routing after run completes
  graph_buffer_mgr_active_ = false;
  run_active_.store(false);

  if (context_.ValidationMode() >= ValidationMode::Basic) {
    return context_.PopErrorScope();
  } else {
    return Status::OK();
  }
}

bool WebGpuExecutionProvider::IsGraphCaptureEnabled() const {
  return enable_graph_capture_;
}

bool WebGpuExecutionProvider::IsGraphCaptured(int graph_annotation_id) const {
  return graph_annotation_id != -1 && captured_graph_ids_.contains(graph_annotation_id);
}

Status WebGpuExecutionProvider::ReplayGraph(int graph_annotation_id, bool /*sync*/) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  const bool serialized_replay = webgpu::ep::UseSerializedExecutionMode() && !IsRunActive();
  if (serialized_replay) {
    context_.SetActiveSerializedRecording(recording_.get());
  }
  auto release_serialized_run = gsl::finally([&] {
    if (serialized_replay) {
      context_.SetActiveSerializedRecording(nullptr);
    }
  });
#endif
  // The sync parameter is ignored: WebGPU EP always replays synchronously.
  ORT_ENFORCE(IsGraphCaptured(graph_annotation_id));
#if defined(ORT_USE_EP_API_ADAPTERS)
  if (serialized_replay) {
    ORT_RETURN_IF_ERROR(context_.Flush(context_.BufferManager(), *recording_));
  }
#endif
  // TODO: enable profiling in run level
  if (session_profiler_ && session_profiler_->Enabled()) {
    context_.StartProfiling();
  }
  context_.Replay(captured_graphs_.at(graph_annotation_id),
                  *per_graph_buffer_mgrs_.at(graph_annotation_id),
                  *recording_);
  if (session_profiler_ && session_profiler_->Enabled()) {
    // Session-level profiling: collect into profiler's own events storage.
    context_.CollectProfilingData(session_profiler_->GpuEvents());
  }
  return Status::OK();
}

Status WebGpuExecutionProvider::ReleaseCapturedGraph(int graph_annotation_id) {
  // Release captured commands
  auto cmd_it = captured_graphs_.find(graph_annotation_id);
  if (cmd_it != captured_graphs_.end()) {
    if (!cmd_it->second.empty()) {
      context_.ReleaseGraphResources(cmd_it->second);
    }
    captured_graphs_.erase(cmd_it);
  }

  // Remove from captured set
  captured_graph_ids_.erase(graph_annotation_id);

  // Release per-graph buffer manager (donate to session pool if enabled; otherwise destroy)
  auto mgr_it = per_graph_buffer_mgrs_.find(graph_annotation_id);
  if (mgr_it != per_graph_buffer_mgrs_.end()) {
    if (session_buffer_pool_ && mgr_it->second) {
      session_buffer_pool_->Donate(*mgr_it->second);
    }
    per_graph_buffer_mgrs_.erase(mgr_it);
  }

  // Clean up run count tracking
  graph_id_to_run_count_.erase(graph_annotation_id);

  return Status::OK();
}

webgpu::BufferManager& WebGpuExecutionProvider::BufferManager() const {
  if (IsGraphCaptureEnabled() && graph_buffer_mgr_active_) {
    auto it = per_graph_buffer_mgrs_.find(current_graph_annotation_id_);
    if (it != per_graph_buffer_mgrs_.end()) {
      return *it->second;
    }
  }
  return context_.BufferManager();
}

webgpu::BufferManager& WebGpuExecutionProvider::InitializerBufferManager() const {
  return context_.InitializerBufferManager();
}

bool WebGpuExecutionProvider::IsGraphCaptureAllowed() const {
  auto it = graph_id_to_run_count_.find(current_graph_annotation_id_);
  int run_count = (it != graph_id_to_run_count_.end()) ? it->second : 0;
  return run_count >= min_num_runs_before_graph_capture_;
}

void WebGpuExecutionProvider::IncrementRegularRunCountBeforeGraphCapture() {
  ++graph_id_to_run_count_[current_graph_annotation_id_];
}
}  // namespace onnxruntime
