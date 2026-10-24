// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/quantize_linear.h"
#include "core/providers/cuda/tensor/quantize_linear_common.h"
#include "dequantize_fp8_bf16.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_Q_KERNEL_TYPED(T, U)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      QuantizeLinear,                                              \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      QuantizeLinear<T, U>);

REGISTER_Q_KERNEL_TYPED(int8_t, MLFloat16)
REGISTER_Q_KERNEL_TYPED(uint8_t, MLFloat16)

#define REGISTER_DQ_KERNEL_TYPED(T, U)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      DequantizeLinear,                                            \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("TS", DataTypeImpl::GetTensorType<U>()), \
      DequantizeLinear<T, U>);

REGISTER_DQ_KERNEL_TYPED(int8_t, MLFloat16)
REGISTER_DQ_KERNEL_TYPED(uint8_t, MLFloat16)

#if !defined(DISABLE_FLOAT8_TYPES)
class DequantizeLinearFp8Bf16 final : public CudaKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DequantizeLinearFp8Bf16);
  explicit DequantizeLinearFp8Bf16(const OpKernelInfo& info) : CudaKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    const auto& scale = *ctx->Input<Tensor>(1);
    const auto* zero_point = ctx->Input<Tensor>(2);
    const auto& shape = input.Shape();
    const int64_t count = shape.Size();
    ORT_RETURN_IF_NOT(IsQDQElementCountSupported(count), QDQElementCountErrorMessage());
    ORT_RETURN_IF_NOT(scale.Shape().NumDimensions() == 0 || scale.Shape().NumDimensions() == 1,
                      "x_scale must be a scalar or 1D tensor.");
    ORT_RETURN_IF_NOT(zero_point == nullptr || zero_point->Shape() == scale.Shape(),
                      "x_zero_point and x_scale must have the same shape.");

    const int64_t scale_count = scale.Shape().Size();
    ORT_RETURN_IF_NOT(scale_count > 0 || count == 0, "x_scale must not be empty.");
    int64_t axis_stride = 1;
    if (scale_count != 1) {
      const int64_t axis = HandleNegativeAxis(axis_, shape.NumDimensions());
      ORT_RETURN_IF_NOT(scale_count == shape[axis],
                        "x_scale must match the input dimension along axis.");
      axis_stride = shape.SizeFromDimension(axis + 1);
    }

    auto* output = ctx->Output(0, shape);
    if (count == 0) {
      return Status::OK();
    }
    DequantizeFp8Bf16(Stream(ctx), input.Data<Float8E4M3FN>(), scale.Data<float>(),
                      zero_point == nullptr ? nullptr : zero_point->Data<Float8E4M3FN>(),
                      output->MutableData<BFloat16>(), static_cast<size_t>(count),
                      static_cast<size_t>(axis_stride), static_cast<size_t>(scale_count));
    return Status::OK();
  }

 private:
  int64_t axis_;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DequantizeLinear,
    kMSDomain,
    1,
    Float8E4M3FN_BFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<Float8E4M3FN>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinearFp8Bf16);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
