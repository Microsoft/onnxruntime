// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "layer_norm.h"
#include "layer_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kOnnxDomain,                                                \
      1,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      LayerNorm<T, U, false>);                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SimplifiedLayerNormalization,                               \
      kOnnxDomain,                                                \
      1,                                                          \
      T##_##U,                                                    \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()), \
      LayerNorm<T, U, true>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(double, double)
REGISTER_KERNEL_TYPED(MLFloat16, float)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_KERNEL_TYPED(BFloat16, float)
#endif

template <typename T, typename U, bool simplified>
LayerNorm<T, U, simplified>::LayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
}

template <typename T, typename U, bool simplified>
Status LayerNorm<T, U, simplified>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  //Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = ctx->Input<Tensor>(2);

  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto bias_data = (simplified || (nullptr == bias)) ? nullptr : reinterpret_cast<const CudaT*>(bias->template Data<T>());

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());

  int n1 = gsl::narrow<int>(x_shape.SizeToDimension(axis));
  int n2 = gsl::narrow<int>(x_shape.SizeFromDimension(axis));

  ORT_ENFORCE(n2 != 1, "n2 should not be 1");

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  // Mean and standard deviation
  std::vector<int64_t> mean_inv_std_dim;
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_dim.emplace_back(1);
    }
  }
  int output_index = 1;

  CudaU* mean_data = nullptr;
  if (!simplified) {
    Tensor* mean = ctx->Output(output_index++, TensorShape(mean_inv_std_dim));
    if (mean != nullptr) {
      mean_data = reinterpret_cast<CudaU*>(mean->template MutableData<U>());
    }
  }

  Tensor* std = ctx->Output(output_index, TensorShape(mean_inv_std_dim));
  CudaU* inv_std_data = nullptr;
  if (std != nullptr) {
    inv_std_data = reinterpret_cast<CudaU*>(std->template MutableData<U>());
  }

  HostApplyLayerNorm<CudaT, CudaU, simplified>(GetDeviceProp(), Stream(), Y_data, mean_data, inv_std_data, X_data, n1, n2, epsilon_, scale_data, bias_data);
  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
