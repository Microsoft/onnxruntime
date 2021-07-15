// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/batch_norm_internal.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace std;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, T1, T2)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      BatchNormInternal,                                            \
      kMSDomain,                                                    \
      1,                                                            \
      T##_##T1##_##T2,                                              \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .Alias(3, 1)                                              \
          .Alias(4, 2)                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>()), \
      BatchNormInternal<T, T1, T2>);

template <typename T, typename T1, typename T2>
Status BatchNormInternal<T, T1, T2>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* B = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* mean = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));

  const TensorShape& x_shape = X->Shape();
  const TensorShape& channel_shape = mean->Shape();

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  Tensor* running_mean = p_op_kernel_context->Output(1, channel_shape);
  Tensor* running_var = p_op_kernel_context->Output(2, channel_shape);
  Tensor* saved_mean = p_op_kernel_context->Output(3, channel_shape);
  Tensor* saved_inv_std = p_op_kernel_context->Output(4, channel_shape);

  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT1*>(scale->template Data<T1>());
  auto b_data = reinterpret_cast<const CudaT1*>(B->template Data<T1>());
  auto mean_data = reinterpret_cast<const CudaT2*>(mean->template Data<T2>());
  auto var_data = reinterpret_cast<const CudaT2*>(var->template Data<T2>());

  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  CudnnTensor data_desc, bn_tensor_desc;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(x_shape, new_dims);
  ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));
  // for fp16 input, `bn_tensor_desc` will have a float type; otherwise it will be the same as input type.
  ORT_RETURN_IF_ERROR(bn_tensor_desc.Set(data_desc, cudnn_batch_norm_mode_));

  auto running_mean_data = reinterpret_cast<CudaT2*>(running_mean->template MutableData<T2>());
  auto running_var_data = reinterpret_cast<CudaT2*>(running_var->template MutableData<T2>());
  auto saved_mean_data = reinterpret_cast<CudaT2*>(saved_mean->template MutableData<T2>());
  auto saved_inv_std_data = reinterpret_cast<CudaT2*>(saved_inv_std->template MutableData<T2>());

  auto p_scale = reinterpret_cast<const void*>(scale_data);
  auto p_B = reinterpret_cast<const void*>(b_data);
  auto p_running_mean = reinterpret_cast<void*>(running_mean_data);
  auto p_running_var = reinterpret_cast<void*>(running_var_data);
  auto p_saved_mean = reinterpret_cast<void*>(saved_mean_data);
  auto p_saved_inv_std = reinterpret_cast<void*>(saved_inv_std_data);


  const int64_t C = new_dims[1];
  IAllocatorUniquePtr<float> p_f_scale, p_f_B, p_f_running_mean, p_f_running_var, p_f_saved_mean, p_f_saved_inv_std;

  if (std::is_same<T1, MLFloat16>::value) {
    // Convert scale/B to float
    p_f_scale = GetScratchBuffer<float>(C);
    p_f_B = GetScratchBuffer<float>(C);

    Impl_Cast<CudaT1, float>(Stream(), scale_data, p_f_scale.get(), C);
    Impl_Cast<CudaT1, float>(Stream(), b_data, p_f_B.get(), C);

    p_scale = p_f_scale.get();
    p_B = p_f_B.get();
  }

  if (std::is_same<T2, MLFloat16>::value) {
    // Convert mean/var to float
    p_f_running_mean = GetScratchBuffer<float>(C);
    p_f_running_var = GetScratchBuffer<float>(C);
    p_f_saved_mean = GetScratchBuffer<float>(C);
    p_f_saved_inv_std = GetScratchBuffer<float>(C);

    Impl_Cast<CudaT2, float>(Stream(), mean_data, p_f_running_mean.get(), C);
    Impl_Cast<CudaT2, float>(Stream(), var_data, p_f_running_var.get(), C);

    p_running_mean = p_f_running_mean.get();
    p_running_var = p_f_running_var.get();
    p_saved_mean = p_f_saved_mean.get();
    p_saved_inv_std = p_f_saved_inv_std.get();
  } else if (mean_data != running_mean_data) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(running_mean_data, mean_data, C * sizeof(T2), cudaMemcpyDeviceToDevice, Stream()));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(running_var_data, var_data, C * sizeof(T2), cudaMemcpyDeviceToDevice, Stream()));
  }

  CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
      CudnnHandle(),
      cudnn_batch_norm_mode_,
      &alpha,
      &beta,
      data_desc,
      x_data,
      data_desc,
      y_data,
      bn_tensor_desc,
      p_scale,
      p_B,
      1.0 - momentum_,
      p_running_mean,
      p_running_var,
      epsilon_,
      p_saved_mean,
      p_saved_inv_std));

  if (std::is_same<T2, MLFloat16>::value) {
    Impl_Cast<float, CudaT2>(Stream(), reinterpret_cast<float*>(p_running_mean), running_mean_data, C);
    Impl_Cast<float, CudaT2>(Stream(), reinterpret_cast<float*>(p_running_var), running_var_data, C);
    Impl_Cast<float, CudaT2>(Stream(), reinterpret_cast<float*>(p_saved_mean), saved_mean_data, C);
    Impl_Cast<float, CudaT2>(Stream(), reinterpret_cast<float*>(p_saved_inv_std), saved_inv_std_data, C);
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T, T1, T2) \
  REGISTER_KERNEL_TYPED(T, T1, T2)     \
  template Status BatchNormInternal<T, T1, T2>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float, float, float)
SPECIALIZED_COMPUTE(double, double, double)
SPECIALIZED_COMPUTE(MLFloat16, MLFloat16, MLFloat16)
SPECIALIZED_COMPUTE(MLFloat16, MLFloat16, float)
SPECIALIZED_COMPUTE(MLFloat16, float, float)

}  // namespace cuda
}  // namespace onnxruntime
