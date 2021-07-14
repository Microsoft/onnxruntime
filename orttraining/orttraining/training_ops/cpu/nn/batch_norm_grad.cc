/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "batch_norm_grad.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status BatchNormalizationGrad<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* Scale = ctx->Input<Tensor>(2);
  const Tensor* saved_mean = ctx->Input<Tensor>(3);
  const Tensor* saved_inv_std = ctx->Input<Tensor>(4);

  const TensorShape X_shape = X->Shape();
  const TensorShape channel_shape = saved_mean->Shape();

  // no B here, but B has same size as Scale, so can validate inputs for gradient with this substitute
  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, Scale, Scale, saved_mean, saved_inv_std));

  const auto* dY_data = dY->template Data<T>();
  const auto* X_data = X->template Data<T>();
  const auto* Scale_data = Scale->template Data<T>();
  const auto* saved_mean_data = saved_mean->template Data<T>();
  const auto* saved_inv_std_data = saved_inv_std->template Data<T>();

  auto* dX_data = ctx->Output(0, X_shape)->template MutableData<T>();
  auto* dScale_data = ctx->Output(1, channel_shape)->template MutableData<T>();
  auto* dBias_data = ctx->Output(2, channel_shape)->template MutableData<T>();

  const auto& dims_vec = X_shape.GetDims();
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size (per individual channel)
  size_t sample_size = X_shape.SizeFromDimension(2);
  size_t scale_tensor_size = C;

  ConstEigenVectorArrayMap<T> scale_arr(Scale_data, scale_tensor_size);
  ConstEigenVectorArrayMap<T> mean_arr(saved_mean_data, scale_tensor_size);
  ConstEigenVectorArrayMap<T> inv_std_arr(saved_inv_std_data, scale_tensor_size);

  EigenVectorArrayMap<T> dBias_arr(dBias_data, scale_tensor_size);
  EigenVectorArrayMap<T> dScale_arr(dScale_data, scale_tensor_size);

  dBias_arr.setZero();
  dScale_arr.setZero();

  const auto scaled_inv_std = scale_arr * inv_std_arr / (N * sample_size);

  ConstEigenArrayMap<T> X_arr(X_data, sample_size, N * C);
  ConstEigenArrayMap<T> dY_arr(dY_data, sample_size, N * C);
  EigenArrayMap<T> dX_arr(dX_data, sample_size, N * C);

  for (size_t nc = 0; nc < N * C; ++nc) {
    int c = nc % C;
    dBias_arr(c) += dY_arr.col(nc).sum();
    dScale_arr(c) +=
        ((X_arr.col(nc) - mean_arr(c)) * inv_std_arr(c) * dY_arr.col(nc)).sum();
  }
  for (size_t nc = 0; nc < N * C; ++nc) {
    int c = nc % C;
    dX_arr.col(nc) = scaled_inv_std(c) *
                      (dY_arr.col(nc) * N * sample_size - dBias_arr(c) -
                       (X_arr.col(nc) - mean_arr(c)) * dScale_arr(c) * inv_std_arr(c));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    BatchNormalizationGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BatchNormalizationGrad<float>);

}  // namespace contrib
}  // namespace onnxruntime
