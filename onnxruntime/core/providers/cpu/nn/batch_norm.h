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

#pragma once

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include <safeint/SafeInt.hpp>

namespace onnxruntime {

template <typename T>
class BatchNorm : public OpKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& op_kernel_info)
      : OpKernel(op_kernel_info), is_spatial_(op_kernel_info.GetAttrOrDefault<int64_t>("spatial", 1) == 1) {
    epsilon_ = op_kernel_info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    momentum_ = op_kernel_info.GetAttrOrDefault<float>("momentum", 0.9f);

    // For opset 6-8, if spatial attribute exists, pick up the value (by default spatial == 1)
    // From opset 9 onwards, by default, only the spatial case (spatial == 1) is defined per spec
    // For opset 14 onwards, training is an attribute.
    // For opset < 14, since no training attribute is present we assume optional outputs indicate training mode.
    if (op_kernel_info.node().SinceVersion() == 14) {
      is_train_ = op_kernel_info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1;
      size_t output_count = op_kernel_info.node().OutputDefs().size();
      ORT_ENFORCE((is_train_ && output_count == 3) || (!is_train_ && output_count == 1),
                  "Output running_mean and running_var are valid and required for training mode.");
    } else {
      is_train_ = OpKernel::Node().OutputDefs().size() > 1;
    }

    ORT_ENFORCE(!is_train_ || is_spatial_, "Training mode does not support non-spatial BN");
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    const auto* X = p_op_kernel_context->Input<Tensor>(0);
    const auto* scale = p_op_kernel_context->Input<Tensor>(1);
    const auto* B = p_op_kernel_context->Input<Tensor>(2);
    const auto* mean = p_op_kernel_context->Input<Tensor>(3);
    const auto* var = p_op_kernel_context->Input<Tensor>(4);

    ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, is_spatial_));

    const TensorShape& x_shape = X->Shape();
    Tensor* Y = p_op_kernel_context->Output(0, x_shape);

    const auto& dims_vec = x_shape.GetDims();
    const size_t N = dims_vec[0];
    const size_t C = dims_vec[1];  // assume NCHW as per the spec

    // calculate sample_size (per individual channel)
    size_t sample_size = 1;
    for (size_t i = 2; i < dims_vec.size(); ++i) {
      sample_size *= gsl::narrow<size_t>(dims_vec[i]);
    }

    // calculate sample_size (including all channels)
    size_t sample_size_incl_all_channels = sample_size * C;

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(p_op_kernel_context->GetTempSpaceAllocator(&alloc));

    // Saved mean corresponds to the mean from this batch
    // If these optional outputs are present (opset <= 9 or internal BN op) we re-use the space for calculations
    // Note that with opset <= 9 we will be outputting saved_inv_std_dev instead of saved_var
    Tensor* saved_mean = is_train_ ? p_op_kernel_context->Output(3, mean->Shape()) : nullptr;
    Tensor* saved_inv_std = is_train_ ? p_op_kernel_context->Output(4, var->Shape()) : nullptr;
    // With opset <= 9, both must be defined in training. If opset >= 14, neither should be defined in training
    ORT_ENFORCE(!is_train_ || ((!saved_mean && !saved_inv_std) || (saved_mean && saved_inv_std)),
                "Invalid number of outputs for BN training");
    Tensor saved_mean_allocated, saved_inv_std_allocated;
    if (is_train_ && !saved_mean) {
      saved_mean_allocated = Tensor(DataTypeImpl::GetType<T>(), mean->Shape(), alloc);
      saved_inv_std_allocated = Tensor(DataTypeImpl::GetType<T>(), var->Shape(), alloc);
      saved_mean = &saved_mean_allocated;
      saved_inv_std = &saved_inv_std_allocated;
    }

    ConstEigenArrayMap<T> X_arr(X->template Data<T>(),
                                is_spatial_ ? sample_size : sample_size_incl_all_channels,
                                is_spatial_ ? N * C : N);
    ConstEigenVectorArrayMap<T> scale_arr(scale->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
    ConstEigenVectorArrayMap<T> bias_arr(B->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);

    // Note that we only support spatial BN for training
    if (is_train_) {
      EigenVectorArrayMap<T> saved_mean_arr(saved_mean->template MutableData<T>(), C);
      // We first calculate saved_var then later take inverse square root to get saved_inv_std
      EigenVectorArrayMap<T> saved_var_arr(saved_inv_std->template MutableData<T>(), C);
      saved_mean_arr.setZero();
      saved_var_arr.setZero();

      for (size_t nc = 0; nc < N * C; ++nc) {
        saved_mean_arr(nc % C) += X_arr.col(nc).sum();
      }

      saved_mean_arr /= static_cast<T>(N * sample_size);
      for (size_t nc = 0; nc < N * C; ++nc) {
        saved_var_arr(nc % C) += (X_arr.col(nc) - saved_mean_arr(nc % C)).matrix().squaredNorm();
      }
      saved_var_arr /= static_cast<T>(N * sample_size);

      // The running mean corresponds to the mean from all the batches
      // During inference this running mean is used as the mean for BN
      auto* running_mean = p_op_kernel_context->Output(1, mean->Shape());
      auto* running_var = p_op_kernel_context->Output(2, var->Shape());
      const auto* input_running_mean = p_op_kernel_context->Input<Tensor>(3);
      const auto* input_running_var = p_op_kernel_context->Input<Tensor>(4);

      // Assume that running mean and variance are initialized properly in the model given to us
      // Because we alias it, we have the past history here
      EigenVectorArrayMap<T> running_mean_arr(
          running_mean->template MutableData<T>(), C);
      EigenVectorArrayMap<T> running_var_arr(
          running_var->template MutableData<T>(), C);
      ConstEigenVectorArrayMap<T> input_running_mean_arr(
          input_running_mean->template Data<T>(), C);
      ConstEigenVectorArrayMap<T> input_running_var_arr(
          input_running_var->template Data<T>(), C);
      running_mean_arr = input_running_mean_arr * momentum_ + saved_mean_arr * (1. - momentum_);
      running_var_arr = input_running_var_arr * momentum_ + saved_var_arr * (1. - momentum_);
    }

    // Regardless of training or testing, we will apply the estimated mean
    // and standard deviation to the input. For testing, they are
    // specified directly by the input, and for training, they are computed
    // by the op.
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(is_spatial_ ? C : sample_size_incl_all_channels);

    if (!is_train_) {
      ConstEigenVectorArrayMap<T> var_arr(var->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
      inv_std = (var_arr + epsilon_).sqrt().inverse();
    } else {
      EigenVectorArrayMap<T> saved_inv_std_arr(saved_inv_std->template MutableData<T>(), C);
      saved_inv_std_arr = (saved_inv_std_arr + epsilon_).inverse().sqrt();
      inv_std = saved_inv_std_arr;
    }

    // If we're training, do batch normalization based on computation from this batch
    ConstEigenVectorArrayMap<T> mean_arr(!is_train_ ? mean->template Data<T>() : saved_mean->template Data<T>(),
                                         is_spatial_ ? C : sample_size_incl_all_channels);

    // We can fuse the output computation as follows:
    //   ((x - est_mean) * (inv_std)) * scale + bias
    // to
    //   (x * inv_std * scale) + (bias - est_mean * inv_std * scale)
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
    EigenArrayMap<T> Y_arr(Y->template MutableData<T>(),
                           is_spatial_ ? sample_size : sample_size_incl_all_channels,
                           is_spatial_ ? N * C : N);

    if (is_spatial_) {  // spatial == 1
      for (size_t nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
    } else {  // spatial == 0
      for (size_t n = 0; n < N; ++n) {
        Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
      }
    }
    return Status::OK();
  }

 protected:
  float epsilon_;
  float momentum_;
  const bool is_spatial_;
  int64_t is_train_;
};
}  // namespace onnxruntime
