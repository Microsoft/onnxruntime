// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/status.h"
#include "core/framework/tensor.h"
#endif
#include <sstream>

namespace onnxruntime {
class BatchNormHelper {
 public:
  static common::Status ValidateInputs(const Tensor* X,
                                       const Tensor* scale,
                                       const Tensor* B,
                                       const Tensor* mean,
                                       const Tensor* var,
                                       bool is_spatial = true) {
    const auto& x_dims = X->Shape().GetDims();

    int64_t num_channels = x_dims.size() > 1 ? x_dims[1] : 1;
    // the first 2 are respectively - N and C. If x_dims size < 2, num_channels defaults to 1.
    int num_feature_dims = X->Shape().NumDimensions() >= 2 ? static_cast<int>(X->Shape().NumDimensions() - 2) : 1;

    // defined as per spec and used for validation
    int kNumInputScaleDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputBiasDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputMeanDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputVarianceDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    //constexpr int kMinCudaNumDims = 4;
    //constexpr int kMaxCudaNumDims = 5;

    // validate 'scales' shape
    const auto& scale_dims = scale->Shape().GetDims();
    if (static_cast<int>(scale_dims.size()) != kNumInputScaleDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: NumDimensions() != ", kNumInputScaleDimensions);
    }
    if (scale_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'scale' must be validated
    if (!is_spatial) {
      for (int feature = 0; feature < num_feature_dims; ++feature) {
        if (scale_dims[1 + feature] != x_dims[2 + feature]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: ", (1 + feature), " dimension != ", x_dims[2 + feature]);
        }
      }
    }

    // validate 'B' shape
    const auto& B_dims = B->Shape().GetDims();
    if (static_cast<int>(B_dims.size()) != kNumInputBiasDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: NumDimensions() != ", kNumInputBiasDimensions);
    }
    if (B_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'B' must be validated
    if (!is_spatial) {
      for (int feature = 0; feature < num_feature_dims; ++feature) {
        if (B_dims[1 + feature] != x_dims[2 + feature]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: ", (1 + feature), " dimension != ", x_dims[2 + feature]);
        }
      }
    }

    // validate 'mean' shape
    const auto& mean_dims = mean->Shape().GetDims();
    if (static_cast<int>(mean_dims.size()) != kNumInputMeanDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: NumDimensions() != ", kNumInputMeanDimensions);
    }
    if (mean_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'mean' must be validated
    if (!is_spatial) {
      for (int feature = 0; feature < num_feature_dims; ++feature) {
        if (mean_dims[1 + feature] != x_dims[2 + feature]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: ", (1 + feature), " dimension != ", x_dims[2 + feature]);
        }
      }
    }

    // validate 'var' shape
    const auto& var_dims = var->Shape().GetDims();
    if (static_cast<int>(var_dims.size()) != kNumInputVarianceDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: NumDimensions() != ", kNumInputVarianceDimensions);
    }
    if (var_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'var' must be validated
    if (!is_spatial) {
      for (int feature = 0; feature < num_feature_dims; ++feature) {
        if (var_dims[1 + feature] != x_dims[2 + feature]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: ", (1 + feature), " dimension != ", x_dims[2 + feature]);
        }
      }
    }

    return common::Status::OK();
  }

  static void NormalizeDims(const TensorShape& x_shape, std::vector<int64_t>& new_dims) {
    new_dims.clear();
    auto& orig_dims = x_shape.GetDims();
    ORT_ENFORCE(orig_dims.size() < 6,
                "Input dim size should be < 6 for BatchNorm, but got ", std::to_string(orig_dims.size()));
    if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
        orig_dims.size() == 5 /*supported size by CUDA*/) {
      new_dims = orig_dims;
      return;
    }

    auto rank = x_shape.NumDimensions();
    auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
    auto num_channels = rank > 1 ? orig_dims[1] : 1;
    auto height = rank > 2 ? orig_dims[2] : 1;
    int64_t width = 1;
    new_dims = {num_samples, num_channels, height, width};
  }
};
}  // namespace onnxruntime
