// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <functional>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

/**
Handle a potentially negative axis. Enforces negative axis is valid.
@param axis Axis to convert from negative to positive if needed.
@param tensor_rank Rank of tensor axis applies to. Tensor::Shape()::NumDimensions().
@returns non-negative axis.
*/
inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  ORT_ENFORCE(axis >= -tensor_rank && axis <= tensor_rank - 1, "axis ", axis,
              " is not in valid range [-", tensor_rank, ",", tensor_rank - 1, "]");
  // Handle negative axis
  return axis < 0 ? axis + tensor_rank : axis;
}

/**
Returns true if given tensor is a scalar or 1D tensor of size 1
**/
inline bool IsScalarOr1ElementVector(const Tensor* input) {
  if (input->Shape().NumDimensions() == 0 ||
      (input->Shape().NumDimensions() == 1 && input->Shape().Size() == 1)) {
    return true;
  } else {
    return false;
  }
}

/**
Clamps input between provided min and max values
**/
inline float clamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_ENFORCE(false, "Unknown AutoPadType String");
}

// helper function

inline Status ComputePad(const int64_t in_dim,
                         const int64_t stride, const int64_t kernel, const int64_t dilation,
                         AutoPadType pad_type,
                         int64_t& pad_head, int64_t& pad_tail,
                         bool force_symmetric_auto_padding = false) {
  switch (pad_type) {
    case AutoPadType::NOTSET:
      break;
    case AutoPadType::VALID: {
      pad_head = 0;
      pad_tail = 0;
    } break;
    case AutoPadType::SAME_UPPER:
    case AutoPadType::SAME_LOWER: {
      if (1 != dilation)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");

      // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
      // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
      // is retained as is
      int64_t legacy_target_size = (in_dim + stride - 1) / stride;
      int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
      // make sure padding is symmetric
      if (force_symmetric_auto_padding) {
        // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
        pad_needed = (pad_needed + 1) & ~1;
      }

      if (pad_type == AutoPadType::SAME_LOWER)
        pad_head = (pad_needed + 1) / 2;
      else
        pad_head = pad_needed / 2;

      pad_tail = pad_needed - pad_head;
    } break;
    default:
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "ComputePad: pad type not supported.");
  }

  return Status::OK();
}

inline int64_t ComputeOutputShape(const int64_t in_dim,
                                  const int64_t stride, const int64_t kernel, const int64_t dilation,
                                  const int64_t pad_head, const int64_t pad_tail) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;
  return static_cast<int64_t>(static_cast<float>(in_dim + pad_head + pad_tail - dkernel) / stride + 1);
}

inline Status ComputePadAndOutputShape(const int64_t in_dim,
                                       const int64_t stride, const int64_t kernel, const int64_t dilation,
                                       AutoPadType pad_type,
                                       int64_t& pad_head, int64_t& pad_tail,
                                       int64_t& out_dim,
                                       bool force_symmetric_auto_padding = false) {
  ORT_RETURN_IF_ERROR(
      ComputePad(in_dim, stride, kernel, dilation, pad_type, pad_head, pad_tail, force_symmetric_auto_padding));
  out_dim = ComputeOutputShape(in_dim, stride, kernel, dilation, pad_head, pad_tail);
  return Status::OK();
}

template <class AssociativeContainer, class Key>
inline bool Contains(const AssociativeContainer& container, const Key& key) {
  return container.find(key) != container.end();
}

// Note: This helper function will not have overflow protection
template <template <typename...> class Container, typename T>
T Product(const Container<T>& c) {
  return accumulate(c.cbegin(), c.cend(), 1, std::multiplies<T>());
}

}  // namespace onnxruntime
