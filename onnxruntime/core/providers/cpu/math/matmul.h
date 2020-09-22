// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class MatMul : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

#if !defined(USE_MKLML_FOR_BLAS)

template <>
class MatMul<float> : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
  }

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;
  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;

 protected:
  // For FusedMatMul and TransposeMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
};

#else

template <>
class MatMul<float> : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  // For FusedMatMul and TransposeMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
};

#endif
}  // namespace onnxruntime
