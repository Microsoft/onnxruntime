// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

std::vector<std::vector<int64_t>> InferOutputShapes(OpKernelInfo info);

template <typename T>
class SinGrad final : public OpKernel {
 public:
  explicit SinGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SinGrad);
};

template <typename T>
class ReluGrad final : public OpKernel {
 public:
  explicit ReluGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReluGrad);
};

template <typename T>
class PowGrad final : public OpKernel {
 public:
  explicit PowGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PowGrad);
};

template <typename T>
class SigmoidGrad final : public OpKernel {
 public:
  explicit SigmoidGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SigmoidGrad);
};

template <typename T>
class SoftmaxGrad final : public OpKernel {
 public:
  explicit SoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxGrad);
  int64_t axis_;
};
}  // namespace contrib
}  // namespace onnxruntime
