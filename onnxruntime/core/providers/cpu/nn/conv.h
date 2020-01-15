// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

template <typename T>
class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
};

template <>
class Conv<float> : public OpKernel {
 public:
  Conv<float>(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    activation_.ActivationKind = MlasIdentityActivation;
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  MLAS_ACTIVATION activation_;

  ConvAttributes conv_attrs_;
};

}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {

template <typename T>
class ConvGrad final : public OpKernel {
 public:
  explicit ConvGrad(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
   ConvAttributes conv_attrs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad);
};

}  // namespace contrib
}  // namespace onnxruntime
