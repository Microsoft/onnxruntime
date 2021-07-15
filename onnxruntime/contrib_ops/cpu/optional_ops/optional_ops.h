// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class Optional final : public OpKernel {
 public:
  explicit Optional(const OpKernelInfo& info) : OpKernel(info) {
    const auto* attr = info.TryGetAttribute("type");

    if (attr) {
      ORT_ENFORCE(attr->has_tp(), "Optional op must have a TypeProto in the 'type' attribute if the attribute is present");
      type_proto_ = &attr->tp();
    }
  }

  common::Status Compute(OpKernelContext* context) const override;

 private:
  const onnx::TypeProto* type_proto_ = nullptr;
};

class OptionalHasElement final : public OpKernel {
 public:
  explicit OptionalHasElement(const OpKernelInfo& info) : OpKernel(info) {}

  common::Status Compute(OpKernelContext* context) const override;
};

class OptionalGetElement final : public OpKernel {
 public:
  explicit OptionalGetElement(const OpKernelInfo& info) : OpKernel(info) {}

  common::Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
