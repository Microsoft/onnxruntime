// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// TODO(kreeger): Bump this out into a helper header in this directory.
class EmbedLayerNormBase : public OpKernel {
 public:
  explicit EmbedLayerNormBase(const OpKernelInfo& op_kernel_info);

 protected:
  float epsilon() const;

 private:
  float epsilon_;
};

template <typename T>
class EmbedLayerNorm : public EmbedLayerNormBase {
 public:
  explicit EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
