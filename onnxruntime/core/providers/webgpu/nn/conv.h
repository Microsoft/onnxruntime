// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Layout of the kernel tensor that ComputeInternal consumes. `OIHW` is the layout the
// Conv operator is defined with; the others are produced by PrePackInternal and are each
// understood by exactly one consumer, so the layout has to be tracked explicitly rather
// than inferred from which prepacked tensor happens to be present.
enum class KernelLayout {
  OIHW,  // No prepacked tensor -- the kernel is read straight from input 1.
  HWIO,  // Consumed by grouped conv, the 1x1/same_size MatMul path and Conv2dMM.
  OHWI,  // Consumed by the Im2ColMatMul path only.
};

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
  // Abstract base for alternative optimized Conv implementations (mirrors
  // Gemm::GemmOptImpl); currently just the subgroup-matrix path.
  class ConvOptImpl {
   public:
    explicit ConvOptImpl(const Conv& parent) : parent_(parent) {}
    virtual ~ConvOptImpl() = default;

    // Sets handled=true if it ran, else leaves handled=false (allocating nothing).
    // Runs before ComputeInternal resolves auto_pad/output.
    virtual Status Compute(ComputeContext& context, /*out*/ bool& handled) = 0;

   protected:
    const Conv& parent_;
  };

  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
    if (is_fused) {
      // Throw the Status, not the expression: the message names the activation ORT_ENFORCE would drop.
      ORT_THROW_IF_ERROR(GetFusedActivationAttr(info, activation_));
    }
    const Tensor* w = nullptr;
    w_is_constant_ = info.TryGetConstantInput(1, &w);
  }
  Status ComputeInternal(ComputeContext& context) const override;

  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

  const ConvAttributes& ConvAttrs() const { return conv_attrs_; }
  const Activation& ConvActivation() const { return activation_; }
  bool IsWeightConstant() const { return w_is_constant_; }
  // Prepacked weight, or null if not prepacked (layout given by kernel_layout_).
  const Tensor* PrepackedKernel() const { return prepacked_kernel_.get(); }

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
  // Set by PrePackInternal; null when the kernel could not be prepacked (e.g. the weight
  // is not a constant initializer), in which case ComputeInternal reads input 1 instead.
  std::unique_ptr<Tensor> prepacked_kernel_;
  // Layout of the tensor ComputeInternal ends up consuming -- `prepacked_kernel_` when it
  // is set, otherwise input 1. Stays `OIHW` while `prepacked_kernel_` is null.
  KernelLayout kernel_layout_{KernelLayout::OIHW};
  mutable MatMulComputeDispatcher matmul_compute_dispatcher_;

  bool w_is_constant_ = false;  // whether the weight input (index 1) is a constant initializer

 private:
  // Owns the optimized implementation, built lazily (needs a compute context) and
  // guarded by call_once against concurrent Compute calls. Null after init means no
  // optimized path on this device.
  class ConvOptImplCache {
   public:
    ConvOptImplCache() = default;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvOptImplCache);

    // Defined in conv.cc: it calls into subgroup_matrix_conv.h, which includes this
    // header, so the include can't go the other way.
    ConvOptImpl* GetOrCreate(const Conv& parent, const ComputeContextBase& context);

   private:
    std::once_flag init_flag_;
    std::unique_ptr<ConvOptImpl> impl_;
  };
  mutable ConvOptImplCache opt_impl_cache_;
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
