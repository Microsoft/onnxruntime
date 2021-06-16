// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#endif

namespace onnxruntime {

namespace SliceOp {
struct PrepareForComputeMetadata {
  PrepareForComputeMetadata() = delete;
  PrepareForComputeMetadata(const std::vector<int64_t>& input_dimensions)
      : input_dimensions_(input_dimensions) {
    size_t dimension_count = input_dimensions.size();
    starts_.resize(dimension_count, 0);
    steps_.resize(dimension_count, 1);
    output_dims_ = input_dimensions;
  }

  const std::vector<int64_t>& input_dimensions_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
  std::vector<int64_t> flattened_output_dims_;
  std::vector<int64_t>* p_flattened_output_dims_ = &flattened_output_dims_;
};
}  // namespace SliceOp

class SliceBase {
  // static methods that can be used from other ops if needed
 public:
  // compute output_dims without steps (Slice V1-9 & DynamicSlice)
  static Status PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                  const std::vector<int64_t>& raw_ends,
                                  const std::vector<int64_t>& raw_axes,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  // compute output_dims with steps (Slice V10)
  static Status PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                  const std::vector<int64_t>& raw_ends,
                                  const std::vector<int64_t>& raw_axes,
                                  const std::vector<int64_t>& raw_steps,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  // Slice V10 & DynamicSlice
  static Status FillVectorsFromInput(const Tensor& start_tensor,
                                     const Tensor& ends_tensor,
                                     const Tensor* axes_tensor,
                                     const Tensor* steps_tensor,
                                     std::vector<int64_t>& input_starts,
                                     std::vector<int64_t>& input_ends,
                                     std::vector<int64_t>& input_axes,
                                     std::vector<int64_t>& input_steps);

 protected:
  SliceBase(const OpKernelInfo& info, bool dynamic = false)
      : dynamic_(dynamic) {
    if (!dynamic) {
      auto has_starts = info.GetAttrs("starts", attr_starts_).IsOK();
      auto has_ends = info.GetAttrs("ends", attr_ends_).IsOK();
      auto has_axes = info.GetAttrs("axes", attr_axes_).IsOK();
      ORT_ENFORCE(has_starts && has_ends && attr_starts_.size() == attr_ends_.size(),
                  "Missing or invalid starts and ends attribute");
      ORT_ENFORCE(!has_axes || attr_axes_.size() == attr_starts_.size(),
                  "Invalid axes attribute, axes attribute (if present) should have the same size as starts/ends attributes");
    }
  }

  Status Compute(OpKernelContext* context) const;
  Status ComputeParallel(OpKernelContext* context,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& axes) const;

 protected:
  const std::vector<int64_t>& StartsAttribute() const { return attr_starts_; }
  const std::vector<int64_t>& EndsAttribute() const { return attr_ends_; }
  const std::vector<int64_t>& AxesAttribute() const { return attr_axes_; }

 private:
  bool dynamic_;
  std::vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

struct Slice1 final : public OpKernel, public SliceBase {
  Slice1(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, false) {}
  Status Compute(OpKernelContext* context) const override { return SliceBase::Compute(context); }
};

struct Slice10 final : public OpKernel, public SliceBase {
  Slice10(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, true) {}
  Status Compute(OpKernelContext* context) const override { return SliceBase::Compute(context); }
};

}  // namespace onnxruntime
