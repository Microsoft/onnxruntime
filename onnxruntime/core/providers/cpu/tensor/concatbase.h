// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {

// structure to hold some inputs and some metadata to be used during Compute()
struct Prepare {
  struct InputInfo {
    const Tensor* tensor;
    int64_t axis_pitch;
    int64_t num_elements;
  };
  std::vector<InputInfo> inputs;
  int64_t output_num_elements;
  int64_t output_axis_pitch;
  Tensor* output_tensor;
  uint64_t axis;
  bool is_string_type;
};

class ConcatBase {
 public:
  // the core method that will be invoked by the 'Concat' (CPU and GPU)
  // and 'ConcatFromSequence' kernels
  Status PrepareForCompute(OpKernelContext* ctx, const std::vector<const Tensor*>& input_tensors,
                           Prepare& p) const;

 protected:
  ConcatBase(const OpKernelInfo& info, bool is_sequence_op = false) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      ORT_ENFORCE(false, "Must have valid 'axis' attribute");
    }

    is_sequence_op_ = is_sequence_op;

    if (is_sequence_op) {  // Only ConcatFromSequence supports stacking
      is_stack_ = info.GetAttrOrDefault<int64_t>("new_axis", 0) == 0 ? false : true;
    }
  }
  Status ComputeImpl(Prepare& p, OpKernelContext* ctx) const;

  int64_t axis_;
  bool is_stack_ = false;
  bool is_sequence_op_;
};

}  // namespace onnxruntime
