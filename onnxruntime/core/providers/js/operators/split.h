// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/split.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Split : public JsKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info, uint32_t opset) : JsKernel(info), SplitBase(info, opset) {
    std::vector<int32_t> split_sizes;
    if (split_sizes_.size() > 0) {
      ORT_ENFORCE(split_sizes_.size() == info.node().OutputDefs().size(),
                  "Number of outputs (", info.node().OutputDefs().size(), ") does not match split_sizes (",
                  split_sizes_.size(), ")");
      split_sizes.resize(split_sizes_.size());
      for (size_t i = 0; i < split_sizes_.size(); ++i) {
        split_sizes[i] = gsl::narrow_cast<int32_t>(split_sizes_[i]);
      }
      if (num_outputs_ < 0) {
        num_outputs_ = split_sizes.size();
      }
    } else if (split_sizes_.size() == 0 && info.GetInputCount() < 2) {
      // Compute split_sizes from input shape and num_outputs.
      // Only do this if shapes are static (dim_value() returns 0 for dynamic dims)
      auto* input_shape = info.node().InputDefs()[0]->Shape();
      bool has_static_shapes = input_shape != nullptr &&
                               input_shape->dim(gsl::narrow_cast<int32_t>(axis_)).has_dim_value();

      if (has_static_shapes) {
        auto total_split_size = input_shape->dim(gsl::narrow_cast<int32_t>(axis_)).dim_value();
        int64_t split_size_sum = 0;
        if (num_outputs_ < 0) {
          num_outputs_ = info.node().OutputDefs().size();
        } else {
          ORT_ENFORCE(num_outputs_ == info.node().OutputDefs().size(),
                      "Number of outputs (", info.node().OutputDefs().size(), ") does not match num_outputs (",
                      num_outputs_, ")");
        }

        bool all_outputs_static = true;
        for (auto output : info.node().OutputDefs()) {
          auto* out_shape = output->Shape();
          if (out_shape == nullptr ||
              !out_shape->dim(gsl::narrow_cast<int32_t>(axis_)).has_dim_value()) {
            all_outputs_static = false;
            break;
          }
          auto split_size = out_shape->dim(gsl::narrow_cast<int32_t>(axis_)).dim_value();
          split_sizes.push_back(gsl::narrow_cast<int32_t>(split_size));
          split_size_sum += split_size;
        }

        if (!all_outputs_static) {
          // Dynamic output shapes - let javascript handle it at runtime
          split_sizes.clear();
        } else {
          ORT_ENFORCE(split_size_sum == total_split_size,
                      "Sum of split sizes (", split_size_sum, ") does not match input size (", total_split_size, ")");
        }
      }
      // else: shapes are dynamic, let javascript handle it at runtime
    }
    // else: let javascript handle all other cases, ie. split_sizes come as input[1]

    JSEP_INIT_KERNEL_ATTRIBUTE(Split, ({"axis" : $1,
                                        "numOutputs" : $2,
                                        "splitSizes" : $3 ? Array.from(HEAP32.subarray(Number($3), Number($4))) : []}),
                               static_cast<int32_t>(axis_),
                               static_cast<int32_t>(num_outputs_),
                               JSEP_HEAP32_INDEX_START(split_sizes),
                               JSEP_HEAP32_INDEX_END(split_sizes));
  }
};

class Split_1 final : public Split {
 public:
  Split_1(const OpKernelInfo& info) : Split(info, 1) {}
};

class Split_2_10 final : public Split {
 public:
  Split_2_10(const OpKernelInfo& info) : Split(info, 2) {}
};

class Split_11_12 final : public Split {
 public:
  Split_11_12(const OpKernelInfo& info) : Split(info, 11) {}
};

class Split_13_17 final : public Split {
 public:
  Split_13_17(const OpKernelInfo& info) : Split(info, 13) {}
};

class Split_18 final : public Split {
 public:
  Split_18(const OpKernelInfo& info) : Split(info, 18) {}
};

}  // namespace js
}  // namespace onnxruntime
