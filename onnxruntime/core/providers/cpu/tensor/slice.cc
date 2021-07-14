// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"

#include <limits>
#include <unordered_map>

#include "core/framework/element_type_lists.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0,
    element_type_lists::All);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using DataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                 Slice, Input, 0);
using IndicesTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                    Slice, Input, 1);
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Slice, Input, 0);
using EnabledIndicesTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                           Slice, Input, 1);
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    1, 9,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<DataTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>()),
    Slice1);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    10, 10,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<DataTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<IndicesTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<DataTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<IndicesTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_KERNEL(
    Slice,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<DataTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<IndicesTypes>(), BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

// Check if it's possible to combine innermost dimensions so we copy larger blocks.
// Sets flattened_output_dims to nullptr if it is not.
// Updates starts and steps to match flattened_output_dims if it is.
// e.g. if input shape is { 2, 2, 2 }, output shape is { 1, 2, 2 }, and the 'steps' value for the last two dims is 1,
// we are keeping all the data of the inner most two dimensions so can combine those into dims of { 1, 4 }
static void FlattenOutputDims(const std::vector<int64_t>& input_dimensions,
                              const std::vector<int64_t>& output_dims,
                              std::vector<int64_t>& starts,
                              std::vector<int64_t>& ends,
                              std::vector<int64_t>& steps,
                              std::vector<int64_t>*& flattened_output_dims) {
  int num_to_combine = 0;
  for (int64_t i = static_cast<int64_t>(starts.size()) - 1; i >= 0; --i) {
    // if we're keeping all the data for the dimension and not reversing the direction we can potentially combine it
    if (steps[i] == 1 && input_dimensions[i] == output_dims[i])
      ++num_to_combine;
    else
      break;
  }

  if (num_to_combine > 1) {
    auto num_dims = output_dims.size() - num_to_combine + 1;
    *flattened_output_dims = output_dims;
    flattened_output_dims->resize(num_dims);

    int64_t dim_value = 1;
    for (size_t k = num_dims - 1, end = output_dims.size(); k < end; ++k) {
      dim_value *= output_dims[k];
    }

    flattened_output_dims->back() = dim_value;

    // the value of starts and steps for all the dims being combined are 0 and 1 respectively,
    // so we can just shrink via resize so the number of entries matches flattened_output_dims
    starts.resize(num_dims);
    steps.resize(num_dims);

    // update ends as well
    ends.resize(num_dims);
    ends.back() = dim_value;
  } else {
    flattened_output_dims = nullptr;
  }
}

// Slice V1-9 & DynamicSlice
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, compute_metadata));
  FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_output_dims_);
  return Status::OK();
}

// DynamicSlice & Slice V10
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    const std::vector<int64_t>& raw_steps,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));
  FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_output_dims_);

  return Status::OK();
}

// Slice V10 & DynamicSlice
Status SliceBase::FillVectorsFromInput(const Tensor& start_tensor,
                                       const Tensor& ends_tensor,
                                       const Tensor* axes_tensor,
                                       const Tensor* steps_tensor,
                                       std::vector<int64_t>& input_starts,
                                       std::vector<int64_t>& input_ends,
                                       std::vector<int64_t>& input_axes,
                                       std::vector<int64_t>& input_steps) {
  ORT_RETURN_IF_NOT(start_tensor.Shape().NumDimensions() == 1, "Starts must be a 1-D array");
  ORT_RETURN_IF_NOT(ends_tensor.Shape().NumDimensions() == 1, "Ends must be a 1-D array");
  ORT_RETURN_IF_NOT(start_tensor.Shape() == ends_tensor.Shape(), "Starts and ends shape mismatch");
  ORT_RETURN_IF_NOT(nullptr == axes_tensor || start_tensor.Shape() == axes_tensor->Shape(),
                    "Starts and axes shape mismatch");
  ORT_RETURN_IF_NOT(nullptr == steps_tensor || start_tensor.Shape() == steps_tensor->Shape(),
                    "Starts and steps shape mismatch");

  const auto& size = start_tensor.Shape().Size();
  input_starts.resize(size);
  input_ends.resize(size);
  if (nullptr != axes_tensor)
    input_axes.resize(size);
  // Slice V10
  if (nullptr != steps_tensor)
    input_steps.resize(size);

  // check for type reduction of supported indices types
  constexpr bool int32_enabled = utils::HasType<EnabledIndicesTypes, int32_t>();
  constexpr bool int64_enabled = utils::HasType<EnabledIndicesTypes, int64_t>();

  if (int32_enabled && start_tensor.IsDataType<int32_t>()) {
    std::copy(start_tensor.Data<int32_t>(), start_tensor.Data<int32_t>() + size, input_starts.begin());
    std::copy(ends_tensor.Data<int32_t>(), ends_tensor.Data<int32_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int32_t>(), axes_tensor->Data<int32_t>() + size, input_axes.begin());
    // Slice V10
    if (nullptr != steps_tensor)
      std::copy(steps_tensor->Data<int32_t>(), steps_tensor->Data<int32_t>() + size, input_steps.begin());
  }

  else if (int64_enabled && start_tensor.IsDataType<int64_t>()) {
    std::copy(start_tensor.Data<int64_t>(), start_tensor.Data<int64_t>() + size, input_starts.begin());
    std::copy(ends_tensor.Data<int64_t>(), ends_tensor.Data<int64_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int64_t>(), axes_tensor->Data<int64_t>() + size, input_axes.begin());
    // Slice V10
    if (nullptr != steps_tensor)
      std::copy(steps_tensor->Data<int64_t>(), steps_tensor->Data<int64_t>() + size, input_steps.begin());
  }

  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Data type for starts and ends inputs' is not supported in this build. Got ",
                           start_tensor.DataType());
  }

  return Status::OK();
}

template <typename T>
static Status SliceImpl(OpKernelContext* ctx,
                        const Tensor& input_tensor,
                        SliceOp::PrepareForComputeMetadata& compute_metadata) {
  TensorShape output_shape(compute_metadata.output_dims_);
  auto& output_tensor = *ctx->Output(0, output_shape);

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  // use MutableDataRaw as actual data type in tensor may not match as we templatize on data size
  T* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());

  // Bypass the slice logic when only one element is required (usually happens with shapes).
  if (compute_metadata.starts_.size() == 1 && output_tensor.Shape().Size() == 1 && compute_metadata.steps_[0] > 0 && !output_tensor.IsDataTypeString()) {
    auto dims = input_tensor.Shape().GetDims();
    size_t start = compute_metadata.starts_[0] >= 0 ? compute_metadata.starts_[0] : ((compute_metadata.starts_[0] + dims[0]) % dims[0]);
    const T* input = reinterpret_cast<const T*>(input_tensor.DataRaw()) + start;
    *output = *input;
    return Status::OK();
  }

  const auto* output_end = output + output_tensor.Shape().Size();

  auto create_output = [&output, &output_end](SliceIterator<T>& slice_input_iterator) {
    if (slice_input_iterator.SolitaryInnerStep()) {
      while (output < output_end) {
        output = slice_input_iterator.CopyInnermostAxisSolitaryInnerStep(output);
      }
    } else {
      while (output < output_end) {
        output = slice_input_iterator.CopyInnermostAxisNonSolitaryInnerStep(output);
      }
    }

    ORT_ENFORCE(output == output_end);
  };

  if (compute_metadata.p_flattened_output_dims_) {
    // if we have flattened output dims we need to also flatten the input dims.
    // as we're combining the innermost dims and keeping all values we can just copy the size of the last dim
    std::vector<int64_t> flattened_input_dims(input_tensor.Shape().GetDims());
    flattened_input_dims.resize(compute_metadata.p_flattened_output_dims_->size());
    flattened_input_dims.back() = compute_metadata.p_flattened_output_dims_->back();
    TensorShape input_shape(std::move(flattened_input_dims));

    auto input_iterator2 = SliceIterator<T>(input_tensor, input_shape, compute_metadata.starts_,
                                            *compute_metadata.p_flattened_output_dims_, compute_metadata.steps_);
    create_output(input_iterator2);
  } else {
    auto input_iterator2 = SliceIterator<T>(input_tensor, compute_metadata.starts_, compute_metadata.output_dims_,
                                            compute_metadata.steps_);
    create_output(input_iterator2);
  }

  return Status::OK();
}

template <typename EnabledTypes, typename T>
static inline bool CallSliceImplIfEnabled(OpKernelContext* ctx,
                                          const Tensor& input_tensor,
                                          SliceOp::PrepareForComputeMetadata& compute_metadata,
                                          Status& status) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledTypes, T>();
  if (enabled) {
    status = SliceImpl<T>(ctx, input_tensor, compute_metadata);
  }

  return enabled;
}

Status SliceBase::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  const auto& input_tensor = *input_tensor_ptr;
  const auto& input_dimensions = input_tensor.Shape().GetDims();

  if (input_dimensions.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");
  }

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);

  // Slice V10 & DynamicSlice
  if (dynamic_) {
    std::vector<int64_t> input_starts;
    std::vector<int64_t> input_ends;
    std::vector<int64_t> input_axes;
    std::vector<int64_t> input_steps;
    ORT_RETURN_IF_ERROR(FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2),
                                             ctx->Input<Tensor>(3), ctx->Input<Tensor>(4),
                                             input_starts, input_ends, input_axes, input_steps));

    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }
  // Slice V1-9
  else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, compute_metadata));
  }

  Status status = Status::OK();

  bool supported = false;
  if (input_tensor.IsDataTypeString()) {
    if (utils::HasType<EnabledDataTypes, std::string>()) {
      supported = true;
      status = SliceImpl<std::string>(ctx, input_tensor, compute_metadata);
    }
  } else {
    const auto element_size = input_tensor.DataType()->Size();
    // call SliceImpl
    switch (element_size) {
      case sizeof(uint32_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint32_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint64_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint64_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint16_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint16_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint8_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint8_t>(ctx, input_tensor, compute_metadata, status);
        break;
      default:
        // leave 'supported' as false
        break;
    }
  }

  if (!supported) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", input_tensor.DataType());
  }

  return status;
}

}  // namespace onnxruntime
