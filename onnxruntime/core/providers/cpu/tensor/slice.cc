// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"

#include <limits>
#include <unordered_map>

#include "core/framework/element_type_lists.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"

using namespace ::onnxruntime::common;
using namespace std;

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

const auto data_type_constraints = BuildKernelDefConstraintsFromTypeList<DataTypes>();
const auto indices_type_constraints = BuildKernelDefConstraintsFromTypeList<IndicesTypes>();
const auto enabled_data_type_constraints = BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>();
const auto enabled_indices_type_constraints = BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>();

// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    1, 9,
    KernelDefBuilder().TypeConstraint("T", data_type_constraints, enabled_data_type_constraints),
    Slice1);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    10, 10,
    KernelDefBuilder()
        .TypeConstraint("T", data_type_constraints, enabled_data_type_constraints)
        .TypeConstraint("Tind", indices_type_constraints, enabled_indices_type_constraints),
    Slice10);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", data_type_constraints, enabled_data_type_constraints)
        .TypeConstraint("Tind", indices_type_constraints, enabled_indices_type_constraints),
    Slice10);

ONNX_CPU_OPERATOR_KERNEL(
    Slice,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", data_type_constraints, enabled_data_type_constraints)
        .TypeConstraint("Tind", indices_type_constraints, enabled_indices_type_constraints),
    Slice10);

// Check if it's possible to combine innermost dimensions so we copy larger blocks.
// Sets flattened_output_dims to nullptr if it is not.
// Updates starts and steps to match flattened_output_dims if it is.
// e.g. if input shape is { 2, 2, 2 }, output shape is { 1, 2, 2 }, and the 'steps' value for the last two dims is 1,
// we are keeping all the data of the inner most two dimensions so can combine those into dims of { 1, 4 }
static void FlattenOutputDims(const std::vector<int64_t>& input_dimensions,
                              const std::vector<int64_t>& output_dims,
                              std::vector<int64_t>& starts,
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
  } else {
    flattened_output_dims = nullptr;
  }
}

// Slice V1-9 & DynamicSlice
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);
  if (axes.empty()) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(compute_metadata.starts_.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    auto axis = HandleNegativeAxis(axes[axis_index], dimension_count);  // handle negative and enforce axis is valid
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += compute_metadata.input_dimensions_[axis];
    compute_metadata.starts_[axis] = clamp(start, int64_t{0}, compute_metadata.input_dimensions_[axis]);

    // process end
    auto end = raw_ends[axis_index];
    if (end < 0)
      end += compute_metadata.input_dimensions_[axis];

    // find output dim value for this axis
    auto temp = clamp(end, int64_t{0}, compute_metadata.input_dimensions_[axis]) - compute_metadata.starts_[axis];
    if (temp < 0)
      compute_metadata.output_dims_[axis] = 0;
    else
      compute_metadata.output_dims_[axis] = temp;
  }

  FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.steps_, compute_metadata.p_flattened_output_dims_);

  return Status::OK();
}

// DynamicSlice & Slice V10
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    const std::vector<int64_t>& raw_steps,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);

  if (axes.empty()) {
    // axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(compute_metadata.starts_.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end/steps ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(dimension_count) : axes[axis_index];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);

    // process step
    auto step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1;
    if (step == 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'step' value cannot be 0");
    compute_metadata.steps_[axis] = step;

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += compute_metadata.input_dimensions_[axis];
    if (step < 0)
      compute_metadata.starts_[axis] = clamp(start, int64_t{0}, compute_metadata.input_dimensions_[axis] - 1);
    else
      compute_metadata.starts_[axis] = clamp(start, int64_t{0}, compute_metadata.input_dimensions_[axis]);

    // process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : compute_metadata.input_dimensions_[axis];
    }

    else {
      if (end < 0)
        end += compute_metadata.input_dimensions_[axis];
      if (step < 0)
        end = clamp(end, int64_t{-1}, compute_metadata.input_dimensions_[axis]);
      else
        end = clamp(end, int64_t{0}, compute_metadata.input_dimensions_[axis]);
    }

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - compute_metadata.starts_[axis]) / step));
    if (temp < 0)
      compute_metadata.output_dims_[axis] = 0;
    else
      compute_metadata.output_dims_[axis] = temp;
  }

  FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                    compute_metadata.steps_, compute_metadata.p_flattened_output_dims_);

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
                        SliceOp::PrepareForComputeMetadata& compute_metadata,
                        T* output = nullptr,
                        T* output_end = nullptr) {
  if (output == nullptr) {
    TensorShape output_shape(compute_metadata.output_dims_);
    auto& output_tensor = *ctx->Output(0, output_shape);

    // output tensor's size is 0, nothing to fill - return
    if (output_shape.Size() == 0)
      return Status::OK();

    // use MutableDataRaw as actual data type in tensor may not match as we templatize on data size
    output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
    output_end = output + output_tensor.Shape().Size();
  }

  // Bypass the slice logic when only one element is required (usually happens with shapes).
  if (compute_metadata.starts_.size() == 1 && std::distance(output, output_end) == 1 && compute_metadata.steps_[0] > 0) {
    auto dims = input_tensor.Shape().GetDims();
    size_t start = compute_metadata.starts_[0] >= 0 ? compute_metadata.starts_[0] : ((compute_metadata.starts_[0] + dims[0]) % dims[0]);
    const T* input = reinterpret_cast<const T*>(input_tensor.DataRaw()) + start;
    *output = *input;
    return Status::OK();
  }

  auto create_output = [&output, &output_end](SliceIterator<T>& input_iterator) {
    if (input_iterator.SolitaryInnerStep()) {
      while (output < output_end) {
        output = input_iterator.CopyInnermostAxisSolitaryInnerStep(output);
      }
    } else {
      while (output < output_end) {
        output = input_iterator.CopyInnermostAxisNonSolitaryInnerStep(output);
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

    auto input_iterator = SliceIterator<T>(input_tensor, input_shape, compute_metadata.starts_,
                                           *compute_metadata.p_flattened_output_dims_, compute_metadata.steps_);
    create_output(input_iterator);
  } else {
    auto input_iterator = SliceIterator<T>(input_tensor, compute_metadata.starts_, compute_metadata.output_dims_,
                                           compute_metadata.steps_);
    create_output(input_iterator);
  }

  return Status::OK();
}

template <typename EnabledTypes, typename T>
static inline bool CallSliceImplIfEnabled(OpKernelContext* ctx,
                                          const Tensor& input_tensor,
                                          SliceOp::PrepareForComputeMetadata& compute_metadata,
                                          Status& status,
                                          T* output = nullptr,
                                          T* output_end = nullptr) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledTypes, T>();
  if (enabled) {
    status = SliceImpl<T>(ctx, input_tensor, compute_metadata, output, output_end);
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
    if (input_starts.size() > 1 && input_steps.size() == 0 && ctx->GetOperatorThreadPool() != nullptr) {
      auto status = SliceBase::ComputeParallel(ctx, input_starts, input_ends, input_axes);
      if (status.IsOK() || status.Code() != StatusCode::NOT_IMPLEMENTED) {
        return Status::OK();
      }
    }
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }
  // Slice V1-9
  else {
    if (attr_axes_.size() > 1 && ctx->GetOperatorThreadPool() != nullptr) {
      auto status = SliceBase::ComputeParallel(ctx, attr_starts_, attr_ends_, attr_axes_);
      if (status.IsOK() || status.Code() != StatusCode::NOT_IMPLEMENTED) {
        return Status::OK();
      }
    }
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

Status SliceBase::ComputeParallel(OpKernelContext* ctx,
                                  const std::vector<int64_t>& i_starts,
                                  const std::vector<int64_t>& i_ends,
                                  const std::vector<int64_t>& i_axes) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  const auto& input_tensor = *input_tensor_ptr;
  const auto& input_dimensions = input_tensor.Shape().GetDims();

  SliceOp::PrepareForComputeMetadata main_compute_metadata(input_dimensions);
  ORT_RETURN_IF_ERROR(PrepareForCompute(i_starts, i_ends, i_axes, main_compute_metadata));

  auto tp = ctx->GetOperatorThreadPool();
  int n_threads = concurrency::ThreadPool::DegreeOfParallelism(tp) * 4;
  if (n_threads > main_compute_metadata.output_dims_[0]) {
    return Status(StatusCategory::NONE, StatusCode::NOT_IMPLEMENTED);
  }

  TensorShape output_shape(main_compute_metadata.output_dims_);

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  int64_t main_stride = 1;
  for (size_t i = 1; i < main_compute_metadata.output_dims_.size(); ++i) {
    main_stride *= main_compute_metadata.output_dims_[i];
  }

  std::vector<SliceOp::PrepareForComputeMetadata> compute_metadata;
  compute_metadata.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    compute_metadata.emplace_back(input_dimensions);
  }

  std::vector<int64_t> starts(i_starts);
  std::vector<int64_t> ends(i_ends);
  std::vector<int64_t> splits(n_threads + 1);
  int64_t i_axes0 = i_axes.size() > 0 ? i_axes[0] : 0;
  int64_t idx = i_starts[0];
  int64_t end = std::min(
      input_dimensions[i_axes0],
      i_ends[0] == -1 ? input_dimensions[i_axes0] : i_ends[0]);
  for (int i = 0; i < n_threads; ++i) {
    splits[i] = idx;
    idx = i_starts[0] + (end - i_starts[0]) * (i + 1) / n_threads;
    if (i > 0 && splits[i - 1] == splits[i])
      return Status(StatusCategory::NONE, StatusCode::NOT_IMPLEMENTED);
  }
  splits[n_threads] = end;

  auto& output_tensor = *ctx->Output(0, output_shape);
  void* output = output_tensor.MutableDataRaw();

  for (int i = 0; i < n_threads; ++i) {
    starts[0] = splits[i];
    ends[0] = splits[i + 1];
    ORT_RETURN_IF_ERROR(PrepareForCompute(starts, ends, i_axes, compute_metadata[i]));
  }
  for (int i = 1; i < n_threads + 1; ++i) {
    splits[i] -= splits[0];
  }
  splits[0] = 0;
  std::vector<Status> statuses(n_threads, Status::OK());

  auto fct = [&](ptrdiff_t begin, ptrdiff_t end) {
    for (ptrdiff_t ith = begin; ith < end; ++ith) {
      bool supported = false;
      if (input_tensor.IsDataTypeString()) {
        if (utils::HasType<EnabledDataTypes, std::string>()) {
          supported = true;
          statuses[ith] = SliceImpl<std::string>(
              ctx, input_tensor, compute_metadata[ith],
              reinterpret_cast<std::string*>(output) + splits[ith] * main_stride,
              reinterpret_cast<std::string*>(output) + splits[ith + 1] * main_stride);
        }
      } else {
        const auto element_size = input_tensor.DataType()->Size();
        // call SliceImpl
        switch (element_size) {
          case sizeof(uint32_t):
            supported = CallSliceImplIfEnabled<EnabledDataTypes, uint32_t>(
                ctx, input_tensor, compute_metadata[ith], statuses[ith],
                reinterpret_cast<uint32_t*>(output) + splits[ith] * main_stride,
                reinterpret_cast<uint32_t*>(output) + splits[ith + 1] * main_stride);
            break;
          case sizeof(uint64_t):
            supported = CallSliceImplIfEnabled<EnabledDataTypes, uint64_t>(
                ctx, input_tensor, compute_metadata[ith], statuses[ith],
                reinterpret_cast<uint64_t*>(output) + splits[ith] * main_stride,
                reinterpret_cast<uint64_t*>(output) + splits[ith + 1] * main_stride);
            break;
          case sizeof(uint16_t):
            supported = CallSliceImplIfEnabled<EnabledDataTypes, uint16_t>(
                ctx, input_tensor, compute_metadata[ith], statuses[ith],
                reinterpret_cast<uint16_t*>(output) + splits[ith] * main_stride,
                reinterpret_cast<uint16_t*>(output) + splits[ith + 1] * main_stride);
            break;
          case sizeof(uint8_t):
            supported = CallSliceImplIfEnabled<EnabledDataTypes, uint8_t>(
                ctx, input_tensor, compute_metadata[ith], statuses[ith],
                reinterpret_cast<uint8_t*>(output) + splits[ith] * main_stride,
                reinterpret_cast<uint8_t*>(output) + splits[ith + 1] * main_stride);
            break;
          default:
            // leave 'supported' as false
            break;
        }
      }

      if (!supported) {
        statuses[idx] = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", input_tensor.DataType());
      }
    }
  };

  auto slice_size = output_shape.Size() / n_threads;
  auto cost = TensorOpCost{(double)slice_size, (double)slice_size, (double)slice_size * input_dimensions.size() * 2};
  concurrency::ThreadPool::TryParallelFor(tp, n_threads, cost, fct);

  for (auto status : statuses) {
    if (!status.IsOK())
      return status;
  }

  return Status::OK();
}

}  // namespace onnxruntime
