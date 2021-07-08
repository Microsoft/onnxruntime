// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/tensor/copy.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    4,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 13 .
ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// this method will be shared between 'Concat' (CPU and GPU) and
// 'ConcatFromSequence' ('concat' and 'stack' modes) to validate inputs
Status ConcatBase::PrepareForCompute(OpKernelContext* ctx,
                                     const std::vector<const Tensor*>& input_tensors,
                                     Prepare& p) const {
  int input_count = static_cast<int>(input_tensors.size());

  // Must have atleast one input to concat
  ORT_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");

  std::vector<int64_t> reference_dims;
  size_t reference_rank = 0;

  int reference_tensor_index = 0;

  std::vector<int64_t> input_tensor_sizes;
  input_tensor_sizes.reserve(input_count);

  bool all_inputs_are_empty = true;

  for (int index = 0; index < input_count; ++index) {
    const auto* input = input_tensors[index];
    ORT_ENFORCE(input != nullptr, "input count mismatch");

    // find the first tensor that isn't empty
    // to be used as a reference for all
    // downstream shape/rank validations of other inputs
    const auto& shape = input->Shape();
    const auto num_elements = shape.Size();
    if (num_elements > 0) {
      reference_dims = shape.GetDims();
      reference_rank = reference_dims.size();
      reference_tensor_index = index;
      input_tensor_sizes.push_back(num_elements);
      all_inputs_are_empty = false;
      break;
    } else {
      input_tensor_sizes.push_back(0);
    }
  }

  if (all_inputs_are_empty) {
    // Reference dim and reference rank can just come from the first input
    // No shape/rank validations will be done (as all inputs are empty).
    // But the rest of the execution flow (filling in the Prepare instance - p)
    // can use this info.
    reference_dims = input_tensors[0]->Shape().GetDims();
    reference_rank = reference_dims.size();
  }

  // Cannot concatenate scalars (but they can be stacked)
  if (!is_stack_)
    ORT_RETURN_IF_NOT(reference_rank > 0, "Cannot concatenate scalars");

  // Handle and fix negative axis
  // In 'stack' mode, the accepted range depends on the output rank (which is one more than the input rank)
  p.axis = static_cast<uint64_t>(HandleNegativeAxis(axis_, !is_stack_
                                                               ? reference_rank
                                                               : reference_rank + 1));

  // Ensure all of the non concatenated axes match each other
  for (int index = reference_tensor_index + 1; index < input_count; index++) {
    const auto* input = input_tensors[index];
    ORT_ENFORCE(input != nullptr, "input count mismatch");
    const auto& input_shape = input->Shape();
    const auto& input_dims = input_shape.GetDims();

    // Skip shape/rank validation for inputs that are empty.
    // The ONNX spec states that all dim values along axes not concatentated on
    // need to be the same for all inputs (empty inputs are not explicitly exempted).
    // The model in GH issue 8020 has a bunch of Loop nodes all feeding into
    // the 'Concat' node and one of these Loops tend to have an iteration
    // count of 0 for some inputs. If the iteration count for a Loop is zero,
    // we don't execute its subgraph (since the outputs are going to be empty anyway)
    // and we send an "empty" tensor(s) downstream and use ONNX shape inferred shape
    // to "compose" the shape for these empty tensor(s).
    // If we encounter symbolic dims in the ONNX shape inferred shape, we place a '0'
    // in that position and due to the "lossy" nature of this process, the inputs' shape
    // validation for such empty inputs fail and hence we skip these validations for all
    // empty inputs.
    // This isn't too bad as we will never use empty inputs while concatenating anyway.
    // We just loosen this check to unblock model in GH issue 8020 to complete processing.
    if (input_shape.Size() == 0) {
      input_tensor_sizes.push_back(0);
    } else {
      const size_t input_rank = input_dims.size();

      ORT_ENFORCE(input_rank == reference_rank,
                  "Ranks of input data are different, cannot concatenate them. expected rank: ",
                  reference_rank, " got: ", input_rank);

      // Ensure all the other (non-concat) axes match
      int64_t tensor_size = 1;
      for (size_t axis_index = 0; axis_index < reference_rank; ++axis_index) {
        auto dim_value = input_dims[axis_index];
        tensor_size *= dim_value;

        // In 'concat' mode, the axis to be concatenated may be different
        // But in 'stack' mode, all input shapes must be the same and must be validated
        if (!is_stack_ && axis_index == p.axis)
          continue;

        ORT_RETURN_IF_NOT(dim_value == reference_dims[axis_index],
                          "Non concat axis dimensions must match: Axis ",
                          axis_index, " has mismatched dimensions of ", dim_value,
                          " and ", reference_dims[axis_index]);
      }

      input_tensor_sizes.push_back(tensor_size);  //assign the computed size of the input tensor
    }
  }

  // Calculate the shape of the output tensor
  std::vector<int64_t> output_dims = reference_dims;

  if (!is_stack_) {  // 'Concat' mode
    // While concatenating, the rank of the output is the same as the input rank(s)

    // Calculate the size of the concatenated axis
    size_t concat_axis_size = 0;
    for (int64_t index = 0; index < input_count; index++) {
      concat_axis_size += input_tensors[index]->Shape()[static_cast<int>(p.axis)];
    }

    output_dims[p.axis] = concat_axis_size;
  } else {  // 'Stack' mode
    // While stacking, the rank of the output is one more than the input rank(s).
    // Stacking may be thought of as adding an unit dimension (of value 1) in the input tensors,
    // and concatenating them on thie new axis.
    // The value in the corresponding axis of the output will be the number of inputs that are being stacked.
    output_dims.insert(output_dims.begin() + p.axis, static_cast<int64_t>(input_count));
  }

  TensorShape output_shape(output_dims);

  // Create output tensor
  p.output_tensor = &(*ctx->Output(0, output_shape));

  // Make note if output tensor is going to be empty
  p.output_num_elements = output_shape.Size();

  // No need to proceed further if output is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // The output_axis_pitch is the number of elements to add to move to the next split axis in the output.
  // Can handle stacking as well.
  p.output_axis_pitch = 1;
  auto output_rank = !is_stack_ ? reference_rank : reference_rank + 1;
  for (size_t i = output_rank; i-- > p.axis;) {
    p.output_axis_pitch *= output_dims[i];
  }

  // Fill the 'Prepare' struct with available information
  p.inputs.reserve(input_count);
  for (int input_index = 0; input_index < input_count; input_index++) {
    const Tensor* data_n_ptr = input_tensors[input_index];
    auto& data_n = *data_n_ptr;

    // Type sanity check (Make sure we are working on homogeneous types)
    ORT_RETURN_IF_NOT(data_n.DataType() == p.output_tensor->DataType(), "Data type mismatch");

    // The input_axis_pitch is the number of elements to add to move to the next split axis in the input
    // Can handle stacking as well (as the "new dummy dimension" in the input is of unit value).
    // TODO: Minor Optimization possibility: This input_axis_patch will be common across all inputs
    // in 'ConcatFromSequence' (stack mode). They have to be computed for each input only while concatenating.
    int64_t input_axis_pitch = 1;
    const auto& data_dims = data_n.Shape().GetDims();
    for (size_t i = reference_rank; i-- > p.axis;) {
      input_axis_pitch *= data_dims[i];
    }

    p.inputs.push_back({&data_n, input_axis_pitch, input_tensor_sizes[input_index]});
  }

  // Make note if the input Tensors of type 'string'
  p.is_string_type = p.inputs[0].tensor->IsDataTypeString();

  return Status::OK();
}

// This method computes the output tensor for Concat/ConcatFromSequence ops
Status ConcatBase::ComputeImpl(Prepare& p, OpKernelContext* ctx) const {
  int input_count = static_cast<int>(p.inputs.size());
  int64_t initial_output_offset = 0;  // initial offset for each input

  auto output_strides_full = StridesForTensor(*p.output_tensor);
  auto num_dims = p.output_tensor->Shape().NumDimensions();
  // if we are stacking, skip the dimension that will be stacked along in the output strides
  // (the striding for that dimension is handled by the initial_output_offset)
  // Note that output_strides_full is only used later when is_stack_ is true, so it's safe to move
  auto output_strides_for_copy = is_stack_ ? std::vector<int64_t>(num_dims - 1) : std::move(output_strides_full);
  if (is_stack_) {
    for (int64_t i = 0; i < static_cast<int64_t>(num_dims - 1); i++) {
      auto read_i = (i >= axis_) ? i + 1 : i;
      output_strides_for_copy[i] = output_strides_full[read_i];
    }
  }

  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];

    // no data in this tensor - so skip it
    if (prep.num_elements == 0)
      continue;

    // parallel copy the data across
    auto status = DispatchStridedCopy(ctx->GetOperatorThreadPool(),
                                      *p.output_tensor,
                                      initial_output_offset,
                                      output_strides_for_copy,
                                      prep.tensor->Shape(),
                                      *prep.tensor,
                                      StridesForTensor(*prep.tensor));
    ORT_RETURN_IF_ERROR(status);

    // advance along the axis that we are concatenating on (by the size of the axis of the tensor that we just copied)
    if (is_stack_) {
      initial_output_offset += output_strides_full[p.axis];
    } else {
      initial_output_offset += prep.tensor->Shape()[p.axis] * output_strides_for_copy[p.axis];
    }
  }

  return Status::OK();
}

// core Compute() method for the 'Concat' kernel
Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // Compute values to be placed in the output tensor
  return ComputeImpl(p, ctx);
}

}  // namespace onnxruntime
