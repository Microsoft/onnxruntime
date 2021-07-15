// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    ConcatTraining,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    ConcatTraining);

// core Compute() method for the 'Concat' kernel
Status ConcatTraining::Compute(OpKernelContext* ctx) const {
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

  // Create optional output tensor for 'per_input_length'
  Tensor* per_input_length_tensor = ctx->Output(1, {input_count});
  if (per_input_length_tensor) {
    int64_t* per_input_length = per_input_length_tensor->template MutableData<int64_t>();
    for (int i = 0; i < input_count; ++i) {
      per_input_length[i] = input_tensors[i]->Shape()[p.axis];
    }
  }
  // Compute values to be placed in the output tensor
  return ComputeImpl(p, ctx);
}

}  // namespace contrib
}  // namespace onnxruntime
