// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optional_ops.h"
#include "core/framework/ml_value.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Optional, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                            // We may be able to re-use the input for the output as is unless the output
                            // is a graph output. We provide this hint to the allocation planner
                            // to make the re-use call.
                            .Alias(0, 0),
                        Optional);

ONNX_OPERATOR_KERNEL_EX(OptionalHasElement, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
                        OptionalHasElement);

ONNX_OPERATOR_KERNEL_EX(OptionalGetElement, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                            .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            // We may be able to re-use the input for the output as is unless the output
                            // is a graph output. We provide this hint to the allocation planner
                            // to make the re-use call.
                            .Alias(0, 0),
                        OptionalGetElement);

static void CloneTensor(AllocatorPtr alloc, const Tensor& tensor_to_be_cloned, OrtValue& clone) {
  auto tensor = std::make_unique<Tensor>(tensor_to_be_cloned.DataType(),
                                         TensorShape(tensor_to_be_cloned.Shape()), alloc);
  CopyCpuTensor(&tensor_to_be_cloned, tensor.get());

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  clone.Init(tensor.release(),
             ml_tensor,
             ml_tensor->GetDeleteFunc());
}

static void CopySequenceTensor(AllocatorPtr alloc,
                               const TensorSeq& input,
                               TensorSeq& output) {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(input.Size());

  auto in_tensor = input.begin();
  for (; in_tensor != input.end(); ++in_tensor) {
    Tensor tmp(in_tensor->DataType(), onnxruntime::TensorShape(in_tensor->Shape()), alloc);
    CopyCpuTensor(&*in_tensor, &tmp);
    output_tensors.push_back(std::move(tmp));
  }

  output.SetElements(std::move(output_tensors));
}

static void CloneSequnceTensor(AllocatorPtr alloc,
                               const TensorSeq& sequence_tensor_to_be_cloned,
                               OrtValue& clone) {
  auto output_sequence_tensor =
      std::make_unique<TensorSeq>(sequence_tensor_to_be_cloned.DataType());

  CopySequenceTensor(alloc, sequence_tensor_to_be_cloned, *output_sequence_tensor);

  auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
  clone.Init(output_sequence_tensor.release(),
             ml_tensor_sequence,
             ml_tensor_sequence->GetDeleteFunc());

  return;
}

static bool CheckValidTypeProto(const onnx::TypeProto& tp) {
  // Optional types can currently be Tensors or SequenceTensors only
  // Ensure that the TypeProto holds those types

  // TODO: Should we also ensure that element type for tensor is set ?
  return (tp.has_tensor_type()) ||
         (tp.has_sequence_type() && tp.sequence_type().elem_type().has_tensor_type());
}

Status Optional::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);
  auto* output_ort_value = ctx->GetOutputOrtValue(0);

  // The static allocation planner has deemed that the input can be re-used as the output
  // Analogy: Checking if data pointers for the input and output Tensors are the same
  // before proceeding to make the copy.
  if (output_ort_value->IsAllocated()) {
    return Status::OK();
  }

  if (input_ort_value != nullptr) {  // An input was provided by the user

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    if (input_ort_value->IsTensor()) {
      CloneTensor(alloc,
                  input_ort_value->Get<Tensor>(), *output_ort_value);
    } else if (input_ort_value->IsTensorSequence()) {
      CloneSequnceTensor(alloc,
                         input_ort_value->Get<TensorSeq>(), *output_ort_value);
    } else {
      // Will not reach here
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Tensors and Sequence Tensors can be used to construct an Optional type");
    }

  } else {  // No input was provided - we use the type proto to construct the output OrtValue

    CheckValidTypeProto(*type_proto_);

    // type is either Tensor or TensorSeq (we have validated this already in CheckValidTypeProto())
    auto type = type_proto_->has_tensor_type() ? DataTypeImpl::GetType<Tensor>() : DataTypeImpl::GetType<TensorSeq>();

    output_ort_value->Init(nullptr,  // This OrtValue is "None"
                           type,
                           type->GetDeleteFunc());
  }

  return Status::OK();
}

Status OptionalHasElement::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  // Output is a scalar
  auto* output_tensor = ctx->Output(0, {});
  output_tensor->MutableData<bool>()[0] = input_ort_value->HasElement();

  return Status::OK();
}

Status OptionalGetElement::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  ORT_ENFORCE(input_ort_value->HasElement(),
              "Trying to use OptionalGetElement on an optional type OrtValue which is None");

  if (input_ort_value->IsTensor()) {
    const auto* input_tensor = &input_ort_value->Get<Tensor>();
    auto* output_tensor = ctx->Output(0, input_tensor->Shape());

    // If the allocation planner had deemed that we re-use the input OrtValue
    // as the output OrtValue, the data pointers in the input_tensor and the
    // output_tensor will be the same and the copy is a no-op.
    // CopyCpuTensor() already has such copy optimizations - so
    // just re-ue it.
    CopyCpuTensor(input_tensor, output_tensor);

  } else if (input_ort_value->IsTensorSequence()) {
    const auto* input_tensor_sequence = &input_ort_value->Get<TensorSeq>();
    auto* output_tensor_sequence = ctx->Output<TensorSeq>(0);

    // The static allocation planner has deemed that the input can be re-used as the output
    // Analogy: Checking if data pointers for the input and output Tensors are the same
    // before proceeding to make the copy.
    if (input_tensor_sequence == output_tensor_sequence) {
      return Status::OK();
    }

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    output_tensor_sequence->SetType(input_tensor_sequence->DataType());
    CopySequenceTensor(alloc, *input_tensor_sequence, *output_tensor_sequence);
  } else {
    // Will not reach here
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Only Optional type OrtValues containing Tensors and Sequence Tensors "
                           "can be used as input to OptionalGetElement op");
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
