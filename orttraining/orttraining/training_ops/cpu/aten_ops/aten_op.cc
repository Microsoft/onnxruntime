// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_executor.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ATenOp, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATenOp);

Status ATenOp::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  std::vector<DLManagedTensor*> dlpacks;
  for (int i = 0; i < p_ctx_internal->InputCount(); i++) {
    const OrtValue* p_ort_value = p_ctx_internal->GetInputMLValue(i);
    if (!p_ort_value) {
      dlpacks.emplace_back(nullptr);
    } else {
      OrtValue ort_value = *p_ort_value;
      dlpacks.emplace_back(dlpack::OrtValueToDlpack(ort_value));
    }
  }

  auto result = aten_ops::ATenOperatorExecutor::Instance()(op_name_, overload_name_, dlpacks);
  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
