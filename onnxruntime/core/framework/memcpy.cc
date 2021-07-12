// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_transfer_manager.h"
#include "memcpy.h"
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
}

Status Memcpy::Compute(OpKernelContext* ctx) const {
  Status retval;
  MLDataType input_type_0 = ctx->InputType(0);
  if (input_type_0->IsTensorType()) {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());

    if (!retval.IsOK()) {
      LOGS(ctx->Logger(), ERROR) << MakeString(retval.ErrorMessage(),
                                               " Copying ", Node().InputDefs()[0]->Name(),
                                               " to ", Node().OutputDefs()[0]->Name(),
                                               " Input shape:", X->Shape(), " Output shape:", Y->Shape(),
                                               " X data:", X->DataRaw(), " Y data:", Y->DataRaw());
    }
  } else if (input_type_0->IsSparseTensorType()) {
    const auto* X = ctx->Input<SparseTensor>(0);
    SparseTensor* Y = ctx->OutputSparse(0, X->Shape());
    retval = X->Copy(Info().GetDataTransferManager(), Info().GetKernelDef().ExecQueueId(), *Y);
    if (!retval.IsOK()) {
      LOGS(ctx->Logger(), ERROR) << MakeString(retval.ErrorMessage(),
                                               " Copying ", Node().InputDefs()[0]->Name(),
                                               " to ", Node().OutputDefs()[0]->Name(),
                                               " Input shape:", X->Shape(), " Output shape:", Y->Shape());
    }
  } else {
    ORT_NOT_IMPLEMENTED("Input type not supported: ", DataTypeImpl::ToString(input_type_0));
  }

  return retval;
}

}  // namespace onnxruntime
