// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/controlflow/if.h"

#include "core/providers/cuda/cuda_common.h"

//#include "core/framework/framework_common.h"
//#include "core/framework/op_kernel_context_internal.h"
//#include "core/framework/session_state.h"
//#include "core/framework/tensorprotoutils.h"
//#include "core/framework/utils.h"
//#include "core/framework/session_options.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .InputMemoryType<OrtMemTypeCPUInput>(0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  If);

// output shape rules requiring the output shapes of the 'THEN' and 'ELSE'
// branches to be the same were relaxed in opset-11
ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        11,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .InputMemoryType<OrtMemTypeCPUInput>(0)  // 'cond' needs to be on CPU
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                        If);

Status If::Compute(OpKernelContext* ctx) const {
  // call the base CPU version. 
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug 
  // that this implementation is being called with it.
  auto status = onnxruntime::If::Compute(ctx);
  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
