// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "async_kernel.h"
#include "async_execution_provider.h"

#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {

AsyncKernel::AsyncKernel(
    const Node& fused_node) {
  const auto* func_body = fused_node.GetFunctionBody();
  ORT_ENFORCE(func_body != nullptr);
  const Graph& subgraph = func_body->Body();
  ORT_ENFORCE(subgraph.NumberOfNodes() == 1);
  std::string op_type = subgraph.GetNode(0)->OpType();

  if (op_type == "Add") {
    func_ = [this]() {
      for (int64_t i = 0; i < func_args_.count; ++i) {
        func_args_.output[i] = func_args_.input0[i] + func_args_.input1[i];
      }
    };
  } else if (op_type == "Sub") {
    func_ = [this]() {
      for (int64_t i = 0; i < func_args_.count; ++i) {
        func_args_.output[i] = func_args_.input0[i] - func_args_.input1[i];
      }
    };
  } else if (op_type == "Mul") {
    func_ = [this]() {
      for (int64_t i = 0; i < func_args_.count; ++i) {
        func_args_.output[i] = func_args_.input0[i] * func_args_.input1[i];
      }
    };
  }

  // get attributes for async execution
  ProtoHelperNodeContext proto_ctx(fused_node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&proto_ctx);

  ORT_ENFORCE(info.GetAttr<int64_t>(AsyncExecutionProvider::NodeAttr_Stream, &cfg_.stream_id).IsOK());
  cfg_.wait_events = info.GetAttrsOrDefault<int64_t>(AsyncExecutionProvider::NodeAttr_WaitEvents);
  cfg_.record_event = info.GetAttrOrDefault<int64_t>(AsyncExecutionProvider::NodeAttr_RecordEvent, AsyncExecutionProvider::EmptyEvent);
  cfg_.prior_sync_stream_id = info.GetAttrOrDefault<int64_t>(AsyncExecutionProvider::NodeAttr_PriorSyncStream, AsyncExecutionProvider::EmptyStream);
  cfg_.posterior_sync_stream_id = info.GetAttrOrDefault<int64_t>(AsyncExecutionProvider::NodeAttr_PosteriorSyncStream, AsyncExecutionProvider::EmptyStream);
}

Status AsyncKernel::Launch(OpKernelContext* op_kernel_context, AsyncExecutionStream& stream) const {
  // the key design is the separation of shape inference and compute
  // and compute happens in async stream

  // launch: for demo purpose, shape inference and compute is simple element wise
  const Tensor* input0_tensor = op_kernel_context->Input<Tensor>(0);
  const Tensor* input1_tensor = op_kernel_context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input0_tensor->Shape() == input1_tensor->Shape());
  Tensor* output_tensor = op_kernel_context->Output(0, input0_tensor->Shape());

  func_args_.input0 = input0_tensor->Data<float>();
  func_args_.input1 = input1_tensor->Data<float>();
  func_args_.output = output_tensor->MutableData<float>();
  func_args_.count = input0_tensor->Shape().Size();

  stream.Launch([&]() {
    func_();
  });

  return Status::OK();
};

}  // namespace onnxruntime