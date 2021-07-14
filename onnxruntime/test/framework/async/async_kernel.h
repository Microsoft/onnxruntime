// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/framework/func_kernel.h"

namespace onnxruntime {

class AsyncExecutionStream;
class OpKernel;

struct AsyncExecConfig {
  int64_t stream_id;
  std::vector<int64_t> wait_events;
  int64_t record_event;
  int64_t prior_sync_stream_id;
  int64_t posterior_sync_stream_id;
};

// Compiled kernel for fused node to execute asynchronously
class AsyncKernel {
 public:
  explicit AsyncKernel(
      const Node& fused_node);

  // note: AsyncKernel runs shape inference and output allocation in dispatcher thread
  // then queues up AsyncTask to EP's stream and return
  Status Launch(OpKernelContext* op_kernel_context, AsyncExecutionStream& stream) const;

  const AsyncExecConfig& GetAsyncExecConfig() const {
    return cfg_;
  }

 private:
  // async compute function and args
  std::function<void()> func_;
  struct FuncArgs {
    const float* input0;
    const float* input1;
    float* output;
    int64_t count;
  };
  mutable FuncArgs func_args_;

  AsyncExecConfig cfg_;
};

}  // namespace onnxruntime