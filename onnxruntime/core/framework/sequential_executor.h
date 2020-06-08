// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
class SequentialExecutor : public IExecutor {
 public:
  SequentialExecutor(const RunOptions& run_options)
      : run_options_(run_options){}

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger,
                         const AllocatorPtr custom_cpu_allocator) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SequentialExecutor);
  const RunOptions& run_options_;
};
}  // namespace onnxruntime
