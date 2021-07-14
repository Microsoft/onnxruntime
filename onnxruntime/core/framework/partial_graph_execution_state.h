//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_TRAINING
#include "core/common/common.h"
#include "core/framework/execution_frame.h"

namespace onnxruntime {

class OrtValueCache {
 public:
  OrtValueCache() = default;

  Status CacheOrtValue(std::string node_arg_name, OrtValue& value) {
    cache_.emplace(node_arg_name, value);
    return Status::OK();
  }

  Status GetCachedIds(std::vector<std::string>& keys) {
    keys.reserve(cache_.size());
    for(auto kv : cache_) {
      keys.push_back(kv.first);
    }
    return Status::OK();
  }

  Status DeleteOrtValue(std::string node_arg_name) {
    ORT_RETURN_IF(cache_.find(node_arg_name) == cache_.end(), "NodeArg not found in cache: ", node_arg_name);
    cache_.erase(node_arg_name);
    return Status::OK();
  }

  Status DeleteCache() {
    cache_.clear();
    return Status::OK();
  }

 private:
  std::unordered_map<std::string, OrtValue> cache_;
};

struct PartialGraphExecutionState {
 public:
  PartialGraphExecutionState() {
    execution_frame_ = nullptr;
  }

  ~PartialGraphExecutionState() = default;

  void SetProgramCounterStart(size_t start) { program_counter_start_ = start; }
  void SetProgramCounterEnd(size_t end) { program_counter_end_ = end; }

  size_t GetProgramCounterStart() { return program_counter_start_; }
  size_t GetProgramCounterEnd() { return program_counter_end_; }

  ExecutionFrame& GetExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                                    const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                                    const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                    const SessionState& session_state) {
    if (execution_frame_ == nullptr) {
      execution_frame_ = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches,
                                                          fetch_allocators, session_state);
    } else {
      execution_frame_->UpdateFeeds(feed_mlvalue_idxs, feeds);
      execution_frame_->UpdateFetches(fetch_mlvalue_idxs, fetches, session_state.GetInitializedTensors());
    }

    return *execution_frame_;
  }

 private:
  std::unique_ptr<ExecutionFrame> execution_frame_;
  size_t program_counter_start_;
  size_t program_counter_end_;
};
}  // namespace onnxruntime
#endif
