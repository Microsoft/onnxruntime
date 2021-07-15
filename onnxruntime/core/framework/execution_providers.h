// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

/**
Class for managing lookup of the execution providers in a session.
*/
class ExecutionProviders {
 public:
  ExecutionProviders() = default;

  common::Status Add(const std::string& provider_id, std::shared_ptr<IExecutionProvider> p_exec_provider) {
    // make sure there are no issues before we change any internal data structures
    if (provider_idx_map_.find(provider_id) != provider_idx_map_.end()) {
      auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " has already been registered.");
      LOGS_DEFAULT(ERROR) << status.ErrorMessage();
      return status;
    }

    // index that provider will have after insertion
    auto new_provider_idx = exec_providers_.size();

    ORT_IGNORE_RETURN_VALUE(provider_idx_map_.insert({provider_id, new_provider_idx}));

    // update execution provider options
    exec_provider_options_[provider_id] = p_exec_provider->GetProviderOptions();

    exec_provider_ids_.push_back(provider_id);
    exec_providers_.push_back(std::move(p_exec_provider));
    return Status::OK();
  }

  const IExecutionProvider* Get(const onnxruntime::Node& node) const {
    return Get(node.GetExecutionProviderType());
  }

  const IExecutionProvider* Get(onnxruntime::ProviderType provider_id) const {
    auto it = provider_idx_map_.find(provider_id);
    if (it == provider_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  IExecutionProvider* Get(onnxruntime::ProviderType provider_id) {
    auto it = provider_idx_map_.find(provider_id);
    if (it == provider_idx_map_.end()) {
      return nullptr;
    }

    return exec_providers_[it->second].get();
  }

  bool Empty() const { return exec_providers_.empty(); }

  size_t NumProviders() const { return exec_providers_.size(); }

  using const_iterator = typename std::vector<std::shared_ptr<IExecutionProvider>>::const_iterator;
  const_iterator begin() const noexcept { return exec_providers_.cbegin(); }
  const_iterator end() const noexcept { return exec_providers_.cend(); }

  const AllocatorPtr GetDefaultCpuAllocator() const {
    return Get(onnxruntime::kCpuExecutionProvider)->GetAllocator(0, OrtMemTypeDefault);
  }

  OrtMemoryInfo GetDefaultCpuMemoryInfo() const {
    return GetDefaultCpuAllocator()->Info();
  }

  const std::vector<std::string>& GetIds() const { return exec_provider_ids_; }
  const ProviderOptionsMap& GetAllProviderOptions() const { return exec_provider_options_; }

 private:
  // Some compilers emit incomprehensive output if this is allowed
  // with a container that has unique_ptr or something move-only.
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ExecutionProviders);

  std::vector<std::shared_ptr<IExecutionProvider>> exec_providers_;
  std::vector<std::string> exec_provider_ids_;
  ProviderOptionsMap exec_provider_options_;

  // maps for fast lookup of an index into exec_providers_
  std::unordered_map<std::string, size_t> provider_idx_map_;
};
}  // namespace onnxruntime
