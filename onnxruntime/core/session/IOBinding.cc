// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/IOBinding.h"
#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel.h"
#include "core/framework/utils.h"

namespace onnxruntime {
IOBinding::IOBinding(const SessionState& session_state) : session_state_(session_state) {
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& names, const std::string& name) {
  auto it = std::find(std::begin(names), std::end(names), name);
  if (it == std::end(names)) {
    return {false, 0};
  }
  return {true, it - std::begin(names)};
}

common::Status IOBinding::BindInput(const std::string& name, const OrtValue& ml_value) {
  auto rc = Contains(feed_names_, name);

  auto add_or_replace = [this, &name](const bool exists, size_t index, const OrtValue& value) {
    if (exists) {
      feeds_[index] = value;
    } else {
      feed_names_.push_back(name);
      feeds_.push_back(value);
    }
  };

  if (ml_value.IsTensor() || ml_value.IsSparseTensor()) {
    OrtValue new_mlvalue;
    ORT_RETURN_IF_ERROR(utils::CopyOneInputAcrossDevices(session_state_, name, ml_value, new_mlvalue));
    add_or_replace(rc.first, rc.second, new_mlvalue);
  } else {
    add_or_replace(rc.first, rc.second, ml_value);
  }

  return Status::OK();
}

void IOBinding::ClearInputs() {
  feed_names_.clear();
  feeds_.clear();
}

static common::Status SyncProviders(const SessionState::NameNodeInfoMapType& node_info_map,
                                    const SessionState& session_state) {
  std::set<std::string> providers;
  for (auto& pair : node_info_map) {
    for (auto& node_info : pair.second) {
      if (node_info.p_node->GetExecutionProviderType() != onnxruntime::kCpuExecutionProvider) {
        providers.insert(node_info.p_node->GetExecutionProviderType());
      }
    }
  }
  for (auto& provider_type : providers) {
    auto* p_provider = session_state.GetExecutionProviders().Get(provider_type);
    if (!p_provider) {
      continue;
    }

    ORT_RETURN_IF_ERROR(p_provider->Sync());
  }
  return Status::OK();
}

common::Status IOBinding::SynchronizeInputs() {
  ORT_RETURN_IF_ERROR(SyncProviders(session_state_.GetInputNodeInfoMap(), session_state_));
  return Status::OK();
}

common::Status IOBinding::SynchronizeOutputs() {
  ORT_RETURN_IF_ERROR(SyncProviders(session_state_.GetOutputNodeInfoMap(), session_state_));
  return Status::OK();
}

common::Status IOBinding::BindOutput(const std::string& name, const OrtValue& ml_value) {
  // device value is ignored when ml_value is pre-allocated
  return BindOutputImpl(name, ml_value, {});
}

common::Status IOBinding::BindOutput(const std::string& name, OrtDevice device) {
  return BindOutputImpl(name, {}, device);
}

common::Status IOBinding::BindOutputImpl(const std::string& name, const OrtValue& ml_value, OrtDevice device) {
  auto rc = Contains(output_names_, name);
  if (rc.first) {
    outputs_[rc.second] = ml_value;
    outputs_device_info_[rc.second] = device;
  } else {
    output_names_.push_back(name);
    outputs_.push_back(ml_value);
    outputs_device_info_.push_back(device);
  }

  return Status::OK();
}

void IOBinding::ClearOutputs() {
  output_names_.clear();
  outputs_.clear();
  outputs_device_info_.clear();
}

const std::vector<std::string>& IOBinding::GetOutputNames() const { return output_names_; }

const std::vector<OrtValue>& IOBinding::GetOutputs() const { return outputs_; }

std::vector<OrtValue>& IOBinding::GetOutputs() { return outputs_; }

const std::vector<OrtDevice>& IOBinding::GetOutputsDeviceInfo() const {
  return outputs_device_info_;
}

const std::vector<std::string>& IOBinding::GetInputNames() const { return feed_names_; }

const std::vector<OrtValue>& IOBinding::GetInputs() const { return feeds_; }

AllocatorPtr IOBinding::GetCPUAllocator(int id, onnxruntime::ProviderType provider_type) const {
  auto& exec_providers = session_state_.GetExecutionProviders();
  auto* p_provider = exec_providers.Get(provider_type);
  ORT_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator(id, OrtMemTypeCPU);

  // if the provider does not implement CPU allocator, fall back to CPU
  if (allocator)
    return allocator;

  auto* cpu_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
  return cpu_provider->GetAllocator(0, OrtMemTypeDefault);
}

}  // namespace onnxruntime
