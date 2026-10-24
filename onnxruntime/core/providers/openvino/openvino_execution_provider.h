// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>
#include <atomic>

#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/contexts.h"
#include "ov_shared_context.h"
#include "ov_bin_manager.h"
#include "ov_interface.h"

#ifdef _WIN32
#include "core/providers/openvino/ov_tracing.h"
#endif

namespace onnxruntime {
namespace openvino_ep {

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI (or) AUTO followed by the devices in the order of priority "
            << "you want to build"
            << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI/AUTO build "
            << "are ['CPU','GPU','NPU','GPU.x'] where x = 0,1,2 and so on"
            << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI or AUTO build type. "
            << "Ex: HETERO:GPU,CPU  Ex: MULTI:GPU,CPU Ex: AUTO:GPU,CPU Ex: AUTO:GPU.0,CPU Ex: AUTO:GPU.1,CPU"
            << std::endl;
}

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

// Logical device representation.
class OpenVINOExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenVINOExecutionProvider(const ProviderInfo& info);
  ~OpenVINOExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Status SetEpDynamicOptions(gsl::span<const char* const> /*keys*/,
                             gsl::span<const char* const> /*values*/) override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  const InlinedVector<const Node*> GetEpContextNodes() const override;

  Status GetEpContextDataCallbackSupport(uint32_t& supported_flags) const override {
    supported_flags = OrtEpContextDataCallbackSupportFlags_READ | OrtEpContextDataCallbackSupportFlags_WRITE;
    return Status::OK();
  }

#ifdef USE_OVEP_NPU_MEMORY
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
#endif
 private:
  SessionContext session_context_;
  std::shared_ptr<OVCore> ov_core_;
  std::shared_ptr<SharedContextManager> shared_context_manager_;
  std::shared_ptr<SharedContext> shared_context_;

  std::list<BackendManager> backend_managers_;  // EP session owns the backend objects
  EPCtxHandler ep_ctx_handle_;

  // Tracing and session tracking
  uint32_t session_id_{0};
  static std::atomic<uint32_t> global_session_counter_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
