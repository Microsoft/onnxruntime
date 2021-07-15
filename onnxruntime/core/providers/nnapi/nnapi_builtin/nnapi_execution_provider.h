// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/nnapi/nnapi_provider_factory.h"

namespace onnxruntime {
namespace nnapi {
class Model;
}

class NnapiExecutionProvider : public IExecutionProvider {
 public:
  NnapiExecutionProvider(uint32_t nnapi_flags);
  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  // we implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

  uint32_t GetNNAPIFlags() const { return nnapi_flags_; }

 private:
  // The bit flags which define bool options for NNAPI EP, bits are defined as
  // NNAPIFlags in include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h
  const uint32_t nnapi_flags_;

  const std::unordered_set<std::string> partitioning_stop_ops_;

#ifdef __ANDROID__
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;
#endif
};
}  // namespace onnxruntime
