// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <list>
#include <memory.h>

#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/mkldnn/subgraph/subgraph.h"

namespace mkldnn {
struct memory;
};

namespace onnxruntime {

// Information needed to construct MKL-DNN execution providers.
struct MKLDNNExecutionProviderInfo {
  bool create_arena{true};

  explicit MKLDNNExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  MKLDNNExecutionProviderInfo() = default;
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::shared_ptr<mkldnn::memory> GetWeightsMemoryBuffer(const std::string& weight_key) {
    auto iter = weights_mem_map_.find(weight_key);
    if (iter != weights_mem_map_.end())
      return iter->second;
    return nullptr;
  }

  void SetWeightsMemoryBuffer(const std::string& weight_key,
                        const std::shared_ptr<mkldnn::memory>& filter_dst_mem) {
    weights_mem_map_.insert(std::make_pair(weight_key, filter_dst_mem));
  }

  OrtMutex& GetMutex() {
    return mutex_;
  }

  void SaveAllocatedMemory(IAllocatorUniquePtr<void> buffer) {
    // keep reordered memory buffers in scope.
    reordered_buffers_.push_back(std::move(buffer));
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  // mkldnn weights(filer data) memory blocks from first iteration
  // saved by weights name
  std::unordered_map<std::string, std::shared_ptr<mkldnn::memory>> weights_mem_map_;
  // Save reordered memory buffers in list so that memory is not freed.
  std::vector<IAllocatorUniquePtr<void>> reordered_buffers_;
  OrtMutex mutex_;

 // SUBGRAPH
 private:
  bool UseSubgraph(const onnxruntime::GraphViewer& graph_viewer,
                   const std::vector<const KernelRegistry*>& kernel_registries,
                   std::vector<std::unique_ptr<ComputeCapability>>& result) const;

  // Some dimensions are not supported by MKL-DNN
  // example: Pool with NumDimensions <= 3 is not supported
  // Fall back to CPU implementation
  bool IsDimensionSupported(const Node* node) const {
    bool supported = true;
    if (node->OpType() == "BatchNormalization") {
      auto node_inputs = node->InputDefs();
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 3) {
        supported = false;
      }
    }
    if (node->OpType().find("Pool") != std::string::npos) {
      auto node_inputs = node->InputDefs();
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() <= 3) {
        supported = false;
      }
    }
    return supported;
  }

  void CreateOrUpdateMklDnnNode(const Node* node,
                                mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                                bool fused,
                                std::map<std::string, size_t>& output_to_source_node_map,
                                NodeAttributes& subgraph_attributes) const;

  // Create MklDnn node, update inputs, outputs and parent nodes
  // collect attribtes
  void CreateMetaDef(const onnxruntime::GraphViewer& graph_viewer,
                     const NodeAttributes& subgraph_attributes,
                     mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                     std::vector<std::unique_ptr<ComputeCapability>>& result) const;

 public:
  const std::shared_ptr<mkl_dnn::Subgraph> GetMklDnnSubgraph(const std::string& subgraph_id) {
    return mkl_subgraphs_[subgraph_id];
  }

 private:
  // supported MklDnn Operators
  std::set<std::string> mkldnn_ops_ = {"Conv", "BatchNormalization", "Relu", "Sum",
                                       "AveragePool", "GlobalMaxPool", "GlobalAveragePool", "MaxPool", "LRN"};

  mutable std::unordered_map<std::string, std::shared_ptr<mkl_dnn::Subgraph>> mkl_subgraphs_;  
};

}  // namespace onnxruntime
