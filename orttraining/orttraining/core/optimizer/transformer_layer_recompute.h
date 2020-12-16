// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

class TransformerLayerRecompute : public GraphTransformer {
 public:
  TransformerLayerRecompute(int number_recompute_layers,
                            const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TransformerLayerRecompute", compatible_execution_providers),
        number_recompute_layers_(number_recompute_layers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  Status IdentifyTransformerLayerEdges(const Graph& graph,
                                       std::vector<const NodeArg*>& layer_start_edges,
                                       std::vector<const NodeArg*>& layer_end_edges,
                                       const logging::Logger& logger) const;

  std::vector<std::pair<const NodeArg*, const NodeArg*>>
  FindMatchingStartEndEdges(const Graph& graph,
                            const std::vector<const NodeArg*>& layer_start_edges,
                            const std::vector<const NodeArg*>& layer_end_edges) const;

  std::vector<const Node*> NodesBetweenEdges(const Graph& graph, const NodeArg* start, const NodeArg* end) const;

  void InsertRecomputeNodes(Graph& graph, const std::vector<const Node*>& nodes, int priority) const;

  int number_recompute_layers_;
};

}  // namespace onnxruntime
