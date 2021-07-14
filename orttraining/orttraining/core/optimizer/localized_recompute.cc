// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/localized_recompute.h"
#include "orttraining/core/optimizer/dropout_recompute.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

bool GeluRecompute::SatisfyCondition(const Node& node) const {
  static const std::unordered_set<std::string> target_optypes = {"Gelu", "FastGelu", "BiasGelu"};
  if (target_optypes.find(node.OpType()) == target_optypes.end()) {
    return false;
  }

  const auto next_node = node.OutputNodesBegin();
  if (next_node != node.OutputNodesEnd() && next_node->OpType() == "MatMul") {
    return true;
  }
  return false;
}

Status GeluRecompute::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Traverse backward from the bottom of the graph, so that the recompute nodes
  // for lower layers are executed earlier
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);

    if (!SatisfyCondition(node)) {
      continue;
    }

    const auto& output = node.OutputDefs()[0];

    auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                       output->TypeAsProto());

    Node& recompute_node = graph.AddNode(node.Name() + "_recompute",
                                         node.OpType(),
                                         "Recompute of " + node.Name(),
                                         node.MutableInputDefs(),
                                         {&recomputed_output},
                                         &node.GetAttributes(),
                                         node.Domain());

    recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

    modified = true;
  }

  return Status::OK();
}

bool AttentionDropoutRecompute::SatisfyCondition(const Node& node) const {
  if (node.OpType() != "Dropout")
    return false;

  const auto prev_node = node.InputNodesBegin();
  const auto next_node = node.OutputNodesBegin();
  if (prev_node != node.InputNodesEnd() && prev_node->OpType() == "Softmax" &&
      next_node != node.OutputNodesEnd() && next_node->OpType() == "MatMul") {
    return true;
  }
  return false;
}

Status AttentionDropoutRecompute::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Traverse backward from the bottom of the graph, so that the recompute nodes
  // for lower layers are executed earlier
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);

    if (!SatisfyCondition(node)) {
      continue;
    }

    Node& recompute_node = InsertDropoutRecompute(graph, node, /*use_original_input*/ true);
    recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

    modified = true;
  }
  return Status::OK();
}

bool ExpandRecompute::SatisfyCondition(const Node& node) const {
  if (node.OpType() != "Expand")
    return false;

  for (auto output_edge = node.OutputEdgesBegin(); output_edge != node.OutputEdgesEnd(); output_edge++) {
    const std::string& next_op = output_edge->GetNode().OpType();
    // Where's condition input
    // GatherElements's indices input
    if ((next_op == "Where" && output_edge->GetDstArgIndex() == 0) ||
        (next_op == "GatherElements" && output_edge->GetDstArgIndex() == 1)) {
      return true;
    }
  }
  return false;
}

Status ExpandRecompute::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder();

  // Traverse backward from the bottom of the graph, so that the recompute nodes
  // for lower layers are executed earlier
  for (int i = static_cast<int>(node_ids.size() - 1); i >= 0; --i) {
    Node& node = *graph.GetNode(node_ids[i]);

    if (!SatisfyCondition(node)) {
      continue;
    }

    const auto& output = node.OutputDefs()[0];
    auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                       output->TypeAsProto());

    Node& recompute_node = graph.AddNode(node.Name() + "_recompute",
                                         node.OpType(),
                                         "Recompute of " + node.Name(),
                                         node.MutableInputDefs(),
                                         {&recomputed_output},
                                         &node.GetAttributes(),
                                         node.Domain());

    recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
