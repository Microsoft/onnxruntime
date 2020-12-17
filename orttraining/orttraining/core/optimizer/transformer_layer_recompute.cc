// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/transformer_layer_recompute.h"
#include "orttraining/core/optimizer/dropout_recompute.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "core/common/common.h"

#include <deque>

namespace onnxruntime {

namespace {
bool IsOpType(const Node& node, const std::unordered_set<std::string>& op_types) {
  return op_types.count(node.OpType()) > 0;
}

}  // namespace

Status TransformerLayerRecompute::IdentifyTransformerLayerEdges(
    const Graph& graph,
    std::vector<const NodeArg*>& layer_start_edges,
    std::vector<const NodeArg*>& layer_end_edges,
    const logging::Logger& logger) const {
  const std::unordered_set<std::string> gelu_ops{"Gelu", "BiasGelu", "FastGelu"};
  const std::unordered_set<std::string> dropout_ops{"Dropout", "BiasDropout"};
  const std::unordered_set<std::string> layernorm_ops{"LayerNormalization", "SkipLayerNormalization", "SimplifiedLayerNormalization"};
  const std::unordered_set<std::string> matmul_ops{"MatMul", "FusedMatMul", "TransposeMatMul"};
  const std::unordered_set<std::string> transpose_ops{"Transpose"};

  layer_start_edges.clear();
  layer_end_edges.clear();

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);

    /*
    Look for start of a transformer layer
          [LayerNorm|Dropout|Transpose]
         /       |         |           \       <----- Start of the transformer layer
       (.)    MatMul    MatMul       MatMul
    */
    if ((IsOpType(node, layernorm_ops) || IsOpType(node, dropout_ops) || IsOpType(node, transpose_ops)) &&
        node.GetOutputEdgesCount() == 4) {
      int matmul_count = 0;
      for (auto next_node = node.OutputNodesBegin(); next_node != node.OutputNodesEnd(); ++next_node) {
        if (IsOpType(*next_node, matmul_ops)) {
          matmul_count++;
        }
      }

      // Three following MatMul nodes for QKV projection
      if (matmul_count == 3) {
        layer_start_edges.push_back(node.OutputDefs()[0]);
      }
    }

    // Look for end of a transformer layer
    // Pattern:  Gelu -> (.*) -> MatMul -> (.*) -> Dropout -> (.*) -> LayerNorm
    // The output of LayerNorm is the end of a transformer layer
    if (IsOpType(node, gelu_ops)) {
      auto next_node = node.OutputNodesBegin();

      if (next_node == node.OutputNodesEnd()) {
        continue;
      }

      while (next_node->OutputNodesBegin() != next_node->OutputNodesEnd() &&
             !IsOpType(*next_node, matmul_ops)) {
        next_node = next_node->OutputNodesBegin();
      }

      while (next_node->OutputNodesBegin() != next_node->OutputNodesEnd() &&
             !IsOpType(*next_node, dropout_ops)) {
        next_node = next_node->OutputNodesBegin();
      }

      while (next_node->OutputNodesBegin() != next_node->OutputNodesEnd() &&
             !IsOpType(*next_node, layernorm_ops)) {
        next_node = next_node->OutputNodesBegin();
      }

      if (next_node->OutputNodesBegin() == next_node->OutputNodesEnd()) {
        continue;
      } else if (IsOpType(*next_node, layernorm_ops)) {
        layer_end_edges.push_back(next_node->OutputDefs()[0]);
      }
    }
  }

  ORT_RETURN_IF_NOT(layer_start_edges.size() == layer_end_edges.size(),
                    "Number of start and end edges doesn't match!, #start=", layer_start_edges.size(),
                    ", #end=", layer_end_edges.size());

  LOGS(logger, INFO) << "Found " << layer_start_edges.size() << " transformer layers.";
  return Status::OK();
}

namespace {

typedef std::set<const Node*, NodeCompare> NodeSet;

NodeSet BFSFrom(const std::vector<const Node*>& start_nodes, bool reverse) {
  NodeSet visited(start_nodes.begin(), start_nodes.end());
  std::deque<const Node*> queue(start_nodes.begin(), start_nodes.end());
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    auto begin = reverse ? n->InputNodesBegin() : n->OutputNodesBegin();
    auto end = reverse ? n->InputNodesEnd() : n->OutputNodesEnd();

    for (auto node_it = begin; node_it != end; ++node_it) {
      const Node& node = *node_it;
      if (visited.find(&node) == visited.end()) {
        queue.push_back(&node);
        visited.insert(&node);
      }
    }
  }
  return visited;
}

const NodeArg* FindMatchingEndEdge(const std::vector<const Node*>& start_nodes,
                                   const std::unordered_map<const Node*, const NodeArg*>& end_nodes_edges) {
  NodeSet visited(start_nodes.begin(), start_nodes.end());
  std::deque<const Node*> queue(start_nodes.begin(), start_nodes.end());
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    // return the first encountered end node
    auto iter = end_nodes_edges.find(n);
    if (iter != end_nodes_edges.end()) {
      return iter->second;
    }

    for (auto node_it = n->OutputNodesBegin(); node_it != n->OutputNodesEnd(); ++node_it) {
      const Node& node = *node_it;
      if (visited.find(&node) == visited.end()) {
        queue.push_back(&node);
        visited.insert(&node);
      }
    }
  }
  return nullptr;
}
}  // namespace

std::vector<std::pair<const NodeArg*, const NodeArg*>>
TransformerLayerRecompute::FindMatchingStartEndEdges(const Graph& graph,
                                                     const std::vector<const NodeArg*>& start_edges,
                                                     const std::vector<const NodeArg*>& end_edges) const {
  std::unordered_map<const Node*, const NodeArg*> candidate_end_nodes_edges;
  for (auto end_edge : end_edges) {
    candidate_end_nodes_edges.insert({graph.GetProducerNode(end_edge->Name()), end_edge});
  }

  std::vector<std::pair<const NodeArg*, const NodeArg*>> start_end_edges;
  for (size_t i = 0; i < start_edges.size(); ++i) {
    const NodeArg* end_edge = FindMatchingEndEdge(graph.GetConsumerNodes(start_edges[i]->Name()),
                                                  candidate_end_nodes_edges);
    start_end_edges.emplace_back(start_edges[i], end_edge);
  }

  return start_end_edges;
}

std::vector<const Node*> TransformerLayerRecompute::NodesBetweenEdges(const Graph& graph, const NodeArg* start, const NodeArg* end) const {
  // Forward BFS from the start node
  std::vector<const Node*> start_nodes = graph.GetConsumerNodes(start->Name());
  NodeSet fw_visited = BFSFrom(start_nodes, /*reverse*/ false);

  // Reverse BFS from the end node
  const Node* end_node = graph.GetProducerNode(end->Name());
  NodeSet bw_visited = BFSFrom({end_node}, /*reverse*/ true);

  // Join fw_visited and bw_visited
  std::vector<const Node*> intersect_nodes;
  std::set_intersection(fw_visited.begin(), fw_visited.end(),
                        bw_visited.begin(), bw_visited.end(),
                        std::back_inserter(intersect_nodes), NodeCompare());

  return intersect_nodes;
}

void TransformerLayerRecompute::InsertRecomputeNodes(Graph& graph, const std::vector<const Node*>& nodes, int priority) const {
  auto initializers = graph.GetAllInitializedTensors();

  for (const Node* n : nodes) {
    Node* node = graph.GetNode(n->Index());

    // recomputed Dropout need to produce the same output as original dropout
    // currently reusing original dropout's mask to achieve this
    if (node->OpType() == "Dropout") {
      const NodeArg* input = node->InputDefs()[0];
      const Node* p_node = graph.GetProducerNode(input->Name());

      bool use_original_input =
          initializers.find(input->Name()) != initializers.end() ||
          std::find(nodes.begin(), nodes.end(), p_node) == nodes.end();

      Node& recompute_node = InsertDropoutRecompute(graph, *node, use_original_input);
      recompute_node.SetPriority(priority);
      continue;
    }

    // prepare inputs for recompute node
    std::vector<NodeArg*> recomputed_inputs;
    for (NodeArg* input : node->MutableInputDefs()) {
      const Node* p_node = graph.GetProducerNode(input->Name());

      // do not duplicate initializers in recompute subgraph
      if (initializers.find(input->Name()) != initializers.end() ||
          std::find(nodes.begin(), nodes.end(), p_node) == nodes.end()) {
        recomputed_inputs.push_back(input);
      } else {
        auto& recomputed_input = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(input->Name()),
                                                          input->TypeAsProto());
        recomputed_inputs.push_back(&recomputed_input);
      }
    }

    // prepare ouputs for recompute node
    std::vector<NodeArg*> recomputed_outputs;
    for (NodeArg* output : node->MutableOutputDefs()) {
      auto& recomputed_output = graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                         output->TypeAsProto());
      recomputed_outputs.push_back(&recomputed_output);
    }

    Node& recompute_node = graph.AddNode(node->Name() + "_recompute",
                                         node->OpType(),
                                         "Recompute of " + node->Name(),
                                         recomputed_inputs,
                                         recomputed_outputs,
                                         &node->GetAttributes(),
                                         node->Domain());
    recompute_node.SetPriority(priority);
  }
  return;
}

Status TransformerLayerRecompute::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger) const {
  std::vector<const NodeArg*> start_edges, end_edges;
  Status s = IdentifyTransformerLayerEdges(graph, start_edges, end_edges, logger);

  if (!s.IsOK()) {
    modified = false;
    return Status::OK();
  }

  // by default, apply recompute expect for the last transformer layer
  // otherwise, take user specified 'number_recompute_layers_'s
  int n_layers;
  const int n_layers_limit = static_cast<int>(start_edges.size() - 1);
  if (number_recompute_layers_ > n_layers_limit) {
    LOGS(logger, WARNING) << "User specified number_recompute_layers " << number_recompute_layers_
                          << " is larger than limit " << n_layers_limit << "."
                          << "number_recompute_layers is now cliped to limit.";
    n_layers = n_layers_limit;
  } else if (number_recompute_layers_ > 0) {
    n_layers = number_recompute_layers_;
  } else {
    LOGS(logger, WARNING) << "number_recompute_layers is not set by user, using default " << n_layers_limit << ".";
    n_layers = n_layers_limit;
  }

  std::vector<std::pair<const NodeArg*, const NodeArg*>> start_end_edges = FindMatchingStartEndEdges(graph, start_edges, end_edges);

  // latter recompute layers have higher execution priorty
  for (int i = 0; i < n_layers; ++i) {
    std::vector<const Node*> nodes = NodesBetweenEdges(graph, start_end_edges[i].first, start_end_edges[i].second);

    std::stringstream node_names;
    for (const auto n : nodes) {
      node_names << n->Name() << ",";
    }
    LOGS(logger, INFO) << "Recompute Layer " << i << "."
                       << "Start edge: " << start_end_edges[i].first->Name() << " End edge: " << start_end_edges[i].second->Name()
                       << "Nodes between layer : " << node_names.str();

    InsertRecomputeNodes(graph, nodes, static_cast<int>(start_edges.size() - i));
  }

  modified = true;
  return Status::OK();
}

}  // namespace onnxruntime
