// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_propagation.h"

#include <cassert>
#include <optional>
#include <queue>
#include <sstream>
#include <utility>

#include <onnx/defs/attr_proto_util.h>
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/extended_graph_edge.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

using onnxruntime::graph_utils::ExtendedGraphEdge;

namespace onnxruntime {
namespace {
bool CanNodePropagate(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12, 22}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {5, 13, 14, 19, 21, 23, 24, 25}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13, 21, 23, 24, 25}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Squeeze", {1, 11, 13, 21, 23, 24, 25}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Unsqueeze", {1, 11, 13, 21, 23, 24, 25}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 10, 11, 13});
}

// Makes matching attributes for new QuantizeLinear nodes from an existing DequantizeLinear node.
NodeAttributes MakeQAttrsFromDQ(const Node& dq_node) {
  // DQ's axis and block_size attributes are also valid Q attributes for all supported opsets.
  // So, start with a copy of the DQ attributes.
  NodeAttributes q_attrs = dq_node.GetAttributes();

  // From opset 21 onward, Q supports an "output_dtype" attribute that controls the quantized
  // element type when no zero-point input is provided. Without it, qdq_util.cc silently falls
  // back to UINT8, which corrupts negative values for INT8 (or other signed) inputs.
  if (dq_node.SinceVersion() >= 21) {
    const auto* input_type_proto = dq_node.InputDefs()[0]->TypeAsProto();
    if (input_type_proto != nullptr) {
      const int32_t elem_type = input_type_proto->tensor_type().elem_type();
      if (elem_type != ONNX_NAMESPACE::TensorProto::UNDEFINED) {
        q_attrs["output_dtype"] = ONNX_NAMESPACE::MakeAttribute("output_dtype",
                                                                static_cast<int64_t>(elem_type));
      }
    }
  }

  return q_attrs;
}

// Makes matching attributes for new DequantizeLinear nodes from an existing QuantizeLinear node.
NodeAttributes MakeDQAttrsFromQ(const Node& q_node) {
  // QDQ::MatchQNode() accepts opsets 10–25; the assert below would abort debug builds for
  // opset 23/24/25 models, so it is intentionally omitted.
  const NodeAttributes& q_attrs = q_node.GetAttributes();
  if (q_attrs.empty()) {
    return {};
  }

  // Only the "axis" and "block_size" attributes for Q are also valid DQ attributes.
  NodeAttributes dq_attrs;

  auto axis_attr_it = q_attrs.find("axis");
  if (axis_attr_it != q_attrs.end()) {
    dq_attrs.insert({axis_attr_it->first, axis_attr_it->second});
  }

  auto block_size_attr_it = q_attrs.find("block_size");
  if (block_size_attr_it != q_attrs.end()) {
    dq_attrs.insert({block_size_attr_it->first, block_size_attr_it->second});
  }

  return dq_attrs;
}

// Validates edges into which to insert Q -> DQ ops.
// - Must have at least one edge.
// - All edges must correspond to the same graph NodeArg (i.e., same source but potentially different destination).
// - All edges must be attached to either a source node or a destination node.
Status ValidateQDQInsertionEdges(Graph& graph, gsl::span<const ExtendedGraphEdge> insertion_edges) {
  const size_t num_edges = insertion_edges.size();
  ORT_RETURN_IF(num_edges == 0, "Expected at least one edge into which to insert QDQ pair.");

  const ExtendedGraphEdge& first_edge = insertion_edges[0];
  const Node* src_node = first_edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  const Node* first_dst_node = first_edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
  const std::string& node_arg_name = first_edge.arg_name;
  ORT_RETURN_IF_NOT(graph.GetNodeArg(node_arg_name) != nullptr,
                    "QDQ insertion edge does not have a valid graph NodeArg for ", node_arg_name);
  ORT_RETURN_IF_NOT(src_node != nullptr || first_dst_node != nullptr,
                    "QDQ insertion edge [0] for NodeArg ", node_arg_name,
                    " must have a source or a destination node");

  for (size_t i = 1; i < num_edges; i++) {
    const ExtendedGraphEdge& insertion_edge = insertion_edges[i];
    ORT_RETURN_IF_NOT(insertion_edge.arg_name == node_arg_name,
                      "QDQ insertion edge [", i, "] has NodeArg ", insertion_edge.arg_name,
                      " but expected NodeArg ", node_arg_name);

    const Node* edge_dst_node = insertion_edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
    ORT_RETURN_IF_NOT(src_node != nullptr || edge_dst_node != nullptr,
                      "QDQ insertion edge [", i, "] for NodeArg ", node_arg_name,
                      " must have a source or a destination node");
  }

  return Status::OK();
}

// Logs information about the edges into which Q/DQ nodes will be inserted in InsertQDQPairs().
// Assumes the edges have already been validated.
void LogQDQInsertion(const logging::Logger& logger, logging::Severity severity, const CodeLocation& code_location,
                     const Graph& graph, gsl::span<const ExtendedGraphEdge> edges) {
  auto logging_data_type = logging::DataType::SYSTEM;
  if (!logger.OutputIsEnabled(severity, logging_data_type)) {
    return;
  }

  const Node* src_node = edges[0].GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  const auto& node_arg_name = edges[0].arg_name;
  std::string src_label = src_node ? MakeString("node (\"", src_node->Name(), "\", index: ", src_node->Index(), ")")
                                   : "input";
  std::ostringstream dst_labels;
  const size_t num_edges = edges.size();

  for (size_t i = 0; i < num_edges; ++i) {
    const ExtendedGraphEdge& edge = edges[i];
    const Node* dst_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
    dst_labels << (dst_node ? MakeString("dst node (\"", dst_node->Name(), "\", index: ", dst_node->Index(), ")")
                            : "output")
               << (i == num_edges - 1 ? "" : ",");
  }

  logging::Capture(logger, severity, logging::Category::onnxruntime, logging_data_type, code_location).Stream()
      << "Inserted Q/DQ pair between "
      << (src_node ? MakeString("src node (\"", src_node->Name(), "\", index: ", src_node->Index(), ")")
                   : "input")
      << " and " << dst_labels.str()
      << " at NodeArg \"" << node_arg_name << "\".";
}

// convert this: src_node (or graph input) --+--> dst_node_0 (or graph output)
//                                           |
//                                           +--> dst_node_1
//                                           |    ...
//                                           +--> dst_node_n
//
// to this: src_node (or graph input) -> Q --+--> DQ -> dst_node_0 (or graph output)
//                                           |
//                                           +--> DQ -> dst_node_1
//                                           |    ...
//                                           +--> DQ -> dst_node_n
// Checks that all insertion edges share the same NodeArg. That is, the edges originate from the same source node
// output. If there is no src_node, then all edges should come from the same graph input.
// This function returns an error status if edges are invalid.
//
// Assumes that scale_initializer_nodearg and zp_initializer_nodearg_ptr (if not null) are constant initializers.
Status InsertQDQPairs(Graph& graph, gsl::span<const ExtendedGraphEdge> insertion_edges,
                      NodeArg& scale_initializer_nodearg, NodeArg* zp_initializer_nodearg_ptr,
                      const std::string& qdq_domain, const NodeAttributes& q_attrs, const NodeAttributes& dq_attrs,
                      const logging::Logger& logger) {
  ORT_RETURN_IF_ERROR(ValidateQDQInsertionEdges(graph, insertion_edges));

  const ExtendedGraphEdge& first_edge = insertion_edges[0];  // ValidateQDQInsertionEdges() guarantees at least one edge

  Node* src_node = first_edge.GetMutableNodeAtEnd(graph, ExtendedGraphEdge::End::Source);  // nullptr for graph input
  const auto& base_name = first_edge.arg_name;
  auto& base_node_arg = *graph.GetNodeArg(base_name);

  LogQDQInsertion(logger, logging::Severity::kVERBOSE, ORT_WHERE, graph, insertion_edges);

  auto make_q_or_dq_inputs = [](NodeArg& data, NodeArg& scale, NodeArg* zero_point) {
    return zero_point ? InlinedVector<NodeArg*>{&data, &scale, zero_point}
                      : InlinedVector<NodeArg*>{&data, &scale};
  };

  // Create Q node that will be inserted after src_node
  auto& pre_q_nodearg = first_edge.HasGraphInputOrInitializer()
                            ? base_node_arg
                            : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_pre_q"),
                                                       nullptr);

  auto& q_to_dq_nodearg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_q_to_dq"),
                                                   nullptr);

  auto& q_node = graph.AddNode(graph.GenerateNodeName(base_name + "_q"),
                               QDQ::QOpName,
                               "Inserted by QDQPropagationTransformer",
                               // inputs
                               make_q_or_dq_inputs(pre_q_nodearg, scale_initializer_nodearg,
                                                   zp_initializer_nodearg_ptr),
                               // outputs
                               {&q_to_dq_nodearg},
                               &q_attrs,  // attributes
                               qdq_domain);

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(q_node), "Failed to set op schema for added Q node.");

  if (src_node) {
    // Remove original edges between src and dst nodes.
    for (const auto& insertion_edge : insertion_edges) {
      auto* dst_node = insertion_edge.GetMutableNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);

      if (dst_node) {
        graph.RemoveEdge(src_node->Index(), dst_node->Index(),
                         insertion_edge.src->arg_idx, insertion_edge.dst->arg_idx);
      }
    }

    optimizer_utils::DuplicateNodeAnnotation(*src_node, q_node);

    // Add edge from src to Q node.
    src_node->MutableOutputDefs()[first_edge.src->arg_idx] = &pre_q_nodearg;
    graph.AddEdge(src_node->Index(), q_node.Index(), first_edge.src->arg_idx, 0);
  }

  // Create a DQ node for each dst node and connect remaining edges.
  for (size_t edge_idx = 0; edge_idx < insertion_edges.size(); ++edge_idx) {
    const auto& insertion_edge = insertion_edges[edge_idx];
    const std::string edge_suffix = edge_idx == 0 ? "" : std::to_string(edge_idx);
    auto& post_dq_nodearg = insertion_edge.HasGraphOutput()
                                ? base_node_arg
                                : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(MakeString(base_name,
                                                                                                "_post_dq",
                                                                                                edge_suffix)),
                                                           nullptr);

    auto& dq_node = graph.AddNode(graph.GenerateNodeName(MakeString(base_name, "_dq", edge_suffix)),
                                  QDQ::DQOpName,
                                  "Inserted by QDQPropagationTransformer",
                                  // inputs
                                  make_q_or_dq_inputs(q_to_dq_nodearg, scale_initializer_nodearg,
                                                      zp_initializer_nodearg_ptr),
                                  // outputs
                                  {&post_dq_nodearg},
                                  &dq_attrs,  // attributes
                                  qdq_domain);

    if (src_node) {
      optimizer_utils::DuplicateNodeAnnotation(*src_node, dq_node);
    }

    ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(dq_node), "Failed to set op schema for added DQ node.");

    Node* dst_node = insertion_edge.GetMutableNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);

    // Add edge from Q to DQ
    graph.AddEdge(q_node.Index(), dq_node.Index(), 0, 0);

    // Add edge from DQ to dst_node
    if (dst_node) {
      dst_node->MutableInputDefs()[insertion_edge.dst->arg_idx] = &post_dq_nodearg;
      graph.AddEdge(dq_node.Index(), dst_node->Index(), 0, insertion_edge.dst->arg_idx);
    }
  }

  return Status::OK();
}

std::optional<ExtendedGraphEdge> GetPreviousEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first input (index 0)

  const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
  const auto input_edge_it = std::find_if(
      input_edges.begin(), input_edges.end(),
      [](const graph_utils::GraphEdge& edge) { return edge.dst_arg_index == 0; });

  if (input_edge_it == input_edges.end()) {
    // maybe edge from input
    return ExtendedGraphEdge::TryCreateFromInputOrInitializerToNode(graph, node, 0);
  }

  const auto& src_node = *graph.GetNode(input_edge_it->src_node);
  const auto src_node_output_edges =
      graph_utils::GraphEdge::GetNodeOutputEdges(src_node, input_edge_it->src_arg_index);
  if (!graph.IsOutput(src_node.OutputDefs()[input_edge_it->src_arg_index]) &&
      src_node_output_edges.size() == 1) {
    // single edge from previous node
    return ExtendedGraphEdge::CreateFromValidGraphEdge(*input_edge_it);
  }

  return std::nullopt;
}

std::optional<ExtendedGraphEdge> GetPreviousPropagationEdge(const Graph& graph,
                                                            const ExtendedGraphEdge& edge) {
  if (edge.HasGraphInputOrInitializer()) {
    return std::nullopt;
  }

  const auto* src_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  ORT_ENFORCE(src_node != nullptr);

  if (!CanNodePropagate(*src_node)) {
    return std::nullopt;
  }

  return GetPreviousEdge(graph, *src_node);
}

InlinedVector<ExtendedGraphEdge> GetNextEdges(const Graph& graph, const Node& node) {
  constexpr int node_output_index = 0;  // for now we can just consider the first output (index 0)
  InlinedVector<ExtendedGraphEdge> next_edges;
  const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node, static_cast<size_t>(node_output_index));

  // edges to next nodes
  for (const auto& output_edge : output_edges) {
    next_edges.push_back(ExtendedGraphEdge::CreateFromValidGraphEdge(output_edge));
  }

  // maybe edge to graph output
  auto edge_to_output = ExtendedGraphEdge::TryCreateFromNodeToOutput(graph, node, node_output_index);
  if (edge_to_output.has_value()) {
    next_edges.push_back(edge_to_output.value());
  }

  return next_edges;
}

InlinedVector<ExtendedGraphEdge> GetNextPropagationEdges(const Graph& graph,
                                                         const ExtendedGraphEdge& edge) {
  if (edge.HasGraphOutput()) {
    return {};
  }

  const auto* dst_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
  ORT_ENFORCE(dst_node != nullptr);

  if (!CanNodePropagate(*dst_node)) {
    return {};
  }

  return GetNextEdges(graph, *dst_node);
}

class GraphConstantInitializerGetter {
  const Graph& graph_;

 public:
  GraphConstantInitializerGetter(const Graph& graph) : graph_{graph} {}
  const ONNX_NAMESPACE::TensorProto* operator()(const std::string& initializer_name) const {
    return graph_utils::GetConstantInitializer(graph_, initializer_name);
  }
};

Status PropagateDQForward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const InlinedHashSet<std::string_view>& compatible_eps,
                          const logging::Logger& logger,
                          bool& modified) {
  for (auto node_index : node_indices) {
    auto* dq_node_ptr = graph.GetNode(node_index);
    if (dq_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& dq_node = *dq_node_ptr;

    if (!QDQ::MatchDQNode(dq_node) ||
        !graph_utils::IsSupportedProvider(dq_node, compatible_eps) ||
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    if (dq_node.Domain() == kMSDomain) {
      const auto* scale_type = dq_node.InputDefs()[QDQ::InputIndex::SCALE_ID]->TypeAsProto();
      const auto* output_type = dq_node.OutputDefs()[0]->TypeAsProto();
      const auto* input_type = dq_node.InputDefs()[QDQ::InputIndex::INPUT_ID]->TypeAsProto();
      // The MS quantizer cannot invert FP8, BF16, or independently typed scale/output dequantization.
      if (graph_utils::GetNodeAttribute(dq_node, "output_dtype") != nullptr ||
          scale_type == nullptr || output_type == nullptr || input_type == nullptr ||
          scale_type->tensor_type().elem_type() != output_type->tensor_type().elem_type() ||
          (scale_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto::FLOAT &&
           scale_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto::FLOAT16) ||
          input_type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN) {
        continue;
      }
    }

    bool dq_zero_point_exists = false;
    if (!QDQ::QOrDQNodeHasConstantScalarScaleAndZeroPoint(dq_node, GraphConstantInitializerGetter{graph},
                                                          dq_zero_point_exists)) {
      continue;
    }

    // Do not propagate DQ forward when its data input is a constant (graph initializer or
    // Constant op output). Propagation would insert a Q -> DQ pair after any downstream
    // reshape-like node; later passes (e.g. S8-to-U8 weight transformer) may flip the
    // existing DQ to uint8 without touching the inserted Q, causing int8 negatives to
    // be clamped to zero. See GitHub issue #28491.
    const NodeArg* dq_data_input = dq_node.InputDefs()[QDQ::InputIndex::INPUT_ID];
    const bool is_initializer_constant = graph_utils::NodeArgIsConstant(graph, *dq_data_input);
    const Node* dq_data_producer = graph.GetProducerNode(dq_data_input->Name());
    const bool is_constant_op_output = dq_data_producer != nullptr &&
                                       dq_data_producer->OpType() == "Constant" &&
                                       dq_data_producer->Domain() == kOnnxDomain;
    if (is_initializer_constant || is_constant_op_output) {
      continue;
    }

    auto& dq_scale = *dq_node.MutableInputDefs()[QDQ::InputIndex::SCALE_ID];
    auto* dq_zero_point = dq_zero_point_exists
                              ? dq_node.MutableInputDefs()[QDQ::InputIndex::ZERO_POINT_ID]
                              : nullptr;

    const InlinedVector<ExtendedGraphEdge> edges_after_dq = GetNextEdges(graph, dq_node);
    if (edges_after_dq.size() != 1) {
      continue;
    }

    // Utility function to check if any edge out of a node (e.g., Transpose) ends in a Q node.
    auto any_edge_ends_in_q = [](Graph& graph, const InlinedVector<ExtendedGraphEdge>& edges) -> bool {
      for (const auto& edge : edges) {
        const auto* edge_dst_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
        if (edge_dst_node && QDQ::MatchQNode(*edge_dst_node)) {
          return true;
        }
      }
      return false;
    };

    // Propagate DQ forward in a BFS traversal of NodeArg edges. A NodeArg "edge group" consists of one or more edges
    // that all begin at the same source node's output slot and end at a graph output or a destination node.
    // Ex: The subgraph below shows a NodeArg edge group (containing 3 edges) that begins at a
    // Transpose, ends at two destination nodes, and produces a graph output.
    //    DQ -> Transpose --+--> Sigmoid -> ...
    //                      |
    //                      +--> Slice -> ...
    //                      |
    //                      +--> graph_output
    std::queue<InlinedVector<ExtendedGraphEdge>> node_arg_edges;
    node_arg_edges.push(GetNextPropagationEdges(graph, edges_after_dq[0]));

    while (!node_arg_edges.empty()) {
      const InlinedVector<ExtendedGraphEdge> curr_edge_group = std::move(node_arg_edges.front());
      node_arg_edges.pop();

      // Skip if edge group is empty. Also, to keep things simple, we do not yet handle edge groups in which
      // one of the destination nodes is already a QuantizeLinear node. Ex:
      //    DQ -> Transpose --+--> QuantizeLinear -> ...
      //                      |
      //                      +--> Slice -> ...
      if (curr_edge_group.empty() || any_edge_ends_in_q(graph, curr_edge_group)) {
        continue;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPairs(graph, curr_edge_group, dq_scale, dq_zero_point, dq_node.Domain(),
                                         MakeQAttrsFromDQ(dq_node), dq_node.GetAttributes(), logger));
      modified = true;

      for (const auto& edge : curr_edge_group) {
        node_arg_edges.push(GetNextPropagationEdges(graph, edge));
      }
    }
  }

  return Status::OK();
}

Status PropagateQBackward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const InlinedHashSet<std::string_view>& compatible_eps,
                          const logging::Logger& logger,
                          bool& modified) {
  for (auto node_index : node_indices) {
    auto* q_node_ptr = graph.GetNode(node_index);
    if (q_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& q_node = *q_node_ptr;

    if (!QDQ::MatchQNode(q_node) ||
        !graph_utils::IsSupportedProvider(q_node, compatible_eps)) {
      continue;
    }

    bool q_zero_point_exists = false;
    if (!QDQ::QOrDQNodeHasConstantScalarScaleAndZeroPoint(q_node, GraphConstantInitializerGetter{graph},
                                                          q_zero_point_exists)) {
      continue;
    }

    auto& q_scale = *q_node.MutableInputDefs()[QDQ::InputIndex::SCALE_ID];
    auto* q_zero_point = q_zero_point_exists
                             ? q_node.MutableInputDefs()[QDQ::InputIndex::ZERO_POINT_ID]
                             : nullptr;

    const auto edge_before_q = GetPreviousEdge(graph, q_node);
    if (!edge_before_q) {
      continue;
    }

    for (auto curr_edge = GetPreviousPropagationEdge(graph, *edge_before_q);
         curr_edge.has_value();
         curr_edge = GetPreviousPropagationEdge(graph, *curr_edge)) {
      if (auto* src_node = curr_edge->GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
          src_node && QDQ::MatchDQNode(*src_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPairs(graph, InlinedVector<ExtendedGraphEdge>{*curr_edge}, q_scale, q_zero_point,
                                         q_node.Domain(), q_node.GetAttributes(), MakeDQAttrsFromQ(q_node), logger));
      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace

Status QDQPropagationTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto node_indices = gsl::make_span(graph_viewer.GetNodesInTopologicalOrder());

  for (auto node_index : node_indices) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));
  }

  const auto& compatible_eps = GetCompatibleExecutionProviders();

  ORT_RETURN_IF_ERROR(PropagateQBackward(graph, node_indices, compatible_eps, logger, modified));
  ORT_RETURN_IF_ERROR(PropagateDQForward(graph, node_indices, compatible_eps, logger, modified));

  return Status::OK();
}
}  // namespace onnxruntime
