// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/div_mul_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

/**
Transform that fuses two Div -> Mul nodes to a single Div node
when the first input to Div is 1.
1 / x1 *  x2 -> x2 / x1
 */
bool DivMulFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7, 13, 14}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7, 13, 14}) ||
      // Make sure the two nodes do not span execution providers.
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  // Check that the appropriate input to the Div node is a constant.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[0])) {
    return false;
  }

  const auto* initializer = graph_utils::GetConstantInitializer(graph, node.InputDefs()[0]->Name());
  if (!initializer) {
    return false;
  }

  int32_t data_type = initializer->data_type();
  Initializer div_A(graph, *initializer, graph.ModelPath());
  if (div_A.size() != 1) {
    return false;
  }
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      if (*div_A.data<float>() != 1.f) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      if (math::halfToFloat(div_A.data<MLFloat16>()->val) != 1.f) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      if (*div_A.data<double>() != static_cast<double>(1.f)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      if (*div_A.data<int32_t>() != static_cast<int32_t>(1)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      if (*div_A.data<int64_t>() != static_cast<int64_t>(1)) {
        return false;
      }
      break;
    default:
      return false;
  }

  if (graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  return true;
}

Status DivMulFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto& div_node = node;
  auto& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());  // get mutable next node
  const auto& div_output = div_node.OutputDefs();
  auto& mul_inputs = mul_node.MutableInputDefs();

  // get other input of mul
  const int mul_other_input_idx = mul_inputs[0] == div_output[0] ? 1 : 0;
  auto& mul_other_input = mul_inputs[mul_other_input_idx];

  // If that input is produced by a node, the edge carrying it ends at mul_node and dies with it.
  // ReplaceNodeInput rewrites the input def without adding an edge - correct only for an initializer
  // or graph input - so div_node has to be reconnected explicitly, or it would reference the value by
  // name alone and a later rule removing the producer would rewire only its edge-connected consumers.
  // div_node's input 0 is a constant initializer (SatisfyCondition), so it is free of edges.
  const Node::EdgeEnd* producer_edge = graph_utils::GetInputEdge(mul_node, mul_other_input_idx);

  graph_utils::ReplaceNodeInput(div_node, 0, *mul_other_input);
  if (producer_edge != nullptr) {
    graph.AddEdge(producer_edge->GetNode().Index(), div_node.Index(), producer_edge->GetSrcArgIndex(), 0);
  }
  // move the output definition and edges from the mul_node to the div_node and delete the mul_node
  graph_utils::FinalizeNodeFusion(graph, div_node, mul_node);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}
}  // namespace onnxruntime
