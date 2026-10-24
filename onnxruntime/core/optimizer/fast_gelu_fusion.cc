// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>
#include <optional>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// FastGelu supports limited data types.
static constexpr std::array gpu_supported_data_types{"tensor(float16)", "tensor(float)", "tensor(bfloat16)"};
static constexpr std::array cpu_supported_data_types{"tensor(float)"};

static bool IsSupportedDataType(const Node& node) {
  if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
    return optimizer_utils::IsSupportedDataType(node, cpu_supported_data_types);
  } else {
    return optimizer_utils::IsSupportedDataType(node, gpu_supported_data_types);
  }
}

static bool CheckNode(Graph& graph, const Node& node,
                      ProviderType provider, bool require_single_output) {
  return node.GetExecutionProviderType() == provider &&
         IsSupportedDataType(node) &&
         (!require_single_output || node.GetOutputEdgesCount() == 1) &&
         !graph.NodeProducesGraphOutput(node);
}

static std::optional<size_t> OtherInputIndex(const Node& node, const NodeArg& input) {
  if (node.InputDefs().size() != 2) {
    return std::nullopt;
  }

  const int input_index = optimizer_utils::IndexOfNodeInput(node, input);
  if (input_index < 0 || input_index >= 2) {
    return std::nullopt;
  }

  return 1u - onnxruntime::narrow<size_t>(input_index);
}

MatchResult FastGeluFusion::CheckFirstFormula(Graph& graph, Node& mul1_node,
                                              InlinedVector<std::reference_wrapper<Node>>& nodes_to_fuse) const {
  MatchResult matchResult{false, nullptr, nullptr};
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul1_node, "Mul", {7, 13, 14}) ||
      mul1_node.InputDefs().size() != 2 ||
      !graph_utils::IsSupportedProvider(mul1_node, GetCompatibleExecutionProviders()) ||
      mul1_node.GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(mul1_node) ||
      !IsSupportedDataType(mul1_node)) {
    return matchResult;
  }

  int32_t input_index = -1;
  constexpr float mul_val = 0.044715f;
  for (auto i = 0; i < 2; i++) {
    if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul1_node.InputDefs()[i]), mul_val, true)) {
      input_index = i;
      break;
    }
  }

  if (input_index == -1) return matchResult;

  NodeArg* gelu_without_bias_input_arg = mul1_node.MutableInputDefs()[1u - onnxruntime::narrow<size_t>(input_index)];
  nodes_to_fuse.push_back(mul1_node);

  Node& mul2_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
  auto other_input_index = OtherInputIndex(mul2_node, *mul1_node.MutableOutputDefs()[0]);
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7, 13, 14}) &&
        CheckNode(graph, mul2_node, mul1_node.GetExecutionProviderType(), true)) ||
      !other_input_index ||
      mul2_node.InputDefs()[*other_input_index]->Name() != gelu_without_bias_input_arg->Name()) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul2_node);

  Node& add1_node = *graph.GetNode(mul2_node.OutputNodesBegin()->Index());
  other_input_index = OtherInputIndex(add1_node, *mul2_node.MutableOutputDefs()[0]);

  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(add1_node, "Add", {7, 13, 14}) &&
        CheckNode(graph, add1_node, mul1_node.GetExecutionProviderType(), true)) ||
      !other_input_index ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *add1_node.InputDefs()[*other_input_index], 1.0f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(add1_node);

  Node& mul3_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul3_node, "Mul", {7, 13, 14}) &&
        CheckNode(graph, mul3_node, mul1_node.GetExecutionProviderType(), true))) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul3_node);

  other_input_index = OtherInputIndex(mul3_node, *add1_node.MutableOutputDefs()[0]);
  if (!other_input_index) return matchResult;
  const Node* p_mul3_input_node =
      graph_utils::GetInputNode(mul3_node, onnxruntime::narrow<int>(*other_input_index));
  if (p_mul3_input_node == nullptr) return matchResult;
  Node& mul4_node = const_cast<Node&>(*p_mul3_input_node);
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul4_node, "Mul", {7, 13, 14}) &&
        mul4_node.InputDefs().size() == 2 &&
        CheckNode(graph, mul4_node, mul1_node.GetExecutionProviderType(), true))) {
    return matchResult;
  }

  input_index = -1;
  constexpr float mul4_val = 0.7978845834732056f;
  for (auto i = 0; i < 2; i++) {
    if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul4_node.InputDefs()[i]), mul4_val, true)) {
      input_index = i;
      break;
    }
  }

  if (input_index == -1 ||
      mul4_node.InputDefs()[1u - onnxruntime::narrow<size_t>(input_index)]->Name() != gelu_without_bias_input_arg->Name())
    return matchResult;
  nodes_to_fuse.push_back(mul4_node);

  matchResult.matched = true;
  matchResult.gelu_without_bias_input_arg = gelu_without_bias_input_arg;
  matchResult.tanh_input_node = &mul3_node;
  return matchResult;
}

MatchResult FastGeluFusion::CheckSecondFormula(Graph& graph, Node& pow1_node,
                                               InlinedVector<std::reference_wrapper<Node>>& nodes_to_fuse) const {
  MatchResult matchResult{false, nullptr, nullptr};
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow1_node, "Pow", {7, 12, 13, 15}) ||
      pow1_node.InputDefs().size() != 2 ||
      !graph_utils::IsSupportedProvider(pow1_node, GetCompatibleExecutionProviders()) ||
      pow1_node.GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(pow1_node) ||
      !IsSupportedDataType(pow1_node)) {
    return matchResult;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(pow1_node.InputDefs()[1]), 3.0f, true)) {
    return matchResult;
  }

  NodeArg* pow_input_arg = pow1_node.MutableInputDefs()[0];
  nodes_to_fuse.push_back(pow1_node);

  Node& mul1_node = *graph.GetNode(pow1_node.OutputNodesBegin()->Index());
  auto other_input_index = OtherInputIndex(mul1_node, *pow1_node.MutableOutputDefs()[0]);
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul1_node, "Mul", {7, 13, 14}) &&
        CheckNode(graph, mul1_node, pow1_node.GetExecutionProviderType(), true)) ||
      !other_input_index ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *mul1_node.InputDefs()[*other_input_index],
                                                       0.044714998453855515f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul1_node);

  Node& add1_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
  other_input_index = OtherInputIndex(add1_node, *mul1_node.MutableOutputDefs()[0]);
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(add1_node, "Add", {7, 13, 14}) &&
        CheckNode(graph, add1_node, pow1_node.GetExecutionProviderType(), true)) ||
      !other_input_index ||
      add1_node.InputDefs()[*other_input_index]->Name() != pow_input_arg->Name()) {
    return matchResult;
  }
  nodes_to_fuse.push_back(add1_node);

  // check if pow node has Cast parent, expect Add has same Cast parent as well
  const Node* p_cast1_node = graph_utils::FirstParentByType(pow1_node, "Cast");
  if (p_cast1_node != nullptr) {
    Node& cast1_node = *graph.GetNode(p_cast1_node->Index());
    // this is fused Cast node, so expect 2 output edges
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(cast1_node, "Cast", {9, 13, 19, 21, 23, 24, 25, 28}) &&
          cast1_node.InputDefs().size() == 1 &&
          CheckNode(graph, cast1_node, pow1_node.GetExecutionProviderType(), false)) ||
        cast1_node.GetOutputEdgesCount() != 2) {
      return matchResult;
    }
    const Node* p_pow_node = graph_utils::FirstChildByType(cast1_node, "Pow");
    if (p_pow_node == nullptr || p_pow_node->Index() != pow1_node.Index()) {
      return matchResult;
    }
    const Node* p_add_node = graph_utils::FirstChildByType(cast1_node, "Add");
    if (p_add_node == nullptr || p_add_node->Index() != add1_node.Index()) {
      return matchResult;
    }
  }

  Node& mul2_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
  other_input_index = OtherInputIndex(mul2_node, *add1_node.MutableOutputDefs()[0]);
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7, 13, 14}) &&
        CheckNode(graph, mul2_node, pow1_node.GetExecutionProviderType(), true)) ||
      !other_input_index ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *mul2_node.InputDefs()[*other_input_index],
                                                       0.7978845834732056f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul2_node);

  matchResult.matched = true;
  matchResult.gelu_without_bias_input_arg = pow_input_arg;
  matchResult.tanh_input_node = &mul2_node;
  return matchResult;
}

/**
In case of ORTModule, there are extra Cast nodes exported for fp16. They should be fused into two nodes:

x --> Cast --> FastGelu

The first Cast should have been fused in CommonSubexpressionElimination transformer, thus it has 2 output edges.

+--------------------------------------------> Mul ---> Cast ----+
|                                                                |
|                                                                v
X --> Cast --> Pow --> Mul --> Add --> Mul --> Tanh --> Add --> Mul
       |                        ^
       |                        |
       +------------------------+

*/
Status FastGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;
    bool second_formula = false;
    MatchResult matchRet = CheckFirstFormula(graph, node, nodes_to_fuse);
    if (!matchRet.matched) {
      nodes_to_fuse.clear();
      matchRet = CheckSecondFormula(graph, node, nodes_to_fuse);

      if (!matchRet.matched) continue;
      second_formula = true;
    };

    Node& tanh_node = *graph.GetNode(matchRet.tanh_input_node->OutputNodesBegin()->Index());
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(tanh_node, "Tanh", {6, 13}) &&
          CheckNode(graph, tanh_node, node.GetExecutionProviderType(), true))) {
      continue;
    }

    Node& add2_node = *graph.GetNode(tanh_node.OutputNodesBegin()->Index());
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13, 14}) &&
          CheckNode(graph, add2_node, node.GetExecutionProviderType(), true))) {
      continue;
    }

    auto other_input_index = OtherInputIndex(add2_node, *tanh_node.MutableOutputDefs()[0]);
    if (!other_input_index ||
        !optimizer_utils::IsInitializerWithExpectedValue(graph, *add2_node.InputDefs()[*other_input_index], 1.0f, true)) {
      continue;
    }

    Node& mul5_node = *graph.GetNode(add2_node.OutputNodesBegin()->Index());
    // This is the output of the Gelu subgraph, we don't need check it has single edge.
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul5_node, "Mul", {7, 13, 14}) &&
          CheckNode(graph, mul5_node, node.GetExecutionProviderType(), false))) {
      continue;
    }

    other_input_index = OtherInputIndex(mul5_node, *add2_node.MutableOutputDefs()[0]);
    if (!other_input_index) continue;
    const Node* p_mul5_input_node =
        graph_utils::GetInputNode(mul5_node, onnxruntime::narrow<int>(*other_input_index));
    if (p_mul5_input_node == nullptr) continue;

    // if this is second formula and if pow node has Cast parent, expect mul5_node has Cast parent as well
    NodeArg* cast_input_arg = nullptr;
    if (second_formula) {
      const Node* p_cast1_node = graph_utils::FirstParentByType(node, "Cast");
      if (p_cast1_node != nullptr) {
        // we've done the node check in second formula for pow node
        Node& cast1_node = *graph.GetNode(p_cast1_node->Index());
        cast_input_arg = cast1_node.MutableInputDefs()[0];

        const Node* p_cast3_node = graph_utils::FirstParentByType(mul5_node, "Cast");
        if (p_cast3_node == nullptr) continue;

        Node& cast3_node = *graph.GetNode(p_cast3_node->Index());
        if (!(graph_utils::IsSupportedOptypeVersionAndDomain(cast3_node, "Cast", {9, 13, 19, 21, 23, 24, 25, 28}) &&
              cast3_node.InputDefs().size() == 1 &&
              CheckNode(graph, cast3_node, node.GetExecutionProviderType(), true))) {
          continue;
        }
        // overwrite and continue as usual
        p_mul5_input_node = graph_utils::FirstParentByType(cast3_node, "Mul");
        if (p_mul5_input_node == nullptr) continue;
        nodes_to_fuse.push_back(cast3_node);
        // keep cast1_node for reuse, its output edges will be adjusted in FinalizeNodeFusion()
      }
    }

    Node& mul6_node = const_cast<Node&>(*p_mul5_input_node);
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(mul6_node, "Mul", {7, 13, 14}) &&
          mul6_node.InputDefs().size() == 2 &&
          CheckNode(graph, mul6_node, node.GetExecutionProviderType(), false))) {
      continue;
    }

    std::optional<size_t> scale_input_index;
    for (size_t i = 0; i < 2; ++i) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul6_node.InputDefs()[i]), 0.5f, true)) {
        scale_input_index = i;
        break;
      }
    }

    if (!scale_input_index) continue;
    const size_t mul6_other_input_index = 1u - *scale_input_index;
    // check same parent for both mul6 and pow, with or without cast
    if (cast_input_arg != nullptr) {
      if (mul6_node.InputDefs()[mul6_other_input_index]->Name() != cast_input_arg->Name())
        continue;
    } else {
      if (mul6_node.InputDefs()[mul6_other_input_index]->Name() != matchRet.gelu_without_bias_input_arg->Name())
        continue;
    }

    nodes_to_fuse.insert(nodes_to_fuse.end(), {tanh_node, add2_node});
    // Preserve X * 0.5 for consumers outside the fused subgraph.
    if (mul6_node.GetOutputEdgesCount() == 1) {
      nodes_to_fuse.push_back(mul6_node);
    }
    nodes_to_fuse.push_back(mul5_node);

    auto type_info = *node.MutableOutputDefs()[0]->TypeAsProto();
    // TODO: re-use node arg of mul5 so that it is allowed to be graph output (Need modify CheckNode as well).
    auto& shape_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("fast_gelu_output"), &type_info);
    Node& fast_gelu_node = graph.AddNode(graph.GenerateNodeName("GPT2Gelu"),
                                         "FastGelu",
                                         "fused GPT2Gelu subgraphs ",
                                         std::array{matchRet.gelu_without_bias_input_arg},
                                         std::array{&shape_output}, {}, kMSDomain);

    // assign provider to this new node, provider should be same as the provider for old node.
    fast_gelu_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // move input edges to node (first in list) across to the fast_gelu_node.
    // move output definitions and output edges from mul5_node (last in list) to fast_gelu_node.
    // remove all nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_fuse, fast_gelu_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
