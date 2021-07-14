// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/embed_layer_norm_bias_gelu_fusion.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace {

}  // namespace

// TODO(kreeger): Add ASCII art to document fusion process:
#pragma warning(disable: 4100)
Status EmbedLayerNormBiasGeluFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {


  std::cerr << "Hello From EmbedLayerNormBiasGeluFusion!\n";

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_layer_norm = graph.GetNode(node_index);
    if (p_layer_norm == nullptr) {
      continue;  // we removed the node as part of an earlier fusion
    }

    Node& skip_layer_norm_node = *p_layer_norm;
    ORT_RETURN_IF_ERROR(Recurse(
        skip_layer_norm_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(skip_layer_norm_node,
                                                        "SkipLayerNormalization",
                                                        {1},
                                                        kMSDomain) ||
        !graph_utils::IsSupportedProvider(skip_layer_norm_node,
                                          GetCompatibleExecutionProviders())) {
      continue;
    }

    // Loop through children and ensure children match:
    //   MatMul -> BiasGelu -> MatMul

    const Node* p_node = nullptr;
    p_node = graph_utils::FirstChildByType(skip_layer_norm_node, "MatMul");
    if (p_node == nullptr) {
      continue;
    }
    Node& matmul_child_1 = *graph.GetNode(p_node->Index());

    p_node = graph_utils::FirstChildByType(matmul_child_1, "BiasGelu");
    if (p_node == nullptr) {
      continue;
    }
    Node& bias_gelu_child = *graph.GetNode(p_node->Index());
    
    p_node = graph_utils::FirstChildByType(bias_gelu_child, "MatMul");
    if (p_node == nullptr) {
      continue;
    }
    Node& matmul_child_2 = *graph.GetNode(p_node->Index());

    // New Node Inputs:
    // 0: SLN Input 0 (input)
    // 1: SLN Input 1 (skip)
    // 2: SLN Input 2 (gamma)
    // 3: SLN Input 3 (beta)
    // 4: SLN Input 4 (bias)
    // 5: MatMul #1 Input 1
    // 6: BiasGelu Input 1
    // 7: MatMul #2 Input 1
    std::vector<NodeArg*> new_input_defs;
    new_input_defs.resize(8);
    new_input_defs[0] = skip_layer_norm_node.MutableInputDefs()[0];
    new_input_defs[1] = skip_layer_norm_node.MutableInputDefs()[1];
    new_input_defs[2] = skip_layer_norm_node.MutableInputDefs()[2];
    new_input_defs[3] = skip_layer_norm_node.MutableInputDefs()[3];
    new_input_defs[4] = skip_layer_norm_node.MutableInputDefs()[4];
    new_input_defs[5] = matmul_child_1.MutableInputDefs()[1];
    new_input_defs[6] = bias_gelu_child.MutableInputDefs()[1];
    new_input_defs[7] = matmul_child_2.MutableInputDefs()[1];

    // New Node Outputs:
    // 0: MatMul #2 Output 0
    std::vector<NodeArg*> new_output_defs;
    new_output_defs.push_back(matmul_child_2.MutableOutputDefs()[0]);

    Node& new_node = graph.AddNode(
        /*name=*/graph.GenerateNodeName("EmbedLayerNormBiasGelu"),
        /*op_type=*/"EmbedLayerNormBiasGelu",
        /*description=*/"fused SkipLayerNorm subgraphs",
        /*input_args=*/new_input_defs,
        /*output_args=*/new_output_defs,
        /*attributes=*/nullptr,
        /*domain=*/kMSDomain);

    // TODO - stick in attributes here. They are not presently in the graph.

    // TODO - note that EP should be only CPU.

    std::vector<std::reference_wrapper<Node>> nodes_to_remove;
    nodes_to_remove.push_back(skip_layer_norm_node);
    nodes_to_remove.push_back(matmul_child_1);
    nodes_to_remove.push_back(bias_gelu_child);
    nodes_to_remove.push_back(matmul_child_2);

    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, new_node);
    
    modified = true;
  }

  return Status::OK();
}
#pragma warning(default: 4100)

}  // namespace onnxruntime

