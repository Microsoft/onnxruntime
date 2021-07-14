// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)"};
// Default epsilon
static const float DEFAULT_LAYERNORM_EPSILON = 1e-5f;

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

/**
Layer Normalization will fuse LayerNormalization into one node :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+
It also handles cases of duplicated sub nodes exported from older version of PyTorch :
+---------------------+
|                     v
|          +-------> Sub ---------------------------------------------+
|          |                                                          |
|          |                                                          v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
|                     ^
|                     |
+---------------------+

In recent pytorch, Cast nodes may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type 
due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                                        ^
                      |                                                        |
                      +--------------------------------------------------------+
+---------------------+       Cast
|                     |        |
|                     v        v
X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                                ^
                      |                                                |
                      +------------------------------------------------+
*/
Status LayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_reduce_mean = graph.GetNode(node_index);
    if (p_reduce_mean == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reduce_mean_node = *p_reduce_mean;
    ORT_RETURN_IF_ERROR(Recurse(reduce_mean_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1, 11, 13}) ||
        !graph_utils::IsSupportedProvider(reduce_mean_node, GetCompatibleExecutionProviders()) ||
        (reduce_mean_node.GetOutputEdgesCount() != 1 && reduce_mean_node.GetOutputEdgesCount() != 2) ||
        graph.NodeProducesGraphOutput(reduce_mean_node) ||
        !IsSupportedDataType(reduce_mean_node)) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean_node);

    // Loop through the children of current "ReduceMean" node. See if they match ["Sub"] or ["Sub", "Sub"]
    int subCnt = 0;
    const Node* p_sub_node = nullptr;
    const Node* p_sub_node_dup = nullptr;
    for (auto iter = reduce_mean_node.OutputNodesBegin(); iter != reduce_mean_node.OutputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Sub") == 0) {
        if (subCnt == 0) {
          p_sub_node = &(*iter);
        } else {
          p_sub_node_dup = &(*iter);
        }
        subCnt++;
      } else {
        // doesn't match layer norm pattern. break.
        subCnt = -1;
        break;
      }
    }

    if (subCnt != 1 && subCnt != 2) {
      continue;
    }

    Node& sub_node = *graph.GetNode(p_sub_node->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node, "Sub", {7, 13, 14}) ||
        sub_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sub_node, subCnt == 1 ? 2u : 1u) ||
        !IsSupportedDataType(sub_node)) {
      continue;
    }
    nodes_to_remove.push_back(sub_node);

    // Find the "Div" node after "Sub".
    const Node* p_div = nullptr;
    p_div = graph_utils::FirstChildByType(sub_node, "Div");

    // Find the sub_dup node if exist
    if (p_sub_node_dup != nullptr) {
      Node& sub_node_dup = *graph.GetNode(p_sub_node_dup->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node_dup, "Sub", {7, 13, 14}) ||
          sub_node_dup.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, sub_node, 1) ||
          !IsSupportedDataType(sub_node_dup)) {
        continue;
      }
      nodes_to_remove.push_back(sub_node_dup);
      // Find Div node after the duplicated sub node if it's not found after the first sub node.
      if (p_div == nullptr) {
        p_div = graph_utils::FirstChildByType(sub_node_dup, "Div");
      }
    }

    if (p_div == nullptr) {
      continue;
    }
    Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13, 14}) ||
        div_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1) ||
        !IsSupportedDataType(div_node)) {
      continue;
    }
    nodes_to_remove.push_back(div_node);

    // Traceback the div node to find sqrt --> div
    const Node* p_sqrt = graph_utils::FirstParentByType(div_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        sqrt_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) ||
        !IsSupportedDataType(sqrt_node) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    // Traceback the sqrt node to find add --> sqrt
    Node& add2_node = *graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13, 14}) ||
        add2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add2_node, 1) ||
        !IsSupportedDataType(add2_node)) {
      continue;
    }
    nodes_to_remove.push_back(add2_node);
    // Traceback the add node to find reduceMean --> add
    const Node* p_reduce_mean2 = nullptr;

    p_reduce_mean2 = graph_utils::FirstParentByType(add2_node, "ReduceMean");
    if (p_reduce_mean2 == nullptr) {
      continue;
    }
    Node& reduce_mean2_node = *graph.GetNode(p_reduce_mean2->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean2_node, "ReduceMean", {1, 11, 13}) ||
        reduce_mean2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean2_node, 1) ||
        !IsSupportedDataType(reduce_mean2_node) ||
        reduce_mean2_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean2_node);

    // Traceback the reduceMean node to find pow --> reduceMean
    Node& pow_node = *graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13}) ||
        pow_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1) ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    // check if Cast node exists: either between sub and pow, or as second input to pow
    const Node* p_cast_node = graph_utils::FirstParentByType(pow_node, "Cast");
    if (p_cast_node != nullptr) {
      Node& cast_node = *graph.GetNode(p_cast_node->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13}) ||
          cast_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, cast_node, 1)) {
        continue;
      }
      nodes_to_remove.push_back(cast_node);

      // Traceback from the last node in vector to find sub --> pow  or  sub --> cast
      const Node* p_sub2_node = graph_utils::FirstParentByType(nodes_to_remove.back(), "Sub");
      if (p_sub2_node != nullptr) {
        // Cast is between Sub and Pow
        if (p_sub2_node != p_sub_node && p_sub2_node != p_sub_node_dup || !IsSupportedDataType(cast_node)) {
          continue;
        }
      }
    }

    // div --> mul
    Node& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        mul_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node, 1) ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // mul --> add
    // Need not check output edges of last node since they will be moved to fused node.
    Node& last_add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(last_add_node, "Add", {7, 13, 14}) ||
        last_add_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(last_add_node)) {
      continue;
    }
    nodes_to_remove.push_back(last_add_node);

    // get axes attributes
    const onnxruntime::NodeAttributes& attributes = reduce_mean_node.GetAttributes();
    std::vector<int64_t> axes_values;
    if (attributes.find("axes") != attributes.end()) {
      axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    }

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    NodeArg* scale = nullptr;
    NodeArg* bias = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(mul_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, mul_node.MutableInputDefs()[i])) {
#ifdef ENABLE_TRAINING
        if (axes_values.empty() ||
            mul_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
          scale = mul_node.MutableInputDefs()[i];
        }
#else
        // Scale must be 1d.
        if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          scale = mul_node.MutableInputDefs()[i];
        }
#endif
      }
    }

    for (size_t i = 0; i < last_add_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(last_add_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, last_add_node.MutableInputDefs()[i])) {
#ifdef ENABLE_TRAINING
        if (axes_values.empty() ||
            last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
          bias = last_add_node.MutableInputDefs()[i];
        }
#else
        // Bias must be 1d.
        if (last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          bias = last_add_node.MutableInputDefs()[i];
        }
#endif
      }
    }
    if (scale == nullptr || bias == nullptr) {
      continue;
    }

    // Scale and bias must have the same shape.
    bool same_dim = true;
    for (int i = 0; i < scale->Shape()->dim_size(); i++) {
      if (scale->Shape()->dim(i).dim_value() != bias->Shape()->dim(i).dim_value()) {
        same_dim = false;
        break;
      }
    }
    if (!same_dim)
      continue;

    const std::vector<NodeArg*> layer_norm_input_defs{reduce_mean_node.MutableInputDefs()[0], scale, bias};
    Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName("LayerNormalization"),
                                          "LayerNormalization",
                                          "fused LayerNorm subgraphs ",
                                          layer_norm_input_defs,
                                          {}, {}, kOnnxDomain);

    // Get constant "epsilon" from "Add2" node if available. Else, default value will be used.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, add2_node.MutableInputDefs()[1]->Name());
    if (tensor_proto != nullptr &&
        tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      Initializer initializer{*tensor_proto, graph.ModelPath()};
      layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
    } else {
      layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

#ifdef ENABLE_TRAINING
    // add two extra output defs, so we have 3 output defs that match what gradient builder expected
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean"), nullptr));
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std"), nullptr));
#endif

    modified = true;
  }
  return Status::OK();
}

/**
Layer Normalization will fuse LayerNormalization into one node :

X --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
|                                              ^
|                                              |
+----------------------------------------------+
Additional FP16 patterns supported if allow_precision_change_ is true:

X --> Cast1 --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast2 --> Mul
        |                                               ^                  ^
        |                                               |                  |
        +-----------------------------------------------+                Scale

In this pattern, we change the Mul to compute in the same type as Cast1 instead of Cast2,
and are able to fuse the graph. We might need to add a Cast to the Scale input 
of Mul to match the type of Cast1. We add the Cast2 after the fused Layer Norm node.
This results in the graph:

X ------> Cast1 --> SimplifiedLayerNormalization --> Cast
                              ^
Scale --> Cast ---------------|
*/
Status SimplifiedLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_pow = graph.GetNode(node_index);
    if (p_pow == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& pow_node = *p_pow;
    ORT_RETURN_IF_ERROR(Recurse(pow_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13}) ||
        !graph_utils::IsSupportedProvider(pow_node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1) ||
        graph.NodeProducesGraphOutput(pow_node) ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    const Node* p_reduce_mean = nullptr;

    p_reduce_mean = graph_utils::FirstChildByType(pow_node, "ReduceMean");
    if (p_reduce_mean == nullptr) {
      continue;
    }
    Node& reduce_mean_node = *graph.GetNode(p_reduce_mean->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1, 11, 13}) ||
        reduce_mean_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean_node, 1) ||
        !IsSupportedDataType(reduce_mean_node) ||
        reduce_mean_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean_node);

    const Node* p_add = graph_utils::FirstChildByType(reduce_mean_node, "Add");
    if (p_add == nullptr) {
      continue;
    }
    Node& add_node = *graph.GetNode(p_add->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13, 14}) ||
        add_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add_node, 1) ||
        !IsSupportedDataType(add_node)) {
      continue;
    }
    nodes_to_remove.push_back(add_node);

    const Node* p_sqrt = graph_utils::FirstChildByType(add_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        sqrt_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) ||
        !IsSupportedDataType(sqrt_node) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    const Node* p_div = graph_utils::FirstChildByType(sqrt_node, "Div");
    if (p_div == nullptr) {
      continue;
    }
    Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13, 14}) ||
        div_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1) ||
        !IsSupportedDataType(div_node)) {
      continue;
    }
    nodes_to_remove.push_back(div_node);

    const NodeArg* p_div_input = div_node.MutableInputDefs()[0];
    const NodeArg* p_pow_input = pow_node.MutableInputDefs()[0];

    if (p_pow_input == nullptr || p_div_input == nullptr) {
      continue;
    }
    bool cast_1_present = false;
    int64_t cast_1_to_attr;
    // check if there are Casts as input to the Pow and Div
    if (p_div_input == p_pow_input) {
      const Node* p_pow_input_node = graph_utils::GetInputNode(pow_node, 0);
      if (allow_precision_change_ && p_pow_input_node != nullptr) {
        Node& pow_input_node = *graph.GetNode(p_pow_input_node->Index());

        // If input to Pow is a Cast, and the Cast has 2 consumers only (Pow, Div)
        if (graph_utils::IsSupportedOptypeVersionAndDomain(pow_input_node, "Cast", {9, 13}) &&
            pow_input_node.GetExecutionProviderType() == pow_node.GetExecutionProviderType() &&
            optimizer_utils::CheckOutputEdges(graph, pow_input_node, 2)) {
          // get the 'to' attribute of Cast
          int64_t pcast_to;
          const onnxruntime::NodeAttributes& pcast_attributes = pow_input_node.GetAttributes();
          NodeAttributes::const_iterator pcast_to_attr = pcast_attributes.find("to");
          if (pcast_to_attr != pcast_attributes.end()) {
            pcast_to = static_cast<int64_t>(pcast_to_attr->second.i());
          } else {
            continue;
          }

          cast_1_present = true;
          cast_1_to_attr = pcast_to;
        }  // end Cast check
      }    // end allow_precision_change_
    } else {
      continue;
    }

    // div --> mul or div --> cast --> mul
    Node* next_node = graph.GetNode(div_node.OutputNodesBegin()->Index());
    Node* p_cast_2 = nullptr;
    if (allow_precision_change_ &&
        graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Cast", {9, 13}) &&
        optimizer_utils::CheckOutputEdges(graph, *next_node, 1)) {
      p_cast_2 = next_node;
      next_node = graph.GetNode(p_cast_2->OutputNodesBegin()->Index());
      nodes_to_remove.push_back(*p_cast_2);
    }

    Node& mul_node = *next_node;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        mul_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // get axes attributes
    const onnxruntime::NodeAttributes& attributes = reduce_mean_node.GetAttributes();
    std::vector<int64_t> axes_values;
    if (attributes.find("axes") != attributes.end()) {
      axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    }

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    NodeArg* scale = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(mul_node.MutableInputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, mul_node.MutableInputDefs()[i])) {
#ifdef ENABLE_TRAINING
        if (axes_values.empty() ||
            mul_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
          scale = mul_node.MutableInputDefs()[i];
        }
#else
        // Scale must be 1d.
        if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
          scale = mul_node.MutableInputDefs()[i];
        }
#endif
      }
    }

    if (scale == nullptr) {
      continue;
    }

    std::vector<NodeArg*> layer_norm_input_defs{pow_node.MutableInputDefs()[0]};
    // There was a cast at input, so make sure the 'to' type for input casts
    // is the same as type for scale input. If not, add a Cast.
    if (allow_precision_change_ && cast_1_present) {
      // get type of activation input
      ONNX_NAMESPACE::TensorProto_DataType cast_1_type = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(cast_1_to_attr);
      const ONNX_NAMESPACE::TypeProto* casted_type = DataTypeImpl::TensorTypeFromONNXEnum(cast_1_type)->GetTypeProto();
      // get type of scale input and compare to activation input type
      if (scale->Type() != nullptr &&
          DataTypeImpl::TypeFromProto(*scale->TypeAsProto()) != DataTypeImpl::TypeFromProto(*casted_type)) {
        std::string node_name = graph.GenerateNodeName("Cast_Scale");

        auto* casted_scale = &graph.GetOrCreateNodeArg(node_name, casted_type);

        std::vector<NodeArg*> input_defs = {scale};
        std::vector<NodeArg*> output_defs = {casted_scale};

        auto& cast_node = graph.AddNode(node_name, "Cast", "cast scale of layer norm", input_defs, output_defs);
        cast_node.AddAttribute("to", cast_1_to_attr);
        cast_node.SetExecutionProviderType(pow_node.GetExecutionProviderType());
        layer_norm_input_defs.push_back(casted_scale);
      } else {  // scale type is same as casted type
        layer_norm_input_defs.push_back(scale);
      }
    } else {  // cast1 is not present or allow_precision_change_ false
      layer_norm_input_defs.push_back(scale);
    }

    Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName("SimplifiedLayerNormalization"),
                                          "SimplifiedLayerNormalization",
                                          "fused LayerNorm subgraphs ",
                                          layer_norm_input_defs,
                                          {}, {}, kOnnxDomain);

    // Get constant "epsilon" from "Add" node if available. Else, default value will be used.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, add_node.MutableInputDefs()[1]->Name());
    if (tensor_proto != nullptr &&
        tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      Initializer initializer{*tensor_proto, graph.ModelPath()};
      layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
    } else {
      layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    if (allow_precision_change_ && p_cast_2 != nullptr) {
      ONNX_NAMESPACE::TensorProto_DataType cast_1_type = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(cast_1_to_attr);
      const ONNX_NAMESPACE::TypeProto* casted_type = DataTypeImpl::TensorTypeFromONNXEnum(cast_1_type)->GetTypeProto();
      NodeArg* LN_output = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("layer_norm_out"), casted_type);
      layer_norm_node.MutableOutputDefs().push_back(LN_output);

      Node& cast_ln_node = graph.AddNode(graph.GenerateNodeName("Cast"),
                                         "Cast",
                                         "cast output of layer norm",
                                         {LN_output},
                                         {});

      auto cast_2_to_attr = p_cast_2->GetAttributes().find("to")->second.i();
      cast_ln_node.AddAttribute("to", cast_2_to_attr);
      cast_ln_node.SetExecutionProviderType(pow_node.GetExecutionProviderType());

      graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node, cast_ln_node);
    } else {
      // move input edges to add (first in list) across to the layer_norm_node.
      // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
      // remove all the other nodes.
      graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);
    }

#ifdef ENABLE_TRAINING
    // add one extra output def, so we have 2 output defs that match what gradient builder expected
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std"), nullptr));
#endif

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
