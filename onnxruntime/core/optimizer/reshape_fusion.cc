// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ReshapeFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_reshape = graph.GetNode(node_index);
    if (p_reshape == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reshape = *p_reshape;
    ORT_RETURN_IF_ERROR(Recurse(reshape, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reshape, "Reshape", {5, 13}) ||
        !graph_utils::IsSupportedProvider(reshape, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (ReshapeFusion::Fuse_Subgraph(reshape, graph, logger)) {
      fused_count++;
      LOGS(logger, INFO) << "Fused reshape node: " << reshape.OutputDefs()[0]->Name();
      modified = true;
    }
  }
  LOGS(logger, INFO) << "Total fused reshape node count: " << fused_count;

  return Status::OK();
}


/**
Provide check for Reshape Fusion for DistilBert. The following are subgraphs that
match the pattern for DistilBert

DistilBert reshape pattern:
             [Root]
             /   \ _ _ _ _ _ _ _
         Shape                  \
           |                  MatMul(w * w)
    Gather(indices=0)             |
            \                     |
        Unsqueeze               Add(w)
              \                  /
             Concat     _ _ _ _ /
                \      /
                Reshape

                                  - -> Shape
                                  |
Check the subgraph that matches [root] -> MatMul(w * w) -> Add(w) -> Reshape.
*/
static bool Match_Linear_Subgraph_1(Graph& graph, const Node& concat, const Node& root, const logging::Logger& logger) {
  if (!optimizer_utils::CheckOutputEdges(graph, concat, 1)) {
    return false;
  }
  auto reshape_itr = concat.OutputNodesBegin();
  if ((*reshape_itr).OpType().compare("Reshape") != 0) {
    return false;
  }
  const Node& reshape = *reshape_itr;

  std::vector<graph_utils::EdgeEndToMatch> linear_path{
    {0, 0, "Add", {7}, kOnnxDomain},
    {0, 0, "MatMul", {1, 9}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(reshape, true, linear_path, edges, logger)) {
    return false;
  }

  const Node& linear_path_add = edges[0]->GetNode();
  const Node& linear_path_matmul = edges[1]->GetNode();

  const Node* p_node_before_matmul = graph_utils::GetInputNode(linear_path_matmul, 0);
  if (p_node_before_matmul != nullptr && p_node_before_matmul->Index() != root.Index()) {
    return false;
  }

  if (linear_path_add.InputDefs().size() < 2) {
    return false;
  }
  const NodeArg& linear_path_add_b = *(linear_path_add.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, linear_path_add_b.Name(), true)) {
    return false;
  }

  if (!optimizer_utils::IsShapeKnownOnAllDims(linear_path_add_b, 1)) {
    return false;
  }
  int64_t hidden_size = linear_path_add_b.Shape()->dim(0).dim_value();

  if (!optimizer_utils::ValidateShape(*(linear_path_matmul.InputDefs()[1]), {hidden_size, hidden_size})) {
    return false;
  }

  return true;
}

static bool Match_Shape(Graph& graph, const Node& concat, const Node& shape, const NodeArg& root_input, const logging::Logger& logger) {
  const NodeArg& shape_input = *(shape.InputDefs()[0]);
  if (shape_input.Name() == root_input.Name()) {
    return true;
  }

  const ONNX_NAMESPACE::TensorShapeProto* shape_input_shape = shape_input.Shape();
  const ONNX_NAMESPACE::TensorShapeProto* root_input_shape = root_input.Shape();

  if (shape_input_shape != nullptr && root_input_shape != nullptr)
    return optimizer_utils::CompareShape(*shape_input_shape, *root_input_shape);

  const Node* p_node_before_shape = graph_utils::GetInputNode(shape, 0);
  if (p_node_before_shape == nullptr) {
    return false;
  }
  if (Match_Linear_Subgraph_1(graph, concat, *p_node_before_shape, logger)) {
    return true;
  }
  return false;
}

/**
 * Find the subgraph that matches [root] -> Shape -> Gather -> Unsqueeze.
 * If checkOneElementOnly is set to true, this function only checks if the matched subgraph produces a
 * one element output(skip the Gather input indices check).
 */
bool ReshapeFusion::Match_One_Element_Output_Subgraph_1(Graph& graph, const NodeArg& root_input, const Node& concat,
                                                        int index, std::vector<int64_t> shape_value, bool checkOneElementOnly,
                                                        const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, index, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (graph_utils::FindPath(concat, true, parent_path, edges, logger)) {
    const Node& unsqueeze = edges[0]->GetNode();
    const Node& gather = edges[1]->GetNode();
    const Node& shape = edges[2]->GetNode();

    std::vector<int64_t> axes;
    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      return false;
    }

    if (checkOneElementOnly && ReshapeFusion::Is_One_Element_Input(gather, 1)) {
      return true;
    }

    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather.InputDefs()[1]), int64_t(shape_value.size()), false)) {
      return false;
    }

    if (!Match_Shape(graph, concat, shape, root_input, logger)) {
      return false;
    }
    return true;
  }

  return false;
}

/**
 * Find the subgraph that matches [root] -> Shape -> Slice -> Squeeze. Check the inputs of slice
 * to make sure the graph produces output with exactly one element.
 */
bool ReshapeFusion::Match_One_Element_Output_Subgraph_2(Graph& graph, const NodeArg& root_input, const Node& cur_node,
                                                        int index, const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, index, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Slice", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (graph_utils::FindPath(cur_node, true, parent_path, edges, logger)) {
    const Node& slice = edges[1]->GetNode();
    const Node& shape = edges[2]->GetNode();

    const NodeArg& shape_input = *(shape.InputDefs()[0]);
    if (shape_input.Name() != root_input.Name()) {
      return false;
    }

    // Check if Slice op slices 1d array (result of shape) to one element.
    std::vector<int64_t> starts_values;
    std::vector<int64_t> ends_values;
    if (slice.GetInputEdgesCount() >= 3) {
      optimizer_utils::AppendTensorFromInitializer(graph, *(slice.InputDefs()[1]), starts_values, true);
      optimizer_utils::AppendTensorFromInitializer(graph, *(slice.InputDefs()[2]), ends_values, true);
    } else { // Support older version of Slice node
      graph_utils::GetRepeatedNodeAttributeValues<int64_t>(slice, "starts", starts_values);
      graph_utils::GetRepeatedNodeAttributeValues<int64_t>(slice, "ends", ends_values);
    }
    if (starts_values.size() != 1 || ends_values.size() != 1) {
      return false;
    }
    int64_t slice_start = starts_values[0];
    int64_t slice_end = ends_values[0];
    if (!(slice_end >= INT_MAX && slice_start == -1) && abs(slice_end - slice_start) != 1) {
      return false;
    }
    return true;
  }

  return false;
}

/**
 * Check if the i-th input of the current node contains exactly one element by checking
 * its inferred shape.
 */
bool ReshapeFusion::Is_One_Element_Input(const Node& cur_node, int index) {
  const NodeArg* cur_node_arg = cur_node.InputDefs()[index];
  // Check if the i-th argument has an inferred shape.
  const auto* input_shape = cur_node_arg->Shape();
  if (!input_shape) {
    // We need shape to be able to be certain of number of elements
    // Can't proceed with fusion
    return false;
  }

  // Check if number of elements in this input to perform binary operation is 1
  if (utils::GetTensorShapeFromTensorShapeProto(*cur_node_arg->Shape()).Size() != 1) {
    // Some dim values may be > 1 or some dim values may be missing
    // Can't proceed with fusion
    return false;
  }
  return true;
}

/**
 * Search all known patterns of one element subgraphs, which include -
 * 1. A concat input with inferred shape that can only contain one element.
 * 2. [root] -> Shape -> Gather(any 1d indice) -> Unsqueeze -> [Concat]
 * 3. [root] -> Shape -> Slice (slice to one element) -> Squeeze -> (Div/Mul) -> Unsqueeze -> [Concat]
 *                                                                      |
 *                                                           (one element output node)
 * If one of the above pattern is found, return true. Return false otherwise.
 */
bool ReshapeFusion::Is_One_Element_Output_Subgraph(Graph& graph, const NodeArg& root_input, const Node& concat,
                                                   int index, std::vector<int64_t> shape_value, const logging::Logger& logger) {
  // Match "1-element subgraph from inferred shape -> concat" or "Shape -> Gather(1d indice) -> Unsqueeze -> [Concat]"
  if (ReshapeFusion::Is_One_Element_Input(concat, index) ||
      ReshapeFusion::Match_One_Element_Output_Subgraph_1(graph, root_input, concat, index, shape_value, true, logger)) {
    return true;
  }

  std::vector<graph_utils::EdgeEndToMatch> div_path{
      {0, index, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Div", {7, 13}, kOnnxDomain}};

  std::vector<graph_utils::EdgeEndToMatch> mul_path{
      {0, index, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Mul", {7, 13}, kOnnxDomain}};

  std::vector<graph_utils::EdgeEndToMatch> unsqueeze_path{
      {0, index, "Unsqueeze", {1, 11, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (graph_utils::FindPath(concat, true, div_path, edges, logger) ||
      graph_utils::FindPath(concat, true, mul_path, edges, logger) ||
      graph_utils::FindPath(concat, true, unsqueeze_path, edges, logger)) {
    const Node& unsqueeze = edges[0]->GetNode();
    std::vector<int64_t> axes;
    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      return false;
    }
    // Unsqueeze_path is found, check for "one-element subgraph -> concat" or "shape -> slice -> squeeze ->
    // unsqueeze" to make sure the path produces one element output.
    if (edges.size() == 1) {
      if (ReshapeFusion::Is_One_Element_Input(unsqueeze, 0) ||
          ReshapeFusion::Match_One_Element_Output_Subgraph_2(graph, root_input, unsqueeze, 0, logger)) {
        return true;
      }
      return false;
    }
    const Node& binary_node = edges[1]->GetNode();

    // Check if each of two inputs of the binary node has exactly one element.
    auto input_count = binary_node.InputArgCount().front();

    for (int i = 0; i < input_count; ++i) {
      // For each input, look for "one-element subgraph -> concat" or "shape -> slice -> squeeze" path for
      // a potential match.
      if (!ReshapeFusion::Is_One_Element_Input(binary_node, i) &&
          !ReshapeFusion::Match_One_Element_Output_Subgraph_2(graph, root_input, binary_node, i, logger)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

/**
Apply Reshape Fusion. The following are subgraphs before and after fusion:
(a[] and b[] are int64[] constant initializers; Concat may have any number of arguments,
each of which is a constant initializer or a Shape->Gather->Unsqueeze chain with the
index corresponding to the index of the argument, or a custom subgraph in which nodes
have only one output edge. Note the resulting shape value should contain no more than one
value of -1.

Before fusion:
   [Sub-graph    Root]
    |        /                  \
    |    Shape                   Shape
    |       |                      |
    |    Gather(indices=0)  a[]   Gather(indices=2)  b[] or subgraph
    |       \              /             /             /
    |   Unsqueeze         /        Unsqueeze          /
    |         \          /  ___________/             /
    |          \        /  / _______________________/
    |           \      /  / /
     \            Concat
      \          /
         Reshape

After fusion:
    [Sub-graph Root]   (Constant Initializer)
                  \         [0, a, 0, b]
                   \        /
                    Reshape
*/
bool ReshapeFusion::Fuse_Subgraph(Node& reshape, Graph& graph, const logging::Logger& logger) {
  // The root could be either a graph input or a node so use node arg to compare.
  const NodeArg& root_input = *(reshape.InputDefs()[0]);

  const Node* p_concat = graph_utils::GetInputNode(reshape, 1);
  if (nullptr == p_concat) {
    return false;
  }
  const Node& concat = *p_concat;

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {1, 4, 11, 13})) {
    return false;
  }

  auto concat_input_count = concat.InputArgCount().front();
  if (!optimizer_utils::CheckOutputEdges(graph, concat, 1)) {
    return false;
  }

  // Loop through the inputs of concat node to calculate the shape_value for a potential reshape fusion.
  std::vector<int64_t> shape_value;
  shape_value.reserve(concat_input_count);

  for (int i = 0; i < concat_input_count; ++i) {
    // First check if the i-th argument is a constant initializer.
    if (optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[i]), shape_value, true)) {
      continue;
    }

    // Try to find a known pattern that produces one element output
    bool matched = ReshapeFusion::Match_One_Element_Output_Subgraph_1(graph, root_input, concat, i, shape_value, false, logger);
    if (matched) {
      shape_value.push_back(0);
      // We have matched the pattern for this input into Concat
      // Proceed to the next input
      continue;
    }

    // If we haven't been able to match the pattern, check if this is a candidate for subgraph pattern
    // fusion. For this input to be a candidate, the number of elements in the input tensor to Concat
    // has to be 1.
    // Try to find path [Root] --> Shape --> Slice(one element slice) --> (Mul/Div) --> Squeeze
    // --> Unsqueeze (axes=0) --> Concat
    matched = ReshapeFusion::Is_One_Element_Output_Subgraph(graph, root_input, concat, i, shape_value, logger);
    if (matched) {
      // This node has met all required criteria thus far.
      // This node could lead to a potential subgraph pattern fusion.
      shape_value.push_back(-1);
      continue;
    }
    return false;
  }

  // Check how many -1 are there in shape_value.
  // -1s may be contributed by multiple subgraph pattern fusions
  // or from values in const initializers (as inputs) to the Concat node.
  // Only one value of -1 is legal in the shape initializer to the Reshape node,
  // and hence we can't proceed with the fusion if we do encounter multiple -1s
  int subgraph_cnt = 0;
  for (auto it = shape_value.begin(); it < shape_value.end(); ++it) {
    if ((*it) == -1) {
      if (++subgraph_cnt > 1) {
        // If more than one "-1" value is present in shape_value, return false to exit current fusion.
        return false;
      }
    }
  }

  // Create an initializer with the same name as the concat node output, and replace the concat node
  const auto& new_initializer_name = concat.OutputDefs()[0]->Name();
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, concat, new_initializer_name, logger)) {
    LOGS(logger, WARNING) << "Cannot replace concat node with initializer:" << new_initializer_name;
    return false;
  }
  const auto* shape_def = concat.OutputDefs()[0];
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  shape_initializer_proto.set_name(shape_def->Name());
  shape_initializer_proto.add_dims(static_cast<int64_t>(shape_value.size()));
  shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer_proto.set_raw_data(shape_value.data(), shape_value.size() * sizeof(int64_t));
  auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);

  // Safely remove concat parent nodes which have only one output
  LOGS(logger, WARNING) << "Replacing subgraph for :" << concat.Name();
  for (int i = 0; i < concat_input_count; ++i) {
    const Node* p_cur_node = graph_utils::GetInputNode(concat, i);
    if (p_cur_node != nullptr) {
      graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_cur_node);
    }
  }

  if (!graph_utils::ReplaceNodeWithInitializer(graph, *graph.GetNode(concat.Index()), new_node_arg)) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime
