// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "orttraining/core/graph/loss_func/softmax_cross_entropy.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs SoftmaxCrossEntropy::operator()(
    const Graph& graph,
    const LossFunctionInfo& loss_func_info) {
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2, " Invalid loss_func_info for SoftmaxCrossEntropy.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];
  const std::string& loss_name = loss_func_info.loss_name;
  const std::string& prob_name = prediction_name + "_probability";

  GraphAugmenter::GraphDefs graph_defs;
  graph_defs.AddGraphInputs({label_name});
  graph_defs.AddGraphOutputs({loss_name});
  std::vector<NodeDef> new_nodes;

  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr, "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = graph_defs.CopyTypeProto(prediction_arg);

    new_nodes.emplace_back(NodeDef(OpDef("SoftmaxCrossEntropy", kMSDomain, 1),  // Op
                                   {ArgDef(prediction_name),
                                    ArgDef(label_name, label_type_proto)},  // Inputs
                                   {ArgDef(loss_name, graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
                                    ArgDef(prob_name)},  // Outputs
                                   NodeAttributes(),
                                   "SoftmaxCrossEntropy"  // name
                                   ));
  }

  graph_defs.AddNodeDefs(new_nodes);
  return graph_defs;
}

GraphAugmenter::GraphDefs SparseSoftmaxCrossEntropy::operator()(
    const Graph& graph,
    const LossFunctionInfo& loss_func_info) {
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2 || args.size() == 3, " Invalid loss_func_info for SparseSoftmaxCrossEntropy.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];
  const std::string& loss_name = loss_func_info.loss_name;
  const std::string& prob_name = prediction_name + "_probability";

  GraphAugmenter::GraphDefs graph_defs;
  graph_defs.AddGraphInputs({label_name});
  graph_defs.AddGraphOutputs({loss_name});
  std::vector<NodeDef> new_nodes;

  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr, "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = GetSparseTypeProto(prediction_arg,
                                                     ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                     graph_defs);

    if (args.size() == 3) {
      const std::string& weight_name = args[2];
      TypeProto* weight_type_proto = GetSparseTypeProto(prediction_arg,
                                                        ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                                        graph_defs);
      new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",  // Op
                                     {ArgDef(prediction_name),
                                      ArgDef(label_name, label_type_proto),
                                      ArgDef(weight_name, weight_type_proto)},  // Inputs
                                     {ArgDef(loss_name),
                                      ArgDef(prob_name, prediction_arg->TypeAsProto())},  // Outputs
                                     NodeAttributes(),
                                     "SoftmaxCrossEntropy"  // name
                                     ));

      graph_defs.AddGraphInputs({weight_name});
    } else {
      new_nodes.emplace_back(NodeDef("SparseSoftmaxCrossEntropy",  // Op
                                     {ArgDef(prediction_name),
                                      ArgDef(label_name, label_type_proto)},  // Inputs
                                     {ArgDef(loss_name),
                                      ArgDef(prob_name, prediction_arg->TypeAsProto())},  // Outputs
                                     NodeAttributes(),
                                     "SoftmaxCrossEntropy"  // name
                                     ));
    }
  }

  graph_defs.AddNodeDefs(new_nodes);
  return graph_defs;
}

GraphAugmenter::GraphDefs SoftmaxCrossEntropyLoss::operator()(
    const Graph& graph,
    const LossFunctionInfo& loss_func_info) {
  const VectorString& args = loss_func_info.loss_builder_args;
  ORT_ENFORCE(args.size() == 2 || args.size() == 3, " Invalid loss_func_info for SoftmaxCrossEntropyLoss.");
  const std::string& prediction_name = args[0];
  const std::string& label_name = args[1];
  const std::string& loss_name = loss_func_info.loss_name;
  const std::string& prob_name = prediction_name + "_probability";

  GraphAugmenter::GraphDefs graph_defs;
  graph_defs.AddGraphInputs({label_name});
  graph_defs.AddGraphOutputs({loss_name});
  std::vector<NodeDef> new_nodes;

  {
    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr, "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = GetSparseTypeProto(prediction_arg,
                                                     ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                     graph_defs);

    if (args.size() == 3) {
      const std::string& weight_name = args[2];
      TypeProto* weight_type_proto = graph_defs.CreateTypeProto();
      weight_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      weight_type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->CopyFrom(
          prediction_arg->TypeAsProto()->tensor_type().shape().dim()[1]);

      new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",  // Op
                                     {ArgDef(prediction_name),
                                      ArgDef(label_name, label_type_proto),
                                      ArgDef(weight_name, weight_type_proto)},  // Inputs
                                     {ArgDef(loss_name),
                                      ArgDef(prob_name, prediction_arg->TypeAsProto())},  // Outputs
                                     NodeAttributes(),
                                     "SoftmaxCrossEntropy"  // name
                                     ));
      graph_defs.AddGraphInputs({weight_name});
    } else {
      new_nodes.emplace_back(NodeDef("SoftmaxCrossEntropyLoss",  // Op
                                     {ArgDef(prediction_name),
                                      ArgDef(label_name, label_type_proto)},  // Inputs
                                     {ArgDef(loss_name),
                                      ArgDef(prob_name, prediction_arg->TypeAsProto())},  // Outputs
                                     NodeAttributes(),
                                     "SoftmaxCrossEntropy"  // name
                                     ));
    }
  }

  graph_defs.AddNodeDefs(new_nodes);
  return graph_defs;
}

}  // namespace training
}  // namespace onnxruntime
