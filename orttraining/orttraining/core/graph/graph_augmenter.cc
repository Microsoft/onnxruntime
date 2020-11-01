// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/graph_augmenter.h"

#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace training {

namespace {
Status AddToExistingNodeArgs(
    const std::string& addition_context,
    const Graph& graph,
    const std::vector<std::string>& new_nodearg_names,
    const std::vector<const NodeArg*>& existing_nodeargs,
    bool is_duplicate_an_error,
    std::vector<const NodeArg*>& nodeargs) {
  std::unordered_set<const NodeArg*> nodeargs_set(existing_nodeargs.begin(), existing_nodeargs.end());
  nodeargs = existing_nodeargs;
  for (const auto& new_nodearg_name : new_nodearg_names) {
    const auto* new_nodearg = graph.GetNodeArg(new_nodearg_name);
    ORT_RETURN_IF_NOT(
        new_nodearg,
        addition_context, " - failed to find NodeArg by name: ", new_nodearg_name);

    if (nodeargs_set.find(new_nodearg) != nodeargs_set.end()) {
      ORT_RETURN_IF(
          is_duplicate_an_error,
          addition_context, " - error - attempted to add a duplicate NodeArg: ", new_nodearg_name);
      LOGS_DEFAULT(INFO)
          << addition_context << " - skipping addition of duplicate NodeArg: " << new_nodearg_name;
      continue;
    }

    nodeargs_set.emplace(new_nodearg);
    nodeargs.emplace_back(new_nodearg);
  }

  return Status::OK();
};
}  // namespace

Status GraphAugmenter::AugmentGraph(Graph& graph,
                                    const GraphDefs& graph_element_defs,
                                    const std::unordered_set<std::string>* p_initializer_names_to_preserve) {
  // Add new initializers to the graph. - no op if it already exists
  for (const auto& tensor_proto : graph_element_defs.Initializers()) {
    const ONNX_NAMESPACE::TensorProto* exist_initializer = nullptr;
    if (!graph.GetInitializedTensor(tensor_proto.name(), exist_initializer)) {
      graph.AddInitializedTensor(tensor_proto);
      graph.GetOrCreateNodeArg(tensor_proto.name(), nullptr);
    }
  }

  // Add new nodes to the graph.
  for (const auto& node_def : graph_element_defs.NodeDefs()) {
    std::vector<NodeArg*> input_args, output_args;

    for (const auto& arg : node_def.input_args) {
      NodeArg& node_arg = graph.GetOrCreateNodeArg(arg.name, arg.type_proto);
      input_args.push_back(&node_arg);
    }

    for (const auto& arg : node_def.output_args) {
      NodeArg& node_arg = graph.GetOrCreateNodeArg(arg.name, arg.type_proto);
      output_args.push_back(&node_arg);
    }

    graph.AddNode(node_def.name,
                  node_def.op_type,
                  "Backward pass",
                  input_args,
                  output_args,
                  &node_def.attributes,
                  node_def.domain, 
                  node_def.priority_);
  }

  // Add new inputs to the graph.
  if (!graph_element_defs.GraphInputs().empty()) {
    std::vector<const NodeArg*> new_inputs;
    ORT_RETURN_IF_ERROR(AddToExistingNodeArgs(
        "add graph inputs", graph, graph_element_defs.GraphInputs(), graph.GetInputsIncludingInitializers(),
        false, new_inputs));
    graph.SetInputs(new_inputs);
  }

  // Add new outputs to the graph.
  if (!graph_element_defs.GraphOutputs().empty()) {
    std::vector<const NodeArg*> new_outputs;
    ORT_RETURN_IF_ERROR(AddToExistingNodeArgs(
        "add graph outputs", graph, graph_element_defs.GraphOutputs(), graph.GetOutputs(), false, new_outputs));
    graph.SetOutputs(new_outputs);
  }

  graph.SetGraphResolveNeeded();
  Graph::ResolveOptions options;
  options.initializer_names_to_preserve = p_initializer_names_to_preserve;
  return graph.Resolve(options);
}

Status GraphAugmenter::OverrideGraphOutputs(Graph& graph, const std::vector<std::string>& graph_outputs) {
  std::vector<const NodeArg*> new_outputs;
  ORT_RETURN_IF_ERROR(AddToExistingNodeArgs(
      "override graph outputs", graph, graph_outputs, {}, true, new_outputs));
  graph.SetOutputs(new_outputs);

  return graph.Resolve();
}

}  // namespace training
}  // namespace onnxruntime
