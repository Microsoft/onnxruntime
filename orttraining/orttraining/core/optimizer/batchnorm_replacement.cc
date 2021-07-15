// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/batchnorm_replacement.h"

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status BatchNormReplacement::Apply(Graph& graph, Node& bn_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const auto& bn_inputs = bn_node.MutableInputDefs();
  auto& bn_outputs = bn_node.MutableOutputDefs();
  const NodeArg* scale_input_def = bn_inputs[1];
  auto scale_input_def_type_proto = scale_input_def->TypeAsProto();

  if (bn_outputs.size() == 1) {
    NodeArg& running_mean_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("running_mean_def"), scale_input_def_type_proto);
    NodeArg& running_var_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("running_var_def"), scale_input_def_type_proto);
    bn_outputs.push_back(&running_mean_def);
    bn_outputs.push_back(&running_var_def);
  }

  if (bn_outputs.size() == 3) {
    NodeArg& saved_mean_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean_def"), scale_input_def_type_proto);
    NodeArg& saved_inv_std = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std"), scale_input_def_type_proto);
    bn_outputs.push_back(&saved_mean_def);
    bn_outputs.push_back(&saved_inv_std);
  }

  // check Batch Normalization node has 5 output node args for training mode
  ORT_ENFORCE(bn_node.OutputDefs().size() == 5);

  Node& batchnorm_internal_node = graph.AddNode(graph.GenerateNodeName(bn_node.Name() + "_BatchNormInternal"),
                                                "BatchNormInternal",
                                                "BatchNormalization with saved mean/inv_std",
                                                bn_inputs,
                                                bn_outputs,
                                                &bn_node.GetAttributes(),
                                                kMSDomain);
  batchnorm_internal_node.AddAttribute("training_mode", static_cast<int64_t>(1));
  // Assign provider to this new node. Provider should be same as the provider for old node.
  batchnorm_internal_node.SetExecutionProviderType(bn_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, batchnorm_internal_node, bn_node);
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

bool BatchNormReplacement::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  return true;
}

}  // namespace onnxruntime
