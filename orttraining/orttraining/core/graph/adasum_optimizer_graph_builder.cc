// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#include "orttraining/core/framework/distributed_run_context.h"
namespace onnxruntime {
namespace training {

ArgDef AdasumOptimizerGraphBuilder::BuildWeightUpdateNode(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& gradient,
    ArgDef& weight,
    const ArgDef& gradient_finite_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
  TypeProto* gradient_fp32_type_proto = graph_defs.CopyTypeProto(weight);
  ArgDef gradient_update_output = gradient;
  ArgDef weight_update_output = ArgDef(nodearg_name_generator(weight.name + "_update_out"), gradient_fp32_type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"InPlaceAccumulator", kMSDomain, 1},
                                  {weight, gradient_update_output, gradient_finite_argdef},
                                  {weight_update_output},
                                  NodeAttributes(),
                                  weight_update_output.name)});
  return weight_update_output;
}

Status AdasumOptimizerGraphBuilder::AddWeightUpdateNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                                         std::vector<ArgDef>& gradient_argdefs,
                                                         std::vector<ArgDef>& weight_argdefs,
                                                         const ArgDef& adasum_gradient_finite_argdef,
                                                         GraphAugmenter::GraphDefs& graph_defs,
                                                         std::vector<ArgDef>& output_weight_argdefs) {
  output_weight_argdefs.clear();
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    output_weight_argdefs.push_back(BuildWeightUpdateNode(nodearg_name_generator,
                                                          gradient_argdefs[i],
                                                          weight_argdefs[i],
                                                          adasum_gradient_finite_argdef,
                                                          graph_defs));
  }
  return Status::OK();
}

static Status AddReducedGradientScalingNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                             std::vector<ArgDef>& gradient_argdefs,
                                             GraphAugmenter::GraphDefs& graph_defs,
                                             const float scale) {
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    ArgDef& gradient_argdef = gradient_argdefs[i];
    TypeProto* scale_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
    ArgDef gradient_scale_argdef(gradient_argdef.name + "_reduced_gradient_scaling_divisor", scale_type_proto);
    graph_defs.AddInitializers({CreateTensorProto<float>(gradient_scale_argdef.name, scale, {})});
    TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
    ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_reduced_scaled"),
                                           scaled_gradient_type_proto);
    auto target_type = scaled_gradient_type_proto->mutable_tensor_type()->elem_type();
    graph_defs.AddNodeDefs({NodeDef(OpDef{"MixedPrecisionScale", kMSDomain, 1},
                                    {gradient_scale_argdef, gradient_argdef},
                                    {scaled_gradient_argdef},
                                    {ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(target_type))},
                                    scaled_gradient_argdef.name)});

    gradient_argdef = scaled_gradient_argdef;
  }
  return Status::OK();
}

AdasumOptimizerGraphBuilder::AdasumOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map)
    : AllreduceOptimizerGraphBuilder(opt_builder_registry,
                                     opt_graph_config,
                                     weight_names_to_opt_configs,
                                     updated_weight_names_map) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1,
              "Adasum optimizer graph builder can only be used for distributed training.");
  ORT_ENFORCE(IsHorovodAvailable(), "Distributed training with Adasum needs building with Horovod.");
}

Status AdasumOptimizerGraphBuilder::BuildOptimizerNode(
    const std::unique_ptr<OptimizerBuilder>& opt_builder,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* global_gradient_norm_argdef,
    const ArgDef* global_gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<TensorProto>& new_initializers,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs) {
  OptimizerBuilderConfig config;
  config.weight_argdefs = weight_argdefs;
  config.gradient_argdefs = gradient_argdefs;
  if (global_gradient_norm_argdef != nullptr) {
    config.gradient_norm_argdef = *global_gradient_norm_argdef;
  }
  if (global_gradient_norm_finite_argdef != nullptr) {
    config.gradient_norm_finite_argdef = *global_gradient_norm_finite_argdef;
  }
  config.opt_configs = opt_configs;
  // Always enable grad clipping for Adasum
  config.enable_grad_clipping = true;
  config.shared_optimizer_states = opt_graph_config_.shared_optimizer_states;
  ORT_RETURN_IF_ERROR(opt_builder->Build(
      config, graph_defs,
      new_initializers,
      output_weight_argdefs, output_gradient_argdefs));

  return Status::OK();
}

Status AdasumOptimizerGraphBuilder::BuildInternal(
    bool should_add_gradient_norm,
    bool should_add_gradient_finite_check,
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  // Set weight update to false for optimizer
  for (auto& opt_config : opt_configs_) {
    opt_config.update_weight = false;
  }

  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  const int64_t horovod_reduce_op = opt_graph_config_.horovod_reduce_op;

  // add gradient scaling
  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);

  ArgDef scale = AddAllReduceForSampleCount(nodearg_name_generator, gradient_argdefs, graph_defs);
  //const float scale = 1.0f / total_num_accumulations;
  // No fusion with Adasum
  const bool fuse_scaling_outputs = false;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), fuse_scaling_outputs));

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;

  if (should_add_gradient_norm) {
    bool megatron_enabled = DistributedRunContext::GroupSize(WorkerGroupType::HorizontalParallel) > 1;
    if (megatron_enabled) {
      std::vector<ArgDef> gradient_norm_inputs;
      ORT_ENFORCE(gradient_argdefs.size() > 1, "Using fused gradient to calcuate gradient norm is not supported when megatron enabled");
      int rank_in_hori_group = DistributedRunContext::RankInGroup(WorkerGroupType::HorizontalParallel);
      if (rank_in_hori_group != 0) {
        for (size_t i = 0; i < megatron_partitioned_weight_grad_index_.size(); ++i) {
          gradient_norm_inputs.push_back(gradient_argdefs[megatron_partitioned_weight_grad_index_[i]]);
        }
      } else {
        gradient_norm_inputs = gradient_argdefs;
      }
      ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef, global_gradient_norm_output_name + "_prior_mega_all_reduce"));
      ORT_RETURN_IF_ERROR(AddL2NormBetweenMegatronRanksNcclAllReduce(global_grad_norm_argdef, graph_defs, global_gradient_norm_output_name));
    } else {
      ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_argdefs, graph_defs, global_grad_norm_argdef, global_gradient_norm_output_name));
    }

    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;
    graph_defs.AddGraphOutputs({global_grad_norm_argdef.name});
  }

  if (should_add_gradient_finite_check) {
    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, {global_grad_norm_argdef}, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GradientAllIsFinite] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
      opt_builder_registry_, weight_argdefs, gradient_argdefs,
      &global_grad_norm_argdef,
      &global_grad_norm_finite_argdef,
      opt_configs_, graph_defs,
      optimizer_state_initializer_names));

  // Perform allreduce on deltas after step() for Adasum
  ORT_RETURN_IF_ERROR(AddHorovodAllReduceForGradients(gradient_argdefs, graph_defs, horovod_reduce_op));
  // If Adasum GPU hierarchical reduce is used, then scale resulting gradients by local size.
  if (opt_graph_config_.adasum_reduction_type == AdasumReductionType::GpuHierarchical) {
    const float adasum_scale = 1.0f / opt_graph_config_.local_size;
    ORT_RETURN_IF_ERROR(AddReducedGradientScalingNodes(nodearg_name_generator, gradient_argdefs, graph_defs, adasum_scale));
  }

  //check if allreduced deltas are finite
  ArgDef adasum_global_grad_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, gradient_argdefs, graph_defs, adasum_global_grad_finite_argdef,
        "adasum_all_gradients_finite"));
    optimizer_graph_outputs[OptimizerOutputKey::DeltaAllIsFinite] = adasum_global_grad_finite_argdef.name;
  }
  //Add weight update.
  std::vector<ArgDef> output_weight_args;
  ORT_RETURN_IF_ERROR(AddWeightUpdateNodes(nodearg_name_generator,
                                           gradient_argdefs,
                                           weight_argdefs,
                                           adasum_global_grad_finite_argdef,
                                           graph_defs,
                                           output_weight_args));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
