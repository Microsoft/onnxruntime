// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"
#include "orttraining/core/framework/distributed_run_context.h"
namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

static NodeDef BuildGlobalHorovodBarrierNode(
    const std::vector<std::string>& ready_names,
    const std::string& global_barrier_name,
    const std::string& global_barrier_ready,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::string barrier_input_name = global_barrier_name + "/input";
  std::string barrier_output_name = global_barrier_name + "/output";

  // Global horovod barrier no-op input.
  TensorProto tensor_proto;
  tensor_proto.add_dims(0);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.set_name(barrier_input_name);
  graph_defs.AddInitializers({tensor_proto});

  std::vector<ArgDef> barrier_inputs{barrier_input_name};
  std::transform(ready_names.begin(), ready_names.end(), std::back_inserter(barrier_inputs), [](const std::string& name) { return ArgDef(name); });
  std::vector<ArgDef> barrier_outputs{barrier_output_name, global_barrier_ready};

  return NodeDef("HorovodBarrier", barrier_inputs, barrier_outputs, NodeAttributes(), global_barrier_name);
}

static NodeDef& GetGlobalHorovodBarrierNode(
    GraphAugmenter::GraphDefs& graph_defs,
    const std::string& global_barrier_name,
    const std::string& global_barrier_ready) {
  // Find the global horovod barrier node.
  auto& nodes = graph_defs.NodeDefs();
  auto barrier_iter = std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
  if (barrier_iter != nodes.end())
    return *barrier_iter;

  // Create the global horovod barrier.
  graph_defs.AddNodeDefs({BuildGlobalHorovodBarrierNode({}, global_barrier_name, global_barrier_ready, graph_defs)});
  return *std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
}

static ArgDef BuildHorovodAllReduceNode(const ArgDef& gradient_argdef, GraphAugmenter::GraphDefs& graph_defs, int64_t reduce_op) {
  const std::string& gradient_name = gradient_argdef.name;
  ArgDef reduce_output(gradient_name + "_AllReduce_Out", gradient_argdef.type_proto);
  ArgDef reduce_ready(gradient_name + "_AllReduce_Ready");
  ArgDef local_barrier_output(gradient_name + "_Barrier_Out", gradient_argdef.type_proto);
  ArgDef local_barrier_ready(gradient_name + "_Barrier_Ready");

  // Add horovod all reduce node.
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduce",
                                  {gradient_argdef},
                                  {reduce_output, reduce_ready},
                                  {ONNX_NAMESPACE::MakeAttribute("reduce_op", static_cast<int64_t>(reduce_op))},
                                  gradient_name + "_AllReduce")});

  // Add ready check to global horovod barrier.
  const std::string global_barrier_name = "horovod/barrier";
  const std::string global_barrier_ready = "horovod/barrier/ready";
  NodeDef& global_barrier_node = GetGlobalHorovodBarrierNode(graph_defs, global_barrier_name, global_barrier_ready);
  global_barrier_node.input_args.push_back(reduce_ready);

  // Add local horovod barrier node.
  graph_defs.AddNodeDefs({NodeDef("HorovodBarrier",
                                  {reduce_output, global_barrier_ready},
                                  {local_barrier_output, local_barrier_ready},
                                  NodeAttributes(),
                                  gradient_name + "_Barrier")});

  return local_barrier_output;
}

Status AllreduceOptimizerGraphBuilder::AddHorovodAllReduceForGradients(std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                                                                       GraphAugmenter::GraphDefs& graph_defs,
                                                                       const int64_t horovod_reduce_op) {
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildHorovodAllReduceNode(gradient_argdefs[i], graph_defs, horovod_reduce_op);
  }
  return Status::OK();
}

static Status AddNcclAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    std::vector<ArgDef>& input_gradient_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> allreduce_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    TypeProto* allreduced_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[i]);
    allreduced_gradient_type_proto->mutable_tensor_type()->set_elem_type(
        input_gradient_argdef[0].type_proto->tensor_type().elem_type());

    allreduce_outputs[i] = ArgDef(gradient_argdefs[i].name + "_AllReduce_Out", allreduced_gradient_type_proto);
  }

  // Add NCCL Allreduce node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  input_gradient_argdef,
                                  allreduce_outputs,
                                  NodeAttributes(),
                                  "NcclAllReduce")});

  gradient_argdefs = allreduce_outputs;
  return Status::OK();
}

AllreduceOptimizerGraphBuilder::AllreduceOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs,
                            updated_weight_names_map) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1,
              "Allreduce optimizer graph builder can only be used for distributed training.");
  if (opt_graph_config.use_nccl) {
    ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
  } else {
    ORT_ENFORCE(IsHorovodAvailable(), "Distributed training with Horovod is not supported, as Horovod is not enabled in this build.");
  }
}

Status AllreduceOptimizerGraphBuilder::BuildInternal(
    bool should_add_gradient_norm,
    bool should_add_gradient_finite_check,
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  const int64_t horovod_reduce_op = opt_graph_config_.horovod_reduce_op;
  
  bool divide_sample_after_all_reduce = true;
  ArgDef scale;

  ArgDef grad_multi_factor = AddAllReduceForSampleCount(nodearg_name_generator, gradient_argdefs, graph_defs);
  if (divide_sample_after_all_reduce == true) {
    scale = ArgDef(nodearg_name_generator("one"), graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    graph_defs.AddInitializers({CreateTensorProto<float>(scale.name, 1.0f, {})});
  } else {
    scale = grad_multi_factor;
  }

  // add gradient scaling
  std::vector<ArgDef> output_gradient_argdef;
  // const auto total_num_accumulations =
  //     opt_graph_config_.gradient_accumulation_steps; // * opt_graph_config_.data_parallel_group_size;
  // ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  //We did not divide by acc step, instead, we divide by total sample across all accumulation batches.
  // todo : make this scale optional.
  //const float scale = 1.0f / total_num_accumulations; 
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, output_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType()));

  // add Allreduce for gradients
  if (opt_graph_config_.use_nccl) {
    ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradients(gradient_argdefs, output_gradient_argdef, graph_defs));
    if (divide_sample_after_all_reduce == true) {
      ORT_RETURN_IF_ERROR(ScaleGradWithSampleCount(nodearg_name_generator, gradient_argdefs, graph_defs, grad_multi_factor));
    } 
  } else {
    ORT_RETURN_IF_ERROR(AddHorovodAllReduceForGradients(gradient_argdefs, graph_defs, horovod_reduce_op));
  }

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

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
