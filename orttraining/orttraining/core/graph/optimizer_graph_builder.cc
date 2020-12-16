// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/optimizer_graph_builder.h"

#include <cassert>
#include <algorithm>
#include <functional>
#include <iterator>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace training {

Status GetArgDefsFromGraph(
    const Graph& graph, const std::vector<std::string>& node_arg_names,
    std::vector<ArgDef>& argdefs) {
  std::vector<ArgDef> result;
  result.reserve(node_arg_names.size());
  for (const auto& node_arg_name : node_arg_names) {
    const auto* node_arg = graph.GetNodeArg(node_arg_name);
    ORT_RETURN_IF_NOT(node_arg, "Failed to get NodeArg with name ", node_arg_name);
    result.emplace_back(node_arg_name, node_arg->TypeAsProto());
  }
  argdefs = std::move(result);
  return Status::OK();
}

ArgDef BuildGradientAccumulationNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                     const ArgDef& gradient,
                                     ArgDef& gradient_accumulation_buffer,
                                     GraphAugmenter::GraphDefs& graph_defs,
                                     bool add_accumulate_buffer_as_initializers) {
  TypeProto* gradient_fp32_type_proto = graph_defs.CopyTypeProto(gradient);
  gradient_fp32_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ArgDef gradient_accumulate_buffer(nodearg_name_generator(gradient.name + "_accumulate_buffer"),
                                    gradient_fp32_type_proto);
  ArgDef gradient_accumulator_output(nodearg_name_generator(gradient.name + "_accumulator_output"),
                                     gradient_fp32_type_proto);

  std::vector<int64_t> dims;
  ORT_ENFORCE(gradient.type_proto &&
              gradient.type_proto->has_tensor_type() &&
              gradient.type_proto->tensor_type().has_shape());
  for (const auto& dim : gradient.type_proto->tensor_type().shape().dim()) {
    dims.push_back(dim.dim_value());
  }
  if (add_accumulate_buffer_as_initializers)
    graph_defs.AddInitializers({CreateTensorProto<float>(gradient_accumulate_buffer.name, 0.f, dims)});
  graph_defs.AddNodeDefs({NodeDef(OpDef{"InPlaceAccumulator", kMSDomain, 1},
                                  {gradient_accumulate_buffer, gradient},
                                  {gradient_accumulator_output},
                                  NodeAttributes(),
                                  gradient_accumulator_output.name)});

  gradient_accumulation_buffer = gradient_accumulate_buffer;
  return gradient_accumulator_output;
}

ArgDef BuildGroupNode(const std::string& group_output_name,
                      const std::vector<ArgDef>& input_argdefs,
                      GraphAugmenter::GraphDefs& graph_defs) {
  ArgDef group_output(group_output_name,
                      graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  graph_defs.AddNodeDefs({NodeDef(OpDef{"Group", kMSDomain, 1},
                                  input_argdefs,
                                  {group_output},
                                  NodeAttributes(),
                                  group_output.name)});
  return group_output;
}

Status OptimizerGraphBuilder::AddGradientPassThroughNode(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> outputs_gradient_argdefs;
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    ArgDef& gradient_argdef = gradient_argdefs[i];
    TypeProto* output_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
    outputs_gradient_argdefs.emplace_back(nodearg_name_generator(gradient_argdef.name + "_passthrough"),
                                          output_gradient_type_proto);
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"PassThrough", kMSDomain, 1},
                                  gradient_argdefs,
                                  outputs_gradient_argdefs,
                                  NodeAttributes(),
                                  nodearg_name_generator("GradientPassThrought"))});

  gradient_argdefs = outputs_gradient_argdefs;

  return Status::OK();
}

Status OptimizerGraphBuilder::AddGradientScalingNodes(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& pre_allreduce_scale,
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    ArgDef& fused_gradient_argdef,          // update argdef in place
    GraphAugmenter::GraphDefs& graph_defs,
    ONNX_NAMESPACE::TensorProto_DataType allreduce_element_type,
    const bool fuse_scaling_outputs) {
  // ArgDef pre_allreduce_scale(nodearg_name_generator("pre_allreduce_scale"),
  //                            graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  // graph_defs.AddInitializers({CreateTensorProto<float>(pre_allreduce_scale.name, scale, {})});

  if (fuse_scaling_outputs) {
    TypeProto* fused_gradient_type_proto = graph_defs.CreateTypeProto();
    fused_gradient_type_proto->mutable_tensor_type()->set_elem_type(allreduce_element_type);
    fused_gradient_argdef = ArgDef("fused_gradient", fused_gradient_type_proto);

    std::vector<ArgDef> inputs;
    inputs.emplace_back(pre_allreduce_scale);
    for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
      inputs.emplace_back(gradient_argdefs[i]);
    }
    graph_defs.AddNodeDefs({NodeDef(OpDef{"MixedPrecisionScale", kMSDomain, 1},
                                    inputs,
                                    {fused_gradient_argdef},
                                    std::vector<AttributeProto>({ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(allreduce_element_type)),
                                                                 ONNX_NAMESPACE::MakeAttribute("fuse_outputs", static_cast<int64_t>(true))}),
                                    pre_allreduce_scale.name)});
  } else {
    for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
      ArgDef& gradient_argdef = gradient_argdefs[i];

      TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
      scaled_gradient_type_proto->mutable_tensor_type()->set_elem_type(allreduce_element_type);

      ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled"),
                                             scaled_gradient_type_proto);
      graph_defs.AddNodeDefs({NodeDef(OpDef{"MixedPrecisionScale", kMSDomain, 1},
                                      {pre_allreduce_scale, gradient_argdef},
                                      {scaled_gradient_argdef},
                                      {ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(allreduce_element_type))},
                                      scaled_gradient_argdef.name)});

      gradient_argdef = scaled_gradient_argdef;
    }
  }

  return Status::OK();
}

Status OptimizerGraphBuilder::AddGradientScalingNodes(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& pre_allreduce_scale,
    std::vector<ArgDef>& input_gradient_argdefs,  // update argdefs in place
    std::vector<ArgDef>& output_gradient_argdef,  // update argdef in place
    GraphAugmenter::GraphDefs& graph_defs,
    ONNX_NAMESPACE::TensorProto_DataType target_type) {
  // ArgDef pre_allreduce_scale(nodearg_name_generator("pre_allreduce_scale"),
  //                            graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  // graph_defs.AddInitializers({CreateTensorProto<float>(pre_allreduce_scale.name, scale, {})});

  TypeProto* fused_gradient_type_proto = graph_defs.CreateTypeProto();
  fused_gradient_type_proto->mutable_tensor_type()->set_elem_type(target_type);

  std::vector<ArgDef> inputs;
  inputs.emplace_back(pre_allreduce_scale);
  for (size_t i = 0; i < input_gradient_argdefs.size(); ++i) {
    inputs.emplace_back(input_gradient_argdefs[i]);
  }

  for (size_t i = 0; i < input_gradient_argdefs.size(); ++i) {
    ArgDef& gradient_argdef = input_gradient_argdefs[i];

    TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
    scaled_gradient_type_proto->mutable_tensor_type()->set_elem_type(target_type);

    output_gradient_argdef.emplace_back(ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled"), scaled_gradient_type_proto));
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"MixedPrecisionScale", kMSDomain, 1},
                                  inputs,
                                  output_gradient_argdef,
                                  std::vector<AttributeProto>({ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(target_type))}),
                                  pre_allreduce_scale.name)});

  return Status::OK();
}

ArgDef OptimizerGraphBuilder::AddAllReduceForSampleCount(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    GraphAugmenter::GraphDefs& graph_defs) {

  ArgDef one_argdef(nodearg_name_generator("one"),
                             graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  graph_defs.AddInitializers({CreateTensorProto<float>(one_argdef.name, 1.0f, {})});

  //TypeProto* temp_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[0]);
  ArgDef scale_argdef = ArgDef(std::string("sample_size_count"),
                             graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
                             //(, temp_type_proto);
  graph_defs.AddGraphInputs({"sample_size_count"});

  if (DistributedRunContext::GroupSize(WorkerGroupType::DataParallel) > 1) {
    TypeProto* allreduced_sacle_type_proto = graph_defs.CopyTypeProto(scale_argdef);
    ArgDef reduced_scale_argdef = ArgDef(scale_argdef.name + "_AllReduce_Out", allreduced_sacle_type_proto);
    
    // Add NCCL Allreduce node.
    graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                    {scale_argdef},
                                    {reduced_scale_argdef},
                                    NodeAttributes(),
                                    nodearg_name_generator("NcclAllReduce_allreduce_scale"))});

    scale_argdef = reduced_scale_argdef;
  }

  TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(scale_argdef);
  ArgDef reciprocal_of_sample_size_count_argdef = ArgDef(nodearg_name_generator("reciprocal_of_sample_size_count"), scaled_gradient_type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"Div"},
                                {one_argdef, scale_argdef},
                                {reciprocal_of_sample_size_count_argdef},
                                NodeAttributes(),
                                nodearg_name_generator("reciprocal_of_sample_size_count"))});



  TypeProto* temp_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[0]);
  ORT_ENFORCE(temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT 
    || temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  // if (temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  //   std::cout<< "don't need convert float to fp16 since, gradient_defs are fp32"<< std::endl;

  // whatever type gradients are, we can pass either fp16 or fp32 scale per MixedPrecisionScale op defined.
    return reciprocal_of_sample_size_count_argdef;
  
  
  // TypeProto* fp16_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[0]);
  // fp16_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  // ArgDef fp16_reciprocal_of_sample_size_count_argdef = ArgDef(nodearg_name_generator("reciprocal_of_sample_size_count"), fp16_type_proto);
  // graph_defs.AddNodeDefs({NodeDef(OpDef{"Cast"},
  //                           {reciprocal_of_sample_size_count_argdef},
  //                           {fp16_reciprocal_of_sample_size_count_argdef},
  //                           std::vector<AttributeProto>({ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16))}),
  //                           nodearg_name_generator("reciprocal_of_sample_size_count_fp16"))});
  // return fp16_reciprocal_of_sample_size_count_argdef;
  
  // return reciprocal_of_sample_size_count_argdef;

  // for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
  //   ArgDef& gradient_argdef = gradient_argdefs[i];
  //   TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
  //   ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled_after_allreduce"), scaled_gradient_type_proto);
  
  //   graph_defs.AddNodeDefs({NodeDef(OpDef{"Div"},
  //                                   {gradient_argdef, scale_argdef},
  //                                   {scaled_gradient_argdef},
  //                                   NodeAttributes(),
  //                                   scaled_gradient_argdef.name)});

  //   gradient_argdef = scaled_gradient_argdef;
  // }
}

Status OptimizerGraphBuilder::ScaleGradWithSampleCount(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef scale_argdef) {

  ArgDef scale =  scale_argdef;
  TypeProto* temp_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[0]);
  ORT_ENFORCE(temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT 
    || temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  if (temp_type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    std::cout<< "don't need convert float to fp16 since, gradient_defs are fp32"<< std::endl;
  } else {
    TypeProto* fp16_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[0]);
    fp16_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    scale = ArgDef(nodearg_name_generator("grad_multi_factor_cast"), fp16_type_proto);
    graph_defs.AddNodeDefs({NodeDef(OpDef{"Cast"},
                              {scale_argdef},
                              {scale},
                              std::vector<AttributeProto>({ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16))}),
                              nodearg_name_generator("grad_multi_factor_cast_node_name"))});
  }

  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    ArgDef& gradient_argdef = gradient_argdefs[i];
    TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
    ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled_after_allreduce"), scaled_gradient_type_proto);
  
    graph_defs.AddNodeDefs({NodeDef(OpDef{"Mul"},
                                    {gradient_argdef, scale},
                                    {scaled_gradient_argdef},
                                    NodeAttributes(),
                                    scaled_gradient_argdef.name)});

    gradient_argdef = scaled_gradient_argdef;
  }

  return Status::OK();
}

ArgDef AddGradientAccumulationNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                    std::vector<ArgDef>& gradient_argdefs,               // update argdefs in place
                                    std::vector<ArgDef>& gradient_accumulation_buffers,  // output
                                    GraphAugmenter::GraphDefs& graph_defs) {
  gradient_accumulation_buffers.resize(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildGradientAccumulationNode(
        nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs);
  }

  ArgDef group_accumulate_gradient_output = BuildGroupNode(nodearg_name_generator("Group_Accumulated_Gradients"),
                                                           gradient_argdefs,
                                                           graph_defs);
  graph_defs.AddGraphOutputs({group_accumulate_gradient_output.name});
  return group_accumulate_gradient_output;
}

ArgDef BuildZeroGradientNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                             const ArgDef& control_signal,
                             const ArgDef& gradient,
                             GraphAugmenter::GraphDefs& graph_defs) {
  ArgDef gradient_zero_output(nodearg_name_generator(gradient.name + "_zero_out"), gradient.type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"ZeroGradient", kMSDomain, 1},
                                  {gradient, control_signal},
                                  {gradient_zero_output},
                                  NodeAttributes(),
                                  gradient_zero_output.name)});
  return gradient_zero_output;
}

Status AddZeroGradientNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                            const std::vector<ArgDef>& control_signals,
                            std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                            GraphAugmenter::GraphDefs& graph_defs) {
  //assert(gradient_argdefs.size() == control_signals.size());
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildZeroGradientNode(nodearg_name_generator, control_signals[i], gradient_argdefs[i], graph_defs);
  }

  return Status::OK();
}

Status OptimizerGraphBuilder::BuildOptimizerNode(
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
  config.enable_grad_clipping = opt_graph_config_.enable_grad_norm_clip;
  config.shared_optimizer_states = opt_graph_config_.shared_optimizer_states;
  ORT_RETURN_IF_ERROR(opt_builder->Build(
      config, graph_defs,
      new_initializers,
      output_weight_argdefs, output_gradient_argdefs));

  return Status::OK();
}

Status OptimizerGraphBuilder::AddDirectWeightUpdate(
    const OptimizerBuilderRegistry& opt_builder_registry,
    std::vector<ArgDef>& weight_argdefs,    // update argdefs in place
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    const ArgDef* global_gradient_norm_argdef,
    const ArgDef* global_gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::unordered_set<std::string>& optimizer_state_initializer_names) {
  ORT_RETURN_IF_NOT(weight_argdefs.size() == gradient_argdefs.size());
  ORT_RETURN_IF_NOT(weight_argdefs.size() == opt_configs.size());

  std::vector<TensorProto> new_initializers;
  std::vector<ArgDef> output_weight_argdefs;
  std::vector<ArgDef> output_gradient_argdefs;

  for (size_t i = 0; i < opt_configs.size(); ++i) {
    ORT_RETURN_IF_NOT(
        opt_configs[i].name == opt_configs[0].name,
        "All optimizers must be the same type, but the graph contains ",
        opt_configs[0].name, " and ", opt_configs[i].name);
  }

  auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[0].name);
  ORT_RETURN_IF_NOT(
      opt_builder, "Failed to get Optimizer builder for ", opt_configs[0].name);

  ORT_RETURN_IF_ERROR(BuildOptimizerNode(
      opt_builder,
      weight_argdefs, gradient_argdefs,
      global_gradient_norm_argdef, global_gradient_norm_finite_argdef,
      opt_configs, graph_defs,
      new_initializers,
      output_weight_argdefs, output_gradient_argdefs));

  graph_defs.AddInitializers(new_initializers);
  weight_argdefs = std::move(output_weight_argdefs);
  gradient_argdefs = std::move(output_gradient_argdefs);

  std::unordered_set<std::string> all_new_initializer_names{};
  std::transform(
      new_initializers.begin(), new_initializers.end(),
      std::inserter(all_new_initializer_names, all_new_initializer_names.end()),
      [](const TensorProto& initializer) { return initializer.name(); });
  optimizer_state_initializer_names = std::move(all_new_initializer_names);

  return Status::OK();
}

Status OptimizerGraphBuilder::AddL2NormBetweenMegatronRanksNcclAllReduce(
    ArgDef& norm_argdef,
    GraphAugmenter::GraphDefs& graph_defs,
    std::string output_name) {
  // Square the L2 norm.
  ArgDef exponent(norm_argdef.name + "_pow2",
                  graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  graph_defs.AddInitializers({CreateTensorProto<float>(exponent.name, 2.0f, {})});
  ArgDef norm_squared(norm_argdef.name + "_squared", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef("Pow",
                                  {norm_argdef, exponent},
                                  {norm_squared},
                                  NodeAttributes(),
                                  norm_squared.name)});

  //AllReduce the squared L2 norms.
  ArgDef allreduce_output(norm_argdef.name + "_AllReduce_Out", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  {norm_squared},
                                  {allreduce_output},
                                  {ONNX_NAMESPACE::MakeAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel))},
                                  allreduce_output.name)});

  // Sqrt the reduced L2 norm.
  ArgDef sqrt_output(output_name, norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef("Sqrt",
                                  {allreduce_output},
                                  {sqrt_output},
                                  NodeAttributes(),
                                  sqrt_output.name)});

  norm_argdef = sqrt_output;
  return Status::OK();
}

Status OptimizerGraphBuilder::AddGradientNorm(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const std::vector<ArgDef>& grad_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& grad_norm_argdef, std::string output_name) {
  ONNX_NAMESPACE::TensorProto_DataType grad_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(grad_argdefs[0].type_proto->tensor_type().elem_type());
  if (grad_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      grad_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      grad_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Unsupport gradient type: it has to be either float, MLFloat16 or BFloat16.");
  }

  for (const auto& argdef : grad_argdefs) {
    ONNX_NAMESPACE::TensorProto_DataType elem_type =
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(argdef.type_proto->tensor_type().elem_type());
    if (elem_type != grad_type) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "All gradient tensors' types must be the same.");
    }
  }

  const TypeProto* const grad_norm_type = graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  grad_norm_argdef = ArgDef{nodearg_name_generator(output_name), grad_norm_type};

  graph_defs.AddNodeDefs({NodeDef{OpDef{"ReduceAllL2", kMSDomain, 1},
                                  grad_argdefs,
                                  {grad_norm_argdef},
                                  NodeAttributes(),
                                  grad_norm_argdef.name,
                                  static_cast<int>(ExecutionPriority::GLOBAL_LOW)}});

  return Status::OK();
}

Status OptimizerGraphBuilder::AddFiniteGradientCheck(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const std::vector<ArgDef>& grad_norm_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& grad_norm_finite_argdef,
    const std::string& node_name) {
  const TypeProto* const grad_norm_finite_type =
      graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  grad_norm_finite_argdef =
      ArgDef{nodearg_name_generator(node_name), grad_norm_finite_type};

  graph_defs.AddNodeDefs({NodeDef{OpDef{"IsAllFinite", kMSDomain, 1},
                                  grad_norm_argdefs,
                                  {grad_norm_finite_argdef},
                                  NodeAttributes(),
                                  grad_norm_finite_argdef.name}});

  graph_defs.AddGraphOutputs({grad_norm_finite_argdef.name});

  return Status::OK();
}

OptimizerGraphBuilder::OptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map)
    : opt_builder_registry_(opt_builder_registry),
      opt_graph_config_(opt_graph_config),
      updated_weight_names_map_(updated_weight_names_map) {
  // add weight names
  weight_names_.reserve(weight_names_to_opt_configs.size());
  std::transform(
      weight_names_to_opt_configs.begin(), weight_names_to_opt_configs.end(),
      std::back_inserter(weight_names_),
      [](const std::pair<std::string, OptimizerNodeConfig>& name_and_info) {
        return name_and_info.first;
      });

  // deterministic ordering for consistent generated nodearg names
  std::sort(weight_names_.begin(), weight_names_.end());

  // add gradient names
  gradient_names_.reserve(weight_names_.size());
  std::transform(
      weight_names_.begin(), weight_names_.end(), std::back_inserter(gradient_names_),
      [&weight_names_to_opt_configs](const std::string& weight_name) {
        return GradientBuilderBase::GradientName(weight_names_to_opt_configs.at(weight_name).mixed_precision_weight_arg != nullptr ? weight_names_to_opt_configs.at(weight_name).mixed_precision_weight_arg->Name() : weight_name);
      });

  for (size_t i = 0; i < weight_names_.size(); ++i) {
    if (weight_names_to_opt_configs.at(weight_names_[i]).megatron_partitioned == true) {
      megatron_partitioned_weight_grad_index_.push_back(i);
    }
  }
  // add optimizer configurations
  opt_configs_.reserve(weight_names_.size());
  std::transform(
      weight_names_.begin(), weight_names_.end(), std::back_inserter(opt_configs_),
      [&weight_names_to_opt_configs](const std::string& weight_name) {
        return weight_names_to_opt_configs.at(weight_name);
      });
}

Status OptimizerGraphBuilder::Build(
    Graph& graph,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  if (weight_names_.empty()) {
    // nothing to do
    return Status::OK();
  }

  // from here, we assume there is at least one weight/gradient to process

  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  GraphAugmenter::GraphDefs graph_defs;
  std::vector<ArgDef> weight_argdefs;
  std::vector<ArgDef> gradient_argdefs;

  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, weight_names_, weight_argdefs));
  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, gradient_names_, gradient_argdefs));

  const bool is_gradient_accumulation_enabled = opt_graph_config_.gradient_accumulation_steps > 1;

  // add gradient accumulation
  std::vector<ArgDef> gradient_accumulation_buffers;
  if (is_gradient_accumulation_enabled) {
    ArgDef group_accumulate_gradient_output =
        AddGradientAccumulationNodes(nodearg_name_generator, gradient_argdefs, gradient_accumulation_buffers, graph_defs);
    optimizer_graph_outputs[OptimizerOutputKey::GradientAccumulation] = group_accumulate_gradient_output.name;
  }

  //gradient norm for bfloat16 is not ready yet. skip it to unblock the testing
  //will add it back when it is ready
  bool should_add_gradient_norm =
      opt_graph_config_.enable_grad_norm_clip ||
      (opt_graph_config_.use_mixed_precision && opt_graph_config_.mixed_precision_type == MixedPrecisionDataType::FP16);
  bool should_add_gradient_finite_check =
      opt_graph_config_.use_mixed_precision && opt_graph_config_.mixed_precision_type == MixedPrecisionDataType::FP16;

  // add configuration-specific graph changes
  ORT_RETURN_IF_ERROR(BuildInternal(
      should_add_gradient_norm, should_add_gradient_finite_check,
      graph, graph_defs, weight_argdefs, gradient_argdefs, optimizer_state_initializer_names, optimizer_graph_outputs));

  // add zero gradient
  if (is_gradient_accumulation_enabled) {
    ORT_RETURN_IF_ERROR(AddZeroGradientNodes(
        nodearg_name_generator, weight_argdefs, gradient_accumulation_buffers, graph_defs));
  }

  return GraphAugmenter::AugmentGraph(graph, graph_defs);
}

Status OptimizerGraphBuilder::BuildInternal(
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

  bool divide_sample_after_all_reduce = false;
  // const bool is_gradient_accumulation_enabled = opt_graph_config_.gradient_accumulation_steps > 1;
  // add gradient scaling
  ArgDef fused_gradient_argdef;
  ArgDef scale;

  ArgDef grad_multi_factor = AddAllReduceForSampleCount(nodearg_name_generator, gradient_argdefs, graph_defs);
  if (divide_sample_after_all_reduce == true) {
    scale = ArgDef(nodearg_name_generator("one"), graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    graph_defs.AddInitializers({CreateTensorProto<float>(scale.name, 1.0f, {})});
  } else {
    scale = grad_multi_factor;
  }

  
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), false));

  if (divide_sample_after_all_reduce == true) {
    ORT_RETURN_IF_ERROR(ScaleGradWithSampleCount(nodearg_name_generator, gradient_argdefs, graph_defs, grad_multi_factor));
  } 
  // if (is_gradient_accumulation_enabled) {
  //     //We did not divide by acc step, instead, we divide by total sample across all accumulation batches.
  //   // todo : make this scale optional.
  //   //const float scale = 1.0f; // / opt_graph_config_.gradient_accumulation_steps;

  // }
  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;

  if (should_add_gradient_norm) {
    std::vector<ArgDef> gradient_norm_inputs;
    bool megatron_enabled = DistributedRunContext::GroupSize(WorkerGroupType::HorizontalParallel) > 1;
    if (megatron_enabled) {
      int rank_in_hori_group = DistributedRunContext::RankInGroup(WorkerGroupType::HorizontalParallel);
      if (rank_in_hori_group != 0) {
        for (size_t i = 0; i < megatron_partitioned_weight_grad_index_.size(); ++i) {
          gradient_norm_inputs.push_back(gradient_argdefs[megatron_partitioned_weight_grad_index_[i]]);
        }
      }
      ORT_RETURN_IF_ERROR(AddGradientNorm(
          nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef, global_gradient_norm_output_name + "_prior_mega_reduce"));
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

  if (!global_grad_norm_argdef.Exists() && !global_grad_norm_finite_argdef.Exists()) {
    ORT_RETURN_IF_ERROR(AddGradientPassThroughNode(nodearg_name_generator, gradient_argdefs, graph_defs));
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
