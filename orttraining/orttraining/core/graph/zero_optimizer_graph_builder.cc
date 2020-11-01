// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>

#include <queue>

#include "onnx/defs/attr_proto_util.h"

#include "orttraining/core/graph/zero_optimizer_graph_builder.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_viewer.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

/*
Utility function. The number of paritions (workers) needed to partition a given vector of tensor sizes(size_arr), where the total tensor size cannot be larger than a given number (max_len)
Inputs
  size_arr: A vector of integers, each element in the vector is a tensor size
  max_len: The total tensor size of each parition cannot be larger than the max_len
  partitions: The vector to store the index of element in size_arr, where that element is the last element in the current partition. 
Return value:
  num_workers: The number of partitiones (workers) needed
*/
static int NumberOfWorkers(const std::vector<int64_t>& size_arr, int64_t max_len, std::vector<int64_t>& partitions) {
  int64_t total = 0;
  int num_workers = 1;
  int64_t idx = 0;
  partitions.clear();
  for (auto i : size_arr) {
    total += i;
    if (total > max_len) {
      total = i;
      partitions.push_back(idx - 1);
      num_workers += 1;
    }
    idx += 1;
  }
  //The last index, enforce that the size_arr is not an empty vector
  ORT_ENFORCE(idx > 0);
  partitions.push_back(idx - 1);
  return num_workers;
}

//Binary search to find the most even partition of gradients cross workers

//size_arr: The vector of gradient tensor sizes. Each element is the size of a gradient.
//dp_group: The number of ranks to partition the gradients into
//max_size: The biggest gradient size in the tensors to be partitioned. I.e., the largest number in size_arr.
//total_size: The total size of all gradient tensors added together
//return: lo: The total tensor size of the maximum partitioned group

//In the each iteration of the algorithm, it finds the number of workers needed for the size_arr, given the max length of each partition to be the average of lower(lo) and higher (hi) bounds.
//If the number of workers is less than the data parallel groups (dp_group), it updates the higher bound (hi) to be the current average, and goes to the next iteration.
//Else it is a invalid partition, increase the lower bounds by 1, goest to next iteration
//The algorithm terminates until lo >= hi

static int64_t WorkersPartition(const std::vector<int64_t>& size_arr, int dp_group, int64_t max_size, int64_t total_size, std::vector<int64_t>& partitions) {
  int64_t lo = max_size;
  int64_t hi = total_size;
  int64_t mid = lo + (hi - lo) / 2;
  std::vector<int64_t> tmp_partitions;

  while (lo < hi) {
    mid = lo + (hi - lo) / 2;
    //Collect the number of workers needed given each partition cannot not exceeding
    int num_workers = NumberOfWorkers(size_arr, mid, tmp_partitions);
    // find better optimum in lower half. Here mid is included because we may not get anything better
    if (num_workers <= dp_group) {
      hi = mid;
      partitions = tmp_partitions;
    } else {
      //find better optimum in upper half. Here mid is excluded because it gives required workers > dp_group, which is invalid
      lo = mid + 1;
    }
  }
  return lo;
}

static Status AddNcclReduceForGradients(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    std::vector<ArgDef>& gradient_argdefs,
    std::vector<int64_t>& partitions,
    std::vector<OptimizerNodeConfig>& opt_configs,
    const OptimizerGraphConfig& opt_graph_config,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& output_readies) {
  const int data_parallel_group_rank = opt_graph_config.data_parallel_group_rank;
  ArgDef old_reduce_output;
  for (int i = 0; i < static_cast<int>(partitions.size()); ++i) {
    int64_t ub = partitions[i];
    int64_t lb = i == 0 ? 0 : partitions[i - 1] + 1;
    bool current_rank = (i == data_parallel_group_rank);
    std::vector<ArgDef> reduce_outputs;
    std::vector<ArgDef> reduce_inputs;

    auto node_name = nodearg_name_generator("NcclReduce");
    ArgDef reduce_output(node_name + "Fake_Reduce_Out");

    for (int64_t j = lb; j <= ub; ++j) {
      reduce_inputs.push_back(gradient_argdefs[j]);
      if (current_rank) {
        reduce_output = ArgDef(gradient_argdefs[j].name + "_Reduce_Out", gradient_argdefs[j].type_proto);
        reduce_outputs.push_back(reduce_output);
        gradient_argdefs[j] = reduce_output;
        opt_configs[j].enabled = true;
      } else {
        opt_configs[j].enabled = false;
      }
    }
    if (!current_rank) {
      reduce_outputs.push_back(reduce_output);
    }
    if (i > 0) {
      //fake data edge to enforce dependence
      reduce_inputs.push_back(old_reduce_output);
    }
    old_reduce_output = reduce_outputs[0];
    int64_t has_old_reduce = i > 0;

    std::vector<AttributeProto> attributes({onnx::MakeAttribute("root_rank", int64_t(i)),
                                            onnx::MakeAttribute("num_input_readies", has_old_reduce)});  //,
    auto nd = NodeDef(OpDef{"NcclReduce", kMSDomain, 1},
                      reduce_inputs,
                      reduce_outputs,
                      attributes,
                      node_name);
    nd.SetPriority(-1);
    graph_defs.AddNodeDefs({nd});

    output_readies.push_back(reduce_outputs[0]);
  }
  return Status::OK();
}

static Status AddNcclReduceScatterForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> reducescatter_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    reducescatter_outputs[i] = ArgDef(gradient_argdefs[i].name + "_ReduceScatter_Out",
                                      gradient_argdefs[i].type_proto);
  }

  // Add NCCL ReduceScatter node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclReduceScatter", kMSDomain, 1},
                                  gradient_argdefs,
                                  reducescatter_outputs,
                                  NodeAttributes(),
                                  "NcclReduceScatter")});

  gradient_argdefs = std::move(reducescatter_outputs);
  return Status::OK();
}

static Status AddNcclAllGatherForWeights(
    std::vector<ArgDef>& input_readies,
    std::vector<int64_t>& partitions,
    std::vector<ArgDef>& weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    int64_t max_group_size_val,
    const bool partition_even) {
  std::vector<ArgDef> allgather_outputs(weight_argdefs.size());
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    allgather_outputs[i] = ArgDef(weight_argdefs[i].name + "_AllGather_Out",
                                  weight_argdefs[i].type_proto);
  }

  // Add NCCL AllGather node.
  if (partition_even) {
    graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllGather", kMSDomain, 1},
                                    weight_argdefs,
                                    allgather_outputs,
                                    NodeAttributes(),
                                    "NcclAllGather")});
  } else {
    auto partition = onnx::MakeAttribute("partition", partitions);
    auto max_group_size = onnx::MakeAttribute("max_group_size", int64_t(max_group_size_val));
    auto num_input_readies = onnx::MakeAttribute("num_input_readies", int64_t(input_readies.size()));
    std::vector<ArgDef> inputs = weight_argdefs;
    inputs.insert(inputs.end(), input_readies.begin(), input_readies.end());
    graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllGather", kMSDomain, 1},
                                    inputs,
                                    allgather_outputs,
                                    {partition, max_group_size, num_input_readies},
                                    "NcclAllGather")});
  }

  weight_argdefs = std::move(allgather_outputs);
  return Status::OK();
}

static Status AddL2NormNcclAllReduce(
    std::vector<ArgDef>& input_readies,
    ArgDef& norm_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
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

  // AllReduce the squared L2 norms.
  std::vector<ArgDef> inputs({norm_argdef});
  inputs.insert(inputs.end(), input_readies.begin(), input_readies.end());
  ArgDef allreduce_output(norm_argdef.name + "_AllReduce_Out", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  inputs,
                                  {allreduce_output},
                                  {onnx::MakeAttribute("num_input_readies", int64_t(input_readies.size()))},
                                  allreduce_output.name)});

  // Sqrt the reduced L2 norm.
  ArgDef sqrt_output(norm_argdef.name + "_sqrt", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef("Sqrt",
                                  {allreduce_output},
                                  {sqrt_output},
                                  NodeAttributes(),
                                  sqrt_output.name)});

  norm_argdef = sqrt_output;
  return Status::OK();
}

static std::vector<ArgDef> AddViewForParameter(
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef argdef,
    const std::vector<TensorShape>& shapes) {
  std::vector<ArgDef> view_inputs = {argdef};
  std::vector<ArgDef> view_outputs;

  for (size_t i = 0; i < shapes.size(); i++) {
    const TensorShape& shape = shapes[i];
    const int64_t dims = shape.NumDimensions();

    ArgDef shape_argdef(argdef.name + "_view_shape_" + std::to_string(i),
                        graph_defs.CreateTypeProto({dims}, ONNX_NAMESPACE::TensorProto_DataType_INT64));
    graph_defs.AddInitializers({CreateTensorProto<int64_t>(shape_argdef.name, shape.GetDims(), {dims})});

    auto dtype = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(argdef.type_proto->tensor_type().elem_type());
    ArgDef view_argdef(argdef.name + "_view_" + std::to_string(i),
                       graph_defs.CreateTypeProto(shape.GetDims(), dtype));

    view_inputs.push_back(shape_argdef);
    view_outputs.push_back(view_argdef);
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"View", kMSDomain, 1},
                                  view_inputs,
                                  view_outputs,
                                  NodeAttributes(),
                                  argdef.name + "_view")});

  return view_outputs;
}

static Status AddViewForParameters(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef weight_argdef,
    ArgDef gradient_argdef,
    const OptimizerNodeConfig& opt_config,
    const std::vector<TensorShape>& view_shapes,
    const std::vector<bool>& enabled,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs) {
  // Add View for weight.
  std::vector<ArgDef> weight_views = AddViewForParameter(graph_defs, weight_argdef, view_shapes);
  weight_argdefs.insert(weight_argdefs.end(), weight_views.begin(), weight_views.end());

  // Add View for gradient.
  std::vector<ArgDef> gradient_views = AddViewForParameter(graph_defs, gradient_argdef, view_shapes);
  gradient_argdefs.insert(gradient_argdefs.end(), gradient_views.begin(), gradient_views.end());

  // (Optional) Add View for mixed precision weight.
  std::vector<ArgDef> mixed_precision_weight_views;
  if (opt_config.mixed_precision_weight_arg != nullptr) {
    ArgDef mixed_precision_weight_argdef(opt_config.mixed_precision_weight_arg->Name(), opt_config.mixed_precision_weight_arg->TypeAsProto());
    mixed_precision_weight_views = AddViewForParameter(graph_defs, mixed_precision_weight_argdef, view_shapes);
  }

  // Update Optimizer node configs.
  ORT_RETURN_IF_NOT(weight_views.size() == gradient_views.size());
  for (size_t i = 0; i < weight_views.size(); i++) {
    OptimizerNodeConfig new_config = opt_config;
    new_config.enabled = enabled[i];

    if (opt_config.mixed_precision_weight_arg != nullptr) {
      new_config.mixed_precision_weight_arg = &graph.GetOrCreateNodeArg(mixed_precision_weight_views[i].name, mixed_precision_weight_views[i].type_proto);
    }

    opt_configs.push_back(new_config);
  }

  return Status::OK();
}

static Status ModifyParametersForOptimizerPartitioningByBoundary(
    const OptimizerGraphConfig& opt_graph_config,
    std::vector<ArgDef>& gradient_argdefs,
    std::vector<int64_t>& partitions,
    int64_t& max_group_size) {
  std::vector<int64_t> size_arr;
  int64_t max_size = 0;
  int64_t total_size = 0;
  for (const auto& gradient_argdef : gradient_argdefs) {
    ORT_ENFORCE(gradient_argdef.type_proto != nullptr);
    const auto& gradient_shape_proto = gradient_argdef.type_proto->tensor_type().shape();
    const TensorShape& gradient_shape = utils::GetTensorShapeFromTensorShapeProto(gradient_shape_proto);

    total_size += gradient_shape.Size();
    ORT_ENFORCE(total_size > 0);
    if (max_size < gradient_shape.Size()) max_size = gradient_shape.Size();
    size_arr.push_back(gradient_shape.Size());
  }

  //Bucketing the gradients on boundary
  const int dp_group = opt_graph_config.data_parallel_group_size;
  max_group_size = WorkersPartition(size_arr, dp_group, max_size, total_size, partitions);
  //The partitions have to be the same (TODO: smaller ?) as the data parallel group size
  ORT_ENFORCE((int)(partitions.size()) == dp_group);
  //The last element in partitons index should be the last index of gradients
  ORT_ENFORCE(partitions.back() == int(gradient_argdefs.size() - 1));

  return Status::OK();
}

static Status ModifyParametersForOptimizerPartitioning(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const OptimizerGraphConfig& opt_graph_config,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs) {
  ORT_ENFORCE(weight_argdefs.size() == gradient_argdefs.size());
  ORT_ENFORCE(weight_argdefs.size() == opt_configs.size());

  // Compute total element count to reduce.
  int64_t total_count = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    ArgDef weight_argdef = weight_argdefs[i];
    ORT_ENFORCE(weight_argdef.type_proto != nullptr);
    const auto& weight_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_shape_proto);

    ArgDef gradient_argdef = gradient_argdefs[i];
    ORT_ENFORCE(gradient_argdef.type_proto != nullptr);
    const auto& gradient_shape_proto = gradient_argdef.type_proto->tensor_type().shape();
    const TensorShape& gradient_shape = utils::GetTensorShapeFromTensorShapeProto(gradient_shape_proto);

    ORT_ENFORCE(weight_shape == gradient_shape);
    total_count += weight_shape.Size();
  }

  // Compute split points for parameters.
  // Note: the alignment here needs to be kept in-sync with the alignment in nccl_kernels.cc
  const int data_parallel_group_rank = opt_graph_config.data_parallel_group_rank;
  const int data_parallel_group_size = opt_graph_config.data_parallel_group_size;
  const int64_t alignment = data_parallel_group_size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t rank_count = padded_count / data_parallel_group_size;
  const int64_t rank_start = data_parallel_group_rank * rank_count;
  const int64_t rank_end = rank_start + rank_count;

  std::vector<OptimizerNodeConfig> new_opt_configs;
  std::vector<ArgDef> new_weight_argdefs;
  std::vector<ArgDef> new_gradient_argdefs;

  int64_t offset = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    const OptimizerNodeConfig& opt_config = opt_configs[i];
    ArgDef weight_argdef = weight_argdefs[i];
    ArgDef gradient_argdef = gradient_argdefs[i];

    const auto& tensor_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& tensor_shape = utils::GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
    const int64_t tensor_count = tensor_shape.Size();

    if (offset < rank_end && offset + tensor_count > rank_start) {
      // Parameter is handled by this rank.  There are 4 cases:
      // 1. parameter is fully handled by this rank
      // 2. parameter is split between previous rank and this rank
      // 3. parameter is split between this rank and next rank
      // 4. parameter is split between previous rank, this rank, and next rank
      if (offset >= rank_start && offset + tensor_count <= rank_end) {
        new_opt_configs.push_back(opt_config);
        new_weight_argdefs.push_back(weight_argdef);
        new_gradient_argdefs.push_back(gradient_argdef);
      } else if (offset < rank_start && offset + tensor_count <= rank_end) {
        int64_t size_for_previous_rank = rank_start - offset;
        int64_t size_for_current_rank = offset + tensor_count - rank_start;
        std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}};
        std::vector<bool> enabled = {false, true};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      } else if (offset >= rank_start && offset + tensor_count > rank_end) {
        int64_t size_for_current_rank = rank_end - offset;
        int64_t size_for_next_rank = offset + tensor_count - rank_end;
        std::vector<TensorShape> view_shapes = {{size_for_current_rank}, {size_for_next_rank}};
        std::vector<bool> enabled = {true, false};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      } else {  // offset < rank_start && offset + tensor_count > rank_end
        int64_t size_for_previous_rank = rank_start - offset;
        int64_t size_for_current_rank = rank_end - rank_start;
        int64_t size_for_next_rank = offset + tensor_count - rank_end;
        std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}, {size_for_next_rank}};
        std::vector<bool> enabled = {false, true, false};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      }
    } else {
      // Parameter is handled by a different rank.
      OptimizerNodeConfig new_config = opt_config;
      new_config.enabled = false;

      new_opt_configs.push_back(new_config);
      new_weight_argdefs.push_back(weight_argdef);
      new_gradient_argdefs.push_back(gradient_argdef);
    }

    offset += tensor_count;
  }

  // Update outputs.
  opt_configs = std::move(new_opt_configs);
  weight_argdefs = std::move(new_weight_argdefs);
  gradient_argdefs = std::move(new_gradient_argdefs);
  return Status::OK();
}

static std::vector<ArgDef> GetGradientNormInputs(
    const std::vector<ArgDef>& gradient_argdefs,
    const std::vector<OptimizerNodeConfig>& opt_configs) {
  std::vector<ArgDef> inputs;
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    if (opt_configs[i].enabled) {
      inputs.push_back(gradient_argdefs[i]);
    }
  }

  return inputs;
}

static Status GetGradientArgsInTopoOrder(
    Graph& graph,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& gradient_argdefs) {
  GraphViewer gv(graph);
  const auto& node_indices = gv.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
  std::vector<std::string> gradient_names;
  for (auto& g_argdef : gradient_argdefs) {
    gradient_names.push_back(g_argdef.name);
  }

  std::vector<ArgDef> weight_argdefs_in_topo_order;
  std::vector<ArgDef> gradient_argdefs_in_topo_order;
  std::vector<OptimizerNodeConfig> opt_configs_in_topo_order;

  for (auto& n_idx : node_indices) {
    auto n_output_defs = gv.GetNode(n_idx)->OutputDefs();
    for (const auto* output_def : n_output_defs) {
      auto itr = std::find(gradient_names.begin(), gradient_names.end(), output_def->Name());
      if (itr != gradient_names.end()) {
        gradient_argdefs_in_topo_order.push_back(gradient_argdefs.at(std::distance(gradient_names.begin(), itr)));
        weight_argdefs_in_topo_order.push_back(weight_argdefs.at(std::distance(gradient_names.begin(), itr)));
        opt_configs_in_topo_order.push_back(opt_configs.at(std::distance(gradient_names.begin(), itr)));
      }
    }
  }

  ORT_ENFORCE(gradient_argdefs_in_topo_order.size() == gradient_argdefs.size());
  ORT_ENFORCE(weight_argdefs_in_topo_order.size() == gradient_argdefs.size());
  ORT_ENFORCE(opt_configs_in_topo_order.size() == opt_configs.size());
  gradient_argdefs = std::move(gradient_argdefs_in_topo_order);
  weight_argdefs = std::move(weight_argdefs_in_topo_order);
  opt_configs = std::move(opt_configs_in_topo_order);
  return Status::OK();
}

ZeROOptimizerGraphBuilder::ZeROOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs),
      stage_(opt_graph_config.deepspeed_zero.stage) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1, "ZeRO optimizer graph builder can only be used for distributed training.");
  ORT_ENFORCE(opt_graph_config.use_nccl, "Distributed training with ZeRO is only supported with NCCL.");
  ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status ZeROOptimizerGraphBuilder::BuildInternal(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  std::vector<int64_t> partitions;            //The vector to hold the partition result of gradient tensors. For stage=1, it should be empty
  std::vector<ArgDef> reduce_output_readies;  //The vector to hold nodes to enforce the same order for reduce nodes. This should be empty for stage 1
  int64_t max_group_size = 0;

  ORT_ENFORCE(stage_ == 1 || stage_ == 2);
  if (stage_ == 2) {
    ORT_ENFORCE(opt_graph_config_.gradient_accumulation_steps == 1);
  }

  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  if (stage_ == 1) {
    // handle optimizer partitioning
    ORT_RETURN_IF_ERROR(ModifyParametersForOptimizerPartitioning(
        graph, graph_defs, opt_graph_config_, opt_configs_, weight_argdefs, gradient_argdefs));
  } else {
    //Get gradients in topological order, updates the weight, gradients and opt_configs_ following that order
    ORT_RETURN_IF_ERROR(GetGradientArgsInTopoOrder(graph, weight_argdefs, opt_configs_, gradient_argdefs));

    // handle optimizer partitioning
    ORT_RETURN_IF_ERROR(ModifyParametersForOptimizerPartitioningByBoundary(opt_graph_config_, gradient_argdefs, partitions, max_group_size));
  }

  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  const float scale = 1.0f / total_num_accumulations;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), false, partitions));

  if (stage_ == 1) {
    // add Reducescatter for gradients;
    ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(gradient_argdefs, graph_defs));
  } else {
    //add Reduce for gradients, update the "enabled" flag in opt_configs_ based on rank. Update the gradient args to reduce outputs
    ORT_RETURN_IF_ERROR(AddNcclReduceForGradients(nodearg_name_generator, gradient_argdefs, partitions, opt_configs_, opt_graph_config_, graph_defs, reduce_output_readies));
  }

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    auto gradient_norm_inputs = GetGradientNormInputs(gradient_argdefs, opt_configs_);
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;

    ORT_RETURN_IF_ERROR(AddL2NormNcclAllReduce(reduce_output_readies, global_grad_norm_argdef, graph_defs));

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

  // add Allgather for weights
  bool partition_even = (stage_ == 1);
  ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(reduce_output_readies, partitions, weight_argdefs, graph_defs, max_group_size, partition_even));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
