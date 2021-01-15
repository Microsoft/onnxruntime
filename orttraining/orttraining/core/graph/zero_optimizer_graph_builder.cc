// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/zero_optimizer_graph_builder.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/graph/graph_augmenter.h"

#include "orttraining/core/framework/distributed_run_context.h"

namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef ORT_USE_NCCL
  return true;
#else
  return false;
#endif
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
    std::vector<ArgDef>& weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> allgather_outputs(weight_argdefs.size());
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    allgather_outputs[i] = ArgDef(weight_argdefs[i].name + "_AllGather_Out",
                                  weight_argdefs[i].type_proto);
  }

  // Add NCCL AllGather node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllGather", kMSDomain, 1},
                                  weight_argdefs,
                                  allgather_outputs,
                                  NodeAttributes(),
                                  "NcclAllGather")});

  weight_argdefs = std::move(allgather_outputs);
  return Status::OK();
}

static Status AddL2NormNcclAllReduce(
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
  ArgDef allreduce_output(norm_argdef.name + "_AllReduce_Out", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  {norm_squared},
                                  {allreduce_output},
                                  {ONNX_NAMESPACE::MakeAttribute("group_type",
                                                                 static_cast<int64_t>(WorkerGroupType::DataParallel))},
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

static std::string GetViewName(const std::string param_name, const size_t view_num) {
  return param_name + "_view_" + std::to_string(view_num);
}

static std::vector<ArgDef> AddPartitionsForParameter(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const std::string& initializer_name,
    const std::vector<TensorShape>& shapes,
    std::unordered_map<std::string, std::string>& updated_weight_names_map) {
  ORT_ENFORCE(shapes.size() == 3, "Invalid shapes vector passed for partitioning.");
  int64_t partition_offset = shapes[0].GetDims()[0];
  int64_t partition_size = shapes[1].GetDims()[0];
  std::vector<ArgDef> view_outputs;
  for (size_t i = 0, view_num = 0; i < shapes.size(); i++) {
    if (shapes[i].Size() > 0) {
      // keep the naming consistent with the "Views" created for the rest of the parameters.
      const auto partition_name = GetViewName(initializer_name, view_num);
      if (i == 1) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto;
        ORT_ENFORCE(graph.GetInitializedTensor(initializer_name, tensor_proto));
        const auto dtype = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_proto->data_type());
        // Shard only fp32 weight for zero stage 1
        ORT_ENFORCE(dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

        // Find the initializer partition to read out.
        auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
        const float* initializer_data = initializer->data<float>();

        // Create new initializer tensor proto.
        ONNX_NAMESPACE::TensorProto initializer_partition;
        initializer_partition.set_name(partition_name);
        initializer_partition.set_data_type(dtype);
        initializer_partition.add_dims(partition_size);
        initializer_partition.set_raw_data(initializer_data + partition_offset, partition_size * sizeof(float));

        //Replace the old initializer with the new one
        graph.RemoveInitializedTensor(initializer_name);
        graph.AddInitializedTensor(initializer_partition);

        //add the modified weight name to get state
        updated_weight_names_map[initializer_name] = partition_name;

        auto partition_argdef = ArgDef(partition_name, graph_defs.CreateTypeProto({partition_size}, dtype));

        view_outputs.push_back(partition_argdef);
      } else {
        auto dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
        auto partition_argdef = ArgDef(partition_name, graph_defs.CreateTypeProto({shapes[i].Size()}, dtype));
        view_outputs.push_back(partition_argdef);
      }
      view_num++;
    }
  }
  return view_outputs;
}

static std::vector<ArgDef> AddViewForParameter(
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef argdef,
    const std::vector<TensorShape>& shapes) {
  std::vector<ArgDef> view_inputs = {argdef};
  std::vector<ArgDef> view_outputs;
  for (size_t i = 0, view_num = 0; i < shapes.size(); i++) {
    const TensorShape& shape = shapes[i];
    if (shape.Size() > 0) {
      const int64_t dims = shape.NumDimensions();

      ArgDef shape_argdef(argdef.name + "_view_shape_" + std::to_string(view_num),
                          graph_defs.CreateTypeProto({dims}, ONNX_NAMESPACE::TensorProto_DataType_INT64));
      graph_defs.AddInitializers({CreateTensorProto<int64_t>(shape_argdef.name, shape.GetDims(), {dims})});

      auto dtype = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(argdef.type_proto->tensor_type().elem_type());
      ArgDef view_argdef(GetViewName(argdef.name, view_num),
                         graph_defs.CreateTypeProto(shape.GetDims(), dtype));

      view_inputs.push_back(shape_argdef);
      view_outputs.push_back(view_argdef);
      view_num++;
    }
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"View", kMSDomain, 1},
                                  view_inputs,
                                  view_outputs,
                                  NodeAttributes(),
                                  argdef.name + "_view")});
  return view_outputs;
}

void PartitionOptimizerState(
    const int64_t partition_offset,
    const int64_t partition_size,
    NameMLValMap& initial_states) {
  for (const auto& moments_prefix : MOMENTS_PREFIXES) {
    const auto initial_state_it = initial_states.find(moments_prefix);
    if (initial_state_it != initial_states.end()) {
      auto* init_tensor = initial_state_it->second.GetMutable<Tensor>();

      OrtValue partitioned;
      TensorShape shape({partition_size});
      auto element_type = init_tensor->DataType();
      const OrtMemoryInfo& info = init_tensor->Location();
      std::unique_ptr<Tensor> p_tensor;

      if (utils::IsPrimitiveDataType<float>(element_type)) {
        float* data_buffer = init_tensor->MutableData<float>();
        p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                    shape,
                                                    data_buffer + partition_offset,
                                                    info);
      } else if (utils::IsPrimitiveDataType<MLFloat16>(element_type)) {
        MLFloat16* data_buffer = init_tensor->MutableData<MLFloat16>();
        p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                    shape,
                                                    data_buffer + partition_offset,
                                                    info);

      } else {
        ORT_THROW("Unsupported type: ", element_type, "for initial optimizer moments.");
      }
      partitioned.Init(p_tensor.release(),
                       DataTypeImpl::GetType<Tensor>(),
                       DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
      initial_states[moments_prefix] = std::move(partitioned);
    }
  }
}

static Status AddParameterPartition(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef weight_argdef,
    ArgDef gradient_argdef,
    const OptimizerNodeConfig& opt_config,
    const std::vector<TensorShape>& view_shapes,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map,
    std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info) {
  std::vector<bool> enabled;
  for (size_t i = 0; i < view_shapes.size(); i++) {
    if (view_shapes[i].Size() > 0) {
      enabled.push_back(i == 1);  // Only the middle partition can be enabled
    }
  }

  // Add View/Partition for weight
  std::vector<ArgDef> weight_views, mixed_precision_weight_views;
  if (opt_config.mixed_precision_weight_arg != nullptr) {
    const NodeArg* weight_arg = graph.GetNodeArg(weight_argdef.name);
    ORT_ENFORCE(weight_arg != nullptr, "Could not find nodearg in graph: " + weight_argdef.name);
    ORT_ENFORCE(!graph_utils::IsGraphInput(graph, weight_arg), "Cannot partition weight that is a part of graph inputs for " + weight_argdef.name);

    // Partition the FP32 weight
    weight_views = AddPartitionsForParameter(graph, graph_defs, weight_argdef.name, view_shapes, updated_weight_names_map);

    // Add View for mixed precision weight.
    ArgDef mixed_precision_weight_argdef(opt_config.mixed_precision_weight_arg->Name(), opt_config.mixed_precision_weight_arg->TypeAsProto());
    mixed_precision_weight_views = AddViewForParameter(graph_defs, mixed_precision_weight_argdef, view_shapes);
    ORT_ENFORCE(mixed_precision_weight_views.size() == enabled.size());
  } else {
    weight_views = AddViewForParameter(graph_defs, weight_argdef, view_shapes);
  }
  ORT_ENFORCE(weight_views.size() == enabled.size());

  // Add View for gradient.
  std::vector<ArgDef> gradient_views = AddViewForParameter(graph_defs, gradient_argdef, view_shapes);
  ORT_ENFORCE(gradient_views.size() == enabled.size());

  weight_argdefs.insert(weight_argdefs.end(), weight_views.begin(), weight_views.end());
  gradient_argdefs.insert(gradient_argdefs.end(), gradient_views.begin(), gradient_views.end());

  const auto& initial_states = opt_config.initial_states;
  // Update Optimizer node configs.
  ORT_ENFORCE(weight_views.size() == gradient_views.size());
  for (size_t i = 0; i < weight_views.size(); i++) {
    OptimizerNodeConfig new_config = opt_config;
    new_config.enabled = enabled[i];

    // Partition initial optimizer state
    if (enabled[i]) {
      weight_partition_info[weight_argdef.name].view_name = weight_views[i].name;

      if (!initial_states.empty()) {
        ORT_ENFORCE(view_shapes.size() == 3, "Invalid view_shapes vector passed for partitioning.");
        int64_t partition_offset = view_shapes[0].GetDims()[0];
        int64_t partition_size = view_shapes[1].GetDims()[0];
        new_config.initial_states = opt_config.initial_states;
        PartitionOptimizerState(partition_offset, partition_size, new_config.initial_states);
      }
    }

    if (opt_config.mixed_precision_weight_arg != nullptr) {
      new_config.mixed_precision_weight_arg = &graph.GetOrCreateNodeArg(mixed_precision_weight_views[i].name, mixed_precision_weight_views[i].type_proto);
    }
    opt_configs.push_back(new_config);
  }

  return Status::OK();
}

static Status ModifyParametersForOptimizerPartitioning(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const OptimizerGraphConfig& opt_graph_config,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    ArgDef& padded_weight_argdef,
    ArgDef& padded_gradient_argdef,
    std::unordered_map<std::string, std::string>& updated_weight_names_map,
    std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info) {
  ORT_ENFORCE(weight_argdefs.size() == gradient_argdefs.size());
  ORT_ENFORCE(weight_argdefs.size() == opt_configs.size());

  const auto elem_type = gradient_argdefs[0].type_proto->tensor_type().elem_type();
  size_t element_size = 0;
  if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    element_size = 4;
  } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
    element_size = 2;
  } else {
    ORT_ENFORCE(false, "Unsupported datatype in gradient tensor");
  }

  // Compute total element count to reduce.
  int64_t total_count = 0;
  int64_t total_bytes = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    ArgDef weight_argdef = weight_argdefs[i];
    ORT_ENFORCE(weight_argdef.type_proto != nullptr);
    const auto& weight_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_shape_proto);

    ArgDef gradient_argdef = gradient_argdefs[i];
    ORT_ENFORCE(gradient_argdef.type_proto != nullptr);
    ORT_ENFORCE(gradient_argdef.type_proto->tensor_type().elem_type() == elem_type);
    const auto& gradient_shape_proto = gradient_argdef.type_proto->tensor_type().shape();
    const TensorShape& gradient_shape = utils::GetTensorShapeFromTensorShapeProto(gradient_shape_proto);

    ORT_ENFORCE(weight_shape == gradient_shape);

    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(weight_shape.Size(), element_size, &tensor_occupied_bytes));

    total_bytes += tensor_occupied_bytes;
    total_count += weight_shape.Size();
  }

  // Compute split points for parameters.
  // Note: the alignment here needs to be kept in-sync with the alignment in nccl_kernels.cc
  const int data_parallel_group_rank = opt_graph_config.data_parallel_group_rank;
  const int data_parallel_group_size = opt_graph_config.data_parallel_group_size;
  const size_t alignment = data_parallel_group_size * 32;
  const size_t padded_buffer_bytes = ((total_bytes + alignment - 1) / alignment) * alignment;

  const size_t rank_bytes = padded_buffer_bytes / data_parallel_group_size;
  const size_t rank_start = data_parallel_group_rank * rank_bytes;
  const size_t rank_end = rank_start + rank_bytes;

  std::vector<OptimizerNodeConfig> new_opt_configs;
  std::vector<ArgDef> new_weight_argdefs;
  std::vector<ArgDef> new_gradient_argdefs;

  size_t offset = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    const OptimizerNodeConfig& opt_config = opt_configs[i];
    ArgDef weight_argdef = weight_argdefs[i];
    ArgDef gradient_argdef = gradient_argdefs[i];

    const auto& tensor_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& tensor_shape = utils::GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
    const size_t tensor_count = tensor_shape.Size();

    const size_t tensor_start = offset;
    const size_t tensor_end = tensor_start + tensor_count * element_size;

    if (tensor_start < rank_end && tensor_end > rank_start) {
      // Parameter is handled by this rank.  There are 4 cases:
      // 1. parameter is fully handled by this rank
      // 2. parameter is split between previous rank and this rank
      // 3. parameter is split between this rank and next rank
      // 4. parameter is split between previous rank, this rank, and next rank
      if (tensor_start >= rank_start && tensor_end <= rank_end) {
        new_opt_configs.push_back(opt_config);
        new_weight_argdefs.push_back(weight_argdef);
        new_gradient_argdefs.push_back(gradient_argdef);
      } else {
        weight_partition_info[weight_argdef.name].original_dim = tensor_shape.GetDims();
        if (tensor_start < rank_start && tensor_end <= rank_end) {
          int64_t bytes_for_previous_rank = rank_start - tensor_start;
          int64_t size_for_previous_rank = std::min<int64_t>(bytes_for_previous_rank / element_size, tensor_count);
          int64_t size_for_current_rank = tensor_count - size_for_previous_rank;

          ORT_ENFORCE(size_for_current_rank > 0);

          std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}, {0}};
          AddParameterPartition(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes,
                                new_opt_configs, new_weight_argdefs, new_gradient_argdefs, updated_weight_names_map, weight_partition_info);
        } else if (tensor_start >= rank_start && tensor_end > rank_end) {
          int64_t bytes_for_current_rank = rank_end - offset;
          int64_t size_for_current_rank = std::min<int64_t>(bytes_for_current_rank / element_size, tensor_count);
          int64_t size_for_next_rank = tensor_count - size_for_current_rank;
          ORT_ENFORCE(size_for_current_rank > 0);

          std::vector<TensorShape> view_shapes = {{0}, {size_for_current_rank}, {size_for_next_rank}};
          AddParameterPartition(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes,
                                new_opt_configs, new_weight_argdefs, new_gradient_argdefs, updated_weight_names_map, weight_partition_info);
        } else {  // tensor_start < rank_start && tensor_end > rank_end
          int64_t bytes_for_previous_rank = rank_start - offset;
          int64_t bytes_for_current_rank = rank_end - rank_start;

          int64_t size_for_previous_rank = std::min<int64_t>(bytes_for_previous_rank / element_size, tensor_count);
          int64_t size_for_current_rank = std::min<int64_t>(bytes_for_current_rank / element_size, tensor_count - size_for_previous_rank);
          int64_t size_for_next_rank = tensor_count - size_for_previous_rank - size_for_current_rank;
          ORT_ENFORCE(size_for_current_rank > 0);

          std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}, {size_for_next_rank}};
          AddParameterPartition(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes,
                                new_opt_configs, new_weight_argdefs, new_gradient_argdefs, updated_weight_names_map, weight_partition_info);
        }
      }
    } else {
      // Parameter is handled by a different rank.
      OptimizerNodeConfig new_config = opt_config;
      new_config.enabled = false;

      new_opt_configs.push_back(new_config);
      new_weight_argdefs.push_back(weight_argdef);
      new_gradient_argdefs.push_back(gradient_argdef);
    }

    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(tensor_count, element_size, &tensor_occupied_bytes));
    offset += tensor_occupied_bytes;
  }

  // weight padding
  const size_t padding_bytes = padded_buffer_bytes - total_bytes;
  const int64_t padding_size = padding_bytes / element_size;

  std::cout << "padded_buffer_bytes " << padded_buffer_bytes << " total_bytes " << total_bytes << "\n";
  std::cout << "padding_bytes " << padding_bytes << " padding_size " << padding_size << "\n";

  if (padding_size > 0) {
    ArgDef weight_padding_argdef("contiguous_weight_buffer_padding",
                                 graph_defs.CreateTypeProto({padding_size}, ONNX_NAMESPACE::TensorProto_DataType(elem_type)));

    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      graph_defs.AddInitializers({CreateTensorProto<float>(weight_padding_argdef.name, 0.0f, {padding_size})});
    } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      graph_defs.AddInitializers({CreateTensorProto<MLFloat16>(weight_padding_argdef.name, MLFloat16(math::floatToHalf(0.0f)), {padding_size})});
    }

    // gradient padding
    ArgDef padding_size_argdef("gradient_padding_size",
                               graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_INT64));
    graph_defs.AddInitializers({CreateTensorProto<int64_t>(padding_size_argdef.name, std::vector<int64_t>{padding_size}, {1})});
    ArgDef gradient_padding_argdef("contiguous_gradient_buffer_padding",
                                   graph_defs.CreateTypeProto({padding_size}, ONNX_NAMESPACE::TensorProto_DataType(elem_type)));

    TensorProto zero_scaler;
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      zero_scaler = ONNX_NAMESPACE::ToTensor<float>(0.f);
    } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      zero_scaler = ONNX_NAMESPACE::ToTensor<MLFloat16>(MLFloat16(math::floatToHalf(0.f)));
    }
    zero_scaler.clear_dims();
    zero_scaler.add_dims(1);

    // gradient padding have to be an activation tensor, so use ConstantOfShape instead of initializer
    graph_defs.AddNodeDefs({NodeDef("ConstantOfShape",
                                    {padding_size_argdef},
                                    {gradient_padding_argdef},
                                    {ONNX_NAMESPACE::MakeAttribute("value", zero_scaler)},
                                    gradient_padding_argdef.name)});

    padded_weight_argdef = weight_padding_argdef;
    padded_gradient_argdef = gradient_padding_argdef;
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

ZeROOptimizerGraphBuilder::ZeROOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map,
    std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs,
                            updated_weight_names_map,
                            weight_partition_info) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1, "ZeRO optimizer graph builder can only be used for distributed training.");
  ORT_ENFORCE(opt_graph_config.use_nccl, "Distributed training with ZeRO is only supported with NCCL.");
  ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status ZeROOptimizerGraphBuilder::BuildInternal(
    bool should_add_gradient_norm,
    bool should_add_gradient_finite_check,
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  // add gradient scaling
  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  const float scale = 1.0f / total_num_accumulations;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), false));

  // handle optimizer partitioning
  ArgDef padded_weight_argdef, padded_gradient_argdef;
  ORT_RETURN_IF_ERROR(ModifyParametersForOptimizerPartitioning(
      graph, graph_defs, opt_graph_config_, opt_configs_, weight_argdefs, gradient_argdefs,
      padded_weight_argdef, padded_gradient_argdef, updated_weight_names_map_, weight_partition_info_));

  // add Reducescatter for gradients
  if (padded_gradient_argdef.Exists()) {
    gradient_argdefs.push_back(padded_gradient_argdef);
    ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(gradient_argdefs, graph_defs));
    gradient_argdefs.pop_back();
  } else {
    ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(gradient_argdefs, graph_defs));
  }

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;

  if (should_add_gradient_norm) {
    auto gradient_norm_inputs = GetGradientNormInputs(gradient_argdefs, opt_configs_);
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;
    ORT_RETURN_IF_ERROR(AddL2NormNcclAllReduce(global_grad_norm_argdef, graph_defs));
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

  // add Allgather for weights
  if (padded_weight_argdef.Exists()) {
    weight_argdefs.push_back(padded_weight_argdef);
    ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(weight_argdefs, graph_defs));
    weight_argdefs.pop_back();
  } else {
    ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(weight_argdefs, graph_defs));
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
