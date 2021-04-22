// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/random_seed.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

struct OpInfo {
  OpInfo(const std::string& op_type,
         const std::initializer_list<OperatorSetVersion>& supported_versions,
         const std::string& domain = kOnnxDomainAlias,
         const size_t output_count = 1) : op_type(op_type),
                                          supported_versions(supported_versions),
                                          domain(domain),
                                          output_count(output_count){};

  std::string op_type;
  std::initializer_list<OperatorSetVersion> supported_versions;
  std::string domain;
  size_t output_count;
};

const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1_13 = {1, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1_11_13 = {1, 11, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v2_11_13 = {2, 11, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v5_13 = {5, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v1_6_7_13 = {1, 6, 7, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v7_13 = {7, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v9 = {9};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v9_13 = {9, 13};
const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> opset_v12_13 = {12, 13};
const OpInfo add_info = OpInfo("Add", opset_v7_13);
const OpInfo split_info = OpInfo("Split", opset_v2_11_13, kOnnxDomainAlias, 3);
const OpInfo reshape_info = OpInfo("Reshape", opset_v5_13);
const OpInfo transpose_info = OpInfo("Transpose", opset_v1_13);
const OpInfo matmul_info = OpInfo("MatMul", opset_v9_13);
const OpInfo div_info = OpInfo("Div", opset_v7_13);
const OpInfo mul_info = OpInfo("Mul", opset_v1_6_7_13);
const OpInfo sub_info = OpInfo("Sub", opset_v7_13);
const OpInfo softmax_info = OpInfo("Softmax", opset_v1_11_13);
const OpInfo dropout_info = OpInfo("Dropout", opset_v12_13);
const OpInfo where_info = OpInfo("Where", opset_v9);

struct NodeInfo {
  NodeInfo(const std::vector<OpInfo>& op_infos,
           const bool required = true) : op_infos(op_infos),
                                         required(required){};

  std::vector<OpInfo> op_infos;
  bool required;
};

// Check if it's an expected node given the op infos and provider type.
static bool IsExpectedOpAndProvider(const Node& node,
                                    const OpInfo& op_info,
                                    ProviderType provider_type) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_info.op_type, op_info.supported_versions, op_info.domain) &&
         node.GetExecutionProviderType() == provider_type &&
         node.GetOutputEdgesCount() == op_info.output_count;
}

// Try to match a linear sub-graph pattern given a list of graph node infos, input node excluded.
// Each node info entry contains a vector of all possible Op infos, and a flag of required or not.
// All visited nodes will be pushed back to a given node vector.
static bool MatchLinearPattern(Graph& graph,
                               Node* node,
                               ProviderType provider_type,
                               const std::vector<NodeInfo>& node_infos,
                               std::vector<Node*>& sub_graph_node_ptrs) {
  Node* curr_node_ptr = node;
  if (curr_node_ptr->GetOutputEdgesCount() == 0) {
    return node_infos.size() == 0;
  }

  for (const auto& node_info : node_infos) {
    Node* next_node_ptr = graph.GetNode(curr_node_ptr->OutputNodesBegin()->Index());
    bool has_matched_op = false;
    for (const auto& op_info : node_info.op_infos) {
      if (IsExpectedOpAndProvider(*next_node_ptr, op_info, provider_type)) {
        has_matched_op = true;
        break;
      }
    }

    sub_graph_node_ptrs.push_back(has_matched_op ? next_node_ptr : nullptr);
    if (has_matched_op) {
      curr_node_ptr = next_node_ptr;
    } else if (node_info.required) {
      return false;
    }
  }

  return true;
}

// std::hash only guarantee deterministic value in single execution of a program.
// So use this simple hash to generate dropout seed by name.
static uint32_t HashName(const std::string& name) {
  uint32_t hash = 0;
  for (char const& c : name) {
    hash = hash * 101 + c;
  }

  return hash;
}

static bool FilterConsumeNodeWithOpType(const Node& node) {
  return node.OpType().compare("MatMul") == 0;
}

static void ReplaceNodeInput(Graph& graph, Node& node, int input_arg_index, Node& target_node, int output_arg_index,
                             const std::function<bool(const Node& node)> filter = {}) {
  const Node::EdgeEnd* edge = graph_utils::GetInputEdge(node, input_arg_index);
  if (nullptr == edge) {  // handle input/initializer
    graph_utils::ReplaceNodeInput(node, input_arg_index, *(target_node.MutableOutputDefs()[output_arg_index]));
  } else {
    auto input_node = const_cast<Node*>(&edge->GetNode());
    graph_utils::ReplaceDownstreamNodeInput(graph, *input_node, edge->GetDstArgIndex(), target_node, output_arg_index, filter);
  }
}

template <class T>
void MegatronTransformer::PartitionBufferByColumn(const T* input,
                                                  const int64_t row_count,
                                                  const int64_t column_count,
                                                  const int64_t column_stride,
                                                  const int stride,
                                                  std::vector<T>& result) const {
  const int64_t column_stride_partition = column_stride / horizontal_parallel_size_;

  const int64_t stride_partition_column_offset = horizontal_parallel_rank_ * column_stride_partition;
  for (auto row_index = 0; row_index < row_count; row_index++) {
    const auto row_offset = row_index * column_count;
    for (auto stride_index = 0; stride_index < stride; stride_index++) {
      const auto column_offset = row_offset + stride_index * column_stride + stride_partition_column_offset;
      std::copy(input + column_offset, input + column_offset + column_stride_partition, std::back_inserter(result));
    }
  }
}

bool MegatronTransformer::PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                                                  ONNX_NAMESPACE::TensorProto& initializer_partition,
                                                  int stride) const {
  const std::string original_name = input_arg.Name();
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  if (!graph.GetInitializedTensor(original_name, tensor_proto)) {
    LOGS_DEFAULT(WARNING) << "PartitionWeightByColumn: " << original_name << " is not an initializer";
    return false;
  }
  auto data_type = tensor_proto->data_type();
  const ONNX_NAMESPACE::TensorShapeProto* shape = input_arg.Shape();
  int rank = shape->dim_size();
  int64_t row_count;
  int64_t column_count;

  if (rank == 2 && utils::HasDimValue(shape->dim(0)) && utils::HasDimValue(shape->dim(1))) {
    row_count = shape->dim(0).dim_value();
    column_count = shape->dim(1).dim_value();
    weight_partition_info_[original_name].original_dim = std::vector<int64_t>{row_count, column_count};
  } else if (rank == 1) {
    row_count = 1;
    column_count = shape->dim(0).dim_value();
    weight_partition_info_[original_name].original_dim = std::vector<int64_t>{column_count};
  } else {
    LOGS_DEFAULT(WARNING) << "Initializer tensor's rank is " << rank << " (expected to be 1 or 2).";
    return false;
  }

  if (column_count % (horizontal_parallel_size_ * stride) != 0) {
    LOGS_DEFAULT(WARNING) << "last dim " << column_count
                          << " is not divisible by horizontal_parallel_size_ times stride "
                          << (horizontal_parallel_size_ * stride) << ", not supported currently.";
    return false;
  }

  if (stride > 1) {
    LOGS_DEFAULT(WARNING) << "Checkpointing is not currently supported for graphs requiring partitioning of weight with stride > 1";
  }

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  std::string new_initializer_name = original_name + "_column_rank_" + std::to_string(horizontal_parallel_rank_);

  initializer_partition.set_name(new_initializer_name);
  initializer_partition.set_data_type(data_type);

  int64_t column_partition = column_count / horizontal_parallel_size_;
  int64_t column_stride = column_count / stride;

  std::vector<int64_t> new_shape;
  if (rank == 2) {
    initializer_partition.add_dims(row_count);
    new_shape.push_back(row_count);
  }

  initializer_partition.add_dims(column_partition);
  new_shape.push_back(column_partition);
  const int64_t element_count = row_count * column_partition;

  std::vector<float> result;
  result.reserve(element_count);

  PartitionBufferByColumn(a_weight, row_count, column_count, column_stride, stride, result);
  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));

  // Partition initial optimizer state if available
  const auto optim_state_it = initial_optimizer_states_.find(original_name);
  if (optim_state_it != initial_optimizer_states_.end()) {
    auto& initial_states = optim_state_it->second;
    // partition moments same way as the weight
    for (const auto& moments_prefix : training::MOMENTS_PREFIXES) {
      const auto initial_state_it = initial_states.find(moments_prefix);
      if (initial_state_it != initial_states.end()) {
        auto* init_tensor = initial_state_it->second.GetMutable<Tensor>();

        OrtValue partitioned;
        auto element_type = init_tensor->DataType();
        TensorShape partition_shape(new_shape);
        std::unique_ptr<Tensor> p_tensor;

        if (utils::IsPrimitiveDataType<float>(element_type)) {
          float* data_buffer = init_tensor->MutableData<float>();

          // allocate temporary memory to get the column partitioned state
          std::vector<float> result_buffer;
          result_buffer.reserve(element_count);
          PartitionBufferByColumn(data_buffer, row_count, column_count, column_stride, stride, result_buffer);

          // We need to maintain the initial optimizer states as an OrtValue,
          // which is converted eventually to a TensorProto in the optimizer builder
          // after Megatron and Zero partitioning. This approach saves CPU memory
          // as creating a TensorProto involves a copy, and by delaying the copy until
          // after the partitioning results in a smaller copy only for the optimizer
          // states currently present on the rank.
          // Allocate a new buffer to hold the partitioned optimizer state
          // as column partitioning cannot re-use the original
          // buffer as it is a non-contiguous read
          auto alloc = cpu_execution_provider_.GetAllocator(0, OrtMemTypeDefault);
          p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                      partition_shape,
                                                      alloc);
          float* out_buffer = p_tensor->MutableData<float>();
          memcpy(out_buffer, result_buffer.data(), sizeof(float) * element_count);
        } else if (utils::IsPrimitiveDataType<MLFloat16>(element_type)) {
          MLFloat16* data_buffer = init_tensor->MutableData<MLFloat16>();

          // allocate temporary memory to get the column partitioned state
          std::vector<MLFloat16> result_buffer;
          result_buffer.reserve(element_count);
          PartitionBufferByColumn(data_buffer, row_count, column_count, column_stride, stride, result_buffer);

          // allocate a new buffer as column partitioning cannot re-use the original
          // buffer as it is a non-contiguous read on original buffer
          auto alloc = cpu_execution_provider_.GetAllocator(0, OrtMemTypeDefault);
          p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                      partition_shape,
                                                      alloc);
          MLFloat16* out_buffer = p_tensor->MutableData<MLFloat16>();
          memcpy(out_buffer, result_buffer.data(), sizeof(MLFloat16) * element_count);
        } else {
          ORT_THROW("Unsupported type: ", element_type, "for initial optimizer moments.");
        }
        partitioned.Init(p_tensor.release(),
                         DataTypeImpl::GetType<Tensor>(),
                         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
        initial_states[moments_prefix] = std::move(partitioned);
      } else {
        LOGS_DEFAULT(WARNING) << "Initial value for optimizer state: " << moments_prefix
                              << " not found for weight: " << original_name;
      }
    }
  }

  weight_partition_info_[original_name].megatron_row_partition = 0;
  weight_partition_info_[original_name].partition_name = new_initializer_name;
  weight_partition_info_[original_name].weight_partitioned = true;

  return true;
}

bool MegatronTransformer::PartitionWeightByRow(const Graph& graph, const NodeArg& input_arg,
                                               ONNX_NAMESPACE::TensorProto& initializer_partition) const {
  const std::string original_name = input_arg.Name();
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  if (!graph.GetInitializedTensor(original_name, tensor_proto)) {
    LOGS_DEFAULT(WARNING) << "PartitionWeightByRow: " << original_name << " is not an initializer";
    return false;
  }

  auto data_type = tensor_proto->data_type();
  const ONNX_NAMESPACE::TensorShapeProto* shape = input_arg.Shape();
  int rank = shape->dim_size();
  int64_t row_count;
  int64_t column_count;

  if (rank == 2 && utils::HasDimValue(shape->dim(0)) && utils::HasDimValue(shape->dim(1))) {
    row_count = shape->dim(0).dim_value();
    column_count = shape->dim(1).dim_value();
    weight_partition_info_[original_name].original_dim = std::vector<int64_t>{row_count, column_count};
  } else if (rank == 1) {
    row_count = shape->dim(0).dim_value();
    column_count = 1;
    weight_partition_info_[original_name].original_dim = std::vector<int64_t>{row_count};
  } else {
    LOGS_DEFAULT(WARNING) << "Initializer tensor's rank is more than " << rank
                          << " (expected to be 1 or 2).";
    return false;
  }

  if (row_count % horizontal_parallel_size_ != 0) {
    LOGS_DEFAULT(WARNING) << "first dim " << row_count << " is not divisible by horizontal parallel size"
                          << horizontal_parallel_size_ << ", not supported currently.";
    return false;
  }
  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* a_weight = initializer->data<float>();

  std::string new_initializer_name = original_name + "_row_rank_" + std::to_string(horizontal_parallel_rank_);

  initializer_partition.set_name(new_initializer_name);
  initializer_partition.set_data_type(data_type);

  int64_t row_partition = row_count / horizontal_parallel_size_;

  std::vector<int64_t> new_shape;
  initializer_partition.add_dims(row_partition);
  new_shape.push_back(row_partition);
  if (rank == 2) {
    initializer_partition.add_dims(column_count);
    new_shape.push_back(column_count);
  }
  const int64_t element_count = row_partition * column_count;

  std::vector<float> result;
  result.reserve(element_count);

  const int64_t row_index_offset = horizontal_parallel_rank_ * row_partition;
  memcpy(result.data(), a_weight + row_index_offset * column_count, sizeof(float) * element_count);
  initializer_partition.set_raw_data(result.data(), element_count * sizeof(float));

  // Partition initial optimizer state if available
  const auto optim_state_it = initial_optimizer_states_.find(original_name);
  if (optim_state_it != initial_optimizer_states_.end()) {
    auto& initial_states = optim_state_it->second;
    for (const auto& moments_prefix : training::MOMENTS_PREFIXES) {
      const auto initial_state_it = initial_states.find(moments_prefix);
      if (initial_state_it != initial_states.end()) {
        auto* init_tensor = initial_state_it->second.GetMutable<Tensor>();

        OrtValue partitioned;
        auto element_type = init_tensor->DataType();
        TensorShape partition_shape(new_shape);
        const OrtMemoryInfo& info = init_tensor->Location();
        std::unique_ptr<Tensor> p_tensor;

        if (utils::IsPrimitiveDataType<float>(element_type)) {
          float* data_buffer = init_tensor->MutableData<float>();

          p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                      partition_shape,
                                                      data_buffer + row_index_offset * column_count,
                                                      info);
        } else if (utils::IsPrimitiveDataType<MLFloat16>(element_type)) {
          MLFloat16* data_buffer = init_tensor->MutableData<MLFloat16>();

          p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                      partition_shape,
                                                      data_buffer + row_index_offset * column_count,
                                                      info);

        } else {
          ORT_THROW("Unsupported type: ", element_type, "for initial optimizer moments.");
        }
        partitioned.Init(p_tensor.release(),
                         DataTypeImpl::GetType<Tensor>(),
                         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
        initial_states[moments_prefix] = std::move(partitioned);
      } else {
        LOGS_DEFAULT(WARNING) << "Initial value for optimizer state: " << moments_prefix
                              << " not found for weight: " << original_name;
      }
    }
  }

  weight_partition_info_[original_name].megatron_row_partition = 1;
  weight_partition_info_[original_name].partition_name = new_initializer_name;
  weight_partition_info_[original_name].weight_partitioned = true;
  return true;
}

Status MegatronTransformer::TransformGPT2MLP(Graph& graph, bool& modified,
                                             std::vector<Node*>& nodes_to_clear_shape,
                                             int32_t& counter,
                                             NodeIndex node_index) const {
  auto skip_status = common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Skip BART Attention megatron transformation");

  auto& node = *graph.GetNode(node_index);

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9, 13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  if (node.GetInputEdgesCount() > 0) {
    Node& matmul_input_node = const_cast<Node&>(*(node.InputNodesBegin()));
    if (matmul_input_node.OpType().compare("MegatronF") == 0) {
      return skip_status;
    }
  }

  Node& add_node = *graph.GetNode(node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13}) ||
      add_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
      add_node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  Node& gelu_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
  if (!(graph_utils::IsSupportedOptypeVersionAndDomain(gelu_node, "Gelu", {1}, kMSDomain) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(gelu_node, "FastGelu", {1}, kMSDomain)) ||
      gelu_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
      gelu_node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  Node& matmul2_node = *graph.GetNode(gelu_node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul2_node, "MatMul", {9, 13}) ||
      matmul2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
      matmul2_node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  Node& add2_node = *graph.GetNode(matmul2_node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13}) ||
      add2_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
      add2_node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {&node, &add_node, &gelu_node,
                                                           &matmul2_node});

  auto a_weight_arg = node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto a_weight_initializer_partition;
  if (!PartitionWeightByColumn(graph, *a_weight_arg, a_weight_initializer_partition)) {
    return skip_status;
  }

  auto a_bias_arg = add_node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto a_bias_initializer_partition;
  if (!PartitionWeightByColumn(graph, *a_bias_arg, a_bias_initializer_partition)) {
    return skip_status;
  }

  auto b_weight_arg = matmul2_node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto b_weight_initializer_partition;
  if (!PartitionWeightByRow(graph, *b_weight_arg, b_weight_initializer_partition)) {
    return skip_status;
  }

  NodeArg& a_weight_partition_arg = graph_utils::AddInitializer(graph, a_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(node, 1, a_weight_partition_arg);
  updated_weight_names_.insert({a_weight_arg->Name(), a_weight_partition_arg.Name()});

  NodeArg& a_bias_partition_arg = graph_utils::AddInitializer(graph, a_bias_initializer_partition);
  graph_utils::ReplaceNodeInput(add_node, 1, a_bias_partition_arg);
  updated_weight_names_.insert({b_weight_arg->Name(), a_bias_partition_arg.Name()});

  NodeArg& b_weight_partition_arg = graph_utils::AddInitializer(graph, b_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(matmul2_node, 1, b_weight_partition_arg);
  updated_weight_names_.insert({a_bias_arg->Name(), b_weight_partition_arg.Name()});

  graph.RemoveInitializedTensor(a_weight_arg->Name());
  graph.RemoveInitializedTensor(b_weight_arg->Name());
  graph.RemoveInitializedTensor(a_bias_arg->Name());

  const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
  auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
  auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronF_Output"), &mlp_f_type_info);
  Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronF"),
                                   "MegatronF",
                                   "MLP MegatronF",
                                   mlp_f_input_defs,
                                   {&mlp_f_out_arg}, {}, kMSDomain);
  mlp_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
  ReplaceNodeInput(graph, node, 0, mlp_f_node, 0, FilterConsumeNodeWithOpType);

  const std::vector<NodeArg*> mlp_g_input_defs{matmul2_node.MutableOutputDefs()[0]};
  auto mlp_g_type_info = *matmul2_node.MutableOutputDefs()[0]->TypeAsProto();
  auto& mlp_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MLP_MegatronG_Output"), &mlp_g_type_info);
  Node& mlp_g_node = graph.AddNode(graph.GenerateNodeName("MLP_MegatronG"),
                                   "MegatronG",
                                   "MLP MegatronG",
                                   mlp_g_input_defs,
                                   {&mlp_g_out_arg}, {}, kMSDomain);
  mlp_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
  mlp_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
  graph_utils::ReplaceDownstreamNodeInput(graph, matmul2_node, 0, mlp_g_node, 0);
  modified = true;
  counter++;

  return Status::OK();
}

/*
DenseWeight -- Transpose \
               MatMul -- BiasGelu -- Dropout -- MatMul -- Add -- Dropout
*/
Status MegatronTransformer::TransformBARTMLP(Graph& graph, bool& modified,
                                             std::vector<Node*>& nodes_to_clear_shape,
                                             std::unordered_set<Node*>& dropout_nodes_to_transform,
                                             int32_t& counter,
                                             NodeIndex node_index) const {
  auto skip_status = common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Skip BART Attention megatron transformation");

  auto& node = *graph.GetNode(node_index);
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9, 13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }
  Node* second_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
  Node* first_op = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
  if (node.GetInputEdgesCount() > 0) {
    if (second_op == nullptr) {
      return skip_status;
    }
    if (first_op != nullptr && first_op->OpType().compare("MegatronF") == 0) {
      return skip_status;
    }

    if (second_op->OpType().compare("Transpose") != 0) {
      return skip_status;
    }
  } else {
    return skip_status;
  }
  // check if transpose is only 2-dim
  if (!optimizer_utils::IsAttributeWithExpectedValues(*second_op, "perm", {1LL, 0LL})) {
    return skip_status;
  }
  ProviderType provider_type = node.GetExecutionProviderType();

  Node* biasgelu_node_ptr = graph.GetNode(node.OutputNodesBegin()->Index());
  Node& biasgelu_node = *biasgelu_node_ptr;
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(biasgelu_node, "BiasGelu", {1}, kMSDomain) ||
      biasgelu_node.GetExecutionProviderType() != provider_type ||
      biasgelu_node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  // Either Dropout->Matmul or just Matmul
  Node* dropout_node = nullptr;
  Node* next_node = graph.GetNode(biasgelu_node.OutputNodesBegin()->Index());
  if (IsExpectedOpAndProvider(*next_node, dropout_info, provider_type)) {
    dropout_node = next_node;
    next_node = graph.GetNode(dropout_node->OutputNodesBegin()->Index());
  }
  if (!IsExpectedOpAndProvider(*next_node, matmul_info, provider_type)) {
    return skip_status;
  }
  Node& matmul2_node = *next_node;

  Node& add_node = *graph.GetNode(matmul2_node.OutputNodesBegin()->Index());
  if (!IsExpectedOpAndProvider(add_node, add_info, provider_type)) {
    return skip_status;
  }
  Node& dropout2_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
  if (!IsExpectedOpAndProvider(dropout2_node, dropout_info, provider_type)) {
    return skip_status;
  }
  Node* transpose_op_ptr = const_cast<Node*>(graph.GetProducerNode(matmul2_node.MutableInputDefs()[1]->Name()));
  if (transpose_op_ptr == nullptr || !IsExpectedOpAndProvider(*transpose_op_ptr, transpose_info, provider_type)) {
    return skip_status;
  }

  nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {&node, second_op, &biasgelu_node,
                                                           &matmul2_node, transpose_op_ptr});
  if (dropout_node != nullptr) {
    nodes_to_clear_shape.insert(nodes_to_clear_shape.end(), {dropout_node});
  }

  auto dense_wi_weight_arg = second_op->MutableInputDefs()[0];
  ONNX_NAMESPACE::TensorProto dense_wi_weight_initializer_partition;
  if (!PartitionWeightByRow(graph, *dense_wi_weight_arg, dense_wi_weight_initializer_partition)) {
    return skip_status;
  }

  //since the bias doesn't get transposed, partitioning by col
  auto dense_wi_bias_arg = biasgelu_node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto dense_wi_bias_initializer_partition;
  if (!PartitionWeightByColumn(graph, *dense_wi_bias_arg, dense_wi_bias_initializer_partition)) {
    return skip_status;
  }

  auto dense_wo_weight_arg = transpose_op_ptr->MutableInputDefs()[0];
  ONNX_NAMESPACE::TensorProto dense_wo_weight_initializer_partition;
  if (!PartitionWeightByColumn(graph, *dense_wo_weight_arg, dense_wo_weight_initializer_partition)) {
    return skip_status;
  }

  NodeArg& dense_wi_weight_partition_arg = graph_utils::AddInitializer(graph, dense_wi_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(*second_op, 0, dense_wi_weight_partition_arg);
  updated_weight_names_.insert({dense_wi_weight_arg->Name(), dense_wi_weight_partition_arg.Name()});

  NodeArg& dense_wi_bias_partition_arg = graph_utils::AddInitializer(graph, dense_wi_bias_initializer_partition);
  graph_utils::ReplaceNodeInput(biasgelu_node, 1, dense_wi_bias_partition_arg);
  updated_weight_names_.insert({dense_wi_bias_arg->Name(), dense_wi_bias_partition_arg.Name()});

  NodeArg& dense_wo_weight_partition_arg = graph_utils::AddInitializer(graph, dense_wo_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(*transpose_op_ptr, 0, dense_wo_weight_partition_arg);
  updated_weight_names_.insert({dense_wo_weight_arg->Name(), dense_wo_weight_partition_arg.Name()});

  graph.RemoveInitializedTensor(dense_wi_weight_arg->Name());
  graph.RemoveInitializedTensor(dense_wi_bias_arg->Name());
  graph.RemoveInitializedTensor(dense_wo_weight_arg->Name());

  if (dropout_node) {
    dropout_nodes_to_transform.insert(dropout_node);
  }

  const std::vector<NodeArg*> mlp_f_input_defs{node.MutableInputDefs()[0]};
  auto mlp_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
  auto& mlp_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("BART_MLP_MegatronF_Output"), &mlp_f_type_info);
  Node& mlp_f_node = graph.AddNode(graph.GenerateNodeName("BART_MLP_MegatronF"),
                                   "MegatronF",
                                   "MLP MegatronF",
                                   mlp_f_input_defs,
                                   {&mlp_f_out_arg}, {}, kMSDomain);
  counter++;
  mlp_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
  ReplaceNodeInput(graph, node, 0, mlp_f_node, 0, FilterConsumeNodeWithOpType);

  const std::vector<NodeArg*> mlp_g_input_defs{matmul2_node.MutableOutputDefs()[0]};
  auto mlp_g_type_info = *matmul2_node.MutableOutputDefs()[0]->TypeAsProto();
  auto& mlp_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("BART_MLP_MegatronG_Output"), &mlp_g_type_info);
  Node& mlp_g_node = graph.AddNode(graph.GenerateNodeName("BART_MLP_MegatronG"),
                                   "MegatronG",
                                   "MLP MegatronG",
                                   mlp_g_input_defs,
                                   {&mlp_g_out_arg}, {}, kMSDomain);
  mlp_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
  mlp_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
  graph_utils::ReplaceDownstreamNodeInput(graph, matmul2_node, 0, mlp_g_node, 0);
  modified = true;

  return Status::OK();
}

Status MegatronTransformer::TransformGPT2Attention(Graph& graph, bool& modified,
                                                   std::vector<Node*>& nodes_to_clear_shape,
                                                   std::unordered_set<Node*>& dropout_nodes_to_transform,
                                                   int32_t& counter,
                                                   NodeIndex node_index) const {
  auto skip_status = common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Skip BART Attention megatron transformation");

  // Self attention sub-graph.
  // MatMul->Add->Split->Reshape->Transpose->MatMul->Div->Mul->Sub->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Add
  //                  |->Reshape->Transpose->|                                        |
  //                  |->Reshape->Transpose------------------------------------------>|

  auto& node = *graph.GetNode(node_index);
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", opset_v9_13) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  if (node.GetInputEdgesCount() > 0 && node.InputNodesBegin()->OpType().compare("MegatronF") == 0) {
    return skip_status;
  }

  std::vector<Node*> sub_graph_node_ptrs;
  sub_graph_node_ptrs.push_back(&node);
  ProviderType provider_type = node.GetExecutionProviderType();

  std::vector<NodeInfo> linear_pattern = {
      NodeInfo({add_info}),  // -15
      NodeInfo({split_info}),
      NodeInfo({reshape_info}),
      NodeInfo({transpose_info}),
      NodeInfo({matmul_info}),  // -11
      NodeInfo({div_info}),
      NodeInfo({mul_info}),
      NodeInfo({sub_info}),
      NodeInfo({softmax_info}),
      NodeInfo({dropout_info}, false),  // -6
      NodeInfo({matmul_info}),
      NodeInfo({transpose_info}),
      NodeInfo({reshape_info}),
      NodeInfo({matmul_info}),
      NodeInfo({add_info})};  // -1
  if (!MatchLinearPattern(graph, &node, provider_type, linear_pattern, sub_graph_node_ptrs)) {
    return skip_status;
  }

  // Get all useful nodes here as more vector push back below will change the index.
  Node& add_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 15];
  Node& split_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 14];
  Node& k_transpose_after_reshape_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 12];
  Node* matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 11];
  Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 6];
  Node* matmul_node_ptr1 = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
  Node& transpose_node1 = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4];
  Node& matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 2];

  // Transpose node attribute checking.
  if (!optimizer_utils::IsAttributeWithExpectedValues(k_transpose_after_reshape_node, "perm", {0LL, 2LL, 1LL, 3LL}) ||
      !optimizer_utils::IsAttributeWithExpectedValues(transpose_node1, "perm", {0LL, 2LL, 1LL, 3LL})) {
    return skip_status;
  }

  std::vector<Node*> transpose_node_ptrs;  // For the 2nd and 3rd transpose nodes after split node for sub-graph structure checking.
  std::vector<Node*> reshape_node_ptrs;    // To keep the reshape node that need to change the shape constant.
  reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 13]);
  reshape_node_ptrs.push_back(sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 3]);
  auto split_output_iter = split_node.OutputNodesBegin();
  ++split_output_iter;
  for (; split_output_iter != split_node.OutputNodesEnd(); ++split_output_iter) {
    Node* reshape_node_ptr = graph.GetNode((*split_output_iter).Index());
    if (!IsExpectedOpAndProvider(*reshape_node_ptr, reshape_info, provider_type)) {
      break;
    }

    Node* transpose_node_ptr = graph.GetNode(reshape_node_ptr->OutputNodesBegin()->Index());
    if (!IsExpectedOpAndProvider(*transpose_node_ptr, transpose_info, provider_type)) {
      break;
    }

    reshape_node_ptrs.push_back(reshape_node_ptr);
    sub_graph_node_ptrs.push_back(reshape_node_ptr);
    transpose_node_ptrs.push_back(transpose_node_ptr);
    sub_graph_node_ptrs.push_back(transpose_node_ptr);
  }

  // Sub-graph structure and transpose attribute checking.
  if (transpose_node_ptrs.size() != 2 ||
      matmul_node_ptr != graph.GetNode(transpose_node_ptrs[0]->OutputNodesBegin()->Index()) ||
      matmul_node_ptr1 != graph.GetNode(transpose_node_ptrs[1]->OutputNodesBegin()->Index()) ||
      !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[0], "perm", {0LL, 2LL, 3LL, 1LL}) ||
      !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node_ptrs[1], "perm", {0LL, 2LL, 1LL, 3LL})) {
    return skip_status;
  }

  // Partition weights. If any of them fails, skip transforming this sub-graph.
  auto qkv_weight_arg = node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto qkv_weight_initializer_partition;
  if (!PartitionWeightByColumn(graph, *qkv_weight_arg, qkv_weight_initializer_partition, 3)) {
    return skip_status;
  }

  auto qkv_bias_arg = add_node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto qkv_bias_initializer_partition;
  if (!PartitionWeightByColumn(graph, *qkv_bias_arg, qkv_bias_initializer_partition, 3)) {
    return skip_status;
  }

  auto dense_weight_arg = matmul_node.MutableInputDefs()[1];
  ONNX_NAMESPACE::TensorProto dense_weight_initializer_partition;
  if (!PartitionWeightByRow(graph, *dense_weight_arg, dense_weight_initializer_partition)) {
    return skip_status;
  }

  // Check the constant value in the Reshape nodes.
  bool is_reshape_valid = true;
  for (Node* node_ptr : reshape_node_ptrs) {
    auto shape_arg = node_ptr->MutableInputDefs()[1];
    const ONNX_NAMESPACE::TensorProto* tensor;

    if (!graph.GetInitializedTensor(shape_arg->Name(), tensor)) {
      is_reshape_valid = false;
      break;
    }

    auto data_type = tensor->data_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      is_reshape_valid = false;
      break;
    }

    // The number of the values should be more than 2, and the 3rd value should be divisible by parallel size,
    // i.e., the attention head number should be divisible by parallel size.
    auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
    if (init_const->size() != 3 && init_const->size() != 4) {
      is_reshape_valid = false;
      break;
    }

    const int64_t* val = init_const->data<int64_t>();
    if (val[2] % horizontal_parallel_size_ != 0) {
      LOGS_DEFAULT(WARNING) << (init_const->size() == 3 ? "Hidden size " : "Number of attention heads ") << val[2]
                            << " is not divisible by horizontal_parallel_size_ "
                            << horizontal_parallel_size_ << ", not supported currently.";
      is_reshape_valid = false;
      break;
    }
  }

  if (!is_reshape_valid) {
    return skip_status;
  }

  // Ready to transform the sub-graph when reach here.
  // It's possible that the node vector contains nullptr due to some optinal node infos during linear pattern matching.
  std::copy_if(sub_graph_node_ptrs.begin(), sub_graph_node_ptrs.end(),
               std::back_inserter(nodes_to_clear_shape),
               [](Node* node_ptr) { return node_ptr != nullptr; });

  // Replace by the partition weights.
  NodeArg& qkv_weight_partition_arg = graph_utils::AddInitializer(graph, qkv_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(node, 1, qkv_weight_partition_arg);
  updated_weight_names_.insert({qkv_weight_arg->Name(), qkv_weight_partition_arg.Name()});

  NodeArg& qkv_bias_partition_arg = graph_utils::AddInitializer(graph, qkv_bias_initializer_partition);
  graph_utils::ReplaceNodeInput(add_node, 1, qkv_bias_partition_arg);
  updated_weight_names_.insert({qkv_bias_arg->Name(), qkv_bias_partition_arg.Name()});

  NodeArg& dense_weight_partition_arg = graph_utils::AddInitializer(graph, dense_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(matmul_node, 1, dense_weight_partition_arg);
  updated_weight_names_.insert({dense_weight_arg->Name(), dense_weight_partition_arg.Name()});

  graph.RemoveInitializedTensor(qkv_weight_arg->Name());
  graph.RemoveInitializedTensor(qkv_bias_arg->Name());
  graph.RemoveInitializedTensor(dense_weight_arg->Name());

  // Change the constant for the reshape nodes.
  for (Node* node_ptr : reshape_node_ptrs) {
    auto shape_arg = node_ptr->MutableInputDefs()[1];
    const ONNX_NAMESPACE::TensorProto* tensor;
    graph.GetInitializedTensor(shape_arg->Name(), tensor);
    auto data_type = tensor->data_type();
    auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
    const int64_t* val = init_const->data<int64_t>();
    int64_t size = init_const->size();
    ONNX_NAMESPACE::TensorProto tensor_partition;
    tensor_partition.set_name(graph.GenerateNodeArgName("partition_" + shape_arg->Name()));
    tensor_partition.set_data_type(data_type);
    tensor_partition.add_dims(size);

    std::vector<int64_t> val_partition;
    val_partition.reserve(size);
    val_partition.insert(val_partition.end(), val, val + size);
    val_partition[2] /= horizontal_parallel_size_;
    tensor_partition.set_raw_data(val_partition.data(), size * sizeof(int64_t));
    NodeArg& node_arg_partition = graph_utils::AddInitializer(graph, tensor_partition);
    graph_utils::ReplaceNodeInput(*node_ptr, 1, node_arg_partition);
    graph.RemoveInitializedTensor(shape_arg->Name());
  }

  if (dropout_node_ptr != nullptr) {
    dropout_nodes_to_transform.insert(dropout_node_ptr);
  }

  // Add MegatronF before the 1st MatMul and MegatronG before the last Add.
  const std::vector<NodeArg*> sa_f_input_defs{node.MutableInputDefs()[0]};
  auto sa_f_type_info = *node.MutableInputDefs()[0]->TypeAsProto();
  auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SeftAttention_MegatronF_Output"), &sa_f_type_info);
  Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "SeftAttention_MegatronF"),
                                  "MegatronF",
                                  "SeftAttention MegatronF",
                                  sa_f_input_defs,
                                  {&sa_f_out_arg}, {}, kMSDomain);
  sa_f_node.SetExecutionProviderType(node.GetExecutionProviderType());
  ReplaceNodeInput(graph, node, 0, sa_f_node, 0, FilterConsumeNodeWithOpType);

  const std::vector<NodeArg*> sa_g_input_defs{matmul_node.MutableOutputDefs()[0]};
  auto sa_g_type_info = *matmul_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
  auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("SeftAttention_MegatronG_Output"), &sa_g_type_info);
  Node& sa_g_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "SelfAttention_MegatronG"),
                                  "MegatronG",
                                  "Attention MegatronG",
                                  sa_g_input_defs,
                                  {&sa_g_out_arg}, {}, kMSDomain);
  sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
  sa_g_node.SetExecutionProviderType(node.GetExecutionProviderType());
  graph_utils::ReplaceDownstreamNodeInput(graph, matmul_node, 0, sa_g_node, 0);
  modified = true;
  counter++;

  return Status::OK();
}

Status MegatronTransformer::TransformBARTAttention(Graph& graph, bool& modified,
                                                   std::vector<Node*>& nodes_to_clear_shape,
                                                   std::unordered_set<Node*>& dropout_nodes_to_transform,
                                                   int32_t& counter,
                                                   NodeIndex node_index) const {
  auto skip_status = common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Skip BART Attention megatron transformation");

  // Self/Enc-Dec Attention sub-graph.
  //
  // MatMul->Add->Mul->Reshape->Transpose->MatMul->Reshape->Where->Reshape->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Add->Droupout
  // MatMul->Add->Reshape->Transpose-------> |                                                  |
  // MatMul->Add->Reshape->Transpose----------------------------------------------------------> |
  auto& node = *graph.GetNode(node_index);

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", opset_v9_13) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      node.GetOutputEdgesCount() != 1) {
    return skip_status;
  }

  Node* q_matmul_input_node_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
  if (q_matmul_input_node_ptr != nullptr && q_matmul_input_node_ptr->OpType().compare("MegatronF") == 0) {
    return skip_status;
  }
  std::vector<Node*> sub_graph_node_ptrs;
  sub_graph_node_ptrs.push_back(&node);
  ProviderType provider_type = node.GetExecutionProviderType();

  std::vector<NodeInfo> linear_pattern = {
      NodeInfo({add_info}),
      NodeInfo({mul_info}),
      NodeInfo({reshape_info}),
      NodeInfo({transpose_info}),
      NodeInfo({matmul_info}),
      NodeInfo({add_info}, false),  // -13
      NodeInfo({reshape_info}),
      NodeInfo({where_info}),
      NodeInfo({reshape_info}),
      NodeInfo({softmax_info}),
      NodeInfo({dropout_info}, false),  // -8
      NodeInfo({matmul_info}),
      NodeInfo({add_info}, false),  // -6
      NodeInfo({transpose_info}),
      NodeInfo({reshape_info}),
      NodeInfo({matmul_info}),  // -3
      NodeInfo({add_info}),
      NodeInfo({dropout_info}, false)};  // -1
  if (!MatchLinearPattern(graph, &node, provider_type, linear_pattern, sub_graph_node_ptrs)) {
    return skip_status;
  }
  // Get all useful nodes here as more vector push back below will change the index.
  // Other than the optional nodes in the pattern, all other node pointers are valid
  // if they match the linear pattern.
  Node* q_biasadd_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 18];
  Node* q_transpose_after_reshape_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 15];
  Node* qk_matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 14];
  Node* dropout_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 8];
  Node* qkv_matmul_node_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 7];
  Node* transpose_node1_ptr = sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 5];
  Node& dense_matmul_node = *sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 3];

  // Transpose node attribute checking.
  if (!optimizer_utils::IsAttributeWithExpectedValues(*q_transpose_after_reshape_node_ptr, "perm", {1LL, 0LL, 2LL}) ||
      !optimizer_utils::IsAttributeWithExpectedValues(*transpose_node1_ptr, "perm", {1LL, 0LL, 2LL})) {
    return skip_status;
  }
  // map between reshape node and dim of reshape that must be modified
  std::unordered_map<Node*, int64_t> reshape_node_ptrs;
  reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 16]] = 1;
  reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 12]] = 1;
  reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 10]] = 0;
  reshape_node_ptrs[sub_graph_node_ptrs[sub_graph_node_ptrs.size() - 4]] = 2;
  // till now node should be q matmul operation

  std::vector<Node*> weight_transpose_node_ptrs;
  std::vector<Node*> bias_add_node_ptrs;

  Node* q_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[1]->Name()));
  if (q_transpose_ptr == nullptr || !IsExpectedOpAndProvider(*q_transpose_ptr, transpose_info, provider_type)) {
    return skip_status;
  }
  weight_transpose_node_ptrs.push_back(q_transpose_ptr);
  sub_graph_node_ptrs.push_back(q_transpose_ptr);
  bias_add_node_ptrs.push_back(q_biasadd_node_ptr);

  Node* k_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(qk_matmul_node_ptr->MutableInputDefs()[1]->Name()));
  if (k_transpose_ptr == nullptr || !IsExpectedOpAndProvider(*k_transpose_ptr, transpose_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(k_transpose_ptr);

  Node* k_reshape_ptr = const_cast<Node*>(graph.GetProducerNode(k_transpose_ptr->MutableInputDefs()[0]->Name()));
  if (k_reshape_ptr == nullptr || !IsExpectedOpAndProvider(*k_reshape_ptr, reshape_info, provider_type)) {
    return skip_status;
  }
  reshape_node_ptrs[k_reshape_ptr] = 1;
  sub_graph_node_ptrs.push_back(k_reshape_ptr);

  Node* k_add_ptr = const_cast<Node*>(graph.GetProducerNode(k_reshape_ptr->MutableInputDefs()[0]->Name()));
  if (k_add_ptr == nullptr || !IsExpectedOpAndProvider(*k_add_ptr, add_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(k_add_ptr);
  bias_add_node_ptrs.push_back(k_add_ptr);

  Node* k_matmul_ptr = const_cast<Node*>(graph.GetProducerNode(k_add_ptr->MutableInputDefs()[0]->Name()));
  if (k_matmul_ptr == nullptr || !IsExpectedOpAndProvider(*k_matmul_ptr, matmul_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(k_matmul_ptr);

  Node* k_weight_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(k_matmul_ptr->MutableInputDefs()[1]->Name()));
  if (k_weight_transpose_ptr == nullptr || !IsExpectedOpAndProvider(*k_weight_transpose_ptr, transpose_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(k_weight_transpose_ptr);
  weight_transpose_node_ptrs.push_back(k_weight_transpose_ptr);

  Node* v_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(qkv_matmul_node_ptr->MutableInputDefs()[1]->Name()));
  if (v_transpose_ptr == nullptr || !IsExpectedOpAndProvider(*v_transpose_ptr, transpose_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(v_transpose_ptr);

  Node* v_reshape_ptr = const_cast<Node*>(graph.GetProducerNode(v_transpose_ptr->MutableInputDefs()[0]->Name()));
  if (v_reshape_ptr == nullptr || !IsExpectedOpAndProvider(*v_reshape_ptr, reshape_info, provider_type)) {
    return skip_status;
  }
  reshape_node_ptrs[v_reshape_ptr] = 1;
  sub_graph_node_ptrs.push_back(v_reshape_ptr);

  Node* v_add_ptr = const_cast<Node*>(graph.GetProducerNode(v_reshape_ptr->MutableInputDefs()[0]->Name()));
  if (v_add_ptr == nullptr || !IsExpectedOpAndProvider(*v_add_ptr, add_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(v_add_ptr);
  bias_add_node_ptrs.push_back(v_add_ptr);

  Node* v_matmul_ptr = const_cast<Node*>(graph.GetProducerNode(v_add_ptr->MutableInputDefs()[0]->Name()));
  if (k_matmul_ptr == nullptr || !IsExpectedOpAndProvider(*k_matmul_ptr, matmul_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(v_matmul_ptr);

  Node* v_weight_transpose_ptr = const_cast<Node*>(graph.GetProducerNode(v_matmul_ptr->MutableInputDefs()[1]->Name()));
  if (v_weight_transpose_ptr == nullptr || !IsExpectedOpAndProvider(*v_weight_transpose_ptr, transpose_info, provider_type)) {
    return skip_status;
  }
  sub_graph_node_ptrs.push_back(v_weight_transpose_ptr);
  weight_transpose_node_ptrs.push_back(v_weight_transpose_ptr);

  // K and V matmul must have the same input
  Node* q_matmul_ptr = &node;
  if (k_matmul_ptr->MutableInputDefs()[0]->Name() != v_matmul_ptr->MutableInputDefs()[0]->Name()) {
    return skip_status;
  }

  // Check the constant value in the Reshape nodes.
  bool is_reshape_valid = true;
  for (auto x : reshape_node_ptrs) {
    Node* node_ptr = x.first;
    int64_t idx = x.second;
    auto shape_arg = node_ptr->MutableInputDefs()[1];
    const ONNX_NAMESPACE::TensorProto* tensor;
    if (!graph.GetInitializedTensor(shape_arg->Name(), tensor)) {
      is_reshape_valid = false;
      break;
    }
    auto data_type = tensor->data_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      is_reshape_valid = false;
      break;
    }
    // The number of the values should be more than idx, and the idx'th value should be divisible by parallel size,
    // i.e., the attention head number should be divisible by parallel size.
    auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
    if (init_const->size() <= idx) {
      is_reshape_valid = false;
      break;
    }
    const int64_t* val = init_const->data<int64_t>();
    if (val[idx] % horizontal_parallel_size_ != 0) {
      LOGS_DEFAULT(WARNING) << "dim[" << idx << "]: " << val[idx]
                            << " is not divisible by horizontal_parallel_size_ "
                            << horizontal_parallel_size_ << ", not supported currently.";
      is_reshape_valid = false;
      break;
    }
  }

  if (!is_reshape_valid) {
    return skip_status;
  }

  // Partition weights. If any of them fails, skip transforming the rest.
  std::vector<ONNX_NAMESPACE::TensorProto> qkv_weight_initializer_partitions;
  for (auto trans_ptr : weight_transpose_node_ptrs) {
    auto qkv_weight_arg = trans_ptr->MutableInputDefs()[0];
    ONNX_NAMESPACE::TensorProto qkv_weight_initializer_partition;
    if (!PartitionWeightByRow(graph, *qkv_weight_arg, qkv_weight_initializer_partition)) {
      break;
    }
    qkv_weight_initializer_partitions.push_back(qkv_weight_initializer_partition);
  }

  // Partition bias. If any of them fails, skip transforming the rest.
  std::vector<ONNX_NAMESPACE::TensorProto> qkv_bias_initializer_partitions;
  for (auto add_ptr : bias_add_node_ptrs) {
    auto qkv_bias_arg = add_ptr->MutableInputDefs()[1];
    ONNX_NAMESPACE::TensorProto qkv_bias_initializer_partition;
    if (!PartitionWeightByColumn(graph, *qkv_bias_arg, qkv_bias_initializer_partition)) {
      break;
    }
    qkv_bias_initializer_partitions.push_back(qkv_bias_initializer_partition);
  }

  // if all the weights or biases weren't transformed, skip transforming this subgraph
  if (weight_transpose_node_ptrs.size() != qkv_weight_initializer_partitions.size()) {
    return skip_status;
  }
  if (bias_add_node_ptrs.size() != qkv_bias_initializer_partitions.size()) {
    return skip_status;
  }

  // transform the dense weight. If it fails, skip transforming this subgraph.
  Node* last_transpose = const_cast<Node*>(graph.GetProducerNode(dense_matmul_node.MutableInputDefs()[1]->Name()));
  auto dense_weight_arg = last_transpose->MutableInputDefs()[0];
  ONNX_NAMESPACE::TensorProto dense_weight_initializer_partition;
  if (!PartitionWeightByColumn(graph, *dense_weight_arg, dense_weight_initializer_partition)) {
    return skip_status;
  }

  // Ready to transform the sub-graph when reach here.
  // Replace node inputs
  size_t i = 0;
  for (auto trans_ptr : weight_transpose_node_ptrs) {
    auto weight_name = trans_ptr->MutableInputDefs()[0]->Name();
    NodeArg& qkv_weight_partition_arg = graph_utils::AddInitializer(graph, qkv_weight_initializer_partitions[i]);
    graph_utils::ReplaceNodeInput(*trans_ptr, 0, qkv_weight_partition_arg);
    graph.RemoveInitializedTensor(weight_name);
    updated_weight_names_.insert({weight_name, qkv_weight_partition_arg.Name()});
    i++;
  }
  i = 0;
  for (auto add_ptr : bias_add_node_ptrs) {
    auto bias_name = add_ptr->MutableInputDefs()[1]->Name();
    NodeArg& qkv_bias_partition_arg = graph_utils::AddInitializer(graph, qkv_bias_initializer_partitions[i]);
    graph_utils::ReplaceNodeInput(*add_ptr, 1, qkv_bias_partition_arg);
    graph.RemoveInitializedTensor(bias_name);
    updated_weight_names_.insert({bias_name, qkv_bias_partition_arg.Name()});
    i++;
  }

  NodeArg& dense_weight_partition_arg = graph_utils::AddInitializer(graph, dense_weight_initializer_partition);
  graph_utils::ReplaceNodeInput(*last_transpose, 0, dense_weight_partition_arg);
  graph.RemoveInitializedTensor(dense_weight_arg->Name());
  updated_weight_names_.insert({dense_weight_arg->Name(), dense_weight_partition_arg.Name()});

  // It's possible that the node vector contains nullptr due to some optinal node infos during linear pattern matching.
  std::copy_if(sub_graph_node_ptrs.begin(), sub_graph_node_ptrs.end(),
               std::back_inserter(nodes_to_clear_shape),
               [](Node* node_ptr) { return node_ptr != nullptr; });

  // Change the constant for the reshape nodes.
  for (auto x : reshape_node_ptrs) {
    Node* node_ptr = x.first;
    int64_t idx = x.second;
    auto shape_arg = node_ptr->MutableInputDefs()[1];
    const ONNX_NAMESPACE::TensorProto* tensor;
    graph.GetInitializedTensor(shape_arg->Name(), tensor);
    auto data_type = tensor->data_type();
    auto init_const = onnxruntime::make_unique<Initializer>(*tensor, graph.ModelPath());
    const int64_t* val = init_const->data<int64_t>();
    int64_t size = init_const->size();
    ONNX_NAMESPACE::TensorProto tensor_partition;
    tensor_partition.set_name(graph.GenerateNodeArgName("partition_" + shape_arg->Name()));
    tensor_partition.set_data_type(data_type);
    tensor_partition.add_dims(size);

    std::vector<int64_t> val_partition;
    val_partition.reserve(size);
    val_partition.insert(val_partition.end(), val, val + size);
    val_partition[idx] /= horizontal_parallel_size_;
    tensor_partition.set_raw_data(val_partition.data(), size * sizeof(int64_t));
    NodeArg& node_arg_partition = graph_utils::AddInitializer(graph, tensor_partition);
    graph_utils::ReplaceNodeInput(*node_ptr, 1, node_arg_partition);
    graph.RemoveInitializedTensor(shape_arg->Name());
  }

  if (dropout_node_ptr != nullptr) {
    dropout_nodes_to_transform.insert(dropout_node_ptr);
  }

  // Add MegatronF before the 1st MatMul and MegatronG before the last Add.

  NodeArg* prev_input_node_ptr = k_matmul_ptr->MutableInputDefs()[0];
  std::vector<Node*> new_consumer_nodes;
  const auto& node_consumers = graph.GetConsumerNodes(prev_input_node_ptr->Name());
  for (auto& n : node_consumers) {
    if (n->Index() == k_matmul_ptr->Index() || n->Index() == v_matmul_ptr->Index() || n->Index() == q_matmul_ptr->Index()) {
      continue;
    }
    new_consumer_nodes.emplace_back(const_cast<Node*>(n));
  }

  bool shared_same_input = k_matmul_ptr->MutableInputDefs()[0]->Name().compare(q_matmul_ptr->MutableInputDefs()[0]->Name()) == 0;

  //then for q, and k&v will have different MegatronF node.
  {
    const std::vector<NodeArg*> sa_f_input_defs{prev_input_node_ptr};
    auto sa_f_type_info = *prev_input_node_ptr->TypeAsProto();
    auto& sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(k_matmul_ptr->Name() + "BARTAttention_MegatronF_Output"), &sa_f_type_info);
    Node& sa_f_node = graph.AddNode(graph.GenerateNodeName(k_matmul_ptr->Name() + "BARTAttention_MegatronF"),
                                    "MegatronF",
                                    k_matmul_ptr->Name() + " BARTAttention MegatronF",
                                    sa_f_input_defs,
                                    {&sa_f_out_arg}, {}, kMSDomain);
    sa_f_node.SetExecutionProviderType(k_matmul_ptr->GetExecutionProviderType());
    ReplaceNodeInput(graph, *k_matmul_ptr, 0, sa_f_node, 0, FilterConsumeNodeWithOpType);
    ReplaceNodeInput(graph, *v_matmul_ptr, 0, sa_f_node, 0, FilterConsumeNodeWithOpType);
    if (shared_same_input) {
      ReplaceNodeInput(graph, *q_matmul_ptr, 0, sa_f_node, 0, FilterConsumeNodeWithOpType);
    }
    new_consumer_nodes.push_back(&sa_f_node);
  }
  graph.UpdateConsumerNodes(prev_input_node_ptr->Name(), new_consumer_nodes);
  counter++;
  if (!shared_same_input) {
    {
      NodeArg* q_prev_input_node_ptr = q_matmul_ptr->MutableInputDefs()[0];
      std::vector<Node*> q_new_consumer_nodes;
      const auto& q_node_consumers = graph.GetConsumerNodes(q_prev_input_node_ptr->Name());
      for (auto& n : q_node_consumers) {
        if (n->Index() == k_matmul_ptr->Index() || n->Index() == v_matmul_ptr->Index() || n->Index() == q_matmul_ptr->Index()) {
          continue;
        }
        q_new_consumer_nodes.emplace_back(const_cast<Node*>(n));
      }

      const std::vector<NodeArg*> q_sa_f_input_defs{q_matmul_ptr->MutableInputDefs()[0]};
      auto q_sa_f_type_info = *q_matmul_ptr->MutableInputDefs()[0]->TypeAsProto();
      auto& q_sa_f_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(q_matmul_ptr->Name() + "BARTAttention_MegatronF_Output"), &q_sa_f_type_info);
      Node& q_sa_f_node = graph.AddNode(graph.GenerateNodeName(q_matmul_ptr->Name() + "BARTAttention_MegatronF"),
                                        "MegatronF",
                                        q_matmul_ptr->Name() + " BARTAttention MegatronF",
                                        q_sa_f_input_defs,
                                        {&q_sa_f_out_arg}, {}, kMSDomain);
      q_sa_f_node.SetExecutionProviderType(q_matmul_ptr->GetExecutionProviderType());
      ReplaceNodeInput(graph, *q_matmul_ptr, 0, q_sa_f_node, 0, FilterConsumeNodeWithOpType);

      q_new_consumer_nodes.push_back(&q_sa_f_node);
      graph.UpdateConsumerNodes(q_prev_input_node_ptr->Name(), q_new_consumer_nodes);
      // todo: need update the consumer node for the input_node as well.
    }
  }

  const std::vector<NodeArg*> sa_g_input_defs{dense_matmul_node.MutableOutputDefs()[0]};
  auto sa_g_type_info = *dense_matmul_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
  auto& sa_g_out_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("BARTAttention_MegatronG_Output"), &sa_g_type_info);
  Node& sa_g_node = graph.AddNode(graph.GenerateNodeName(k_matmul_ptr->Name() + "BARTAttention_MegatronG"),
                                  "MegatronG",
                                  "BARTAttention MegatronG",
                                  sa_g_input_defs,
                                  {&sa_g_out_arg}, {}, kMSDomain);
  sa_g_node.AddAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel));
  sa_g_node.SetExecutionProviderType(k_matmul_ptr->GetExecutionProviderType());
  graph_utils::ReplaceDownstreamNodeInput(graph, dense_matmul_node, 0, sa_g_node, 0);

  modified = true;

  return Status::OK();
}

Status MegatronTransformer::DoTransform(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger,
                                        std::vector<Node*>& nodes_to_clear_shape,
                                        std::unordered_set<Node*>& dropout_nodes_to_transform) const {
  std::vector<int> counters(4);
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    int i = 0;
    auto ret = TransformGPT2Attention(graph, modified, nodes_to_clear_shape,
                                      dropout_nodes_to_transform, counters[i++], node_index);
    if (ret.Code() != common::NOT_IMPLEMENTED) {
      ORT_ENFORCE(ret.IsOK());
      continue;
    }

    ret = TransformGPT2MLP(graph, modified, nodes_to_clear_shape, counters[i++], node_index);
    if (ret.Code() != common::NOT_IMPLEMENTED) {
      ORT_ENFORCE(ret.IsOK());
      continue;
    }

    ret = TransformBARTAttention(graph, modified, nodes_to_clear_shape,
                                 dropout_nodes_to_transform, counters[i++], node_index);
    if (ret.Code() != common::NOT_IMPLEMENTED) {
      ORT_ENFORCE(ret.IsOK());
      continue;
    }

    ret = TransformBARTMLP(graph, modified, nodes_to_clear_shape,
                           dropout_nodes_to_transform, counters[i++], node_index);
    if (ret.Code() != common::NOT_IMPLEMENTED) {
      ORT_ENFORCE(ret.IsOK());
      continue;
    }
  }

  LOGS_DEFAULT(WARNING) << "Megatron transformer result : Partitioned "
                        << counters[0] << " GPT2 Attention Blocks, "
                        << counters[1] << " GPT2 MLP Blocks, "
                        << counters[2] << " BART Attention Blocks, "
                        << counters[3] << " BART MLP Blocks.";

  return Status::OK();
}

Status MegatronTransformer::TransformDropout(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                             std::unordered_set<Node*>& dropout_nodes_to_transform, int32_t& counter) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", opset_v12_13)) {
      continue;
    }

    // Only need to set the seed if it's a transformed self-attention dropout, or the seed attribute is not set.
    if (dropout_nodes_to_transform.find(&node) != dropout_nodes_to_transform.end()) {
      int64_t seed = static_cast<int64_t>(HashName(node.MutableOutputDefs()[0]->Name())) + utils::GetRandomSeed();
      if (dropout_nodes_to_transform.find(&node) != dropout_nodes_to_transform.end()) {
        seed += horizontal_parallel_rank_;
      }

      if (graph_utils::GetNodeAttribute(node, "seed") != nullptr) {
        node.ClearAttribute("seed");
      }
      node.AddAttribute("seed", seed);
      counter++;
      modified = true;
    }
  }

  return Status::OK();
}

Status MegatronTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  if (horizontal_parallel_size_ <= 1) {
    return Status::OK();
  }

  std::vector<Node*> nodes_to_clear_shape;
  std::unordered_set<Node*> dropout_nodes_to_transform;
  int32_t dropout_changed = 0;

  ORT_ENFORCE(DoTransform(graph, modified, graph_level, logger,
                          nodes_to_clear_shape, dropout_nodes_to_transform)
                  .IsOK());
  ORT_ENFORCE(TransformDropout(graph, modified, graph_level, logger,
                               dropout_nodes_to_transform, dropout_changed)
                  .IsOK());

  auto& graph_inputs = graph.GetInputs();
  for (auto node : nodes_to_clear_shape) {
    if (node != nullptr) {
      auto& inputs = node->MutableInputDefs();
      for (auto* input : inputs)
        if (std::find(graph_inputs.begin(), graph_inputs.end(), input) == graph_inputs.end())
          input->ClearShape();

      for (auto* output : node->MutableOutputDefs())
        if (std::find(graph_inputs.begin(), graph_inputs.end(), output) == graph_inputs.end())
          output->ClearShape();
    }
  }

  for (auto x : updated_weight_names_) {
    auto old_initializer_name = x.first;
    auto new_initializer_name = x.second;
    if (weights_to_train_.find(old_initializer_name) != weights_to_train_.end()) {
      weights_to_train_.erase(old_initializer_name);
      weights_to_train_.insert(new_initializer_name);
    }
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
    auto ret = graph.Resolve();
    LOGS_DEFAULT(WARNING) << "Megatron transformer result: Reset seed for " << dropout_changed
                          << " Dropout nodes. Error Message (if there is): " << ret.ErrorMessage();
    ORT_ENFORCE(ret.IsOK());
  } else {
    LOGS_DEFAULT(WARNING) << "Megatron transformer result : unmodified\n";
  }

  return Status::OK();
}

}  // namespace onnxruntime