// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"
#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"
#if defined(USE_MPI)
#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#endif
#include "orttraining/core/graph/zero_optimizer_graph_builder.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/test_environment.h"
#include "orttraining/test/session/training_session_test_utils.h"
#include "orttraining/core/graph/optimizer_builder.h"

using onnxruntime::test::CountOpsInGraph;
using onnxruntime::test::CreateMLValue;
using onnxruntime::test::TestCPUExecutionProvider;
using namespace onnxruntime::test::training_session_test_utils;

namespace onnxruntime {
namespace training {
namespace test {
namespace {

const std::vector<std::string> k_weight_names{"weight_1", "weight_2"};
constexpr const char* const k_loss_scaling_factor_name = "loss_scaling_factor";
constexpr const char* const k_adam_optimizer_op_name = "AdamOptimizer";
constexpr const char* const k_lamb_optimizer_op_name = "LambOptimizer";
constexpr const char* const k_all_reduce_op_name = "NcclAllReduce";
constexpr const char* const k_all_gather_op_name = "NcclAllGather";
constexpr const char* const k_reduce_scatter_op_name = "NcclReduceScatter";
constexpr const char* const k_is_all_finite_op_name = "IsAllFinite";
constexpr const char* const k_gradient_norm_op_name = "ReduceAllL2";
constexpr const char* const k_unscale_op_name = "MixedPrecisionScale";
constexpr const char* const k_inplace_accumulator_op_name = "InPlaceAccumulator";
constexpr const char* const k_zero_gradient_op_name = "ZeroGradient";
#if defined(USE_MPI)
constexpr const char* const k_adasum_op_name = "AdasumAllReduce";
#endif
Status SetUpBaseGraph(Graph& graph);

class OptimizerGraphBuilderTest : public testing::Test {
 protected:
  OptimizerGraphBuilderTest() : model_{"test_model", false, onnxruntime::test::DefaultLoggingManager().DefaultLogger()},
                                graph_{model_.MainGraph()} {
  }

  virtual void SetUp() override {
    ASSERT_STATUS_OK(SetUpBaseGraph(graph_));
  }

  Model model_;
  Graph& graph_;
};

// sets up a base graph with weight and gradient NodeArgs for each weight name
Status SetUpBaseGraph(Graph& graph) {
  ONNX_NAMESPACE::TypeProto float_tensor_type{};
  float_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  ONNX_NAMESPACE::TensorProto weight_initializer_base{};
  weight_initializer_base.add_dims(1);
  weight_initializer_base.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_initializer_base.add_float_data(1.0f);

  ONNX_NAMESPACE::TensorProto weight_gradient_initializer_base{weight_initializer_base};
  weight_gradient_initializer_base.set_float_data(0, 2.0f);

  std::unordered_set<std::string> weight_and_gradient_names{};

  for (const auto& weight_name : k_weight_names) {
    graph.GetOrCreateNodeArg(weight_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_initializer{weight_initializer_base};
    weight_initializer.set_name(weight_name);
    graph.AddInitializedTensor(weight_initializer);

    const std::string weight_gradient_name = GradientBuilderBase::GradientName(weight_name);
    graph.GetOrCreateNodeArg(weight_gradient_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_gradient_initializer{weight_gradient_initializer_base};
    weight_gradient_initializer.set_name(weight_gradient_name);
    graph.AddInitializedTensor(weight_gradient_initializer);

    weight_and_gradient_names.emplace(weight_name);
    weight_and_gradient_names.emplace(weight_gradient_name);
  }

  Graph::ResolveOptions resolve_options{};
  resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
  return graph.Resolve(resolve_options);
}

class ZeroOptimizerGraphBuilderTest : public testing::Test {
 protected:
  ZeroOptimizerGraphBuilderTest() {}
};

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap(const std::string& optim_name,
                                                                   const std::vector<std::string>& weight_names = k_weight_names) {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  std::transform(
      weight_names.begin(), weight_names.end(), std::inserter(result, result.end()),
      [&optim_name](const std::string& weight_name) {
        return std::make_pair(
            weight_name, OptimizerNodeConfig{optim_name, nullptr, "Learning_Rate", {}});
      });

  return result;
}

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap(const std::vector<std::string>& weight_names = k_weight_names) {
  return GetOptInfoMap(k_adam_optimizer_op_name, weight_names);
}

OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

void VerifyTensorValue(const ONNX_NAMESPACE::TensorProto* tensor, float expected_val) {
  float tensor_value;
  ASSERT_TRUE(tensor->dims_size() == 1);
  ASSERT_TRUE(tensor->dims(0) == 1);
  if (tensor->has_raw_data()) {
    memcpy(&tensor_value, tensor->raw_data().data(), sizeof(float));
  } else {
    tensor_value = *(tensor->float_data().data());
  }
  ASSERT_EQ(tensor_value, expected_val);
}

void VerifyTensorValue(const ONNX_NAMESPACE::TensorProto* tensor, int64_t expected_val) {
  int64_t tensor_value;
  ASSERT_TRUE(tensor->dims_size() == 1);
  ASSERT_TRUE(tensor->dims(0) == 1);
  if (tensor->has_raw_data()) {
    memcpy(&tensor_value, tensor->raw_data().data(), sizeof(int64_t));
  } else {
    tensor_value = *(tensor->int64_data().data());
  }
  ASSERT_EQ(tensor_value, expected_val);
}

static int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  static std::string ms_domain_prefix{std::string(kMSDomain) + '.'};

  auto op_count_it = op_counts.find(ms_domain_prefix + op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}

}  // namespace

static void TestOptimizerGraphBuilderWithInitialStates(OptimizerGraphConfig config,
                                                       Graph& graph,
                                                       std::string optimizer_op_name) {
  std::unordered_map<std::string, OptimizerNodeConfig> weight_names_to_opt_configs = GetOptInfoMap(optimizer_op_name);

  std::vector<int64_t> dims = {1};
  std::vector<float> values = {4.f};
  std::vector<int64_t> uc_value = {3};
  NameMLValMap shared_states;

  for (auto& opt_config_it : weight_names_to_opt_configs) {
    NameMLValMap per_weight_states;
    OrtValue ml_value;

    for (const auto key : MOMENTS_PREFIXES) {
      CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, values, &ml_value);
      per_weight_states.insert(std::make_pair(key, std::move(ml_value)));
    }
    if (optimizer_op_name == k_adam_optimizer_op_name) {
      CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, uc_value, &ml_value);
      per_weight_states.insert(std::make_pair(ADAM_UC_PREFIX, std::move(ml_value)));
    } else if (optimizer_op_name == k_lamb_optimizer_op_name) {
      // add "Step" for lamb
      CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, uc_value, &ml_value);
      shared_states.insert(std::make_pair(LAMB_STEP_TENSOR_NAME, std::move(ml_value)));
      config.shared_optimizer_states = std::move(shared_states);
    }
    opt_config_it.second.initial_states = std::move(per_weight_states);
  }

  std::unordered_map<std::string, std::string> updated_weight_names_map;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  OptimizerGraphBuilder optimizer_graph_builder(GetOptimizerBuilderRegistry(), config, weight_names_to_opt_configs, updated_weight_names_map, weight_partition_info);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> opt_initializer_names_map;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names_map, opt_graph_outputs));

  const ONNX_NAMESPACE::TensorProto* tensor;
  for (auto& weight_item : opt_initializer_names_map) {
    for (auto& opt_item : weight_item.second) {
      ASSERT_TRUE(graph.GetInitializedTensor(opt_item.second, tensor));
      ASSERT_TRUE(tensor->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT || tensor->data_type() == ONNX_NAMESPACE::TensorProto::INT64);
      if (tensor->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
        VerifyTensorValue(tensor, values[0]);
      } else if (tensor->data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
        VerifyTensorValue(tensor, uc_value[0]);
      }
    }
  }
}

TEST_F(OptimizerGraphBuilderTest, LoadOptimState_FullPrecision_Adam) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestOptimizerGraphBuilderWithInitialStates(config, graph_, k_adam_optimizer_op_name);
}

TEST_F(OptimizerGraphBuilderTest, LoadOptimState_FullPrecision_Lamb) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestOptimizerGraphBuilderWithInitialStates(config, graph_, k_lamb_optimizer_op_name);
}

TEST_F(OptimizerGraphBuilderTest, ZeroSplitInitialOptimizerState) {
  NameMLValMap initial_states;
  std::vector<int64_t> param_dims = {784, 128};
  int64_t num_ele = std::accumulate(param_dims.begin(), param_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  MLValue mlValue;
  std::vector<float> init_value(num_ele);
  std::iota(init_value.begin(), init_value.end(), static_cast<float>(0));

  for (const auto& param_prefix : MOMENTS_PREFIXES) {
    TrainingUtil::CreateCpuMLValue<float>(param_dims, init_value, &mlValue);
    initial_states.insert(std::make_pair(param_prefix, std::move(mlValue)));
  }

  int64_t partition_offset = 10;
  int64_t partition_size = 500;
  PartitionOptimizerState(partition_offset, partition_size, initial_states);

  std::vector<float> expected_vec(init_value.begin() + partition_offset, init_value.begin() + partition_offset + partition_size);
  std::vector<int64_t> expected_shape = {partition_size};

  for (const auto& state : initial_states) {
    const auto& init_tensor = state.second.Get<Tensor>();
    const auto& shape = init_tensor.Shape().GetDims();
    ASSERT_EQ(shape, expected_shape);
    const std::vector<float> found(init_tensor.Data<float>(),
                                   init_tensor.Data<float>() + partition_size);
    ASSERT_EQ(expected_vec, found);
  }
}

static void TestDefaultOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  OptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map, weight_partition_info);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, weight_to_opt_mapping, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_EQ(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 1);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_EQ(GetOpCount(op_counts, k_gradient_norm_op_name), 1);
    ASSERT_EQ(GetOpCount(op_counts, k_is_all_finite_op_name), 1);
  }

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_adam_optimizer_op_name), k_weight_names.size());

  // verify distributed operations don't exist
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_reduce_scatter_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_all_gather_op_name), 0);
}

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

#if defined(ORT_USE_NCCL)
static void TestAllreduceOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  AllreduceOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map, weight_partition_info);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, weight_to_opt_mapping, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_EQ(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 1);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_EQ(GetOpCount(op_counts, k_gradient_norm_op_name), 1);
    ASSERT_EQ(GetOpCount(op_counts, k_is_all_finite_op_name), 1);
  }

  // verify allreduce operations exist
  ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), 1);
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 1);

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_adam_optimizer_op_name), k_weight_names.size());
}
#endif

#if defined(ORT_USE_NCCL) && defined(USE_MPI)
static void TestAdasumOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  std::unordered_map<std::string, TrainingSession::PartitionInfo> weight_partition_info;
  AdasumOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map, weight_partition_info);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, weight_to_opt_mapping, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), 1);
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size() * 2);
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_EQ(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 1);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify allreduce operations exist
  ASSERT_GT(GetOpCount(op_counts, k_adasum_op_name), 0);

  // verify in place adder operations exist
  ASSERT_GT(GetOpCount(op_counts, k_inplace_accumulator_op_name), 0);

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_adam_optimizer_op_name), k_weight_names.size());
}
#endif

#if defined(ORT_USE_NCCL) && defined(USE_MPI)

TEST_F(OptimizerGraphBuilderTest, Adasum_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchicalReduction;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}
TEST_F(OptimizerGraphBuilderTest, Adasum_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchicalReduction;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchicalReduction;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchicalReduction;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}
#endif  //ORT_USE_NCCL && USE_MPI

#ifdef ORT_USE_NCCL
TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

Status SetUpZeroGraph(Graph& graph, std::vector<std::string>& weight_names, const std::vector<int64_t>& weight_dims) {
  std::unordered_set<std::string> weight_and_gradient_names{};

  for (size_t i = 0; i < weight_dims.size(); ++i) {
    const std::string weight_name = "weight_" + std::to_string(i);
    const std::string gradient_name = GradientBuilderBase::GradientName(weight_name);

    ONNX_NAMESPACE::TensorProto weight_initializer = CreateTensorProto<float>(weight_name, 0.1f, {weight_dims[i]});
    ONNX_NAMESPACE::TensorProto gradient_initializer = CreateTensorProto<float>(gradient_name, 0.01f, {weight_dims[i]});

    ONNX_NAMESPACE::TypeProto float_tensor_type{};
    float_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(weight_dims[i]);

    graph.GetOrCreateNodeArg(weight_name, &float_tensor_type);
    graph.AddInitializedTensor(weight_initializer);

    graph.GetOrCreateNodeArg(gradient_name, &float_tensor_type);
    graph.AddInitializedTensor(gradient_initializer);

    weight_and_gradient_names.emplace(weight_name);
    weight_and_gradient_names.emplace(gradient_name);
    weight_names.emplace_back(weight_name);
  }

  Graph::ResolveOptions resolve_options{};
  resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
  return graph.Resolve(resolve_options);
}

static void TestZeROOptimizerGraphBuilder(OptimizerGraphConfig config,
                                          const std::vector<int64_t>& weight_dims) {
  Model model{"test_model", false, onnxruntime::test::DefaultLoggingManager().DefaultLogger()};
  Graph& graph{model.MainGraph()};
  std::vector<std::string> weight_names;
  ASSERT_STATUS_OK(SetUpZeroGraph(graph, weight_names, weight_dims));

  std::unordered_map<std::string, std::string> updated_weight_names_map;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(weight_names), updated_weight_names_map, weight_partition_info);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, weight_to_opt_mapping, opt_graph_outputs));

  int total_weight_bytes = 0;
  for (auto& weight_dim : weight_dims) {
    size_t bytes;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(weight_dim, sizeof(float), &bytes));
    total_weight_bytes += bytes;
  }

  const size_t alignment = config.data_parallel_group_size * 32;
  const size_t padded_buffer_bytes = ((total_weight_bytes + alignment - 1) / alignment) * alignment;
  int rank_bytes = padded_buffer_bytes / config.data_parallel_group_size;
  const size_t rank_start = config.data_parallel_group_rank * rank_bytes;
  const size_t rank_end = rank_start + rank_bytes;

  int optimizer_node_count = 0;
  size_t offset = 0;
  for (size_t i = 0; i < weight_dims.size(); i++) {
    const size_t tensor_count = weight_dims[i];
    size_t tensor_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(tensor_count, sizeof(float), &tensor_bytes));

    size_t tensor_start_address = offset;
    size_t tensor_end_address = tensor_start_address + tensor_count * sizeof(float);
    if (tensor_start_address < rank_end && tensor_end_address > rank_start) {
      optimizer_node_count++;
    }
    offset += tensor_bytes;
  }

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), weight_names.size());
    ASSERT_EQ(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 1);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    if (optimizer_node_count > 0) {
      ASSERT_EQ(GetOpCount(op_counts, k_gradient_norm_op_name), 1);
    } else {
      ASSERT_EQ(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    }

    ASSERT_EQ(GetOpCount(op_counts, k_is_all_finite_op_name), 1);
  }

  // verify ZeRO operations exist
  ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), weight_names.size());
  ASSERT_EQ(GetOpCount(op_counts, k_reduce_scatter_op_name), 1);
  ASSERT_EQ(GetOpCount(op_counts, k_all_gather_op_name), 1);

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_adam_optimizer_op_name), optimizer_node_count);
}

TEST_F(ZeroOptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  for (int rank = 0; rank < config.data_parallel_group_size; ++rank) {
    config.data_parallel_group_rank = rank;
    TestZeROOptimizerGraphBuilder(config, {12, 2, 312, 1233});
  }
}

TEST_F(ZeroOptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;

  for (int rank = 0; rank < config.data_parallel_group_size; ++rank) {
    config.data_parallel_group_rank = rank;
    TestZeROOptimizerGraphBuilder(config, {123, 22, 312, 123});
  }
}

TEST_F(ZeroOptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 16;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;

  for (int rank = 0; rank < config.data_parallel_group_size; ++rank) {
    config.data_parallel_group_rank = rank;
    TestZeROOptimizerGraphBuilder(config, {12, 412, 312, 123});
  }
}

TEST_F(ZeroOptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 16;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;

  for (int rank = 0; rank < config.data_parallel_group_size; ++rank) {
    config.data_parallel_group_rank = rank;
    TestZeROOptimizerGraphBuilder(config, {12, 231, 312, 1233});
  }
}

#endif  // ORT_USE_NCCL

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
