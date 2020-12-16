// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/graph/model.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/graph_transformer_mgr.h"

#ifdef ENABLE_TRAINING
#include "orttraining/core/optimizer/graph_transformer_utils.h"
#endif

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {
void ApplyCse(Model& model, unsigned num_steps = 1) {
  GraphTransformerManager graph_transformation_mgr(num_steps);
  ASSERT_TRUE(
      graph_transformation_mgr.Register(onnxruntime::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1).IsOK());
  ASSERT_TRUE(
      graph_transformation_mgr.ApplyTransformers(model.MainGraph(), TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());
}

std::vector<std::string> GetSortedNames(const std::vector<const NodeArg*>& node_args) {
  std::vector<std::string> node_arg_names;
  for (const auto* node_arg : node_args) {
    node_arg_names.push_back(node_arg->Name());
  }

  std::sort(node_arg_names.begin(), node_arg_names.end());
  return node_arg_names;
}

std::vector<std::string> GetNodeNames(const Graph& graph) {
  std::vector<std::string> res;
  for (auto& node : graph.Nodes()) {
    res.push_back(node.Name());
  }
  return res;
}

void GetAllNodeNamesImpl(const Graph& graph, std::vector<std::string>& res) {
  for (auto& node : graph.Nodes()) {
    res.push_back(node.Name());
    for (const auto& subgraph : node.GetSubgraphs()) {
      GetAllNodeNamesImpl(*subgraph, res);
    }
  }
}

std::vector<std::string> GetAllNodeNamesSorted(const Graph& graph) {
  std::vector<std::string> res;
  GetAllNodeNamesImpl(graph, res);
  std::sort(res.begin(), res.end());
  return res;
}
}  // namespace

TEST(CseTests, SimpleTest) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse1.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  ApplyCse(*model);

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(graph_inputs, (std::vector<std::string>{"x"}));

  const auto& graph_outputs = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(graph_outputs, (std::vector<std::string>{"Result"}));

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("MatMul"), 1);
  ASSERT_EQ(op_count.at("Add"), 2);
  ASSERT_EQ(op_count.at("Relu"), 1);
}

#ifdef ENABLE_TRAINING
TEST(CseTests, SimpleTestTraining) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse1.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());

  GraphTransformerManager graph_transformation_mgr(1);
  // need to declare variables to avoid build error after making
  // weights_to_train and updated_weight_names as non-const
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, std::string> updated_weight_names;
  auto transformers_to_register = onnxruntime::training::transformer_utils::GeneratePreTrainingTransformers(
      TransformerLevel::Level1, weights_to_train, {}, CPUExecutionProvider(CPUExecutionProviderInfo()), updated_weight_names);
  for (auto& entry : transformers_to_register) {
    ASSERT_TRUE(
        graph_transformation_mgr.Register(std::move(entry), TransformerLevel::Level1).IsOK());
  }
  ASSERT_TRUE(
      graph_transformation_mgr.ApplyTransformers(model->MainGraph(), TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(graph_inputs, (std::vector<std::string>{"x"}));

  const auto& graph_outputs = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(graph_outputs, (std::vector<std::string>{"Result"}));

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("MatMul"), 1);
  ASSERT_EQ(op_count.at("Add"), 2);
  ASSERT_EQ(op_count.at("Relu"), 1);
}
#endif

TEST(CseTests, GraphOutput) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_graph_output.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  ApplyCse(*model);

  Graph& graph = model->MainGraph();

  const auto& input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names, (std::vector<std::string>{"x"}));

  const auto& output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"res1", "res2"}));

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Add"), 2);
}

TEST(CseTests, OptionalArgs) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_optional_args.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  Graph& graph = model->MainGraph();
  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Clip"), 5);

  ApplyCse(*model);

  const auto& input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names, (std::vector<std::string>{"x"}));

  const auto& output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Result"}));

  op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.at("Clip"), 3);
  auto node_names = GetNodeNames(graph);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "clip_3"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "clip_4"), 1);
}

TEST(CseTests, Random) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_random.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  Graph& graph = model->MainGraph();
  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["RandomUniform"], 4);

  ApplyCse(*model);

  const auto& input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names, (std::vector<std::string>{"x"}));

  const auto& output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Result"}));

  op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["RandomUniform"], 4);
  auto node_names = GetNodeNames(graph);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_1"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_2"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_3"), 1);
  ASSERT_EQ(std::count(node_names.begin(), node_names.end(), "random_uniform_4"), 1);
}

TEST(CseTests, Subgraph) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_subgraph.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  Graph& graph = model->MainGraph();
  auto node_names = GetAllNodeNamesSorted(graph);
  ASSERT_EQ(node_names, (std::vector<std::string>{"if_0",
                                                  "iffalse_intermediate_1", "iffalse_intermediate_2", "iffalse_res_1", "iffalse_res_2", "iffalse_res_3",
                                                  "iftrue_intermediate_1", "iftrue_intermediate_2", "iftrue_res_1", "iftrue_res_2", "iftrue_res_3"}));

  ApplyCse(*model);

  const auto& input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names, (std::vector<std::string>{"b", "x"}));

  auto output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Result1", "Result2", "Result3"}));

  const Graph* if_true_graph = nullptr;
  const Graph* if_false_graph = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "If") {
      if_true_graph = node.GetGraphAttribute("then_branch");
      if_false_graph = node.GetGraphAttribute("else_branch");
      break;
    }
  }

  output_names = GetSortedNames(if_true_graph->GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Result1", "Result2", "Result3"}));

  auto op_count = CountOpsInGraph(*if_true_graph);
  ASSERT_EQ(op_count["Mul"], 1);
  ASSERT_EQ(op_count["Sum"], 3);

  output_names = GetSortedNames(if_false_graph->GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Result1", "Result2", "Result3"}));

  op_count = CountOpsInGraph(*if_false_graph);
  ASSERT_EQ(op_count["Mul"], 3);
  ASSERT_EQ(op_count["Sum"], 1);
}

TEST(CseTests, MergedValueAndGraphOutputAreOutputsOfSameNode) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_only_one_graph_output.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  Graph& graph = model->MainGraph();
  ApplyCse(*model);

  const auto& input_names = GetSortedNames(graph.GetInputs());
  ASSERT_EQ(input_names, (std::vector<std::string>{"x"}));

  const auto& output_names = GetSortedNames(graph.GetOutputs());
  ASSERT_EQ(output_names, (std::vector<std::string>{"Add", "Split1Output2", "Split2Output2"}));

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["ReduceSum"], 1);
  ASSERT_EQ(op_count["Add"], 1);
  // In current implementation we don't collapse nodes that produce graph outputs.
  ASSERT_EQ(op_count["Split"], 2);
}

TEST(CseTests, MergeConstants) {
  auto model_uri = ORT_TSTR("testdata/transform/cse/cse_merge_constants.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  Graph& graph = model->MainGraph();
  GraphTransformerManager graph_transformation_mgr(1);
  // In current implementation, equal constants are not merged. So CSE must precede constant folding, otherwise we end up
  // with multiple copies of the same constant.
  std::unique_ptr<CPUExecutionProvider> e = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_TRUE(
      graph_transformation_mgr.Register(onnxruntime::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1).IsOK());
  std::unordered_set<std::string> compatible_execution_providers = {};
  std::unordered_set<std::string> excluded_initializers = {};
  ASSERT_TRUE(
      graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get(), 
      compatible_execution_providers, excluded_initializers), TransformerLevel::Level1).IsOK());
  ASSERT_TRUE(
      graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  ASSERT_EQ(graph.GetAllInitializedTensors().size(), 1U);
  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count.size(), 1U);
  ASSERT_EQ(op_count["Add"], 2);
}

}  // namespace test
}  // namespace onnxruntime
