// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <cmath>
#include <functional>
#include <fstream>
#include <random>
#include <sstream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/onnx_protobuf.h"

#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"

#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/int4.h"
#include "core/framework/ort_value.h"
#include "core/framework/resource_accountant.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/bias_dropout_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/cast_chain_elimination.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/concat_slice_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/div_mul_fusion.h"
#include "core/optimizer/double_qdq_pairs_remover.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/dynamic_quantize_matmul_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/gather_fusion.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/gemm_sum_fusion.h"
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/graph_transformer_config.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/isinf_reducesum_fusion.h"
#include "core/optimizer/label_encoder_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_bn_fusion.h"
#include "core/optimizer/matmul_nbits_fusion.h"
#include "core/optimizer/matmul_integer_to_float.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/noop_elimination.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/pad_fusion.h"
#include "core/optimizer/pre_shape_node_elimination.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/quick_gelu_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/slice_concat_to_space_to_depth_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/stft_decomposition.h"
#include "core/optimizer/transpose_optimization/onnx_transpose_optimization.h"
#include "core/optimizer/transpose_optimization/optimizer_api.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/util/math.h"
#include "test/capturing_sink.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/compare_ortvalue.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/optimizer/webgpu_fusion_test_util.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test_utils.h"
#ifdef ENABLE_TRAINING
#include "orttraining/core/optimizer/bitmask_dropout_replacement.h"
#endif

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

namespace {

class NoOpTraceGraphTransformer : public GraphTransformer {
 public:
  NoOpTraceGraphTransformer() : GraphTransformer("NoOpTraceGraphTransformer") {}

 private:
  Status ApplyImpl(Graph& /*graph*/, bool& modified, int /*graph_level*/,
                   const logging::Logger& /*logger*/) const override {
    modified = false;
    return Status::OK();
  }
};

}  // namespace

TEST_F(GraphTransformationTests, GraphTransformerManagerEmitsVerboseTrace) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));

  auto capturing_sink = std::make_unique<CapturingSink>();
  const auto* capturing_sink_raw = capturing_sink.get();
  logging::LoggingManager logging_manager(std::move(capturing_sink), logging::Severity::kVERBOSE, false,
                                          logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("GraphTransformerTraceTest");

  GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<NoOpTraceGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(model->MainGraph(), TransformerLevel::Level1, *logger));

  const auto& messages = capturing_sink_raw->Messages();
  EXPECT_THAT(messages, testing::Contains(testing::HasSubstr(
                            "Applying 1 graph transformer(s) for level 1 for up to 5 step(s).")));
  EXPECT_THAT(messages, testing::Contains(testing::HasSubstr(
                            "Applying graph transformer NoOpTraceGraphTransformer on step 1.")));
  EXPECT_THAT(messages, testing::Contains(testing::HasSubstr(
                            "Graph transformer step 1 of 5 for level 1 completed. graph_changed: 0.")));
  EXPECT_THAT(messages, testing::Contains(testing::HasSubstr(
                            "Stopping graph transformer iteration for level 1 after step 1 because the graph was not modified.")));
}

TEST_F(GraphTransformationTests, IdentityElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

// Registering two transformers with the same Name() at the same level must fail (per-level
// uniqueness); the same name at a different level is allowed.
TEST_F(GraphTransformationTests, RegisterDuplicateTransformerNameFailsPerLevel) {
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1));

  // Same name, same level -> rejected.
  const auto duplicate_status = graph_transformation_mgr.Register(
      std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1);
  ASSERT_FALSE(duplicate_status.IsOK());
  ASSERT_THAT(duplicate_status.ErrorMessage(), ::testing::HasSubstr("already registered"));

  // Same name, different level -> allowed.
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level2));
}

TEST_F(GraphTransformationTests, IdentityEliminationWithGraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "abs-id.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST_F(GraphTransformationTests, IdentityWithSharedNodeArgNotEliminated) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "id-elim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(),
                                                     TransformerLevel::Level1));
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // after CommonSubexpressionElimination, Add would have 1 output def and 2 edges
  // each edge would share the same input node arg 0. Thus after execution, only one of the 2 outputs
  // has data. Thus skip.
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

// Runs a model to ensure that common subexpression elimination does not eliminate
// DequantizeLinear nodes.
TEST_F(GraphTransformationTests, DequantizeLinearNodeNotEliminated) {
  auto test_case = [](const ORTCHAR_T* model_uri,
                      bool use_contrib_qdq,
                      const logging::Logger& logger) {
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count[dq_key], 25);

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(),
                                                       TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

    // CommonSubexpressionElimination should skip the DequantizeLinear nodes
    op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count[dq_key], 25);
  };

  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.onnx",
            false,  // use_contrib_qdq
            *logger_);

  // Test with 16-bit DequantizeLinear(21)
  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.qdq16.onnx",
            false,  // use_contrib_qdq
            *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit com.microsoft.DequantizeLinear
  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.qdq_contrib.onnx",
            true,  // use_contrib_qdq
            *logger_);
  // Test with 16-bit com.microsoft.DequantizeLinear
  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.qdq16_contrib.onnx",
            true,  // use_contrib_qdq
            *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

TEST_F(GraphTransformationTests, IdentityInputIsGraphOutputNotEliminated) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "scan9_sum.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  // tips: to dump the subgraph, can use python tool - dump_subgraphs.py
  // or click on one of the input to see the drop down graph list and view subgraph

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Identity's input in subgraph is also graph output. Thus skip.
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
}

TEST_F(GraphTransformationTests, NoopElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "noop-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 5);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<NoopElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // 3 Adds eliminated (scalar/1-element zero initializers).
  // 2 Adds preserved:
  //   - one whose initializer rank exceeds the other input's rank (broadcasts up).
  //   - one with a zero-element initializer (shape [0]); a 0-element tensor is
  //     not a proven identity for Add, so NoopElimination must not remove it.
  ASSERT_TRUE(op_to_count["Add"] == 2);

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] + CountOpsInGraph(graph)["Sub"] + CountOpsInGraph(graph)["Mul"] +
                           CountOpsInGraph(graph)["Div"] ==
                       1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] + CountOpsInGraph(graph)["Sub"] + CountOpsInGraph(graph)["Mul"] +
                           CountOpsInGraph(graph)["Div"] ==
                       0);
    return Status::OK();
  };

  // x+0, float.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {0.0f});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {matmul_output, initializer_arg}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // 0+x, fp16.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<MLFloat16>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<MLFloat16>({1}, {MLFloat16(0.0f)});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {initializer_arg, matmul_output}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x-0, double.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(0.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {matmul_output, initializer_arg}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*1, int32.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int32_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int32_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int32_t>({1, 1, 1}, {1});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {matmul_output, initializer_arg}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // 1*x, int64.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int64_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int64_t>({1, 1, 1, 1}, {static_cast<int64_t>(1)});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x/1, float.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* div_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Div", {matmul_output, initializer_arg}, {div_out});
      builder.AddNode("Identity", {div_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Invalid case: x+1.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {matmul_output, initializer_arg}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: initializer rank is larger.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<MLFloat16>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<MLFloat16>({1, 1, 1, 1, 1}, {MLFloat16(0.0f)});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {initializer_arg, matmul_output}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 0-x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(0.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {initializer_arg, matmul_output}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: x-1.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(1.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {matmul_output, initializer_arg}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 0*x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int32_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int32_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int32_t>({1, 1, 1}, {0});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: output is graph output.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int64_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int64_t>({1, 1, 1, 1}, {static_cast<int64_t>(1)});
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 1/x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* div_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Div", {initializer_arg, matmul_output}, {div_out});
      builder.AddNode("Identity", {div_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }
}

TEST_F(GraphTransformationTests, DropoutElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 6);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateDropout>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Of the 6 Dropout nodes in the graph, all but the ones named `d1` and `d6` should have been removed.
  // A Dropout node can be removed if its second, optional output `mask` is either missing or unused downstream.
  // `d1` cannot be removed because an Identity node has its `mask` output as an input;
  // `d6` cannot be removed because its `mask` output is marked as a graph output.
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

TEST_F(GraphTransformationTests, SliceElimination) {
  std::vector<std::basic_string<ORTCHAR_T>> model_names = {ORT_TSTR("slice-v1-elim.onnx"), ORT_TSTR("slice-v11-elim.onnx")};
  for (const auto& model_name : model_names) {
    PathString model_uri = PathString(MODEL_FOLDER) + model_name;
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    int initial_slice_num = op_to_count["Slice"];

    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateSlice>()));
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    op_to_count = CountOpsInGraph(graph);
    // Only one Slice operator is redundant and is removed.
    ASSERT_TRUE(op_to_count["Slice"] == --initial_slice_num);
  }
}

TEST_F(GraphTransformationTests, ConstantFolding) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Unsqueeze"], 2);

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingCopiesAliasedTensorBuffer) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> values(64);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(i + 1);
    }

    auto* input = builder.MakeInitializer<float>({64}, values);
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{64});
    builder.AddNode("Identity", {input}, std::vector<NodeArg*>{output});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    ORT_RETURN_IF_ERROR(graph.Resolve());

    const auto& outputs = graph.GetOutputs();
    TEST_RETURN_IF_NOT(outputs.size() == 1U);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 1U);

    const ONNX_NAMESPACE::TensorProto* folded_tensor = nullptr;
    TEST_RETURN_IF_NOT(graph.GetInitializedTensor(outputs[0]->Name(), folded_tensor));
    TEST_RETURN_IF_NOT(folded_tensor != nullptr);

    Initializer initializer{graph, *folded_tensor, graph.ModelPath()};
    TEST_RETURN_IF_NOT(initializer.size() == 64U);
    const float* data = initializer.data<float>();
    for (size_t i = 0; i < 64; ++i) {
      TEST_RETURN_IF_NOT(data[i] == static_cast<float>(i + 1));
    }

    return Status::OK();
  };

  const ConfigOptions empty_config_options;
  auto cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case,
      13,
      *logger_,
      std::make_unique<ConstantFolding>(*cpu_ep, false, empty_config_options),
      TransformerLevel::Level1,
      1,
      pre_graph_checker,
      post_graph_checker));
}

TEST_F(GraphTransformationTests, ConstantFoldingNodesOnDifferentEP) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  // assign all nodes to CUDA. the constant folding should override this to perform the constant folding on cpu
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);

  // all remaining nodes should still be on CUDA
  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingTransposeEmptyInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "constant_folding_transpose_empty_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 1);
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingUnsupportedFloat16TopK) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "constant_float16_topk.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["TopK"] == 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const ConfigOptions empty_config_options;
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  // assign all nodes to CUDA. constant folding will be attempted on CPU but fail, leaving nodes on CUDA.
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["TopK"] == 1);

  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingSubgraph) {
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TypeProto float_tensor_type;
  float_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_subgraph = [&](GraphProto& graph_proto) {
    // create subgraph that has an Add node to add a local and parent graph initializer
    Model model("ConstantFoldingSubgraphTest_subgraph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
    auto& graph = model.MainGraph();

    TensorProto local_constant(value_tensor);
    local_constant.set_name("local_constant");
    graph.AddInitializedTensor(local_constant);

    auto& local_constant_arg = graph.GetOrCreateNodeArg("local_constant", &float_tensor_type);
    auto& parent_constant_arg = graph.GetOrCreateNodeArg("parent_constant", &float_tensor_type);
    graph.AddOuterScopeNodeArg("parent_constant");

    auto& add_out = graph.GetOrCreateNodeArg("add_out", &float_tensor_type);
    graph.AddNode("add", "Add", "Add two inputs.", {&parent_constant_arg, &local_constant_arg}, {&add_out});

    auto& subgraph_out = graph.GetOrCreateNodeArg("subgraph_out", &float_tensor_type);
    graph.AddNode("identity", "Identity", "So Add isn't providing graph output.", {&add_out}, {&subgraph_out});

    ASSERT_STATUS_OK(graph.Resolve());
    graph_proto = graph.ToGraphProto();
  };

  Model model("ConstantFoldingSubgraphTest_main_graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
  auto& graph = model.MainGraph();

  // add initializer at parent level
  TensorProto parent_value_tensor(value_tensor);
  parent_value_tensor.set_name("parent_constant");
  graph.AddInitializedTensor(parent_value_tensor);

  // put the subgraph in an If node
  TypeProto if_cond_type;
  if_cond_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  if_cond_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& if_cond_input = graph.GetOrCreateNodeArg("if_in", &if_cond_type);
  auto& if_output = graph.GetOrCreateNodeArg("if_out", &float_tensor_type);

  auto& if_node = graph.AddNode("if", "If", "If node", {&if_cond_input}, {&if_output});

  GraphProto subgraph;
  create_subgraph(subgraph);

  if_node.AddAttribute("then_branch", subgraph);
  if_node.AddAttribute("else_branch", subgraph);

  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 2);  // one in each subgraph
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0)
      << "Constant folding should have been able to remove the Add node in both subgraphs";
}

TEST_F(GraphTransformationTests, ConstantFoldingWithShapeToInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 2);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 3);

  InlinedHashSet<std::string_view> compatible_eps;
  InlinedHashSet<std::string> excluded_initializers = {"matmul_weight"};
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(),
                                        false /*skip_dequantize_linear*/,
                                        empty_config_options,
                                        compatible_eps,
                                        excluded_initializers),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingWithScalarShapeToInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options,
                                        compatible_eps),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, ConstantFoldingForOpsWithMissingOptionalInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_for_ops_having_missing_optional_inputs.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());

  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // The Resize node has some missing optional inputs (roi and scales)
  ASSERT_TRUE(op_to_count["Resize"] == 1);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options,
                                        compatible_eps),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // The Resize node is constant folded
  ASSERT_TRUE(op_to_count["Resize"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);
}

TEST_F(GraphTransformationTests, ConstantFoldingForOpsWithMissingOptionalOutputs) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInitializer<int64_t>(
        {6}, {5, 5, 5, 10, 10, 20});

    auto* unique_values = builder.MakeOutput<int64_t>(std::nullopt);
    auto* unique_counts = builder.MakeOutput<int64_t>(std::nullopt);

    auto* missing_indices = builder.MakeEmptyInput();
    auto* missing_inverse_indices = builder.MakeEmptyInput();

    builder.AddNode(
        "Unique",
        {input},
        {unique_values, missing_indices, missing_inverse_indices, unique_counts});
  };

  auto pre_graph_checker = [](Graph& graph) {
    const auto ops = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(ops.count("Unique") == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    const auto ops = CountOpsInGraph(graph);

    TEST_RETURN_IF(ops.find("Unique") != ops.end());

    const auto& outputs = graph.GetOutputs();
    TEST_RETURN_IF_NOT(outputs.size() == 2U);

    const auto& init_tensors = graph.GetAllInitializedTensors();

    bool saw_values = false;
    bool saw_counts = false;

    for (const NodeArg* out : outputs) {
      auto it = init_tensors.find(out->Name());
      TEST_RETURN_IF(it == init_tensors.end());

      onnxruntime::Initializer init{
          graph,
          *it->second,
          graph.ModelPath()};

      TEST_RETURN_IF_NOT(init.size() == 3U);

      const int64_t* data = init.data<int64_t>();

      if (data[0] == 5 && data[1] == 10 && data[2] == 20) {
        if (saw_values) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "Duplicate values output detected");
        }
        saw_values = true;

      } else if (data[0] == 3 && data[1] == 2 && data[2] == 1) {
        if (saw_counts) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "Duplicate counts output detected");
        }
        saw_counts = true;

      } else {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "Unexpected tensor content after constant folding");
      }
    }

    TEST_RETURN_IF_NOT(saw_values);
    TEST_RETURN_IF_NOT(saw_counts);

    return Status::OK();
  };

  const ConfigOptions empty_config_options;
  auto cpu_ep =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case,
      13,
      *logger_,
      std::make_unique<ConstantFolding>(
          *cpu_ep,
          false,
          empty_config_options),
      TransformerLevel::Level1,
      1,
      pre_graph_checker,
      post_graph_checker));
}

static void VerifyConstantFoldingWithDequantizeLinear(const std::unordered_map<std::string, int>& expected_op_count,
                                                      Graph& graph,
                                                      SessionOptions& session_options,
                                                      const Logger& logger) {
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  bool has_constant_folding = false;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, session_options, *e.get(), logger,
                                                            {});
  for (auto& transformer : transformers) {
    if (transformer->Name() == "ConstantFolding") {
      ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), TransformerLevel::Level1));
      has_constant_folding = true;
    }
  }

  ASSERT_TRUE(has_constant_folding);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  for (const auto& entry : expected_op_count) {
    if (entry.second == 0) {
      ASSERT_TRUE(op_to_count.find(entry.first) == op_to_count.end())
          << entry.first << " should not exist in the graph";
    } else {
      ASSERT_TRUE(op_to_count[entry.first] == entry.second)
          << entry.first << " mismatch. Expected:" << entry.second << " Got:" << op_to_count[entry.first];
    }
  }
}

// Runs a model that checks constant folding with DequantizeLinear nodes.
TEST_F(GraphTransformationTests, ConstantFoldingWithDequantizeLinear) {
  auto test_case = [](const ORTCHAR_T* model_uri,
                      bool use_contrib_qdq,
                      const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 1);
    ASSERT_TRUE(op_to_count[dq_key] == 3);
    ASSERT_TRUE(op_to_count["Conv"] == 1);

    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 1},
                                                               {dq_key, 3},
                                                               {"Conv", 1}};

    SessionOptions session_options;
    // Check DequantizeLinear aren't constant folded for default setting.
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);

    // set kOrtSessionOptionsDisableQuantQDQ to enable it explicitly
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "0"));
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);

    // set SessionOptionsEnableQuantQDQ to disable it
    expected_op_counts[dq_key] = 1;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.onnx",
            false, *logger_);

  // Test with 16-bit DequantizeLinear(21).
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq16.onnx",
            false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq_contrib.onnx",
            true, *logger_);
  // Test with 16-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq16_contrib.onnx",
            true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// Verify that setting session.disable_qdq_constant_folding to "1" preserves DequantizeLinear nodes
// even when session.disable_quant_qdq is "1" (which normally allows DQ to be constant folded).
TEST_F(GraphTransformationTests, ConstantFoldingDisableQDQConstantFolding) {
  auto test_case = [](const ORTCHAR_T* model_uri,
                      bool use_contrib_qdq,
                      const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 1);
    ASSERT_TRUE(op_to_count[dq_key] == 3);
    ASSERT_TRUE(op_to_count["Conv"] == 1);

    // With disable_quant_qdq="1" and disable_qdq_constant_folding="1", DQ nodes should be preserved.
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsDisableQDQConstantFolding, "1"));

    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 1},
                                                               {dq_key, 3},
                                                               {"Conv", 1}};
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.onnx",
            false, *logger_);

  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq16.onnx",
            false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq_contrib.onnx",
            true, *logger_);
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq16_contrib.onnx",
            true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// model with 2 QDQ node units that can be constant folded as they are simple DQ -> Node -> Q where DQ and Node have
// single consumer and do not produce graph outputs. Node is deterministic.
// there are also other DQ nodes that should be ignored.
TEST_F(GraphTransformationTests, ConstantFoldingQDQNodeUnit) {
  auto test_case = [](const ORTCHAR_T* model_uri, bool use_contrib_qdq, const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 3);
    ASSERT_TRUE(op_to_count[dq_key] == 4);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 1);

    SessionOptions session_options;

    // 2 QDQ node units should be constant folded and go away
    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 1},
                                                               {dq_key, 2},
                                                               {"Transpose", 0},
                                                               {"Unsqueeze", 0}};

    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.onnx", false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit com.microsoft.Q/DQ
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.qdq_contrib.onnx", true, *logger_);
  // Test with 16-bit com.microsoft.Q/DQ
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.qdq16_contrib.onnx", true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// Simple QDQ Node Unit but shouldn't be constant folded as the node in the middle produces a graph output
TEST_F(GraphTransformationTests, ConstantFoldingQDQNodeUnitGraphOutput) {
  auto test_case = [](const ORTCHAR_T* model_uri, bool use_contrib_qdq, const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 2);
    ASSERT_TRUE(op_to_count[dq_key] == 3);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 1);

    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 2},
                                                               {dq_key, 3},
                                                               {"Unsqueeze", 1}};

    SessionOptions session_options;
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.onnx", false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.qdq_contrib.onnx", true, *logger_);

  // Test with 16-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.qdq16_contrib.onnx", true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

TEST_F(GraphTransformationTests, ConstantFolding_RemoveDanglingInputNodesToConstantFoldedNode) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_remove_dangling_inputs.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);          // Shape node that will be constant folded
  ASSERT_TRUE(op_to_count["Add"] == 1);            // Input node to Shape
  ASSERT_TRUE(op_to_count["RandomUniform"] == 1);  // Input node to Add

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["RandomUniform"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingAShapeNodeDeepInTheGraph) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "shape-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 4);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  // A Shape node very deep in the graph (feeding into an Identity
  // node that produces the graph output) gets constant folded which
  // removes all its ancestors and the Identity node consuming this Shape's
  // output is subsequently constant folded to leave the graph with no
  // nodes.
  ASSERT_TRUE(op_to_count.size() == 0U);
}

// Test we don't fail when constant folding hits a string initializer
TEST_F(GraphTransformationTests, ConstantFoldingStringInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "gh_issue_17392.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const ConfigOptions empty_config_options;
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/, empty_config_options),
      TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count.size(), 0U) << "Identity node should have been removed";
}

TEST_F(GraphTransformationTests, ConstantFoldingIfConstantInlining) {
  // This test covers the following necessary cases:
  // The input refers to the explicit or implicit inputs of If node.
  // The output of the node is the output of the subgraph being inlined.
  // Constant nodes and initializers are promoted to the outer graph.
  // The initializer or a constant node is the output of the subgraph being inlined.
  // Nested subgraphs names are renamed as appropriate.
  // In all If node is constant folded twice. The last If node is not constant
  // folded because the input is indirectly dependent on the size of the input.
  // XXX: Can we constant fold Size() if the graph input shape is fixed?

  const char* code = R"(
  <
  ir_version: 8,
  opset_import: [ "" : 16, "local" : 1 ]
  >
  agraph (float[128] x, float[128] x1) => (float[N] y)
  {
      y = local.aten_gather <dim: int = 1, sparse_grad: int = 0> (x, x1)
  }
  <
    opset_import: [ "" : 16, "local" : 1],
    domain: "local"
  >
  aten_gather <dim>(self, index) => (result_16)
  {
     tmp = Shape (index)
     tmp_0 = Size (tmp)
     int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
     int64_0_cast = CastLike (int64_0, tmp_0)
     cond = Equal (tmp_0, int64_0_cast)
     result_16 = If (cond) <then_branch: graph = thenGraph_10 () => ( result) {
        result = Identity (self)
     }, else_branch: graph = elseGraph_10 () => ( result_15) {
        tmp_1 = Shape (self)
        tmp_2 = Size (tmp_1)
        int64_0_3 = Constant <value: tensor = int64 int64_0_3 {0}> ()
        int64_0_3_cast = CastLike (int64_0_3, tmp_2)
        cond_4 = Equal (tmp_2, int64_0_3_cast)
        self_8 = If (cond_4) <then_branch: graph = thenGraph_13 () => ( self_6) {
           tmp_5 = Constant <value_ints: ints = [-1]> ()
           self_6 = Reshape (self, tmp_5)
        }, else_branch: graph = elseGraph_13 () => ( self_7) {
           self_7 = Identity (self)
        }>
        tmp_9 = Size (index)
        int64_0_10 = Constant <value: tensor = int64 int64_0_10 {0}> ()
        int64_0_10_cast = CastLike (int64_0_10, tmp_9)
        cond_11 = Equal (tmp_9, int64_0_10_cast)
        result_15 = If (cond_11) <then_branch: graph = thenGraph_15 () => ( result_12) {
           result_12 = CastLike (index, self_8)
        }, else_branch: graph = elseGraph_15 () => ( result_14) {
           index_13 = Cast <to: int = 7> (index)
           result_14 = GatherElements <axis: int = @dim> (self_8, index_13)
        }>
     }>
  }
)";

  ONNX_NAMESPACE::OnnxParser parser(code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  {
    // Test that the model is loadable and check the function call node.
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(std::move(model_proto), p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["local.aten_gather"], 1);
    model_proto = p_model->ToProto();
  }

  std::string serialized_model;
  const bool serialization_status = model_proto.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";

  // AOT inlining is necessary in this case, so the If nodes within the function
  // are brought out to the outer scope. So we load this into a session object.

  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  // const auto resulting_model_proto = session_object.GetModel().ToProto();
  // std::string printed_model = ONNX_NAMESPACE::ProtoToString(resulting_model_proto);
  // ASSERT_FALSE(printed_model.empty());
  // std::cout << printed_model << std::endl;

  // This is the resulting model proto.
  // The remaining If node is not constant foldable because Size() does not constant fold
  // although the shape is fixed.
  /*
    <
       ir_version: 8,
       opset_import: ["" : 16, "local" : 1,
         "com.microsoft.nchwc" : 1,
          "ai.onnx.ml" : 4,
          "com.ms.internal.nhwc" : 20,
          "ai.onnx.training" : 1,
          "ai.onnx.preview.training" : 1,
          "com.microsoft" : 1,
          "com.microsoft.experimental" : 1,
          "org.pytorch.aten" : 1]
    >
    agraph (float[128] x, float[128] x1) => (float[128] y) {
       _if_elseGraph_10__inlfunc_aten_gather_tmp_9 = Size (x1)
       _if_elseGraph_10__inlfunc_aten_gather_cond_11 =
                                  Equal (_if_elseGraph_10__inlfunc_aten_gather_tmp_9, ortshared_7_0_1_0_token_10)
       y = If (_if_elseGraph_10__inlfunc_aten_gather_cond_11) <then_branch: graph = thenGraph_15 () => (float[128] _inlfunc_aten_gather_result_12) {
          _inlfunc_aten_gather_result_12 = Cast <to: int = 1> (x1)
       }, else_branch: graph = elseGraph_15 () => (float[128] _inlfunc_aten_gather_result_14) {
          _inlfunc_aten_gather_index_13 = Cast <to: int = 7> (x1)
          _inlfunc_aten_gather_result_14 = GatherElements <axis: int = 1> (x, _inlfunc_aten_gather_index_13)
       }>
    }
  */

  auto& graph = session_object.GetModel().MainGraph();
  auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["local.aten_gather"], 0);
  ASSERT_EQ(op_to_count["If"], 1);
}

TEST_F(GraphTransformationTests, ConstantFoldingIfConstantInliningRebuildEdges) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "transform_nested_ifs_toplogical_sorted_nodes.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.ConstantFoldingIfConstantInliningRebuildEdges";

  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  auto& graph = session_object.GetModel().MainGraph();
  auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["pkg.onnxscript.torch_lib._aten_linalg_vector_norm_no_dim_onnx"], 0);
  ASSERT_EQ(op_to_count["If"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  ASSERT_EQ(op_to_count["Abs"], 1);
  ASSERT_EQ(op_to_count["Mul"], 1);
  ASSERT_EQ(op_to_count["ReduceSum"], 1);
  ASSERT_EQ(op_to_count["Sqrt"], 1);
  ASSERT_EQ(op_to_count["Cast"], 2);
}

TEST_F(GraphTransformationTests, ConstantFoldingIfConstantInliningEdgesWithMiddleArgNonExisting) {
  // This model has a Resize() call with a middle argument non-existing.
  // We want to make sure that the input edges for that Resize() node
  // are properly rebuilt with a middle argument non-existing
  // during If constant folding
  // This test is only valid if Resize() node resides in the nested subgraph which gets inlined
  // however, the destination graph must not be the main graph. Then we test that the edges are rebuild
  // properly. Also Resize() should not be the first node in the resulting subgraph, so it has edges
  const char* code = R"(
  <
  ir_version: 8,
  opset_import: [ "" : 16, "local" : 1 ]
  >
  agraph (float[128] x, float[128] x1) => (float[N] y)
  {
      y = local.aten_gather <dim: int = 1, sparse_grad: int = 0> (x, x1)
  }
  <
    opset_import: [ "" : 16, "local" : 1],
    domain: "local"
  >
  aten_gather <dim>(self, index) => (result_16)
  {
     resize_scales = Constant <value_floats: floats = [1.5]> ()
     tmp_0 = Size (index)
     int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
     int64_0_cast = CastLike (int64_0, tmp_0)
     cond = Equal (tmp_0, int64_0_cast)
     result_16 = If (cond) <then_branch: graph = thenGraph_10 () => ( result) {
        result = Identity (self)
     }, else_branch: graph = elseGraph_10 () => ( result_15) {
        tmp_1 = Shape (self)
        tmp_2 = Size (tmp_1)
        int64_0_3 = Constant <value: tensor = int64 int64_0_3 {0}> ()
        int64_0_3_cast = CastLike (int64_0_3, tmp_2)
        cond_4 = Equal (tmp_2, int64_0_3_cast)
        self_8 = If (cond_4) <then_branch: graph = thenGraph_13 () => ( self_6) {
           tmp_5 = Constant <value_ints: ints = [-1]> ()
           self_6 = Reshape (self, tmp_5)
        }, else_branch: graph = elseGraph_13 () => ( self_7) {
           self_71 = Mul(self, self)
           float_size = CastLike (tmp_0, resize_scales)
           non_constant_resize_scales = Mul(float_size, resize_scales)
           self_7 = Resize(self_71,, non_constant_resize_scales)
        }>
        tmp_9 = Size (index)
        int64_0_10 = Constant <value: tensor = int64 int64_0_10 {0}> ()
        int64_0_10_cast = CastLike (int64_0_10, tmp_9)
        cond_11 = Equal (tmp_9, int64_0_10_cast)
        result_15 = If (cond_11) <then_branch: graph = thenGraph_15 () => ( result_12) {
           result_12 = CastLike (index, self_8)
        }, else_branch: graph = elseGraph_15 () => ( result_14) {
           index_13 = Cast <to: int = 7> (index)
           result_14 = GatherElements <axis: int = @dim> (self_8, index_13)
        }>
     }>
  }
  )";

  /** Optimized model graph
  <
     ir_version: 8,
     opset_import: ["" : 16,
     "local" : 1,
     "com.microsoft.nchwc" : 1,
     "ai.onnx.ml" : 4,
     "ai.onnx.training" : 1,
     "ai.onnx.preview.training" : 1,
     "com.microsoft" : 1,
     "com.microsoft.experimental" : 1, "org.pytorch.aten" : 1]
  >
  agraph (float[128] x, float[128] x1) => (float[128] y)
     <float[1] _inlfunc_aten_gather_resize_scales =  {1.5}, int64 ortshared_7_0_1_0_token_8 =  {0}>
  {
     _inlfunc_aten_gather_tmp_0 = Size (x1)
     _inlfunc_aten_gather_cond = Equal (_inlfunc_aten_gather_tmp_0, ortshared_7_0_1_0_token_8)
      y = If (_inlfunc_aten_gather_cond) <then_branch: graph = thenGraph_10 () =>
          (float[128] _inlfunc_aten_gather_result) {
        _inlfunc_aten_gather_result = Identity (x)
     }, else_branch: graph = elseGraph_10 () => (float[128] _inlfunc_aten_gather_result_15)
        <int64 _inlfunc_aten_gather_int64_0_10 =  {0}>
  {
        _if_else_branch__inlfunc_aten_gather_self_71 = Mul (x, x)
        _if_else_branch__inlfunc_aten_gather_float_size = Cast <to: int = 1> (_inlfunc_aten_gather_tmp_0)
        _if_else_branch__inlfunc_aten_gather_non_constant_resize_scales = Mul (
          _if_else_branch__inlfunc_aten_gather_float_size, _inlfunc_aten_gather_resize_scales)
        _inlfunc_aten_gather_self_8 = Resize <exclude_outside: int = 0, coordinate_transformation_mode:
                string = "half_pixel", cubic_coeff_a: float = -0.75, extrapolation_value: float = 0, mode:
                string = "nearest", nearest_mode: string = "round_prefer_floor"> (
                    _if_else_branch__inlfunc_aten_gather_self_71, ,
                    _if_else_branch__inlfunc_aten_gather_non_constant_resize_scales)
        _inlfunc_aten_gather_tmp_9 = Size (x1)
        _inlfunc_aten_gather_cond_11 = Equal (_inlfunc_aten_gather_tmp_9, _inlfunc_aten_gather_int64_0_10)
        _inlfunc_aten_gather_result_15 = If (_inlfunc_aten_gather_cond_11) <then_branch: graph = thenGraph_15 () =>
              (float[128] _inlfunc_aten_gather_result_12) {
           _inlfunc_aten_gather_result_12 = Cast <to: int = 1> (x1)
        }, else_branch: graph = elseGraph_15 () => (float[128] _inlfunc_aten_gather_result_14) {
           _inlfunc_aten_gather_index_13 = Cast <to: int = 7> (x1)
           _inlfunc_aten_gather_result_14 = GatherElements <axis: int = 1> (
                          _inlfunc_aten_gather_self_8, _inlfunc_aten_gather_index_13)
        }>
     }>
  }

  */

  ONNX_NAMESPACE::OnnxParser parser(code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  std::string serialized_model;
  const bool serialization_status = model_proto.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";

  // AOT inlining is necessary in this case, so the If nodes within the function
  // are brought out to the outer scope. So we load this into a session object.
  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};
  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Let's verify the correctness of the rebuild edges in the Resize node that still
  // resides within an if else subgraph.
  auto& graph = session_object.GetModel().MainGraph();
  auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["If"], 2);
  ASSERT_EQ(op_to_count["Resize"], 1);

  auto if_node = std::find_if(graph.Nodes().begin(), graph.Nodes().end(),
                              [](const auto& node) { return node.OpType() == "If"; });
  ASSERT_NE(graph.Nodes().cend(), if_node);
  // Resize is in the else branch
  auto subgraph_map = if_node->GetAttributeNameToSubgraphMap();
  auto branch = subgraph_map.find("else_branch");
  ASSERT_NE(subgraph_map.cend(), branch);

  auto resize_node = std::find_if(branch->second->Nodes().begin(), branch->second->Nodes().end(),
                                  [](const auto& node) { return node.OpType() == "Resize"; });
  ASSERT_NE(branch->second->Nodes().cend(), resize_node);

  // Check the edges
  ASSERT_EQ(2U, resize_node->GetInputEdgesCount());
  // Should have input edges with arg_pos 0 and 2
  // With 1 is missing
  InlinedHashSet<size_t> dest_edges;
  auto zero_edge = resize_node->InputEdgesBegin();
  dest_edges.insert(zero_edge->GetDstArgIndex());
  ++zero_edge;
  dest_edges.insert(zero_edge->GetDstArgIndex());
  ASSERT_TRUE(dest_edges.find(0) != dest_edges.end());
  ASSERT_TRUE(dest_edges.find(2) != dest_edges.end());
}

// Test that constant folding respects the output size limit and skips nodes
// whose output would exceed it. This is a security measure against malicious
// models that could cause memory exhaustion during optimization.
TEST_F(GraphTransformationTests, ConstantFoldingOutputSizeLimit) {
  // Build a model with an Expand node: scalar input [1.0] expanded by shape [1024, 1024].
  // Output = 1024*1024 * 4 bytes = 4 MB of float data.
  // With a 1 MB limit, this should NOT be constant folded.
  // With a 8 MB limit, this SHOULD be constant folded.

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInitializer<float>({1}, {1.0f});
    auto* shape_data = builder.MakeInitializer<int64_t>({2}, {1024, 1024});
    auto* output_arg = builder.MakeOutput();

    builder.AddNode("Expand", {input_data, shape_data}, {output_arg});
  };

  // Test 1: With a 1 MB limit, the Expand node should NOT be folded (output is ~4 MB).
  {
    auto pre_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      // Expand should remain because output is too large
      TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
      return Status::OK();
    };

    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    ConfigOptions config_options;
    // Set limit to 1 MB
    ASSERT_STATUS_OK(config_options.AddConfigEntry(
        kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "1048576"));

    ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                          std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Test 2: With an 8 MB limit, the Expand node SHOULD be folded (output is ~4 MB).
  {
    auto pre_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      // Expand should be folded since output is within limit
      TEST_RETURN_IF_NOT(op_to_count["Expand"] == 0);
      return Status::OK();
    };

    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    ConfigOptions config_options;
    // Set limit to 8 MB
    ASSERT_STATUS_OK(config_options.AddConfigEntry(
        kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "8388608"));

    ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                          std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

// Test that an explicitly configured constant folding output-size limit blocks
// folding a very large ConstantOfShape output.
TEST_F(GraphTransformationTests, ConstantFoldingConfiguredLimitBlocksLargeConstantOfShape) {
  // Build a model with a ConstantOfShape node producing a huge output.
  // Shape = [16384, 16384] = 268M elements * 4 bytes = 1 GB.
  // Use an explicit 512 MB limit so the 1 GB output is not folded.

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* shape_data = builder.MakeInitializer<int64_t>({2}, {16384, 16384});
    auto* output_arg = builder.MakeOutput();

    auto& node = builder.AddNode("ConstantOfShape", {shape_data}, {output_arg});
    // Default value is float 0.0
    ONNX_NAMESPACE::AttributeProto value_attr;
    value_attr.set_name("value");
    value_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    auto* tensor = value_attr.mutable_t();
    tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor->add_dims(1);
    tensor->add_float_data(0.0f);
    node.AddAttributeProto(std::move(value_attr));
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    // ConstantOfShape should remain because output is too large (1 GB > 512 MB limit)
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ConfigOptions config_options;
  // Set limit to 512 MB so the 1 GB output is blocked
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "536870912"));

  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                        TransformerLevel::Level1, 1,
                                        pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, ConstantFoldingConstantOfShapeSmallOutputs) {
  for (const auto& dimensions : {std::vector<int64_t>{}, std::vector<int64_t>{0}, std::vector<int64_t>{2, 3}}) {
    auto build_model = [&dimensions](ModelTestBuilder& builder) {
      auto* shape = builder.Make1DInitializer<int64_t>(dimensions);
      builder.AddNode("ConstantOfShape", {shape}, {builder.MakeOutput()});
    };

    auto post_graph_checker = [&dimensions](Graph& graph) -> Status {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ConstantOfShape"] == 0);
      const auto* output = graph.GetConstantInitializer(graph.GetOutputs()[0]->Name(), false);
      TEST_RETURN_IF_NOT(output != nullptr);
      TEST_RETURN_IF_NOT(std::vector<int64_t>(output->dims().begin(), output->dims().end()) == dimensions);
      return Status::OK();
    };

    CPUExecutionProvider cpu_ep{CPUExecutionProviderInfo()};
    ConfigOptions config_options;
    ASSERT_STATUS_OK(config_options.AddConfigEntry(kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "1024"));
    ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                          std::make_unique<ConstantFolding>(cpu_ep, false, config_options),
                                          TransformerLevel::Level1, 1, nullptr, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingSkipsOverflowingConstantOfShape) {
  auto build_model = [](ModelTestBuilder& builder) {
    auto* shape = builder.Make1DInitializer<int64_t>({std::numeric_limits<int64_t>::max(), 2});
    builder.AddNode("ConstantOfShape", {shape}, {builder.MakeOutput()});
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    ONNX_NAMESPACE::TensorShapeProto stale_shape;
    stale_shape.add_dim()->set_dim_value(1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "ConstantOfShape") {
        node.MutableOutputDefs()[0]->SetShape(stale_shape);
        return Status::OK();
      }
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ConstantOfShape node not found");
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ConstantOfShape"] == 1);
    return Status::OK();
  };

  CPUExecutionProvider cpu_ep{CPUExecutionProviderInfo()};
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "1024"));
  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(cpu_ep, false, config_options),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Verify that ConstantOfShape output size is estimated directly from the input shape
// initializer (and not just from shape inference) so that excessive constant-folded
// allocations are blocked before kernel execution. The 'value' attribute uses int64
// elements (8 bytes) so that 100M * 8 = 800 MB exceeds the configured 256 MB cap.
TEST_F(GraphTransformationTests, ConstantFoldingConstantOfShapeUsesInputInitializerForSizeCheck) {
  constexpr int64_t kLargeDim = 100 * 1024 * 1024;  // 100M elements

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* shape_data = builder.MakeInitializer<int64_t>({1}, {kLargeDim});
    auto* output_arg = builder.MakeOutput();

    auto& node = builder.AddNode("ConstantOfShape", {shape_data}, {output_arg});
    ONNX_NAMESPACE::AttributeProto value_attr;
    value_attr.set_name("value");
    value_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    auto* tensor = value_attr.mutable_t();
    tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    tensor->add_dims(1);
    tensor->add_int64_data(0);
    node.AddAttributeProto(std::move(value_attr));
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    // 800 MB output exceeds the 256 MB cap, so the node must not be folded
    // (and crucially, the 800 MB allocation must not have happened during folding).
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "268435456"));  // 256 MB

  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                        TransformerLevel::Level1, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Companion to the test above that isolates the new input-initializer-based size estimator
// from the pre-existing shape-inference-based path. After Graph::Resolve() has populated
// the ConstantOfShape output shape from the constant shape initializer, we explicitly clear
// the output NodeArg shape in pre_graph_checker. This simulates the documented attack where
// shape inference fails to propagate the output shape, so the generic shape-based estimator
// would return -1 and the only thing that can still block folding is the new
// EstimateConstantOfShapeOutputSizeInBytes path that reads the shape input initializer directly.
TEST_F(GraphTransformationTests, ConstantFoldingConstantOfShapeBlockedWhenOutputShapeMissing) {
  constexpr int64_t kLargeDim = 100 * 1024 * 1024;  // 100M elements

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* shape_data = builder.MakeInitializer<int64_t>({1}, {kLargeDim});
    auto* output_arg = builder.MakeOutput();

    auto& node = builder.AddNode("ConstantOfShape", {shape_data}, {output_arg});
    // int64 'value' so derived size is 100M * 8 = 800 MB, exceeding the 256 MB cap below.
    ONNX_NAMESPACE::AttributeProto value_attr;
    value_attr.set_name("value");
    value_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    auto* tensor = value_attr.mutable_t();
    tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    tensor->add_dims(1);
    tensor->add_int64_data(0);
    node.AddAttributeProto(std::move(value_attr));
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    // Strip the shape that ONNX shape inference propagated onto the ConstantOfShape output.
    // This forces the size-cap check to rely on the new initializer-based estimator.
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "ConstantOfShape") {
        for (auto* output_def : node.MutableOutputDefs()) {
          if (output_def != nullptr && output_def->Exists()) {
            output_def->ClearShape();
          }
        }
      }
    }
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    // The new initializer-based estimator must still derive 800 MB and refuse to fold,
    // even though the output NodeArg has no shape for the generic estimator to use.
    TEST_RETURN_IF_NOT(op_to_count["ConstantOfShape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "268435456"));  // 256 MB

  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                        TransformerLevel::Level1, 1,
                                        pre_graph_checker, post_graph_checker));
}

// The constant-folding output-size estimate must be packing-aware for sub-byte types.
// A QuantizeLinear node that produces a packed int4 output stores 2 elements per byte, so its
// real storage size is ceil(num_elements / 2) bytes. The pre-execution size estimate previously
// used num_elements * 1 byte, over-counting by ~2x and needlessly blocking folding of sub-byte
// outputs that actually fit within the configured limit.
TEST_F(GraphTransformationTests, ConstantFoldingSubByteOutputSizeIsPacked) {
  // 20000 int4 elements -> 10000 packed bytes of real storage (the old estimate was 20000 bytes).
  constexpr int64_t kNumElements = 20000;

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInitializer<float>({kNumElements}, -1.0f, 1.0f);
    auto* output_arg = builder.MakeOutput();
    builder.AddQuantizeLinearNode<Int4x2>(input_data, 1.0f, Int4x2(0, 0), output_arg);
  };

  // Case 1: limit (15000 bytes) sits between the packed size (10000) and the old over-estimate
  // (20000). With the packing-aware estimate the node SHOULD be folded. Under the old estimate
  // it would have been (incorrectly) skipped.
  {
    auto pre_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_to_count["QuantizeLinear"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      // Real packed size (10000 bytes) is within the 15000 byte limit, so folding proceeds.
      TEST_RETURN_IF_NOT(op_to_count["QuantizeLinear"] == 0);
      return Status::OK();
    };

    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    ConfigOptions config_options;
    ASSERT_STATUS_OK(config_options.AddConfigEntry(
        kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "15000"));

    ASSERT_STATUS_OK(TestGraphTransformer(build_model, 21, *logger_,
                                          std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Case 2: limit (5000 bytes) is below even the packed size (10000), so folding is blocked.
  {
    auto pre_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_to_count["QuantizeLinear"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) -> Status {
      auto op_to_count = CountOpsInGraph(graph);
      // Real packed size (10000 bytes) exceeds the 5000 byte limit, so the node is not folded.
      TEST_RETURN_IF_NOT(op_to_count["QuantizeLinear"] == 1);
      return Status::OK();
    };

    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    ConfigOptions config_options;
    ASSERT_STATUS_OK(config_options.AddConfigEntry(
        kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "5000"));

    ASSERT_STATUS_OK(TestGraphTransformer(build_model, 21, *logger_,
                                          std::make_unique<ConstantFolding>(*e.get(), false, config_options),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

// Test that small constant folding still works with the size limit.
TEST_F(GraphTransformationTests, ConstantFoldingSmallOutputAllowed) {
  // Build a model with a small Expand: scalar -> [4, 4] = 16 * 4 = 64 bytes.
  // This is well within even a small limit and should be folded.

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInitializer<float>({1}, {42.0f});
    auto* shape_data = builder.MakeInitializer<int64_t>({2}, {4, 4});
    auto* output_arg = builder.MakeOutput();

    builder.AddNode("Expand", {input_data, shape_data}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    // Small Expand should be constant folded
    TEST_RETURN_IF_NOT(op_to_count["Expand"] == 0);
    return Status::OK();
  };

  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const ConfigOptions empty_config_options;

  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(*e.get(), false, empty_config_options),
                                        TransformerLevel::Level1, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Test that constant folding gracefully handles an Expand node whose output shape
// dimensions would cause integer overflow. This simulates the attack vector where
// a malicious model embeds constant initializers with extreme shape values, causing
// kernel Compute() to execute during graph optimization. The SafeInt-protected
// arithmetic in Expand::Compute (or TensorShape overflow) should be caught by the
// try/catch around Compute, and the node should NOT be constant folded.
TEST_F(GraphTransformationTests, ConstantFoldingExpandOverflowDimsSkipped) {
  constexpr int64_t kLargeDim = int64_t(1) << 32;  // 4294967296

  auto build_model = [&](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInitializer<float>({1}, {1.0f});
    auto* shape_data = builder.MakeInitializer<int64_t>({2}, {kLargeDim, kLargeDim});
    auto* output_arg = builder.MakeOutput();

    builder.AddNode("Expand", {input_data, shape_data}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    // Expand should remain because the overflow prevents constant folding.
    TEST_RETURN_IF_NOT(op_to_count["Expand"] == 1);
    return Status::OK();
  };

  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const ConfigOptions empty_config_options;

  ASSERT_STATUS_OK(TestGraphTransformer(build_model, 14, *logger_,
                                        std::make_unique<ConstantFolding>(*e.get(), false, empty_config_options),
                                        TransformerLevel::Level1, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Check transformations in the case of a subgraph with constant inputs.
TEST_F(GraphTransformationTests, SubgraphWithConstantInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "constant-subgraph.onnx";

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  ASSERT_STATUS_OK(session_object.Initialize());

  NameMLValMap feeds;
  RunOptions run_options;

  std::vector<std::string> output_names = {"output"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
}

TEST_F(GraphTransformationTests, FuseConvBNNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string bn_output_name;

  // add a missing optional output to BN. this should be fusable
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg("", nullptr));
      bn_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), bn_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

static void RunConvTransposeBNFusionTest(bool add_conv_bias, int64_t group, bool add_epsilon = true) {
  const int64_t input_channels_per_group = 2;
  const int64_t input_channels = input_channels_per_group * group;
  const int64_t output_channels_per_group = 3;
  const int64_t output_channels = output_channels_per_group * group;
  const int64_t input_h = 4;
  const int64_t input_w = 4;
  const int64_t kernel_h = 3;
  const int64_t kernel_w = 3;
  const int64_t output_h = input_h + kernel_h - 1;
  const int64_t output_w = input_w + kernel_w - 1;

  auto make_values = [](size_t size, float scale, float offset) {
    std::vector<float> values(size);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = offset + scale * static_cast<float>(static_cast<int64_t>(i % 17) - 8);
    }

    return values;
  };

  const auto weight_data = make_values(static_cast<size_t>(input_channels) *
                                           static_cast<size_t>(output_channels_per_group) *
                                           static_cast<size_t>(kernel_h) *
                                           static_cast<size_t>(kernel_w),
                                       0.02f,
                                       0.01f);
  const auto conv_bias_data = make_values(static_cast<size_t>(output_channels), 0.03f, -0.02f);
  const auto bn_scale_data = make_values(static_cast<size_t>(output_channels), 0.04f, 0.9f);
  const auto bn_bias_data = make_values(static_cast<size_t>(output_channels), 0.05f, -0.1f);
  const auto bn_mean_data = make_values(static_cast<size_t>(output_channels), 0.02f, 0.12f);
  std::vector<float> bn_var_data(static_cast<size_t>(output_channels));
  for (size_t i = 0; i < bn_var_data.size(); ++i) {
    bn_var_data[i] = 0.7f + 0.05f * static_cast<float>(i);
  }

  std::string bn_output_name;
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(std::vector<int64_t>{1, input_channels, input_h, input_w}, -1.0f, 1.0f);
    auto* weight = builder.MakeInitializer<float>({input_channels, output_channels_per_group, kernel_h, kernel_w},
                                                  weight_data);

    std::vector<NodeArg*> conv_inputs{input, weight};
    if (add_conv_bias) {
      conv_inputs.push_back(builder.MakeInitializer<float>({output_channels}, conv_bias_data));
    }

    auto* conv_output =
        builder.MakeIntermediate<float>(std::vector<int64_t>{1, output_channels, output_h, output_w});
    auto& conv_transpose = builder.AddNode("ConvTranspose", conv_inputs, {conv_output});
    if (group != 1) {
      conv_transpose.AddAttribute("group", group);
    }

    auto* bn_scale = builder.MakeInitializer<float>({output_channels}, bn_scale_data);
    auto* bn_bias = builder.MakeInitializer<float>({output_channels}, bn_bias_data);
    auto* bn_mean = builder.MakeInitializer<float>({output_channels}, bn_mean_data);
    auto* bn_var = builder.MakeInitializer<float>({output_channels}, bn_var_data);
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, output_channels, output_h, output_w});
    bn_output_name = output->Name();

    auto& batch_norm = builder.AddNode("BatchNormalization", {conv_output, bn_scale, bn_bias, bn_mean, bn_var}, {output});
    if (add_epsilon) {
      batch_norm.AddAttribute("epsilon", 1e-5f);
    }
  };

  auto check_transformed_graph = [&bn_output_name](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    auto op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["BatchNormalization"], 0);
    ASSERT_EQ(op_to_count["ConvTranspose"], 1);

    bool found_conv_transpose = false;
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "ConvTranspose") {
        found_conv_transpose = true;
        ASSERT_EQ(node.InputDefs().size(), 3U);
        ASSERT_EQ(node.OutputDefs()[0]->Name(), bn_output_name)
            << "fusion should produce the same output name as the last node";
      }
    }
    ASSERT_TRUE(found_conv_transpose);
  };

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    14,
                    1e-4,
                    1e-4,
                    std::move(rule_transformer_L1));
}

TEST_F(GraphTransformationTests, FuseConvTransposeBNNoBias) {
  RunConvTransposeBNFusionTest(false, 1);
}

TEST_F(GraphTransformationTests, FuseConvTransposeBNWithDefaultEpsilon) {
  RunConvTransposeBNFusionTest(false, 1, false);
}

TEST_F(GraphTransformationTests, FuseConvTransposeBNWithBias) {
  RunConvTransposeBNFusionTest(true, 1);
}

TEST_F(GraphTransformationTests, FuseGroupedConvTransposeBN) {
  RunConvTransposeBNFusionTest(true, 2);
}

TEST_F(GraphTransformationTests, DontFuseConvTransposeWithBNWithNonConstantWeight) {
  const int64_t input_channels = 2;
  const int64_t output_channels = 3;
  const int64_t kernel_h = 3;
  const int64_t kernel_w = 3;
  const int64_t output_h = 6;
  const int64_t output_w = 6;

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(std::vector<int64_t>{1, input_channels, 4, 4}, -1.0f, 1.0f);
    auto* weight =
        builder.MakeInput<float>(std::vector<int64_t>{input_channels, output_channels, kernel_h, kernel_w},
                                 -0.2f,
                                 0.2f);

    auto* conv_output =
        builder.MakeIntermediate<float>(std::vector<int64_t>{1, output_channels, output_h, output_w});
    builder.AddNode("ConvTranspose", {input, weight}, {conv_output});

    auto* bn_scale = builder.MakeInitializer<float>({output_channels}, {0.8f, 0.9f, 1.0f});
    auto* bn_bias = builder.MakeInitializer<float>({output_channels}, {-0.1f, 0.0f, 0.1f});
    auto* bn_mean = builder.MakeInitializer<float>({output_channels}, {0.2f, 0.3f, 0.4f});
    auto* bn_var = builder.MakeInitializer<float>({output_channels}, {0.7f, 0.8f, 0.9f});
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, output_channels, output_h, output_w});

    builder.AddNode("BatchNormalization", {conv_output, bn_scale, bn_bias, bn_mean, bn_var}, {output})
        .AddAttribute("epsilon", 1e-5f);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    auto op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["BatchNormalization"], 1);
    ASSERT_EQ(op_to_count["ConvTranspose"], 1);
  };

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    14,
                    0.0,
                    0.0,
                    std::move(rule_transformer_L1));
}

TEST_F(GraphTransformationTests, FusePadWithConv) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-conv.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::vector<int64_t> expected_pads;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "Pad") {
      const auto* pads_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      Initializer pads{graph, *pads_proto, graph.ModelPath()};
      gsl::span<const int64_t> pads_values = pads.DataAsSpan<int64_t>();
      expected_pads.resize(pads_values.size() - 4);

      for (uint32_t pads_index = 2, index = 0; pads_index < pads_values.size() / 2; pads_index++, index++) {
        expected_pads[index] = pads_values[pads_index];
        expected_pads[index + (expected_pads.size() / 2)] = pads_values[pads_index + (pads_values.size() / 2)];
      }
    } else if (node.OpType() == "Conv") {
      auto child_pads = node.GetMutableAttributes()["pads"].mutable_ints();
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        expected_pads[index] += child_pads->Get(index);
      }
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Pad"], 0);
  ASSERT_EQ(op_to_count["Conv"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      auto child_pads = node.GetMutableAttributes()["pads"].mutable_ints();
      ASSERT_EQ(child_pads->size(), static_cast<int32_t>(expected_pads.size()))
          << "fusion should produce the same size of pads integer as the Conv node";
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        ASSERT_EQ(expected_pads[index], child_pads->Get(index))
            << "fusion does not produce correct padding value";
      }
    }
  }
}

TEST_F(GraphTransformationTests, FusePadWithNoPadsConv) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-nopadsconv.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::vector<int64_t> expected_pads;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "Pad") {
      const auto* pads_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      Initializer pads{graph, *pads_proto, graph.ModelPath()};
      gsl::span<const int64_t> pads_values = pads.DataAsSpan<int64_t>();
      expected_pads.resize(pads_values.size() - 4);

      for (uint32_t pads_index = 2, index = 0; pads_index < pads_values.size() / 2; pads_index++, index++) {
        expected_pads[index] = pads_values[pads_index];
        expected_pads[index + (expected_pads.size() / 2)] = pads_values[pads_index + (pads_values.size() / 2)];
      }
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Pad"], 0);
  ASSERT_EQ(op_to_count["Conv"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      auto child_pads = node.GetMutableAttributes()["pads"].mutable_ints();
      ASSERT_EQ(child_pads->size(), static_cast<int32_t>(expected_pads.size()))
          << "fusion should produce the same size of pads integer as the Conv node";
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        ASSERT_EQ(expected_pads[index], child_pads->Get(index))
            << "fusion does not produce correct padding value";
      }
    }
  }
}

struct PadMaxPoolFusionTestConfig {
  std::vector<int64_t> pad_pads{0, 0, 1, 0, 0, 0};
  std::vector<int64_t> max_pool_pads{0, 1};
  std::vector<int64_t> strides{};
  int64_t ceil_mode{0};
  bool use_explicit_pad_value{true};
};

template <typename PadType, typename PoolType = PadType>
void RunPadMaxPoolFusionTest(const std::vector<PadType>& input_values, bool expect_fusion,
                             PadMaxPoolFusionTestConfig config = {}) {
  constexpr bool add_cast = !std::is_same_v<PadType, PoolType>;

  auto build_test_case = [&input_values, &config](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<PadType>({1, 1, 2}, input_values);
    auto* pads = builder.Make1DInitializer<int64_t>(config.pad_pads);
    auto* pad_output = builder.MakeIntermediate<PadType>(std::nullopt);
    if (config.use_explicit_pad_value) {
      auto* pad_value = builder.MakeScalarInitializer<PadType>(PadType{0});
      builder.AddNode("Pad", {input, pads, pad_value}, {pad_output});
    } else {
      builder.AddNode("Pad", {input, pads}, {pad_output});
    }

    NodeArg* pool_input = pad_output;
    if constexpr (add_cast) {
      auto* cast_output = builder.MakeIntermediate<PoolType>(std::nullopt);
      builder.AddNode("Cast", {pad_output}, {cast_output})
          .AddAttribute("to", static_cast<int64_t>(utils::ToTensorProtoElementType<PoolType>()));
      pool_input = cast_output;
    }

    auto* output = builder.MakeOutput<PoolType>(std::nullopt);
    auto& max_pool = builder.AddNode("MaxPool", {pool_input}, {output});
    max_pool.AddAttribute("kernel_shape", std::vector<int64_t>{2});
    max_pool.AddAttribute("pads", config.max_pool_pads);
    if (!config.strides.empty()) {
      max_pool.AddAttribute("strides", config.strides);
    }
    if (config.ceil_mode != 0) {
      max_pool.AddAttribute("ceil_mode", config.ceil_mode);
    }
  };

  auto check_transformed_graph = [expect_fusion, &config](InferenceSessionWrapper& session) {
    const auto op_to_count = CountOpsInGraph(session.GetGraph());
    const auto pad_count = op_to_count.find("Pad");
    ASSERT_EQ(op_to_count.at("MaxPool"), 1);
    ASSERT_EQ(pad_count == op_to_count.end() ? 0 : pad_count->second, expect_fusion ? 0 : 1);
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == "MaxPool") {
        const auto& pads = node.GetAttributes().at("pads").ints();
        ASSERT_EQ(pads.size(), 2);
        ASSERT_EQ(pads[0], config.max_pool_pads[0] + (expect_fusion ? config.pad_pads[2] : 0));
        ASSERT_EQ(pads[1], config.max_pool_pads[1] + (expect_fusion ? config.pad_pads[5] : 0));
      }
    }
    if constexpr (add_cast) {
      ASSERT_EQ(op_to_count.at("Cast"), 1);
      for (const auto& node : session.GetGraph().Nodes()) {
        if (node.OpType() == "Cast") {
          const auto* cast_output_shape = node.OutputDefs()[0]->Shape();
          ASSERT_NE(cast_output_shape, nullptr);
          ASSERT_EQ(cast_output_shape->dim_size(), 3);
          ASSERT_EQ(cast_output_shape->dim(0).dim_value(), 1);
          ASSERT_EQ(cast_output_shape->dim(1).dim_value(), 1);
          const int64_t expected_width = expect_fusion ? 2 : 2 + config.pad_pads[2] + config.pad_pads[5];
          ASSERT_EQ(cast_output_shape->dim(2).dim_value(), expected_width);
        }
      }
    }
  };

  auto rule_transformer_l1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_l1->Register(std::make_unique<PadFusion>()));
  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    13,
                    0.0,
                    0.0,
                    std::move(rule_transformer_l1));
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsFloat) {
  PadMaxPoolFusionTestConfig config;
  config.use_explicit_pad_value = false;
  RunPadMaxPoolFusionTest<float>({-1.0f, -2.0f}, false, config);
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsInt8) {
  RunPadMaxPoolFusionTest<int8_t>({-1, -2}, false);
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionPreservesUint8) {
  RunPadMaxPoolFusionTest<uint8_t>({1, 2}, true);
}

TEST_F(GraphTransformationTests, PadCastMaxPoolFusionSkipsDouble) {
  RunPadMaxPoolFusionTest<float, double>({-1.0f, -2.0f}, false);
}

TEST_F(GraphTransformationTests, PadCastMaxPoolFusionPreservesUint8) {
  RunPadMaxPoolFusionTest<float, uint8_t>({1.0f, 2.0f}, true);
}

TEST_F(GraphTransformationTests, PadCastMaxPoolFusionHandlesMissingInputShape) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(std::nullopt);
    auto* pads = builder.Make1DInitializer<int64_t>({0, 0, 1, 0, 0, 0});
    auto* pad_output = builder.MakeIntermediate<float>(std::nullopt);
    builder.AddNode("Pad", {input, pads}, {pad_output});

    auto* cast_output = builder.MakeIntermediate<uint8_t>(std::nullopt);
    builder.AddNode("Cast", {pad_output}, {cast_output})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_UINT8));

    auto* output = builder.MakeOutput<uint8_t>(std::nullopt);
    auto& max_pool = builder.AddNode("MaxPool", {cast_output}, {output});
    max_pool.AddAttribute("kernel_shape", std::vector<int64_t>{2});
    max_pool.AddAttribute("pads", std::vector<int64_t>{0, 0});
  };

  auto post_graph_checker = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    ORT_RETURN_IF_NOT(op_to_count.find("Pad") == op_to_count.end(), "Pad should be fused away");
    ORT_RETURN_IF_NOT(op_to_count.at("Cast") == 1, "Cast should remain");
    ORT_RETURN_IF_NOT(op_to_count.at("MaxPool") == 1, "MaxPool should remain");

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Cast") {
        ORT_RETURN_IF_NOT(node.OutputDefs()[0]->Shape() == nullptr,
                          "Cast output shape should remain unknown after fusion");
      }
    }

    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case,
                                        13,
                                        *logger_,
                                        std::move(rule_transformer),
                                        TransformerLevel::Level1,
                                        1,
                                        nullptr,
                                        post_graph_checker));
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsCeilModeWithTrailingPad) {
  PadMaxPoolFusionTestConfig config;
  config.pad_pads = {0, 0, 0, 0, 0, 1};
  config.max_pool_pads = {0, 0};
  config.strides = {2};
  config.ceil_mode = 1;
  RunPadMaxPoolFusionTest<uint8_t>({1, 2}, false, config);
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsFusedPadNotSmallerThanKernel) {
  PadMaxPoolFusionTestConfig config;
  config.max_pool_pads = {1, 0};
  RunPadMaxPoolFusionTest<uint8_t>({1, 2}, false, config);
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsFloatModel) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-maxpool.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_l1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_l1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_l1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  const auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count.at("Pad"), 1);
  ASSERT_EQ(op_to_count.at("MaxPool"), 1);
}

TEST_F(GraphTransformationTests, PadMaxPoolFusionSkipsFloatModelOpsetLessThan11) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-maxpool-opset8.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_l1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_l1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_l1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  const auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count.at("Pad"), 1);
  ASSERT_EQ(op_to_count.at("MaxPool"), 1);
}

TEST_F(GraphTransformationTests, PadAveragePoolFusionPreservesCeilModeDivisor) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 1, 2}, {2.0f, 4.0f});
    auto* pads = builder.Make1DInitializer<int64_t>({0, 0, 1, 0, 0, 0});
    auto* pad_output = builder.MakeIntermediate<float>(std::nullopt);
    builder.AddNode("Pad", {input, pads}, {pad_output});

    auto* output = builder.MakeOutput<float>(std::nullopt);
    auto& average_pool = builder.AddNode("AveragePool", {pad_output}, {output});
    average_pool.AddAttribute("kernel_shape", std::vector<int64_t>{2});
    average_pool.AddAttribute("strides", std::vector<int64_t>{2});
    average_pool.AddAttribute("pads", std::vector<int64_t>{0, 0});
    average_pool.AddAttribute("ceil_mode", static_cast<int64_t>(1));
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_to_count = CountOpsInGraph(session.GetGraph());
    const auto pad_count = op_to_count.find("Pad");
    ASSERT_EQ(pad_count == op_to_count.end() ? 0 : pad_count->second, 0);
    ASSERT_EQ(op_to_count.at("AveragePool"), 1);

    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == "AveragePool") {
        const auto& attributes = node.GetAttributes();
        const auto& pads = attributes.at("pads").ints();
        ASSERT_EQ(pads.size(), 2);
        ASSERT_EQ(pads[0], 1);
        ASSERT_EQ(pads[1], 0);
        ASSERT_EQ(attributes.at("count_include_pad").i(), 1);
      }
    }
  };

  auto rule_transformer_l1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_l1->Register(std::make_unique<PadFusion>()));

  // Before fusion the explicit zero is a real input value, while the extra cell created by
  // ceil_mode is outside the padded tensor. The expected output is [1, 4]: only the former is
  // included in an average. TransformerTester verifies that the fused graph preserves this
  // distinction when it changes the explicit Pad to count_include_pad=1.
  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    13,
                    0.0,
                    0.0,
                    std::move(rule_transformer_l1));
}

TEST_F(GraphTransformationTests, FusePadWithAvgPool) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-avgpool.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::vector<int64_t> expected_pads;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "Pad") {
      auto const& pads_proto = node.GetAttributes().at("pads").ints();
      gsl::span<const int64_t> pads_values = gsl::make_span(pads_proto.data(), pads_proto.size());
      expected_pads.resize(pads_values.size() - 4);
      for (uint32_t pads_index = 2, index = 0; pads_index < pads_values.size() / 2; pads_index++, index++) {
        expected_pads[index] = pads_values[pads_index];
        expected_pads[index + (expected_pads.size() / 2)] = pads_values[pads_index + (pads_values.size() / 2)];
      }
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Pad"], 0);
  ASSERT_EQ(op_to_count["AveragePool"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "AveragePool") {
      auto const& child_pads = node.GetAttributes().at("pads").ints();
      auto const& count_include_pad = node.GetAttributes().at("count_include_pad");
      ASSERT_NE(count_include_pad.i(), 0) << "fusion should ensure count_include_pad!=0";
      ASSERT_EQ(child_pads.size(), static_cast<int32_t>(expected_pads.size()))
          << "fusion should produce the same size of pads integer as the AvgPool node";
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        ASSERT_EQ(expected_pads[index], child_pads.Get(index))
            << "fusion does not produce correct padding value";
      }
    }
  }
}

TEST_F(GraphTransformationTests, FusePadWithAvgPoolWithPad) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-avgpool_with_pad.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::vector<int64_t> expected_pads;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "Pad") {
      auto const& pads_proto = node.GetAttributes().at("pads").ints();
      gsl::span<const int64_t> pads_values = gsl::make_span(pads_proto.data(), pads_proto.size());
      expected_pads.resize(pads_values.size() - 4);

      for (uint32_t pads_index = 2, index = 0; pads_index < pads_values.size() / 2; pads_index++, index++) {
        expected_pads[index] = pads_values[pads_index];
        expected_pads[index + (expected_pads.size() / 2)] = pads_values[pads_index + (pads_values.size() / 2)];
      }
    } else if (node.OpType() == "AveragePool") {
      auto const& child_pads = node.GetAttributes().at("pads").ints();
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        expected_pads[index] += child_pads.Get(index);
      }
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Pad"], 0);
  ASSERT_EQ(op_to_count["AveragePool"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "AveragePool") {
      auto const& child_pads = node.GetAttributes().at("pads").ints();
      auto const& count_include_pad = node.GetAttributes().at("count_include_pad");
      ASSERT_NE(count_include_pad.i(), 0) << "fusion should ensure count_include_pad!=0";
      ASSERT_EQ(child_pads.size(), static_cast<int32_t>(expected_pads.size()))
          << "fusion should produce the same size of pads integer as the AvgPool node";
      for (uint32_t index = 0; index < expected_pads.size(); index++) {
        ASSERT_EQ(expected_pads[index], child_pads.Get(index))
            << "fusion does not produce correct padding value";
      }
    }
  }
}

// should not fuse
TEST_F(GraphTransformationTests, FusePadWithAvgPoolWithPadNoInclude) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-pad-avgpool_with_pad-nofuse.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Pad"], 1);
  ASSERT_EQ(op_to_count["AveragePool"], 1);
}

// Verify PadFusion does not fuse and does not crash when the Pad node's `pads` initializer carries
// fewer than four elements (which violates the 2*rank requirement of the ONNX Pad spec). The data
// input uses an unspecified rank so ONNX shape inference cannot pre-validate the pads length.
TEST_F(GraphTransformationTests, PadFusionRejectsShortPadsInitializer) {
  auto run_case = [&](const std::vector<int64_t>& pads_data) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>(std::nullopt);
      auto* weight_arg = builder.MakeInitializer<float>({1, 1, 1, 1}, {1.0f});
      auto* pads_arg = builder.MakeInitializer<int64_t>(
          {static_cast<int64_t>(pads_data.size())}, pads_data);
      auto* pad_out = builder.MakeIntermediate();
      auto* conv_out = builder.MakeOutput();

      builder.AddNode("Pad", {data_arg, pads_arg}, {pad_out});
      builder.AddNode("Conv", {pad_out, weight_arg}, {conv_out});
    };

    auto pre_graph_checker = [](Graph&) { return Status::OK(); };
    auto post_graph_checker = [](Graph& graph) {
      // Fusion must not have run; both nodes remain.
      std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
      ORT_RETURN_IF_NOT(op_to_count["Pad"] == 1, "Pad should not be fused away");
      ORT_RETURN_IF_NOT(op_to_count["Conv"] == 1, "Conv should remain");
      return Status::OK();
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PadFusion>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(rule_transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  };

  // Cover empty, 1, 2, and 3 element pads vectors as well as an odd-length vector.
  run_case({});
  run_case({0});
  run_case({0, 0});
  run_case({0, 0, 0});
  run_case({0, 0, 0, 0, 0});
}

// Verify PadFusion bails out when the Pad node's pads contain a negative value.
TEST_F(GraphTransformationTests, PadFusionRejectsNegativePads) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>(std::nullopt);
    auto* weight_arg = builder.MakeInitializer<float>({1, 1, 1, 1}, {1.0f});
    auto* pads_arg = builder.MakeInitializer<int64_t>({8}, {0, 0, 1, 1, 0, 0, -1, 1});
    auto* pad_out = builder.MakeIntermediate();
    auto* conv_out = builder.MakeOutput();

    builder.AddNode("Pad", {data_arg, pads_arg}, {pad_out});
    builder.AddNode("Conv", {pad_out, weight_arg}, {conv_out});
  };

  auto pre_graph_checker = [](Graph&) { return Status::OK(); };
  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ORT_RETURN_IF_NOT(op_to_count["Pad"] == 1, "Pad with negative pads should not be fused");
    ORT_RETURN_IF_NOT(op_to_count["Conv"] == 1, "Conv should remain");
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Verify PadFusion bails out when the child Conv already has an explicit `pads` attribute whose
// length does not match the spatial rank implied by the Pad node.
TEST_F(GraphTransformationTests, PadFusionRejectsMismatchedChildPadsRank) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    // Pad pads length 8 implies spatial rank 2 (expected child pads length 4),
    // but Conv carries a `pads` attribute of length 6, simulating a rank mismatch
    // that could arise from a malformed model. Data uses unspecified rank so shape
    // inference does not reject the model before the transformer runs.
    auto* data_arg = builder.MakeInput<float>(std::nullopt);
    auto* weight_arg = builder.MakeInitializer<float>({1, 1, 1, 1}, {1.0f});
    auto* pads_arg = builder.MakeInitializer<int64_t>({8}, {0, 0, 1, 1, 0, 0, 1, 1});
    auto* pad_out = builder.MakeIntermediate();
    auto* conv_out = builder.MakeOutput();

    builder.AddNode("Pad", {data_arg, pads_arg}, {pad_out});
    auto& conv = builder.AddNode("Conv", {pad_out, weight_arg}, {conv_out});
    conv.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0});
  };

  auto pre_graph_checker = [](Graph&) { return Status::OK(); };
  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ORT_RETURN_IF_NOT(op_to_count["Pad"] == 1, "Pad should not be fused into a rank-mismatched Conv");
    ORT_RETURN_IF_NOT(op_to_count["Conv"] == 1, "Conv should remain");
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PadFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Verify PadFusion does not crash for the opset-10 attribute form of Pad when the `pads` attribute
// has fewer than four elements.
TEST_F(GraphTransformationTests, PadFusionRejectsShortPadsAttributeOpset10) {
  auto run_case = [&](const std::vector<int64_t>& pads_data) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>(std::nullopt);
      auto* weight_arg = builder.MakeInitializer<float>({1, 1, 1, 1}, {1.0f});
      auto* pad_out = builder.MakeIntermediate();
      auto* conv_out = builder.MakeOutput();

      auto& pad_node = builder.AddNode("Pad", {data_arg}, {pad_out});
      pad_node.AddAttribute("pads", pads_data);
      pad_node.AddAttribute("mode", "constant");
      builder.AddNode("Conv", {pad_out, weight_arg}, {conv_out});
    };

    auto pre_graph_checker = [](Graph&) { return Status::OK(); };
    auto post_graph_checker = [](Graph& graph) {
      std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
      ORT_RETURN_IF_NOT(op_to_count["Pad"] == 1, "Pad should not be fused away");
      ORT_RETURN_IF_NOT(op_to_count["Conv"] == 1, "Conv should remain");
      return Status::OK();
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PadFusion>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 10, *logger_, std::move(rule_transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  };

  run_case({});
  run_case({0, 0});
  run_case({0, 0, 0});
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithEmptyOptionalOutputWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    } else if (node.OpType() == "BatchNormalization") {
      node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg("", nullptr));
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

// should not fuse
TEST_F(GraphTransformationTests, FuseMatmulBNWithOptionalOutputWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      // additional non-empty output to batchNormalization
      ONNX_NAMESPACE::TypeProto optional_output_tensor_type;
      optional_output_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TypeProto::kTensorType);
      auto& arg = graph.GetOrCreateNodeArg("bn_optional_output", &optional_output_tensor_type);
      node.MutableOutputDefs().push_back(&arg);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

TEST_F(GraphTransformationTests, FuseMatmulBNDirectly) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-directly.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

TEST_F(GraphTransformationTests, DoNotApplyFuseMatmulBNDirectly) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-directly-dont-fuse.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

// should not fuse: the Reshape to [1, 1, 1, 5] moves the MatMul columns off the channel axis
TEST_F(GraphTransformationTests, FuseMatmulBNWithOnlyReshape) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

// should not fuse: the Transpose moves the MatMul columns off the channel axis
TEST_F(GraphTransformationTests, FuseMatmulBNWithOnlyTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-transpose.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithoutBatchNormalization) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-transpose.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      graph_utils::RemoveNode(graph, node);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["MatMul"], 1);
}

// should not fuse
TEST_F(GraphTransformationTests, FuseMatmulBNWithNonIgnorableNode) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-non-ignorable-node.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

// The fusion is only valid if the Reshape keeps the MatMul columns as the BatchNormalization channels.
TEST_F(GraphTransformationTests, FuseMatmulBNWithChannelChangingReshape) {
  for (const auto& [reshape_shape, should_fuse] : std::vector<std::pair<std::vector<int64_t>, bool>>{
           {{2, 4, 1, 1}, true},
           {{1, 4, 2, 1}, false}}) {
    auto build_test_case = [&reshape_shape](ModelTestBuilder& builder) {
      auto* input = builder.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* matmul_b = builder.MakeInitializer<float>({3, 4}, -1.0f, 1.0f);
      auto* shape = builder.MakeInitializer<int64_t>({4}, reshape_shape);
      auto* bn_scale = builder.MakeInitializer<float>({4}, 0.5f, 1.5f);
      auto* bn_bias = builder.MakeInitializer<float>({4}, -1.0f, 1.0f);
      auto* bn_mean = builder.MakeInitializer<float>({4}, -1.0f, 1.0f);
      auto* bn_var = builder.MakeInitializer<float>({4}, 0.5f, 1.5f);
      auto* matmul_output = builder.MakeIntermediate();
      auto* reshape_output = builder.MakeIntermediate();
      auto* output = builder.MakeOutput();

      builder.AddNode("MatMul", {input, matmul_b}, {matmul_output});
      builder.AddNode("Reshape", {matmul_output, shape}, {reshape_output});
      builder.AddNode("BatchNormalization", {reshape_output, bn_scale, bn_bias, bn_mean, bn_var}, {output})
          .AddAttribute("epsilon", 1e-5f);
    };

    auto check_transformed_graph = [should_fuse](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      ASSERT_EQ(op_to_count["BatchNormalization"], should_fuse ? 0 : 1);
      ASSERT_EQ(op_to_count["Gemm"], should_fuse ? 1 : 0);
    };

    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
    TransformerTester(build_test_case,
                      check_transformed_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      15,
                      1e-4,
                      1e-4,
                      std::move(rule_transformer_L1));
  }
}

TEST_F(GraphTransformationTests, DontFuseConvWithBNWithOptionalOutputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  // add an optional output to the BN node. should not fuse if this is present
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      auto mean_input = node.InputDefs()[3];
      auto& mean_output = graph.GetOrCreateNodeArg(mean_input->Name() + ".output", mean_input->TypeAsProto());
      node.MutableOutputDefs().push_back(&mean_output);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 1);
}

TEST_F(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  std::vector<std::basic_string<ORTCHAR_T>> test_models = {ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.negative_axes.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx")};
  for (const auto& model : test_models) {
    PathString model_uri = PathString(MODEL_FOLDER) + model;

    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
    ASSERT_TRUE(op_to_count["Mul"] == 0);
    ASSERT_TRUE(op_to_count["Add"] == 0);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  }
}

TEST_F(GraphTransformationTests, DivMulFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/div_mul.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 5);
  ASSERT_TRUE(op_to_count["Mul"] == 5);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 5);
  ASSERT_TRUE(op_to_count["Mul"] == 2);
}

TEST_F(GraphTransformationTests, LabelEncoderFusion) {
  using common::INVALID_GRAPH;
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/label_encoder.onnx";

  NameMLValMap feeds;

  constexpr size_t ALPH = 26;
  OrtValue mlvalue_a;
  std::vector<int64_t> dims_a = {ALPH};
  std::vector<std::string> values_a = {};
  for (char letter = 'a'; letter <= 'z'; letter++) {
    values_a.emplace_back(1, letter);
  }
  CreateMLValue<std::string>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_a,
                             values_a, &mlvalue_a);
  feeds.insert(std::make_pair("A", mlvalue_a));

  bool is_implemented = true;

  auto run_model_test = [&](TransformerLevel level, std::vector<OrtValue>& fetches, const int requiredLabelEncoderCount) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "OptimizerTests";
    InferenceSessionWrapper session{session_options, GetEnvironment()};

    // If we did not initialize the session correctly, the operator is missing.
    if (!session.Load(model_uri).IsOK() || !session.Initialize().IsOK()) {
      is_implemented = false;
      return;
    }

    // Count if the number of LabelEncoders is as expected
    std::map<std::string, int> op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_TRUE(op_to_count["ai.onnx.ml.LabelEncoder"] == requiredLabelEncoderCount);

    std::vector<std::string> output_names = {};
    for (const auto& output : session.GetGraph().GetOutputs()) {
      output_names.push_back(output->Name());
    }

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));
  };

  // run model with and w/o optimizations and compare the results
  std::vector<OrtValue> unoptimized_fetches;
  run_model_test(TransformerLevel::Default, unoptimized_fetches, 11);

  std::vector<OrtValue> optimized_fetches;
  run_model_test(TransformerLevel::MaxLevel, optimized_fetches, 7);

  // If there was a problem loading the model, do not compare the 2 results
  if (!is_implemented) {
    GTEST_SKIP();
    return;
  }

  // Compare results
  auto ret = CompareOrtValue(optimized_fetches[0], unoptimized_fetches[0], 0.0, 0.0, false);
  EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
}

TEST_F(GraphTransformationTests, LabelEncoderFusionIgnoresMismatchedEmptyAttribute) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/label_encoder.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  Node* target = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() != "LabelEncoder" || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto& next_node = *node.OutputNodesBegin();
    const auto& node_attributes = node.GetAttributes();
    const auto& next_attributes = next_node.GetAttributes();
    if (next_node.OpType() == "LabelEncoder" &&
        node_attributes.find("values_strings") != node_attributes.end() &&
        next_attributes.find("keys_strings") != next_attributes.end() &&
        next_attributes.find("values_int64s") != next_attributes.end() &&
        next_attributes.find("values_strings") == next_attributes.end()) {
      target = graph.GetNode(next_node.Index());
      break;
    }
  }

  ASSERT_NE(target, nullptr);
  target->AddAttribute("values_strings", std::vector<std::string>{});

  GraphTransformerManager transformer_manager{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("LabelEncoderFusionTest");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<LabelEncoderFusion>()));
  ASSERT_STATUS_OK(transformer_manager.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(transformer_manager.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  const auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count.at("ai.onnx.ml.LabelEncoder"), 7);
}

TEST_F(GraphTransformationTests, NotWhereFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/not_where.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Not"] == 4);
  ASSERT_TRUE(op_to_count["Where"] == 5);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<NotWhereFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Where"] == 5);
  ASSERT_TRUE(op_to_count["Not"] == 1);  // can't remove Not if it is graph output/ has consumer that's not where
}

#if !defined(DISABLE_CONTRIB_OPS)
// Conv->Add->Relu will be left intact since there is Identity depend on Add
TEST_F(GraphTransformationTests, FuseCudaConvAddReluIdentity) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_identity.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
#if defined(USE_JSEP)
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kJsExecutionProvider);
  }
#elif defined(USE_WEBGPU)
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kWebGpuExecutionProvider);
  }
#else
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
#endif
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);       // Add remains
  ASSERT_TRUE(op_to_count["Relu"] == 1);      // Relu remains
  ASSERT_TRUE(op_to_count["Identity"] == 1);  // Identity remains
}

// Conv->Add will be left intact since there is no Relu follows
TEST_F(GraphTransformationTests, FuseCudaConvAdd) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
#if defined(USE_JSEP)
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kJsExecutionProvider);
  }
#elif defined(USE_WEBGPU)
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kWebGpuExecutionProvider);
  }
#else
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
#endif
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);  // Add remains, no transform applied to the graph
}

#endif

#if !defined(DISABLE_CONTRIB_OPS)
// Conv->Add->Relu will be transformed to FusedConv
TEST_F(GraphTransformationTests, FuseCpuConvAddRelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);   // Add removed from graph
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Relu removed from graph
}

// Conv->Add->Relu will be partly fused  to Conv_Add->Relu since there is Identity depend on Add
TEST_F(GraphTransformationTests, FuseCpuConvAddReluIdentity) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_identity.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);       // Add removed
  ASSERT_TRUE(op_to_count["Relu"] == 1);      // Relu remains
  ASSERT_TRUE(op_to_count["Identity"] == 1);  // Identity remains
}

// Conv->Add will be transformed to FusedConv
TEST_F(GraphTransformationTests, FuseCpuConvAdd) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);  // Add removed
}

#endif

#if !defined(DISABLE_CONTRIB_OPS)
TEST_F(GraphTransformationTests, FuseConvActivation) {
  std::unordered_map<PathString, std::string> model_to_op_name{{ORT_TSTR("fusion/conv_relu.onnx"), "Relu"},
                                                               {ORT_TSTR("fusion/conv_relu_opset12.onnx"), "Relu"},
                                                               {ORT_TSTR("fusion/conv_clip.onnx"), "Clip"},
                                                               {ORT_TSTR("fusion/conv_sigmoid.onnx"), "Sigmoid"},
                                                               {ORT_TSTR("fusion/conv_tanh.onnx"), "Tanh"},
                                                               {ORT_TSTR("fusion/conv_leakyrelu.onnx"), "LeakyRelu"},
                                                               {ORT_TSTR("fusion/conv_hardsigmoid.onnx"), "HardSigmoid"}};
  for (const auto& model : model_to_op_name) {
    PathString model_uri = PathString(MODEL_FOLDER) + model.first;
    SCOPED_TRACE(ORT_TSTR("model file: ") + model_uri);
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
#if defined(USE_JSEP)
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kJsExecutionProvider);
    }
#elif defined(USE_WEBGPU)
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kWebGpuExecutionProvider);
    }
#else
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kCpuExecutionProvider);
    }
#endif
    std::map<std::string, int> op_to_count_before_fusion = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count_before_fusion[model.second] >= 1);

    // Apply transformer
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

    std::map<std::string, int> op_to_count_after_fusion = CountOpsInGraph(graph);
#if defined(USE_JSEP)
    std::set<std::string> js_supported = {"Relu", "Clip", "Sigmoid", "Tanh", "LeakyRelu"};
    if (js_supported.find(model.second) == js_supported.end()) {
      ASSERT_EQ(op_to_count_before_fusion[model.second], op_to_count_after_fusion[model.second]);
    } else {
      ASSERT_TRUE(op_to_count_after_fusion[model.second] == 0);
    }
#elif defined(USE_WEBGPU)
    std::set<std::string> webgpu_supported = {"Relu", "Clip", "Sigmoid", "Tanh", "LeakyRelu", "HardSigmoid"};
    if (webgpu_supported.find(model.second) == webgpu_supported.end()) {
      ASSERT_EQ(op_to_count_before_fusion[model.second], op_to_count_after_fusion[model.second]);
    } else {
      ASSERT_TRUE(op_to_count_after_fusion[model.second] == 0);
    }
#else
    ASSERT_TRUE(op_to_count_after_fusion[model.second] == 0);
#endif
  }
}

#if defined(USE_WEBGPU)
namespace {
void RunWebGpuSmallMatMulConvActivationPairTest(const std::string& activation_a,
                                                const std::string& activation_b) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto build_test_case = [&](ModelTestBuilder& builder) {
    // Keep N and K below 8 to select MatMulNaiveProgram.
    auto* input = builder.MakeInput<float>({1, 4, 8, 8}, -3.0f, 3.0f);

    // Use distinct weights to prevent common subexpression elimination.
    auto* weight_a = builder.MakeInitializer<float>({6, 4, 1, 1}, -0.5f, 0.5f);
    auto* bias_a = builder.MakeInitializer<float>({6}, -0.5f, 0.5f);
    auto* weight_b = builder.MakeInitializer<float>({6, 4, 1, 1}, -0.5f, 0.5f);
    auto* bias_b = builder.MakeInitializer<float>({6}, -0.5f, 0.5f);

    auto* conv_a_out = builder.MakeIntermediate();
    auto* conv_b_out = builder.MakeIntermediate();
    auto* output_a = builder.MakeOutput();
    auto* output_b = builder.MakeOutput();

    builder.AddNode("Conv", {input, weight_a, bias_a}, {conv_a_out});
    builder.AddNode(activation_a, {conv_a_out}, {output_a});
    builder.AddNode("Conv", {input, weight_b, bias_b}, {conv_b_out});
    builder.AddNode(activation_b, {conv_b_out}, {output_b});
  };

  auto check_transformed_graph = [&](InferenceSessionWrapper& session) {
    std::set<std::string> fused_activations;
    for (const auto& node : session.GetGraph().Nodes()) {
      const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
      if (activation_attr != nullptr) {
        fused_activations.insert(activation_attr->s());
      }
    }
    // Ensure both activations are fused into Conv; otherwise this test would not exercise
    // the activation-specific MatMulNaiveProgram path.
    ASSERT_THAT(fused_activations, ::testing::UnorderedElementsAre(activation_a, activation_b))
        << "expected both convolutions to absorb their activation, so that two MatMulNaive "
           "variants are compiled in one session";
  };

  RunWebGpuFusionTransformerTest(build_test_case,
                                 check_transformed_graph,
                                 TransformerLevel::Level1,
                                 TransformerLevel::Level2,
                                 /*opset_version=*/17,
                                 /*per_sample_tolerance=*/2e-3,
                                 /*relative_per_sample_tolerance=*/2e-3,
                                 /*transformer=*/nullptr,
                                 []() { return DefaultWebGpuExecutionProvider(); });
}

void RunWebGpuSmallMatMulConvHardSigmoidParamPairTest(float alpha_first, float alpha_second) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  // Choose beta so the alpha values clamp to different outputs when Conv produces 8.
  constexpr float kBeta = -8000160.0f;

  auto build_test_case = [&](ModelTestBuilder& builder) {
    // Keep N and K below 8 to select MatMulNaiveProgram; N = 5 keeps its bias indexing in bounds.
    auto* input = builder.MakeInput<float>({1, 4, 8, 8}, std::vector<float>(1 * 4 * 8 * 8, 1.0f));

    // Produce 8 with distinct weights to prevent common subexpression elimination.
    auto* weight_a = builder.MakeInitializer<float>({5, 4, 1, 1}, std::vector<float>(5 * 4, 1.0f));
    auto* bias_a = builder.MakeInitializer<float>({5}, std::vector<float>(5, 4.0f));
    auto* weight_b = builder.MakeInitializer<float>({5, 4, 1, 1}, std::vector<float>(5 * 4, 0.5f));
    auto* bias_b = builder.MakeInitializer<float>({5}, std::vector<float>(5, 6.0f));

    auto* conv_a_out = builder.MakeIntermediate();
    auto* conv_b_out = builder.MakeIntermediate();
    auto* output_a = builder.MakeOutput();
    auto* output_b = builder.MakeOutput();

    builder.AddNode("Conv", {input, weight_a, bias_a}, {conv_a_out});
    Node& activation_a = builder.AddNode("HardSigmoid", {conv_a_out}, {output_a});
    activation_a.AddAttribute("alpha", alpha_first);
    activation_a.AddAttribute("beta", kBeta);

    builder.AddNode("Conv", {input, weight_b, bias_b}, {conv_b_out});
    Node& activation_b = builder.AddNode("HardSigmoid", {conv_b_out}, {output_b});
    activation_b.AddAttribute("alpha", alpha_second);
    activation_b.AddAttribute("beta", kBeta);
  };

  auto check_transformed_graph = [&](InferenceSessionWrapper& session) {
    std::vector<float> fused_alphas;
    for (const auto& node : session.GetGraph().Nodes()) {
      const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
      if (activation_attr == nullptr || activation_attr->s() != "HardSigmoid") {
        continue;
      }
      const auto* params_attr = graph_utils::GetNodeAttribute(node, "activation_params");
      ASSERT_NE(params_attr, nullptr) << "fused HardSigmoid must carry its parameters";
      ASSERT_GT(params_attr->floats_size(), 0);
      fused_alphas.push_back(params_attr->floats(0));
    }
    // Both must fuse so that one MatMulNaive pipeline is dispatched twice with different uniforms.
    ASSERT_THAT(fused_alphas, ::testing::UnorderedElementsAre(alpha_first, alpha_second))
        << "expected both convolutions to absorb their HardSigmoid, so that a single MatMulNaive "
           "pipeline runs twice with different activation parameter uniforms";
  };

  RunWebGpuFusionTransformerTest(build_test_case,
                                 check_transformed_graph,
                                 TransformerLevel::Level1,
                                 TransformerLevel::Level2,
                                 /*opset_version=*/17,
                                 /*per_sample_tolerance=*/2e-3,
                                 /*relative_per_sample_tolerance=*/2e-3,
                                 /*transformer=*/nullptr,
                                 []() { return DefaultWebGpuExecutionProvider(); });
}

void RunWebGpuConcurrentLayoutConvCacheTest(bool channels_last_first) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 17;
  Model model("WebGpuLayoutCacheTester", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());

  // Keep N, K, and components equal across the channels-first and channels-last paths.
  auto* input = builder.MakeInput<float>({1, 3, 3, 1},
                                         std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  auto* weight = builder.MakeInitializer<float>({3, 3, 1, 1},
                                                std::vector<float>{1.0f, 0.0f, 0.0f,
                                                                   0.0f, 1.0f, 0.0f,
                                                                   0.0f, 0.0f, 1.0f});
  // Use distinct bias values to expose layout-dependent indexing.
  auto* bias = builder.MakeInitializer<float>({3}, std::vector<float>{100.0f, 200.0f, 300.0f});
  auto* output = builder.MakeOutput();
  builder.AddNode("Conv", {input, weight, bias}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Level2;
  RunOptions run_options;

  std::vector<OrtValue> expected_fetches;
  {
    InferenceSessionWrapper cpu_session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(cpu_session.Load(model_data.data(), static_cast<int>(model_data.size())));
    ASSERT_STATUS_OK(cpu_session.Initialize());
    ASSERT_STATUS_OK(cpu_session.Run(run_options, builder.feeds_, builder.output_names_, &expected_fetches));
  }

  // Keep both sessions alive so they share the WebGPU program cache.
  InferenceSessionWrapper session_first{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session_first.RegisterExecutionProvider(DefaultWebGpuExecutionProvider(channels_last_first)));
  ASSERT_STATUS_OK(session_first.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_first.Initialize());

  InferenceSessionWrapper session_second{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session_second.RegisterExecutionProvider(DefaultWebGpuExecutionProvider(!channels_last_first)));
  ASSERT_STATUS_OK(session_second.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_second.Initialize());

  // Ensure both convolutions run on the WebGPU EP.
  for (const auto* session : {&session_first, &session_second}) {
    bool saw_webgpu_conv = false;
    for (const auto& node : session->GetGraph().Nodes()) {
      if (node.OpType() == "Conv" || node.OpType() == "NhwcConv" || node.OpType() == "FusedConv") {
        EXPECT_EQ(node.GetExecutionProviderType(), kWebGpuExecutionProvider);
        saw_webgpu_conv = node.GetExecutionProviderType() == kWebGpuExecutionProvider;
      }
    }
    ASSERT_TRUE(saw_webgpu_conv) << "expected the convolution to run on the WebGPU EP";
  }

  std::vector<OrtValue> fetches_first;
  ASSERT_STATUS_OK(session_first.Run(run_options, builder.feeds_, builder.output_names_, &fetches_first));
  std::vector<OrtValue> fetches_second;
  ASSERT_STATUS_OK(session_second.Run(run_options, builder.feeds_, builder.output_names_, &fetches_second));

  ASSERT_EQ(expected_fetches.size(), fetches_first.size());
  ASSERT_EQ(expected_fetches.size(), fetches_second.size());
  for (size_t i = 0; i < expected_fetches.size(); ++i) {
    auto first = CompareOrtValue(fetches_first[i], expected_fetches[i], 2e-3, 2e-3, false);
    EXPECT_EQ(first.first, COMPARE_RESULT::SUCCESS)
        << "first session (channels_last=" << channels_last_first << "): " << first.second;
    auto second = CompareOrtValue(fetches_second[i], expected_fetches[i], 2e-3, 2e-3, false);
    EXPECT_EQ(second.first, COMPARE_RESULT::SUCCESS)
        << "second session (channels_last=" << !channels_last_first << "): " << second.second;
  }
}

}  // namespace

TEST_F(GraphTransformationTests, WebGpuSmallMatMulConvDistinguishesActivationsInPipelineCache) {
  RunWebGpuSmallMatMulConvActivationPairTest("Relu", "Sigmoid");
  RunWebGpuSmallMatMulConvActivationPairTest("Sigmoid", "Relu");
}

// These alphas render identically at the default stringstream precision, so before activation
// parameters became uniforms they collided in the pipeline cache key. Both convolutions now share
// one pipeline and must still produce distinct results from their per-dispatch uniforms.
TEST_F(GraphTransformationTests, WebGpuSmallMatMulConvSharesPipelineAcrossActivationParams) {
  RunWebGpuSmallMatMulConvHardSigmoidParamPairTest(1000015.0f, 1000025.0f);
  RunWebGpuSmallMatMulConvHardSigmoidParamPairTest(1000025.0f, 1000015.0f);
}

TEST_F(GraphTransformationTests, WebGpuConcurrentLayoutConvsDistinguishedInPipelineCache) {
  RunWebGpuConcurrentLayoutConvCacheTest(/*channels_last_first=*/true);
  RunWebGpuConcurrentLayoutConvCacheTest(/*channels_last_first=*/false);
}
#endif  // defined(USE_WEBGPU)

TEST_F(GraphTransformationTests, FuseConvClip11Activation) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_clip11.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 3);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      EXPECT_TRUE(node.Name() == "Conv1") << "Conv1 should not have been fused as 'min' input to Clip was mutable.";
    }

    if (node.OpType() == "FusedConv") {
      const ONNX_NAMESPACE::AttributeProto& attr_proto = node.GetAttributes().at("activation_params");
      const auto& params = attr_proto.floats();
      // check expected values for each. Conv0 is explicitly specified. Conv2 are defaults
      if (node.Name() == "Conv0") {
        EXPECT_EQ(params.Get(0), -1.f);
        EXPECT_EQ(params.Get(1), 10.f);
      } else if (node.Name() == "Conv2") {
        EXPECT_EQ(params.Get(0), std::numeric_limits<float>::lowest());
        EXPECT_EQ(params.Get(1), std::numeric_limits<float>::max());
      } else {
        FAIL() << "Unexpected fused node name: '" << node.Name() << "'.";
      }
    }
  }
}

// Regression test: when a SelectorActionTransformer fusion runs before partitioning, the
// target nodes still have empty (unassigned) EP. The replacement node must inherit the empty
// EP rather than being silently pinned to CPU; otherwise non-CPU EPs cannot claim the new
// node via GetCapability and the fusion locks them out.
TEST_F(GraphTransformationTests, FuseConvActivationPreEpAssignmentLeavesEpEmpty) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  // Sanity: nodes start with empty EP (loaded from disk, not yet partitioned).
  for (const Node& node : graph.Nodes()) {
    ASSERT_TRUE(node.GetExecutionProviderType().empty())
        << "expected empty EP on freshly loaded node " << node.Name();
  }

  // Register at Level1 so the transformer runs before EP assignment, mirroring
  // the regression scenario this test guards against. Level2+ runs after
  // GraphPartitioner::Partition in the real session pipeline, so it would not
  // exercise the empty-EP code path.
  //
  // Note: ConvActivationFusion is currently registered at Level2 in production;
  // this test validates the fix for any SelectorActionTransformer-based fusion
  // that may be registered pre-partitioning in the future (e.g., for CoreML EP).
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  ASSERT_EQ(graph.NumberOfNodes(), 1);
  const Node& fused = *graph.Nodes().begin();
  ASSERT_EQ(fused.OpType(), "FusedConv");
  EXPECT_TRUE(fused.GetExecutionProviderType().empty())
      << "FusedConv replacement should inherit the target's empty EP, got '"
      << fused.GetExecutionProviderType() << "'";
}

TEST_F(GraphTransformationTests, FuseConvActivationPreservingAttributes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_with_padding_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Relu"], 1);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Relu"], 0);

  ASSERT_EQ(graph.NumberOfNodes(), 1);
  const auto& fused_conv_node = *graph.Nodes().begin();
  ASSERT_EQ(fused_conv_node.OpType(), "FusedConv");

  auto check_ints_attr =
      [&fused_conv_node](const std::string& attr_name, gsl::span<const int64_t> expected_values) {
        const auto& attrs = fused_conv_node.GetAttributes();
        const auto attr_it = attrs.find(attr_name);
        ASSERT_NE(attr_it, attrs.end());
        EXPECT_THAT(attr_it->second.ints(), testing::ContainerEq(expected_values));
      };

  check_ints_attr("pads", AsSpan<int64_t>({1, 1, 1, 1}));
  check_ints_attr("kernel_shape", AsSpan<int64_t>({3, 3}));
}

static void BuildConvQuickGeluGraph(ModelTestBuilder& builder, const std::string& conv_domain,
                                    const std::string& ep, float alpha) {
  auto* input = builder.MakeInput<float>({1, 5, 5, 2}, -1.0f, 1.0f);
  auto* weight = builder.MakeInitializer<float>({3, 2, 3, 3}, -1.0f, 1.0f);
  auto* conv_out = builder.MakeIntermediate();
  auto* output = builder.MakeOutput();

  Node& conv = builder.AddNode("Conv", {input, weight}, {conv_out}, conv_domain);
  conv.SetExecutionProviderType(ep);

  Node& quick_gelu = builder.AddNode("QuickGelu", {conv_out}, {output}, kMSDomain);
  quick_gelu.AddAttribute("alpha", alpha);
  quick_gelu.SetExecutionProviderType(ep);
}

static Status ExpectQuickGeluPresent(Graph& graph) {
  int quick_gelu_count = 0;
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "QuickGelu" && node.Domain() == kMSDomain) {
      ++quick_gelu_count;
    }
  }
  ORT_RETURN_IF_NOT(quick_gelu_count == 1, "expected exactly one QuickGelu node, got ",
                    quick_gelu_count);
  return Status::OK();
}

static Status ExpectUnfusedConvActivationPair(Graph& graph) {
  ORT_RETURN_IF_NOT(graph.NumberOfNodes() == 2,
                    "expected a Conv + activation pair before fusion, got ", graph.NumberOfNodes(),
                    " nodes");
  return Status::OK();
}

// In-memory test models must import the internal NHWC domain explicitly.
static Status RunConvActivationFusion(const std::function<void(ModelTestBuilder&)>& build_test_case,
                                      const logging::Logger& logger,
                                      const std::function<Status(Graph&)>& post_graph_checker,
                                      int opset_version = 13,
                                      const std::function<Status(Graph&)>& post_resolve_mutator = nullptr) {
  const std::unordered_map<std::string, int> domain_to_version{
      {kOnnxDomain, opset_version}, {kMSDomain, 1}, {kMSInternalNHWCDomain, opset_version}};

  Model model("ConvActivationFusionTest", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  build_test_case(builder);
  builder.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());
  ORT_RETURN_IF_ERROR(ExpectUnfusedConvActivationPair(graph));

  // Graph::Resolve() materializes schema attribute defaults onto every node, so a mutator is the
  // only way to hand the fusion a node whose optional attribute is genuinely absent.
  if (post_resolve_mutator) {
    ORT_RETURN_IF_ERROR(post_resolve_mutator(graph));
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(),
                                                        TransformerLevel::Level2));
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  return post_graph_checker(graph);
}

TEST_F(GraphTransformationTests, FuseNhwcConvQuickGeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvQuickGeluGraph(builder, kMSInternalNHWCDomain, kWebGpuExecutionProvider, 1.0f);
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    ORT_RETURN_IF_NOT(graph.NumberOfNodes() == 1, "expected QuickGelu to be folded into Conv, got ",
                      graph.NumberOfNodes(), " nodes");
    const Node& fused = *graph.Nodes().begin();
    ORT_RETURN_IF_NOT(fused.OpType() == "Conv" && fused.Domain() == kMSInternalNHWCDomain,
                      "expected an NHWC Conv, got ", fused.Domain(), ".", fused.OpType());

    const auto* activation = graph_utils::GetNodeAttribute(fused, "activation");
    ORT_RETURN_IF_NOT(activation != nullptr && activation->s() == "QuickGelu",
                      "fused Conv is missing the QuickGelu activation attribute");

    const auto* params = graph_utils::GetNodeAttribute(fused, "activation_params");
    ORT_RETURN_IF_NOT(params != nullptr && params->floats_size() == 1 && params->floats(0) == 1.0f,
                      "fused Conv should carry the QuickGelu alpha as its only activation param");
    return Status::OK();
  };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, post_graph_checker));
}

// CPU FusedConv does not support QuickGelu.
TEST_F(GraphTransformationTests, NoFuseNhwcConvQuickGeluForCpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvQuickGeluGraph(builder, kMSInternalNHWCDomain, kCpuExecutionProvider, 1.0f);
  };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, ExpectQuickGeluPresent));
}

// WebGPU supports this fusion only for internal NHWC Conv.
TEST_F(GraphTransformationTests, NoFuseOnnxDomainConvQuickGeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvQuickGeluGraph(builder, kOnnxDomain, kWebGpuExecutionProvider, 1.0f);
  };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, ExpectQuickGeluPresent));
}

// An absent optional input is either omitted from the node or serialized with an empty name;
// both forms are bias-free, but only the first shrinks InputDefs().
enum class OptionalActivationInput {
  kAbsent,
  kEmptyName,
  kPresent,
};

// Push the output transpose through the activation before ConvActivationFusion runs.
static Status RunLayoutPropagationThenConvActivationFusion(
    const std::string& act_op, const std::string& act_domain, const logging::Logger& logger,
    const std::function<void(Node&)>& decorate, const std::function<Status(Graph&)>& post_graph_checker) {
  const std::unordered_map<std::string, int> domain_to_version{
      {kOnnxDomain, 13}, {kMSDomain, 1}, {kMSInternalNHWCDomain, 13}};

  Model model("LayoutPropagationConvActivationTest", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* input = builder.MakeInput<float>({1, 5, 7, 2}, -1.0f, 1.0f);
  auto* weight = builder.MakeInitializer<float>({3, 2, 3, 3}, -1.0f, 1.0f);
  auto* conv_out = builder.MakeIntermediate();
  auto* transpose_out = builder.MakeIntermediate();
  auto* output = builder.MakeOutput();

  Node& conv = builder.AddNode("Conv", {input, weight}, {conv_out}, kMSInternalNHWCDomain);
  conv.SetExecutionProviderType(kWebGpuExecutionProvider);

  Node& transpose = builder.AddNode("Transpose", {conv_out}, {transpose_out});
  transpose.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
  transpose.SetExecutionProviderType(kWebGpuExecutionProvider);

  Node& activation = builder.AddNode(act_op, {transpose_out}, {output}, act_domain);
  if (decorate) {
    decorate(activation);
  }
  activation.SetExecutionProviderType(kWebGpuExecutionProvider);

  builder.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());
  ORT_RETURN_IF_NOT(graph.NumberOfNodes() == 3, "expected Conv -> Transpose -> activation, got ",
                    graph.NumberOfNodes(), " nodes");

  {
    auto api_graph = MakeApiGraph(graph, TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                                  kWebGpuExecutionProvider);
    auto always_push = [](const onnx_transpose_optimization::api::GraphRef&,
                          const onnx_transpose_optimization::api::NodeRef&, const std::vector<int64_t>&,
                          const std::unordered_set<std::string>&) {
      return onnx_transpose_optimization::CostCheckResult::kPushTranspose;
    };
    onnx_transpose_optimization::OptimizeResult result = onnx_transpose_optimization::Optimize(
        *api_graph, kWebGpuExecutionProvider, always_push, OrtExtendedHandlers());
    ORT_RETURN_IF(result.error_msg.has_value(), "transpose optimization failed: ", *result.error_msg);
  }
  ORT_RETURN_IF_ERROR(graph.Resolve());

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(),
                                                        TransformerLevel::Level2));
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  return post_graph_checker(graph);
}

// The output transpose remains after activation fusion.
static Status ExpectActivationPropagatedAndFused(Graph& graph, const std::string& activation) {
  ORT_RETURN_IF_NOT(graph.NumberOfNodes() == 2,
                    "expected a fused NHWC Conv plus the pushed Transpose, got ", graph.NumberOfNodes(),
                    " nodes; the activation did not move past the Transpose");

  const Node* fused = nullptr;
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Conv" && node.Domain() == kMSInternalNHWCDomain) {
      fused = &node;
    }
  }
  ORT_RETURN_IF_NOT(fused != nullptr, "expected the NHWC Conv to survive");

  const auto* attr = graph_utils::GetNodeAttribute(*fused, "activation");
  ORT_RETURN_IF_NOT(attr != nullptr && attr->s() == activation, "expected ", activation,
                    " to be fused into the NHWC Conv");
  return Status::OK();
}

TEST_F(GraphTransformationTests, LayoutPropagationFusesContribQuickGeluIntoNhwcConv) {
  auto decorate = [](Node& node) { node.AddAttribute("alpha", 1.702f); };
  auto post_graph_checker = [](Graph& graph) -> Status {
    return ExpectActivationPropagatedAndFused(graph, "QuickGelu");
  };
  ASSERT_STATUS_OK(RunLayoutPropagationThenConvActivationFusion("QuickGelu", kMSDomain, *logger_,
                                                                decorate, post_graph_checker));
}

TEST_F(GraphTransformationTests, LayoutPropagationFusesContribGeluIntoNhwcConv) {
  auto post_graph_checker = [](Graph& graph) -> Status {
    return ExpectActivationPropagatedAndFused(graph, "Gelu");
  };
  ASSERT_STATUS_OK(RunLayoutPropagationThenConvActivationFusion("Gelu", kMSDomain, *logger_, nullptr,
                                                                post_graph_checker));
}

TEST_F(GraphTransformationTests, LayoutPropagationFusesContribFastGeluIntoNhwcConv) {
  auto post_graph_checker = [](Graph& graph) -> Status {
    return ExpectActivationPropagatedAndFused(graph, "FastGelu");
  };
  ASSERT_STATUS_OK(RunLayoutPropagationThenConvActivationFusion("FastGelu", kMSDomain, *logger_, nullptr,
                                                                post_graph_checker));
}

static std::unique_ptr<GraphTransformer> TakeRegisteredLevel2Transformer(const std::string& name,
                                                                         const logging::Logger& logger) {
  SessionOptions session_options;
  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, session_options,
                                                            *TestCPUExecutionProvider(), logger);
  for (auto& transformer : transformers) {
    if (transformer->Name() == name) {
      return std::move(transformer);
    }
  }
  return nullptr;
}

TEST_F(GraphTransformationTests, WebGpuTanhGeluAfterConvFusesToFastGelu) {
  auto fast_gelu_fusion = TakeRegisteredLevel2Transformer("FastGeluFusion", *logger_);
  ASSERT_NE(fast_gelu_fusion, nullptr) << "FastGeluFusion is not registered at Level2";
  const auto& compatible_eps = fast_gelu_fusion->GetCompatibleExecutionProviders();
  EXPECT_TRUE(compatible_eps.count(kWebGpuExecutionProvider) != 0)
      << "FastGeluFusion excludes the WebGPU EP, so the tanh GELU decomposition can never reach "
         "com.microsoft.FastGelu and the Conv+FastGelu fusion advertised for pre-opset-20 models is dead";

  const std::unordered_map<std::string, int> domain_to_version{
      {kOnnxDomain, 13}, {kMSDomain, 1}, {kMSInternalNHWCDomain, 13}};
  Model model("WebGpuTanhGeluAfterConv", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, *logger_);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* input = builder.MakeInput<float>({1, 5, 7, 2}, -1.0f, 1.0f);
  auto* weight = builder.MakeInitializer<float>({3, 2, 3, 3}, -1.0f, 1.0f);
  auto* coeff = builder.MakeInitializer<float>({}, {0.044715f});
  auto* sqrt2pi = builder.MakeInitializer<float>({}, {0.7978845834732056f});
  auto* one = builder.MakeInitializer<float>({}, {1.0f});
  auto* half = builder.MakeInitializer<float>({}, {0.5f});
  auto* three = builder.MakeInitializer<float>({}, {3.0f});

  auto* conv_out = builder.MakeIntermediate();
  auto* pow_out = builder.MakeIntermediate();
  auto* mul1_out = builder.MakeIntermediate();
  auto* add1_out = builder.MakeIntermediate();
  auto* mul2_out = builder.MakeIntermediate();
  auto* tanh_out = builder.MakeIntermediate();
  auto* add2_out = builder.MakeIntermediate();
  auto* mul_half_out = builder.MakeIntermediate();
  auto* gelu_out = builder.MakeIntermediate();
  auto* output = builder.MakeOutput();

  builder.AddNode("Conv", {input, weight}, {conv_out}, kMSInternalNHWCDomain);
  builder.AddNode("Pow", {conv_out, three}, {pow_out});
  builder.AddNode("Mul", {pow_out, coeff}, {mul1_out});
  builder.AddNode("Add", {conv_out, mul1_out}, {add1_out});
  builder.AddNode("Mul", {add1_out, sqrt2pi}, {mul2_out});
  builder.AddNode("Tanh", {mul2_out}, {tanh_out});
  builder.AddNode("Add", {tanh_out, one}, {add2_out});
  builder.AddNode("Mul", {conv_out, half}, {mul_half_out});
  builder.AddNode("Mul", {mul_half_out, add2_out}, {gelu_out});
  // FastGeluFusion requires a non-graph-output consumer.
  builder.AddNode("Identity", {gelu_out}, {output});

  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kWebGpuExecutionProvider);
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(fast_gelu_fusion), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(),
                                                     TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  const auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count.count("Tanh"), 0u) << "the tanh decomposition was not collapsed";
  ASSERT_EQ(graph.NumberOfNodes(), 2) << "expected a fused NHWC Conv plus the trailing Identity";

  const Node* fused = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      fused = &node;
    }
  }
  ASSERT_NE(fused, nullptr);
  EXPECT_EQ(fused->Domain(), kMSInternalNHWCDomain);
  const auto* activation = graph_utils::GetNodeAttribute(*fused, "activation");
  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(activation->s(), "FastGelu");
}

static void BuildConvActivationGraph(ModelTestBuilder& builder, const std::string& act_op,
                                     const std::string& act_domain, const std::string& ep,
                                     const std::function<void(Node&)>& decorate = nullptr,
                                     OptionalActivationInput optional_input =
                                         OptionalActivationInput::kAbsent) {
  auto* input = builder.MakeInput<float>({1, 5, 5, 2}, -1.0f, 1.0f);
  auto* weight = builder.MakeInitializer<float>({3, 2, 3, 3}, -1.0f, 1.0f);
  auto* conv_out = builder.MakeIntermediate();
  auto* output = builder.MakeOutput();

  Node& conv = builder.AddNode("Conv", {input, weight}, {conv_out}, kMSInternalNHWCDomain);
  conv.SetExecutionProviderType(ep);

  std::vector<NodeArg*> act_inputs{conv_out};
  if (optional_input == OptionalActivationInput::kPresent) {
    act_inputs.push_back(builder.MakeInitializer<float>({3}, -1.0f, 1.0f));
  } else if (optional_input == OptionalActivationInput::kEmptyName) {
    act_inputs.push_back(builder.MakeEmptyInput());
  }

  Node& activation = builder.AddNode(act_op, act_inputs, {output}, act_domain);
  if (decorate) {
    decorate(activation);
  }
  activation.SetExecutionProviderType(ep);
}

static Status ExpectFusedInto(Graph& graph, const std::string& activation,
                              const std::vector<float>& expected_params) {
  ORT_RETURN_IF_NOT(graph.NumberOfNodes() == 1, "expected ", activation,
                    " to be folded into Conv, got ", graph.NumberOfNodes(), " nodes");
  const Node& fused = *graph.Nodes().begin();
  ORT_RETURN_IF_NOT(fused.OpType() == "Conv" && fused.Domain() == kMSInternalNHWCDomain,
                    "expected an NHWC Conv, got ", fused.Domain(), ".", fused.OpType());

  const auto* activation_attr = graph_utils::GetNodeAttribute(fused, "activation");
  ORT_RETURN_IF_NOT(activation_attr != nullptr && activation_attr->s() == activation,
                    "fused Conv should carry activation ", activation, ", got ",
                    activation_attr == nullptr ? "<none>" : activation_attr->s());

  const auto* params = graph_utils::GetNodeAttribute(fused, "activation_params");
  if (expected_params.empty()) {
    ORT_RETURN_IF_NOT(params == nullptr || params->floats_size() == 0, activation,
                      " takes no parameters, but activation_params was set");
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(params != nullptr, activation, " should carry activation_params");
  ORT_RETURN_IF_NOT(static_cast<size_t>(params->floats_size()) == expected_params.size(),
                    activation, " expected ", expected_params.size(), " activation_params, got ",
                    params->floats_size());
  for (size_t i = 0; i < expected_params.size(); ++i) {
    ORT_RETURN_IF_NOT(params->floats(static_cast<int>(i)) == expected_params[i], activation,
                      " activation_params[", i, "] expected ", expected_params[i], ", got ",
                      params->floats(static_cast<int>(i)));
  }
  return Status::OK();
}

static std::function<Status(Graph&)> ExpectNotFused(const std::string& act_op) {
  return [act_op](Graph& graph) -> Status {
    int count = 0;
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == act_op) {
        ++count;
      }
    }
    ORT_RETURN_IF_NOT(count == 1, "expected ", act_op, " to be left unfused, found ", count,
                      " such nodes in a graph of ", graph.NumberOfNodes());
    return Status::OK();
  };
}

TEST_F(GraphTransformationTests, FuseNhwcConvHardSwishForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "HardSwish", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "HardSwish", {}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvEluPreservesAlphaForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Elu", kOnnxDomain, kWebGpuExecutionProvider,
                             [](Node& node) { node.AddAttribute("alpha", 0.5f); });
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Elu", {0.5f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvEluDefaultsAlphaForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Elu", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Elu", {1.0f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

// QuickGelu authored without an explicit alpha must reach the shader as 1.702, not 1.0 (SiLU).
TEST_F(GraphTransformationTests, FuseNhwcConvQuickGeluDefaultsAlphaForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "QuickGelu", kMSDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "QuickGelu", {1.702f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

// Resolve() materializes alpha from the schema, so stripping it is what exercises the fallback.
TEST_F(GraphTransformationTests, FuseNhwcConvQuickGeluFallbackAlphaMatchesSchemaDefault) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "QuickGelu", kMSDomain, kWebGpuExecutionProvider);
  };
  auto strip_alpha = [](Graph& graph) -> Status {
    bool cleared = false;
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "QuickGelu" && node.Domain() == kMSDomain) {
        cleared = node.ClearAttribute("alpha");
      }
    }
    ORT_RETURN_IF_NOT(cleared, "expected Graph::Resolve() to have materialized QuickGelu's alpha");
    return Status::OK();
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "QuickGelu", {1.702f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17, strip_alpha));
}

// activation_params encodes Gelu approximation as 0 (erf) or 1 (tanh).
TEST_F(GraphTransformationTests, FuseNhwcConvOnnxGeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Gelu", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Gelu", {0.0f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 20));
}

TEST_F(GraphTransformationTests, FuseNhwcConvOnnxGeluTanhForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Gelu", kOnnxDomain, kWebGpuExecutionProvider,
                             [](Node& node) { node.AddAttribute("approximate", "tanh"); });
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Gelu", {1.0f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 20));
}

TEST_F(GraphTransformationTests, FuseNhwcConvContribGeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Gelu", kMSDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Gelu", {0.0f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvFastGeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "FastGelu", kMSDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "FastGelu", {}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

// FastGelu with a bias input is not elementwise.
TEST_F(GraphTransformationTests, NoFuseNhwcConvFastGeluWithBiasForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "FastGelu", kMSDomain, kWebGpuExecutionProvider, nullptr,
                             OptionalActivationInput::kPresent);
  };

  ASSERT_STATUS_OK(
      RunConvActivationFusion(build_test_case, *logger_, ExpectNotFused("FastGelu"), 17));
}

// An empty-name optional bias is bias-free, so this FastGelu must fuse like the omitted form.
TEST_F(GraphTransformationTests, FuseNhwcConvFastGeluWithEmptyBiasForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "FastGelu", kMSDomain, kWebGpuExecutionProvider, nullptr,
                             OptionalActivationInput::kEmptyName);
  };

  auto verify_fusion_sees_the_empty_bias = [](Graph& graph) -> Status {
    bool checked = false;
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() != "FastGelu") {
        continue;
      }
      ORT_RETURN_IF_NOT(node.InputDefs().size() == 2,
                        "Graph::Resolve() dropped FastGelu's empty optional input, leaving ",
                        node.InputDefs().size(), " inputs");
      ORT_RETURN_IF_NOT(!node.InputDefs()[1]->Exists(),
                        "expected FastGelu's second input to be a non-existent NodeArg");
      checked = true;
    }
    ORT_RETURN_IF_NOT(checked, "no FastGelu node found before fusion");
    return Status::OK();
  };

  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "FastGelu", {}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17,
                                           verify_fusion_sees_the_empty_bias));
}

TEST_F(GraphTransformationTests, FuseNhwcConvSoftplusForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Softplus", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Softplus", {}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvThresholdedReluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "ThresholdedRelu", kOnnxDomain, kWebGpuExecutionProvider,
                             [](Node& node) { node.AddAttribute("alpha", 0.6f); });
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "ThresholdedRelu", {0.6f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvThresholdedReluDefaultAlphaForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "ThresholdedRelu", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "ThresholdedRelu", {1.0f}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

TEST_F(GraphTransformationTests, FuseNhwcConvErfForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Erf", kOnnxDomain, kWebGpuExecutionProvider);
  };
  auto check = [](Graph& graph) { return ExpectFusedInto(graph, "Erf", {}); };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, check, 17));
}

// WebGPU has no Selu kernel.
TEST_F(GraphTransformationTests, NoFuseNhwcConvSeluForWebGpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Selu", kOnnxDomain, kWebGpuExecutionProvider);
  };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, ExpectNotFused("Selu"), 17));
}

TEST_F(GraphTransformationTests, NoFuseNhwcConvHardSwishForCpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "HardSwish", kOnnxDomain, kCpuExecutionProvider);
  };

  ASSERT_STATUS_OK(
      RunConvActivationFusion(build_test_case, *logger_, ExpectNotFused("HardSwish"), 17));
}

TEST_F(GraphTransformationTests, NoFuseNhwcConvGeluForCpuEp) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildConvActivationGraph(builder, "Gelu", kOnnxDomain, kCpuExecutionProvider);
  };

  ASSERT_STATUS_OK(RunConvActivationFusion(build_test_case, *logger_, ExpectNotFused("Gelu"), 20));
}

#if defined(USE_WEBGPU)
namespace {

using ConvActivationBuilder = std::function<void(ModelTestBuilder&, NodeArg*, NodeArg*)>;

// Compare Level2 fused execution with the Level1 unfused baseline.
void RunWebGpuConvActivationParity(const ConvActivationBuilder& add_activation,
                                   const std::string& expected_activation,
                                   int opset_version,
                                   const std::vector<int64_t>& input_shape = {1, 4, 14, 14},
                                   const std::vector<int64_t>& weight_shape = {8, 4, 3, 3}) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto build_test_case = [&add_activation, &input_shape, &weight_shape](ModelTestBuilder& builder) {
    // Include negative inputs for asymmetric activations.
    auto* input = builder.MakeInput<float>(input_shape, -3.0f, 3.0f);
    auto* weight = builder.MakeInitializer<float>(weight_shape, -0.5f, 0.5f);
    auto* bias = builder.MakeInitializer<float>({weight_shape[0]}, -0.5f, 0.5f);
    auto* conv_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    // Start from NCHW so layout transformation creates the NHWC fusion candidate.
    builder.AddNode("Conv", {input, weight, bias}, {conv_out});
    add_activation(builder, conv_out, output);
  };

  auto check_transformed_graph = [&expected_activation](InferenceSessionWrapper& session) {
    bool fused = false;
    std::ostringstream graph_description;
    for (const auto& node : session.GetGraph().Nodes()) {
      graph_description << " " << node.Domain() << "." << node.OpType()
                        << "[" << node.GetExecutionProviderType() << "]";
      if (node.OpType() != "Conv") {
        continue;
      }
      const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
      if (activation_attr != nullptr && activation_attr->s() == expected_activation) {
        fused = true;
      }
    }
    // Ensure the activation is fused; otherwise this test would not exercise the fused kernel.
    ASSERT_TRUE(fused) << "Conv did not absorb " << expected_activation
                       << ", so the fused kernel was never executed. Graph was:"
                       << graph_description.str();
  };

  RunWebGpuFusionTransformerTest(build_test_case,
                                 check_transformed_graph,
                                 TransformerLevel::Level1,
                                 TransformerLevel::Level2,
                                 opset_version,
                                 /*per_sample_tolerance=*/2e-3,
                                 /*relative_per_sample_tolerance=*/2e-3,
                                 /*transformer=*/nullptr,
                                 []() { return DefaultWebGpuExecutionProvider(); });
}

ConvActivationBuilder SimpleActivation(const std::string& op_type,
                                       const std::string& domain = kOnnxDomain,
                                       const std::function<void(Node&)>& decorate = nullptr) {
  return [op_type, domain, decorate](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    Node& activation = builder.AddNode(op_type, {conv_out}, {output}, domain);
    if (decorate) {
      decorate(activation);
    }
  };
}

// Use fp16, group 1, NHWC, and a non-1x1 kernel to select Im2ColMatMulProgram.
void RunWebGpuIm2ColActivationParity(const ConvActivationBuilder& add_activation,
                                     const std::string& expected_activation,
                                     int opset_version,
                                     int64_t input_channels = 4) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  const std::vector<int64_t> input_shape{1, input_channels, 14, 14};
  const std::vector<int64_t> weight_shape{8, input_channels, 3, 3};

  auto to_fp16 = [](const std::vector<float>& values) {
    std::vector<MLFloat16> converted;
    converted.reserve(values.size());
    for (float v : values) {
      converted.push_back(MLFloat16(v));
    }
    return converted;
  };

  // Use deterministic signed inputs for stable fp16 comparisons.
  auto ramp = [](size_t count, float lo, float hi) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = lo + (hi - lo) * (static_cast<float>(i % 32) / 31.0f);
    }
    return values;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<MLFloat16>(
        input_shape, to_fp16(ramp(static_cast<size_t>(input_channels * 14 * 14), -3.0f, 3.0f)));
    auto* weight = builder.MakeInitializer<MLFloat16>(
        weight_shape, to_fp16(ramp(static_cast<size_t>(8 * input_channels * 3 * 3), -0.5f, 0.5f)));
    auto* bias = builder.MakeInitializer<MLFloat16>({weight_shape[0]}, to_fp16(ramp(8, -0.5f, 0.5f)));
    auto* conv_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Conv", {input, weight, bias}, {conv_out});
    add_activation(builder, conv_out, output);
  };

  bool im2col_selected = false;
  std::string observed_conv_programs;
  auto check_transformed_graph = [&](InferenceSessionWrapper& session) {
    bool fused = false;
    std::ostringstream graph_description;
    for (const auto& node : session.GetGraph().Nodes()) {
      graph_description << " " << node.Domain() << "." << node.OpType()
                        << "[" << node.GetExecutionProviderType() << "]";
      if (node.OpType() != "Conv") {
        continue;
      }
      const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
      if (activation_attr != nullptr && activation_attr->s() == expected_activation) {
        fused = true;
      }
    }
    ASSERT_TRUE(fused) << "Conv did not absorb " << expected_activation
                       << ", so no fused kernel ran at all. Graph was:" << graph_description.str();

    // Confirm profiling observed Im2ColMatMulProgram rather than the fallback Conv path.
    const std::string profile_path = session.EndProfiling();
    ASSERT_FALSE(profile_path.empty()) << "profiling produced no file, so program selection is unverifiable";
    std::ifstream profile_stream(profile_path);
    ASSERT_TRUE(profile_stream.good()) << "cannot read profile " << profile_path;
    std::stringstream buffer;
    buffer << profile_stream.rdbuf();
    const std::string profile_contents = buffer.str();

    im2col_selected = profile_contents.find("Im2ColMatMul") != std::string::npos;
    for (size_t pos = profile_contents.find("&Conv&"); pos != std::string::npos;
         pos = profile_contents.find("&Conv&", pos + 1)) {
      const size_t end = profile_contents.find('"', pos);
      if (end != std::string::npos) {
        observed_conv_programs += " " + profile_contents.substr(pos + 6, end - pos - 6);
      }
    }
    // GPU timestamp events carry a "cache_key" argument. A device without timestamp query support
    // emits none at all, which is not the same as dispatching an unexpected program.
    if (profile_contents.find("cache_key") == std::string::npos) {
      GTEST_SKIP() << "device reported no GPU timestamps, so program selection is unobservable here";
    }
    ASSERT_FALSE(observed_conv_programs.empty())
        << "GPU kernels were profiled but none was a Conv dispatch, so this test can no longer tell "
           "whether the im2col program was selected";
  };

  RunWebGpuFusionTransformerTest(
      build_test_case, check_transformed_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset_version,
      /*per_sample_tolerance=*/2e-2,
      /*relative_per_sample_tolerance=*/2e-2,
      /*transformer=*/nullptr, []() { return DefaultWebGpuExecutionProvider(); }, [](SessionOptions& session_options) {
                                   session_options.enable_profiling = true;
                                   session_options.profile_file_prefix = ORT_TSTR("webgpu_im2col_activation"); });

  // Im2ColMatMulProgram is restricted to Intel Xe-2/Xe-3.
  if (!im2col_selected) {
    GTEST_SKIP() << "Im2ColMatMul did not run on this adapter, so the im2col activation epilogue was "
                    "not exercised. Conv dispatched to:"
                 << observed_conv_programs
                 << ". Requires an Intel Xe-2/Xe-3 GPU per IsDeviceSupported().";
  }
}

}  // namespace

TEST_F(GraphTransformationTests, WebGpuConvReluFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(SimpleActivation("Relu"), "Relu", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvLeakyReluFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(
      SimpleActivation("LeakyRelu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.15f); }),
      "LeakyRelu", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvHardSigmoidFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(
      SimpleActivation("HardSigmoid", kOnnxDomain,
                       [](Node& node) {
                         node.AddAttribute("alpha", 0.3f);
                         node.AddAttribute("beta", 0.4f);
                       }),
      "HardSigmoid", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvEluFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(
      SimpleActivation("Elu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.7f); }),
      "Elu", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvClipFusionMatchesUnfusedResults) {
  auto add_clip = [](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    auto* min_value = builder.MakeScalarInitializer<float>(-0.25f);
    auto* max_value = builder.MakeScalarInitializer<float>(0.75f);
    builder.AddNode("Clip", {conv_out, min_value, max_value}, {output});
  };
  RunWebGpuConvActivationParity(add_clip, "Clip", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvHardSwishFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(SimpleActivation("HardSwish"), "HardSwish", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvSoftplusFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(SimpleActivation("Softplus"), "Softplus", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvGeluFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(SimpleActivation("Gelu"), "Gelu", 20);
}

TEST_F(GraphTransformationTests, WebGpuConvGeluTanhFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(
      SimpleActivation("Gelu", kOnnxDomain,
                       [](Node& node) { node.AddAttribute("approximate", std::string("tanh")); }),
      "Gelu", 20);
}

// Build the exporter pattern consumed by QuickGeluFusion.
TEST_F(GraphTransformationTests, WebGpuConvQuickGeluFusionMatchesUnfusedResults) {
  auto add_quick_gelu = [](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    auto* alpha = builder.MakeScalarInitializer<float>(1.4f);
    auto* scaled = builder.MakeIntermediate();
    auto* sigmoid_out = builder.MakeIntermediate();
    builder.AddNode("Mul", {conv_out, alpha}, {scaled});
    builder.AddNode("Sigmoid", {scaled}, {sigmoid_out});
    builder.AddNode("Mul", {conv_out, sigmoid_out}, {output});
  };
  RunWebGpuConvActivationParity(add_quick_gelu, "QuickGelu", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvThresholdedReluFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(
      SimpleActivation("ThresholdedRelu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.8f); }),
      "ThresholdedRelu", 17);
}

TEST_F(GraphTransformationTests, WebGpuConvErfFusionMatchesUnfusedResults) {
  RunWebGpuConvActivationParity(SimpleActivation("Erf"), "Erf", 17);
}

// Each call builds its own session, so these are two independent parity checks rather than a
// pipeline-sharing test. Sharing is covered by WebGpuSmallMatMulConvSharesPipelineAcrossActivationParams,
// which puts both parameter values in one session.
TEST_F(GraphTransformationTests, WebGpuConvLeakyReluParityAcrossAlphaValues) {
  // Keep N and K below 8 to select MatMulNaiveProgram.
  const std::vector<int64_t> input_shape{1, 4, 8, 8};
  const std::vector<int64_t> weight_shape{6, 4, 1, 1};

  RunWebGpuConvActivationParity(
      SimpleActivation("LeakyRelu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.01f); }),
      "LeakyRelu", 17, input_shape, weight_shape);
  RunWebGpuConvActivationParity(
      SimpleActivation("LeakyRelu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.5f); }),
      "LeakyRelu", 17, input_shape, weight_shape);
}

TEST_F(GraphTransformationTests, WebGpuIm2ColConvReluFusionMatchesUnfusedResults) {
  RunWebGpuIm2ColActivationParity(SimpleActivation("Relu"), "Relu", 17);
}

// Odd input channels give vec_size == 1, which takes the scalar path of the shader (including the
// subgroup variant when subgroup-size control is available). 4 channels give vec_size == 4 and 2 give 2.
TEST_F(GraphTransformationTests, WebGpuIm2ColConvOddInputChannelsMatchesUnfusedResults) {
  RunWebGpuIm2ColActivationParity(SimpleActivation("Relu"), "Relu", 17, /*input_channels=*/3);
}

TEST_F(GraphTransformationTests, WebGpuIm2ColConvTwoInputChannelsMatchesUnfusedResults) {
  RunWebGpuIm2ColActivationParity(SimpleActivation("Relu"), "Relu", 17, /*input_channels=*/2);
}

TEST_F(GraphTransformationTests, WebGpuIm2ColConvLeakyReluFusionMatchesUnfusedResults) {
  RunWebGpuIm2ColActivationParity(
      SimpleActivation("LeakyRelu", kOnnxDomain, [](Node& node) { node.AddAttribute("alpha", 0.25f); }),
      "LeakyRelu", 17);
}

TEST_F(GraphTransformationTests, WebGpuIm2ColConvHardSigmoidFusionMatchesUnfusedResults) {
  RunWebGpuIm2ColActivationParity(
      SimpleActivation("HardSigmoid", kOnnxDomain,
                       [](Node& node) {
                         node.AddAttribute("alpha", 0.3f);
                         node.AddAttribute("beta", 0.7f);
                       }),
      "HardSigmoid", 17);
}

// Clip is the only two-slot activation whose uniform slots mean {min, max} rather than
// {alpha, beta}, so it is the one case where swapping or mis-indexing activation_param_0/1
// would survive the HardSigmoid test above.
TEST_F(GraphTransformationTests, WebGpuIm2ColConvClipFusionMatchesUnfusedResults) {
  auto add_clip = [](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    // min/max must match the fp16 tensor type this path requires.
    auto* min_value = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(-0.25f));
    auto* max_value = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(0.75f));
    builder.AddNode("Clip", {conv_out, min_value, max_value}, {output});
  };
  RunWebGpuIm2ColActivationParity(add_clip, "Clip", 17);
}

// Build the exporter pattern consumed by QuickGeluFusion: x * sigmoid(alpha * x).
TEST_F(GraphTransformationTests, WebGpuIm2ColConvQuickGeluFusionMatchesUnfusedResults) {
  auto add_quick_gelu = [](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    // alpha must match the fp16 tensor type this path requires.
    auto* alpha = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(1.4f));
    auto* scaled = builder.MakeIntermediate();
    auto* sigmoid_out = builder.MakeIntermediate();
    builder.AddNode("Mul", {conv_out, alpha}, {scaled});
    builder.AddNode("Sigmoid", {scaled}, {sigmoid_out});
    builder.AddNode("Mul", {conv_out, sigmoid_out}, {output});
  };
  RunWebGpuIm2ColActivationParity(add_quick_gelu, "QuickGelu", 17);
}

// x * sigmoid(x) fuses to QuickGelu with alpha 1, which selects the shader variant that drops
// both the multiply and activation_param_0. Without this case the alpha-1 template branch is
// never compiled, so a stale uniform reference in it would not be caught.
TEST_F(GraphTransformationTests, WebGpuIm2ColConvQuickGeluUnitAlphaFusionMatchesUnfusedResults) {
  auto add_silu = [](ModelTestBuilder& builder, NodeArg* conv_out, NodeArg* output) {
    auto* sigmoid_out = builder.MakeIntermediate();
    builder.AddNode("Sigmoid", {conv_out}, {sigmoid_out});
    builder.AddNode("Mul", {conv_out, sigmoid_out}, {output});
  };
  RunWebGpuIm2ColActivationParity(add_silu, "QuickGelu", 17);
}
#endif  // defined(USE_WEBGPU)
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST_F(GraphTransformationTests, FuseConvMulNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-mul-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

// if IR version is 4 or higher the weights can be overridden if there's a matching graph input.
// check that we don't fuse if that is the case
TEST_F(GraphTransformationTests, NegativeFuseConvAddNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/negative-fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Nodes are not fused because the weights to conv/add are not constants (they appear in the graph inputs).
  // Unsqueeze is also not eliminated as the initializer that is its input is also not constant
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] != 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] != 0);
}

// Basic test: Unsqueeze with a single axis on a constant initializer is eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_BasicConstantInput) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Input shape [2, 3] with axes {0} => output shape [1, 2, 3].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 3);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze with multiple axes on a constant initializer is eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_MultipleAxes) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0, 2});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Input shape [4] with axes {0, 2} => output shape [1, 4, 1].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 4);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze with negative axis on a constant initializer is eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_NegativeAxis) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    // Input rank 2, axes {-1}: output rank = 3, -1 maps to axis 2.
    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{-1});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Input shape [2, 3] with axes {-1} => output shape [2, 3, 1].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 3);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze on a scalar constant initializer is eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_ScalarInput) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({}, {42.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0, 1});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Scalar with axes {0, 1} => output shape [1, 1].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 2);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze whose input is a graph input (not constant) is NOT eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_NonConstantInputNotEliminated) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{2, 3}});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {input_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, checker, checker));
}

// Boundary test: axes at the valid extremes are correctly handled.
TEST_F(GraphTransformationTests, UnsqueezeElimination_AxisBoundaryValues) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({2}, {1.0f, 2.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    // Input rank 1, axes {-3, 2}: output rank = 3.
    // -3 is the minimum valid negative axis (maps to 0), 2 is the maximum valid positive axis.
    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{-3, 2});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Input [2] with axes {-3, 2} => axes {0, 2} => output shape [1, 2, 1].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze whose output is directly a graph output is NOT eliminated
// because the generated initializer name won't match the graph output name.
TEST_F(GraphTransformationTests, UnsqueezeElimination_OutputIsGraphOutput) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({3}, {1.0f, 2.0f, 3.0f});
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {output_arg});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0});
  };

  auto checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, checker, checker));
}

// Unsqueeze with non-float data type (int32) constant initializer is eliminated.
TEST_F(GraphTransformationTests, UnsqueezeElimination_Int32Initializer) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<int32_t>({2, 2}, {10, 20, 30, 40});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{1});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    // Input shape [2, 2] with axes {1} => output shape [2, 1, 2].
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 2);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Only the Unsqueeze with a constant initializer input is eliminated in a mixed graph.
TEST_F(GraphTransformationTests, UnsqueezeElimination_MixedConstantAndNonConstant) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    // This one should be eliminated.
    auto* initializer_arg = builder.MakeInitializer<float>({3}, {1.0f, 2.0f, 3.0f});
    auto* unsqueeze_out_1 = builder.MakeIntermediate();
    auto& unsqueeze1 = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out_1});
    unsqueeze1.AddAttribute("axes", std::vector<int64_t>{0});

    // This one should NOT be eliminated.
    auto* graph_input = builder.MakeInput<float>({{3}});
    auto* unsqueeze_out_2 = builder.MakeIntermediate();
    auto& unsqueeze2 = builder.AddNode("Unsqueeze", {graph_input}, {unsqueeze_out_2});
    unsqueeze2.AddAttribute("axes", std::vector<int64_t>{0});

    auto* output_arg = builder.MakeOutput();
    builder.AddNode("Add", {unsqueeze_out_1, unsqueeze_out_2}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 2);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze with all negative axes.
TEST_F(GraphTransformationTests, UnsqueezeElimination_AllNegativeAxes) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({2}, {1.0f, 2.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    // Input rank 1, axes {-1, -3}: output rank = 3.
    // -1 maps to axis 2, -3 maps to axis 0. => output shape [1, 2, 1].
    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{-1, -3});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 3);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Unsqueeze inserting dimensions at multiple positions on a rank-1 input.
TEST_F(GraphTransformationTests, UnsqueezeElimination_ManyAxes) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* initializer_arg = builder.MakeInitializer<float>({2}, {1.0f, 2.0f});
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    // Input rank 1, axes {0, 2, 3}: output rank = 4 => shape [1, 2, 1, 1].
    auto& unsqueeze_node = builder.AddNode("Unsqueeze", {initializer_arg}, {unsqueeze_out});
    unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{0, 2, 3});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        auto* shape = node.InputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape != nullptr);
        TEST_RETURN_IF_NOT(shape->dim_size() == 4);
        TEST_RETURN_IF_NOT(shape->dim(0).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(1).dim_value() == 2);
        TEST_RETURN_IF_NOT(shape->dim(2).dim_value() == 1);
        TEST_RETURN_IF_NOT(shape->dim(3).dim_value() == 1);
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// NOTE: Duplicate-axis and out-of-range axis error paths in UnsqueezeElimination::Apply
// are defense-in-depth guards. They cannot be exercised through ModelTestBuilder because
// ONNX schema validation during graph.Resolve() rejects such invalid models before the
// optimizer runs.

static void TestFuseConvAddMul(logging::Logger& logger, const PathChar* model_uri) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul1D) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-1d.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D_2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d-2.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul1D_2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-1d-2.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_two_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/2Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_three_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_three_input_with_1d) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/model_1d.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
  ASSERT_TRUE(op_to_count["Reshape"] == 2);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      auto shape_proto = node.OutputDefs()[0]->Shape();
      ASSERT_TRUE(shape_proto != nullptr);
      auto shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
      if (node.Name().find("gemm_input") != std::string::npos) {
        ASSERT_TRUE(shape.NumDimensions() == 2 && shape[0] == 1 && shape[1] == 4);
      } else {
        ASSERT_TRUE(shape.NumDimensions() == 1 && shape[0] == 3);
      }
    }
  }
}

// Matmul+Add with shape [M,k]*[k,N]+[1,4], won't do the fusion
// 1,4 is not uni-directionally broadcast
TEST_F(GraphTransformationTests, MatMulAddFusion_NotBroadcastable) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_not_broadcastable.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_MissingShape) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_missing_shape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Add"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_NeedReshape_3D) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{8, 16, 32}});
    auto* weight_arg = builder.MakeInput<float>({{32, 768}});
    auto* bias_arg = builder.MakeInput<float>({{768}});
    auto* matmul_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    builder.AddNode("MatMul", {input_arg, weight_arg}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias_arg}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Gemm"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "Reshape") {
        auto shape_proto = node.OutputDefs()[0]->Shape();
        TEST_RETURN_IF_NOT(shape_proto != nullptr);
        auto shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
        if (node.Name().find("gemm_input") != std::string::npos) {
          TEST_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[0] == 8 * 16 && shape[1] == 32);
        } else {
          TEST_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[0] == 8 && shape[1] == 16 && shape[2] == 768);
        }
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<MatMulAddFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

// With attention pattern, but targeting an execution provider that does not perform
// AttentionFusion, fuse into GEMM should still be happen, rather than skipping them
TEST_F(GraphTransformationTests, MatMulAddFusion_PreserveAttentionPattern) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_from_attention.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  // This toy model contains 11 MatMul + Add pairs, 0 GEMMs.
  // 7 of them are out of "Attention Pattern" (see MatMulAddFusion::IsAttentionPattern)
  // 4 of them are in "Attention Pattern" conditionally skipped by MatMulAddFusion pass
  OpCountMap op_count_before = CountOpsInGraph(p_model->MainGraph());
  const InlinedHashSet<std::string_view> empty_list = {};

  // In attention pattern, 4 MatMul + Add pairs should be preserved
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatMulAddFusion>(empty_list, /*preserve_attention_pattern=*/true), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  OpCountMap op_count_cpu_ep = CountOpsInGraph(graph);
  constexpr int expected_fusions1 = 11 - 4;
  ASSERT_EQ(op_count_cpu_ep["MatMul"], op_count_before["MatMul"] - expected_fusions1);
  ASSERT_EQ(op_count_cpu_ep["Add"], op_count_before["Add"] - expected_fusions1);
  ASSERT_EQ(op_count_cpu_ep["Gemm"], op_count_before["Gemm"] + expected_fusions1);

  // In attention pattern, 4 MatMul + Add pairs should be fused into Gemm
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatMulAddFusion>(empty_list, /*preserve_attention_pattern=*/false), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  OpCountMap op_count_qnn_ep = CountOpsInGraph(graph);
  constexpr int expected_fusions2 = 11;
  ASSERT_EQ(op_count_qnn_ep["MatMul"], op_count_before["MatMul"] - expected_fusions2);
  ASSERT_EQ(op_count_qnn_ep["Add"], op_count_before["Add"] - expected_fusions2);
  ASSERT_EQ(op_count_qnn_ep["Gemm"], op_count_before["Gemm"] + expected_fusions2);
}

#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, Gemm_Relu_three_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GemmActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);
}

TEST_F(GraphTransformationTests, TransposeMatmulNoFusionForCpuFp16) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* q = builder.MakeInput<MLFloat16>({{2, 4, 8, 16}});
    auto* k = builder.MakeInput<MLFloat16>({{2, 4, 6, 16}});
    auto* k_transposed = builder.MakeIntermediate<MLFloat16>({{2, 4, 16, 6}});
    auto* output = builder.MakeOutput<MLFloat16>({{2, 4, 8, 6}});

    builder.AddNode("Transpose", {k}, {k_transposed})
        .AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 2});
    builder.AddNode("MatMul", {q, k_transposed}, {output})
        .SetExecutionProviderType(kCpuExecutionProvider);
  };

  auto check_unfused = [](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.FusedMatMul"] == 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatmulTransposeFusion>(),
      TransformerLevel::Level2, 1, check_unfused, check_unfused));
}

TEST_F(GraphTransformationTests, TransposeCastMatmulFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion0.onnx",  // Test fusion from the right input
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion1.onnx",  // Test fusion from the left input
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion2.onnx",  // Test fusion both from the left and right inputs
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion3.onnx",  // Cast nodes feed multiple MatMul nodes.
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion4.onnx",  // Cast nodes feed one MatMul node and
                                                                    // the Transpose nodes feed another MatMul node.
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion5.onnx"   // One Cast node and one Transpose node feed each
                                                                    // MatMul nodes.
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> orig_op_to_count = CountOpsInGraph(graph);  // Original op count

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                       TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["Transpose"] == 0);
    ASSERT_TRUE(op_to_count["MatMul"] == 0);
    ASSERT_TRUE(op_to_count["Cast"] == orig_op_to_count["Cast"]);
    ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == orig_op_to_count["MatMul"]);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnTwoTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_2_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "FusedMatMul");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnThreeTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_3_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "FusedMatMul");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulNoFusionOnInvalidInput) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_perm.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_default_perm.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_datatype_int32.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_datatype_int64.onnx",
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], 1);
    ASSERT_EQ(op_to_count["MatMul"], 1);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 0);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionFromTransposeMatMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_from_transpose_scale_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  float expected_alpha;
  {
    auto transpose_scale_matmul_node =
        std::find_if(
            graph.Nodes().cbegin(), graph.Nodes().cend(),
            [](const Node& node) { return node.Name() == "FusedMatMul"; });
    ASSERT_NE(transpose_scale_matmul_node, graph.Nodes().cend());
    expected_alpha = transpose_scale_matmul_node->GetAttributes().at("alpha").f();
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);

  auto& transpose_scale_matmul_node = *graph.Nodes().begin();
  ASSERT_EQ(transpose_scale_matmul_node.OpType(), "FusedMatMul");
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transB").i()));
  ASSERT_EQ(transpose_scale_matmul_node.GetAttributes().at("alpha").f(), expected_alpha);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionWithPreservedTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_with_preserved_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);

  ASSERT_FALSE(graph.GraphResolveNeeded());
}

TEST_F(GraphTransformationTests, TransposeMatmulTransBatchFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion1.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion2.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion3.onnx",
  };
  const std::vector<std::pair<int64_t, int64_t>> trans_batch_attrs = {
      {1, 0},
      {1, 1},
      {1, 1},
  };
  size_t index = 0;
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], 0);
    ASSERT_EQ(op_to_count["MatMul"], 0);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "FusedMatMul") {
        auto attrs = node.GetAttributes();
        int64_t trans_batch_a = 0;
        if (attrs.find("transBatchA") != attrs.end()) {
          trans_batch_a = attrs.at("transBatchA").i();
        }
        int64_t trans_batch_b = 0;
        if (attrs.find("transBatchB") != attrs.end()) {
          trans_batch_b = attrs.at("transBatchB").i();
        }
        ASSERT_EQ(trans_batch_a, trans_batch_attrs[index].first);
        ASSERT_EQ(trans_batch_b, trans_batch_attrs[index].second);
        break;
      }
    }
    ++index;
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulTransBatchNoFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case1.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case2.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case3.onnx",
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> orig_op_to_count = CountOpsInGraph(graph);

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], orig_op_to_count["Transpose"]);
    ASSERT_EQ(op_to_count["MatMul"], orig_op_to_count["MatMul"]);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], orig_op_to_count["com.microsoft.FusedMatMul"]);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusion_SameInput_gh_issue_24341) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gh_issue_24341.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> orig_op_to_count = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], orig_op_to_count["Transpose"]);
  ASSERT_EQ(op_to_count["MatMul"], orig_op_to_count["MatMul"]);
  ASSERT_EQ(op_to_count["Cast"], orig_op_to_count["Cast"]);
}

TEST_F(GraphTransformationTests, Gemm_LeakyRelu_Fusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GemmActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedGemm"] == 1);
}
#endif

// (A')'B' = AB'
TEST_F(GraphTransformationTests, GemmTransposeFusion2Inputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2inputs_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
}

TEST_F(GraphTransformationTests, GemmTransposeFusionDoesNotFuseIdentityTranspose) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{3, 4}});
    auto* weight = builder.MakeInput<float>({{4, 5}});
    auto* transposed_weight = builder.MakeIntermediate<float>(std::vector<int64_t>{4, 5});
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{3, 5});

    builder.AddNode("Transpose", {weight}, {transposed_weight}).AddAttribute("perm", std::vector<int64_t>{0, 1});
    auto& gemm = builder.AddNode("Gemm", {input, transposed_weight}, {output});
    gemm.AddAttribute("transA", int64_t{0});
    gemm.AddAttribute("transB", int64_t{0});
    gemm.AddAttribute("alpha", 1.0f);
    gemm.AddAttribute("beta", 1.0f);
  };

  auto check_graph = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Transpose"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gemm"] == 1);
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1,
                                        1, check_graph, check_graph));
}

TEST_F(GraphTransformationTests, GemmTransposeFusionDoesNotFuseIdentityTransposeAtOutput) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{4, 3}}, "A");
    auto* weight = builder.MakeInput<float>({{4, 5}}, "B");
    auto* gemm_output = builder.MakeIntermediate<float>(std::vector<int64_t>{3, 5});
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{3, 5});

    auto& gemm = builder.AddNode("Gemm", {input, weight}, {gemm_output});
    gemm.AddAttribute("transA", int64_t{1});
    gemm.AddAttribute("transB", int64_t{0});
    gemm.AddAttribute("alpha", 2.0f);
    gemm.AddAttribute("beta", 3.0f);
    builder.AddNode("Transpose", {gemm_output}, {output}).AddAttribute("perm", std::vector<int64_t>{0, 1});
  };

  auto check_graph = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Transpose"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gemm"] == 1);
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Gemm") {
        TEST_RETURN_IF_NOT(node.GetAttributes().at("transA").i() == 1);
        TEST_RETURN_IF_NOT(node.GetAttributes().at("transB").i() == 0);
        TEST_RETURN_IF_NOT(node.GetAttributes().at("alpha").f() == 2.0f);
        TEST_RETURN_IF_NOT(node.GetAttributes().at("beta").f() == 3.0f);
        TEST_RETURN_IF_NOT(node.InputDefs()[0]->Name() == "A");
        TEST_RETURN_IF_NOT(node.InputDefs()[1]->Name() == "B");
      }
    }
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1,
                                        1, check_graph, check_graph));
}

TEST_F(GraphTransformationTests, GemmTransposeFusionPreservesIdentityOutputWhenFusingInput) {
  for (bool transpose_input_b : {false, true}) {
    SCOPED_TRACE(transpose_input_b);
    auto build_test_case = [transpose_input_b](ModelTestBuilder& builder) {
      auto* input = builder.MakeInput<float>(transpose_input_b ? std::vector<int64_t>{3, 4}
                                                               : std::vector<int64_t>{4, 3},
                                             "A");
      auto* weight = builder.MakeInput<float>(transpose_input_b ? std::vector<int64_t>{5, 4}
                                                                : std::vector<int64_t>{4, 5},
                                              "B");
      auto* transposed_input = builder.MakeIntermediate();
      auto* gemm_output = builder.MakeIntermediate<float>(std::vector<int64_t>{3, 5});
      auto* output = builder.MakeOutput<float>(std::vector<int64_t>{3, 5});

      builder.AddNode("Transpose", {transpose_input_b ? weight : input}, {transposed_input})
          .AddAttribute("perm", std::vector<int64_t>{1, 0});
      auto& gemm = builder.AddNode("Gemm", {transpose_input_b ? input : transposed_input, transpose_input_b ? transposed_input : weight},
                                   {gemm_output});
      gemm.AddAttribute("transA", int64_t{0});
      gemm.AddAttribute("transB", int64_t{0});
      gemm.AddAttribute("alpha", 2.0f);
      gemm.AddAttribute("beta", 3.0f);
      builder.AddNode("Transpose", {gemm_output}, {output}).AddAttribute("perm", std::vector<int64_t>{0, 1});
    };

    auto check_graph = [transpose_input_b](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Transpose"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gemm"] == 1);
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "Gemm") {
          TEST_RETURN_IF_NOT(node.GetAttributes().at("transA").i() == (transpose_input_b ? 0 : 1));
          TEST_RETURN_IF_NOT(node.GetAttributes().at("transB").i() == (transpose_input_b ? 1 : 0));
          TEST_RETURN_IF_NOT(node.GetAttributes().at("alpha").f() == 2.0f);
          TEST_RETURN_IF_NOT(node.GetAttributes().at("beta").f() == 3.0f);
          TEST_RETURN_IF_NOT(node.InputDefs()[0]->Name() == "A");
          TEST_RETURN_IF_NOT(node.InputDefs()[1]->Name() == "B");
        } else if (node.OpType() == "Transpose") {
          const auto perm = RetrieveValues<int64_t>(node.GetAttributes().at("perm"));
          TEST_RETURN_IF_NOT(perm == std::vector<int64_t>({0, 1}));
          TEST_RETURN_IF_NOT(graph.NodeProducesGraphOutput(node));
        }
      }
      return Status::OK();
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer),
                                          TransformerLevel::Level1, 1, nullptr, check_graph));
  }
}

TEST_F(GraphTransformationTests, GemmTransposeFusionPreservesPublicInputTransposeWhenFusingOtherInput) {
  for (bool public_input_a : {false, true}) {
    for (bool initial_trans_a : {false, true}) {
      for (bool initial_trans_b : {false, true}) {
        SCOPED_TRACE(testing::Message() << "public_input_a=" << public_input_a << ", transA=" << initial_trans_a
                                        << ", transB=" << initial_trans_b);

        std::string public_transpose_output_name;
        auto build_test_case = [public_input_a, initial_trans_a, initial_trans_b,
                                &public_transpose_output_name](ModelTestBuilder& builder) {
          const std::vector<int64_t> input_a_shape = initial_trans_a ? std::vector<int64_t>{3, 4}
                                                                     : std::vector<int64_t>{4, 3};
          const std::vector<int64_t> input_b_shape = initial_trans_b ? std::vector<int64_t>{4, 5}
                                                                     : std::vector<int64_t>{5, 4};
          const std::vector<int64_t> transpose_a_shape = initial_trans_a ? std::vector<int64_t>{4, 3}
                                                                         : std::vector<int64_t>{3, 4};
          const std::vector<int64_t> transpose_b_shape = initial_trans_b ? std::vector<int64_t>{5, 4}
                                                                         : std::vector<int64_t>{4, 5};
          auto* input_a = builder.MakeInput<float>(input_a_shape, "A");
          auto* input_b = builder.MakeInput<float>(input_b_shape, "B");
          auto* transpose_a_output = public_input_a ? builder.MakeOutput<float>(transpose_a_shape)
                                                    : builder.MakeIntermediate<float>(transpose_a_shape);
          auto* transpose_b_output = public_input_a ? builder.MakeIntermediate<float>(transpose_b_shape)
                                                    : builder.MakeOutput<float>(transpose_b_shape);

          builder.AddNode("Transpose", {input_a}, {transpose_a_output})
              .AddAttribute("perm", std::vector<int64_t>{1, 0});
          builder.AddNode("Transpose", {input_b}, {transpose_b_output})
              .AddAttribute("perm", std::vector<int64_t>{1, 0});
          public_transpose_output_name = (public_input_a ? transpose_a_output : transpose_b_output)->Name();

          auto* output = builder.MakeOutput<float>(std::vector<int64_t>{3, 5});
          auto& gemm = builder.AddNode("Gemm", {transpose_a_output, transpose_b_output}, {output});
          gemm.AddAttribute("transA", static_cast<int64_t>(initial_trans_a));
          gemm.AddAttribute("transB", static_cast<int64_t>(initial_trans_b));
          gemm.AddAttribute("alpha", 1.0f);
          gemm.AddAttribute("beta", 1.0f);
        };

        auto check_graph = [public_input_a, initial_trans_a, initial_trans_b,
                            &public_transpose_output_name](Graph& graph) {
          auto op_counts = CountOpsInGraph(graph);
          TEST_RETURN_IF_NOT(op_counts["Transpose"] == 1);
          TEST_RETURN_IF_NOT(op_counts["Gemm"] == 1);

          const Node* gemm_node = nullptr;
          const Node* public_transpose = nullptr;
          for (const auto& node : graph.Nodes()) {
            if (node.OpType() == "Gemm") {
              gemm_node = &node;
            } else if (node.OpType() == "Transpose") {
              public_transpose = &node;
              TEST_RETURN_IF_NOT(graph.NodeProducesGraphOutput(node));
            }
          }
          TEST_RETURN_IF_NOT(gemm_node != nullptr);
          TEST_RETURN_IF_NOT(public_transpose != nullptr);

          const size_t public_input_index = public_input_a ? 0 : 1;
          const size_t fused_input_index = 1 - public_input_index;
          const std::string expected_public_input_name = public_input_a ? "A" : "B";
          const std::string expected_fused_input_name = public_input_a ? "B" : "A";
          const bool expected_trans_a = public_input_a ? initial_trans_a : !initial_trans_a;
          const bool expected_trans_b = public_input_a ? !initial_trans_b : initial_trans_b;
          TEST_RETURN_IF_NOT(gemm_node->InputDefs().size() == 2);
          TEST_RETURN_IF_NOT(public_transpose->InputDefs()[0]->Name() == expected_public_input_name);
          TEST_RETURN_IF_NOT(public_transpose->OutputDefs()[0]->Name() == public_transpose_output_name);
          TEST_RETURN_IF_NOT(gemm_node->InputDefs()[public_input_index]->Name() == public_transpose_output_name);
          TEST_RETURN_IF_NOT(gemm_node->InputDefs()[fused_input_index]->Name() == expected_fused_input_name);
          TEST_RETURN_IF_NOT(gemm_node->GetAttributes().at("transA").i() ==
                             static_cast<int64_t>(expected_trans_a));
          TEST_RETURN_IF_NOT(gemm_node->GetAttributes().at("transB").i() ==
                             static_cast<int64_t>(expected_trans_b));
          return Status::OK();
        };

        auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
        ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
        ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer),
                                              TransformerLevel::Level1, 1, nullptr, check_graph));
      }
    }
  }
}

TEST_F(GraphTransformationTests, GemmTransposeFusionPreservesPublicInputTransposesWhenFusingOutput) {
  std::string transpose_a_output_name;
  std::string transpose_b_output_name;
  std::string final_output_name;
  auto build_test_case = [&transpose_a_output_name, &transpose_b_output_name,
                          &final_output_name](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<float>(std::vector<int64_t>{4, 3}, "A");
    auto* input_b = builder.MakeInput<float>(std::vector<int64_t>{5, 4}, "B");
    auto* transpose_a_output = builder.MakeOutput<float>(std::vector<int64_t>{3, 4});
    auto* transpose_b_output = builder.MakeOutput<float>(std::vector<int64_t>{4, 5});
    transpose_a_output_name = transpose_a_output->Name();
    transpose_b_output_name = transpose_b_output->Name();

    builder.AddNode("Transpose", {input_a}, {transpose_a_output}).AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Transpose", {input_b}, {transpose_b_output}).AddAttribute("perm", std::vector<int64_t>{1, 0});

    auto* gemm_output = builder.MakeIntermediate<float>(std::vector<int64_t>{3, 5});
    auto& gemm = builder.AddNode("Gemm", {transpose_a_output, transpose_b_output}, {gemm_output});
    gemm.AddAttribute("transA", int64_t{0});
    gemm.AddAttribute("transB", int64_t{0});
    gemm.AddAttribute("alpha", 1.0f);
    gemm.AddAttribute("beta", 1.0f);
    auto* output = builder.MakeOutput<float>(std::vector<int64_t>{5, 3});
    final_output_name = output->Name();
    builder.AddNode("Transpose", {gemm_output}, {output}).AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_graph = [&transpose_a_output_name, &transpose_b_output_name, &final_output_name](Graph& graph) {
    auto op_counts = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_counts["Transpose"] == 2);
    TEST_RETURN_IF_NOT(op_counts["Gemm"] == 1);

    const Node* gemm_node = nullptr;
    const Node* transpose_a = nullptr;
    const Node* transpose_b = nullptr;
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Gemm") {
        gemm_node = &node;
      } else if (node.OpType() == "Transpose") {
        TEST_RETURN_IF_NOT(graph.NodeProducesGraphOutput(node));
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 1);
        if (node.InputDefs()[0]->Name() == "A") {
          transpose_a = &node;
          TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Name() == transpose_a_output_name);
        } else if (node.InputDefs()[0]->Name() == "B") {
          transpose_b = &node;
          TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Name() == transpose_b_output_name);
        } else {
          TEST_RETURN_IF_NOT(false);
        }
      }
    }
    TEST_RETURN_IF_NOT(gemm_node != nullptr);
    TEST_RETURN_IF_NOT(transpose_a != nullptr);
    TEST_RETURN_IF_NOT(transpose_b != nullptr);
    TEST_RETURN_IF_NOT(gemm_node->InputDefs().size() == 2);
    TEST_RETURN_IF_NOT(gemm_node->InputDefs()[0]->Name() == transpose_b_output_name);
    TEST_RETURN_IF_NOT(gemm_node->InputDefs()[1]->Name() == transpose_a_output_name);
    TEST_RETURN_IF_NOT(gemm_node->GetAttributes().at("transA").i() == 1);
    TEST_RETURN_IF_NOT(gemm_node->GetAttributes().at("transB").i() == 1);
    TEST_RETURN_IF_NOT(graph.NodeProducesGraphOutput(*gemm_node));
    TEST_RETURN_IF_NOT(gemm_node->OutputDefs().size() == 1);
    TEST_RETURN_IF_NOT(gemm_node->OutputDefs()[0]->Name() == final_output_name);
    return Status::OK();
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, nullptr, check_graph));
}

// (A')'B' = AB' where transpose has multiple consumers
TEST_F(GraphTransformationTests, GemmTransposeFusion2OutputsFromTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2outputs_from_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 1);

  auto gemm_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm/GemmTransposeFusion/"; });

  auto& node = *gemm_node;
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "tp0");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
}

// (A')'B' = AB' and  (B')'C = BC where transpose has multiple consumers
TEST_F(GraphTransformationTests, GemmTransposeFusion2OutputsFromTransposeTo2Gemms) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2outputs_from_transpose_to_2gemms.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 2);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 2);
  ASSERT_EQ(op_to_count["Identity"], 1);

  auto gemm1_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm1/GemmTransposeFusion/"; });

  auto& node1 = *gemm1_node;
  ASSERT_TRUE(node1.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node1.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node1.GetAttributes().at("transB").i()));
  auto new_input_defs1 = node1.InputDefs();
  ASSERT_TRUE(new_input_defs1[0]->Name() == "tp0");
  ASSERT_TRUE(new_input_defs1[1]->Name() == "B");

  auto gemm2_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm2/GemmTransposeFusion/"; });

  auto& node2 = *gemm2_node;
  ASSERT_TRUE(node2.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node2.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node2.GetAttributes().at("transB").i()));
  auto new_input_defs2 = node2.InputDefs();
  ASSERT_TRUE(new_input_defs2[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs2[1]->Name() == "C");
}

// (A'B)' = B'A
TEST_F(GraphTransformationTests, GemmTransposeFusionOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_output_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// ((A')'B')' = BA'
TEST_F(GraphTransformationTests, GemmTransposeFusionInputOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_inputs_output_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 3);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// (A'(B'))' = BA
TEST_F(GraphTransformationTests, GemmTransposeFusionInputOutput2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_inputs_output_transposed_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C)
TEST_F(GraphTransformationTests, GemmSumFusionBasic) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_basic.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C), with attributes
TEST_F(GraphTransformationTests, GemmSumFusionAttributes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_attributes.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 3.5);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Identity(Sum(Gemm(Identity(A), Identity(B), _), Identity(C)) should still fuse Gemm/Sum internally.
TEST_F(GraphTransformationTests, GemmSumFusionInternalNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_internal_nodes.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 4);
  ASSERT_EQ(graph.NumberOfNodes(), 6);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 4);
  ASSERT_EQ(graph.NumberOfNodes(), 5);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
      ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
      ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
      ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);

      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "tp0");
      ASSERT_TRUE(new_input_defs[1]->Name() == "tp1");
      ASSERT_TRUE(new_input_defs[2]->Name() == "tp3");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "tp4");
    }
  }
}

// Sum(Gemm(A, B, C), D) does not perform transform.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionCUsed) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_c_used.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Assert that the Sum still exists. Fusion should not occur with this pattern.
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
      ASSERT_TRUE(new_input_defs[2]->Name() == "C");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "D");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B), C, D) does not perform transform.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionSumMultipleInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_sum_multiple_inputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Assert that the Sum still exists. Fusion should not occur with this pattern.
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      ASSERT_TRUE(new_input_defs[2]->Name() == "D");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C), with broadcast.
TEST_F(GraphTransformationTests, GemmSumFusionBroadcast) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_fusion_broadcast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Sum(Gemm(A, B, _), C), with invalid broadcasting (no fusion performed).
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionBroadcastFailure) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_broadcast_failure.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B, _), C) where intermediate Gemm output is used, so fusion cannot be performed.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionOriginalGemmOutputUsed) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_original_gemm_output_used.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

TEST_F(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(session_object.RegisterGraphTransformer(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(session_object.Initialize());

  NameMLValMap feeds;
  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  OrtValue ml_value_x;

  auto x_f = MLFloat16(1.0f);
  std::vector<int64_t> dims_x = {1, 1, 3, 3};
  std::vector<MLFloat16> values_x;
  for (int i = 0; i < 9; ++i) {
    values_x.push_back(x_f);
  }
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           dims_x, values_x, &ml_value_x);
  feeds.insert(std::make_pair("X", ml_value_x));

  std::vector<std::string> output_names;
  output_names.push_back("PROD");
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));

  auto prod_f = MLFloat16(6.0f);
  std::vector<int64_t> expected_dims_prod = {1, 1, 2, 2};
  std::vector<MLFloat16> expected_values_prod;
  for (int i = 0; i < 4; ++i) {
    expected_values_prod.push_back(prod_f);
  }

  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_prod);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<MLFloat16> found(rtensor.Data<MLFloat16>(),
                                     rtensor.Data<MLFloat16>() + expected_dims_prod.size());
  ASSERT_EQ(expected_values_prod, found);
}

TEST_F(GraphTransformationTests, ReluClip6Fusion) {
  // Clip op schema changed for opset version 11. Until Clip op is updated in ORT hard coding this model to use
  // older opset.
  Model model("ReluClip6Fusion", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 10}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 3 paths in the model, each with Relu followed by Clip
  // One has a Clip with min of 0  (remove Relu)
  // One have a Clip with a min > 1 (remove Relu)
  // One has a Clip with min < 0 (remove Relu and update Clip 'min' to 0)
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &input_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &input_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &input_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu to eliminate", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu to not eliminate", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu to eliminate and update 'min' of following Clip", {&input2}, {&relu2_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with min 0", {&relu0_output}, {&clip0_output});
  clip0.AddAttribute("min", 0.f);
  clip0.AddAttribute("max", 1.f);

  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with min 1", {&relu1_output}, {&clip1_output});
  clip1.AddAttribute("min", 1.f);
  clip1.AddAttribute("max", 1.f);

  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with min -1", {&relu2_output}, {&clip2_output});
  clip2.AddAttribute("min", -1.f);
  clip2.AddAttribute("max", 1.f);

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 3);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);

  // make sure the Clip nodes were updated to have a 'min' >= 0
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Clip") {
      auto* min = graph_utils::GetNodeAttribute(node, "min");
      ASSERT_TRUE(min->f() >= 0.f);
    }
  }
}

// test handling of Clip 11
TEST_F(GraphTransformationTests, ReluClip11Fusion) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 11;
  Model model("ReluClip6Fusion", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              *logger_);  //, true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 11}}, {});
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float16_tensor_type;
  float16_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float16_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 4 paths in the model, each with Relu followed by Clip to test different aspects of Clip 11 handling
  // One has a Clip with mutable 'min' (don't fuse)
  // One has a Clip with constant 'min' < 0 (fuse and update 'min')
  // One has a Clip with constant 'min' > 0 (fuse and leave 'min')
  // One has a Clip with no 'min' (fuse and update to set min to 0 using type info from 'input')
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &float16_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);
  auto& input3 = graph.GetOrCreateNodeArg("input_3", &input_tensor_type);

  auto& min_input_0 = graph.GetOrCreateNodeArg("min_input_0", &input_tensor_type);
  auto& min_input_1 = graph.GetOrCreateNodeArg("min_input_1", &float16_tensor_type);
  auto& min_input_2 = graph.GetOrCreateNodeArg("min_input_2", &input_tensor_type);

  // add initializer for min_input_1 so it's constant
  TensorProto const_min_1;
  Initializer i1(TensorProto_DataType_FLOAT16, "min_input_1", AsSpan<int64_t>({1}));
  i1.data<MLFloat16>()->val = math::floatToHalf(-1.f);
  i1.ToProto(const_min_1);
  graph.AddInitializedTensor(const_min_1);

  TensorProto const_min_2;
  Initializer i2(TensorProto_DataType_FLOAT, "min_input_2", AsSpan<int64_t>({1}));
  *i2.data<float>() = 1.f;
  i2.ToProto(const_min_2);
  graph.AddInitializedTensor(const_min_2);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &float16_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);
  auto& relu3_output = graph.GetOrCreateNodeArg("relu3_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &float16_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);
  auto& clip3_output = graph.GetOrCreateNodeArg("clip3_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu0", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu1", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu2", {&input2}, {&relu2_output});
  graph.AddNode("relu3", "Relu", "Relu3", {&input3}, {&relu3_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with mutable min", {&relu0_output, &min_input_0}, {&clip0_output});
  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with constant min < 0", {&relu1_output, &min_input_1}, {&clip1_output});
  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with constant min > 0", {&relu2_output, &min_input_2}, {&clip2_output});
  auto& clip3 = graph.AddNode("clip3", "Clip", "Clip with no min", {&relu3_output}, {&clip3_output});

  graph.SetInputs({&input0, &input1, &input2, &input3, &min_input_0});
  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 4);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 1) << "All except the first Relu should have been fused";

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Relu") {
      EXPECT_TRUE(node.Name() == "relu0") << "relu0 should be the only Relu node left";
    }

    if (node.OpType() == "Clip") {
      auto* min_input = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());

      if (&node == &clip0) {
        EXPECT_TRUE(min_input == nullptr) << "clip0 should not have been fused as min_input_0 is not constant";
      } else {
        EXPECT_TRUE(min_input != nullptr)
            << node.Name() << " should have been fused and have a constant initializer for 'min'";

        auto type = min_input->data_type();

        if (&node == &clip1) {
          // fusion with float16 data and min set to 0
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT16);
          MLFloat16 value = *Initializer(*min_input, graph.ModelPath()).data<MLFloat16>();
          EXPECT_EQ(math::halfToFloat(value.val), 0.f) << "Min was not 0.f. Got:" << math::halfToFloat(value.val);
        } else if (&node == &clip2) {
          // fusion with float data and min untouched
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 1.0) << "Min should have remained unchanged but is now " << value;
        } else if (&node == &clip3) {
          // fusion with no min so type comes from input
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 0.f) << "Min was not 0.f. Got:" << value;

        } else {
          EXPECT_TRUE(false) << "Unexpected node " << node.Name();
        }
      }
    }
  }
}

TEST_F(GraphTransformationTests, ReluClip11FusionGHIssue9753) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/relu_clip_fusion_gh_issue_9753.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  // The model contains one Relu and one Clip
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Clip"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  // After fusion, the model only contains Clip.
  ASSERT_TRUE(op_to_count["Relu"] == 0);
  ASSERT_TRUE(op_to_count["Clip"] == 1);
}

// Test Reshape Fusion with 2 constant initializers for Concat inputs.
TEST_F(GraphTransformationTests, ReshapeFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 4U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 12);
      EXPECT_EQ(val[3], 64);
    }
  }
}

// Test Reshape Fusion with one constant initializer for Concat inputs.
TEST_F(GraphTransformationTests, ReshapeFusionOneConstTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_one_const.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 768);
    }
  }
}

// Test Reshape Fusion with an internal node being the output of the graph.
TEST_F(GraphTransformationTests, ReshapeFusionInternalNodeIsOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_node_is_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

// Test Reshape Fusion where some of the internal nodes are reused:
// A Shape is used in two Gather's, and the third Gather is the graph output.
TEST_F(GraphTransformationTests, ReshapeFusionInternalReuseTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_nodes_reused.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 5U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 128);
      EXPECT_EQ(val[2], 0);
      EXPECT_EQ(val[3], 0);
      EXPECT_EQ(val[4], -1);
    } else if (node.OpType() == "Shape") {
      EXPECT_EQ(node.Name(), "shape2");
    } else if (node.OpType() == "Gather") {
      EXPECT_EQ(node.Name(), "gather3");
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionGraphInputsTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_with_graph_inputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerSubgraphTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerAppliesTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], 0);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionAnotherGraphInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_input_is_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionOverridableInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_overridable_initializer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_orig, op_to_count);
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphMultipleOutputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_multiple_outputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization applies but certain paths with multiple outputs/graph outputs are not removed.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Slice"], 1);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraph) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

// Regression test: FuseContiguousReshapes must not collapse a chain of Reshapes
// when the inferred output shape contains a literal 0 dim. Doing so would create
// a single Reshape whose shape data contains 0 and (because allowzero defaults
// to 0) be misinterpreted as "copy from input dim", silently producing wrong shape.
// See https://github.com/microsoft/onnxruntime/issues/28348.
TEST_F(GraphTransformationTests, ReshapeFusionContiguousReshapesWithZeroDim) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  Model model("ReshapeFusionContiguousReshapesWithZeroDim", false, ModelMetaData(),
              PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(), *logger_);
  auto& graph = model.MainGraph();

  // X: float[0, 6, 2]  (zero-sized first dim, fully concrete)
  TypeProto x_type;
  x_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(0);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(6);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  TypeProto y_type;
  y_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  auto& X = graph.GetOrCreateNodeArg("X", &x_type);
  auto& mid = graph.GetOrCreateNodeArg("mid", &y_type);
  auto& Y = graph.GetOrCreateNodeArg("Y", &y_type);

  // shape1 = [3, 2, -1]  -> mid shape (3, 2, 0)
  ONNX_NAMESPACE::TensorProto shape1_proto;
  shape1_proto.set_name("shape1");
  shape1_proto.set_data_type(TensorProto_DataType_INT64);
  shape1_proto.add_dims(3);
  for (int64_t v : {3, 2, -1}) shape1_proto.add_int64_data(v);
  graph.AddInitializedTensor(shape1_proto);

  // shape2 = [0, 0, 3] with allowzero=1 -> Y shape (0, 0, 3)
  ONNX_NAMESPACE::TensorProto shape2_proto;
  shape2_proto.set_name("shape2");
  shape2_proto.set_data_type(TensorProto_DataType_INT64);
  shape2_proto.add_dims(3);
  for (int64_t v : {0, 0, 3}) shape2_proto.add_int64_data(v);
  graph.AddInitializedTensor(shape2_proto);

  auto& shape1 = graph.GetOrCreateNodeArg("shape1", nullptr);
  auto& shape2 = graph.GetOrCreateNodeArg("shape2", nullptr);

  graph.AddNode("reshape1", "Reshape", "first reshape", {&X, &shape1}, {&mid});
  auto& reshape2 = graph.AddNode("reshape2", "Reshape", "second reshape (allowzero=1)",
                                 {&mid, &shape2}, {&Y});
  reshape2.AddAttribute("allowzero", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count_before = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_before["Reshape"], 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Fusion must NOT collapse the two reshapes, otherwise the resulting single
  // Reshape would (mis)compute output shape (0, 6, 3) instead of (0, 0, 3).
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Reshape"], 2);

  // Y's inferred shape must remain (0, 0, 3).
  const auto* y_shape = graph.GetNodeArg("Y")->Shape();
  ASSERT_NE(y_shape, nullptr);
  ASSERT_EQ(y_shape->dim_size(), 3);
  EXPECT_EQ(y_shape->dim(0).dim_value(), 0);
  EXPECT_EQ(y_shape->dim(1).dim_value(), 0);
  EXPECT_EQ(y_shape->dim(2).dim_value(), 3);
}

// Execution regression test: a chained Reshape with allowzero=1 on a zero-element tensor
// must produce the correct output shape at runtime.
// Input X: float[0, 8, 2] -> Reshape([4, 2, -1]) -> mid -> Reshape([0, 0, 4], allowzero=1) -> Y
// Expected Y shape: (0, 0, 4).  Without the fix FuseContiguousReshapes would collapse the
// two nodes into one (losing allowzero=1) and emit (0, 8, 4) instead.
// See https://github.com/microsoft/onnxruntime/issues/28348.
TEST_F(GraphTransformationTests, ReshapeFusionContiguousReshapesWithZeroDimExecution) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 18;
  Model model("ReshapeFusionContiguousReshapesWithZeroDimExecution", false, ModelMetaData(),
              PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(), *logger_);
  auto& graph = model.MainGraph();

  // X: float[0, 8, 2]
  TypeProto x_type;
  x_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(0);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(8);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  TypeProto y_type;
  y_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  auto& X = graph.GetOrCreateNodeArg("X", &x_type);
  auto& mid = graph.GetOrCreateNodeArg("mid", &y_type);
  auto& Y = graph.GetOrCreateNodeArg("Y", &y_type);

  // shape1 = [4, 2, -1]  ->  mid shape (4, 2, 0)
  ONNX_NAMESPACE::TensorProto shape1_proto;
  shape1_proto.set_name("shape1");
  shape1_proto.set_data_type(TensorProto_DataType_INT64);
  shape1_proto.add_dims(3);
  for (int64_t v : {4, 2, -1}) shape1_proto.add_int64_data(v);
  graph.AddInitializedTensor(shape1_proto);

  // shape2 = [0, 0, 4] with allowzero=1  ->  Y shape (0, 0, 4)
  ONNX_NAMESPACE::TensorProto shape2_proto;
  shape2_proto.set_name("shape2");
  shape2_proto.set_data_type(TensorProto_DataType_INT64);
  shape2_proto.add_dims(3);
  for (int64_t v : {0, 0, 4}) shape2_proto.add_int64_data(v);
  graph.AddInitializedTensor(shape2_proto);

  auto& shape1 = graph.GetOrCreateNodeArg("shape1", nullptr);
  auto& shape2 = graph.GetOrCreateNodeArg("shape2", nullptr);

  graph.AddNode("reshape1", "Reshape", "first reshape", {&X, &shape1}, {&mid});
  auto& reshape2 = graph.AddNode("reshape2", "Reshape", "second reshape (allowzero=1)",
                                 {&mid, &shape2}, {&Y});
  reshape2.AddAttribute("allowzero", static_cast<int64_t>(1));

  graph.SetInputs({&X});
  graph.SetOutputs({&Y});

  ASSERT_STATUS_OK(graph.Resolve());

  // Serialize and run via InferenceSession to exercise the full execution path.
  auto model_proto = model.ToProto();
  std::string serialized_model;
  ASSERT_TRUE(model_proto.SerializeToString(&serialized_model));

  SessionOptions so;
  InferenceSession session_object{so, GetEnvironment()};
  std::stringstream model_stream(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(model_stream));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Input: zero-element float tensor with shape [0, 8, 2].
  OrtValue input_val;
  std::vector<int64_t> input_dims = {0, 8, 2};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                       input_dims, std::vector<float>(), &input_val);

  NameMLValMap feeds = {{"X", input_val}};
  std::vector<std::string> output_names = {"Y"};
  std::vector<OrtValue> fetches;
  RunOptions run_options;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));

  // Output shape must be (0, 0, 4), not (0, 8, 4).
  ASSERT_EQ(fetches.size(), 1U);
  const auto& output_tensor = fetches[0].Get<Tensor>();
  const TensorShape& output_shape = output_tensor.Shape();
  ASSERT_EQ(output_shape.NumDimensions(), 3U);
  EXPECT_EQ(output_shape[0], 0);
  EXPECT_EQ(output_shape[1], 0);
  EXPECT_EQ(output_shape[2], 4);
}

// Two chained Reshapes where the first Reshape's shape input is computed by a Where node.
// The fusion collapses them into a single Reshape with the constant shape {6} and deletes
// the Where, whose only consumer is the removed shape input.
// See https://github.com/microsoft/onnxruntime/issues/32837.
TEST_F(GraphTransformationTests, ReshapeFusionContiguousReshapesWithNodeProducedShape) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{2, 3}});
    auto* cond_arg = builder.MakeInputBool({2});
    auto* target_shape = builder.MakeInitializer<int64_t>({2}, {2, 3});
    auto* final_shape = builder.MakeInitializer<int64_t>({1}, {6});
    auto* where_out = builder.MakeIntermediate();
    auto* reshape_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    builder.AddNode("Where", {cond_arg, target_shape, target_shape}, {where_out});
    builder.AddNode("Reshape", {input_arg, where_out}, {reshape_out});
    builder.AddNode("Reshape", {reshape_out, final_shape}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Where"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 1);
    // The Where's only consumer is the removed shape input, so the fusion deletes it.
    TEST_RETURN_IF_NOT(op_to_count["Where"] == 0);

    for (const Node& node : graph.Nodes()) {
      if (node.OpType() != "Reshape") {
        continue;
      }
      // The remaining Reshape takes the model input as data and the initializer {6} as shape.
      const auto* shape_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      TEST_RETURN_IF_NOT(shape_proto != nullptr);
      const int64_t fused_first_dim = *Initializer(*shape_proto, graph.ModelPath()).data<int64_t>();
      TEST_RETURN_IF_NOT(shape_proto->dims_size() == 1 && fused_first_dim == 6);
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

// The Where's output feeds both the first Reshape's shape input and an Identity node; the
// fusion keeps the Where because the Identity still consumes it.
// See https://github.com/microsoft/onnxruntime/issues/32837.
TEST_F(GraphTransformationTests, ReshapeFusionContiguousReshapesWithSharedShapeProducer) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{2, 3}});
    auto* cond_arg = builder.MakeInputBool({2});
    auto* target_shape = builder.MakeInitializer<int64_t>({2}, {2, 3});
    auto* final_shape = builder.MakeInitializer<int64_t>({1}, {6});
    auto* where_out = builder.MakeIntermediate();
    auto* reshape_out = builder.MakeIntermediate();
    auto* shape_output_arg = builder.MakeOutput();
    auto* output_arg = builder.MakeOutput();
    builder.AddNode("Where", {cond_arg, target_shape, target_shape}, {where_out});
    builder.AddNode("Identity", {where_out}, {shape_output_arg});
    builder.AddNode("Reshape", {input_arg, where_out}, {reshape_out});
    builder.AddNode("Reshape", {reshape_out, final_shape}, {output_arg});
  };

  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 1);  // chain fused
    TEST_RETURN_IF_NOT(op_to_count["Where"] == 1);    // shared producer kept
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1, 1, [](Graph&) { return Status::OK(); }, post_graph_checker));
}

// Full-session test of a Reshape chain whose first shape input is produced by a Where node.
// Initialization applies the Level1 optimizers (including ReshapeFusion); Run() returns the
// input tensor reshaped to [6].
// Where(cond, target_shape, target_shape) -> Reshape(X, shape1) -> Reshape(r1, shape2) -> Y
// See https://github.com/microsoft/onnxruntime/issues/32837.
TEST_F(GraphTransformationTests, ReshapeFusionContiguousReshapesWithNodeProducedShapeExecution) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 18;
  Model model("ReshapeFusionContiguousReshapesWithNodeProducedShapeExecution", false, ModelMetaData(),
              PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(), *logger_);
  auto& graph = model.MainGraph();

  // X: float[2, 3]
  TypeProto x_type;
  x_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  x_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  // cond: bool[2]
  TypeProto cond_type;
  cond_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  cond_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  TypeProto float_unshaped;  // float with unknown shape (r1, Y)
  float_unshaped.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto int64_unshaped;  // int64 with unknown shape (shape1)
  int64_unshaped.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);

  auto& X = graph.GetOrCreateNodeArg("X", &x_type);
  auto& cond = graph.GetOrCreateNodeArg("cond", &cond_type);
  auto& shape1 = graph.GetOrCreateNodeArg("shape1", &int64_unshaped);
  auto& r1 = graph.GetOrCreateNodeArg("r1", &float_unshaped);
  auto& Y = graph.GetOrCreateNodeArg("Y", &float_unshaped);

  // target_shape = [2, 3]; shape2 = [6]
  ONNX_NAMESPACE::TensorProto target_shape_proto;
  target_shape_proto.set_name("target_shape");
  target_shape_proto.set_data_type(TensorProto_DataType_INT64);
  target_shape_proto.add_dims(2);
  for (int64_t v : {2, 3}) {
    target_shape_proto.add_int64_data(v);
  }
  graph.AddInitializedTensor(target_shape_proto);

  ONNX_NAMESPACE::TensorProto shape2_proto;
  shape2_proto.set_name("shape2");
  shape2_proto.set_data_type(TensorProto_DataType_INT64);
  shape2_proto.add_dims(1);
  shape2_proto.add_int64_data(6);
  graph.AddInitializedTensor(shape2_proto);

  auto& target_shape = graph.GetOrCreateNodeArg("target_shape", nullptr);
  auto& shape2 = graph.GetOrCreateNodeArg("shape2", nullptr);

  graph.AddNode("shape_producer", "Where", "dynamic shape producer",
                {&cond, &target_shape, &target_shape}, {&shape1});
  graph.AddNode("reshape_dynamic", "Reshape", "first reshape", {&X, &shape1}, {&r1});
  graph.AddNode("reshape_static", "Reshape", "second reshape", {&r1, &shape2}, {&Y});

  graph.SetInputs({&X, &cond});
  graph.SetOutputs({&Y});

  ASSERT_STATUS_OK(graph.Resolve());

  // Serialize and run through a real InferenceSession to cover the full optimization and
  // execution path.
  auto model_proto = model.ToProto();
  std::string serialized_model;
  ASSERT_TRUE(model_proto.SerializeToString(&serialized_model));

  SessionOptions so;
  InferenceSession session_object{so, GetEnvironment()};
  std::stringstream model_stream(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(model_stream));
  ASSERT_STATUS_OK(session_object.Initialize());

  std::vector<int64_t> x_dims = {2, 3};
  OrtValue x_val;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                       x_dims, std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, &x_val);
  std::vector<int64_t> cond_dims = {2};
  OrtValue cond_val;
  CreateMLValue<bool>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                      cond_dims, std::vector<bool>{true, false}, &cond_val);

  NameMLValMap feeds = {{"X", x_val}, {"cond", cond_val}};
  std::vector<std::string> output_names = {"Y"};
  std::vector<OrtValue> fetches;
  RunOptions run_options;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));

  ASSERT_EQ(fetches.size(), 1U);
  const auto& output_tensor = fetches[0].Get<Tensor>();
  const TensorShape& output_shape = output_tensor.Shape();
  ASSERT_EQ(output_shape.NumDimensions(), 1U);
  EXPECT_EQ(output_shape[0], 6);
  const float* output_data = output_tensor.Data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(output_data[i], static_cast<float>(i)) << "Element order must be preserved";
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionWithSlice1) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_with_slice1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphNotTriggered) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_not_triggered.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Two of the branches leading to Concat are candidates to trigger the optimization
  // (Shape -> Gather -> Unsqueeze -> Concat).
  // But one of the subgraphs leading to the Concat node will not trigger the optimization
  // as an additional pad value of 1 is inserted thus making the inputs to the Concat -
  // [10], [20], and [1, 30]
  // Since the third branch will match the subgraph fusion, (it has more than 1 value in the tensor)
  // and hence the optimization will not be triggered eventually

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 2);
  ASSERT_EQ(op_to_count["Unsqueeze"], 2);
  ASSERT_EQ(op_to_count["Slice"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Pad"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto == nullptr);  // No initializer as optimizer is not triggered
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithDiv) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_div.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_mul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 3U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionDistilBertTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_distillbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer.size(), 4U);

      const int64_t* val = initializer.data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], -1);
      EXPECT_EQ(val[2], 2);
      EXPECT_EQ(val[3], 4);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusion_Contiguous_Reshape) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{8, 16, 32}});
    auto* shape_initializer_1 = builder.MakeInitializer<int64_t>({4}, {2, 4, 16, 32});
    auto* shape_initializer_2 = builder.MakeInitializer<int64_t>({3}, {2, 64, 32});
    auto* axes_initializer = builder.MakeInitializer<int64_t>({1}, {1});
    auto* reshape_out_1 = builder.MakeIntermediate();
    auto* reshape_out_2 = builder.MakeIntermediate();
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    builder.AddNode("Reshape", {input_arg, shape_initializer_1}, {reshape_out_1});
    builder.AddNode("Reshape", {reshape_out_1, shape_initializer_2}, {reshape_out_2});
    builder.AddNode("Unsqueeze", {reshape_out_2, axes_initializer}, {unsqueeze_out});
    builder.AddNode("Identity", {unsqueeze_out}, {output_arg});
  };

  auto pre_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 0);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

// Test eliminating redundant Concat-Slice pattern.
TEST_F(GraphTransformationTests, ConcatSliceEliminationTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "concat_slice_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConcatSliceElimination>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Slice"] == 0);
}

// Verifies that ConcatSliceElimination correctly defaults axes to {0} and steps to {1}
// when Slice nodes omit the optional axes/steps inputs (opset >= 10).
TEST_F(GraphTransformationTests, ConcatSliceElimination_OpsetGte10_MissingAxesAndSteps) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* init0 = builder.MakeInitializer<float>({2}, {1.0f, 2.0f});
    auto* init1 = builder.MakeInitializer<float>({2}, {3.0f, 4.0f});
    auto* init2 = builder.MakeInitializer<float>({2}, {5.0f, 6.0f});

    auto* concat_out = builder.MakeIntermediate();
    auto& concat_node = builder.AddNode("Concat", {init0, init1, init2}, {concat_out});
    concat_node.AddAttribute("axis", int64_t{0});

    auto* starts0 = builder.MakeInitializer<int64_t>({1}, {int64_t{0}});
    auto* ends0 = builder.MakeInitializer<int64_t>({1}, {int64_t{2}});
    auto* slice0_out = builder.MakeIntermediate();
    builder.AddNode("Slice", {concat_out, starts0, ends0}, {slice0_out});

    auto* starts1 = builder.MakeInitializer<int64_t>({1}, {int64_t{2}});
    auto* ends1 = builder.MakeInitializer<int64_t>({1}, {int64_t{4}});
    auto* slice1_out = builder.MakeIntermediate();
    builder.AddNode("Slice", {concat_out, starts1, ends1}, {slice1_out});

    auto* starts2 = builder.MakeInitializer<int64_t>({1}, {int64_t{4}});
    auto* ends2 = builder.MakeInitializer<int64_t>({1}, {int64_t{6}});
    auto* slice2_out = builder.MakeIntermediate();
    builder.AddNode("Slice", {concat_out, starts2, ends2}, {slice2_out});

    auto* lhs0 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add0_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs0, slice0_out}, {add0_out});

    auto* lhs1 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add1_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs1, slice1_out}, {add1_out});

    auto* lhs2 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add2_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs2, slice2_out}, {add2_out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Concat"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 3);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 3);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Concat"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 3);
    return Status::OK();
  };

  auto transformer = std::make_unique<ConcatSliceElimination>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Same test for opset v1, where axes is an optional attribute on the Slice node.
TEST_F(GraphTransformationTests, ConcatSliceElimination_OpsetV1_MissingAxesAttribute) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* init0 = builder.MakeInitializer<float>({2}, {1.0f, 2.0f});
    auto* init1 = builder.MakeInitializer<float>({2}, {3.0f, 4.0f});
    auto* init2 = builder.MakeInitializer<float>({2}, {5.0f, 6.0f});

    auto* concat_out = builder.MakeIntermediate();
    auto& concat_node = builder.AddNode("Concat", {init0, init1, init2}, {concat_out});
    concat_node.AddAttribute("axis", int64_t{0});

    auto* slice0_out = builder.MakeIntermediate();
    auto& slice0 = builder.AddNode("Slice", {concat_out}, {slice0_out});
    slice0.AddAttribute("starts", std::vector<int64_t>{0});
    slice0.AddAttribute("ends", std::vector<int64_t>{2});

    auto* slice1_out = builder.MakeIntermediate();
    auto& slice1 = builder.AddNode("Slice", {concat_out}, {slice1_out});
    slice1.AddAttribute("starts", std::vector<int64_t>{2});
    slice1.AddAttribute("ends", std::vector<int64_t>{4});

    auto* slice2_out = builder.MakeIntermediate();
    auto& slice2 = builder.AddNode("Slice", {concat_out}, {slice2_out});
    slice2.AddAttribute("starts", std::vector<int64_t>{4});
    slice2.AddAttribute("ends", std::vector<int64_t>{6});

    auto* lhs0 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add0_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs0, slice0_out}, {add0_out});

    auto* lhs1 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add1_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs1, slice1_out}, {add1_out});

    auto* lhs2 = builder.MakeInput<float>({2}, {0.0f, 0.0f});
    auto* add2_out = builder.MakeOutput();
    builder.AddNode("Add", {lhs2, slice2_out}, {add2_out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Concat"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 3);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 3);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Concat"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Add"] == 3);
    return Status::OK();
  };

  auto transformer = std::make_unique<ConcatSliceElimination>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 4, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* ends00 = builder.Make1DInitializer<int64_t>({8, 8});
    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});

    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* ends01 = builder.Make1DInitializer<int64_t>({8, 8});

    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* ends10 = builder.Make1DInitializer<int64_t>({8, 8});

    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});
    auto* ends11 = builder.Make1DInitializer<int64_t>({8, 8});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends00, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends01, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends10, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends11, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [get_op_count](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    const auto op_to_count = CountOpsInGraph(graph);
    if (graph.DomainToVersionMap().at(kOnnxDomain) >= 28) {
      ASSERT_EQ(get_op_count(op_to_count, "Slice"), 4);
      ASSERT_EQ(get_op_count(op_to_count, "Concat"), 1);
      ASSERT_EQ(get_op_count(op_to_count, "SpaceToDepth"), 0);
      return;
    }
    ASSERT_TRUE(op_to_count.count("Slice") == 0 || op_to_count.at("Slice") == 0);
    ASSERT_TRUE(op_to_count.count("Concat") == 0 || op_to_count.at("Concat") == 0);
    ASSERT_EQ(get_op_count(op_to_count, "SpaceToDepth"), 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "SpaceToDepth") {
        const auto* blocksize_attr = graph_utils::GetNodeAttribute(node, "blocksize");
        ASSERT_TRUE(blocksize_attr != nullptr && utils::HasInt(*blocksize_attr) && blocksize_attr->i() == 2);
      }
    }
  };

  for (int opset : {13, 27, 28}) {
    SCOPED_TRACE(opset);
    TransformerTester(build_test_case,
                      check_transformed_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      opset,
                      0.0,
                      0.0,
                      std::make_unique<SliceConcatToSpaceToDepthFusion>());
  }
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionWithConstantNodesTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto make_int64_constant = [&](const std::vector<int64_t>& values) -> NodeArg* {
      ONNX_NAMESPACE::TensorProto tensor_proto;
      tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      if (!values.empty()) {
        tensor_proto.add_dims(gsl::narrow<int64_t>(values.size()));
      }
      utils::SetRawDataInTensorProto(tensor_proto,
                                     reinterpret_cast<const char*>(values.data()),
                                     values.size() * sizeof(int64_t));

      NodeArg* output = builder.MakeIntermediate();
      tensor_proto.set_name(output->Name());
      builder.AddNode("Constant", {}, {output}).AddAttribute("value", tensor_proto);
      return output;
    };

    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);

    auto* axes_hw = make_int64_constant({2, 3});
    auto* steps2 = make_int64_constant({2, 2});
    auto* ends = make_int64_constant({8, 8});

    auto* starts00 = make_int64_constant({0, 0});
    auto* starts01 = make_int64_constant({0, 1});
    auto* starts10 = make_int64_constant({1, 0});
    auto* starts11 = make_int64_constant({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [get_op_count](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    const auto op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count.count("Slice") == 0 || op_to_count.at("Slice") == 0);
    ASSERT_TRUE(op_to_count.count("Concat") == 0 || op_to_count.at("Concat") == 0);
    ASSERT_EQ(get_op_count(op_to_count, "SpaceToDepth"), 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    13,
                    0.0,
                    0.0,
                    std::make_unique<SliceConcatToSpaceToDepthFusion>());
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionWithPermutedBlockOrderTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);

    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice10_out, slice01_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [get_op_count](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    const auto op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count.count("Slice") == 0 || op_to_count.at("Slice") == 0);
    ASSERT_TRUE(op_to_count.count("Concat") == 0 || op_to_count.at("Concat") == 0);
    ASSERT_EQ(get_op_count(op_to_count, "SpaceToDepth"), 1);
    ASSERT_EQ(get_op_count(op_to_count, "Gather"), 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    13,
                    0.0,
                    0.0,
                    std::make_unique<SliceConcatToSpaceToDepthFusion>());
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForDynamicChannelPermutedBlockOrderTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{{1, -1, 8, 8}});

    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice10_out, slice01_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto pre_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    TEST_RETURN_IF(get_op_count(op_to_count, "Gather") != 0);
    return Status::OK();
  };

  auto post_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    TEST_RETURN_IF(get_op_count(op_to_count, "Gather") != 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<SliceConcatToSpaceToDepthFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForSpatialCropTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);

    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({6, 8});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto pre_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto post_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<SliceConcatToSpaceToDepthFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForChannelSliceTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);

    auto* axes_chw = builder.Make1DInitializer<int64_t>({1, 2, 3});
    auto* steps = builder.Make1DInitializer<int64_t>({1, 2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({2, 8, 8});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 0, 1});
    auto* starts10 = builder.Make1DInitializer<int64_t>({0, 1, 0});
    auto* starts11 = builder.Make1DInitializer<int64_t>({0, 1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_chw, steps}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_chw, steps}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_chw, steps}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_chw, steps}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto pre_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto post_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<SliceConcatToSpaceToDepthFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForUnknownRankInputTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{});

    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({8, 8});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto pre_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto post_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<SliceConcatToSpaceToDepthFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForRank5InputTest) {
  auto get_op_count = [](const OpCountMap& op_to_count, std::string_view op_type) {
    const auto it = op_to_count.find(std::string(op_type));
    return it == op_to_count.end() ? 0 : it->second;
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 2, 8, 8}, -1.0f, 1.0f);

    auto* axes_hw = builder.Make1DInitializer<int64_t>({2, 3});
    auto* steps2 = builder.Make1DInitializer<int64_t>({2, 2});
    auto* ends = builder.Make1DInitializer<int64_t>({2, 8});

    auto* starts00 = builder.Make1DInitializer<int64_t>({0, 0});
    auto* starts01 = builder.Make1DInitializer<int64_t>({0, 1});
    auto* starts10 = builder.Make1DInitializer<int64_t>({1, 0});
    auto* starts11 = builder.Make1DInitializer<int64_t>({1, 1});

    auto* slice00_out = builder.MakeIntermediate();
    auto* slice01_out = builder.MakeIntermediate();
    auto* slice10_out = builder.MakeIntermediate();
    auto* slice11_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Slice", {input, starts00, ends, axes_hw, steps2}, {slice00_out});
    builder.AddNode("Slice", {input, starts01, ends, axes_hw, steps2}, {slice01_out});
    builder.AddNode("Slice", {input, starts10, ends, axes_hw, steps2}, {slice10_out});
    builder.AddNode("Slice", {input, starts11, ends, axes_hw, steps2}, {slice11_out});
    builder.AddNode("Concat", {slice00_out, slice01_out, slice10_out, slice11_out}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto pre_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto post_graph_checker = [get_op_count](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Slice") == 4);
    TEST_RETURN_IF_NOT(op_to_count.at("Concat") == 1);
    TEST_RETURN_IF(get_op_count(op_to_count, "SpaceToDepth") != 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<SliceConcatToSpaceToDepthFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

namespace {

// Builds the initializers shared by the chained-Slice focus tests below.
struct ChainedSliceInitializers {
  NodeArg* axis_h;
  NodeArg* axis_w;
  NodeArg* step2;
  NodeArg* start0;
  NodeArg* start1;
  NodeArg* end_full;
};

ChainedSliceInitializers MakeChainedSliceInitializers(ModelTestBuilder& builder, int64_t extent) {
  return {builder.Make1DInitializer<int64_t>({2}),
          builder.Make1DInitializer<int64_t>({3}),
          builder.Make1DInitializer<int64_t>({2}),
          builder.Make1DInitializer<int64_t>({0}),
          builder.Make1DInitializer<int64_t>({1}),
          builder.Make1DInitializer<int64_t>({extent})};
}

int GetOpCountOrZero(const OpCountMap& op_to_count, std::string_view op_type) {
  const auto it = op_to_count.find(std::string(op_type));
  return it == op_to_count.end() ? 0 : it->second;
}

}  // namespace

// A focus layer is commonly exported as two chained Slice nodes per branch, one
// striding height and one striding width, rather than a single Slice striding
// both axes. The composed phase is still a blocksize=2 space-to-depth, so the
// chain must fuse.
TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionChainedSlicesTest) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);
    const auto init = MakeChainedSliceInitializers(builder, 8);

    auto slice_height_then_width = [&](NodeArg* h_start, NodeArg* w_start) {
      auto* h_out = builder.MakeIntermediate();
      auto* w_out = builder.MakeIntermediate();
      builder.AddNode("Slice", {input, h_start, init.end_full, init.axis_h, init.step2}, {h_out});
      builder.AddNode("Slice", {h_out, w_start, init.end_full, init.axis_w, init.step2}, {w_out});
      return w_out;
    };

    // Canonical block order, so no channel permutation is required.
    auto* branch00 = slice_height_then_width(init.start0, init.start0);
    auto* branch01 = slice_height_then_width(init.start0, init.start1);
    auto* branch10 = slice_height_then_width(init.start1, init.start0);
    auto* branch11 = slice_height_then_width(init.start1, init.start1);

    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Concat", {branch00, branch01, branch10, branch11}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Slice"), 0);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Concat"), 0);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "SpaceToDepth"), 1);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Gather"), 0);
  };

  TransformerTester(build_test_case, check_transformed_graph, TransformerLevel::Default,
                    TransformerLevel::Level1, 13, 0.0, 0.0,
                    std::make_unique<SliceConcatToSpaceToDepthFusion>());
}

// The height Slice is normally shared by two width branches, so the graph holds
// six Slice nodes rather than eight and the upstream node has two consumers.
// This is the shape a YOLO focus layer actually exports, including its
// non-canonical (0,0) (1,0) (0,1) (1,1) block order, which needs a Gather to
// restore the channel order that Concat produced.
TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionSharedHeightSliceTest) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);
    const auto init = MakeChainedSliceInitializers(builder, 8);

    auto slice_height = [&](NodeArg* h_start) {
      auto* h_out = builder.MakeIntermediate();
      builder.AddNode("Slice", {input, h_start, init.end_full, init.axis_h, init.step2}, {h_out});
      return h_out;
    };
    auto slice_width = [&](NodeArg* h_out, NodeArg* w_start) {
      auto* w_out = builder.MakeIntermediate();
      builder.AddNode("Slice", {h_out, w_start, init.end_full, init.axis_w, init.step2}, {w_out});
      return w_out;
    };

    auto* top = slice_height(init.start0);
    auto* bottom = slice_height(init.start1);

    auto* top_left = slice_width(top, init.start0);
    auto* top_right = slice_width(top, init.start1);
    auto* bottom_left = slice_width(bottom, init.start0);
    auto* bottom_right = slice_width(bottom, init.start1);

    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Concat", {top_left, bottom_left, top_right, bottom_right}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_to_count = CountOpsInGraph(session.GetGraph());
    // Both the width Slices and the shared height Slices must be gone.
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Slice"), 0);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Concat"), 0);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "SpaceToDepth"), 1);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Gather"), 1);
  };

  TransformerTester(build_test_case, check_transformed_graph, TransformerLevel::Default,
                    TransformerLevel::Level1, 13, 0.0, 0.0,
                    std::make_unique<SliceConcatToSpaceToDepthFusion>());
}

// A chain that strides the same axis twice is a 4x height decimation, not a
// space-to-depth, and must be left alone.
TEST_F(GraphTransformationTests, SliceConcatToSpaceToDepthFusionNotTriggeredForRepeatedAxisChainTest) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 8, 8}, -1.0f, 1.0f);
    const auto init = MakeChainedSliceInitializers(builder, 8);

    auto slice_height_twice = [&](NodeArg* first_start, NodeArg* second_start) {
      auto* first_out = builder.MakeIntermediate();
      auto* second_out = builder.MakeIntermediate();
      builder.AddNode("Slice", {input, first_start, init.end_full, init.axis_h, init.step2}, {first_out});
      builder.AddNode("Slice", {first_out, second_start, init.end_full, init.axis_h, init.step2}, {second_out});
      return second_out;
    };

    auto* branch00 = slice_height_twice(init.start0, init.start0);
    auto* branch01 = slice_height_twice(init.start0, init.start1);
    auto* branch10 = slice_height_twice(init.start1, init.start0);
    auto* branch11 = slice_height_twice(init.start1, init.start1);

    auto* concat_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Concat", {branch00, branch01, branch10, branch11}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Identity", {concat_out}, {output});
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Slice"), 8);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "Concat"), 1);
    ASSERT_EQ(GetOpCountOrZero(op_to_count, "SpaceToDepth"), 0);
  };

  TransformerTester(build_test_case, check_transformed_graph, TransformerLevel::Default,
                    TransformerLevel::Level1, 13, 0.0, 0.0,
                    std::make_unique<SliceConcatToSpaceToDepthFusion>());
}

TEST_F(GraphTransformationTests, ExpandElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "expand_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 6);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ExpandElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 3);
}

TEST_F(GraphTransformationTests, CastEliminationSimple) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "cast_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 7);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<CastElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 4);
}

TEST_F(GraphTransformationTests, CastChainEliminationRepeatedPattern) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "cast_elimination_complex.onnx";

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 7);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<CastChainElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 3);
}

TEST_F(GraphTransformationTests, PreShapeNodeElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "pre_shape_node_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 3);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PreShapeNodeElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Cast"] == 2);

  // Assert that the remaining "Cast" nodes have different names than "cast2"
  bool names_are_different = true;
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Cast") {
      const std::string& node_name = node.Name();
      if (node_name == "cast") {
        names_are_different = false;
        break;
      }
    }
  }

  ASSERT_TRUE(names_are_different);

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
    return Status::OK();
  };

  // cast is the first node.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* cast_out = builder.MakeIntermediate();
      auto* shape_out = builder.MakeIntermediate();
      auto* output = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Shape", {cast_out}, {shape_out});
      builder.AddNode("Identity", {shape_out}, {output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PreShapeNodeElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

#ifndef DISABLE_CONTRIB_OPS

static void ValidateAttention(Graph& graph) {
  // Validate the merged weights (initializer) input for Attention node.
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Attention") {
      int64_t expected_heads = 2;
      ASSERT_TRUE(optimizer_utils::IsAttributeWithExpectedValue(node, "num_heads", expected_heads));

      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      Initializer initializer(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer.size(), 192U);

      // Validate two rows (2x24 items) for sanity check.
      std::vector<double> expected_value = {
          -0.10791015625,
          -0.04193115234375,
          0.09051513671875,
          0.025787353515625,
          -0.11572265625,
          -0.126953125,
          -0.043304443359375,
          -0.02984619140625,
          0.022125244140625,
          -0.017730712890625,
          -0.03265380859375,
          -0.05108642578125,
          0.0423583984375,
          0.112060546875,
          0.080810546875,
          0.09375,
          -0.03643798828125,
          0.02862548828125,
          0.039764404296875,
          0.06097412109375,
          -0.002288818359375,
          -0.10797119140625,
          -0.01171875,
          0.041717529296875,

          0.033538818359375,
          -0.05755615234375,
          -0.04986572265625,
          -0.01558685302734375,
          -0.0352783203125,
          0.03546142578125,
          0.05218505859375,
          0.005565643310546875,
          -0.043182373046875,
          -0.05010986328125,
          -0.063720703125,
          -0.00824737548828125,
          0.1492919921875,
          0.048431396484375,
          -0.0482177734375,
          -0.1123046875,
          0.032196044921875,
          0.0135650634765625,
          0.020233154296875,
          -0.05084228515625,
          -0.011260986328125,
          -0.1241455078125,
          -0.0101165771484375,
          -0.00490570068359375};

      const float* data = initializer.data<float>();
      for (size_t i = 0; i < expected_value.size(); i++) {
        EXPECT_EQ(data[i], static_cast<float>(expected_value[i]));
      }

      tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      Initializer initializer2(graph, *tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer2.size(), 24U);

      std::vector<double> expected_value2 = {
          -0.23681640625,
          -0.16552734375,
          0.2191162109375,
          -0.1756591796875,
          -0.03460693359375,
          -0.05316162109375,
          -0.336181640625,
          -0.253662109375,
          0.0246734619140625,
          0.011993408203125,
          0.0178375244140625,
          0.00998687744140625,
          0.0255126953125,
          0.076416015625,
          -0.040771484375,
          0.0107879638671875,
          -0.005893707275390625,
          -0.00916290283203125,
          0.04541015625,
          0.0159454345703125,
          -0.0029163360595703125,
          -0.03472900390625,
          0.0535888671875,
          0.0091094970703125};

      const float* data2 = initializer2.data<float>();
      for (size_t i = 0; i < expected_value2.size(); i++) {
        EXPECT_EQ(data2[i], static_cast<float>(expected_value2[i]));
      }
    }
  }
}

// Test Attention Fusion with int32 mask
TEST_F(GraphTransformationTests, AttentionFusionInt32Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_int32_mask.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with int64 mask and symbolic batch dimension
TEST_F(GraphTransformationTests, AttentionFusionInt64Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_symbolic_batch.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 1);  // Cast for int64 mask to int32
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with float32 mask and no "cast" node in mask path
TEST_F(GraphTransformationTests, AttentionFusionFloat32Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_mask_no_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["Div"], 0);
  EXPECT_EQ(op_to_count["Sub"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test GPT-2 Attention Fusion with past and unidirectional mask
TEST_F(GraphTransformationTests, AttentionFusionWithPastAndUnidirMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_past_unidir.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (p_node->OpType().compare("Attention") == 0) {
      EXPECT_EQ(p_node->GetAttributes().at("unidirectional").i(), 1);
    }
  }
}

// Test Attention Fusion with past but no unidirectional mask
TEST_F(GraphTransformationTests, AttentionFusionWithPastAndNoUnidirMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_past_no_unidir.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (p_node->OpType().compare("Attention") == 0) {
      EXPECT_EQ(p_node->GetAttributes().at("unidirectional").i(), 0);
    }
  }
}

// Test GPT-2 Attention Fusion with float32 mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastAndMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_past_mask_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastNoMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_past_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask and past state
TEST_F(GraphTransformationTests, AttentionFusionGPTTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

TEST_F(GraphTransformationTests, AttentionFusionDistilBertTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_distilbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Concat"], 0);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Shape"], 0);
}

// These tests verify that attention fusions fire at the CURRENT max ONNX opset.
// When the ONNX opset advances (e.g., via submodule update), nodes will report a new
// SinceVersion(). If the optimizer's version lists are not updated, the fusion will
// fail to match and these tests will fail.
//
// To fix: update the EdgeEndToMatch version lists in the indicated optimizer file
// to include the new opset version for each affected op type.
//
// NOTE: GPT-2 and DistilBert attention models use attribute-based Unsqueeze/Squeeze
// in their mask subgraphs (opset <= 12 only). Those patterns cannot be converted to
// current opset without rewriting the mask matching logic. The MobileCLIP test
// (programmatically built) covers the current-opset attention fusion paths.

static int GetCurrentOnnxOpset() {
  const auto& map = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  auto it = map.find(ONNX_NAMESPACE::ONNX_DOMAIN);
  EXPECT_TRUE(it != map.end()) << "ONNX domain not found in OpSchemaRegistry";
  return it != map.end() ? it->second.second : 0;
}

enum class MobileClipProjectionType {
  MatMulAdd,
  GemmWithReshapes,
};

struct MobileClipAttentionShapeConfig {
  int64_t input_channels = 512;
  int64_t hidden_size = 512;
  int64_t num_heads = 16;
  int64_t head_size = 32;
  int64_t qkv_weight_input_dim = 512;
};

static void BuildMobileClipAttentionTestCase(ModelTestBuilder& builder,
                                             MobileClipProjectionType projection_type,
                                             const MobileClipAttentionShapeConfig& shape_config = {},
                                             bool use_non_default_projection_gemm_attributes = false,
                                             bool use_runtime_projection_shape_input = false) {
  const int64_t input_channels = shape_config.input_channels;
  const int64_t hidden_size = shape_config.hidden_size;
  const int64_t num_heads = shape_config.num_heads;
  const int64_t head_size = shape_config.head_size;
  const int64_t qkv_weight_input_dim = shape_config.qkv_weight_input_dim;
  const int64_t qkv_hidden_size = num_heads * head_size;
  const int64_t qkv_output_size = 3 * qkv_hidden_size;

  auto* input_x = builder.MakeInput<float>({1, input_channels, 8, 8}, -1.0f, 1.0f);
  auto* input_skip = builder.MakeInput<float>({1, hidden_size, 8, 8}, -1.0f, 1.0f);

  auto* reshape0_shape = builder.Make1DInitializer<int64_t>({1, input_channels, 64});
  auto* qkv_weight = builder.MakeInitializer<float>({qkv_weight_input_dim, qkv_output_size}, -0.05f, 0.05f);
  auto* qkv_reshape_shape = builder.Make1DInitializer<int64_t>({1, 64, 3, num_heads, head_size});
  auto* split_sizes = builder.Make1DInitializer<int64_t>({1, 1, 1});
  auto* squeeze_axis_0 = builder.Make1DInitializer<int64_t>({0});
  auto* squeeze_axis_2 = builder.Make1DInitializer<int64_t>({2});
  auto* scale = builder.MakeScalarInitializer<float>(1.0f / std::sqrt(static_cast<float>(head_size)));
  auto* reshape2_shape = use_runtime_projection_shape_input
                             ? builder.MakeInput<int64_t>({3}, {1, 64, hidden_size})
                             : builder.Make1DInitializer<int64_t>({1, 64, hidden_size});
  auto* proj_gemm_input_shape = builder.Make1DInitializer<int64_t>({64, hidden_size});
  auto* proj_weight = builder.MakeInitializer<float>({hidden_size, hidden_size}, -0.05f, 0.05f);
  auto* proj_bias = builder.MakeInitializer<float>({hidden_size}, -0.02f, 0.02f);
  auto* proj_gemm_output_shape = builder.Make1DInitializer<int64_t>({1, 64, hidden_size});
  auto* reshape3_shape = builder.Make1DInitializer<int64_t>({1, hidden_size, 8, 8});
  auto* layer_scale = builder.MakeInitializer<float>({hidden_size, 1, 1}, 0.9f, 1.1f);

  auto* reshape0_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, input_channels, 64});
  auto* transpose0_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, input_channels});
  auto* qkv_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, qkv_output_size});
  auto* qkv_reshape_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, 3, num_heads, head_size});
  auto* split_q = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, 1, num_heads, head_size});
  auto* split_k = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, 1, num_heads, head_size});
  auto* split_v = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, 1, num_heads, head_size});
  auto* q_transpose_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 1, num_heads, 64, head_size});
  auto* q_squeeze_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, head_size});
  auto* k_squeeze_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, num_heads, head_size});
  auto* k_transpose_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, head_size, 64});
  auto* q_scaled_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, head_size});
  auto* qk_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, 64});
  auto* softmax_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, 64});
  auto* v_transpose_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 1, num_heads, 64, head_size});
  auto* v_squeeze_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, head_size});
  auto* attn_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, num_heads, 64, head_size});
  auto* transpose3_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, num_heads, head_size});
  auto* reshape2_out = use_runtime_projection_shape_input
                           ? builder.MakeIntermediate<float>(std::nullopt)
                           : builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, hidden_size});
  auto* proj_gemm_input_out = builder.MakeIntermediate<float>(std::vector<int64_t>{64, hidden_size});
  auto* proj_gemm_out = builder.MakeIntermediate<float>(std::vector<int64_t>{64, hidden_size});
  auto* proj_linear_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, hidden_size});
  auto* transpose4_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, hidden_size, 64});
  auto* reshape3_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, hidden_size, 8, 8});
  auto* layer_scale_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, hidden_size, 8, 8});
  auto* output = builder.MakeOutput<float>(std::vector<int64_t>{1, hidden_size, 8, 8});

  auto& reshape0 = builder.AddNode("Reshape", std::vector<NodeArg*>{input_x, reshape0_shape}, std::vector<NodeArg*>{reshape0_out});
  reshape0.AddAttribute("allowzero", static_cast<int64_t>(0));

  auto& transpose0 = builder.AddNode("Transpose", std::vector<NodeArg*>{reshape0_out}, std::vector<NodeArg*>{transpose0_out});
  transpose0.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});

  builder.AddNode("MatMul", std::vector<NodeArg*>{transpose0_out, qkv_weight}, std::vector<NodeArg*>{qkv_out});

  auto& qkv_reshape = builder.AddNode("Reshape", std::vector<NodeArg*>{qkv_out, qkv_reshape_shape}, std::vector<NodeArg*>{qkv_reshape_out});
  qkv_reshape.AddAttribute("allowzero", static_cast<int64_t>(0));

  auto& split = builder.AddNode("Split", std::vector<NodeArg*>{qkv_reshape_out, split_sizes}, std::vector<NodeArg*>{split_q, split_k, split_v});
  split.AddAttribute("axis", static_cast<int64_t>(2));

  auto& q_transpose = builder.AddNode("Transpose", std::vector<NodeArg*>{split_q}, std::vector<NodeArg*>{q_transpose_out});
  q_transpose.AddAttribute("perm", std::vector<int64_t>{2, 0, 3, 1, 4});

  builder.AddNode("Squeeze", std::vector<NodeArg*>{q_transpose_out, squeeze_axis_0}, std::vector<NodeArg*>{q_squeeze_out});
  builder.AddNode("Squeeze", std::vector<NodeArg*>{split_k, squeeze_axis_2}, std::vector<NodeArg*>{k_squeeze_out});

  auto& k_transpose = builder.AddNode("Transpose", std::vector<NodeArg*>{k_squeeze_out}, std::vector<NodeArg*>{k_transpose_out});
  k_transpose.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});

  builder.AddNode("Mul", std::vector<NodeArg*>{q_squeeze_out, scale}, std::vector<NodeArg*>{q_scaled_out});
  builder.AddNode("MatMul", std::vector<NodeArg*>{q_scaled_out, k_transpose_out}, std::vector<NodeArg*>{qk_out});

  auto& softmax = builder.AddNode("Softmax", std::vector<NodeArg*>{qk_out}, std::vector<NodeArg*>{softmax_out});
  softmax.AddAttribute("axis", static_cast<int64_t>(-1));

  auto& v_transpose = builder.AddNode("Transpose", std::vector<NodeArg*>{split_v}, std::vector<NodeArg*>{v_transpose_out});
  v_transpose.AddAttribute("perm", std::vector<int64_t>{2, 0, 3, 1, 4});

  builder.AddNode("Squeeze", std::vector<NodeArg*>{v_transpose_out, squeeze_axis_0}, std::vector<NodeArg*>{v_squeeze_out});
  builder.AddNode("MatMul", std::vector<NodeArg*>{softmax_out, v_squeeze_out}, std::vector<NodeArg*>{attn_out});

  auto& transpose3 = builder.AddNode("Transpose", std::vector<NodeArg*>{attn_out}, std::vector<NodeArg*>{transpose3_out});
  transpose3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});

  auto& reshape2 = builder.AddNode("Reshape", std::vector<NodeArg*>{transpose3_out, reshape2_shape}, std::vector<NodeArg*>{reshape2_out});
  reshape2.AddAttribute("allowzero", static_cast<int64_t>(0));

  if (projection_type == MobileClipProjectionType::GemmWithReshapes) {
    auto& proj_gemm_input = builder.AddNode("Reshape", std::vector<NodeArg*>{reshape2_out, proj_gemm_input_shape},
                                            std::vector<NodeArg*>{proj_gemm_input_out});
    proj_gemm_input.AddAttribute("allowzero", static_cast<int64_t>(0));

    auto& proj_gemm = builder.AddNode("Gemm", std::vector<NodeArg*>{proj_gemm_input_out, proj_weight, proj_bias},
                                      std::vector<NodeArg*>{proj_gemm_out});
    if (use_non_default_projection_gemm_attributes) {
      proj_gemm.AddAttribute("transB", static_cast<int64_t>(1));
    }

    auto& proj_gemm_output = builder.AddNode("Reshape", std::vector<NodeArg*>{proj_gemm_out, proj_gemm_output_shape},
                                             std::vector<NodeArg*>{proj_linear_out});
    proj_gemm_output.AddAttribute("allowzero", static_cast<int64_t>(0));
  } else {
    auto* proj_matmul_out = builder.MakeIntermediate<float>(std::vector<int64_t>{1, 64, hidden_size});
    builder.AddNode("MatMul", std::vector<NodeArg*>{reshape2_out, proj_weight}, std::vector<NodeArg*>{proj_matmul_out});
    builder.AddNode("Add", std::vector<NodeArg*>{proj_bias, proj_matmul_out}, std::vector<NodeArg*>{proj_linear_out});
  }

  auto& transpose4 = builder.AddNode("Transpose", std::vector<NodeArg*>{proj_linear_out}, std::vector<NodeArg*>{transpose4_out});
  transpose4.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});

  auto& reshape3 = builder.AddNode("Reshape", std::vector<NodeArg*>{transpose4_out, reshape3_shape}, std::vector<NodeArg*>{reshape3_out});
  reshape3.AddAttribute("allowzero", static_cast<int64_t>(0));

  builder.AddNode("Mul", std::vector<NodeArg*>{layer_scale, reshape3_out}, std::vector<NodeArg*>{layer_scale_out});
  builder.AddNode("Add", std::vector<NodeArg*>{input_skip, layer_scale_out}, std::vector<NodeArg*>{output});
}

static Status CheckMobileClipAttentionFusedGraph(const Graph& graph) {
  auto op_to_count = CountOpsInGraph(graph);
  TEST_RETURN_IF_NOT(op_to_count["com.microsoft.MultiHeadAttention"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Gemm"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Softmax"] == 0);
  TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 0);
  TEST_RETURN_IF_NOT(op_to_count["Split"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 2);
  TEST_RETURN_IF_NOT(op_to_count["Mul"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Add"] == 1);

  int mha_nodes = 0;
  int gemm_nodes = 0;
  int split_nodes = 0;
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "MultiHeadAttention" && node.Domain() == kMSDomain) {
      ++mha_nodes;
      TEST_RETURN_IF_NOT(node.GetAttributes().at("num_heads").i() == 16);
      TEST_RETURN_IF_NOT(std::abs(node.GetAttributes().at("scale").f() - (1.0f / std::sqrt(32.0f))) < 1e-6f);
      TEST_RETURN_IF_NOT(node.OutputDefs().size() == 1);
      TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Shape() != nullptr);
      TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Shape()->dim_size() == 3);
    } else if (node.OpType() == "Split") {
      ++split_nodes;
    } else if (node.OpType() == "Gemm") {
      ++gemm_nodes;
      TEST_RETURN_IF_NOT(node.InputDefs().size() == 3);
      TEST_RETURN_IF_NOT(node.OutputDefs().size() == 1);
      TEST_RETURN_IF_NOT(node.InputDefs()[0]->Shape() != nullptr);
      TEST_RETURN_IF_NOT(node.InputDefs()[0]->Shape()->dim_size() == 2);
      TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Shape() != nullptr);
      TEST_RETURN_IF_NOT(node.OutputDefs()[0]->Shape()->dim_size() == 2);

      const Node* gemm_input_node = graph_utils::GetInputNode(node, 0);
      TEST_RETURN_IF_NOT(gemm_input_node != nullptr);
      TEST_RETURN_IF_NOT(gemm_input_node->OpType() == "Reshape");

      bool has_output_reshape = false;
      for (const Node& consumer : graph.Nodes()) {
        for (const NodeArg* input_def : consumer.InputDefs()) {
          if (input_def != nullptr && input_def->Name() == node.OutputDefs()[0]->Name()) {
            has_output_reshape = consumer.OpType() == "Reshape";
            break;
          }
        }

        if (has_output_reshape) {
          break;
        }
      }

      TEST_RETURN_IF_NOT(has_output_reshape);
    }
  }

  TEST_RETURN_IF_NOT(mha_nodes == 1);
  TEST_RETURN_IF_NOT(gemm_nodes == 1);
  TEST_RETURN_IF_NOT(split_nodes == 1);
  return Status::OK();
}

static Status CheckMobileClipAttentionFusedGraphOnProvider(const Graph& graph, const char* provider) {
  ORT_RETURN_IF_ERROR(CheckMobileClipAttentionFusedGraph(graph));

  for (const Node& node : graph.Nodes()) {
    TEST_RETURN_IF_NOT(node.GetExecutionProviderType() == provider);
  }

  return Status::OK();
}

static void CheckMobileClipAttentionFusedSession(InferenceSessionWrapper& session) {
  ASSERT_STATUS_OK(CheckMobileClipAttentionFusedGraph(session.GetGraph()));
}

static void CheckMobileClipAttentionFusedCudaSession(InferenceSessionWrapper& session) {
  ASSERT_STATUS_OK(CheckMobileClipAttentionFusedGraphOnProvider(session.GetGraph(), kCudaExecutionProvider));
}

static Status CheckMobileClipAttentionUnfusedProjectionGemmGraph(Graph& graph) {
  auto op_to_count = CountOpsInGraph(graph);
  TEST_RETURN_IF_NOT(op_to_count["com.microsoft.MultiHeadAttention"] == 0);
  TEST_RETURN_IF_NOT(op_to_count["Gemm"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Softmax"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 3);
  TEST_RETURN_IF_NOT(op_to_count["Split"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 3);
  TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 6);
  TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 6);
  TEST_RETURN_IF_NOT(op_to_count["Mul"] == 2);
  TEST_RETURN_IF_NOT(op_to_count["Add"] == 1);

  int gemm_nodes = 0;
  for (Node& node : graph.Nodes()) {
    if (node.OpType() != "Gemm") {
      continue;
    }

    ++gemm_nodes;
    const auto& attrs = node.GetAttributes();
    auto trans_b_attr = attrs.find("transB");
    TEST_RETURN_IF_NOT(trans_b_attr != attrs.end());
    TEST_RETURN_IF_NOT(trans_b_attr->second.i() == 1);
  }

  TEST_RETURN_IF_NOT(gemm_nodes == 1);
  return Status::OK();
}

static Status CheckMobileClipAttentionUnfusedMatMulGraph(Graph& graph) {
  auto op_to_count = CountOpsInGraph(graph);
  TEST_RETURN_IF_NOT(op_to_count["com.microsoft.MultiHeadAttention"] == 0);
  TEST_RETURN_IF_NOT(op_to_count["Gemm"] == 0);
  TEST_RETURN_IF_NOT(op_to_count["Softmax"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 3);
  TEST_RETURN_IF_NOT(op_to_count["Split"] == 1);
  TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 4);
  TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 6);
  TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 4);
  TEST_RETURN_IF_NOT(op_to_count["Mul"] == 2);
  TEST_RETURN_IF_NOT(op_to_count["Add"] == 2);
  return Status::OK();
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::MatMulAdd);
  };

  TransformerTester(build_test_case,
                    CheckMobileClipAttentionFusedSession,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    14,
                    1e-3,
                    0.0,
                    std::make_unique<AttentionFusion>());
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaCurrentOpsetTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::MatMulAdd);
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  TransformerTester(build_test_case,
                    CheckMobileClipAttentionFusedSession,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    GetCurrentOnnxOpset(),
                    1e-3,
                    0.0,
                    std::make_unique<AttentionFusion>(),
                    /*add_session_options*/ {},
                    /*disabled_optimizers*/ {},
                    /*ep*/ nullptr,
                    ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false});
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaProjectionGemmTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::GemmWithReshapes);
  };

  TransformerTester(build_test_case,
                    CheckMobileClipAttentionFusedSession,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    14,
                    1e-3,
                    0.0,
                    std::make_unique<AttentionFusion>());
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaCudaEpTest) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::MatMulAdd);
  };

  TransformerTester(build_test_case,
                    CheckMobileClipAttentionFusedCudaSession,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    14,
                    1e-3,
                    0.0,
                    std::make_unique<AttentionFusion>(InlinedHashSet<std::string_view>{kCudaExecutionProvider}),
                    {},
                    {},
                    std::move(cuda_ep));
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaProjectionGemmCudaEpTest) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::GemmWithReshapes);
  };

  TransformerTester(build_test_case,
                    CheckMobileClipAttentionFusedCudaSession,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    14,
                    1e-3,
                    0.0,
                    std::make_unique<AttentionFusion>(InlinedHashSet<std::string_view>{kCudaExecutionProvider}),
                    {},
                    {},
                    std::move(cuda_ep));
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaInvalidQkvWeightShapeTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder,
                                     MobileClipProjectionType::MatMulAdd,
                                     MobileClipAttentionShapeConfig{512, 510, 15, 34, 512});
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::make_unique<AttentionFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, CheckMobileClipAttentionUnfusedMatMulGraph));
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaOverflowingQkvShapeTest) {
  auto capturing_sink = std::make_unique<CapturingSink>();
  const auto* capturing_sink_raw = capturing_sink.get();
  logging::LoggingManager logging_manager(std::move(capturing_sink), logging::Severity::kVERBOSE, false,
                                          logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("MobileClipOverflowTest");

  for (const auto& [num_heads, head_size] :
       {std::pair{std::numeric_limits<int64_t>::max(), int64_t{2}},
        std::pair{std::numeric_limits<int64_t>::max() / 3 + 1, int64_t{1}}}) {
    auto build_test_case = [](ModelTestBuilder& builder) {
      BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::MatMulAdd);
    };

    auto replace_qkv_shape = [num_heads, head_size](Graph& graph) -> Status {
      for (const Node& node : graph.Nodes()) {
        if (node.OpType() != "Reshape" || node.InputDefs().size() < 2) {
          continue;
        }

        const auto& shape_name = node.InputDefs()[1]->Name();
        const ONNX_NAMESPACE::TensorProto* shape = nullptr;
        if (!graph.GetInitializedTensor(shape_name, shape) ||
            shape->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          continue;
        }

        Initializer shape_data{graph, *shape, graph.ModelPath()};
        const auto dimensions = shape_data.DataAsSpan<int64_t>();
        if (dimensions.size() != 5 || dimensions[2] != 3) {
          continue;
        }

        ONNX_NAMESPACE::TensorProto overflow_shape(*shape);
        overflow_shape.clear_raw_data();
        for (int64_t dimension : dimensions) {
          overflow_shape.add_int64_data(dimension);
        }
        overflow_shape.set_int64_data(3, num_heads);
        overflow_shape.set_int64_data(4, head_size);
        graph.RemoveInitializedTensor(shape_name);
        graph.AddInitializedTensor(overflow_shape);
        return Status::OK();
      }

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QKV reshape initializer not found");
    };

    const size_t previous_message_count = capturing_sink_raw->Messages().size();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::make_unique<AttentionFusion>(),
                                          TransformerLevel::Level2, 1, replace_qkv_shape,
                                          CheckMobileClipAttentionUnfusedMatMulGraph));
    const auto& messages = capturing_sink_raw->Messages();
    EXPECT_THAT(std::vector<std::string>(messages.begin() + previous_message_count, messages.end()),
                testing::Contains(testing::HasSubstr("unable to derive num_heads/head_size from qkv reshape initializer")));
  }
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaProjectionGemmNonDefaultAttributesTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::GemmWithReshapes, {}, true);
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::make_unique<AttentionFusion>(),
                                        TransformerLevel::Level2, 1, nullptr,
                                        CheckMobileClipAttentionUnfusedProjectionGemmGraph));
}

TEST_F(GraphTransformationTests, AttentionFusionMobileClipMhaProjectionRewriteFailureLeavesGraphUnfusedTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildMobileClipAttentionTestCase(builder, MobileClipProjectionType::MatMulAdd, {}, false, true);
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::make_unique<AttentionFusion>(),
                                        TransformerLevel::Level2, 1, nullptr,
                                        CheckMobileClipAttentionUnfusedMatMulGraph));
}

// Current-opset regression tests for fusion optimizers.
// These construct minimal graphs at the current ONNX opset and verify the optimizer fires.
// If ONNX bumps an op's since_version, the optimizer's version list will miss it and the test fails.

TEST_F(GraphTransformationTests, GeluFusionCurrentOpsetTest) {
  // Pattern: x -> Div(sqrt2) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5) -> output
  int current_opset = GetCurrentOnnxOpset();
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{2, 3, 64}});
    auto* sqrt2 = builder.MakeInitializer<float>({}, {1.4142135f});
    auto* one = builder.MakeInitializer<float>({}, {1.0f});
    auto* half = builder.MakeInitializer<float>({}, {0.5f});

    auto* div_out = builder.MakeIntermediate();
    auto* erf_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* mul1_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Div", {input, sqrt2}, {div_out});
    builder.AddNode("Erf", {div_out}, {erf_out});
    builder.AddNode("Add", {erf_out, one}, {add_out});
    builder.AddNode("Mul", {input, add_out}, {mul1_out});
    builder.AddNode("Mul", {mul1_out, half}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["Gelu"] == 1 || op_to_count["com.microsoft.Gelu"] == 1) {
      TEST_RETURN_IF_NOT(op_to_count["Div"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Erf"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Add"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Mul"] == 0);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Gelu fusion failed at opset ", current_opset,
                           ". Remaining ops: Div=", op_to_count["Div"],
                           " Erf=", op_to_count["Erf"],
                           " Add=", op_to_count["Add"],
                           " Mul=", op_to_count["Mul"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/gelu_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::make_unique<GeluFusion>(),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, FastGeluFusionCurrentOpsetTest) {
  // FastGelu pattern: x*x*0.044715 + x -> mul(sqrt(2/pi)) -> tanh -> add(1) -> mul(0.5) -> mul(x)
  int current_opset = GetCurrentOnnxOpset();
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{2, 3, 64}});
    auto* coeff = builder.MakeInitializer<float>({}, {0.044715f});
    auto* sqrt2pi = builder.MakeInitializer<float>({}, {0.7978845f});  // sqrt(2/pi)
    auto* one = builder.MakeInitializer<float>({}, {1.0f});
    auto* half = builder.MakeInitializer<float>({}, {0.5f});
    auto* three = builder.MakeInitializer<float>({}, {3.0f});

    auto* pow_out = builder.MakeIntermediate();
    auto* mul1_out = builder.MakeIntermediate();
    auto* add1_out = builder.MakeIntermediate();
    auto* mul2_out = builder.MakeIntermediate();
    auto* tanh_out = builder.MakeIntermediate();
    auto* add2_out = builder.MakeIntermediate();
    auto* mul_half_out = builder.MakeIntermediate();
    auto* mul_final_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Pow", {input, three}, {pow_out});
    builder.AddNode("Mul", {pow_out, coeff}, {mul1_out});
    builder.AddNode("Add", {input, mul1_out}, {add1_out});
    builder.AddNode("Mul", {add1_out, sqrt2pi}, {mul2_out});
    builder.AddNode("Tanh", {mul2_out}, {tanh_out});
    builder.AddNode("Add", {tanh_out, one}, {add2_out});
    builder.AddNode("Mul", {input, half}, {mul_half_out});
    builder.AddNode("Mul", {mul_half_out, add2_out}, {mul_final_out});
    builder.AddNode("Identity", {mul_final_out}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["com.microsoft.FastGelu"] == 1) {
      TEST_RETURN_IF_NOT(op_to_count["Tanh"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Pow"] == 0);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "FastGelu fusion failed at opset ", current_opset,
                           ". Remaining ops: Tanh=", op_to_count["Tanh"],
                           " Pow=", op_to_count["Pow"],
                           " Mul=", op_to_count["Mul"],
                           " Add=", op_to_count["Add"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/fast_gelu_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::make_unique<FastGeluFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, BiasGeluFusionCurrentOpsetTest) {
  // BiasGelu pattern: Add(input, bias) -> Gelu -> output (requires opset >= 20 for ONNX Gelu)
  int current_opset = GetCurrentOnnxOpset();
  if (current_opset < 20) {
    GTEST_SKIP() << "BiasGelu with ONNX Gelu requires opset >= 20";
  }
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{2, 3, 64}});
    auto* bias = builder.MakeInitializer<float>({64}, -0.5f, 0.5f);
    auto* add_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Add", {input, bias}, {add_out});
    builder.AddNode("Gelu", {add_out}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["com.microsoft.BiasGelu"] == 1) {
      TEST_RETURN_IF_NOT(op_to_count["Add"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Gelu"] == 0);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "BiasGelu fusion failed at opset ", current_opset,
                           ". Remaining ops: Add=", op_to_count["Add"],
                           " Gelu=", op_to_count["Gelu"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/bias_gelu_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::make_unique<BiasGeluFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

#if !defined(DISABLE_CONTRIB_OPS)
// Regression test for the WebGPU entry added to the Level-2 GeluFusion allowlist
// (cpu_acl_cuda_dml_webgpu_eps in graph_transformer_utils.cc).
TEST_F(GraphTransformationTests, GeluFusionWebGpu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
#if defined(USE_WEBGPU)
  const std::string expected_ep = kWebGpuExecutionProvider;
#else
  const std::string expected_ep = kCpuExecutionProvider;
#endif
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(expected_ep);
  }

  SessionOptions session_options;
  auto cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const InlinedHashSet<std::string> gelu_transformer_names = {"GeluFusionL1", "GeluFusionL2"};
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  for (auto level : {TransformerLevel::Level1, TransformerLevel::Level2}) {
    for (auto& transformer : optimizer_utils::GenerateTransformers(level, session_options, *cpu_ep, *logger_, {})) {
      if (gelu_transformer_names.count(transformer->Name()) != 0) {
        ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), level));
      }
    }
  }
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["com.microsoft.Gelu"], 1);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Erf"], 0);
  ASSERT_EQ(op_to_count["Add"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);

  const Node* gelu_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gelu" && node.Domain() == kMSDomain) {
      gelu_node = &node;
      break;
    }
  }
  ASSERT_NE(gelu_node, nullptr);
  EXPECT_EQ(gelu_node->GetExecutionProviderType(), expected_ep);
}

// Regression test for the WebGPU entry added to the BiasGeluFusion allowlist.
TEST_F(GraphTransformationTests, BiasGeluFusionWebGpu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
#if defined(USE_WEBGPU)
  const std::string expected_ep = kWebGpuExecutionProvider;
#else
  const std::string expected_ep = kCpuExecutionProvider;
#endif
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(expected_ep);
  }

  SessionOptions session_options;
  auto cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const InlinedHashSet<std::string> gelu_transformer_names = {
      "GeluFusionL1", "GeluFusionL2", "BiasGeluFusion"};
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  for (auto level : {TransformerLevel::Level1, TransformerLevel::Level2}) {
    for (auto& transformer : optimizer_utils::GenerateTransformers(level, session_options, *cpu_ep, *logger_, {})) {
      if (gelu_transformer_names.count(transformer->Name()) != 0) {
        ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), level));
      }
    }
  }
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["com.microsoft.BiasGelu"], 1);
  ASSERT_EQ(op_to_count["com.microsoft.Gelu"], 0);
  ASSERT_EQ(op_to_count["Gelu"], 0);
  ASSERT_EQ(op_to_count["Add"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Erf"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);

  const Node* bias_gelu_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BiasGelu" && node.Domain() == kMSDomain) {
      bias_gelu_node = &node;
      break;
    }
  }
  ASSERT_NE(bias_gelu_node, nullptr);
  EXPECT_EQ(bias_gelu_node->GetExecutionProviderType(), expected_ep);
}

// Regression test for the WebGPU entry added to the Level-2 LayerNormFusion
// allowlist (cpu_acl_cuda_dml_webgpu_eps in graph_transformer_utils.cc).
// The other LayerNorm fusion tests construct the transformer directly with an
// unrestricted provider set, so they pass regardless of that allowlist; this one
// goes through GenerateTransformers so the registration is actually exercised.
TEST_F(GraphTransformationTests, LayerNormFusionWebGpu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_fp16.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kWebGpuExecutionProvider);
  }

  SessionOptions session_options;
  auto cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const InlinedHashSet<std::string> layer_norm_transformer_names = {"LayerNormFusionL1", "LayerNormFusionL2"};
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  for (auto level : {TransformerLevel::Level1, TransformerLevel::Level2}) {
    for (auto& transformer : optimizer_utils::GenerateTransformers(level, session_options, *cpu_ep, *logger_, {})) {
      if (layer_norm_transformer_names.count(transformer->Name()) != 0) {
        ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), level));
      }
    }
  }
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["LayerNormalization"], 1);
  ASSERT_EQ(op_to_count["ReduceMean"], 0);
  ASSERT_EQ(op_to_count["Pow"], 0);
  ASSERT_EQ(op_to_count["Sqrt"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);

  const Node* layer_norm_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      layer_norm_node = &node;
      break;
    }
  }
  ASSERT_NE(layer_norm_node, nullptr);
  EXPECT_EQ(layer_norm_node->GetExecutionProviderType(), kWebGpuExecutionProvider);
  EXPECT_FLOAT_EQ(layer_norm_node->GetAttributes().at("epsilon").f(), static_cast<float>(MLFloat16(1e-4f)));
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST_F(GraphTransformationTests, MatMulAddFusionCurrentOpsetTest) {
  // MatMul + Add -> Gemm fusion
  int current_opset = GetCurrentOnnxOpset();
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{2, 64}});
    auto* weights = builder.MakeInitializer<float>({64, 32}, -1.0f, 1.0f);
    auto* bias = builder.MakeInitializer<float>({32}, -0.5f, 0.5f);
    auto* matmul_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("MatMul", {input, weights}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["Gemm"] == 1) {
      TEST_RETURN_IF_NOT(op_to_count["MatMul"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Add"] == 0);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MatMulAdd fusion failed at opset ", current_opset,
                           ". Remaining ops: MatMul=", op_to_count["MatMul"],
                           " Add=", op_to_count["Add"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/matmul_add_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::make_unique<MatMulAddFusion>(),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, DivMulFusionCurrentOpsetTest) {
  // 1/x * y -> y/x fusion
  int current_opset = GetCurrentOnnxOpset();
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_x = builder.MakeInput<float>({{2, 64}});
    auto* input_y = builder.MakeInput<float>({{2, 64}});
    auto* one = builder.MakeInitializer<float>({}, {1.0f});
    auto* div_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Div", {one, input_x}, {div_out});
    builder.AddNode("Mul", {div_out, input_y}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["Div"] == 1 && op_to_count["Mul"] == 0) {
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "DivMul fusion failed at opset ", current_opset,
                           ". Remaining ops: Div=", op_to_count["Div"],
                           " Mul=", op_to_count["Mul"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/div_mul_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("DivMulFusionCurrentOpset");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<DivMulFusion>()));
  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::move(rule_transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, QuickGeluFusionCurrentOpsetTest) {
  // x * Sigmoid(alpha * x) -> QuickGelu(x, alpha) fusion
  int current_opset = GetCurrentOnnxOpset();
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{2, 3, 64}});
    auto* alpha = builder.MakeInitializer<float>({}, {1.702f});
    auto* mul1_out = builder.MakeIntermediate();
    auto* sigmoid_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("Mul", {input, alpha}, {mul1_out});
    builder.AddNode("Sigmoid", {mul1_out}, {sigmoid_out});
    builder.AddNode("Mul", {input, sigmoid_out}, {output});
  };

  auto post_graph_checker = [current_opset](Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    if (op_to_count["com.microsoft.QuickGelu"] == 1) {
      TEST_RETURN_IF_NOT(op_to_count["Mul"] == 0);
      TEST_RETURN_IF_NOT(op_to_count["Sigmoid"] == 0);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "QuickGelu fusion failed at opset ", current_opset,
                           ". Remaining ops: Mul=", op_to_count["Mul"],
                           " Sigmoid=", op_to_count["Sigmoid"],
                           ". Either update version lists in "
                           "onnxruntime/core/optimizer/quick_gelu_fusion.cc"
                           " or skip this opset in the test if the fusion is not expected to apply.");
  };

  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, current_opset, *logger_,
                                        std::make_unique<QuickGeluFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker,
                                        ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, GeluFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTest_Opset20) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_opset20.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestSwitchOrderFormat2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_0.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_1_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_0_with_bias_use_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 0);
}

TEST_F(GraphTransformationTests, BiasGeluTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 1);
}

TEST_F(GraphTransformationTests, BiasOnnxGeluTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_onnx_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 1);
}

TEST_F(GraphTransformationTests, BiasOnnxFastGeluTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_onnx_fast_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<GeluFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 0);
}

// BiasGelu allows input switching based on input dimensions.
// This test validates the input edges are plugged correct in the optimized graph.
TEST_F(GraphTransformationTests, BiasGeluSwitchedInputOrder) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion_format_2.onnx";

  // create inputs and outputs
  RandomValueGenerator random{};
  NameMLValMap feeds;

  OrtValue mlvalue_b_i;
  std::vector<int64_t> dims_b_i = {3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_b_i,
                       random.Uniform<float>(dims_b_i, 0.0f, 1.0f), &mlvalue_b_i);
  feeds.insert(std::make_pair("B_I", mlvalue_b_i));

  OrtValue mlvalue_a_i;
  std::vector<int64_t> dims_a_i = {3, 512, 3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_a_i,
                       random.Uniform<float>(dims_a_i, 0.0f, 1.0f), &mlvalue_a_i);
  feeds.insert(std::make_pair("A_I", mlvalue_a_i));

  std::vector<std::string> output_names;
  output_names.push_back("C");

  auto run_model_test = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "OptimizerTests";
    InferenceSession session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));
  };

  // run model with and w/o optimizations and compare the results
  std::vector<OrtValue> unoptimized_fetches;
  run_model_test(TransformerLevel::Default, unoptimized_fetches);

  std::vector<OrtValue> optimized_fetches;
  run_model_test(TransformerLevel::MaxLevel, optimized_fetches);

  // Compare results
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 0.0;
  auto ret = CompareOrtValue(optimized_fetches[0], unoptimized_fetches[0],
                             per_sample_tolerance, relative_per_sample_tolerance, false);
  EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
}

static void VerifyGeluApproximation(bool is_enabled, SessionOptions& session_options) {
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  bool has_gelu_approximation = false;
  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, session_options, *e.get(),
                                                            DefaultLoggingManager().DefaultLogger(), {});
  for (auto& transformer : transformers) {
    if (transformer->Name() == "GeluApproximation") {
      has_gelu_approximation = true;
    }
  }

  EXPECT_EQ(has_gelu_approximation, is_enabled);
}

// Test session option configuration for DoubleQDQPairsRemover
TEST_F(GraphTransformationTests, DoubleQDQRemover_SessionOptionConfig) {
  auto verify_session_config = [&](bool is_enabled, SessionOptions& session_option) {
    std::unique_ptr<CPUExecutionProvider> cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    bool has_double_qdq_remover = false;
    auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, session_option, *cpu_ep.get(),
                                                              DefaultLoggingManager().DefaultLogger(), {});
    for (auto& transformer : transformers) {
      if (transformer->Name() == "DoubleQDQPairsRemover") {
        has_double_qdq_remover = true;
      }
    }
    EXPECT_EQ(has_double_qdq_remover, is_enabled);
  };
  SessionOptions session_options;
  // DoubleQDQPairsRemover is enabled by default.
  verify_session_config(true, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableDoubleQDQRemover, "1"));
  verify_session_config(false, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableDoubleQDQRemover, "0"));
  verify_session_config(true, session_options);
}

// Test session option configuration for GeluApproximation
TEST_F(GraphTransformationTests, GeluApproximation_SessionOptionConfig) {
  SessionOptions session_options;

  // GeluApproximation is not enabled by default.
  VerifyGeluApproximation(false, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsEnableGeluApproximation, "1"));
  VerifyGeluApproximation(true, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsEnableGeluApproximation, "0"));
  VerifyGeluApproximation(false, session_options);
}

// Test DoubleQDQPairsRemover to remove unnecessary DQ->Q nodes in the middle
TEST_F(GraphTransformationTests, DoublQDQRemover_RemoveDupQDQ_Float16) {
  auto RunTest = [this](const ORTCHAR_T* model_uri) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DoubleQDQPairsRemover>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    EXPECT_EQ(op_to_count["QuantizeLinear"], 3);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 4);

    std::string dq_scale_name_before_reshape_node;
    std::string zp_name_before_reshape_node;
    std::string dq_scale_name_after_reshape_node;
    std::string zp_name_after_reshape_node;
    for (auto& node : graph.Nodes()) {
      if (node.Name() == "dq_2") {
        dq_scale_name_before_reshape_node = node.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
        zp_name_before_reshape_node = node.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
      }
      if (node.Name() == "q_3") {
        dq_scale_name_after_reshape_node = node.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
        zp_name_after_reshape_node = node.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
      }
    }
    EXPECT_EQ(dq_scale_name_before_reshape_node.empty(), false);
    EXPECT_EQ(zp_name_before_reshape_node.empty(), false);
    EXPECT_EQ(dq_scale_name_before_reshape_node, dq_scale_name_after_reshape_node);
    EXPECT_EQ(zp_name_before_reshape_node, zp_name_after_reshape_node);
  };

  RunTest(MODEL_FOLDER "qdq_optimization/dup_qdq.onnx");
  RunTest(MODEL_FOLDER "qdq_optimization/dup_qdq_float16.onnx");
  RunTest(MODEL_FOLDER "qdq_optimization/dup_qdq_bfloat16.onnx");
}

// Test Gelu -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.Gelu"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

// Test AddGeluFusion -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_Bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu_add_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.BiasGelu"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

// Test MatMul & AddGeluFusion -> MatMul & FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_MatMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu_add_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.BiasGelu"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionSkipsMalformedScaleMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  Node* scale_mul = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() != "Mul" || node.InputDefs().size() != 2) {
      continue;
    }

    for (const NodeArg* input : node.InputDefs()) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *input, 0.7978845834732056f, true)) {
        scale_mul = &node;
        break;
      }
    }

    if (scale_mul != nullptr) {
      break;
    }
  }

  ASSERT_NE(scale_mul, nullptr);
  scale_mul->MutableInputDefs().pop_back();

  GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 0);
}

TEST_F(GraphTransformationTests, FastGeluFusionSkipsMalformedFirstFormulaIntermediateMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  Node* coefficient_mul = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Mul" && node.InputDefs().size() == 2 &&
        (optimizer_utils::IsInitializerWithExpectedValue(graph, *node.InputDefs()[0], 0.044715f, true) ||
         optimizer_utils::IsInitializerWithExpectedValue(graph, *node.InputDefs()[1], 0.044715f, true))) {
      coefficient_mul = &node;
      break;
    }
  }

  ASSERT_NE(coefficient_mul, nullptr);
  ASSERT_EQ(coefficient_mul->GetOutputEdgesCount(), 1);
  Node& intermediate_mul = *graph.GetNode(coefficient_mul->OutputNodesBegin()->Index());
  intermediate_mul.MutableInputDefs().pop_back();

  GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.FastGelu"], 0);
}

TEST_F(GraphTransformationTests, FastGeluFusionSkipsMalformedSecondFormulaIntermediateMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  Node* pow_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Pow") {
      pow_node = &node;
      break;
    }
  }

  ASSERT_NE(pow_node, nullptr);
  ASSERT_EQ(pow_node->GetOutputEdgesCount(), 1);
  Node& intermediate_mul = *graph.GetNode(pow_node->OutputNodesBegin()->Index());
  intermediate_mul.MutableInputDefs().pop_back();

  GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  EXPECT_EQ(CountOpsInGraph(graph)["com.microsoft.FastGelu"], 0);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionWithCastsTest3) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu3_with_casts.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  // ORTModule for gpt2 model has two casts fused into one before FastGeluFusion
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 2);

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Cast"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, CseWithConstantOfShape) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    std::vector<std::variant<int64_t, std::string>> input_shape;
    input_shape.reserve(4);
    input_shape.emplace_back("dim0");
    input_shape.emplace_back(512);
    input_shape.emplace_back(16);
    input_shape.emplace_back("dim3");
    auto* input_arg = builder.MakeSymbolicInput<float>(input_shape);
    auto* shape_out_1 = builder.MakeIntermediate();
    auto* shape_out_2 = builder.MakeIntermediate();
    auto* constant_of_shape_out_1 = builder.MakeIntermediate();
    auto* constant_of_shape_out_2 = builder.MakeIntermediate();
    auto* mul_out_1 = builder.MakeIntermediate();
    auto* mul_out_2 = builder.MakeOutput();
    builder.AddNode("Shape", {input_arg}, {shape_out_1});
    builder.AddNode("Shape", {input_arg}, {shape_out_2});
    TensorProto value_tensor;
    value_tensor.add_dims(1);
    float value = 2.333f;
    value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    utils::SetRawDataInTensorProto(value_tensor, reinterpret_cast<const char*>(&value), sizeof(float));
    builder.AddNode("ConstantOfShape", {shape_out_1}, {constant_of_shape_out_1}).AddAttribute("value", value_tensor);
    builder.AddNode("ConstantOfShape", {shape_out_2}, {constant_of_shape_out_2}).AddAttribute("value", value_tensor);
    builder.AddNode("Mul", {input_arg, constant_of_shape_out_1}, {mul_out_1});
    builder.AddNode("Mul", {mul_out_1, constant_of_shape_out_2}, {mul_out_2});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_map = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_map["Shape"] == 2);
    TEST_RETURN_IF_NOT(op_count_map["ConstantOfShape"] == 2);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    auto op_count_map = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_map["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_count_map["ConstantOfShape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<CommonSubexpressionElimination>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, QuickGelu) {
  // Sigmoid(x*alpha)*x, float
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {input_arg, alpha_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {sigmoid_out, input_arg}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(alpha == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*Sigmoid(alpha*x), MLFloat16
  {
    constexpr float alpha = -1.f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<MLFloat16>({}, {static_cast<MLFloat16>(alpha)});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(alpha == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid's output is consumed by other node.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
      builder.AddNode("Identity", {sigmoid_out}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // First Mul's output is consumed by other node.
  {
    constexpr float alpha = -1.f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<MLFloat16>({}, {static_cast<MLFloat16>(alpha)});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
      builder.AddNode("Identity", {mul_out_0}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid's output is a graph output.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeOutput();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // First Mul's output is a graph output.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeOutput();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid(x)*x, float
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("Sigmoid", {input_arg}, {sigmoid_out});
      builder.AddNode("Mul", {sigmoid_out, input_arg}, {mul_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(1.0f == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*Sigmoid(x), MLFloat16
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("Sigmoid", {input_arg}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(1.0f == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

struct BiasSoftmaxFusionTester {
  std::shared_ptr<Model> p_model_;
  Status model_load_;
  onnxruntime::logging::Logger* logger_;
  onnxruntime::GraphTransformerManager graph_transformation_mgr_;

  bool GetAxis(const std::string op_type, const std::string name, int* axis) {
    for (auto& node : p_model_->MainGraph().Nodes()) {
      if (node.OpType() == op_type) {
        auto& softmax_attr = node.GetAttributes();
        if (softmax_attr.find(name) != softmax_attr.end()) {
          // found axis attribute
          auto& axis_attr = softmax_attr.at(name);
          *axis = (int)axis_attr.i();
          return true;
        }
      }
    }
    // not found
    return false;
  }

  BiasSoftmaxFusionTester(
      const PathString& model_uri,
      onnxruntime::logging::Logger* logger,
      const char* execution_provider = kCudaExecutionProvider) : logger_(logger), graph_transformation_mgr_{5} {
    model_load_ = Model::Load(model_uri, p_model_, nullptr, *logger_);

    // move to cuda since fusion only takes place in that case
    SetExecutionProvider(execution_provider);

    ORT_THROW_IF_ERROR(graph_transformation_mgr_.Register(
        std::make_unique<BiasSoftmaxFusion>(), TransformerLevel::Level2));
  }

  void SetExecutionProvider(const char* ep) {
    for (auto& node : p_model_->MainGraph().Nodes()) {
      node.SetExecutionProviderType(ep);
    }
  }

  void TestFusionOccurs(int expected_axis, bool expected_is_inner_broadcast) {
    ASSERT_STATUS_OK(model_load_);

    ASSERT_STATUS_OK(graph_transformation_mgr_.ApplyTransformers(p_model_->MainGraph(), TransformerLevel::Level2, *logger_));
    std::map<std::string, int> op_to_count = CountOpsInGraph(p_model_->MainGraph());

    ASSERT_EQ(op_to_count["Add"], 0);
    ASSERT_EQ(op_to_count["Softmax"], 0);
    ASSERT_EQ(op_to_count["com.microsoft.BiasSoftmax"], 1);

    int actual_axis = 1, actual_broadcast_type = 1;
    ASSERT_TRUE(GetAxis("BiasSoftmax", "axis", &actual_axis));
    ASSERT_EQ(actual_axis, expected_axis);

    ASSERT_TRUE(GetAxis("BiasSoftmax", "is_inner_broadcast", &actual_broadcast_type));
    ASSERT_EQ(actual_broadcast_type, expected_is_inner_broadcast ? 1 : 0);
  }

  void TestNoFusionOccurs() {
    ASSERT_STATUS_OK(model_load_);

    ASSERT_STATUS_OK(graph_transformation_mgr_.ApplyTransformers(p_model_->MainGraph(), TransformerLevel::Level2, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(p_model_->MainGraph());
    ASSERT_EQ(op_to_count["Add"], 1);
    ASSERT_EQ(op_to_count["Softmax"], 1);
    ASSERT_EQ(op_to_count["com.microsoft.BiasSoftmax"], 0);
  }
};

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_GpuOnly) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get(), kCpuExecutionProvider);
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_Simple_Cuda) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(1, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_Simple_Opset13_DefaultAxis) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple_no_axis_opset13.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(1, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_BFloat16_Input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_bfloat16.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_MiddleOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_ReversedInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones_reversed.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_BadAxis) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones_badaxis.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_AllLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_allleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_SomeLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_someleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, false);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_NoLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_noleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, false);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_OuterBroadcast) {
  auto pre_graph_checker = [&](Graph& graph) {
    for (auto& node : graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "BiasSoftmax") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(attrs.find("is_inner_broadcast") != attrs.end());
        TEST_RETURN_IF_NOT(6 == static_cast<int>(attrs.at("axis").i()));
        TEST_RETURN_IF_NOT(static_cast<int>(attrs.at("is_inner_broadcast").i()) == 0);
      }
    }
    return Status::OK();
  };

  // Input and bias have different ranks.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
      auto* bias_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* add_out = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
      builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Input and bias have same rank.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
      auto* bias_arg = builder.MakeInput<float>({{1, 1, 1, 1, 2, 3, 3, 3}});
      auto* add_out = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
      builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_OpSet13InValidAxis) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
    auto* bias_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
    auto* add_out = builder.MakeIntermediate();
    auto* softmax_out = builder.MakeOutput();

    builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
    builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
  };

  auto pre_graph_checker = [&](Graph& graph) {
    for (auto& node : graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 0);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

static void TestBiasDropoutFusion(const PathString& file_path, const logging::Logger& logger, const int add_count = 0) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasDropoutFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Add"], add_count);
  ASSERT_EQ(op_to_count["Dropout"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.BiasDropout"], 1);
}

TEST_F(GraphTransformationTests, BiasDropoutFusionTest) {
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_mismatch.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_multiple_consumers1.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_multiple_consumers2.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_same_shape_fusion.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_same_shape_fusion.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_same_shape_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_same_shape_fusion_dim_is_param.onnx", *logger_);
}

#ifdef ENABLE_TRAINING
static void TestBitmaskDropoutFusion(const PathString& file_path, bool is_bias_dropout, const logging::Logger& logger,
                                     const int add_count, const int dropout_count, const int bitmask_dropout_count,
                                     const int bias_dropout_count, const int bitmask_bias_dropout_count,
                                     const int dropout_grad_count, const int bitmask_dropout_grad_count) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  if (is_bias_dropout) {
    ASSERT_STATUS_OK(
        graph_transformation_mgr.Register(std::make_unique<BiasDropoutFusion>(), TransformerLevel::Level2));
  } else {
    ASSERT_STATUS_OK(
        graph_transformation_mgr.Register(std::make_unique<BitmaskDropoutReplacement>(), TransformerLevel::Level2));
  }
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Add"], add_count);
  ASSERT_EQ(op_to_count["Dropout"], dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskDropout"], bitmask_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BiasDropout"], bias_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskBiasDropout"], bitmask_bias_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.DropoutGrad"], dropout_grad_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskDropoutGrad"], bitmask_dropout_grad_count);
}

TEST_F(GraphTransformationTests, BitmaskDropoutFusionTest) {
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_dropout_replacement_basic.onnx", false, *logger_, 0, 0, 1, 0, 0,
                           0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_dropout_replacement_multiple_mask_uses.onnx", false, *logger_,
                           0, 1, 0, 0, 0, 1, 0);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_replacement_basic.onnx", false, *logger_, 0, 0, 0,
                           0, 1, 0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_fusion_basic.onnx", true, *logger_, 0, 0, 0, 0, 1,
                           0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_fusion_residual.onnx", true, *logger_, 0, 0, 0, 0,
                           1, 0, 1);
}

TEST_F(GraphTransformationTests, BitmaskDropoutReplacementTransfersWorkspaceReservations) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(
      MODEL_FOLDER "fusion/bitmask_dropout_replacement_basic.onnx",
      model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  constexpr size_t kDropoutReservationBytes = 11;
  constexpr size_t kDropoutGradReservationBytes = 17;
  NodeWorkspaceReservationMap reservations;

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Dropout") {
      node.SetExecutionProviderType(kCudaExecutionProvider);
      reservations.insert_or_assign(
          node.Index(),
          WorkspaceEstimateSelection{
              kDropoutReservationBytes, WorkspaceEstimateSource::kEstimator});
    } else if (node.OpType() == "DropoutGrad") {
      node.SetExecutionProviderType(kCudaExecutionProvider);
      reservations.insert_or_assign(
          node.Index(),
          WorkspaceEstimateSelection{
              kDropoutGradReservationBytes, WorkspaceEstimateSource::kEstimator});
    }
  }
  ASSERT_EQ(reservations.size(), 2U);

  graph.SetNodeReplacementCallback(
      [&reservations](const Graph&,
                      gsl::span<const NodeIndex> source_node_indices,
                      NodeIndex destination_node_index) {
        ConsolidateWorkspaceReservations(
            reservations, source_node_indices, destination_node_index);
      });

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<BitmaskDropoutReplacement>(),
      TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(
      graph, TransformerLevel::Level2, *logger_));
  graph.SetNodeReplacementCallback({});

  const Node* bitmask_dropout_node = nullptr;
  const Node* bitmask_dropout_grad_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "BitmaskDropout") {
      bitmask_dropout_node = &node;
    } else if (node.OpType() == "BitmaskDropoutGrad") {
      bitmask_dropout_grad_node = &node;
    }
  }

  ASSERT_NE(bitmask_dropout_node, nullptr);
  ASSERT_NE(bitmask_dropout_grad_node, nullptr);
  ASSERT_EQ(reservations.size(), 2U);
  EXPECT_EQ(
      reservations.at(bitmask_dropout_node->Index()).bytes,
      kDropoutReservationBytes);
  EXPECT_EQ(
      reservations.at(bitmask_dropout_grad_node->Index()).bytes,
      kDropoutGradReservationBytes);
}

/*
This test build a graph like:
             input0  input1
                  \ /
                  Add
 -----------------|
 |                |
 |             Shape
 |             /   \
 |        Gather0  Gather1
 |           /       \
 |     Unsqueeze0  Unsqueeze1  (Constant Initializer) (Constant Initializer)
 |          \        /               /                  /
 |     	     \      /               /                  /
 |         ConcatTraining   -------       ------------
  \	       /
   \	      /
     Reshape


After fusion, the graph become:
             input0  input1
                  \ /
                  Add     (Constant Initializer)
                   \       /
                    Reshape

*/
TEST_F(GraphTransformationTests, ReshapeFusionOpsetTest) {
  constexpr const int batch_size = 64;
  constexpr const int seq_lenth = 1024;
  constexpr const int hidden_size = 1024;

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.ConcatTraining"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  // Include the current max opset to ensure the fusion stays up-to-date.
  // If this test fails at the current opset, update version lists in
  // onnxruntime/core/optimizer/reshape_fusion.cc.
  std::vector<int> opsets{11, 12, 13, 14, 15, 18, 19, 21, 23, 24, 25};
  int current_opset = GetCurrentOnnxOpset();
  if (std::find(opsets.begin(), opsets.end(), current_opset) == opsets.end()) {
    opsets.push_back(current_opset);
  }

  for (auto& opset : opsets) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto opset_version = builder.DomainToVersionMap().find(kOnnxDomain)->second;
      auto* input_arg0 = builder.MakeInput<float>({{batch_size, seq_lenth, hidden_size}});
      auto* input_arg1 = builder.MakeInput<float>({{hidden_size}});
      auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
      auto* scalar_int_1 = builder.MakeInitializer<int64_t>({}, {1});
      auto* single_value_1d_int_0 = builder.MakeInitializer<int64_t>({1}, {0});
      auto* single_value_1d_int_16 = builder.MakeInitializer<int64_t>({1}, {16});
      auto* single_value_1d_int_64 = builder.MakeInitializer<int64_t>({1}, {64});
      auto* add_out = builder.MakeIntermediate();
      auto* shape_out = builder.MakeIntermediate();
      auto* gather_out_0 = builder.MakeIntermediate();
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* unsqueeze_out_0 = builder.MakeIntermediate();

      auto* unsqueeze_out_1 = builder.MakeIntermediate();
      auto* concattraining1_out = builder.MakeIntermediate();
      auto* concattraining1_length = builder.MakeIntermediate();
      auto* out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg0, input_arg1}, {add_out});
      if (opset_version >= 15) {
        builder.AddNode("Shape", {add_out}, {shape_out}).AddAttribute("start", (int64_t)0);
      } else {
        builder.AddNode("Shape", {add_out}, {shape_out});
      }
      builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out_0});
      builder.AddNode("Gather", {shape_out, scalar_int_1}, {gather_out_1});
      if (opset_version >= 13) {
        builder.AddNode("Unsqueeze", {gather_out_0, single_value_1d_int_0}, {unsqueeze_out_0});
        builder.AddNode("Unsqueeze", {gather_out_1, single_value_1d_int_0}, {unsqueeze_out_1});
      } else {
        builder.AddNode("Unsqueeze", {gather_out_0}, {unsqueeze_out_0}).AddAttribute("axes", std::vector<int64_t>{0});
        builder.AddNode("Unsqueeze", {gather_out_1}, {unsqueeze_out_1}).AddAttribute("axes", std::vector<int64_t>{0});
      }
      builder.AddNode("ConcatTraining", {unsqueeze_out_0, unsqueeze_out_1, single_value_1d_int_16, single_value_1d_int_64},
                      {concattraining1_out, concattraining1_length}, "com.microsoft")
          .AddAttribute("axis", static_cast<int64_t>(0));
      builder.AddNode("Reshape", {add_out, concattraining1_out}, {out});
    };

    // Test that the fusion fires for every opset.
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
    // The opset list includes the current ONNX opset, which may still be under development
    // (e.g. opset 27 in ONNX 1.22). Allow the unreleased opset so the model loads on strict legs.
    // Remove once opset 27 is released. Tracked by #28966.
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker,
                                          ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));

    // For opset >= 15, also test that partial Shape (start=1, end=2) prevents fusion.
    if (opset >= 15) {
      auto build_partial_shape_case = [&](ModelTestBuilder& builder) {
        auto* input_arg0 = builder.MakeInput<float>({{batch_size, seq_lenth, hidden_size}});
        auto* input_arg1 = builder.MakeInput<float>({{hidden_size}});
        auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
        auto* scalar_int_1 = builder.MakeInitializer<int64_t>({}, {1});
        auto* single_value_1d_int_0 = builder.MakeInitializer<int64_t>({1}, {0});
        auto* single_value_1d_int_16 = builder.MakeInitializer<int64_t>({1}, {16});
        auto* single_value_1d_int_64 = builder.MakeInitializer<int64_t>({1}, {64});
        auto* add_out = builder.MakeIntermediate();
        auto* shape_out = builder.MakeIntermediate();
        auto* gather_out_0 = builder.MakeIntermediate();
        auto* gather_out_1 = builder.MakeIntermediate();
        auto* unsqueeze_out_0 = builder.MakeIntermediate();
        auto* unsqueeze_out_1 = builder.MakeIntermediate();
        auto* concattraining1_out = builder.MakeIntermediate();
        auto* concattraining1_length = builder.MakeIntermediate();
        auto* out = builder.MakeOutput();

        builder.AddNode("Add", {input_arg0, input_arg1}, {add_out});
        auto& shape_1 = builder.AddNode("Shape", {add_out}, {shape_out});
        shape_1.AddAttribute("start", (int64_t)1);
        shape_1.AddAttribute("end", (int64_t)2);
        builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out_0});
        builder.AddNode("Gather", {shape_out, scalar_int_1}, {gather_out_1});
        builder.AddNode("Unsqueeze", {gather_out_0, single_value_1d_int_0}, {unsqueeze_out_0});
        builder.AddNode("Unsqueeze", {gather_out_1, single_value_1d_int_0}, {unsqueeze_out_1});
        builder.AddNode("ConcatTraining", {unsqueeze_out_0, unsqueeze_out_1, single_value_1d_int_16, single_value_1d_int_64},
                        {concattraining1_out, concattraining1_length}, "com.microsoft")
            .AddAttribute("axis", static_cast<int64_t>(0));
        builder.AddNode("Reshape", {add_out, concattraining1_out}, {out});
      };

      std::unique_ptr<GraphTransformer> transformer_no_fuse = std::make_unique<ReshapeFusion>();
      // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
      // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships. #28966.
      ASSERT_STATUS_OK(TestGraphTransformer(build_partial_shape_case, opset, *logger_, std::move(transformer_no_fuse),
                                            TransformerLevel::Level1, 1, pre_graph_checker, pre_graph_checker,
                                            ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
    }
  }
}
#endif

// Test reshape fusion when Shape feeds from a tensor with the same symbolic dim_param names
// as the reshape root (e.g., position_ids reshaped using Shape from input_ids in Qwen).
TEST_F(GraphTransformationTests, ReshapeFusionDimParamMatch) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    // input_ids: [batch_size, sequence_length] — symbolic dims
    auto* input_ids = builder.MakeSymbolicInput<float>({std::string("batch_size"), std::string("sequence_length")});
    // position_ids: [batch_size, sequence_length] — same symbolic dims
    auto* position_ids = builder.MakeSymbolicInput<float>({std::string("batch_size"), std::string("sequence_length")});

    auto* shape_out = builder.MakeIntermediate();
    auto* gather_out = builder.MakeIntermediate();
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* out = builder.MakeOutput();

    auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
    auto* unsqueeze_axes = builder.MakeInitializer<int64_t>({1}, {0});
    auto* const_neg1 = builder.MakeInitializer<int64_t>({1}, {-1});

    // Shape(input_ids) -> Gather(0) -> Unsqueeze -> Concat([unsqueeze, -1]) -> Reshape(position_ids)
    builder.AddNode("Shape", {input_ids}, {shape_out});
    builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out});
    builder.AddNode("Unsqueeze", {gather_out, unsqueeze_axes}, {unsqueeze_out});
    builder.AddNode("Concat", {unsqueeze_out, const_neg1}, {concat_out}).AddAttribute("axis", static_cast<int64_t>(0));
    builder.AddNode("Reshape", {position_ids, concat_out}, {out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Test reshape fusion when Shape feeds from a different tensor than the reshape root,
// but the gathered dimension matches by symbolic dim_param (moondream2 vision encoder pattern:
// Shape from pre-projection tensor, Reshape on post-projection output).
TEST_F(GraphTransformationTests, ReshapeFusionCrossTensorDimMatch) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    // pre_proj: [batch_size, seq_len, 768] — pre-projection (layernorm output)
    auto* pre_proj = builder.MakeSymbolicInput<float>(
        {std::string("batch_size"), std::string("seq_len"), int64_t(768)});
    // post_proj: [batch_size, seq_len, 2304] — post-projection (qkv linear output, different last dim)
    auto* post_proj = builder.MakeSymbolicInput<float>(
        {std::string("batch_size"), std::string("seq_len"), int64_t(2304)});

    auto* shape_out = builder.MakeIntermediate();
    auto* gather_out = builder.MakeIntermediate();
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* out = builder.MakeOutput();

    auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
    auto* unsqueeze_axes = builder.MakeInitializer<int64_t>({1}, {0});
    auto* const_neg1 = builder.MakeInitializer<int64_t>({1}, {-1});
    auto* const_3 = builder.MakeInitializer<int64_t>({1}, {3});
    auto* const_256 = builder.MakeInitializer<int64_t>({1}, {256});

    // Shape(pre_proj) -> Gather(0) -> Unsqueeze -> Concat([unsqueeze, -1, 3, 256]) -> Reshape(post_proj)
    builder.AddNode("Shape", {pre_proj}, {shape_out});
    builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out});
    builder.AddNode("Unsqueeze", {gather_out, unsqueeze_axes}, {unsqueeze_out});
    builder.AddNode("Concat", {unsqueeze_out, const_neg1, const_3, const_256}, {concat_out})
        .AddAttribute("axis", static_cast<int64_t>(0));
    builder.AddNode("Reshape", {post_proj, concat_out}, {out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

// Test reshape fusion with GlobalAveragePool passthrough pattern (MobileNetV2):
// Shape is taken from the pre-pool input, Reshape operates on the pool output,
// and only batch dim (gather index 0) is extracted.
TEST_F(GraphTransformationTests, ReshapeFusionGlobalAveragePoolPassthrough) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({{1, 1280, 7, 7}});
    auto* pool_out = builder.MakeIntermediate();
    auto* shape_out = builder.MakeIntermediate();
    auto* gather_out = builder.MakeIntermediate();
    auto* unsqueeze_out = builder.MakeIntermediate();
    auto* concat_out = builder.MakeIntermediate();
    auto* out = builder.MakeOutput();

    auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
    auto* unsqueeze_axes = builder.MakeInitializer<int64_t>({1}, {0});
    auto* const_neg1 = builder.MakeInitializer<int64_t>({1}, {-1});

    // input -> GlobalAveragePool -> pool_out
    builder.AddNode("GlobalAveragePool", {input}, {pool_out});
    // Shape(input) -> Gather(0) -> Unsqueeze -> Concat([unsqueeze, -1]) -> Reshape(pool_out)
    builder.AddNode("Shape", {input}, {shape_out});
    builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out});
    builder.AddNode("Unsqueeze", {gather_out, unsqueeze_axes}, {unsqueeze_out});
    builder.AddNode("Concat", {unsqueeze_out, const_neg1}, {concat_out}).AddAttribute("axis", static_cast<int64_t>(0));
    builder.AddNode("Reshape", {pool_out, concat_out}, {out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["GlobalAveragePool"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["GlobalAveragePool"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Concat"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_ND_bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_ND.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias_No_B_ZP) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_b_no_zp.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/matmul_integer_to_float.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 1);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 3);
  EXPECT_EQ(op_to_count["Add"], 1);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatFusionRejectsOverlappingPatterns) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* lhs_1 = builder.MakeInput<uint8_t>({{2, 2}});
    auto* rhs_1 = builder.MakeInput<uint8_t>({{2, 2}});
    auto* lhs_2 = builder.MakeInput<uint8_t>({{2, 2}});
    auto* rhs_2 = builder.MakeInput<uint8_t>({{2, 2}});
    auto* scale_1 = builder.MakeInput<float>({{1}});
    auto* scale_2 = builder.MakeInput<float>({{1}});
    auto* matmul_1_out = builder.MakeIntermediate();
    auto* cast_1_out = builder.MakeIntermediate();
    auto* scale_product_out = builder.MakeIntermediate();
    auto* shared_mul_out = builder.MakeIntermediate();
    auto* matmul_2_out = builder.MakeIntermediate();
    auto* cast_2_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    builder.AddNode("MatMulInteger", {lhs_1, rhs_1}, {matmul_1_out});
    builder.AddNode("Cast", {matmul_1_out}, {cast_1_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    builder.AddNode("Mul", {scale_1, scale_2}, {scale_product_out});
    builder.AddNode("Mul", {cast_1_out, scale_product_out}, {shared_mul_out});
    builder.AddNode("MatMulInteger", {lhs_2, rhs_2}, {matmul_2_out});
    builder.AddNode("Cast", {matmul_2_out}, {cast_2_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    builder.AddNode("Mul", {cast_2_out, shared_mul_out}, {output});
  };

  auto post_graph_checker = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("com.microsoft.MatMulIntegerToFloat") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMulInteger") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("Cast") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("Mul") == 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<MatMulIntegerToFloatFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(transformer),
                                        TransformerLevel::Level2, 1, {}, post_graph_checker));
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatFusion_Int8Bias_Input0) {
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/matmul_integer_to_float_int8_bias_initializer_index1.onnx");
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  // check graph structure before applying transformations
  const Node* add_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Add") {
      add_node = &node;
      break;
    }
  }

  ASSERT_NE(add_node, nullptr) << "Expected Add node not found.";

  const auto& inputs = add_node->InputDefs();
  ASSERT_EQ(inputs.size(), 2u);

  // Assert bias is in position 1
  EXPECT_EQ(inputs[1]->Name(), "bias") << "Expected bias in input 1 but found in input 0.";

  // Apply the transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
  EXPECT_EQ(op_to_count["Add"], 0);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatFusion_Int8Bias_Input1) {
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/matmul_integer_to_float_int8_bias_initializer_index0.onnx");
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  // check graph structure before applying transformations
  const Node* add_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Add") {
      add_node = &node;
      break;
    }
  }

  ASSERT_NE(add_node, nullptr) << "Expected Add node not found.";

  const auto& inputs = add_node->InputDefs();
  ASSERT_EQ(inputs.size(), 2u);

  // Assert bias is in position 0
  EXPECT_EQ(inputs[0]->Name(), "bias") << "Expected bias in input 0 but found in input 1.";

  // Apply the transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
  EXPECT_EQ(op_to_count["Add"], 0);
}

#ifdef USE_DML
TEST_F(GraphTransformationTests, MatMulIntegerToFloat16Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/matmul_integer_to_float16_int8.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kDmlExecutionProvider);
  }
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatLargeTensorTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/matmul_integer_to_float_large_tensor.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kDmlExecutionProvider);
  }
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 0);
}
#endif  // USE_DML

#endif

#ifndef DISABLE_CONTRIB_OPS
template <typename GraphTransformationCheckFn, typename GraphPreprocessFn>
static void TestMatMulScaleFusion(
    const PathString& model_path, const Logger& logger,
    GraphPreprocessFn graph_preprocess_fn,
    GraphTransformationCheckFn graph_transformation_check_fn,
    const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
    const InlinedHashSet<std::string>& excluded_initializer_names = {}) {
  SCOPED_TRACE(ORT_TSTR("model path: ") + model_path);

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_path, model, nullptr, logger));
  Graph& graph = model->MainGraph();

  graph_preprocess_fn(graph);

  auto original_op_counts = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformer_manager{5};
  ASSERT_STATUS_OK(graph_transformer_manager.Register(
      make_unique<MatMulScaleFusion>(compatible_execution_providers, excluded_initializer_names),
      TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformer_manager.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  auto transformed_op_counts = CountOpsInGraph(graph);

  graph_transformation_check_fn(graph, original_op_counts, transformed_op_counts);
}

template <typename GraphTransformationCheckFn>
static void TestMatMulScaleFusion(
    const PathString& model_path, const Logger& logger,
    GraphTransformationCheckFn graph_transformation_check,
    const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
    const InlinedHashSet<std::string>& excluded_initializer_names = {}) {
  TestMatMulScaleFusion(
      model_path, logger,
      [](Graph&) {}, graph_transformation_check,
      compatible_execution_providers, excluded_initializer_names);
}

TEST_F(GraphTransformationTests, MatMulScaleFusionFusableModels) {
  const std::vector<PathString> one_fusion_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1_out.onnx",
      MODEL_FOLDER "fusion/matmul_scale_transposescalematmul_in0_in1_out.onnx",
  };

  for (const auto& path : one_fusion_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph& graph,
           std::map<std::string, int> original_op_counts,
           std::map<std::string, int> transformed_op_counts) {
          EXPECT_EQ(transformed_op_counts["Mul"], 0);
          EXPECT_EQ(transformed_op_counts["Div"], 0);
          EXPECT_EQ(transformed_op_counts["MatMul"], 0);
          EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 1);

          // check combined scale, individual scales should all have the same value
          constexpr float scale_value = 3.0f;

          const int num_scales =
              original_op_counts["Mul"] + original_op_counts["Div"] + original_op_counts["com.microsoft.FusedMatMul"];

          auto fused_node = std::find_if(
              graph.Nodes().cbegin(), graph.Nodes().cend(),
              [](const Node& node) { return node.OpType() == "FusedMatMul"; });
          ASSERT_NE(fused_node, graph.Nodes().cend());

          auto alpha_attr = fused_node->GetAttributes().find("alpha");
          ASSERT_NE(alpha_attr, fused_node->GetAttributes().end());

          EXPECT_EQ(alpha_attr->second.f(), pow(scale_value, num_scales));
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionTransfersWorkspaceReservations) {
  WorkspaceReservationMap reservations;
  size_t reserved_bytes = 0;
  constexpr size_t bytes_per_node = 64;

  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [&](Graph& graph) {
        auto& graph_reservations = reservations[&graph];
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCudaExecutionProvider);
          if (node.OpType() == "MatMul" || node.OpType() == "Mul" || node.OpType() == "Div") {
            graph_reservations.insert_or_assign(
                node.Index(),
                WorkspaceEstimateSelection{bytes_per_node, WorkspaceEstimateSource::kEstimator});
            reserved_bytes += bytes_per_node;
          }
        }
        graph.SetNodeReplacementCallback(
            [&reservations](const Graph& modified_graph,
                            gsl::span<const NodeIndex> source_node_indices,
                            NodeIndex destination_node_index) {
              auto graph_it = reservations.find(&modified_graph);
              ASSERT_NE(graph_it, reservations.end());
              ConsolidateWorkspaceReservations(
                  graph_it->second, source_node_indices, destination_node_index);
            });
      },
      [&](Graph& graph, const auto&, const auto&) {
        const auto fused_node = std::find_if(
            graph.Nodes().cbegin(), graph.Nodes().cend(),
            [](const Node& node) { return node.OpType() == "FusedMatMul"; });
        ASSERT_NE(fused_node, graph.Nodes().cend());

        const auto& graph_reservations = reservations.at(&graph);
        ASSERT_EQ(graph_reservations.size(), 1U);
        const auto reservation_it = graph_reservations.find(fused_node->Index());
        ASSERT_NE(reservation_it, graph_reservations.end());
        EXPECT_EQ(reservation_it->second.bytes, reserved_bytes);
        graph.SetNodeReplacementCallback({});
      },
      {kCudaExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionUnfusableModels) {
  const std::vector<PathString> unfusable_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_unfusable_div_not_scale.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_scalar.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_constant.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_broadcasting_changes_shape.onnx",
  };

  for (const auto& path : unfusable_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph&,
           const std::map<std::string, int>& original_op_counts,
           const std::map<std::string, int>& transformed_op_counts) {
          EXPECT_EQ(original_op_counts, transformed_op_counts);
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionReusedInputScale) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_reused_input_scale.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 0);
        EXPECT_EQ(transformed_op_counts["Div"], 0);
        EXPECT_EQ(transformed_op_counts["MatMul"], 0);
        EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 2);
      });
}

TEST_F(GraphTransformationTests, MatMulScaleFusionExcludedInitializerName) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {},
      {"scale"});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionIncompatibleExecutionProvider) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [](Graph& graph) {
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCudaExecutionProvider);
        }
      },
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {kCpuExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionUnsupportedInputType) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_int32.onnx", *logger_,
      [](Graph& graph) {
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCpuExecutionProvider);
        }
      },
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {kCpuExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionDoubleType) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_double.onnx", *logger_,
      [](Graph& graph) {
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCpuExecutionProvider);
        }
      },
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 0);
        EXPECT_EQ(transformed_op_counts["MatMul"], 0);
        EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 1);
      },
      {kCpuExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionRejectsLossyDoubleScale) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<double>({{2, 2}});
    auto* input_b = builder.MakeInput<double>({{2, 2}});
    auto* scale = builder.MakeScalarInitializer<double>(std::nextafter(1.0, 2.0));
    auto* scaled_a = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {input_a, scale}, {scaled_a});
    builder.AddNode("MatMul", {scaled_a, input_b}, {output});
  };

  auto check_not_fused = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Mul") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMul") == 1);
    TEST_RETURN_IF(op_to_count.count("com.microsoft.FusedMatMul") != 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatMulScaleFusion>(),
      TransformerLevel::Level2, 1, check_not_fused, check_not_fused));
}

TEST_F(GraphTransformationTests, MatMulScaleFusionRejectsLossyDoubleDivScale) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<double>({{2, 2}});
    auto* input_b = builder.MakeInput<double>({{2, 2}});
    auto* divisor = builder.MakeScalarInitializer<double>(3.0);
    auto* scaled_a = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Div", {input_a, divisor}, {scaled_a});
    builder.AddNode("MatMul", {scaled_a, input_b}, {output});
  };

  auto check_not_fused = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Div") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMul") == 1);
    TEST_RETURN_IF(op_to_count.count("com.microsoft.FusedMatMul") != 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatMulScaleFusion>(),
      TransformerLevel::Level2, 1, check_not_fused, check_not_fused));
}

TEST_F(GraphTransformationTests, MatMulScaleFusionRejectsLossyCombinedDoubleScale) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<double>({{2, 2}});
    auto* input_b = builder.MakeInput<double>({{2, 2}});
    const double scale_value = static_cast<double>(std::nextafter(1.0f, 2.0f));
    auto* scale1 = builder.MakeScalarInitializer<double>(scale_value);
    auto* scale2 = builder.MakeScalarInitializer<double>(scale_value);
    auto* scaled_a = builder.MakeIntermediate();
    auto* matmul_output = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {input_a, scale1}, {scaled_a});
    builder.AddNode("MatMul", {scaled_a, input_b}, {matmul_output});
    builder.AddNode("Mul", {matmul_output, scale2}, {output});
  };

  auto check_not_fused = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Mul") == 2);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMul") == 1);
    TEST_RETURN_IF(op_to_count.count("com.microsoft.FusedMatMul") != 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatMulScaleFusion>(),
      TransformerLevel::Level2, 1, check_not_fused, check_not_fused));
}

TEST_F(GraphTransformationTests, MatMulScaleFusionRejectsCancellingLossyDoubleScales) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<double>({{1, 1}});
    auto* input_b = builder.MakeInput<double>({{1, 1}});
    auto* scale1 = builder.MakeScalarInitializer<double>(1e300);
    auto* scale2 = builder.MakeScalarInitializer<double>(1e-300);
    auto* scaled_a = builder.MakeIntermediate();
    auto* matmul_output = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {input_a, scale1}, {scaled_a});
    builder.AddNode("MatMul", {scaled_a, input_b}, {matmul_output});
    builder.AddNode("Mul", {matmul_output, scale2}, {output});
  };

  auto check_not_fused = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Mul") == 2);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMul") == 1);
    TEST_RETURN_IF(op_to_count.count("com.microsoft.FusedMatMul") != 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatMulScaleFusion>(),
      TransformerLevel::Level2, 1, check_not_fused, check_not_fused));
}

TEST_F(GraphTransformationTests, MatMulScaleFusionRejectsLossyIntegerScale) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_a = builder.MakeInput<int64_t>({{2, 2}});
    auto* input_b = builder.MakeInput<int64_t>({{2, 2}});
    auto* scale = builder.MakeScalarInitializer<int64_t>((int64_t{1} << 24) + 1);
    auto* scaled_a = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {input_a, scale}, {scaled_a});
    builder.AddNode("MatMul", {scaled_a, input_b}, {output});
  };

  auto check_not_fused = [](Graph& graph) {
    const auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count.at("Mul") == 1);
    TEST_RETURN_IF_NOT(op_to_count.at("MatMul") == 1);
    TEST_RETURN_IF(op_to_count.count("com.microsoft.FusedMatMul") != 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build_test_case, 13, *logger_, std::make_unique<MatMulScaleFusion>(),
      TransformerLevel::Level2, 1, check_not_fused, check_not_fused));
}

TEST_F(GraphTransformationTests, MatMulScaleFusionWithScaleInput) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_with_scale_input.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 1);
        EXPECT_EQ(transformed_op_counts["MatMul"], 1);
        EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 0);
      });
}

#if defined(USE_CUDA)
TEST_F(GraphTransformationTests, IsInfReduceSum_Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/isinf_reducesum.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<IsInfReduceSumFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["IsInf"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.IsAllFinite"], 1);
  EXPECT_EQ(op_to_count["Not"], 1);
}
#endif
#endif

TEST_F(GraphTransformationTests, FilterEnabledOptimizers) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.FilterEnabledOptimizers";
  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(model_uri));

  const auto& graph = session_object.GetGraph();

  // check the ops that should go away if the constant folding transformer runs
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  ASSERT_STATUS_OK(session_object.FilterEnabledOptimizers({"ConstantFolding"}));
  ASSERT_STATUS_OK(session_object.Initialize());  // Initialize runs the transformers

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, FilterEnabledOptimizersViaSessionOptions) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.FilterEnabledOptimizersViaSessionOptions";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers, "ConstantFolding"));

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(model_uri));

  const auto& graph = session_object.GetGraph();

  // check the ops that should go away if the constant folding transformer runs
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  ASSERT_STATUS_OK(session_object.Initialize());  // Initialize runs the transformers

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, PropagateCastOpsTests) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  struct PropagateCastOpsTestSpecs {
    PathString model_uri;
    // Expected number of casts after the transformation with different stratigies and optimization levels
    std::map<std::pair<Strategy, int>, int> casts_count_map;
    vector<std::string> allow_ops = {};  // Allowed ops for PropagateCastOps graph transformer
  };

  std::pair<Strategy, int> insertAndReduce0 = std::make_pair(Strategy::InsertAndReduce, 0);
  std::pair<Strategy, int> insertAndReduce1 = std::make_pair(Strategy::InsertAndReduce, 1);
  std::pair<Strategy, int> floodFill1 = std::make_pair(Strategy::FloodFill, 1);
  std::pair<Strategy, int> floodFill2 = std::make_pair(Strategy::FloodFill, 2);
  std::vector<std::string> allow_matmul = {"MatMul"};
  std::vector<std::string> allow_matmul_transpose = {"MatMul", "Transpose"};
  std::vector<std::string> allow_matmul_transpose_add = {"Add", "MatMul", "Transpose"};
  const std::vector<PropagateCastOpsTestSpecs> test_cases = {
      {MODEL_FOLDER "propagate_cast/squeeze_cast_propagation_test.onnx", {{insertAndReduce0, 2}, {insertAndReduce1, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      {MODEL_FOLDER "propagate_cast/unsqueeze_cast_propagation_test.onnx", {{insertAndReduce0, 2}, {insertAndReduce1, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      // Negative testcase to test that the transformer will not move cast bool to float/float16.
      {MODEL_FOLDER "propagate_cast/negative_test_case_bool_fp_cast.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, {"Add"}},
      {MODEL_FOLDER "propagate_cast/negative_test_case_bool_fp16_cast.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, {"Add"}},
      // Test fusing back to back casts functionality
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float16_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float16_float.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float_float16.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      // Test fusing subgraph functionality
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float16_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float16_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},

      // Test constant propagation with various combinations
      // 1. Computation is float or float16
      // 2. The inputs and/or output may be casted
      // 3. The inputs and/or output may be transposed
      // These variations help testing the following functions.
      // PropagateForward, PropagateBackward, PropagateFP16FromInputsToOutput, and PropagateFP32FromOutputsToInputs

      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 1}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 1}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 3}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 3}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add}};

  // Create a temporary directory, which will be deleted automatically, to save/load the transformed models.
  TemporaryDirectory temp_dir{ORT_TSTR("propagate_casts_test_output_dir")};
  for (PropagateCastOpsTestSpecs test_case : test_cases) {
    for (const auto& scenario : test_case.casts_count_map) {
      Strategy strategy = scenario.first.first;
      int level = scenario.first.second;
      int expected_casts_count = scenario.second;
      std::shared_ptr<Model> p_model;
      ASSERT_STATUS_OK(Model::Load(test_case.model_uri, p_model, nullptr, *logger_));
      Graph& graph = p_model->MainGraph();
      ASSERT_STATUS_OK(graph.Resolve());
      onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
      ASSERT_STATUS_OK(graph_transformation_mgr.Register(
          std::make_unique<PropagateCastOps>(strategy, level, test_case.allow_ops),
          TransformerLevel::Level1));
      ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
      std::filesystem::path p = test_case.model_uri;
      PathString model_filename = ORT_TSTR("transformed_");
      model_filename += p.filename();
      std::filesystem::path transformed_model_uri = std::filesystem::path(temp_dir.Path()) / model_filename;
      ASSERT_STATUS_OK(Model::Save(*p_model, transformed_model_uri));
      // Load the transformed model to validate
      ASSERT_STATUS_OK(Model::Load(transformed_model_uri, p_model, nullptr, *logger_));
      Graph& transformed_graph = p_model->MainGraph();
      ASSERT_STATUS_OK(transformed_graph.Resolve());
      std::map<std::string, int> op_to_count = CountOpsInGraph(transformed_graph);
      ASSERT_EQ(op_to_count["Cast"], expected_casts_count);
    }
  }
}

#ifdef ENABLE_TRAINING_CORE
TEST_F(GraphTransformationTests, PropagateCastOpsTests_Gelu) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* gelu_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Gelu", {cast_out_0}, {gelu_out}, kMSDomain);
      builder.AddNode("Cast", {gelu_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<BFloat16>({{2, -1, 3, -1}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* gelu_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Gelu", {cast_out_0}, {gelu_out}, kMSDomain);
      builder.AddNode("Cast", {gelu_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, PropagateCastOpsTests_Softmax) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  {
    auto build_test_case = [](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Softmax", {cast_out_0}, {softmax_out});
      builder.AddNode("Cast", {softmax_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  {
    auto build_test_case = [](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<BFloat16>({{2, -1, 3, -1}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Softmax", {cast_out_0}, {softmax_out});
      builder.AddNode("Cast", {softmax_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

#endif

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (int64_t)
                 |
                Neg
            /    |  \______________________________________________________
           /     |  256 (int64_t)                                       Cast
          / ...  | /                                    _________/     /   \      \_____________
        Add      Add                               ____/    __________/     \________           \
         |       |  0 (int64_t)  511 (int64_t)    /        /                         \          |
         |       |  __/    _____/                /        | 128 (int32_t)            |  64 [1] |
        Clip ... Clip                           /         | /                        |  /       |
         |        |                            Sub   ...  Sub                        Mul  ...  Mul
                                                |         |                          |          |
graph out [1, 1, 256, 256] (int64_t)        graph out [1, 1, 256, 256] (int32_t)   graph out [1, 1, 256, 256] (int32_t)

Be noted:
 the Add's input initializer 256 is a scalar int64_t;
 the Sub's input initializer 128 is a scalar int32_t;
 the Mul's input initializer 64 is a 1-D int32_t.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareIntTypedInitializer) {
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 6U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Clip"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Neg"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 15U);
    return Status::OK();
  };

  std::vector<int64_t> adders{256, 512};
  std::vector<int32_t> subers{128, 512};
  std::vector<int32_t> mulers{64, 512};
  for (size_t test_data_index = 0; test_data_index < adders.size(); ++test_data_index) {
    int64_t adder = adders[test_data_index];
    int32_t suber = subers[test_data_index];
    int32_t muler = mulers[test_data_index];
    auto post_graph_checker = [adder, suber, muler](Graph& graph) {
      const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
      TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 5U);
      const NodeArg* add_initializer = nullptr;
      const NodeArg* clip_min_initializer = nullptr;
      const NodeArg* clip_max_initializer = nullptr;
      const NodeArg* sub_initializer = nullptr;
      const NodeArg* mul_initializer = nullptr;

      for (auto& node : graph.Nodes()) {
        if (node.OpType().compare("Add") == 0) {
          if (!add_initializer) {
            add_initializer = node.InputDefs()[1];
            TEST_RETURN_IF_NOT(add_initializer != nullptr);
            const TensorShapeProto* s = add_initializer->Shape();
            TEST_RETURN_IF_NOT(s->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
            CheckShapeEquality(add_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        } else if (node.OpType().compare("Clip") == 0) {
          if (!clip_min_initializer && !clip_max_initializer) {
            clip_min_initializer = node.InputDefs()[1];
            clip_max_initializer = node.InputDefs()[2];
            TEST_RETURN_IF(clip_min_initializer == nullptr);
            TEST_RETURN_IF(clip_max_initializer == nullptr);
            const TensorShapeProto* s1 = clip_min_initializer->Shape();
            const TensorShapeProto* s2 = clip_max_initializer->Shape();
            TEST_RETURN_IF_NOT(s1->dim_size() == 0);
            TEST_RETURN_IF_NOT(s2->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(clip_min_initializer == node.InputDefs()[1]);
            TEST_RETURN_IF_NOT(clip_max_initializer == node.InputDefs()[2]);
            CheckShapeEquality(clip_min_initializer->Shape(), node.InputDefs()[1]->Shape());
            CheckShapeEquality(clip_max_initializer->Shape(), node.InputDefs()[2]->Shape());
          }
        } else if (node.OpType().compare("Sub") == 0) {
          if (!sub_initializer) {
            sub_initializer = node.InputDefs()[1];
            TEST_RETURN_IF(sub_initializer == nullptr);
            TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
            CheckShapeEquality(sub_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        } else if (node.OpType().compare("Mul") == 0) {
          if (!mul_initializer) {
            mul_initializer = node.InputDefs()[1];
            TEST_RETURN_IF(mul_initializer == nullptr);
            const TensorShapeProto* s = mul_initializer->Shape();
            TEST_RETURN_IF_NOT(s->dim_size() == 1);
            auto dim1 = s->dim(0);
            TEST_RETURN_IF_NOT(s->dim(0).has_dim_value());
            TEST_RETURN_IF_NOT(s->dim(0).dim_value() == 1);
          } else {
            TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
            CheckShapeEquality(mul_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        }
      }

      for (const auto& entry : initialized_tensor_set) {
        InlinedVector<int64_t> values;
        constexpr bool require_constant = true;
        NodeArg* initializer_node_arg = graph.GetNodeArg(entry.first);
        TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));

        if (add_initializer != nullptr && entry.first.compare(add_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == adder);
        } else if (clip_min_initializer != nullptr && entry.first.compare(clip_min_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == 0);
        } else if (clip_max_initializer != nullptr && entry.first.compare(clip_max_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == 511);
        } else if (sub_initializer != nullptr && entry.first.compare(sub_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == suber);
        } else if (mul_initializer != nullptr && entry.first.compare(mul_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == muler);
        }
      }

      auto op_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count.size() == 6U);
      TEST_RETURN_IF_NOT(op_count["Add"] == 3);
      TEST_RETURN_IF_NOT(op_count["Clip"] == 3);
      TEST_RETURN_IF_NOT(op_count["Sub"] == 3);
      TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
      TEST_RETURN_IF_NOT(op_count["Neg"] == 1);
      TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
      return Status::OK();
    };

    auto build_test_case = [adder, suber, muler](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int64_t>({{1, 1, 256, 256}});
      auto* neg_out = builder.MakeIntermediate();
      builder.AddNode("Neg", {input_arg}, {neg_out});

      // test scalar int64_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* add_initializer = builder.MakeScalarInitializer<int64_t>(adder);
        auto* add_out = builder.MakeIntermediate();
        auto* clip_out = builder.MakeOutput();
        auto* clip_min_initializer = builder.MakeScalarInitializer<int64_t>(0);
        auto* clip_max_initializer = builder.MakeScalarInitializer<int64_t>(511);
        builder.AddNode("Add", {neg_out, add_initializer}, {add_out});
        builder.AddNode("Clip", {add_out, clip_min_initializer, clip_max_initializer}, {clip_out});
      }
      auto* cast_out = builder.MakeIntermediate();
      builder.AddNode("Cast", {neg_out}, {cast_out})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));

      // test scalar int32_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* sub_initializer = builder.MakeScalarInitializer<int32_t>(suber);
        auto* sub_out = builder.MakeOutput();
        builder.AddNode("Sub", {cast_out, sub_initializer}, {sub_out});
      }

      // test 1-D int32_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* mul_initializer = builder.MakeInitializer<int32_t>({1}, {muler});
        auto* mul_out = builder.MakeOutput();
        builder.AddNode("Mul", {cast_out, mul_initializer}, {mul_out});
      }
    };

    const std::vector<int> opsets{12, 13, 14, 23};  // Clip support int64_t since opset 12
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger_, std::move(transformer), TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

template <typename T>
void BuildConstantSharingDivMulGraph(ModelTestBuilder& builder) {
  auto* input0_arg = builder.MakeInput<T>({{1, 1, 256, 256}});
  auto* input1_arg = builder.MakeInput<T>({{1, 1, 256, 256}});
  auto* div_out = builder.MakeIntermediate();
  builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

  for (size_t i = 0; i < 12; ++i) {
    NodeArg* mul_initializer = nullptr;
    if (std::is_same<T, MLFloat16>::value) {
      mul_initializer = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(1.0f));
    } else if (std::is_same<T, float>::value) {
      mul_initializer = builder.MakeScalarInitializer<float>(1.0f);
    } else {
      ASSERT_TRUE(false);
    }
    auto* mul_out = builder.MakeOutput();
    builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float|MLFloat16)
                 |
                Div
            /    |       \
           /     |  1.0   \
          / ...  |  / ...  \
        Mul      Mul      Mul
         |       |         |
 graph out [1, 1, 256, 256] (float|MLFloat16)

Be noted:
 the Mul's input initializer 1.0f is a scalar float/MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareFloatOrHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 2U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 12U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 1U);
    const NodeArg* mul_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        int32_t data_type = tensor_proto->data_type();
        onnxruntime::Initializer float_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        float float_const_value;
        if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          float_const_value = math::halfToFloat(float_const.data<MLFloat16>()->val);
        } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          float_const_value = *(float_const.data<float>());
        } else {
          return Status(common::ONNXRUNTIME, common::FAIL, "unexpected type");
        }

        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 2U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14, 23};  // Clip support int64_t since opset 12

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraph<float>(builder);
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // MLFloat16 data type tests.
  auto build_test_case_mlfloat16 = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraph<MLFloat16>(builder);
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_mlfloat16, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

template <typename T>
void BuildConstantSharingDivMulGraphFor2DInitializer(ModelTestBuilder& builder) {
  auto* input0_arg = builder.MakeInput<T>({{1, 1, 256, 8}});
  auto* input1_arg = builder.MakeInput<T>({{1, 1, 256, 8}});
  auto* div_out = builder.MakeIntermediate();
  builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

  std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<MLFloat16> values_float16;
  values_float16.reserve(values.size());
  if (std::is_same<T, MLFloat16>::value) {
    for (auto v : values) {
      values_float16.push_back(MLFloat16(v));
    }
  }

  for (size_t i = 0; i < 12; ++i) {
    NodeArg* mul_initializer = nullptr;
    if (std::is_same<T, MLFloat16>::value) {
      mul_initializer = builder.MakeInitializer<MLFloat16>({1, 8}, values_float16);
    } else if (std::is_same<T, float>::value) {
      mul_initializer = builder.MakeInitializer<float>({1, 8}, values);
    } else {
      ASSERT_TRUE(false);
    }
    auto* mul_out = builder.MakeOutput();
    builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 8] (float|MLFloat16)
                 |
                Div
            /    |       \
           /     |        \
          / ...  |  / ...  \
        Mul      Mul      Mul
         |       |         |
 graph out [1, 1, 256, 8] (float|MLFloat16)

Be noted:
 the Mul's input initializer is a 2D float/MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_Share2DFloatOrHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 2U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 12U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 1U);
    const NodeArg* mul_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 2);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        int32_t data_type = tensor_proto->data_type();
        onnxruntime::Initializer float_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 8U);
        for (int i = 0; i < 8; ++i) {
          float float_const_value;
          if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
            float_const_value = math::halfToFloat((float_const.data<MLFloat16>() + i)->val);
          } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            float_const_value = *(float_const.data<float>() + i);
          } else {
            return Status(common::ONNXRUNTIME, common::FAIL, "unexpected type");
          }
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 2U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14, 23};  // Clip support int64_t since opset 12

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraphFor2DInitializer<float>(builder);
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // MLFloat16 data type tests.
  auto build_test_case_mlfloat16 = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraphFor2DInitializer<MLFloat16>(builder);
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_mlfloat16, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float)
                 |
                Div ______________________________
            /    |                 |              |
           /     |  1.0float       |  1.0half     |  1.0half
          / ...  |  / ...          |  /   ...     |  /   ...
        Mul      Mul              Add            Add
         |       |                     \          /
 graph out [1, 1, 256, 256](float)   graph out [1, 1, 256, 256](MLFloat16)

Be noted:
 the Mul's input initializer 1.0f is a scalar float.
 the Add's input initializer 1.0f is a scalar MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareFloatAndHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 4U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 6U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 2U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Add") == 0) {
        if (!add_initializer) {
          add_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(add_initializer == nullptr);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(add_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
      int32_t data_type = tensor_proto->data_type();
      onnxruntime::Initializer float_const{graph, *tensor_proto, graph.ModelPath()};
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        float float_const_value = *(float_const.data<float>());
        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      } else if (entry.first.compare(add_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
        float float_const_value = math::halfToFloat(float_const.data<MLFloat16>()->val);
        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 4U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count["Add"] == 3);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14, 23};

  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* mul_initializer = builder.MakeScalarInitializer<float>(1.0f);

      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
    }

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    for (size_t i = 0; i < 3; ++i) {
      NodeArg* add_initializer = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(1.0f));
      auto* add_out = builder.MakeOutput();
      builder.AddNode("Add", {cast_out, add_initializer}, {add_out});
    }
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 8, 8] (float)
                 |
                Div ______________________________
            /    |    \_______     |              |
           /     |  float  |  |    |  half        |  half
          / ...  |  / ...  |  |    |  /   ...     |  /   ...
        Mul      Mul      Sub Sub Add            Add
         |       |         |  |        \          /
 graph out [1, 1, 8, 8](float)   graph out [1, 1, 8, 8](MLFloat16)

Be noted:
 the Mul's input initializer is a 2D float tensor.
 the Add's input initializer is a 2D MLFloat16 tensor.
*/
TEST_F(GraphTransformationTests, ConstantSharing_Share2DFloatAndHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 5U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 9U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 3U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* sub_initializer = nullptr;
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim(0).dim_value() == 1);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim(1).dim_value() == 8);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Sub") == 0) {
        if (!sub_initializer) {
          sub_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(sub_initializer == nullptr);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim(0).dim_value() == 8);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim(1).dim_value() == 1);
        } else {
          TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Add") == 0) {
        if (!add_initializer) {
          add_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(add_initializer == nullptr);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim(0).dim_value() == 1);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim(1).dim_value() == 8);
        } else {
          TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(sub_initializer == nullptr);
    TEST_RETURN_IF(add_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
      int32_t data_type = tensor_proto->data_type();
      onnxruntime::Initializer float_const{graph, *tensor_proto, graph.ModelPath()};
      TEST_RETURN_IF_NOT(float_const.size() == 8U);
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = *(float_const.data<float>() + i);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      } else if (entry.first.compare(sub_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = *(float_const.data<float>() + i);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      } else if (entry.first.compare(add_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = math::halfToFloat((float_const.data<MLFloat16>() + i)->val);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 5U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count["Add"] == 3);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14, 23};

  std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<MLFloat16> values_float16;
  values_float16.reserve(values.size());
  for (auto v : values) {
    values_float16.push_back(MLFloat16(v));
  }

  auto build_test_case_float = [&values, &values_float16](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 8, 8}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 8, 8}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* mul_initializer = builder.MakeInitializer<float>({1, 8}, values);
      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
    }

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* sub_initializer = builder.MakeInitializer<float>({8, 1}, values);
      auto* sub_out = builder.MakeOutput();
      builder.AddNode("Sub", {div_out, sub_initializer}, {sub_out});
    }

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    for (size_t i = 0; i < 3; ++i) {
      NodeArg* add_initializer = builder.MakeInitializer<MLFloat16>({1, 8}, values_float16);
      auto* add_out = builder.MakeOutput();
      builder.AddNode("Add", {cast_out, add_initializer}, {add_out});
    }
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float)
                 |
                Div
            /    |  \______________________________________________________
           /     |  infinity (float)                                       Cast
          / ...  | /                                    _________/     /   \      \_____________
        Sub      Sub                               ____/    __________/     \________           |
         |       |                                /        /                         \          |
         |       |                               /        | int64_max (int64_t)      |          |
         |       |                              /         | /                        |          |
         |       |                             Mul   ...  Mul                        Mul  ...  Mul
                                                |         |                          |          |
graph out [1, 1, 256, 256] (float)        graph out [1, 1, 256, 256] (int64_t)   graph out [1, 1, 256, 256] (int64_t)

Be noted:
 the Sub's input initializer is a scalar std::numeric_limits<float>::infinity();
 the Mul's input initializer is a scalar std::numeric_limits<int64_t>::max().
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareIntMaxOrFloatInfinityInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 4U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 24U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 2U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* sub_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Sub") == 0) {
        if (!sub_initializer) {
          sub_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(sub_initializer == nullptr);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(sub_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer int64_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(int64_const.size() == 1U);
        int64_t int64_const_value = *(int64_const.data<int64_t>());
        TEST_RETURN_IF_NOT(int64_const_value == std::numeric_limits<int64_t>::max());
      } else if (entry.first.compare(sub_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer float_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        float float_const_value = *(float_const.data<float>());
        TEST_RETURN_IF_NOT(float_const_value == std::numeric_limits<float>::infinity());
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 4U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    TEST_RETURN_IF_NOT(op_count["Sub"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14, 23};

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64));
    for (size_t i = 0; i < 12; ++i) {
      NodeArg* mul_initializer = nullptr;
      mul_initializer = builder.MakeScalarInitializer<int64_t>(std::numeric_limits<int64_t>::max());
      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {cast_out, mul_initializer}, {mul_out});
    }

    for (size_t i = 0; i < 12; ++i) {
      NodeArg* sub_initializer = nullptr;
      sub_initializer = builder.MakeScalarInitializer<float>(std::numeric_limits<float>::infinity());
      auto* sub_out = builder.MakeOutput();
      builder.AddNode("Sub", {div_out, sub_initializer}, {sub_out});
    }
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph as below.
           graph input [2] (float)                Constant (1.0float)                Constant (1.0uint8)
                  \_______________  ________________/        |                             |
                                  \/                         |                             |
                                 Add                         |                             |
                                  |                          |                             |
                            graph output [2](float)   graph output [](float)       graph output [](int8)

Be noted: expected result graph should maintain original graph outputs,
   both float and unin8 constant values are not shared.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShouldNotShareForGraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "scalar_const_not_share.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["Add"] == 1);
    // Be noted, constant nodes are converted to initialized already.
    ASSERT_TRUE(graph.GetAllInitializedTensors().size() == 2U);
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    ASSERT_TRUE(initialized_tensor_set.size() == 2U);
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Add") == 0) {
        add_initializer = node.InputDefs()[1];
        ASSERT_TRUE(add_initializer->Shape()->dim_size() == 0);
        ASSERT_TRUE(add_initializer->Name().compare("y_scale") == 0);
      }
    }
    ASSERT_TRUE(add_initializer != nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare("y_scale") == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer int64_const{graph, *tensor_proto, graph.ModelPath()};
        ASSERT_TRUE(int64_const.size() == 1U);
        float float_const_value = *(int64_const.data<float>());
        ASSERT_TRUE(float_const_value == 1);
      } else {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer uint8_const{graph, *tensor_proto, graph.ModelPath()};
        ASSERT_TRUE(uint8_const.size() == 1U);
        uint8_t uint8_const_value = *(uint8_const.data<uint8_t>());
        ASSERT_TRUE(uint8_const_value == static_cast<uint8_t>(1));
      }
    }

    auto op_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_count.size() == 1U);
    ASSERT_TRUE(op_count["Add"] == 1);
  }
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_AllGather) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{54}});
    auto* shape_arg = builder.MakeInput<int64_t>({{4}});
    auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
    auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
    auto* gather_index_2 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(1)});
    auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* gather_out_2 = builder.MakeIntermediate();
    auto* gather_out_3 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();

    builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
    builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
        .AddAttribute("axis", static_cast<int64_t>(-2));
    builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3);
    return Status::OK();
  };

  // OpSet-12, not support
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }

  // OpSet-14
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 2);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        } else if (node.OpType() == "Squeeze") {
          const NodeArg& input_arg = *(node.InputDefs()[1]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_AllSlice_GraphInput) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{2, 3, 8, 3}});
    auto* starts_1 = builder.MakeInitializer<int64_t>({1}, {0});
    auto* ends_1 = builder.MakeInitializer<int64_t>({1}, {2});
    auto* axes_1 = builder.MakeInitializer<int64_t>({1}, {2});
    auto* steps_1 = builder.MakeInitializer<int64_t>({1}, {1});
    auto* starts_2 = builder.MakeInitializer<int64_t>({1}, {2});
    auto* ends_2 = builder.MakeInitializer<int64_t>({1}, {-2});
    auto* axes_2 = builder.MakeInitializer<int64_t>({1}, {-2});
    auto* steps_2 = builder.MakeInitializer<int64_t>({1}, {1});
    auto* starts_3 = builder.MakeInitializer<int64_t>({1}, {-2});
    auto* ends_3 = builder.MakeInitializer<int64_t>({1}, {16});
    auto* axes_3 = builder.MakeInitializer<int64_t>({1}, {2});
    auto* slice_out_1 = builder.MakeIntermediate();
    auto* slice_out_2 = builder.MakeIntermediate();
    auto* slice_out_3 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();

    builder.AddNode("Slice", {data_arg, starts_1, ends_1, axes_1, steps_1}, {slice_out_1});
    builder.AddNode("Slice", {data_arg, starts_2, ends_2, axes_2, steps_2}, {slice_out_2});
    builder.AddNode("Slice", {data_arg, starts_3, ends_3, axes_3}, {slice_out_3});
    builder.AddNode("Transpose", {slice_out_1}, {transpose_out_1})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    builder.AddNode("Transpose", {slice_out_2}, {transpose_out_2})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    builder.AddNode("Transpose", {slice_out_3}, {transpose_out_3})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Slice"] == 3);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Split") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_AllSlice_GraphInput_gh_issue_24203) {
  // https://github.com/microsoft/onnxruntime/issues/24203
  // This bug is manifested when the model features nameless nodes and there are more than one
  // SliceToSplitFusion resulting in multiple Split nodes with the same name.
  const PathString CUSTOM_OP_MODEL_URI = ORT_TSTR("testdata/gh_issue_24203.onnx");
  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Level2;
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.Load(CUSTOM_OP_MODEL_URI));
  ASSERT_STATUS_OK(session.Initialize());
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_Combined) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{144}});
    auto* shape_arg = builder.MakeInput<int64_t>({{4}});
    auto* reshape_out = builder.MakeIntermediate<float>({{2, 8, 3, 3}});
    auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(5)});
    auto* starts_2 = builder.MakeInitializer<int64_t>({1}, {6});
    auto* ends_2 = builder.MakeInitializer<int64_t>({1}, {8});
    auto* axes_2 = builder.MakeInitializer<int64_t>({1}, {-3});
    auto* steps_2 = builder.MakeInitializer<int64_t>({1}, {1});
    auto* gather_index_3 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(4)});
    auto* starts_4 = builder.MakeInitializer<int64_t>({1}, {-16});
    auto* ends_4 = builder.MakeInitializer<int64_t>({1}, {4});
    auto* axes_4 = builder.MakeInitializer<int64_t>({1}, {1});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* slice_out_2 = builder.MakeIntermediate();
    auto* gather_out_3 = builder.MakeIntermediate();
    auto* slice_out_4 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();
    auto* transpose_out_4 = builder.MakeOutput();

    builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
    builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Slice", {reshape_out, starts_2, ends_2, axes_2, steps_2}, {slice_out_2});
    builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
        .AddAttribute("axis", static_cast<int64_t>(-3));
    builder.AddNode("Slice", {reshape_out, starts_4, ends_4, axes_4}, {slice_out_4});
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {slice_out_2}, {transpose_out_2})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    builder.AddNode("Transpose", {slice_out_4}, {transpose_out_4})
        .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Slice"] == 2);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Split") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(1 == static_cast<int>(attrs.at("axis").i()));
      } else if (node.OpType() == "Squeeze") {
        const NodeArg& input_arg = *(node.InputDefs()[1]);
        const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
        TEST_RETURN_IF_NOT(tensor_proto != nullptr);
        Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
        TEST_RETURN_IF_NOT(1 == static_cast<int>(*(init_const.data<int64_t>())));
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_Consume_Initializer) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInitializer<float>({2, 3, 3, 3}, std::vector<float>(54));
    auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
    auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
    auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* gather_out_2 = builder.MakeIntermediate();
    auto* gather_out_3 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();

    builder.AddNode("Gather", {data_arg, gather_index_1}, {gather_out_1}).AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Gather", {data_arg, gather_index_2}, {gather_out_2})
        .AddAttribute("axis", static_cast<int64_t>(-2));
    builder.AddNode("Gather", {data_arg, gather_index_3}, {gather_out_3}).AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 3);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Split") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
      } else if (node.OpType() == "Squeeze") {
        const NodeArg& input_arg = *(node.InputDefs()[1]);
        const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
        TEST_RETURN_IF_NOT(tensor_proto != nullptr);
        Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
        TEST_RETURN_IF_NOT(2 == static_cast<int>(*(init_const.data<int64_t>())));
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_Invalid) {
  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] > 0 || CountOpsInGraph(graph)["Slice"] > 0);
    return Status::OK();
  };
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] > 0 || CountOpsInGraph(graph)["Slice"] > 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
    return Status::OK();
  };

  // Not cover all elements of specific dimension.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{72}});
      auto* shape_arg = builder.MakeInput<int64_t>({{4}});
      auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 4, 3}});
      auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* gather_index_2 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(1)});
      auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* gather_out_2 = builder.MakeIntermediate();
      auto* gather_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
      builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
          .AddAttribute("axis", static_cast<int64_t>(-2));
      builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
      builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }

  // Has overlap.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{2, 3, 8, 3}});
      auto* starts_1 = builder.MakeInitializer<int64_t>({1}, {0});
      auto* ends_1 = builder.MakeInitializer<int64_t>({1}, {3});
      auto* axes_1 = builder.MakeInitializer<int64_t>({1}, {2});
      auto* steps_1 = builder.MakeInitializer<int64_t>({1}, {1});
      auto* starts_2 = builder.MakeInitializer<int64_t>({1}, {2});
      auto* ends_2 = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* axes_2 = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* steps_2 = builder.MakeInitializer<int64_t>({1}, {1});
      auto* starts_3 = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* ends_3 = builder.MakeInitializer<int64_t>({1}, {16});
      auto* axes_3 = builder.MakeInitializer<int64_t>({1}, {2});
      auto* slice_out_1 = builder.MakeIntermediate();
      auto* slice_out_2 = builder.MakeIntermediate();
      auto* slice_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Slice", {data_arg, starts_1, ends_1, axes_1, steps_1}, {slice_out_1});
      builder.AddNode("Slice", {data_arg, starts_2, ends_2, axes_2, steps_2}, {slice_out_2});
      builder.AddNode("Slice", {data_arg, starts_3, ends_3, axes_3}, {slice_out_3});
      builder.AddNode("Transpose", {slice_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
      builder.AddNode("Transpose", {slice_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
      builder.AddNode("Transpose", {slice_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }

  // Invalid axis.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{54}});
      auto* shape_arg = builder.MakeInput<int64_t>({{4}});
      auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
      auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* gather_out_2 = builder.MakeIntermediate();
      auto* gather_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
      builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
          .AddAttribute("axis", static_cast<int64_t>(1));
      builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
          .AddAttribute("axis", static_cast<int64_t>(3));
      builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }
}

// The fusion reads the 'axis' attribute/input straight from the model, and needs to reject a value
// that falls outside [0, rank) for the shared input's known rank rather than indexing its shape with it.
TEST_F(GraphTransformationTests, GatherSliceToSplitFusion_OutOfRangeAxis) {
  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 2);
    return Status::OK();
  };
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 0);
    return Status::OK();
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{54}});
    auto* shape_arg = builder.MakeInput<int64_t>({{4}});
    auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
    // gather_index_1 is a graph input with no static shape, so ONNX's own Gather shape inference
    // (which requires both the data AND indices shapes to be known before range-checking 'axis')
    // cannot validate 'axis' here and silently skips that check, matching how this can be reached
    // in a real model without ONNX itself already rejecting it at load.
    auto* gather_index_1 = builder.MakeInput<int64_t>(std::nullopt);
    auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* gather_out_2 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();

    builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
    // 'axis' is well outside the reshape output's rank of 4; the fusion must reject this before
    // indexing the shape with it, rather than crashing.
    builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
        .AddAttribute("axis", static_cast<int64_t>(100));
    builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
        .AddAttribute("axis", static_cast<int64_t>(1));
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherSliceToSplitFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, GatherToSliceFusion) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_map = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_map["Range"] == 1);
    TEST_RETURN_IF_NOT(op_count_map["Gather"] == 1);
    return Status::OK();
  };

  // OpSet-12, Tind is int32.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{8, 8, 8, 8}});
      auto* range_input_1 = builder.MakeInitializer<int32_t>({}, {0});
      auto* range_input_2 = builder.MakeInitializer<int32_t>({}, {8});
      auto* range_input_3 = builder.MakeInitializer<int32_t>({}, {1});
      auto* range_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeOutput();

      builder.AddNode("Range", {range_input_1, range_input_2, range_input_3}, {range_output});
      builder.AddNode("Gather", {data_arg, range_output}, {gather_output})
          .AddAttribute("axis", static_cast<int64_t>(2));
    };

    auto post_graph_checker = [&](Graph& graph) {
      auto op_count_map = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_map["Range"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Gather"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Slice"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          const NodeArg& input_arg = *(node.InputDefs()[3]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32);
          TEST_RETURN_IF_NOT(2 == *(init_const.data<int32_t>()));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSliceFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }

  // OpSet-14, Tind is int64.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{8, 8, 8, 8}});
      auto* range_input_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* range_input_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(8)});
      auto* range_input_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* range_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeOutput();

      builder.AddNode("Range", {range_input_1, range_input_2, range_input_3}, {range_output});
      builder.AddNode("Gather", {data_arg, range_output}, {gather_output})
          .AddAttribute("axis", static_cast<int64_t>(2));
    };

    auto post_graph_checker = [&](Graph& graph) {
      auto op_count_map = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_map["Range"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Gather"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Slice"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          const NodeArg& input_arg = *(node.InputDefs()[3]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int32_t>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSliceFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  }

  // OpSet-27, Tind is int64. Range gained an opset-27 schema in ONNX 1.22; ensure the fusion still matches it.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{8, 8, 8, 8}});
      auto* range_input_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* range_input_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(8)});
      auto* range_input_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* range_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeOutput();

      builder.AddNode("Range", {range_input_1, range_input_2, range_input_3}, {range_output});
      builder.AddNode("Gather", {data_arg, range_output}, {gather_output})
          .AddAttribute("axis", static_cast<int64_t>(2));
    };

    auto post_graph_checker = [&](Graph& graph) {
      auto op_count_map = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_map["Range"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Gather"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Slice"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          const NodeArg& input_arg = *(node.InputDefs()[3]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{graph, *tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int32_t>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSliceFusion>();
    // Opset 27 is still under development in ONNX 1.22, so the default released-opset-only model load
    // would throw on strict (ALLOW_RELEASED_ONNX_OPSET_ONLY!=0) legs. Allow the unreleased opset here so
    // this sub-block exercises the opset-27 Range schema on every CI leg, not just the relaxed ones.
    const ModelOptions allow_unreleased_opset{kAllowReleasedOpsetsOnly,
                                              /*strict_shape_type_inference*/ false};
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 27, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker,
                                          allow_unreleased_opset));
  }
}

#if !defined(DISABLE_CONTRIB_OPS)

TEST_F(GraphTransformationTests, MatMulNBitsBiasFusion) {
  struct TestOptions {
    bool bias_is_first_add_input{false};
    bool add_produces_graph_output{false};
    bool use_cuda_ep{false};
    bool use_gpt_oss_router_shape{false};
    int64_t weight_prepacked{0};
  };

  auto run_test = [&logger = *logger_](const TestOptions& opts) {
    SCOPED_TRACE(MakeString("bias_is_first_add_input:", opts.bias_is_first_add_input,
                            ", add_produces_graph_output:", opts.add_produces_graph_output,
                            ", use_cuda_ep:", opts.use_cuda_ep,
                            ", use_gpt_oss_router_shape:", opts.use_gpt_oss_router_shape,
                            ", weight_prepacked:", opts.weight_prepacked));

    auto build_test_case = [&](ModelTestBuilder& builder) {
      constexpr size_t qbits = 4;
      constexpr size_t block_size = 32;

      const int64_t M = opts.use_gpt_oss_router_shape ? 1 : 2;
      const int64_t K = opts.use_gpt_oss_router_shape ? 2880 : 4;
      const int64_t N = opts.use_gpt_oss_router_shape ? 32 : 8;

      int q_rows, q_cols;
      MlasBlockwiseQuantizedShape<float, qbits>(block_size, /* columnwise */ true,
                                                static_cast<int>(K), static_cast<int>(N),
                                                q_rows, q_cols);

      size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
      MlasBlockwiseQuantizedBufferSizes<qbits>(block_size, /* columnwise */ true,
                                               static_cast<int>(K), static_cast<int>(N),
                                               q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

      auto* A = builder.MakeInput<float>(std::vector{M, K}, "A");

      auto* B_data = builder.MakeInitializer<uint8_t>({int64_t{q_rows}, int64_t{q_cols}},
                                                      uint8_t{0}, uint8_t{255});
      auto* B_scales = builder.MakeInitializer<float>({static_cast<int64_t>(q_scale_size)},
                                                      1.0f, 2.0f);
      NodeArg* B_zero_points = opts.use_gpt_oss_router_shape
                                   ? builder.MakeEmptyInput()
                                   : builder.MakeInitializer<uint8_t>({static_cast<int64_t>(q_zp_size_in_bytes)},
                                                                      uint8_t{0}, uint8_t{255});

      auto* matmul_output = builder.MakeIntermediate();

      auto& matmul = builder.AddNode("MatMulNBits",
                                     {A, B_data, B_scales, B_zero_points},
                                     {matmul_output},
                                     kMSDomain);
      matmul.AddAttribute("N", N);
      matmul.AddAttribute("K", K);
      matmul.AddAttribute("block_size", static_cast<int64_t>(block_size));
      matmul.AddAttribute("bits", static_cast<int64_t>(qbits));
      if (opts.weight_prepacked != 0) {
        matmul.AddAttribute("weight_prepacked", opts.weight_prepacked);
      }
      if (opts.use_cuda_ep) {
        matmul.SetExecutionProviderType(kCudaExecutionProvider);
      }

      auto* Bias = builder.MakeInput<float>(std::vector{N}, "Bias");

      auto* graph_output = builder.MakeOutput();

      auto* add_output = opts.add_produces_graph_output ? graph_output : builder.MakeIntermediate();

      auto& add = builder.AddNode("Add",
                                  {opts.bias_is_first_add_input ? Bias : matmul_output,
                                   opts.bias_is_first_add_input ? matmul_output : Bias},
                                  {add_output});
      if (opts.use_cuda_ep) {
        add.SetExecutionProviderType(kCudaExecutionProvider);
      }

      if (!opts.add_produces_graph_output) {
        auto& identity = builder.AddNode("Identity",
                                         {add_output},
                                         {graph_output});
        if (opts.use_cuda_ep) {
          identity.SetExecutionProviderType(kCudaExecutionProvider);
        }
      }
    };

    auto pre_graph_checker = [](Graph& graph) {
      auto op_count = CountOpsInGraph(graph);
      EXPECT_EQ(op_count["Add"], 1);
      return Status::OK();
    };

    auto post_graph_checker = [&opts](Graph& graph) {
      auto op_count = CountOpsInGraph(graph);
      EXPECT_EQ(op_count["Add"], opts.weight_prepacked == 0 ? 0 : 1);
      return Status::OK();
    };

    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 21, logger, std::make_unique<MatMulNBitsFusion>(),
                                          TransformerLevel::Level2, 1, pre_graph_checker, post_graph_checker));
  };

  for (bool bias_is_first_add_input : {false, true}) {
    for (bool add_produces_graph_output : {false, true}) {
      TestOptions opts{};
      opts.bias_is_first_add_input = bias_is_first_add_input;
      opts.add_produces_graph_output = add_produces_graph_output;
      run_test(opts);

      // CUDA now fuses bias generically for any shape (the kernel adds the bias with a
      // separate kernel when the fused GEMV fast path does not apply).
      opts.use_cuda_ep = true;
      run_test(opts);

      opts.use_gpt_oss_router_shape = true;
      run_test(opts);
    }
  }

  for (int64_t weight_prepacked : {int64_t{1}, int64_t{2}}) {
    TestOptions opts{};
    opts.weight_prepacked = weight_prepacked;
    run_test(opts);

    opts.use_cuda_ep = true;
    run_test(opts);
  }
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

// Tests for graceful handling of zero-element initializers in optimizer passes.
// Zero-element constant initializers (shape [0], 0 bytes of data) are valid ONNX.
// Optimizer passes must check initializer size() before accessing data<T>().

TEST_F(GraphTransformationTests, DivMulFusion_ZeroElementInitializer) {
  // Div(zero_element_const, X) -> Mul(_, Y) with a shape [0] initializer.
  // Verifies the optimizer correctly handles zero-element initializers.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(14);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("DivMulFusion_ZeroElem");

  // Zero-element float initializer
  auto* init = graph_proto->add_initializer();
  init->set_name("div_const");
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  init->add_dims(0);  // shape [0]

  // Inputs
  for (const char* name : {"X", "Y"}) {
    auto* input = graph_proto->add_input();
    input->set_name(name);
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }
  // Note: div_const is intentionally NOT added as a graph input so it remains a constant initializer.

  // Output
  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }

  // Div node
  auto* div_node = graph_proto->add_node();
  div_node->set_op_type("Div");
  div_node->add_input("div_const");
  div_node->add_input("X");
  div_node->add_output("div_out");

  // Mul node
  auto* mul_node = graph_proto->add_node();
  mul_node->set_op_type("Mul");
  mul_node->add_input("div_out");
  mul_node->add_input("Y");
  mul_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Fusion must be skipped: a zero-element initializer is not a scalar, so
  // both Div and Mul must remain in the graph.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Div"], 1);
  EXPECT_EQ(op_to_count["Mul"], 1);
}

TEST_F(GraphTransformationTests, NoopElimination_ZeroElementInitializer) {
  // Add(X, zero_element_initializer): NoopElimination must NOT remove the Add.
  // A zero-element initializer is not a proven identity value (the unfused op
  // is only equivalent to a no-op for empty-compatible runtime shapes), and
  // the optimizer must not mask invalid models or broaden semantics.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(14);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("NoopElim_ZeroElem");

  // Zero-element float initializer
  auto* init = graph_proto->add_initializer();
  init->set_name("zero_init");
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  init->add_dims(0);

  // Input
  {
    auto* input = graph_proto->add_input();
    input->set_name("X");
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }
  // Note: zero_init is intentionally NOT added as a graph input so it remains a constant initializer.

  // Output
  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }

  // Add node
  auto* add_node = graph_proto->add_node();
  add_node->set_op_type("Add");
  add_node->add_input("X");
  add_node->add_input("zero_init");
  add_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The Add must be preserved: a zero-element initializer is not a proven identity.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Add"], 1);
}

// Companion test: Producer -> Add(X, zero_element_initializer) -> Consumer.
// Exercises the path where the node is topologically removable (data input is
// produced by another node, not a graph input). The Add must still be
// preserved because a zero-element initializer is not a proven identity.
TEST_F(GraphTransformationTests, NoopElimination_ZeroElementInitializer_InternalNode) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(14);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("NoopElim_ZeroElem_Internal");

  // Zero-element float initializer (constant, not a graph input)
  auto* init = graph_proto->add_initializer();
  init->set_name("zero_init");
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  init->add_dims(0);

  // Graph input
  {
    auto* input = graph_proto->add_input();
    input->set_name("X");
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }

  // Graph output
  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(0);
  }

  // Producer: Identity(X) -> id_out
  auto* id_node = graph_proto->add_node();
  id_node->set_op_type("Identity");
  id_node->add_input("X");
  id_node->add_output("id_out");

  // Add(id_out, zero_init) -> add_out  (this is the node that could be removed)
  auto* add_node = graph_proto->add_node();
  add_node->set_op_type("Add");
  add_node->add_input("id_out");
  add_node->add_input("zero_init");
  add_node->add_output("add_out");

  // Consumer: Identity(add_out) -> output
  auto* consumer_node = graph_proto->add_node();
  consumer_node->set_op_type("Identity");
  consumer_node->add_input("add_out");
  consumer_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The Add must be preserved even though it is topologically removable.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Add"], 1);
}

// Verifies that DivMulFusion's tightened size() != 1 check correctly rejects
// multi-element (non-scalar) initializers. Fusion must be skipped and the
// original Div/Mul nodes preserved.
TEST_F(GraphTransformationTests, DivMulFusion_MultiElementInitializer) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(14);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("DivMulFusion_MultiElem");

  // Multi-element (shape [2]) float initializer - must NOT be treated as a scalar.
  auto* init = graph_proto->add_initializer();
  init->set_name("div_const");
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  init->add_dims(2);
  init->add_float_data(1.0f);
  init->add_float_data(2.0f);

  for (const char* name : {"X", "Y"}) {
    auto* input = graph_proto->add_input();
    input->set_name(name);
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
  }

  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
  }

  auto* div_node = graph_proto->add_node();
  div_node->set_op_type("Div");
  div_node->add_input("div_const");
  div_node->add_input("X");
  div_node->add_output("div_out");

  auto* mul_node = graph_proto->add_node();
  mul_node->set_op_type("Mul");
  mul_node->add_input("div_out");
  mul_node->add_input("Y");
  mul_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Fusion must be skipped: Div and Mul both remain.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Div"], 1);
  EXPECT_EQ(op_to_count["Mul"], 1);
}

// Regression tests for https://github.com/microsoft/onnxruntime/issues/32416 and
// https://github.com/microsoft/onnxruntime/issues/32414.
//
// DivMulFusion moves the Mul's other input onto the Div. When a node produces that input, the
// edge carrying it ends at the Mul and dies with it, so the fusion has to recreate it on the Div.
// Otherwise the Div consumes the value by name alone, and a later rewrite rule that removes the
// producer - rewiring only its edge-connected consumers - leaves the Div reading a value that
// nothing produces. Both rules below run in the same Level1 pass, before the next Graph::Resolve.
TEST_F(GraphTransformationTests, DivMulFusion_ReconnectsCastProducerEdge) {
  // x -> Cast(to=FLOAT) -> mid; Div(one, two) -> div_out; Mul(mid, div_out) -> output.
  // DivMulFusion rewrites this to Div(mid, two), then CastElimination drops the same-dtype Cast.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(14);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("DivMulFusion_CastProducer");

  // Scalar constants: the 1 that makes the fusion applicable, and the divisor it keeps.
  auto* one = graph_proto->add_initializer();
  one->set_name("one");
  one->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  one->add_float_data(1.0f);

  auto* two = graph_proto->add_initializer();
  two->set_name("two");
  two->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  two->add_float_data(2.0f);

  {
    auto* input = graph_proto->add_input();
    input->set_name("x");
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
    type->mutable_shape()->add_dim()->set_dim_value(3);
  }
  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
    type->mutable_shape()->add_dim()->set_dim_value(3);
  }

  auto* cast_node = graph_proto->add_node();
  cast_node->set_op_type("Cast");
  cast_node->set_name("cast");
  cast_node->add_input("x");
  cast_node->add_output("mid");
  auto* to_attr = cast_node->add_attribute();
  to_attr->set_name("to");
  to_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  to_attr->set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  auto* div_node = graph_proto->add_node();
  div_node->set_op_type("Div");
  div_node->set_name("div");
  div_node->add_input("one");
  div_node->add_input("two");
  div_node->add_output("div_out");

  auto* mul_node = graph_proto->add_node();
  mul_node->set_op_type("Mul");
  mul_node->set_name("mul");
  mul_node->add_input("mid");
  mul_node->add_input("div_out");
  mul_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<CastElimination>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  // Without the reconnection this fails to resolve: "Node input 'mid' is not a graph input,
  // initializer, or output of a previous node."
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["Div"], 1);
}

TEST_F(GraphTransformationTests, DivMulFusion_ReconnectsDropoutProducerEdge) {
  // Same defect reached through EliminateDropout instead of CastElimination, and with the Mul's
  // operands commuted so the substituted input is found at index 1 rather than index 0:
  // x -> Dropout -> mid; Div(one, two) -> div_out; Mul(div_out, mid) -> output.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(8);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(13);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("DivMulFusion_DropoutProducer");

  auto* one = graph_proto->add_initializer();
  one->set_name("one");
  one->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  one->add_float_data(1.0f);

  auto* two = graph_proto->add_initializer();
  two->set_name("two");
  two->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  two->add_float_data(2.0f);

  {
    auto* input = graph_proto->add_input();
    input->set_name("x");
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
    type->mutable_shape()->add_dim()->set_dim_value(3);
  }
  {
    auto* output = graph_proto->add_output();
    output->set_name("output");
    auto* type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
    type->mutable_shape()->add_dim()->set_dim_value(3);
  }

  // Inference-mode Dropout, so it carries no ratio and is an identity.
  auto* dropout_node = graph_proto->add_node();
  dropout_node->set_op_type("Dropout");
  dropout_node->set_name("dropout");
  dropout_node->add_input("x");
  dropout_node->add_output("mid");

  auto* div_node = graph_proto->add_node();
  div_node->set_op_type("Div");
  div_node->set_name("div");
  div_node->add_input("one");
  div_node->add_input("two");
  div_node->add_output("div_out");

  auto* mul_node = graph_proto->add_node();
  mul_node->set_op_type("Mul");
  mul_node->set_name("mul");
  mul_node->add_input("div_out");
  mul_node->add_input("mid");
  mul_node->add_output("output");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_proto, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<EliminateDropout>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Dropout"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["Div"], 1);
}

// Note: There is intentionally no end-to-end regression test for the
// `ratio.size() != 1` guard in `EliminateDropout`. ONNX Dropout requires
// `ratio` to be a scalar, so a malformed (zero-element or multi-element)
// initializer is rejected by ORT shape inference during `Model::Load` before
// the optimizer ever runs (verified locally: `Model::Load` returns
// `[ShapeInferenceError] Ratio of Dropout must be a scalar`). The guard in
// `dropout_elimination.cc` remains as pure defense-in-depth against future
// internal callers that may bypass shape inference.

// These tests verify that STFTDecomposition skips malformed models
// instead of crashing with OOB writes from negative initializer values.
TEST_F(GraphTransformationTests, STFTDecomposition_NegativeFrameLength) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "stft_negative_frame_length.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 1);

  const InlinedHashSet<std::string_view> empty_ep = {};
  auto stft_transformer = std::make_unique<STFTDecomposition>(empty_ep);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(stft_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // STFT node should NOT be decomposed — transformer skips invalid models
  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 1);
}

TEST_F(GraphTransformationTests, STFTDecomposition_NegativeFrameStep) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "stft_negative_frame_step.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 1);

  const InlinedHashSet<std::string_view> empty_ep = {};
  auto stft_transformer = std::make_unique<STFTDecomposition>(empty_ep);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(stft_transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // STFT node should NOT be decomposed — transformer skips invalid models
  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 1);
}

TEST_F(GraphTransformationTests, STFTDecomposition_NoWindowInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "stft_no_window.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 1);

  const InlinedHashSet<std::string_view> empty_ep = {};
  auto stft_transformer = std::make_unique<STFTDecomposition>(empty_ep);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(stft_transformer), TransformerLevel::Level1));
  // Should not crash (previously dereferenced nullptr window_recipient)
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Valid windowless STFT should be successfully decomposed
  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["STFT"], 0);
}

TEST_F(GraphTransformationTests, FusionPreservesPublicAndSharedValues) {
  const std::vector<std::pair<std::basic_string<ORTCHAR_T>, bool>> models = {
      {ORT_TSTR("fusion/matmul_transpose_public_cast.onnx"), false},
      {ORT_TSTR("fusion/gather_to_slice_public_range.onnx"), false},
      {ORT_TSTR("fusion/fast_gelu_public_entry.onnx"), false},
      {ORT_TSTR("fusion/attention_public_past_key_transpose.onnx"), false},
      {ORT_TSTR("fusion/attention_public_qk_intermediate.onnx"), false},
      {ORT_TSTR("fusion/qdq_public_first_node.onnx"), true},
      {ORT_TSTR("fusion/qdq_shared_source_value.onnx"), true},
  };

  for (const auto& [model, enable_qdq_cleanup] : models) {
    SessionOptions so;
    so.graph_optimization_level = TransformerLevel::MaxLevel;
    if (enable_qdq_cleanup) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
    }

    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(PathString(MODEL_FOLDER) + model));
    ASSERT_STATUS_OK(session.Initialize());
  }
}

}  // namespace test
}  // namespace onnxruntime
