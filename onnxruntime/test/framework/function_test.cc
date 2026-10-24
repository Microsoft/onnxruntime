// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <array>
#include <limits>
#include <optional>
#include <sstream>

#include "core/graph/onnx_protobuf.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/customregistry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/function_utils.h"
#include "core/graph/model.h"
#include "core/graph/model_helpers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/capturing_sink.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"
#include "test/util/include/inference_session_wrapper.h"

// Unit tests to check the implementation of functions, model-local functions,
// function-inlining etc.

namespace onnxruntime {
namespace test {

// Convert source-representation of model to ModelProto:
static void ParseOnnxSource(const char* source, std::string& result) {
  ONNX_NAMESPACE::OnnxParser parser(source);
  ONNX_NAMESPACE::ModelProto model;
  auto parse_status = parser.Parse(model);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Serialize
  std::string serialized_model;
  const bool serialization_status = model.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";
  result = std::move(serialized_model);
}

static void Check(const char* source,
                  const char* input_name, std::vector<float> input_values,
                  const char* output_name, std::vector<float> output_values) {
  // Serialize and then load model:
  std::string serialized_model;
  ParseOnnxSource(source, serialized_model);

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  NameMLValMap feeds;

  std::unique_ptr<CPUExecutionProvider> provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  OrtValue ort_value;
  CreateMLValue<float>(provider->CreatePreferredAllocators()[0], {int64_t(input_values.size())}, input_values, &ort_value);

  feeds.insert(std::make_pair(std::string(input_name), ort_value));

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, AsSpan({std::string(output_name)}), &fetches);
  ASSERT_TRUE(status.IsOK()) << "Session Run failed: " << status.ErrorMessage() << std::endl;

  auto& tensor = fetches[0].Get<Tensor>();
  size_t size = static_cast<size_t>(tensor.Shape().Size());
  EXPECT_EQ(size, output_values.size());

  auto* data = tensor.Data<float>();
  float threshold = 0.001f;

  for (size_t i = 0; i < size; ++i) {
    if (!std::isnan(data[i]) && !std::isnan(output_values[i])) {
      ASSERT_NEAR(data[i], output_values[i], threshold) << "at position i:" << i;
    }
  }
}

static Status LoadModel(const char* source, bool allow_released_opsets_only = true) {
  ONNX_NAMESPACE::OnnxParser parser(source);
  ONNX_NAMESPACE::ModelProto model;
  auto parse_status = parser.Parse(model);
  if (!parse_status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to parse test model: ", parse_status.ErrorMessage());
  }
  if (!parser.EndOfInput()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Extra unparsed input unexpected.");
  }

  try {
    ONNX_NAMESPACE::checker::check_model(model);
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ONNX model check failed: ", e.what());
  }

  std::string serialized_model;
  if (!model.SerializeToString(&serialized_model) || serialized_model.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize test model.");
  }

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};
  std::istringstream sstr(serialized_model);
  return session_object.Load(sstr, allow_released_opsets_only);
}

static ONNX_NAMESPACE::ModelProto CreateFunctionExpansionModel(size_t body_node_count, size_t call_count) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  auto* local_opset = model.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);

  auto set_float_value = [](ONNX_NAMESPACE::ValueInfoProto& value, const std::string& name) {
    value.set_name(name);
    auto* tensor_type = value.mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  };

  auto* graph = model.mutable_graph();
  graph->set_name("function_expansion");
  set_float_value(*graph->add_input(), "input");

  auto* function = model.add_functions();
  function->set_domain("local");
  function->set_name("FunctionToExpand");
  function->add_input("function_input");
  function->add_output("function_output");
  auto* function_opset = function->add_opset_import();
  function_opset->set_domain("");
  function_opset->set_version(13);

  std::string previous_value = "function_input";
  for (size_t i = 0; i < body_node_count; ++i) {
    auto* node = function->add_node();
    node->set_op_type("Identity");
    node->add_input(previous_value);
    previous_value = i + 1 == body_node_count ? "function_output" : "body_" + std::to_string(i);
    node->add_output(previous_value);
  }

  previous_value = "input";
  for (size_t i = 0; i < call_count; ++i) {
    auto* node = graph->add_node();
    node->set_domain("local");
    node->set_op_type("FunctionToExpand");
    node->add_input(previous_value);
    previous_value = "call_" + std::to_string(i);
    node->add_output(previous_value);
  }
  set_float_value(*graph->add_output(), previous_value);

  return model;
}

static ONNX_NAMESPACE::ModelProto CreateRecursiveFunctionExpansionModel(size_t branch_node_count,
                                                                        size_t call_count) {
  auto model = CreateFunctionExpansionModel(0, call_count);
  model.set_doc_string(std::string(1024 * 1024, 'x'));

  auto* function = model.mutable_functions(0);
  auto* condition = function->add_node();
  condition->set_op_type("Constant");
  condition->add_output("condition");
  auto* condition_value = condition->add_attribute();
  condition_value->set_name("value");
  condition_value->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  condition_value->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  condition_value->mutable_t()->add_int32_data(1);

  auto* if_node = function->add_node();
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("function_output");

  for (const auto* attribute_name : {"then_branch", "else_branch"}) {
    auto* attribute = if_node->add_attribute();
    attribute->set_name(attribute_name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* branch = attribute->mutable_g();
    branch->set_name(attribute_name);

    std::string previous_value = "function_input";
    for (size_t i = 0; i < branch_node_count; ++i) {
      auto* node = branch->add_node();
      node->set_op_type("Identity");
      node->add_input(previous_value);
      previous_value = "branch_" + std::to_string(i);
      node->add_output(previous_value);
    }

    auto* output = branch->add_output();
    output->set_name(previous_value);
    auto* tensor_type = output->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  }

  return model;
}

struct FunctionExpansionTestOptions {
  bool claim_first_function_call = false;
  bool disable_aot_inlining = false;
  std::optional<size_t> node_limit;
  std::optional<size_t> byte_limit;
};

static Status InitializeFunctionExpansionModel(
    ONNX_NAMESPACE::ModelProto model,
    std::vector<std::string>& log_messages,
    const FunctionExpansionTestOptions& options = {}) {
  std::string serialized_model;
  ORT_RETURN_IF_NOT(model.SerializeToString(&serialized_model),
                    "Failed to serialize function expansion model.");

  SessionOptions session_options;
  session_options.use_per_session_threads = true;
  if (options.disable_aot_inlining) {
    ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsDisableAheadOfTimeFunctionInlining, "1"));
  }
  if (options.node_limit.has_value()) {
    ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsFunctionExpansionNodeLimit,
        std::to_string(*options.node_limit).c_str()));
  }
  if (options.byte_limit.has_value()) {
    ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsFunctionExpansionByteLimit,
        std::to_string(*options.byte_limit).c_str()));
  }

  auto capturing_sink = std::make_unique<CapturingSink>();
  auto* capturing_sink_ptr = capturing_sink.get();
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::move(capturing_sink), logging::Severity::kWARNING, false,
      logging::LoggingManager::InstanceType::Temporal);
  std::unique_ptr<Environment> environment;
  ORT_RETURN_IF_ERROR(Environment::Create(std::move(logging_manager), environment));

  InferenceSession session{session_options, *environment};
  if (options.claim_first_function_call) {
    class FirstFunctionCallExecutionProvider final
        : public internal_testing_ep::InternalTestingExecutionProvider {
     public:
      FirstFunctionCallExecutionProvider()
          : InternalTestingExecutionProvider({}, {}, DataLayout::NCHW) {}

      std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
          const GraphViewer& graph_view,
          const IKernelLookup&,
          const GraphOptimizerRegistry&,
          IResourceAccountant*) const override {
        for (const auto node_index : graph_view.GetNodesInTopologicalOrder()) {
          const auto* node = graph_view.GetNode(node_index);
          if (node != nullptr && node->CanBeInlined()) {
            std::vector<std::unique_ptr<ComputeCapability>> capabilities;
            capabilities.push_back(utils::MakeComputeCapability(
                graph_view, std::vector<const Node*>{node},
                [node_index]() { return "FirstFunctionCall_" + std::to_string(node_index); },
                Type(), false));
            return capabilities;
          }
        }

        return {};
      }
    };

    ORT_RETURN_IF_ERROR(session.RegisterExecutionProvider(
        std::make_unique<FirstFunctionCallExecutionProvider>()));
  }

  std::istringstream stream(serialized_model);
  ORT_RETURN_IF_ERROR(session.Load(stream));
  const auto status = session.Initialize();
  log_messages = capturing_sink_ptr->Messages();
  return status;
}

TEST(FunctionTest, AotInliningLimitsFunctionExpansionByNodeCount) {
  auto model = CreateRecursiveFunctionExpansionModel(20, 14);
  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = 500,
       .byte_limit = std::nullopt});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("node expansion limit"));
  EXPECT_THAT(log_messages, testing::Contains(testing::HasSubstr("node expansion limit")));
  EXPECT_THAT(log_messages, testing::Not(testing::Contains(testing::HasSubstr("protobuf expansion limit"))));
}

TEST(FunctionTest, AotInliningLimitsUnclaimedCallsSharingClaimedFunction) {
  auto model = CreateRecursiveFunctionExpansionModel(20, 15);
  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = true,
       .disable_aot_inlining = false,
       .node_limit = 500,
       .byte_limit = std::nullopt});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("node expansion limit"));
  EXPECT_THAT(log_messages, testing::Contains(testing::HasSubstr("node expansion limit")));
}

TEST(FunctionTest, AotInliningLimitsFunctionExpansionByProtoBytes) {
  auto model = CreateFunctionExpansionModel(2, 20);
  auto* function = model.mutable_functions(0);
  auto* payload_node = function->add_node();
  payload_node->set_op_type("Constant");
  payload_node->add_output("payload");
  auto* payload_attribute = payload_node->add_attribute();
  payload_attribute->set_name("value");
  payload_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* payload_tensor = payload_attribute->mutable_t();
  payload_tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  payload_tensor->add_dims(256 * 1024);
  payload_tensor->set_raw_data(std::string(256 * 1024, 'x'));

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 1024 * 1024});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("protobuf expansion limit"));
  EXPECT_THAT(log_messages, testing::Contains(testing::HasSubstr("protobuf expansion limit")));
  EXPECT_THAT(log_messages, testing::Not(testing::Contains(testing::HasSubstr("node expansion limit"))));
}

TEST(FunctionTest, FallbackInliningEnforcesExpansionLimitWhenAotIsDisabled) {
  auto model = CreateRecursiveFunctionExpansionModel(20, 14);
  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = true,
       .node_limit = 500,
       .byte_limit = std::nullopt});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("node expansion limit"));
}

TEST(FunctionTest, DefaultExpansionLimitPreservesOrdinaryLargeFunctions) {
  auto model = CreateFunctionExpansionModel(200, 24);
  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(std::move(model), log_messages);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(FunctionTest, AotInliningIgnoresFunctionMetadataForProtoBytes) {
  auto model = CreateFunctionExpansionModel(1, 11);
  model.mutable_functions(0)->set_doc_string(std::string(1024 * 1024, 'x'));

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 1024 * 1024});
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_THAT(log_messages, testing::Not(testing::Contains(testing::HasSubstr("protobuf expansion limit"))));
}

TEST(FunctionTest, AotInliningChargesBoundAttributePayloadPerReference) {
  auto model = CreateFunctionExpansionModel(0, 1);
  constexpr int64_t kPayloadElementCount = 64 * 1024;
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(kPayloadElementCount);
  auto* function = model.mutable_functions(0);
  function->add_attribute("payload");
  for (size_t i = 0; i < 8; ++i) {
    auto* node = function->add_node();
    node->set_op_type("Constant");
    node->add_output(i + 1 == 8 ? "function_output" : "payload_" + std::to_string(i));
    auto* attribute = node->add_attribute();
    attribute->set_name("value");
    attribute->set_ref_attr_name("payload");
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  }

  auto* payload_attribute = model.mutable_graph()->mutable_node(0)->add_attribute();
  payload_attribute->set_name("payload");
  payload_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* payload_tensor = payload_attribute->mutable_t();
  payload_tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  payload_tensor->add_dims(kPayloadElementCount);
  payload_tensor->set_raw_data(std::string(256 * 1024, 'x'));

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 1024 * 1024});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("protobuf expansion limit"));
}

TEST(FunctionTest, AotInliningChargesRepeatedActualNamesPerOccurrence) {
  auto model = CreateFunctionExpansionModel(32, 1);
  for (auto& function_node : *model.mutable_functions(0)->mutable_node()) {
    function_node.set_input(0, "function_input");
  }
  const std::string long_input_name(4096, 'x');
  model.mutable_graph()->mutable_input(0)->set_name(long_input_name);
  model.mutable_graph()->mutable_node(0)->set_input(0, long_input_name);

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 64 * 1024});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("protobuf expansion limit"));
}

TEST(FunctionTest, AotInliningUsesLexicalScopeForShadowedSubgraphNames) {
  auto model = CreateFunctionExpansionModel(0, 1);
  auto* function = model.mutable_functions(0);
  const std::string long_input_name(4096, 'x');
  model.mutable_graph()->mutable_input(0)->set_name(long_input_name);
  model.mutable_graph()->mutable_node(0)->set_input(0, long_input_name);

  auto* condition_node = function->add_node();
  condition_node->set_op_type("Constant");
  condition_node->add_output("condition");
  auto* condition_attribute = condition_node->add_attribute();
  condition_attribute->set_name("value");
  condition_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  condition_attribute->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  condition_attribute->mutable_t()->add_int32_data(1);
  auto* condition_value_info = function->add_value_info();
  condition_value_info->set_name("condition");
  condition_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto* if_node = function->add_node();
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("function_output");
  for (const char* attribute_name : {"then_branch", "else_branch"}) {
    auto* attribute = if_node->add_attribute();
    attribute->set_name(attribute_name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* branch = attribute->mutable_g();
    branch->set_name(attribute_name);

    auto* initializer = branch->add_initializer();
    initializer->set_name("function_input");
    initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    initializer->add_dims(1);
    initializer->add_float_data(1.0f);

    for (size_t i = 0; i < 16; ++i) {
      auto* identity = branch->add_node();
      identity->set_op_type("Identity");
      identity->add_input("function_input");
      const std::string output_name = "shadowed_" + std::to_string(i);
      identity->add_output(output_name);
      if (i + 1 != 16) {
        auto* value_info = branch->add_value_info();
        value_info->set_name(output_name);
        auto* value_type = value_info->mutable_type()->mutable_tensor_type();
        value_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        value_type->mutable_shape()->add_dim()->set_dim_value(1);
      }
    }

    auto* branch_output = branch->add_output();
    branch_output->set_name("shadowed_15");
    auto* output_type = branch_output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    output_type->mutable_shape()->add_dim()->set_dim_value(1);
  }

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 64 * 1024});
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(FunctionTest, AotInliningUsesLexicalScopeForNestedBoundGraphCaptures) {
  auto model = CreateFunctionExpansionModel(0, 1);
  auto* function = model.mutable_functions(0);
  const std::string long_input_name(4096, 'x');
  model.mutable_graph()->mutable_input(0)->set_name(long_input_name);
  model.mutable_graph()->mutable_node(0)->set_input(0, long_input_name);

  function->add_attribute("nested_branch");
  auto* nested_branch_attribute = function->add_attribute_proto();
  nested_branch_attribute->set_name("nested_branch");
  nested_branch_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* nested_branch = nested_branch_attribute->mutable_g();
  nested_branch->set_name("nested_branch");
  for (size_t i = 0; i < 32; ++i) {
    auto* identity = nested_branch->add_node();
    identity->set_op_type("Identity");
    identity->add_input("function_input");
    identity->add_output("nested_output_" + std::to_string(i));
  }
  auto* nested_branch_output = nested_branch->add_output();
  nested_branch_output->set_name("nested_output_31");
  auto* nested_branch_output_type = nested_branch_output->mutable_type()->mutable_tensor_type();
  nested_branch_output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  nested_branch_output_type->mutable_shape()->add_dim()->set_dim_value(1);

  auto* condition_node = function->add_node();
  condition_node->set_op_type("Constant");
  condition_node->add_output("condition");
  auto* condition_attribute = condition_node->add_attribute();
  condition_attribute->set_name("value");
  condition_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  condition_attribute->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  condition_attribute->mutable_t()->add_int32_data(1);
  auto* condition_value_info = function->add_value_info();
  condition_value_info->set_name("condition");
  condition_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto* if_node = function->add_node();
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("function_output");
  for (const char* attribute_name : {"then_branch", "else_branch"}) {
    auto* attribute = if_node->add_attribute();
    attribute->set_name(attribute_name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* branch = attribute->mutable_g();
    branch->set_name(attribute_name);

    auto* initializer = branch->add_initializer();
    initializer->set_name("function_input");
    initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    initializer->add_dims(1);
    initializer->add_float_data(1.0f);

    auto* nested_condition = branch->add_node();
    nested_condition->set_op_type("Constant");
    nested_condition->add_output("nested_condition");
    auto* nested_condition_attribute = nested_condition->add_attribute();
    nested_condition_attribute->set_name("value");
    nested_condition_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    nested_condition_attribute->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
    nested_condition_attribute->mutable_t()->add_int32_data(1);

    auto* nested_if = branch->add_node();
    nested_if->set_op_type("If");
    nested_if->add_input("nested_condition");
    nested_if->add_output("branch_output");
    for (const char* nested_attribute_name : {"then_branch", "else_branch"}) {
      auto* nested_attribute = nested_if->add_attribute();
      nested_attribute->set_name(nested_attribute_name);
      nested_attribute->set_ref_attr_name("nested_branch");
      nested_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    }

    auto* branch_output = branch->add_output();
    branch_output->set_name("branch_output");
    auto* branch_output_type = branch_output->mutable_type()->mutable_tensor_type();
    branch_output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    branch_output_type->mutable_shape()->add_dim()->set_dim_value(1);
  }

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 64 * 1024});
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(FunctionTest, AotInliningChargesNestedBoundGraphAttributeReferences) {
  auto model = CreateFunctionExpansionModel(0, 1);
  constexpr int64_t kPayloadElementCount = 64 * 1024;
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(kPayloadElementCount);

  auto* function = model.mutable_functions(0);
  function->add_attribute("branch");
  function->add_attribute("payload");

  auto* condition_node = function->add_node();
  condition_node->set_op_type("Constant");
  condition_node->add_output("condition");
  auto* condition_attribute = condition_node->add_attribute();
  condition_attribute->set_name("value");
  condition_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  condition_attribute->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  condition_attribute->mutable_t()->add_int32_data(1);
  auto* condition_value_info = function->add_value_info();
  condition_value_info->set_name("condition");
  condition_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto* if_node = function->add_node();
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("function_output");
  for (const char* attribute_name : {"then_branch", "else_branch"}) {
    auto* attribute = if_node->add_attribute();
    attribute->set_name(attribute_name);
    attribute->set_ref_attr_name("branch");
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  }

  auto* branch_attribute = function->add_attribute_proto();
  branch_attribute->set_name("branch");
  branch_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* branch = branch_attribute->mutable_g();
  branch->set_name("bound_branch");
  for (size_t i = 0; i < 8; ++i) {
    auto* constant = branch->add_node();
    const std::string output_name = i + 1 == 8 ? "branch_output" : "unused_" + std::to_string(i);
    constant->set_op_type("Constant");
    constant->add_output(output_name);
    auto* value = constant->add_attribute();
    value->set_name("value");
    value->set_ref_attr_name("payload");
    value->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    value->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    value->mutable_t()->add_dims(kPayloadElementCount);
    if (i + 1 != 8) {
      auto* value_info = branch->add_value_info();
      value_info->set_name(output_name);
      auto* value_type = value_info->mutable_type()->mutable_tensor_type();
      value_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      value_type->mutable_shape()->add_dim()->set_dim_value(kPayloadElementCount);
    }
  }
  auto* branch_output = branch->add_output();
  branch_output->set_name("branch_output");
  auto* branch_output_type = branch_output->mutable_type()->mutable_tensor_type();
  branch_output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  branch_output_type->mutable_shape()->add_dim()->set_dim_value(kPayloadElementCount);

  auto* payload_attribute = function->add_attribute_proto();
  payload_attribute->set_name("payload");
  payload_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* payload_tensor = payload_attribute->mutable_t();
  payload_tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  payload_tensor->add_dims(kPayloadElementCount);
  payload_tensor->set_raw_data(std::string(kPayloadElementCount * sizeof(float), 'x'));

  std::vector<std::string> log_messages;
  const auto status = InitializeFunctionExpansionModel(
      std::move(model), log_messages,
      {.claim_first_function_call = false,
       .disable_aot_inlining = false,
       .node_limit = std::nullopt,
       .byte_limit = 1024 * 1024});
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("protobuf expansion limit"));
}

// A recursive/cyclic chain of model-local functions can be rejected by either layer:
// ONNX 1.22+ detects the cycle in its own model checker ("Cycle detected in model-local
// function references"), which runs before ORT's equivalent check ("must not be recursive").
// Older ONNX versions don't catch it, so ORT's check fires instead. Accept either message so
// the cycle-rejection tests pass regardless of which layer rejects the model.
static testing::Matcher<const std::string&> HasCycleRejectionMessage() {
  return testing::AnyOf(testing::HasSubstr("must not be recursive"),
                        testing::HasSubstr("Cycle detected in model-local function references"));
}

namespace {
const char* basic_code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";
}

TEST(FunctionTest, Basic) {
  Check(basic_code, "x", {1.0, 2.0, 3.0}, "y", {2.0, 4.0, 6.0});
}

TEST(FunctionTest, SwiGLUOpset28FunctionAndNumerics) {
  constexpr const char* code = R"(
        <
        ir_version: 13,
        opset_import: [ "" : 28 ]
        >
        agraph (float[2, 2] A, float[2, 2] B) => (float[2, 2] Y, float[2, 2] YDefault)
        {
            Y = SwiGLU <alpha = 0.5> (A, B)
            YDefault = SwiGLU (A, B)
        }
        )";

  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);

  SessionOptions session_options;
  InferenceSessionWrapper expansion_session{session_options, GetEnvironment()};
  std::stringstream expansion_model_stream(serialized_model);
  // Opset 28 is under development in ONNX 1.23, so allow it for focused function coverage.
  ASSERT_STATUS_OK(expansion_session.Load(expansion_model_stream, false));

  auto& expansion_graph = expansion_session.GetMutableGraph();
  const auto loaded_ops = CountOpsInGraph(expansion_graph);
  ASSERT_EQ(loaded_ops.at("SwiGLU"), 2);
  while (true) {
    Node* swiglu_node = nullptr;
    for (auto& node : expansion_graph.Nodes()) {
      if (node.OpType() == "SwiGLU") {
        swiglu_node = &node;
        break;
      }
    }

    if (swiglu_node == nullptr) {
      break;
    }

    const auto* op_schema = swiglu_node->Op();
    ASSERT_NE(op_schema, nullptr);
    EXPECT_TRUE(op_schema->HasFunction());
    ASSERT_STATUS_OK(expansion_graph.InlineFunction(*swiglu_node));
  }
  ASSERT_STATUS_OK(expansion_graph.Resolve());
  const auto expanded_ops = CountOpsInGraph(expansion_graph);
  EXPECT_EQ(expanded_ops.count("SwiGLU"), 0);
  EXPECT_EQ(expanded_ops.at("Swish"), 2);
  EXPECT_EQ(expanded_ops.at("Mul"), 2);

  InferenceSessionWrapper execution_session{session_options, GetEnvironment()};
  std::stringstream execution_model_stream(serialized_model);
  ASSERT_STATUS_OK(execution_session.Load(execution_model_stream, false));
  ASSERT_STATUS_OK(execution_session.Initialize());
  EXPECT_EQ(CountOpsInGraph(execution_session.GetGraph()).at("SwiGLU"), 2);

  auto provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  const auto allocator = provider->CreatePreferredAllocators()[0];
  OrtValue a;
  OrtValue b;
  CreateMLValue<float>(allocator, {2, 2}, {-2.0f, -1.0f, 0.0f, 2.0f}, &a);
  CreateMLValue<float>(allocator, {2, 2}, {0.5f, 2.0f, -3.0f, 4.0f}, &b);

  const std::string input_names[] = {"A", "B"};
  const std::string output_names[] = {"Y", "YDefault"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(execution_session.Run(RunOptions{}, AsSpan(input_names), AsSpan({a, b}),
                                         AsSpan(output_names), &fetches));

  ASSERT_EQ(fetches.size(), 2);
  const std::vector<std::vector<float>> expected{
      {-0.26894143f, -0.75508136f, 0.0f, 5.8484688f},
      {-0.11920292f, -0.53788286f, 0.0f, 7.0463767f},
  };
  for (size_t output_index = 0; output_index < fetches.size(); ++output_index) {
    const auto& output = fetches[output_index].Get<Tensor>();
    EXPECT_THAT(output.Shape().GetDims(), testing::ElementsAre(2, 2));
    const auto output_data = output.DataAsSpan<float>();
    ASSERT_EQ(output_data.size(), expected[output_index].size());
    for (size_t i = 0; i < expected[output_index].size(); ++i) {
      EXPECT_NEAR(output_data[i], expected[output_index][i], 1e-6f);
    }
  }
}

TEST(FunctionTest, SwiGLUOpset28RejectsMismatchedShapes) {
  constexpr const char* code = R"(
        <
        ir_version: 13,
        opset_import: [ "" : 28 ]
        >
        agraph (float[2, 2] A, float[2, 3] B) => (float[2, 2] Y)
        {
            Y = SwiGLU (A, B)
        }
        )";

  // Opset 28 is under development in ONNX 1.23, so allow shape inference to validate the model.
  const auto status = LoadModel(code, false);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Can't merge shape info"));
}

// Check that variables are renamed to avoid conflicts when multiple
// calls are inlined.
TEST(FunctionTest, Renaming) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y1 = local.myfun (x)
            y = local.myfun (y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {4.0, 8.0, 12.0});
}

// Check variable renaming in subgraphs.
// Scenario: input lx is used within subgraphs, but not in main graph.
// Both must be renamed to match the actual parameter name.
TEST(FunctionTest, InputInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (lx, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (lx, three)
                }
                >
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Check variable renaming in subgraphs.
// Scenario: intermediate temp is used within subgraphs, defined in main graph.
// Both must be renamed with a unique temporary name.
TEST(FunctionTest, TempInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (temp, three)
                }
                >
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Test a function body that calls another function.
TEST(FunctionTest, NestedCall) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (lx) => (ly) {
            one = Constant <value = float[1] {1.0}> ()
            tmp = local.twice (lx)
            ly = Add (tmp, one)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        twice (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 5.0, 7.0});
}

// Nested call inside a conditional statement.
TEST(FunctionTest, CallInConditional) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  local.MulFun (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  local.MulFun (temp, three)
                }
                >
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        MulFun (ax, bx) => (cx) {
            cx = Mul (ax, bx)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// A model-local function that declares zero inputs must not be invoked with
// actual inputs.
TEST(FunctionTest, RejectsZeroInputFunctionCalledWithInput) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[1] y)
        {
            y = local.zerofun (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        zerofun () => (ly) {
            ly = Constant <value = float[1] {2.0}> ()
        }
        )";

  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};
  std::stringstream sstr(serialized_model);
  const auto status = session_object.Load(sstr);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("declares no inputs"));
}

TEST(FunctionTest, RejectsSelfRecursiveLocalFunction) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.self_recursive (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        self_recursive (lx) => (ly) {
            ly = local.self_recursive (lx)
        }
        )";

  const auto status = LoadModel(code);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), HasCycleRejectionMessage());
}

TEST(FunctionTest, RejectsMutuallyRecursiveLocalFunctions) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.first (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        first (lx) => (ly) {
            ly = local.second (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        second (lx) => (ly) {
            ly = local.first (lx)
        }
        )";

  const auto status = LoadModel(code);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), HasCycleRejectionMessage());
}

TEST(FunctionTest, RejectsRecursionThroughSubgraph) {
  // A local function that calls itself inside an If subgraph (then_branch).
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.recursive_if (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        recursive_if (lx) => (ly) {
            temp = Identity (lx)
            cond = Constant <value = bool {1}> ()
            ly = If (cond) <
                then_branch = then_graph () => (float[N] then_out)
                {
                    then_out = local.recursive_if (temp)
                },
                else_branch = else_graph () => (float[N] else_out)
                {
                    else_out = Identity (temp)
                }
                >
        }
        )";

  const auto status = LoadModel(code);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("must not be recursive"));
}

// --- Synthetic adjacency-list tests for ValidateCallGraphAcyclic ---
// These test the cycle detection algorithm directly without constructing ONNX models.

static ONNX_NAMESPACE::ModelProto CreateNestedLocalFunctionModel(size_t depth, bool use_graphs_attribute) {
  ONNX_NAMESPACE::ModelProto model_proto;
  auto* nodes = model_proto.add_functions()->mutable_node();
  for (size_t i = 0; i < depth; ++i) {
    auto* node = nodes->Add();
    auto* attr = node->add_attribute();
    if (use_graphs_attribute) {
      nodes = attr->add_graphs()->mutable_node();
    } else {
      nodes = attr->mutable_g()->mutable_node();
    }
  }

  return model_proto;
}

static ONNX_NAMESPACE::ModelProto CreateNestedLocalFunctionDefaultAttributeModel(
    size_t depth, bool use_graphs_attribute) {
  ONNX_NAMESPACE::ModelProto model_proto;
  auto* function = model_proto.add_functions();
  auto* attr = function->add_attribute_proto();
  auto* graph = use_graphs_attribute ? attr->add_graphs() : attr->mutable_g();
  for (size_t i = 1; i < depth; ++i) {
    auto* node = graph->add_node();
    attr = node->add_attribute();
    graph = attr->mutable_g();
  }

  return model_proto;
}

TEST(FunctionTest, LocalFunctionSubgraphDepthValidated) {
  EXPECT_STATUS_OK(ValidateModelSubgraphDepth(
      CreateNestedLocalFunctionModel(kMaxModelSubgraphDepth, false)));
  EXPECT_EQ(ValidateModelSubgraphDepth(
                CreateNestedLocalFunctionModel(kMaxModelSubgraphDepth + 1, false))
                .Code(),
            common::NOT_IMPLEMENTED);
  EXPECT_EQ(ValidateModelSubgraphDepth(
                CreateNestedLocalFunctionModel(kMaxModelSubgraphDepth + 1, true))
                .Code(),
            common::NOT_IMPLEMENTED);
}

TEST(FunctionTest, LocalFunctionDefaultAttributeSubgraphDepthValidated) {
  EXPECT_STATUS_OK(ValidateModelSubgraphDepth(
      CreateNestedLocalFunctionDefaultAttributeModel(kMaxModelSubgraphDepth, false)));
  EXPECT_EQ(ValidateModelSubgraphDepth(
                CreateNestedLocalFunctionDefaultAttributeModel(kMaxModelSubgraphDepth + 1, false))
                .Code(),
            common::NOT_IMPLEMENTED);
  EXPECT_EQ(ValidateModelSubgraphDepth(
                CreateNestedLocalFunctionDefaultAttributeModel(kMaxModelSubgraphDepth + 1, true))
                .Code(),
            common::NOT_IMPLEMENTED);
}

TEST(FunctionTest, CallGraphAcyclic_EmptyGraph) {
  onnxruntime::LocalFunctionCallGraph call_graph;
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
}

TEST(FunctionTest, CallGraphAcyclic_SingleNodeNoCalls) {
  // Single function with no callees.
  std::string a = "A";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
}

TEST(FunctionTest, CallGraphAcyclic_SelfCycle) {
  std::string a = "A";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {a};
  const auto status = onnxruntime::ValidateCallGraphAcyclic(call_graph);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("must not be recursive"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("A -> A"));
}

TEST(FunctionTest, CallGraphAcyclic_MutualCycle) {
  std::string a = "A", b = "B";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {b};
  call_graph[b] = {a};
  const auto status = onnxruntime::ValidateCallGraphAcyclic(call_graph);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("must not be recursive"));
}

TEST(FunctionTest, CallGraphAcyclic_LongerCycle) {
  // A -> B -> C -> A
  std::string a = "A", b = "B", c = "C";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {b};
  call_graph[b] = {c};
  call_graph[c] = {a};
  const auto status = onnxruntime::ValidateCallGraphAcyclic(call_graph);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("must not be recursive"));
  // The cycle path should include all three participants.
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("A"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("B"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("C"));
}

TEST(FunctionTest, CallGraphAcyclic_DiamondNoCycle) {
  // A -> B, A -> C, B -> D, C -> D  (no cycle)
  std::string a = "A", b = "B", c = "C", d = "D";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {b, c};
  call_graph[b] = {d};
  call_graph[c] = {d};
  call_graph[d] = {};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
}

TEST(FunctionTest, CallGraphAcyclic_DeepChainNoCycle) {
  // A -> B -> C -> D  (no cycle)
  std::string a = "A", b = "B", c = "C", d = "D";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {b};
  call_graph[b] = {c};
  call_graph[c] = {d};
  call_graph[d] = {};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
}

TEST(FunctionTest, CallGraphAcyclic_MultipleIndependentCycles) {
  // Two independent cycles: A -> B -> A, C -> D -> C
  std::string a = "A", b = "B", c = "C", d = "D";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[a] = {b};
  call_graph[b] = {a};
  call_graph[c] = {d};
  call_graph[d] = {c};
  const auto status = onnxruntime::ValidateCallGraphAcyclic(call_graph);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("must not be recursive"));
}

TEST(FunctionTest, CallGraphAcyclic_SharedCallsDiamondNoCycle) {
  // Regression test: acyclic model with shared function calls (diamond pattern).
  // E -> A, E -> B, A -> C, B -> C, C -> D  (no cycle despite shared references to C)
  std::string a = "A", b = "B", c = "C", d = "D", e = "E";
  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[e] = {a, b};
  call_graph[a] = {c};
  call_graph[b] = {c};
  call_graph[c] = {d};
  call_graph[d] = {};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
}

TEST(FunctionTest, CallGraphDepth_MaximumDepthAccepted) {
  std::vector<std::string> function_ids;
  function_ids.reserve(onnxruntime::kMaxModelLocalFunctionCallDepth);
  for (size_t i = 0; i < onnxruntime::kMaxModelLocalFunctionCallDepth; ++i) {
    function_ids.push_back("function_" + std::to_string(i));
  }

  onnxruntime::LocalFunctionCallGraph call_graph;
  for (size_t i = 0; i < function_ids.size(); ++i) {
    auto& callees = call_graph[function_ids[i]];
    if (i + 1 < function_ids.size()) {
      callees.push_back(function_ids[i + 1]);
    }
  }

  const std::array<std::string_view, 1> roots{function_ids[0]};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphDepth(call_graph, roots));
}

TEST(FunctionTest, CallGraphDepth_UnreachableExcessiveDepthAccepted) {
  std::vector<std::string> function_ids;
  function_ids.reserve(onnxruntime::kMaxModelLocalFunctionCallDepth + 2);
  for (size_t i = 0; i < onnxruntime::kMaxModelLocalFunctionCallDepth + 2; ++i) {
    function_ids.push_back("function_" + std::to_string(i));
  }

  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[function_ids[0]] = {};
  for (size_t i = 1; i < function_ids.size(); ++i) {
    auto& callees = call_graph[function_ids[i]];
    if (i + 1 < function_ids.size()) {
      callees.push_back(function_ids[i + 1]);
    }
  }

  const std::array<std::string_view, 1> roots{function_ids[0]};
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphDepth(call_graph, roots));
}

TEST(FunctionTest, CallGraphDepth_ExcessiveSharedTailRejected) {
  std::vector<std::string> function_ids;
  function_ids.reserve(onnxruntime::kMaxModelLocalFunctionCallDepth + 2);
  for (size_t i = 0; i < onnxruntime::kMaxModelLocalFunctionCallDepth + 2; ++i) {
    function_ids.push_back("function_" + std::to_string(i));
  }

  onnxruntime::LocalFunctionCallGraph call_graph;
  call_graph[function_ids[0]] = {function_ids[2]};
  call_graph[function_ids[1]] = {function_ids[0], function_ids[2]};
  for (size_t i = 2; i < function_ids.size(); ++i) {
    auto& callees = call_graph[function_ids[i]];
    if (i + 1 < function_ids.size()) {
      callees.push_back(function_ids[i + 1]);
    }
  }

  ASSERT_STATUS_OK(onnxruntime::ValidateCallGraphAcyclic(call_graph));
  const std::array<std::string_view, 1> roots{function_ids[1]};
  const auto status = onnxruntime::ValidateCallGraphDepth(call_graph, roots);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

static ONNX_NAMESPACE::ModelProto CreateLocalFunctionChainModel(size_t call_depth) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain(onnxruntime::kOnnxDomain);
  onnx_opset->set_version(16);
  auto* local_opset = model_proto.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);

  auto* graph = model_proto.mutable_graph();
  graph->set_name("local_function_chain");
  auto* graph_input = graph->add_input();
  graph_input->set_name("x");
  graph_input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* graph_output = graph->add_output();
  graph_output->set_name("y");
  graph_output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* graph_node = graph->add_node();
  graph_node->set_domain("local");
  graph_node->set_op_type("function_0");
  graph_node->add_input("x");
  graph_node->add_output("y");

  for (size_t i = 0; i < call_depth; ++i) {
    auto* function = model_proto.add_functions();
    function->set_domain("local");
    function->set_name("function_" + std::to_string(i));
    function->add_input("x");
    function->add_output("y");
    auto* function_onnx_opset = function->add_opset_import();
    function_onnx_opset->set_domain(onnxruntime::kOnnxDomain);
    function_onnx_opset->set_version(16);
    auto* function_local_opset = function->add_opset_import();
    function_local_opset->set_domain("local");
    function_local_opset->set_version(1);

    auto* node = function->add_node();
    node->set_op_type(i + 1 < call_depth ? "function_" + std::to_string(i + 1) : "Identity");
    if (i + 1 < call_depth) {
      node->set_domain("local");
    }
    node->add_input("x");
    node->add_output("y");
  }

  return model_proto;
}

static ONNX_NAMESPACE::ModelProto CreateRepeatedLocalFunctionCallDagModel(
    size_t call_depth, bool graph_references_alpha = false) {
  auto model_proto = CreateLocalFunctionChainModel(call_depth);
  const auto populate_graph_attribute = [graph_references_alpha](ONNX_NAMESPACE::AttributeProto& attribute) {
    attribute.set_name("tag");
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* graph = attribute.mutable_g();
    graph->set_name("tag");
    auto* identity = graph->add_node();
    identity->set_op_type("Identity");
    identity->add_input("x");
    identity->add_output(graph_references_alpha ? "tag_identity_output" : "tag_output");
    if (graph_references_alpha) {
      auto* leaky_relu = graph->add_node();
      leaky_relu->set_op_type("LeakyRelu");
      leaky_relu->add_input("tag_identity_output");
      leaky_relu->add_output("tag_output");
      auto* alpha = leaky_relu->add_attribute();
      alpha->set_name("alpha");
      alpha->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      alpha->set_ref_attr_name("alpha");
    }
    auto* output = graph->add_output();
    output->set_name("tag_output");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  };
  const auto add_scalar_attribute = [](ONNX_NAMESPACE::NodeProto& node) {
    auto* attribute = node.add_attribute();
    attribute->set_name("alpha");
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attribute->set_f(0.1f);
  };
  const auto add_graph_attribute = [&populate_graph_attribute](ONNX_NAMESPACE::NodeProto& node) {
    populate_graph_attribute(*node.add_attribute());
  };

  for (auto& function : *model_proto.mutable_functions()) {
    function.add_attribute("alpha");
    auto* default_scalar_attribute = function.add_attribute_proto();
    default_scalar_attribute->set_name("alpha");
    default_scalar_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    default_scalar_attribute->set_f(0.1f);

    function.add_attribute("tag");
    populate_graph_attribute(*function.add_attribute_proto());

    auto* graph_attribute_user = function.add_node();
    graph_attribute_user->set_op_type("Identity");
    graph_attribute_user->add_input("x");
    graph_attribute_user->add_output("tag_ref_output");
    auto* graph_attribute_reference = graph_attribute_user->add_attribute();
    graph_attribute_reference->set_name("graph");
    graph_attribute_reference->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    graph_attribute_reference->set_ref_attr_name("tag");
  }

  for (int i = 0; i + 1 < model_proto.functions_size(); ++i) {
    auto* function = model_proto.mutable_functions(i);
    function->mutable_node(0)->set_output(0, "unused");
    add_scalar_attribute(*function->mutable_node(0));
    add_graph_attribute(*function->mutable_node(0));

    auto* repeated_call = function->add_node();
    repeated_call->set_domain("local");
    repeated_call->set_op_type("function_" + std::to_string(i + 1));
    repeated_call->add_input("x");
    repeated_call->add_output("y");
    add_scalar_attribute(*repeated_call);
    add_graph_attribute(*repeated_call);
  }

  return model_proto;
}

static std::shared_ptr<OnnxRuntimeOpSchemaRegistry> CreateLocalFunctionCollisionRegistry() {
  auto registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas{
      ONNX_NAMESPACE::OpSchema()
          .SetName("function_0")
          .SetDomain("local")
          .Input(0, "X", "", "T")
          .Output(0, "Y", "", "T")
          .TypeConstraint("T", ONNX_NAMESPACE::OpSchema::all_tensor_types(), "")};
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, "local", 0, 1));
  return registry;
}

static void WrapLocalFunctionChainInReferencedGraphAttribute(
    ONNX_NAMESPACE::ModelProto& model_proto);
static ONNX_NAMESPACE::AttributeProto MakeGraphRefAttribute(
    const std::string& name, const std::string& ref_attr_name,
    ONNX_NAMESPACE::AttributeProto_AttributeType type);

TEST(FunctionTest, RegisteredSchemaTakesPrecedenceOverCollidingRootLocalFunction) {
  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth + 1);
  IOnnxRuntimeOpSchemaRegistryList registries{CreateLocalFunctionCollisionRegistry()};
  Model model(std::move(model_proto), &registries, DefaultLoggingManager().DefaultLogger());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
}

TEST(FunctionTest, FunctionInferenceRegistryTakesPrecedenceInsideLocalFunctionBody) {
  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth + 1);
  model_proto.mutable_graph()->mutable_node(0)->set_op_type("wrapper");

  auto* wrapper = model_proto.add_functions();
  wrapper->set_domain("local");
  wrapper->set_name("wrapper");
  wrapper->add_input("x");
  wrapper->add_output("y");
  auto* onnx_opset = wrapper->add_opset_import();
  onnx_opset->set_domain(kOnnxDomain);
  onnx_opset->set_version(16);
  auto* local_opset = wrapper->add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);
  auto* collision_node = wrapper->add_node();
  collision_node->set_domain("local");
  collision_node->set_op_type("function_0");
  collision_node->add_input("x");
  collision_node->add_output("y");

  IOnnxRuntimeOpSchemaRegistryList registries{CreateLocalFunctionCollisionRegistry()};
  Model model(std::move(model_proto), &registries, DefaultLoggingManager().DefaultLogger());
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, FunctionInferenceRegistryTakesPrecedenceInsideBoundGraph) {
  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth + 1);
  WrapLocalFunctionChainInReferencedGraphAttribute(model_proto);

  IOnnxRuntimeOpSchemaRegistryList registries{CreateLocalFunctionCollisionRegistry()};
  Model model(std::move(model_proto), &registries, DefaultLoggingManager().DefaultLogger());
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, BoundGraphExpansionCacheIncludesFunctionOpsetImports) {
  for (const bool registered_schema_first : {false, true}) {
    SCOPED_TRACE(registered_schema_first);
    auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth);
    model_proto.mutable_opset_import(0)->set_version(12);
    auto* root_call = model_proto.mutable_graph()->mutable_node(0);
    root_call->set_op_type("wrapper");
    root_call->add_input("cond");
    auto* cond_input = model_proto.mutable_graph()->add_input();
    cond_input->set_name("cond");
    cond_input->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_BOOL);

    auto add_opsets = [](ONNX_NAMESPACE::FunctionProto& function, int onnx_opset) {
      auto* onnx_import = function.add_opset_import();
      onnx_import->set_domain(kOnnxDomain);
      onnx_import->set_version(onnx_opset);
      auto* local_import = function.add_opset_import();
      local_import->set_domain("local");
      local_import->set_version(1);
    };
    auto add_graph_ref = [](ONNX_NAMESPACE::NodeProto& node, const char* attribute_name) {
      *node.add_attribute() = MakeGraphRefAttribute(
          attribute_name, "body", ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    };

    auto* celu = model_proto.add_functions();
    celu->set_domain(kOnnxDomain);
    celu->set_name("Celu");
    celu->add_input("x");
    celu->add_output("y");
    add_opsets(*celu, 11);
    auto* celu_node = celu->add_node();
    celu_node->set_domain("local");
    celu_node->set_op_type("function_0");
    celu_node->add_input("x");
    celu_node->add_output("y");

    for (const int opset : {11, 12}) {
      auto* visitor = model_proto.add_functions();
      visitor->set_domain("local");
      visitor->set_name("visit_" + std::to_string(opset));
      visitor->add_input("x");
      visitor->add_input("cond");
      visitor->add_output("y");
      visitor->add_attribute("body");
      add_opsets(*visitor, opset);
      auto* visitor_node = visitor->add_node();
      visitor_node->set_op_type("If");
      visitor_node->add_input("cond");
      visitor_node->add_output("y");
      add_graph_ref(*visitor_node, "then_branch");
      auto* else_branch = visitor_node->add_attribute();
      else_branch->set_name("else_branch");
      else_branch->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      auto* else_graph = else_branch->mutable_g();
      else_graph->set_name("else_branch");
      auto* else_node = else_graph->add_node();
      else_node->set_op_type("Identity");
      else_node->add_input("x");
      else_node->add_output("y");
      auto* else_output = else_graph->add_output();
      else_output->set_name("y");
      else_output->mutable_type()->mutable_tensor_type()->set_elem_type(
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    }

    auto* wrapper = model_proto.add_functions();
    wrapper->set_domain("local");
    wrapper->set_name("wrapper");
    wrapper->add_input("x");
    wrapper->add_input("cond");
    wrapper->add_output("y");
    wrapper->add_attribute("body");
    auto* default_body = wrapper->add_attribute_proto();
    default_body->set_name("body");
    default_body->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* body_graph = default_body->mutable_g();
    body_graph->set_name("shared_body");
    auto* body_output = body_graph->add_output();
    body_output->set_name("y");
    body_output->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* body_node = body_graph->add_node();
    body_node->set_op_type("Celu");
    body_node->add_input("x");
    body_node->add_output("y");
    add_opsets(*wrapper, 12);

    const int first_opset = registered_schema_first ? 12 : 11;
    for (const int opset : {first_opset, 23 - first_opset}) {
      auto* call = wrapper->add_node();
      call->set_domain("local");
      call->set_op_type("visit_" + std::to_string(opset));
      call->add_input(opset == first_opset ? "x" : "intermediate");
      call->add_input("cond");
      call->add_output(opset == first_opset ? "intermediate" : "y");
      add_graph_ref(*call, "body");
    }

    Model model(std::move(model_proto), nullptr, DefaultLoggingManager().DefaultLogger());
    const auto status = model.MainGraph().Resolve();
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
  }
}

TEST(FunctionTest, UnusedCallSiteGraphContributesToFunctionDepth) {
  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth + 1);
  auto* root_call = model_proto.mutable_graph()->mutable_node(0);
  root_call->set_op_type("wrapper");

  auto* unused_graph_attribute = root_call->add_attribute();
  unused_graph_attribute->set_name("unused_tag");
  unused_graph_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* unused_graph = unused_graph_attribute->mutable_g();
  unused_graph->set_name("unused_tag");
  auto* deep_call = unused_graph->add_node();
  deep_call->set_domain("local");
  deep_call->set_op_type("function_0");
  deep_call->add_input("x");
  deep_call->add_output("y");
  auto* unused_graph_output = unused_graph->add_output();
  unused_graph_output->set_name("y");
  unused_graph_output->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* wrapper = model_proto.add_functions();
  wrapper->set_domain("local");
  wrapper->set_name("wrapper");
  wrapper->add_input("x");
  wrapper->add_output("y");
  auto* onnx_opset = wrapper->add_opset_import();
  onnx_opset->set_domain(kOnnxDomain);
  onnx_opset->set_version(16);
  auto* wrapper_node = wrapper->add_node();
  wrapper_node->set_op_type("Identity");
  wrapper_node->add_input("x");
  wrapper_node->add_output("y");

  Model model(
      std::move(model_proto), nullptr,
      DefaultLoggingManager().DefaultLogger());
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, RepeatedLocalFunctionCallDagDepthValidationCompletes) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(CreateRepeatedLocalFunctionCallDagModel(30), nullptr, logger);
  ASSERT_STATUS_OK(model.ValidateLocalFunctionCallDepth(model.MainGraph()));
}

TEST(FunctionTest, RepeatedLocalFunctionCallDagWithReferencedGraphDepthValidationCompletes) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(CreateRepeatedLocalFunctionCallDagModel(30, true), nullptr, logger);
  ASSERT_STATUS_OK(model.ValidateLocalFunctionCallDepth(model.MainGraph()));
}

static void WrapLocalFunctionChainInReferencedGraphAttribute(ONNX_NAMESPACE::ModelProto& model_proto) {
  auto* graph = model_proto.mutable_graph();
  auto* condition = graph->add_input();
  condition->set_name("condition");
  condition->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto* root_node = graph->mutable_node(0);
  root_node->set_op_type("wrapper");
  root_node->clear_input();
  root_node->add_input("x");
  root_node->add_input("condition");

  auto* body_attr = root_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* body_graph = body_attr->mutable_g();
  body_graph->set_name("body");
  auto* function_call = body_graph->add_node();
  function_call->set_domain("local");
  function_call->set_op_type("function_0");
  function_call->add_input("x");
  function_call->add_output("body_output");
  auto* body_output = body_graph->add_output();
  body_output->set_name("body_output");
  body_output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* wrapper = model_proto.add_functions();
  wrapper->set_domain("local");
  wrapper->set_name("wrapper");
  wrapper->add_input("x");
  wrapper->add_input("condition");
  wrapper->add_output("y");
  wrapper->add_attribute("body");
  auto* wrapper_onnx_opset = wrapper->add_opset_import();
  wrapper_onnx_opset->set_domain(onnxruntime::kOnnxDomain);
  wrapper_onnx_opset->set_version(16);
  auto* wrapper_local_opset = wrapper->add_opset_import();
  wrapper_local_opset->set_domain("local");
  wrapper_local_opset->set_version(1);

  auto* if_node = wrapper->add_node();
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("y");
  auto* then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  then_attr->set_ref_attr_name("body");
  auto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* else_graph = else_attr->mutable_g();
  else_graph->set_name("else");
  auto* identity = else_graph->add_node();
  identity->set_op_type("Identity");
  identity->add_input("x");
  identity->add_output("else_output");
  auto* else_output = else_graph->add_output();
  else_output->set_name("else_output");
  else_output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

TEST(FunctionTest, ReferencedGraphAttributeContributesToLocalFunctionDepth) {
  auto accepted_model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth - 1);
  WrapLocalFunctionChainInReferencedGraphAttribute(accepted_model_proto);
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model accepted_model(std::move(accepted_model_proto), nullptr, logger);
  ASSERT_STATUS_OK(accepted_model.MainGraph().Resolve());

  auto rejected_model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth);
  WrapLocalFunctionChainInReferencedGraphAttribute(rejected_model_proto);
  Model rejected_model(std::move(rejected_model_proto), nullptr, logger);
  const auto status = rejected_model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, DefaultGraphAttributeContributesToLocalFunctionDepth) {
  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth);
  WrapLocalFunctionChainInReferencedGraphAttribute(model_proto);

  auto* root_node = model_proto.mutable_graph()->mutable_node(0);
  auto* wrapper = model_proto.mutable_functions(model_proto.functions_size() - 1);
  *wrapper->add_attribute_proto() = root_node->attribute(0);
  root_node->clear_attribute();

  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(std::move(model_proto), nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

static ONNX_NAMESPACE::ModelProto CreateForwardedNestedGraphAttributeModel(
    size_t chain_depth = kMaxModelLocalFunctionCallDepth) {
  auto model_proto = CreateLocalFunctionChainModel(chain_depth);
  auto* root_call = model_proto.mutable_graph()->mutable_node(0);
  root_call->set_op_type("F");

  auto* forwarded_attr = root_call->add_attribute();
  forwarded_attr->set_name("A");
  forwarded_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* nested_if = forwarded_attr->mutable_g()->add_node();
  nested_if->set_op_type("If");
  auto* nested_ref = nested_if->add_attribute();
  nested_ref->set_name("then_branch");
  nested_ref->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  nested_ref->set_ref_attr_name("B");

  auto* chain_attr = root_call->add_attribute();
  chain_attr->set_name("B");
  chain_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* chain_call = chain_attr->mutable_g()->add_node();
  chain_call->set_domain("local");
  chain_call->set_op_type("function_0");

  auto* forwarder = model_proto.add_functions();
  forwarder->set_domain("local");
  forwarder->set_name("F");
  forwarder->add_attribute("A");
  forwarder->add_attribute("B");
  auto* forwarder_onnx_opset = forwarder->add_opset_import();
  forwarder_onnx_opset->set_domain(onnxruntime::kOnnxDomain);
  forwarder_onnx_opset->set_version(16);
  auto* forwarder_local_opset = forwarder->add_opset_import();
  forwarder_local_opset->set_domain("local");
  forwarder_local_opset->set_version(1);
  auto* forwarded_call = forwarder->add_node();
  forwarded_call->set_domain("local");
  forwarded_call->set_op_type("G");
  auto* forwarded_binding = forwarded_call->add_attribute();
  forwarded_binding->set_name("body");
  forwarded_binding->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  forwarded_binding->set_ref_attr_name("A");

  auto* consumer = model_proto.add_functions();
  consumer->set_domain("local");
  consumer->set_name("G");
  consumer->add_attribute("body");
  auto* consumer_onnx_opset = consumer->add_opset_import();
  consumer_onnx_opset->set_domain(onnxruntime::kOnnxDomain);
  consumer_onnx_opset->set_version(16);
  auto* consumer_local_opset = consumer->add_opset_import();
  consumer_local_opset->set_domain("local");
  consumer_local_opset->set_version(1);
  auto* if_node = consumer->add_node();
  if_node->set_op_type("If");
  auto* body_ref = if_node->add_attribute();
  body_ref->set_name("then_branch");
  body_ref->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  body_ref->set_ref_attr_name("body");

  return model_proto;
}

static ONNX_NAMESPACE::ModelProto CreateDirectNestedGraphAttributeModel(
    size_t chain_depth = kMaxModelLocalFunctionCallDepth) {
  auto model_proto = CreateForwardedNestedGraphAttributeModel(chain_depth);
  auto* root_call = model_proto.mutable_graph()->mutable_node(0);
  auto* forwarder = model_proto.mutable_functions(model_proto.functions_size() - 2);
  auto* direct_binding = forwarder->mutable_node(0)->mutable_attribute(0);
  direct_binding->clear_ref_attr_name();
  *direct_binding->mutable_g() = root_call->attribute(0).g();
  return model_proto;
}

TEST(FunctionTest, ForwardedGraphAttributePreservesNestedReferenceBindings) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(CreateForwardedNestedGraphAttributeModel(), nullptr, logger);
  const auto status = model.ValidateLocalFunctionCallDepth(model.MainGraph());
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, DirectGraphAttributePreservesNestedReferenceBindings) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(
      CreateDirectNestedGraphAttributeModel(kMaxModelLocalFunctionCallDepth - 1),
      nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

static ONNX_NAMESPACE::ModelProto CreateTerminatingCrossBoundGraphAttributeModel() {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain(onnxruntime::kOnnxDomain);
  onnx_opset->set_version(16);
  auto* local_opset = model_proto.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);
  auto* graph = model_proto.mutable_graph();
  graph->set_name("terminating_cross_bound_graph_attributes");

  for (const auto* function_name : {"F", "G"}) {
    auto* function = model_proto.add_functions();
    function->set_domain("local");
    function->set_name(function_name);
    function->add_attribute("body");
    auto* function_onnx_opset = function->add_opset_import();
    function_onnx_opset->set_domain(onnxruntime::kOnnxDomain);
    function_onnx_opset->set_version(16);
    auto* function_local_opset = function->add_opset_import();
    function_local_opset->set_domain("local");
    function_local_opset->set_version(1);

    auto* if_node = function->add_node();
    if_node->set_op_type("If");
    auto* body_attr = if_node->add_attribute();
    body_attr->set_name("then_branch");
    body_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    body_attr->set_ref_attr_name("body");
  }

  const auto add_bound_call = [&](const char* function_name, const char* nested_function_name) {
    auto* call = graph->add_node();
    call->set_domain("local");
    call->set_op_type(function_name);
    auto* body_attr = call->add_attribute();
    body_attr->set_name("body");
    body_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* nested_call = body_attr->mutable_g()->add_node();
    nested_call->set_domain("local");
    nested_call->set_op_type(nested_function_name);
  };
  add_bound_call("F", "G");
  add_bound_call("G", "F");

  return model_proto;
}

TEST(FunctionTest, CallSiteGraphBindingsDoNotCreateGlobalCycle) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(CreateTerminatingCrossBoundGraphAttributeModel(), nullptr, logger);
  ASSERT_STATUS_OK(model.ValidateLocalFunctionCallDepth(model.MainGraph()));
}

TEST(FunctionTest, LoadFromBytes_ExcessiveLocalFunctionDepthReturnsStatus) {
  auto model_proto = CreateLocalFunctionChainModel(onnxruntime::kMaxModelLocalFunctionCallDepth + 1);
  std::string serialized_model;
  ASSERT_TRUE(model_proto.SerializeToString(&serialized_model));

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  const auto status = Model::LoadFromBytes(static_cast<int>(serialized_model.size()), serialized_model.data(),
                                           model, nullptr, logger);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, LocalFunctionDepthValidationPreservesGraphProtoSyncFlag) {
  auto model_proto = CreateLocalFunctionChainModel(1);
  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  auto& graph = model->MainGraph();
  graph.SetGraphResolveNeeded().SetGraphProtoSyncNeeded();
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_TRUE(graph.GraphProtoSyncNeeded());
}

TEST(FunctionTest, FailedLocalFunctionDepthValidationPreservesGraphProtoSyncFlag) {
  auto model_proto = CreateLocalFunctionChainModel(onnxruntime::kMaxModelLocalFunctionCallDepth + 1);
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(std::move(model_proto), nullptr, logger);

  auto& graph = model.MainGraph();
  graph.SetGraphResolveNeeded().SetGraphProtoSyncNeeded();
  const auto status = graph.Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_TRUE(graph.GraphProtoSyncNeeded());
}

TEST(FunctionTest, ExcessiveLocalFunctionDepthInGraphsAttributeReturnsStatus) {
  auto model_proto = CreateLocalFunctionChainModel(onnxruntime::kMaxModelLocalFunctionCallDepth + 1);
  auto* root_node = model_proto.mutable_graph()->mutable_node(0);
  root_node->set_domain(onnxruntime::kOnnxDomain);
  root_node->set_op_type("Identity");
  auto* graphs_attr = root_node->add_attribute();
  graphs_attr->set_name("graphs");
  graphs_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS);
  auto* function_call = graphs_attr->add_graphs()->add_node();
  function_call->set_domain("local");
  function_call->set_op_type("function_0");

  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(std::move(model_proto), nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, ExcessiveLocalFunctionDepthInUntypedGraphAttributeReturnsStatus) {
  auto model_proto = CreateLocalFunctionChainModel(onnxruntime::kMaxModelLocalFunctionCallDepth + 1);
  auto* root_node = model_proto.mutable_graph()->mutable_node(0);
  root_node->set_domain(onnxruntime::kOnnxDomain);
  root_node->set_op_type("Identity");
  auto* graph_attr = root_node->add_attribute();
  graph_attr->set_name("graph");
  auto* function_call = graph_attr->mutable_g()->add_node();
  function_call->set_domain("local");
  function_call->set_op_type("function_0");

  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(std::move(model_proto), nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

// --- Model-level integration tests ---

TEST(FunctionTest, RejectsLongerCycle) {
  // A -> B -> C -> A (three-function cycle)
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.func_a (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_a (lx) => (ly) {
            ly = local.func_b (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_b (lx) => (ly) {
            ly = local.func_c (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_c (lx) => (ly) {
            ly = local.func_a (lx)
        }
        )";

  const auto status = LoadModel(code);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), HasCycleRejectionMessage());
}

TEST(FunctionTest, AcceptsAcyclicDiamond) {
  // A -> B, A -> C, B -> D, C -> D (diamond, no cycle)
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.func_a (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_a (lx) => (ly) {
            t1 = local.func_b (lx)
            ly = local.func_c (t1)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_b (lx) => (ly) {
            ly = local.func_d (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_c (lx) => (ly) {
            ly = local.func_d (lx)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        func_d (lx) => (ly) {
            ly = Identity (lx)
        }
        )";

  ASSERT_STATUS_OK(LoadModel(code));
}

TEST(FunctionTest, AcceptsTrivialSingleNodeFunction) {
  // A local function with a single Identity node — verifies that trivial
  // (but non-empty) function bodies pass acyclicity validation.
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.trivial_func (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        trivial_func (lx) => (ly) {
            ly = Identity (lx)
        }
        )";

  ASSERT_STATUS_OK(LoadModel(code));
}

TEST(FunctionTest, RejectsMultipleIndependentCycles) {
  // Two independent cycles in the same model: A -> B -> A, C -> D -> C
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            t = local.func_a (x)
            y = local.func_c (t)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_a (lx) => (ly) {
            ly = local.func_b (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_b (lx) => (ly) {
            ly = local.func_a (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_c (lx) => (ly) {
            ly = local.func_d (lx)
        }

        <
        opset_import: [ "" : 16, "local" : 1 ],
        domain: "local"
        >
        func_d (lx) => (ly) {
            ly = local.func_c (lx)
        }
        )";

  const auto status = LoadModel(code);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), HasCycleRejectionMessage());
}

// Test use of attibute references, especially where source/target attribute
// names are not the same. In this example, the "start : int = @s" attribute-reference
// binds the attribute named "start" of the Shape op to the attribute named "s"
// of the containing function myfun.
TEST(FunctionTest, AttrName) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun <s = 0> (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun <s> (lx) => (ly) {
            d = Shape <start : int = @s> (lx)
            df = Cast <to = 1> (d)
            ly = Mul (lx, df)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 6.0, 9.0});
}

// Test function with attribute that has default value.
TEST(FunctionTest, AttrWithDefault) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y0 = local.myfun <a = 2.0> (x)
            y1 = local.myfun (x)
            y = Add (y0, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun <a: float=1.0> (x) => (y) {
            x2 = Constant <value_float: float=@a>()
            x3 = CastLike (x2, x)
            y = Add (x, x3)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {5.0, 7.0, 9.0});
}

#if !defined(DISABLE_FLOAT8_TYPES)

// Attribute 'saturate' was introduced in opset 19, ir_version=9.
// The test checks the parser gets it right and returns the expected results.
TEST(FunctionTest, AttrSaturate) {
  const char* code = R"(
        <
        ir_version: 9,
        opset_import: [ "" : 19, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y0 = local.myfun <a = 2.0> (x)
            y1 = local.myfun (x)
            y = Add (y0, y1)
        }

        <
        opset_import: [ "" : 19 ],
        domain: "local"
        >
        myfun <a: float=1.0> (x) => (y) {
            x2 = Constant <value_float: float=@a>()
            x2_ = Cast<to=17>(x2)
            x3 = CastLike<saturate=0>(x2, x2_)
            x3_ = Cast<to=1>(x3)
            y = Add (x, x3_)
        }
        )";

  Check(code, "x", {1.0, 2.0, 1e6}, "y", {5.0, 7.0, 2000003.0});
}

// Attribute 'saturate' was introduced in opset 19, ir_version=9.
// The test checks the model does not saturate a value out of float 8 boundary.
// TODO: change the expected value when this PR is merged in onnx:
// https://github.com/onnx/onnx/pull/5246
TEST(FunctionTest, AttrSaturateNan) {
  const char* code = R"(
        <
        ir_version: 9,
        opset_import: [ "" : 19, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            x_E4M3FNUZ = Cast<to=18>(x)
            x_E4M3FNUZ_2 = CastLike<saturate=0>(x, x_E4M3FNUZ)  # NaN when OOR
            y = Cast<to=1>(x_E4M3FNUZ_2)
        }
        )";

  Check(code, "x", {1.0, 2.0, 1e6}, "y", {1.0, 2.0, std::numeric_limits<float>::quiet_NaN()});
}

#endif

// Test use of constants inside sub-graphs, which are promoted to initializers by ORT.
TEST(FunctionTest, NestedConstant) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 17 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            xseq = SequenceConstruct (x)
            yseq = SequenceMap (xseq) <body =
              zeropad (float[3] lx) => (float[6] ly) {
                zeros = Constant <value = float[3] {0.0, 0.0, 0.0}> ()
                ly = Concat <axis = 0> (lx, zeros)
              }>
            zero = Constant <value = int64{0}> ()
            y = SequenceAt (yseq, zero)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 2.0, 3.0, 0.0, 0.0, 0.0});
}

// GH13121. Model with function body that has variadic inputs (or outputs) was not loading.
// Add handling for variadics to IOTypeConstraintHelper. Test model has a Concat and Split to test both variadic
// inputs and outputs.
TEST(FunctionTest, Variadics) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/function_with_variadics.onnx");

  SessionOptions so;
  so.session_logid = "FunctionTest.Variadics";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());
}

// A variation of the variadics issue above, where the first input/output of the
// variadic list is NOT an input/output of the function.
TEST(FunctionTest, VariadicsNonInputOutput) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 17, "local" : 1]>
    mymodel (float[2] x) => (float[3] y) {
      y = local.func (x)
    }

    <opset_import: ["" : 17 ],  domain: "local">
    func (a) => (y) {
      b = Identity(a)
      z = Concat <axis = 0> (b, a, b)
      y, w = Split (z)
    }
  )";

  Check(code, "x", {1.0, 2.0}, "y", {1.0, 2.0, 1.0});
}

// Test use of outer-scope names inside sub-graphs in functions that are inlined.
TEST(FunctionTest, OuterScopeName) {
  const char* code = R"(
        <ir_version: 8, opset_import: [ "" : 17 ]>
        agraph (float[N] x) => (float[N] y)
        {
            xseq = SequenceConstruct (x)
            zeros = Constant <value = float[3] {0.0, 0.0, 0.0}> ()
            yseq = SequenceMap (xseq) <body =
              zeropad (float[3] lx) => (float[6] ly) {
                ly = Concat <axis = 0> (lx, zeros)
              }>
            zero = Constant <value = int64{0}> ()
            y = SequenceAt (yseq, zero)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 2.0, 3.0, 0.0, 0.0, 0.0});
}

// Test use of functions with unused inputs:
TEST(FunctionTest, UnusedFunctionInputs) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 17, "local" : 1]>
    mymodel (float[3] x) => (float[3] y) {
      y = local.func (x, x, x)
    }

    <opset_import: ["" : 17 ],  domain: "local">
    func (a, b, c) => (y) {
      y = Mul (a, b)
    }
  )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 4.0, 9.0});
}

// Test constant-folding inside a sub-graph is handled correctly
// for functions that are inlined.
TEST(FunctionTest, ConstantFoldingInSubGraph) {
  const char* code = R"(
    <ir_version: 8, opset_import: [ "" : 17 ]>
    agraph (float[N] X) => (float[M] Y)  {
        seq1 = SequenceConstruct(X, X, X)
        seq2 = SequenceMap (seq1) <body =
            add1 (float[K] Z) => (float[K] W) {
                C1 = Constant <value = float {1.0}> ()
                C2 = Constant <value = float {1.0}> ()
                # C is a constant, which will be constant-folded into an initializer out of the sub-graph.
                C = Add (C1, C2)
                # After optimization, only following Add will be left in this sub-graph.
                W = Add (Z, C)
            }
        >
        Y = ConcatFromSequence <axis=0> (seq2)
    }
  )";

  Check(code, "X", {1.0, 2.0, 3.0}, "Y", {3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0});
}

TEST(FunctionTest, TestInlinedLocalFunctionRemoved) {
  std::string serialized_model;
  ParseOnnxSource(basic_code, serialized_model);

  // Default is to do AOT Function inlining
  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));

  auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());

  ASSERT_STATUS_OK(session_object.Initialize());

  // All functions removed
  model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(0, model_proto.functions_size());
}

TEST(FunctionTest, TestInlinedLocalFunctionNotRemoved) {
  std::string serialized_model;
  ParseOnnxSource(basic_code, serialized_model);

  // Default is to do AOT Function inlining
  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NCHW);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(internal_testing_ep)));

  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));

  auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());

  ASSERT_STATUS_OK(session_object.Initialize());

  // myfun is not removed because it was claimed by InternalTestingEP
  model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());
}

TEST(FunctionTest, TestInlinedFunctionDoesNotReserrectNonExistingArgs) {
  // Verify this runs
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/transform/gh_issue_18338.onnx");

  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Scalar shape for input_0 and output
  const std::string input_names[] = {"input_0"};
  const std::string output_names[] = {"_val_3"};
  TensorShape input_shape;
  MLFloat16 input_0_data{684.f};

  OrtValue input_0;
  Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), input_shape, &input_0_data, OrtMemoryInfo(), input_0);

  std::vector<OrtValue> fetches(1);
  RunOptions run_options;
  ASSERT_STATUS_OK(session_object.Run(run_options, AsSpan(input_names), AsSpan({input_0}),
                                      AsSpan(output_names), &fetches, 0));
}

/// <summary>
/// This test covers the issues:
/// https://github.com/microsoft/onnxruntime/issues/16438
/// https://github.com/microsoft/onnxruntime/issues/18781
/// </summary>
TEST(FunctionTest, Test_GH_issue_16438) {
  const char* code = R"(
    <
       ir_version: 8,
       opset_import: ["pkg.onnxscript.torch_lib" : 1, "" : 18],
       producer_name: "pytorch",
       producer_version: "2.1.0"
    >
    torch_jit (float16[5,10,5] input_0) => (double[5,10,5] _val_1) {
       _val_1 = pkg.onnxscript.torch_lib.aten_special_log_softmax <dim: int = 2, dtype: int = 11> (input_0)
    }
    <
      domain: "pkg.onnxscript.torch_lib",
      opset_import: ["" : 18]
    >
    aten_special_log_softmax <dim, dtype>(self) => (result_8)
    {
      tmp = Shape(self)
      tmp_0 = Size(tmp)
      int64_0 = Constant<value : tensor = int64 int64_0{0}> ()
      int64_0_cast = CastLike(int64_0, tmp_0)
      self_is_scalar = Equal(tmp_0, int64_0_cast)
      self_4 = If(self_is_scalar) <then_branch : graph = thenGraph_8() => (self_2) {
        tmp_1 = Constant<value_ints : ints = [0]> ()
        self_2 = Unsqueeze(self, tmp_1)
      }, else_branch : graph = elseGraph_8() => (self_3) {
        self_3 = Identity(self)
      }>
      result = LogSoftmax<axis : int = @dim>(self_4)
      result_5 = Cast<to : int = @dtype>(result)
      result_8 = If(self_is_scalar) <then_branch : graph = thenGraph_12() => (result_6) {
       result_6 = Squeeze(result_5)
      }, else_branch : graph = elseGraph_12() => (result_7) {
        result_7 = Identity(result_5)
      }>
    }
  )";

  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);
  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

// Verify that when a function node with a layering annotation is inlined,
// the inlined nodes inherit the parent function node's annotation.
TEST(FunctionTest, InlinedNodesInheritLayeringAnnotation) {
  // Parse and build a Model with a local function (multi-node body: Constant + Mul).
  ONNX_NAMESPACE::OnnxParser parser(basic_code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Find the function call node (local.myfun) and annotate it.
  Node* func_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_node = &node;
      break;
    }
  }
  ASSERT_NE(func_node, nullptr) << "Could not find function call node 'myfun'";
  ASSERT_TRUE(func_node->CanBeInlined());

  const std::string annotation = "TestLayerAnnotation";
  func_node->SetLayeringAnnotation(annotation);

  // Inline the function node.
  ASSERT_STATUS_OK(graph.InlineFunction(*func_node));
  ASSERT_STATUS_OK(graph.Resolve());

  // After inlining, the original function call node is removed and replaced
  // by the function body nodes (a Mul node; the Constant becomes an initializer).
  // Verify every remaining node inherited the annotation.
  int node_count = 0;
  for (const auto& node : graph.Nodes()) {
    ++node_count;
    EXPECT_EQ(node.GetLayeringAnnotation(), annotation)
        << "Node '" << node.Name() << "' (op: " << node.OpType()
        << ") did not inherit the parent function's layering annotation.";
  }
  EXPECT_GT(node_count, 0) << "Expected at least one inlined node in the graph.";
}

// Verify that when a function node with no layering annotation is inlined,
// the inlined nodes remain unannotated.
TEST(FunctionTest, InlinedNodesNoAnnotationWhenParentUnannotated) {
  ONNX_NAMESPACE::OnnxParser parser(basic_code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  Node* func_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_node = &node;
      break;
    }
  }
  ASSERT_NE(func_node, nullptr);
  // Do NOT set any annotation on the function node.
  ASSERT_TRUE(func_node->GetLayeringAnnotation().empty());

  ASSERT_STATUS_OK(graph.InlineFunction(*func_node));
  ASSERT_STATUS_OK(graph.Resolve());

  for (const auto& node : graph.Nodes()) {
    EXPECT_TRUE(node.GetLayeringAnnotation().empty())
        << "Node '" << node.Name() << "' should not have a layering annotation "
        << "when the parent function node was unannotated.";
  }
}

// Verify annotation inheritance with two calls to the same function,
// where each call has a different annotation.
TEST(FunctionTest, InlinedNodesInheritDistinctAnnotationsPerCallSite) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y1 = local.myfun (x)
            y = local.myfun (y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  ONNX_NAMESPACE::OnnxParser parser(code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput());

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Collect the two function call nodes in graph order.
  std::vector<Node*> func_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_nodes.push_back(&node);
    }
  }
  ASSERT_EQ(func_nodes.size(), 2u);

  // Annotate each call site differently.
  func_nodes[0]->SetLayeringAnnotation("AnnotationA");
  func_nodes[1]->SetLayeringAnnotation("AnnotationB");

  // Inline the first call, then the second.
  ASSERT_STATUS_OK(graph.InlineFunction(*func_nodes[0]));
  ASSERT_STATUS_OK(graph.InlineFunction(*func_nodes[1]));
  ASSERT_STATUS_OK(graph.Resolve());

  // After inlining both calls, the graph should have nodes from both expansions.
  // Each group should carry its respective annotation.
  bool found_a = false;
  bool found_b = false;
  for (const auto& node : graph.Nodes()) {
    const auto& ann = node.GetLayeringAnnotation();
    EXPECT_TRUE(ann == "AnnotationA" || ann == "AnnotationB")
        << "Node '" << node.Name() << "' has unexpected annotation: '" << ann << "'";
    if (ann == "AnnotationA") found_a = true;
    if (ann == "AnnotationB") found_b = true;
  }
  EXPECT_TRUE(found_a) << "No node found with AnnotationA";
  EXPECT_TRUE(found_b) << "No node found with AnnotationB";
}

static ONNX_NAMESPACE::FunctionProto MakeIdentityFunction(
    const std::string& domain, const std::string& name, const std::string& overload = {}) {
  ONNX_NAMESPACE::FunctionProto function;
  function.set_domain(domain);
  function.set_name(name);
  function.set_overload(overload);
  function.add_input("x");
  function.add_output("y");
  auto* opset = function.add_opset_import();
  opset->set_domain("");
  opset->set_version(17);
  auto* node = function.add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  return function;
}

static ONNX_NAMESPACE::NodeProto MakeFunctionCallNodeProto() {
  ONNX_NAMESPACE::NodeProto call_node;
  call_node.add_input("actual_x");
  call_node.add_output("actual_y");
  return call_node;
}

static NodeAttributes CreateDefaultAttributeMap(const ONNX_NAMESPACE::FunctionProto& function) {
  NodeAttributes attr_map;
  for (const auto& attribute_proto : function.attribute_proto()) {
    ORT_IGNORE_RETURN_VALUE(attr_map.emplace(attribute_proto.name(), attribute_proto));
  }

  return attr_map;
}

static Status SpecializeWithDefaultAttributes(ONNX_NAMESPACE::FunctionProto& function) {
  return function_utils::Specialize(function, MakeFunctionCallNodeProto(), CreateDefaultAttributeMap(function),
                                    "test_inliner");
}

static ONNX_NAMESPACE::AttributeProto MakeGraphRefAttribute(const std::string& name, const std::string& ref_attr_name,
                                                            ONNX_NAMESPACE::AttributeProto_AttributeType type) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_ref_attr_name(ref_attr_name);
  attr.set_type(type);
  return attr;
}

static ONNX_NAMESPACE::GraphProto MakeRecursiveDefaultGraph(const std::string& ref_attr_name) {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("default_graph");

  auto* node = graph.add_node();
  node->set_name("body_node");
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  *node->add_attribute() = MakeGraphRefAttribute("nested", ref_attr_name, ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);

  return graph;
}

static ONNX_NAMESPACE::GraphProto MakeNonRecursiveDefaultGraph() {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("default_graph");

  auto* node = graph.add_node();
  node->set_name("body_node");
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  node->add_attribute()->mutable_g()->set_name("leaf_graph");

  return graph;
}

static ONNX_NAMESPACE::FunctionProto MakeFunctionWithDefaultGraphAttributes(
    const std::vector<ONNX_NAMESPACE::AttributeProto>& default_attrs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& body_attrs) {
  ONNX_NAMESPACE::FunctionProto function;
  function.set_domain("local");
  function.set_name("myfun");
  function.add_input("x");
  function.add_output("y");

  for (const auto& default_attr : default_attrs) {
    function.add_attribute(default_attr.name());
    *function.add_attribute_proto() = default_attr;
  }

  auto* node = function.add_node();
  node->set_name("body_root");
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  for (const auto& body_attr : body_attrs) {
    *node->add_attribute() = body_attr;
  }

  return function;
}

static ONNX_NAMESPACE::FunctionProto MakeFunctionWithDefaultGraphAttribute(const ONNX_NAMESPACE::AttributeProto& default_attr) {
  return MakeFunctionWithDefaultGraphAttributes(
      {default_attr},
      {MakeGraphRefAttribute("body_attr", default_attr.name(), default_attr.type())});
}

static ONNX_NAMESPACE::ModelProto MakeModelWithDefaultGraphAttributeFunction(
    ONNX_NAMESPACE::FunctionProto function) {
  auto model_proto = CreateLocalFunctionChainModel(1);
  function.set_name("function_0");
  auto* onnx_opset = function.add_opset_import();
  onnx_opset->set_domain(kOnnxDomain);
  onnx_opset->set_version(16);
  auto* local_opset = function.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);
  *model_proto.mutable_functions(0) = std::move(function);
  return model_proto;
}

TEST(FunctionTest, ResolveRejectsRecursiveDefaultGraphAttributeExpansion) {
  for (const bool repeated_graphs : {false, true}) {
    SCOPED_TRACE(repeated_graphs);
    ONNX_NAMESPACE::AttributeProto default_attr;
    default_attr.set_name("body");
    default_attr.set_type(
        repeated_graphs
            ? ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS
            : ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    if (repeated_graphs) {
      *default_attr.add_graphs() = MakeRecursiveDefaultGraph(default_attr.name());
    } else {
      *default_attr.mutable_g() = MakeRecursiveDefaultGraph(default_attr.name());
    }

    auto& logger = DefaultLoggingManager().DefaultLogger();
    Model model(
        MakeModelWithDefaultGraphAttributeFunction(
            MakeFunctionWithDefaultGraphAttribute(default_attr)),
        nullptr, logger);
    const auto status = model.MainGraph().Resolve();
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(
        status.ErrorMessage(),
        testing::HasSubstr("Recursive model-local function graph attribute expansion"));
  }
}

TEST(FunctionTest, ResolveAllowsSequentialDefaultGraphAttributeReuse) {
  ONNX_NAMESPACE::AttributeProto default_attr;
  default_attr.set_name("body");
  default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *default_attr.mutable_g() = MakeNonRecursiveDefaultGraph();

  auto function = MakeFunctionWithDefaultGraphAttributes(
      {default_attr},
      {
          MakeGraphRefAttribute("body_attr_0", default_attr.name(), default_attr.type()),
          MakeGraphRefAttribute("body_attr_1", default_attr.name(), default_attr.type()),
      });
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(
      MakeModelWithDefaultGraphAttributeFunction(std::move(function)),
      nullptr, logger);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
}

static ONNX_NAMESPACE::AttributeProto MakeDefaultGraphReference(
    std::string name, std::string referenced_name, bool duplicate_reference) {
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_name(std::move(name));
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* graph = attribute.mutable_g();
  graph->set_name(attribute.name());
  auto* node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  *node->add_attribute() =
      MakeGraphRefAttribute("next_0", referenced_name, ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  if (duplicate_reference) {
    *node->add_attribute() =
        MakeGraphRefAttribute("next_1", referenced_name, ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  }
  auto* output = graph->add_output();
  output->set_name("y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  return attribute;
}

TEST(FunctionTest, ResolveRejectsExcessiveDefaultGraphReferenceDepth) {
  std::vector<ONNX_NAMESPACE::AttributeProto> defaults;
  defaults.reserve(kMaxModelLocalFunctionCallDepth + 1);
  for (size_t i = 0; i <= kMaxModelLocalFunctionCallDepth; ++i) {
    const std::string name = "body_" + std::to_string(i);
    if (i == kMaxModelLocalFunctionCallDepth) {
      ONNX_NAMESPACE::AttributeProto leaf;
      leaf.set_name(name);
      leaf.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      *leaf.mutable_g() = MakeNonRecursiveDefaultGraph();
      defaults.push_back(std::move(leaf));
    } else {
      defaults.push_back(MakeDefaultGraphReference(
          name, "body_" + std::to_string(i + 1), false));
    }
  }

  for (const bool visit_leaf_first : {false, true}) {
    SCOPED_TRACE(visit_leaf_first);
    std::vector<ONNX_NAMESPACE::AttributeProto> body_attributes;
    if (visit_leaf_first) {
      body_attributes.push_back(MakeGraphRefAttribute(
          "leaf_attr", defaults.back().name(),
          ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH));
    }
    body_attributes.push_back(MakeGraphRefAttribute(
        "body_attr", defaults.front().name(),
        ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH));

    auto function = MakeFunctionWithDefaultGraphAttributes(defaults, body_attributes);
    auto& logger = DefaultLoggingManager().DefaultLogger();
    Model model(
        MakeModelWithDefaultGraphAttributeFunction(std::move(function)),
        nullptr, logger);
    const auto status = model.MainGraph().Resolve();
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
    EXPECT_THAT(
        status.ErrorMessage(),
        testing::HasSubstr("graph attribute expansion depth exceeds"));
  }
}

TEST(FunctionTest, MaximumDefaultGraphReferenceDepthAllowsScalarLeafAttribute) {
  std::vector<ONNX_NAMESPACE::AttributeProto> defaults;
  defaults.reserve(kMaxModelLocalFunctionCallDepth);
  for (size_t i = 0; i < kMaxModelLocalFunctionCallDepth; ++i) {
    const std::string name = "body_" + std::to_string(i);
    if (i + 1 == kMaxModelLocalFunctionCallDepth) {
      ONNX_NAMESPACE::AttributeProto leaf;
      leaf.set_name(name);
      leaf.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      *leaf.mutable_g() = MakeNonRecursiveDefaultGraph();
      leaf.mutable_g()->mutable_node(0)->clear_attribute();
      auto* scalar = leaf.mutable_g()->mutable_node(0)->add_attribute();
      scalar->set_name("alpha");
      scalar->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      scalar->set_f(0.1f);
      defaults.push_back(std::move(leaf));
    } else {
      defaults.push_back(MakeDefaultGraphReference(
          name, "body_" + std::to_string(i + 1), false));
    }
  }

  auto function = MakeFunctionWithDefaultGraphAttributes(
      defaults,
      {MakeGraphRefAttribute(
          "body_attr", defaults.front().name(),
          ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)});
  Model model(
      MakeModelWithDefaultGraphAttributeFunction(std::move(function)),
      nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
}

TEST(FunctionTest, RepeatedDefaultGraphDagExpansionCompletes) {
  constexpr size_t graph_count = 30;
  std::vector<ONNX_NAMESPACE::AttributeProto> defaults;
  defaults.reserve(graph_count);
  for (size_t i = 0; i < graph_count; ++i) {
    const std::string name = "body_" + std::to_string(i);
    if (i + 1 == graph_count) {
      ONNX_NAMESPACE::AttributeProto leaf;
      leaf.set_name(name);
      leaf.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      *leaf.mutable_g() = MakeNonRecursiveDefaultGraph();
      defaults.push_back(std::move(leaf));
    } else {
      defaults.push_back(MakeDefaultGraphReference(
          name, "body_" + std::to_string(i + 1), true));
    }
  }

  auto function = MakeFunctionWithDefaultGraphAttributes(
      defaults,
      {MakeGraphRefAttribute(
          "body_attr", defaults.front().name(),
          ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)});
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(
      MakeModelWithDefaultGraphAttributeFunction(std::move(function)),
      nullptr, logger);
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
}

TEST(FunctionTest, ResolveRejectsMutuallyRecursiveDefaultGraphAttributes) {
  ONNX_NAMESPACE::AttributeProto first;
  first.set_name("first");
  first.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *first.mutable_g() = MakeRecursiveDefaultGraph("second");

  ONNX_NAMESPACE::AttributeProto second;
  second.set_name("second");
  second.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *second.mutable_g() = MakeRecursiveDefaultGraph("first");

  auto function = MakeFunctionWithDefaultGraphAttributes(
      {first, second},
      {MakeGraphRefAttribute("body_attr", first.name(), first.type())});
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(
      MakeModelWithDefaultGraphAttributeFunction(std::move(function)),
      nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(
      status.ErrorMessage(),
      testing::HasSubstr("Recursive model-local function graph attribute expansion"));
}

TEST(FunctionTest, TransitiveDefaultGraphBindingsContributeToFunctionDepth) {
  ONNX_NAMESPACE::AttributeProto first;
  first.set_name("first");
  first.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *first.mutable_g() = MakeRecursiveDefaultGraph("second");

  ONNX_NAMESPACE::AttributeProto second;
  second.set_name("second");
  second.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *second.mutable_g() = MakeRecursiveDefaultGraph("third");

  ONNX_NAMESPACE::AttributeProto third;
  third.set_name("third");
  third.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* third_graph = third.mutable_g();
  third_graph->set_name("third_graph");
  auto* call = third_graph->add_node();
  call->set_domain("local");
  call->set_op_type("function_0");
  call->add_input("x");
  call->add_output("y");
  auto* third_output = third_graph->add_output();
  third_output->set_name("y");
  third_output->mutable_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto wrapper = MakeFunctionWithDefaultGraphAttributes(
      {first, second, third},
      {MakeGraphRefAttribute("body_attr", first.name(), first.type())});
  wrapper.set_name("wrapper");
  auto* onnx_opset = wrapper.add_opset_import();
  onnx_opset->set_domain(kOnnxDomain);
  onnx_opset->set_version(16);
  auto* local_opset = wrapper.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);

  auto model_proto = CreateLocalFunctionChainModel(kMaxModelLocalFunctionCallDepth);
  model_proto.mutable_graph()->mutable_node(0)->set_op_type("wrapper");
  *model_proto.add_functions() = std::move(wrapper);

  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model(std::move(model_proto), nullptr, logger);
  const auto status = model.MainGraph().Resolve();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST(FunctionTest, SpecializeRejectsRecursiveDefaultGraphAttributeExpansion) {
  ONNX_NAMESPACE::AttributeProto default_attr;
  default_attr.set_name("body");
  default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *default_attr.mutable_g() = MakeRecursiveDefaultGraph(default_attr.name());

  auto function = MakeFunctionWithDefaultGraphAttribute(default_attr);
  const auto status = SpecializeWithDefaultAttributes(function);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Function attribute graph expansion is recursive"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("body -> body"));
}

TEST(FunctionTest, SpecializeRejectsRecursiveDefaultGraphsAttributeExpansion) {
  ONNX_NAMESPACE::AttributeProto default_attr;
  default_attr.set_name("body_list");
  default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS);
  *default_attr.add_graphs() = MakeRecursiveDefaultGraph(default_attr.name());

  auto function = MakeFunctionWithDefaultGraphAttribute(default_attr);
  const auto status = SpecializeWithDefaultAttributes(function);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Function attribute graph expansion is recursive"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("body_list -> body_list"));
}

TEST(FunctionTest, SpecializeRejectsRecursiveDefaultGraphAttributeExpansionAcrossDefaults) {
  ONNX_NAMESPACE::AttributeProto first_default_attr;
  first_default_attr.set_name("first");
  first_default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *first_default_attr.mutable_g() = MakeRecursiveDefaultGraph("second");

  ONNX_NAMESPACE::AttributeProto second_default_attr;
  second_default_attr.set_name("second");
  second_default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *second_default_attr.mutable_g() = MakeRecursiveDefaultGraph("first");

  auto function = MakeFunctionWithDefaultGraphAttributes(
      {first_default_attr, second_default_attr},
      {MakeGraphRefAttribute("body_attr", first_default_attr.name(), first_default_attr.type())});
  const auto status = SpecializeWithDefaultAttributes(function);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Function attribute graph expansion is recursive"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("first -> second -> first"));
}

TEST(FunctionTest, SpecializeAllowsNonRecursiveDefaultGraphAttributeExpansion) {
  ONNX_NAMESPACE::AttributeProto default_attr;
  default_attr.set_name("body");
  default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *default_attr.mutable_g() = MakeNonRecursiveDefaultGraph();

  auto function = MakeFunctionWithDefaultGraphAttribute(default_attr);
  auto status = SpecializeWithDefaultAttributes(function);

  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const auto& attr = function.node(0).attribute(0);
  EXPECT_EQ(attr.name(), "body_attr");
  EXPECT_TRUE(attr.ref_attr_name().empty());
  EXPECT_TRUE(attr.has_g());
}

TEST(FunctionTest, SpecializeAllowsSequentialReuseOfNonRecursiveDefaultGraphAttribute) {
  ONNX_NAMESPACE::AttributeProto default_attr;
  default_attr.set_name("body");
  default_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *default_attr.mutable_g() = MakeNonRecursiveDefaultGraph();

  auto function = MakeFunctionWithDefaultGraphAttributes(
      {default_attr},
      {
          MakeGraphRefAttribute("body_attr_0", default_attr.name(), default_attr.type()),
          MakeGraphRefAttribute("body_attr_1", default_attr.name(), default_attr.type()),
      });
  const auto status = SpecializeWithDefaultAttributes(function);

  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const auto& attrs = function.node(0).attribute();
  ASSERT_EQ(attrs.size(), 2);
  for (const auto& attr : attrs) {
    EXPECT_TRUE(attr.ref_attr_name().empty());
    EXPECT_TRUE(attr.has_g());
  }
}

TEST(FunctionTest, RejectDuplicateFunctionIdentifiersFromModelProto) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(10);
  auto* default_opset = model_proto.add_opset_import();
  default_opset->set_domain("");
  default_opset->set_version(17);
  auto* local_opset = model_proto.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);
  model_proto.mutable_graph()->set_name("duplicate_functions");
  *model_proto.add_functions() = MakeIdentityFunction("local", "myfun");
  *model_proto.add_functions() = MakeIdentityFunction("local", "myfun");

  std::string serialized_model;
  ASSERT_TRUE(model_proto.SerializeToString(&serialized_model));

  InferenceSession session{SessionOptions(), GetEnvironment()};
  const auto status = session.Load(serialized_model.data(), static_cast<int>(serialized_model.size()));
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Duplicate model-local function identifier"));
}

TEST(FunctionTest, RejectDuplicateFunctionIdentifiersFromFunctionVector) {
  std::vector<ONNX_NAMESPACE::FunctionProto> functions{
      MakeIdentityFunction("local", "myfun", "same_overload"),
      MakeIdentityFunction("local", "myfun", "same_overload")};
  const std::unordered_map<std::string, int> domain_to_version{{"", 17}, {"local", 1}};

  EXPECT_THROW(
      Model("duplicate_functions", false, ModelMetaData(), PathString(),
            IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, functions,
            DefaultLoggingManager().DefaultLogger()),
      OnnxRuntimeException);
}

// Test that overloaded functions (IR version 10+) are resolved correctly.
// Two functions with the same domain and name but different overload identifiers.
TEST(FunctionTest, OverloadedFunctions) {
  const char* code = R"(
        <
        ir_version: 10,
        opset_import: [ "" : 17, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y, float[N] z)
        {
            y = local.myfun:double_it (x)
            z = local.myfun:triple_it (x)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local",
        overload: "double_it"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local",
        overload: "triple_it"
        >
        myfun (lx) => (ly) {
            three = Constant <value = float[1] {3.0}> ()
            ly = Mul (lx, three)
        }
        )";

  // Serialize and then load model:
  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  NameMLValMap feeds;
  std::unique_ptr<CPUExecutionProvider> provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  std::vector<float> input_values = {1.0f, 2.0f, 3.0f};
  OrtValue ort_value;
  CreateMLValue<float>(provider->CreatePreferredAllocators()[0], {int64_t(input_values.size())}, input_values, &ort_value);
  feeds.insert(std::make_pair(std::string("x"), ort_value));

  std::vector<OrtValue> fetches;
  status = session_object.Run(run_options, feeds, AsSpan({std::string("y"), std::string("z")}), &fetches);
  ASSERT_TRUE(status.IsOK()) << "Session Run failed: " << status.ErrorMessage() << std::endl;

  // Check "y" output (doubled)
  auto& tensor_y = fetches[0].Get<Tensor>();
  auto* data_y = tensor_y.Data<float>();
  EXPECT_NEAR(data_y[0], 2.0f, 0.001f);
  EXPECT_NEAR(data_y[1], 4.0f, 0.001f);
  EXPECT_NEAR(data_y[2], 6.0f, 0.001f);

  // Check "z" output (tripled)
  auto& tensor_z = fetches[1].Get<Tensor>();
  auto* data_z = tensor_z.Data<float>();
  EXPECT_NEAR(data_z[0], 3.0f, 0.001f);
  EXPECT_NEAR(data_z[1], 6.0f, 0.001f);
  EXPECT_NEAR(data_z[2], 9.0f, 0.001f);
}

// Test that non-overloaded functions (empty overload) still work as before.
TEST(FunctionTest, OverloadedFunctionBackwardCompat) {
  // Same as basic_code but with ir_version: 10 to verify backward compatibility
  const char* code = R"(
        <
        ir_version: 10,
        opset_import: [ "" : 17, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {2.0, 4.0, 6.0});
}

namespace {
ONNX_NAMESPACE::ModelProto MakeModelWithFunctionConstant(int constant_output_count) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(10);
  auto* default_opset = model_proto.add_opset_import();
  default_opset->set_domain("");
  default_opset->set_version(17);
  auto* local_opset = model_proto.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("function_constant");

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("N");

  auto* graph_input = graph_proto->add_input();
  graph_input->set_name("x");
  *graph_input->mutable_type() = float_tensor;

  auto* graph_output = graph_proto->add_output();
  graph_output->set_name("y");
  *graph_output->mutable_type() = float_tensor;

  auto* call_node = graph_proto->add_node();
  call_node->set_name("call_myfun");
  call_node->set_op_type("myfun");
  call_node->set_domain("local");
  call_node->add_input("x");
  call_node->add_output("y");

  auto* function = model_proto.add_functions();
  function->set_domain("local");
  function->set_name("myfun");
  function->add_input("lx");
  function->add_output("ly");
  auto* function_opset = function->add_opset_import();
  function_opset->set_domain("");
  function_opset->set_version(17);

  auto* constant_node = function->add_node();
  constant_node->set_name("local_constant");
  constant_node->set_op_type("Constant");
  for (int i = 0; i < constant_output_count; ++i) {
    constant_node->add_output("c" + std::to_string(i));
  }
  auto* attr = constant_node->add_attribute();
  attr->set_name("value_float");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr->set_f(2.0f);

  auto* identity_node = function->add_node();
  identity_node->set_name("local_identity");
  identity_node->set_op_type("Identity");
  identity_node->add_input("lx");
  identity_node->add_output("ly");

  return model_proto;
}

// Loads and initializes a session, returning the first failing status.
Status LoadAndInitialize(const ONNX_NAMESPACE::ModelProto& model_proto) {
  std::string serialized_model;
  if (!model_proto.SerializeToString(&serialized_model)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize test model.");
  }

  InferenceSession session{SessionOptions(), GetEnvironment()};
  ORT_RETURN_IF_ERROR(session.Load(serialized_model.data(), narrow<int>(serialized_model.size())));
  return session.Initialize();
}

void CheckFunctionNodeCountRejection(const ONNX_NAMESPACE::ModelProto& model_proto, const char* expected_error) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  auto check_status = [&](const Status& status) {
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_error));
    EXPECT_EQ(model, nullptr);
  };

  check_status(Model::Load(model_proto, model, nullptr, logger));
  auto copy = model_proto;
  check_status(Model::Load(std::move(copy), model, nullptr, logger));

  std::string serialized_model;
  ASSERT_TRUE(model_proto.SerializeToString(&serialized_model));
  const int size = narrow<int>(serialized_model.size());
  check_status(Model::LoadFromBytes(size, serialized_model.data(), model, nullptr, logger));

  InferenceSession session{SessionOptions(), GetEnvironment()};
  check_status(session.Load(serialized_model.data(), size));

  PathString path = ORT_TSTR("function_node_counts_XXXXXX");
  FILE* file = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateTestFile(file, path));
  ScopedFileDeleter deleter(path);
  std::unique_ptr<FILE, int (*)(FILE*)> file_owner(file, fclose);
  ASSERT_EQ(serialized_model.size(), fwrite(serialized_model.data(), 1, serialized_model.size(), file));
  ASSERT_EQ(0, fclose(file_owner.release()));
  check_status(Model::Load(path, model, nullptr, logger));
}
}  // namespace

// These failures must return before constructing Model, without relying on exception handling.
TEST(FunctionTest, RejectFunctionConstantWithUnexpectedOutputCount) {
  for (int output_count : {0, 2}) {
    SCOPED_TRACE(output_count);
    ASSERT_NO_FATAL_FAILURE(CheckFunctionNodeCountRejection(
        MakeModelWithFunctionConstant(output_count), "Invalid output count for op Constant"));
  }

  ASSERT_STATUS_OK(LoadAndInitialize(MakeModelWithFunctionConstant(1)));
}

TEST(FunctionTest, RejectFunctionNodeWithUnexpectedInputCount) {
  auto model_proto = MakeModelWithFunctionConstant(1);
  model_proto.mutable_functions(0)->mutable_node(0)->add_input("lx");
  CheckFunctionNodeCountRejection(model_proto, "Invalid input count for op Constant");
}

TEST(FunctionTest, RejectFunctionWithInvalidOpsetImports) {
  auto model_proto = MakeModelWithFunctionConstant(1);
  auto* function = model_proto.mutable_functions(0);
  for (int64_t version : {int64_t{0}, int64_t{-1}, int64_t{std::numeric_limits<int>::max()} + 1}) {
    SCOPED_TRACE(version);
    function->mutable_opset_import(0)->set_version(version);
    ASSERT_NO_FATAL_FAILURE(CheckFunctionNodeCountRejection(model_proto, "Invalid opset version"));
  }

  function->clear_opset_import();
  CheckFunctionNodeCountRejection(model_proto, "No opset registered for domain");
}

TEST(FunctionTest, RejectFunctionSubgraphNodeWithUnexpectedOutputCount) {
  for (int output_count : {0, 1, 2}) {
    SCOPED_TRACE(output_count);
    auto model_proto = MakeModelWithFunctionConstant(output_count);
    auto* function = model_proto.mutable_functions(0);
    ONNX_NAMESPACE::NodeProto constant;
    constant.Swap(function->mutable_node(0));

    auto* condition = function->mutable_node(0);
    condition->set_op_type("Constant");
    condition->add_output("cond");
    auto* value = condition->add_attribute();
    value->set_name("value");
    value->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
    value->mutable_t()->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
    value->mutable_t()->add_int32_data(1);

    auto* if_node = function->mutable_node(1);
    if_node->set_op_type("If");
    if_node->set_input(0, "cond");
    for (const char* branch_name : {"then_branch", "else_branch"}) {
      auto* branch = if_node->add_attribute();
      branch->set_name(branch_name);
      branch->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      auto* graph = branch->mutable_g();
      graph->set_name(branch_name);
      *graph->add_node() = constant;
      auto* identity = graph->add_node();
      identity->set_op_type("Identity");
      identity->add_input("lx");
      identity->add_output("branch_output");
      auto* output = graph->add_output();
      output->set_name("branch_output");
      *output->mutable_type() = model_proto.graph().input(0).type();
    }

    if (output_count == 1) {
      ASSERT_STATUS_OK(LoadAndInitialize(model_proto));
    } else {
      ASSERT_NO_FATAL_FAILURE(CheckFunctionNodeCountRejection(model_proto, "Invalid output count for op Constant"));
    }
  }
}

TEST(FunctionTest, RejectFunctionDefaultGraphAttributeWithUnexpectedOutputCount) {
  for (bool repeated_graphs : {false, true}) {
    SCOPED_TRACE(repeated_graphs);
    auto model_proto = MakeModelWithFunctionConstant(1);
    auto* function = model_proto.mutable_functions(0);
    auto* attribute = function->add_attribute_proto();
    attribute->set_name("body");
    attribute->set_type(repeated_graphs ? ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS
                                        : ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto* graph = repeated_graphs ? attribute->add_graphs() : attribute->mutable_g();
    graph->set_name("default_body");
    auto* constant = graph->add_node();
    *constant = function->node(0);
    constant->add_output("c1");
    ASSERT_NO_FATAL_FAILURE(CheckFunctionNodeCountRejection(model_proto, "Invalid output count for op Constant"));
  }
}

TEST(FunctionTest, FunctionNodeOptionalAndVariadicOutputs) {
  for (const char* op_type : {"Dropout", "Split"}) {
    for (int output_count : {1, 2}) {
      SCOPED_TRACE(MakeString(op_type, " with ", output_count, " outputs"));
      auto model_proto = MakeModelWithFunctionConstant(output_count);
      auto* node = model_proto.mutable_functions(0)->mutable_node(0);
      node->set_op_type(op_type);
      node->clear_attribute();
      node->add_input("lx");
      ASSERT_STATUS_OK(LoadAndInitialize(model_proto));
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
