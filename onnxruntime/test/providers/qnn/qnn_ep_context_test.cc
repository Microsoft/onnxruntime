// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

constexpr size_t kQnnEpContextTestMaxDataSize = size_t{1} << 30;

struct QnnEpContextCallbackState {
  bool write_called = false;
  bool read_called = false;
  size_t write_count = 0;
  size_t read_count = 0;
  size_t allocation_count = 0;
  size_t max_data_size = kQnnEpContextTestMaxDataSize;
  bool fail_write = false;
  bool fail_read = false;
  bool return_empty = false;
  bool return_null_buffer = false;
  std::string write_name;
  std::string read_name;
  std::vector<char> payload;
};

static OrtStatus* ORT_API_CALL StoreQnnEpContextData(void* state, const char* name, const void* buffer,
                                                     size_t buffer_size) {
  auto& callback_state = *static_cast<QnnEpContextCallbackState*>(state);
  callback_state.write_called = true;
  ++callback_state.write_count;
  callback_state.write_name = name;
  if (callback_state.fail_write) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "synthetic QNN EPContext write callback failure");
  }
  callback_state.payload.clear();
  if (buffer_size != 0) {
    if (buffer == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "QNN EPContext write callback received a null non-empty payload");
    }
    callback_state.payload.assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + buffer_size);
  }
  return nullptr;
}

static OrtStatus* ORT_API_CALL LoadQnnEpContextData(void* state, const char* name, OrtAllocator* allocator,
                                                    void** buffer, size_t* buffer_size) {
  auto& callback_state = *static_cast<QnnEpContextCallbackState*>(state);
  callback_state.read_called = true;
  ++callback_state.read_count;
  callback_state.read_name = name;
  *buffer = nullptr;
  *buffer_size = 0;
  if (callback_state.fail_read) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "synthetic QNN EPContext read callback failure");
  }
  if (callback_state.return_empty) {
    return nullptr;
  }
  if (callback_state.payload.size() > callback_state.max_data_size) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "QNN application callback rejected payload exceeding its allocation limit");
  }
  *buffer_size = callback_state.payload.size();
  if (callback_state.return_null_buffer) {
    return nullptr;
  }
  if (callback_state.payload.empty()) {
    return nullptr;
  }

  ++callback_state.allocation_count;
  OrtStatus* status = Ort::GetApi().AllocatorAlloc(allocator, callback_state.payload.size(), buffer);
  if (status == nullptr) {
    std::copy(callback_state.payload.begin(), callback_state.payload.end(), static_cast<char*>(*buffer));
  }
  return status;
}

TEST(QnnEpContextCallbackTest, ApplicationReadLimitIsCheckedBeforeAllocation) {
  QnnEpContextCallbackState callback_state;
  const std::vector<char> payload{'q', 'n', 'n'};
  Ort::Status write_status{StoreQnnEpContextData(&callback_state, "context.bin", payload.data(), payload.size())};
  ASSERT_TRUE(write_status.IsOK()) << write_status.GetErrorMessage();
  callback_state.max_data_size = callback_state.payload.size() - 1;
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = nullptr;
  size_t buffer_size = 0;
  Ort::Status status{LoadQnnEpContextData(&callback_state, "context.bin", allocator, &buffer, &buffer_size)};
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("application callback"));
  EXPECT_EQ(callback_state.allocation_count, 0u);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(buffer_size, 0u);

  callback_state.max_data_size = callback_state.payload.size();
  status = Ort::Status{LoadQnnEpContextData(&callback_state, "context.bin", allocator, &buffer, &buffer_size)};
  std::unique_ptr<void, Ort::detail::AllocatedFree> buffer_guard{buffer, Ort::detail::AllocatedFree{allocator}};
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(callback_state.allocation_count, 1u);
  ASSERT_EQ(buffer_size, callback_state.payload.size());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(std::vector<char>(static_cast<char*>(buffer), static_cast<char*>(buffer) + buffer_size),
            callback_state.payload);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static int64_t GetNodeAttr(const Node& node, const std::string& attr_name, int64_t default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.i();
  }

  return default_val;
}

static const std::string& GetNodeAttr(const Node& node, const std::string& attr_name, const std::string& default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.s();
  }

  return default_val;
}

// from the context cache Onnx model, find the EPContext node with main_context=1,
// and get the QNN context binary file name
static void GetContextBinaryFileName(const std::string onnx_ctx_file,
                                     std::string& last_ctx_bin_file,
                                     const Logger& logger) {
  std::shared_ptr<Model> ctx_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(onnx_ctx_file), ctx_model, nullptr, logger));
  auto& ctx_graph = ctx_model->MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      int64_t is_main_context = GetNodeAttr(node, "main_context", static_cast<int64_t>(0));
      if (1 == is_main_context) {
        last_ctx_bin_file = GetNodeAttr(node, "ep_cache_context", "");
        return;
      }
    }
  }
}

// Get context binary file name from Context model file and remove it with the context model file
void CleanUpCtxFile(std::string context_file_path) {
  std::string qnn_ctx_binary_file_name;
  GetContextBinaryFileName(context_file_path, qnn_ctx_binary_file_name,
                           DefaultLoggingManager().DefaultLogger());

  std::filesystem::path ctx_model_path(context_file_path);

  std::string qnn_ctx_binary_file_path = (ctx_model_path.remove_filename().string() + qnn_ctx_binary_file_name);
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_path.c_str()), 0);
  ASSERT_EQ(std::remove(context_file_path.c_str()), 0);
}

// Create a model with FusedGemm + Add (quantized)
// input1 -> Add -> Q -> DQ ----
//                              |
//        input2 -> Q -> DQ -> FusedGemm -> Q -> DQ -> output
static GetTestModelFn BuildGraphWithQAndNonQ(bool single_ep_node = true) {
  return [single_ep_node](ModelTestBuilder& builder) {
    // Create non-quantized FusedGemm node1
    std::vector<float> gemm_input_data(12, 1.0f);
    std::vector<float> gemm_weight_data(16, 1.0f);  // 4x4 = 16 elements
    NodeArg* input1 = MakeTestInput(builder, TestInputDef<float>({3, 4}, false, gemm_input_data));
    NodeArg* add1_ini_input2 = MakeTestInput(builder, TestInputDef<float>({4, 4}, true, gemm_weight_data));

    auto* add1_output = builder.MakeIntermediate();
    Node& fused_gemm_node1 = builder.AddNode("FusedGemm", {input1, add1_ini_input2}, {add1_output}, kMSDomain);
    fused_gemm_node1.AddAttribute("activation", "Relu");

    // Create quantized Add node2
    std::vector<float> add_data(12, 1.0f);
    gsl::span<float> data_range = gsl::make_span(add_data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    auto* add2_input1_qdq = AddQDQNodePair<uint8_t>(builder, add1_output, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add2_input2 = MakeTestInput(builder, TestInputDef<float>({3, 4}, true, add_data));
    auto* add2_input2_qdq = AddQDQNodePair<uint8_t>(builder, add2_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add2_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add2_input1_qdq, add2_input2_qdq}, {add2_output});

    if (single_ep_node) {
      // add_output -> Q -> DQ -> output
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add2_output, q_parameter.scale, q_parameter.zero_point);
    } else {
      auto* add3_input1_qdq = AddQDQNodePair<uint8_t>(builder, add2_output, q_parameter.scale, q_parameter.zero_point);
      NodeArg* add3_ini_input2 = MakeTestInput(builder, TestInputDef<float>({4, 4}, true, gemm_weight_data));

      auto* add3_output = builder.MakeIntermediate();
      Node& fused_gemm_node2 = builder.AddNode("FusedGemm", {add3_input1_qdq, add3_ini_input2}, {add3_output}, kMSDomain);
      fused_gemm_node2.AddAttribute("activation", "Relu");

      // Create quantized Add node4
      auto* add4_input1_qdq = AddQDQNodePair<uint8_t>(builder, add3_output, q_parameter.scale, q_parameter.zero_point);

      NodeArg* add4_input2 = MakeTestInput(builder, TestInputDef<float>({3, 4}, true, add_data));
      auto* add4_input2_qdq = AddQDQNodePair<uint8_t>(builder, add4_input2, q_parameter.scale, q_parameter.zero_point);

      auto* add4_output = builder.MakeIntermediate();

      builder.AddNode("Add", {add4_input1_qdq, add4_input2_qdq}, {add4_output});
      // add_output -> Q -> DQ -> output
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add4_output, q_parameter.scale, q_parameter.zero_point);
    }
  };
}

void QnnContextBinaryMultiPartitionTestBody(bool single_ep_node = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./testdata/qnn_context_binary_multi_partition_test.onnx";
  std::remove(context_model_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  int ep_context_node_count = 0;
  int non_ep_context_node_count = 0;
  std::shared_ptr<Model> ctx_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_model_file), ctx_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto& ctx_graph = ctx_model->MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      ++ep_context_node_count;
      // validate the fix for the partition issue relate to QDQ model
      ASSERT_EQ(node.InputDefs().size(), 1);
    } else {
      ++non_ep_context_node_count;
    }
  }

  int expected_node_count = single_ep_node ? 1 : 2;
  ASSERT_EQ(ep_context_node_count, expected_node_count);
  ASSERT_EQ(non_ep_context_node_count, expected_node_count);

  Ort::SessionOptions so2;
  // context file path is required if it's non-embed mode and the model is loaded from memory
  so2.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
  so2.AppendExecutionProvider("QNN", provider_options);

  std::string ctx_model_data;
  ctx_model->ToProto().SerializeToString(&ctx_model_data);
  Ort::Session session2(*ort_env, ctx_model_data.data(), ctx_model_data.size(), so2);

  // clean up
  CleanUpCtxFile(context_model_file);
}

// Helper struct that represents a test model.
struct TestModel {
  std::unique_ptr<onnxruntime::Model> model;
  std::unique_ptr<ModelTestBuilder> builder;

  std::string Serialize() const {
    std::string model_data;
    model->ToProto().SerializeToString(&model_data);
    return model_data;
  }

  Status Save(const ORTCHAR_T* path) const {
    return onnxruntime::Model::Save(*model, PathString(path));
  }
};

// Create a test model from a function that programmatically builds a graph.
static void CreateTestModel(test::GetTestModelFn graph_builder,
                            int onnx_opset_version,
                            logging::Severity log_severity,
                            TestModel& test_model) {
  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(log_severity);
  const std::unordered_map<std::string, int> domain_to_version = {{"", onnx_opset_version}, {kMSDomain, 1}};

  test_model.model = std::make_unique<onnxruntime::Model>("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                                                          IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                                          std::vector<ONNX_NAMESPACE::FunctionProto>{}, logging_manager.DefaultLogger());
  test_model.builder = std::make_unique<ModelTestBuilder>(test_model.model->MainGraph());
  graph_builder(*test_model.builder);
  test_model.builder->SetGraphOutputs();
  ASSERT_STATUS_OK(test_model.model->MainGraph().Resolve());
}

// Helper that checks that a compiled model has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(const onnxruntime::Model& ep_ctx_model,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  int ep_context_node_count = 0;
  int non_ep_context_node_count = 0;
  auto& ctx_graph = ep_ctx_model.MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      ++ep_context_node_count;
      // validate the fix for the partition issue relate to QDQ model
      ASSERT_EQ(node.InputDefs().size(), 1);
    } else {
      ++non_ep_context_node_count;
    }
  }

  EXPECT_EQ(ep_context_node_count, expected_ep_context_node_count);
  EXPECT_EQ(non_ep_context_node_count, expected_other_node_count);
}

// Helper to check that a compiled model (stored as a file) has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(const ORTCHAR_T* model_path,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  std::shared_ptr<Model> ep_ctx_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(model_path), ep_ctx_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  CheckEpContextNodeCounts(*ep_ctx_model, expected_ep_context_node_count, expected_other_node_count);
}

// Helper to check that a compiled model (stored in a buffer) has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(void* model_buffer, size_t model_buffer_size,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  std::shared_ptr<Model> ep_ctx_model;
  const ORTCHAR_T* output_model_path = ORT_TSTR("tmp_output_ctx_model.onnx");
  ASSERT_STATUS_OK(onnxruntime::Model::LoadFromBytes(static_cast<int>(model_buffer_size),
                                                     model_buffer, output_model_path, ep_ctx_model,
                                                     nullptr, DefaultLoggingManager().DefaultLogger()));
  CheckEpContextNodeCounts(*ep_ctx_model, expected_ep_context_node_count, expected_other_node_count);
  std::filesystem::remove(output_model_path);
}

// Test workflow that:
//   - Creates session that disables EP compilation.
//   - Session creation fails because input model is not pre-compiled.
//   - Uses OrtCompileApi to compile the model.
//   - Recreates session with the compiled model.
TEST_F(QnnHTPBackendTests, CompileApi_DisableEpCompile_ThenCompileExplicitly) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_disable_compile_input.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./compileapi_disable_compile_output.onnx");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  so.AppendExecutionProvider("QNN", provider_options);
  so.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "1");  // Disable model compilation!

  // Create an inference session that fails with error ORT_MODEL_REQUIRES_COMPILATION
  try {
    Ort::Session session(*ort_env, input_model_file, so);
    FAIL() << "Expected Session creation to fail but it succeeded";  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    OrtErrorCode error_code = excpt.GetOrtErrorCode();
    std::string_view error_msg = excpt.what();
    ASSERT_EQ(error_code, ORT_MODEL_REQUIRES_COMPILATION);
    ASSERT_THAT(error_msg, testing::HasSubstr(kQnnExecutionProvider));
  }

  // Session creation failed because the model was not pre-compiled.
  // Try to compile it now.

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
}

// Test using the CompileModel() API with settings:
//   - input model file
//   - output model file
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputModelFromPath) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_inputmodelfrompath.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./qnn_context_binary_multi_partition_test.onnx");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
}

// Test using the CompileModel() API with settings:
//   - input model from buffer
//   - output model file
//   - EPContext nodes in output model use embedded binary blobs.
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputModelAsBuffer_Embedded) {
  // Create a test model and serialize it to a buffer.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  std::string model_data = test_model.Serialize();

  const ORTCHAR_T* output_model_file = ORT_TSTR("./qnn_context_binary_multi_partition_test.onnx");
  std::filesystem::remove(output_model_file);

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
}

// Test using the CompileModel() API with settings:
//   - input model from file
//   - save output model to a buffer
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_OutputModelBuffer) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_inputmodelfrompath.onnx");
  std::filesystem::remove(input_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Create model compilation options from the session options. Output model is stored in a buffer.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
  compile_options.SetInputModelPath(input_model_file);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_model_buffer = nullptr;
  size_t output_model_buffer_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was saved to the buffer.
  ASSERT_TRUE(output_model_buffer != nullptr);
  ASSERT_TRUE(output_model_buffer_size > 0);

  // Check that the compiled model has the expected number of EPContext nodes.
  CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

  {
    // Should be able to create a session with the compiled model and the original session options.
    EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, so)));
  }

  allocator.Free(output_model_buffer);
}

// Test using the CompileModel() API with settings:
//   - input model from buffer
//   - save output model to buffer
//   - test enabling AND disabling embed mode for context binary in EPContext node attributes
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputAndOutputModelsInBuffers) {
  // Create a test model and serialize it to a buffer.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  std::string model_data = test_model.Serialize();

  // Initialize session options with QNN EP
  Ort::SessionOptions session_options;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  session_options.AppendExecutionProvider("QNN", provider_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Test embed mode enabled.
  {
    QnnEpContextCallbackState callback_state;
    session_options.SetEpContextDataReadFunc(LoadQnnEpContextData, &callback_state);
    void* output_model_buffer = nullptr;
    size_t output_model_buffer_size = 0;

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
    compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
    compile_options.SetEpContextEmbedMode(true);
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Compile the model.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    // Make sure the compiled model was saved to the buffer.
    ASSERT_TRUE(output_model_buffer != nullptr);
    ASSERT_TRUE(output_model_buffer_size > 0);

    // Check that the compiled model has the expected number of EPContext nodes.
    CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

    // Should be able to create a session with the compiled model and the original session options.
    EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, session_options)));
    EXPECT_EQ(callback_state.read_count, 0u);

    allocator.Free(output_model_buffer);
    session_options.ClearEpContextDataReadFunc();
  }

  // Test embed mode disabled.
  {
    void* output_model_buffer = nullptr;
    size_t output_model_buffer_size = 0;

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
    compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
    std::string target_dir = "./testdata/";
    std::string model_name = "test_model_in_mem.onnx";
    auto pos = model_name.rfind(".onnx");
    std::string bin_file_name = model_name.substr(0, pos) + "_qnn.bin";
    std::filesystem::remove(target_dir + bin_file_name);
    compile_options.SetEpContextBinaryInformation(ToWideString(target_dir).c_str(), ToWideString(model_name).c_str());
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    QnnEpContextCallbackState callback_state;
    compile_options.SetEpContextDataWriteFunc(StoreQnnEpContextData, &callback_state);

    // Compile the model.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    // Make sure the compiled model was saved to the buffer.
    ASSERT_TRUE(output_model_buffer != nullptr);
    ASSERT_TRUE(output_model_buffer_size > 0);

    ASSERT_TRUE(callback_state.write_called);
    ASSERT_EQ(callback_state.write_count, 1u);
    ASSERT_EQ(callback_state.write_name, bin_file_name);
    ASSERT_FALSE(callback_state.payload.empty());
    ASSERT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

    // Check that the compiled model has the expected number of EPContext nodes.
    CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

    // Add session option "ep.context_file_path" so that the session can use it to locate the [model_name]_qnn.bin file
    std::string ctx_model = target_dir + model_name;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_model.c_str());
    callback_state.max_data_size = callback_state.payload.size() - 1;
    session_options.SetEpContextDataReadFunc(LoadQnnEpContextData, &callback_state);

    std::string oversized_error;
    try {
      Ort::Session session(*ort_env, output_model_buffer, output_model_buffer_size, session_options);
      FAIL() << "Expected application callback allocation-limit rejection";
    } catch (const Ort::Exception& ex) {
      oversized_error = ex.what();
      EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_GRAPH);
    }
    EXPECT_TRUE(callback_state.read_called);
    EXPECT_EQ(callback_state.read_count, 1u);
    EXPECT_EQ(callback_state.allocation_count, 0u);
    EXPECT_THAT(oversized_error,
                testing::HasSubstr("QNN application callback rejected payload exceeding its allocation limit"));
    EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

    callback_state.read_called = false;
    callback_state.max_data_size = callback_state.payload.size();

    callback_state.return_empty = true;
    std::string empty_error;
    try {
      Ort::Session session(*ort_env, output_model_buffer, output_model_buffer_size, session_options);
      FAIL() << "Expected empty QNN EPContext callback payload rejection";
    } catch (const Ort::Exception& ex) {
      empty_error = ex.what();
      EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_GRAPH);
    }
    EXPECT_TRUE(callback_state.read_called);
    EXPECT_EQ(callback_state.read_count, 2u);
    EXPECT_THAT(empty_error, testing::HasSubstr("QNN EPContext read callback returned an empty payload"));
    EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

    callback_state.return_empty = false;
    callback_state.read_called = false;
    callback_state.return_null_buffer = true;
    std::string null_buffer_error;
    try {
      Ort::Session session(*ort_env, output_model_buffer, output_model_buffer_size, session_options);
      FAIL() << "Expected null QNN EPContext callback buffer rejection";
    } catch (const Ort::Exception& ex) {
      null_buffer_error = ex.what();
      EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_GRAPH);
    }
    EXPECT_TRUE(callback_state.read_called);
    EXPECT_EQ(callback_state.read_count, 3u);
    EXPECT_THAT(null_buffer_error,
                testing::HasSubstr("EPContext read callback returned a null buffer for non-empty QNN context data"));
    EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

    callback_state.return_null_buffer = false;
    callback_state.read_called = false;
    callback_state.fail_read = true;
    std::string read_error;
    try {
      Ort::Session session(*ort_env, output_model_buffer, output_model_buffer_size, session_options);
      FAIL() << "Expected QNN EPContext read callback failure";
    } catch (const Ort::Exception& ex) {
      read_error = ex.what();
      EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_GRAPH);
    }
    EXPECT_TRUE(callback_state.read_called);
    EXPECT_EQ(callback_state.read_count, 4u);
    EXPECT_THAT(read_error, testing::HasSubstr("synthetic QNN EPContext read callback failure"));
    EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

    callback_state.fail_read = false;
    callback_state.read_called = false;
    // Should be able to create a session with the compiled model and the original session options.
    EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, session_options)));

    EXPECT_TRUE(callback_state.read_called);
    EXPECT_EQ(callback_state.read_count, 5u);
    EXPECT_EQ(callback_state.allocation_count, 1u);
    EXPECT_EQ(callback_state.read_name, callback_state.write_name);
    EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));
    allocator.Free(output_model_buffer);
  }
}

TEST_F(QnnHTPBackendTests, CompileApi_EpContextWriteCallbackErrorDoesNotFallbackToDisk) {
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  const std::string model_data = test_model.Serialize();

  Ort::SessionOptions session_options;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  session_options.AppendExecutionProvider("QNN", provider_options);

  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
  compile_options.SetEpContextEmbedMode(false);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_model_buffer = nullptr;
  size_t output_model_buffer_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);

  const std::string target_dir = "./testdata/";
  const std::string model_name = "test_model_callback_error.onnx";
  const std::string bin_file_name = "test_model_callback_error_qnn.bin";
  std::filesystem::remove(target_dir + bin_file_name);
  compile_options.SetEpContextBinaryInformation(ToWideString(target_dir).c_str(),
                                                ToWideString(model_name).c_str());

  QnnEpContextCallbackState callback_state;
  callback_state.fail_write = true;
  compile_options.SetEpContextDataWriteFunc(StoreQnnEpContextData, &callback_state);

  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("synthetic QNN EPContext write callback failure"));
  EXPECT_TRUE(callback_state.write_called);
  EXPECT_FALSE(std::filesystem::exists(target_dir + bin_file_name));

  if (output_model_buffer != nullptr) {
    allocator.Free(output_model_buffer);
  }
}

TEST_F(QnnHTPBackendTests, EpContextCallbacksRejectSharedContextConfigurations) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  QnnEpContextCallbackState callback_state;

  {
    Ort::SessionOptions session_options;
    session_options.SetEpContextDataReadFunc(LoadQnnEpContextData, &callback_state);
    session_options.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    session_options.AppendExecutionProvider("QNN", provider_options);

    std::string error;
    try {
      Ort::Session session(*ort_env, input_model_file, session_options);
      FAIL() << "Expected QNN shared-context callback configuration rejection";
    } catch (const Ort::Exception& ex) {
      error = ex.what();
    }
    EXPECT_THAT(error, testing::HasSubstr("QNN shared EP contexts do not support EPContext data callbacks"));
  }

  {
    Ort::SessionOptions session_options;
    session_options.SetEpContextDataReadFunc(LoadQnnEpContextData, &callback_state);
    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["enable_vtcm_backup_buffer_sharing"] = "1";
    session_options.AppendExecutionProvider("QNN", provider_options);

    std::string error;
    try {
      Ort::Session session(*ort_env, input_model_file, session_options);
      FAIL() << "Expected QNN VTCM callback configuration rejection";
    } catch (const Ort::Exception& ex) {
      error = ex.what();
    }
    EXPECT_THAT(error,
                testing::HasSubstr("QNN VTCM backup buffer sharing does not support EPContext data callbacks"));
  }
}

// Test using the CompileModel() API with settings:
//   - input model from file
//   - save output model to a buffer
//   - save initializers (used by CPU EP) to external file.
//   - EPContext nodes in output model use embedded binary blobs.
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_OutputModelBuffer_OutputInitializersFile) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_outputmodelbuffer_initializers.onnx");
  const ORTCHAR_T* output_initializers_file = ORT_TSTR("./compileapi_initializers.bin");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_initializers_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Create model compilation options from the session options. Output model is stored in a buffer.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_model_buffer = nullptr;
  size_t output_model_buffer_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
  compile_options.SetOutputModelExternalInitializersFile(output_initializers_file, 0);
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was saved to the buffer.
  ASSERT_TRUE(output_model_buffer != nullptr);
  ASSERT_TRUE(output_model_buffer_size > 0);

  // Make sure that the initializers were saved to an external file.
  ASSERT_TRUE(std::filesystem::exists(output_initializers_file));

  // Check that the compiled model has the expected number of EPContext nodes.
  CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, so)));

  allocator.Free(output_model_buffer);
}

// Test that the explicit compile API can be configured to return an error if the output model does not
// have EPContext nodes.
TEST_F(QnnHTPBackendTests, CompileApi_SetFlags_ErrorIfNoCompiledNodes) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("should_not_be_generated.onnx");
  std::filesystem::remove(output_model_file);

  // Initialize session options with only CPU EP, which will not be able to compile any nodes.
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);

  // Call CompileModel() but expect an error status.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_EQ(status.GetErrorCode(), ORT_FAIL);
  ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("Unable to compile any nodes"));

  // Make sure that the output file was *NOT* generated.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
}

// Test that the explicit compile API can be configured to return an error if the output model already exists and
// would have been overwritten.
TEST_F(QnnHTPBackendTests, CompileApi_SetFlags_ErrorIfOutputFileAlreadyExists) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("mul_1_ctx_.onnx");
  std::filesystem::remove(output_model_file);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider(kQnnExecutionProvider, ProviderOptions{{"backend_type", "htp"}});

  // Compile with QNN EP. Should succeed the first time.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << "CompileModel() should succeed the first time a model is compiled.";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "compiled model should exist";
  }

  // Compiling the input model again should fail if we disallow overwriting the output file.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_EQ(status.GetErrorCode(), ORT_FAIL);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("exists already"));
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "original compiled model should still exist";
  }
}

// Tests that the explicit compile API returns an error if user tries to compile a compiled model.
// This scenario is silently ignored in the original compilation approach with session option configs.
TEST_F(QnnHTPBackendTests, CompileApi_ErrorIfCompilingACompiledModel) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("mul_1_ctx_.onnx");
  std::filesystem::remove(output_model_file);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider(kQnnExecutionProvider, ProviderOptions{{"backend_type", "htp"}});

  // Compile with QNN EP. Should succeed the first time.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << "CompileModel() should succeed the first time a model is compiled.";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "compiled model should exist";
  }

  // Compiling the compiled model should always fail: it's already compiled!
  {
    const ORTCHAR_T* new_output_model_file = ORT_TSTR("should_not_be_generated.onnx");  // Should not be generated.
    std::filesystem::remove(new_output_model_file);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(output_model_file);  // Set the compiled model as the input!
    compile_options.SetOutputModelPath(new_output_model_file);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_GRAPH);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("ensure the input model is not already compiled"));
    ASSERT_FALSE(std::filesystem::exists(new_output_model_file)) << "new compiled model should not be generated";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "original compiled model should still exist";
  }
}

// Uses the original compiling approach with session option configs (instead of explicit compile API).
// Test that ORT does not generate an output model if the model does not contain EPContext nodes.
// Also, ORT should not return an error.
TEST_F(QnnHTPBackendTests, QnnContextBinary_OriginalCompileApproach_NoCompiledNodesDoesntGenerateOutput) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const char* output_model_file = "should_not_be_generated.onnx";

  // Initialize session options with only CPU EP, which will not be able to compile any nodes.
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file);
  Ort::Session session(*ort_env, input_model_file, so);  // Should not throw an error.

  // Make sure that the output file was *NOT* generated.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
}

// Uses the original compiling approach with session option configs (instead of explicit compile API).
// Test that ORT does not generate an output model if the input model is already compiled.
// Also, ORT should not return an error.
TEST_F(QnnHTPBackendTests, QnnContextBinary_OriginalCompileApproach_IgnoreCompilingOfCompiledModel) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const char* output_model_file = "mul_1_ctx.onnx";
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}};

  // Compile a model with QNN. This should succeed.
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file);
    so.AppendExecutionProvider(kQnnExecutionProvider, qnn_options);

    Ort::Session session(*ort_env, input_model_file, so);
    ASSERT_TRUE(std::filesystem::exists(output_model_file));  // check compiled model was generated.
  }

  // Try compiling the compiled model again. ORT should basically ignore it.
  {
    const char* new_output_model_file = "should_not_be_generated.onnx";  // will not be generated!
    std::filesystem::remove(new_output_model_file);

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, new_output_model_file);
    so.AppendExecutionProvider(kQnnExecutionProvider, qnn_options);

    Ort::Session session(*ort_env, ToPathString(output_model_file).c_str(), so);

    // Session creation should not throw an error. And a new output model should not have been generated.
    ASSERT_FALSE(std::filesystem::exists(new_output_model_file));
  }
}

// Test that models with 1 non-quantized FusedGemm node and 1 quantized Add node can still generate the context binary
// The generated Onnx model has 1 FusedGemm node and 1 EPContext node
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport1) {
  bool single_ep_node = true;
  QnnContextBinaryMultiPartitionTestBody(single_ep_node);
}

// Test that models with 2 non-quantized FusedGemm nodes and 2 quantized Add nodes can still generate the context binary
// The generated Onnx model has 2 FusedGemm nodes and 2 EPContext nodes
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport2) {
  bool single_ep_node = false;
  QnnContextBinaryMultiPartitionTestBody(single_ep_node);
}

void EpCtxCpuNodeWithExternalIniFileTestBody(bool expect_external_ini_file, bool load_model_from_buffer = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildGraphWithQAndNonQ(true)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ModelSavingOptions model_saving_options{10};
  // dump the model in testdata folder in case it hides the bug that not able to find model not in current dir
  const std::string model_with_ext = "./testdata/model_external.onnx";
  const std::string model_ext_file = "model_external.bin";
  ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(model, model_with_ext,
                                                       model_ext_file, model_saving_options));

  EXPECT_TRUE(std::filesystem::exists(model_with_ext.c_str()));
  std::string model_ext_file_full_path = "./testdata/" + model_ext_file;
  EXPECT_TRUE(std::filesystem::exists(model_ext_file_full_path.c_str()));

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AppendExecutionProvider("QNN", provider_options);
  const std::string ep_context_model_file = "./qnn_ctx_part_external_ini_ctx.onnx";
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_model_file.c_str());
  const std::string external_ini_file = "./qnn_ctx_part_external_ini.bin";
  if (expect_external_ini_file) {
    // Set the external ini file name will force all initializers to the external file
    so.AddConfigEntry(kOrtSessionOptionsEpContextModelExternalInitializersFileName, external_ini_file.c_str());
  }  // otherwise all initializers are in Onnx file, no external data file generated

  if (load_model_from_buffer) {
    std::vector<char> buffer;
    {
      std::ifstream file(model_with_ext, std::ios::binary | std::ios::ate);
      if (!file)
        ORT_THROW("Error reading model");
      buffer.resize(narrow<size_t>(file.tellg()));
      file.seekg(0, std::ios::beg);
      if (!file.read(buffer.data(), buffer.size()))
        ORT_THROW("Error reading model");
    }
    so.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, "./testdata/");
    Ort::Session session(*ort_env, buffer.data(), buffer.size(), so);
  } else {
    Ort::Session session(*ort_env, ToPathString(model_with_ext).c_str(), so);
  }

  EXPECT_TRUE(std::filesystem::exists(ep_context_model_file.c_str()));
  if (expect_external_ini_file) {
    EXPECT_TRUE(std::filesystem::exists(external_ini_file.c_str()));
    ASSERT_EQ(std::remove(external_ini_file.c_str()), 0);
  } else {
    EXPECT_FALSE(std::filesystem::exists(external_ini_file.c_str()));
  }

  // clean up
  ASSERT_EQ(std::remove(model_with_ext.c_str()), 0);
  ASSERT_EQ(std::remove(model_ext_file_full_path.c_str()), 0);
  CleanUpCtxFile(ep_context_model_file);
}

// Set the session option "ep.context_model_external_initializers_file_name" so FusedGemm (which fallback on CPU)
// will dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(true);
}

// Without setting the session option "ep.context_model_external_initializers_file_name"
// so FusedGemm (which fallback on CPU) will NOT dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithoutExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(false);
}

// Load model from memory
// Without setting the session option "ep.context_model_external_initializers_file_name"
// so FusedGemm (which fallback on CPU) will NOT dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithoutExternalWeightsModelFromMemory) {
  EpCtxCpuNodeWithExternalIniFileTestBody(false, true);
}

// Set ep.context_file_path to folder path which is not a valid option, check the error message
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationFolderPathNotExpected) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_folder_not_expected/";
  std::remove(ep_context_onnx_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Set ep.context_file_path to invalid file path, check the error message
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationFolderPathNotExpected2) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_folder_not_expected/invalid_file";
  std::remove(ep_context_onnx_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Create session 1 to generate context binary file
// Create session 2 to do same thing, make sure session 2 failed because file exist already
// Make sure no new file over write from session 2
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationNoOverWrite) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_no_over_write.onnx";
  const std::string ep_context_binary_file = "./ep_context_no_over_write_qnn.bin";

  std::remove(ep_context_onnx_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session1(*ort_env, model_data_span.data(), model_data_span.size(), so);

  auto modify_time_1 = std::filesystem::last_write_time(ep_context_binary_file);

  try {
    Ort::Session session2(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("exists already."));
    auto modify_time_2 = std::filesystem::last_write_time(ep_context_binary_file);
    ASSERT_EQ(modify_time_1, modify_time_2);
  }

  ASSERT_EQ(std::remove(ep_context_onnx_file.c_str()), 0);
  ASSERT_EQ(std::remove(ep_context_binary_file.c_str()), 0);
}

// Create a model with Cast + Add (quantized)
// cast_input -> Cast -> Q -> DQ ----+
//                                   |
//             input2 -> Q -> DQ -> Add -> Q -> DQ -> output
static GetTestModelFn BuildCastAddTestCase() {
  return [](ModelTestBuilder& builder) {
    // Creat Cast node int32 -> float32
    NodeArg* cast_input = MakeTestInput(builder, TestInputDef<int32_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));

    auto* cast_output = builder.MakeIntermediate();
    Node& cast_node = builder.AddNode("Cast", {cast_input}, {cast_output});
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));

    // Create Add node
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    auto* add_input1_qdq = AddQDQNodePair<uint8_t>(builder, cast_output, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add_input2 = MakeTestInput(builder, TestInputDef<float>({2, 3}, false, data));
    auto* add_input2_qdq = AddQDQNodePair<uint8_t>(builder, add_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add_input1_qdq, add_input2_qdq}, {add_output});

    // add_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add_output, q_parameter.scale, q_parameter.zero_point);
  };
}

// Create a model with Add (quantized)
// input1 -> Q -> DQ ----
//                       |
// input2 -> Q -> DQ -> Add -> Q -> DQ -> output
static GetTestModelFn BuildAddTestCase() {
  return [](ModelTestBuilder& builder) {
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    NodeArg* add_input1 = MakeTestInput(builder, TestInputDef<float>({2, 3}, false, data));
    auto* add_input1_qdq = AddQDQNodePair<uint8_t>(builder, add_input1, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add_input2 = MakeTestInput(builder, TestInputDef<float>({2, 3}, true, data));
    auto* add_input2_qdq = AddQDQNodePair<uint8_t>(builder, add_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add_input1_qdq, add_input2_qdq}, {add_output});

    // add_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add_output, q_parameter.scale, q_parameter.zero_point);
  };
}

// Test that models with 2 inputs which has different data type can still generate the context binary
TEST_F(QnnHTPBackendTests, QnnContextBinaryGeneration2InputTypes) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildCastAddTestCase()(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./qnn_context_binary_int32_fp32_inputs_test.onnx";
  std::remove(context_model_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // clean up
  CleanUpCtxFile(context_model_file);
}

// Generate context cache model from the ONNX models with 2 inputs.
// The generated model should have same input order.
// The input ONNX model is created in the way that the model inputs order
// is different with the order in the graph (topological order).
// It cause issue if the generated model doesn't set the inputs/outputs explicitly.
TEST_F(QnnHTPBackendTests, QnnContextGeneration2InputsOrderIssue) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  const std::string context_model_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx_2_inputs_order_test.onnx"), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_model_file), model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto inputs = model->MainGraph().GetInputs();
  EXPECT_TRUE(inputs.size() == 2);
  EXPECT_TRUE(inputs[0]->Name() == "attention_mask");
  EXPECT_TRUE(inputs[1]->Name() == "Add_input_0");

  // clean up
  CleanUpCtxFile(context_model_file);
}

TEST_F(QnnHTPBackendTests, QnnContextGenerationNodeNamePrefix) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  std::string node_name_prefix = "node_name_prefix_test";

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  const std::string context_model_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
  so.AddConfigEntry(kOrtSessionOptionEpContextNodeNamePrefix, node_name_prefix.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx_2_inputs_order_test.onnx"), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_model_file), model, nullptr, DefaultLoggingManager().DefaultLogger()));
  for (auto& node : model->MainGraph().Nodes()) {
    if (node.OpType() == "EPContext") {
      EXPECT_TRUE(node.Name().find(node_name_prefix) != std::string::npos);
    }
  }

  // clean up
  CleanUpCtxFile(context_model_file);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheEmbedModeTest) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_model_file = "./qnn_context_binary_test.onnx";
  std::remove(context_model_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  std::unordered_map<std::string, std::string> session_option_pairs2;
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_model_file,
                       session_option_pairs2);
  // Clean up
  CleanUpCtxFile(context_model_file);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheNonEmbedModeTest) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_binary_file = "./testdata/qnn_context_cache_non_embed.onnx";
  std::string qnn_ctx_bin = "./testdata/qnn_context_cache_non_embed_qnn.bin";

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  std::remove(context_binary_file.c_str());
  std::remove(qnn_ctx_bin.c_str());

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(qnn_ctx_bin));

  std::unordered_map<std::string, std::string> session_option_pairs2;
  // Need to set the context file path since TestQDQModelAccuracy load the model from memory
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  // 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_binary_file,
                       session_option_pairs2);

  // load the model from file
  std::vector<char> buffer;
  {
    std::ifstream file(context_binary_file, std::ios::binary | std::ios::ate);
    if (!file)
      ORT_THROW("Error reading model");
    buffer.resize(narrow<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      ORT_THROW("Error reading model");
  }

  Ort::SessionOptions so;  // No need to set the context file path in so since it's load from file
  so.AppendExecutionProvider("QNN", provider_options);
#ifdef _WIN32
  std::wstring ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#else
  std::string ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#endif
  Ort::Session session(*ort_env.get(), ctx_model_file.c_str(), so);

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
  ASSERT_EQ(std::remove(qnn_ctx_bin.c_str()), 0);
}

// Run QDQ model on HTP 2 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// Then delete the context bin file to make the 2nd sesssion.Initialize() return the status with code INVALID_GRAPH
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_InvalidGraph) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_binary_file = "./qnn_context_cache_non_embed.onnx";
  std::filesystem::path context_bin = "qnn_context_cache_non_embed_qnn.bin";
  std::remove(context_binary_file.c_str());
  std::remove(context_bin.string().c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_bin));
  // Delete the Qnn context cache binary file
  EXPECT_TRUE(std::filesystem::remove(context_bin));

  // loads and run from Onnx skeleton file + Qnn context cache binary file
  onnx::ModelProto model_proto;
  onnxruntime::Model qnn_ctx_model;
  // Load the QNN context cache model from path specified
  ASSERT_STATUS_OK(qnn_ctx_model.Load(ToPathString(context_binary_file), model_proto));
  std::string qnn_ctx_model_data;
  model_proto.SerializeToString(&qnn_ctx_model_data);

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(qnn_ctx_model_data.data(), static_cast<int>(qnn_ctx_model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

std::string CreateQnnCtxModelWithNonEmbedMode(std::string external_bin_path) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  auto& logging_manager = DefaultLoggingManager();
  onnxruntime::Model model("QNN_ctx_model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  std::vector<int64_t> shape = {2, 3};
  NodeArg* graph_input = MakeTestInput(helper, TestInputDef<float>(shape, true, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  auto* graph_output = helper.MakeOutput<float>(shape);
  Node& ep_context_node = helper.AddNode("EPContext", {graph_input}, {graph_output}, kMSDomain);
  ep_context_node.AddAttribute("embed_mode", static_cast<int64_t>(0));
  ep_context_node.AddAttribute("ep_cache_context", external_bin_path);
  ep_context_node.AddAttribute("partition_name", "QNNExecutionProvider_QNN_1110111000111000111_1_0");
  ep_context_node.AddAttribute("source", "QNN");
  helper.SetGraphOutputs();
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  return model_data;
}

// Create a model with EPContext node. Set the node property ep_cache_context has ".."
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryRelativePathTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("../qnn_context.bin");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Create a model with EPContext node. Set the node property ep_cache_context has absolute path
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryAbsolutePathTest) {
#if defined(_WIN32)
  std::string external_ctx_bin_path = "D:/qnn_context.bin";
#else
  std::string external_ctx_bin_path = "/data/qnn_context.bin";
#endif
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode(external_ctx_bin_path);

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Create a model with EPContext node. Set the node property ep_cache_context to a file not exist
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileNotExistTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("qnn_context_not_exist.bin");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./qnn_context_not_exist.onnx"));
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options, &so)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  const auto status = session_object.Initialize();
  ASSERT_EQ(status.Code(), common::StatusCode::INVALID_GRAPH);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("session.model_external_initializers_file_folder_path"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("ep.context_file_path"));
}

// Create a model with EPContext node. Set the node property ep_cache_context to empty string
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileEmptyStringTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./test_ctx.onnx"));
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options, &so)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Run QDQ model on HTP with 2 inputs
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinary2InputsTest) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_model_file = "./qnn_context_binary_2inputs_test.onnx";
  std::remove(context_model_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  const TestInputDef<float> input_def1({1, 2, 3}, false, -10.0f, 10.0f);
  const TestInputDef<float> input_def2({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Add";

  // Runs model with DQ-> Add-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  std::unordered_map<std::string, std::string> session_option_pairs2;
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_model_file,
                       session_option_pairs2);
  // Clean up
  CleanUpCtxFile(context_model_file);
}

// Context binary only contains a single QNN graph, generated context cache model (detached mode) only has 1 EPContext node
// Create another Onnx model which also reference to the bin file,
// but the node name is not same with the QNN graph name inside the bin file.
// This is to support backward compatible for the models generated before the PR that
// make context generation support multi-partition
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_SingleNodeNameNotMatchGraphNameInCtx) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_model_file = "./qnn_context_cache_non_embed.onnx";
  std::filesystem::path context_bin = "qnn_context_cache_non_embed_qnn.bin";
  std::remove(context_model_file.c_str());
  std::remove(context_bin.string().c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_bin));

  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  auto& logging_manager = DefaultLoggingManager();
  onnxruntime::Model model("QNN_ctx_model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  std::vector<int64_t> shape = {1, 2, 3};
  NodeArg* graph_input = MakeTestInput(helper, TestInputDef<float>(shape, false, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  auto* graph_output = helper.MakeOutput<float>(shape);
  Node& ep_context_node = helper.AddNode("EPContext", {graph_input}, {graph_output}, kMSDomain);
  ep_context_node.AddAttribute("embed_mode", static_cast<int64_t>(0));
  ep_context_node.AddAttribute("ep_cache_context", context_bin.string());
  ep_context_node.AddAttribute("partition_name", "QNNExecutionProvider_QNN_1110111000111000111_1_0");
  ep_context_node.AddAttribute("source", "QNNExecutionProvider");
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  // loads and run from Onnx skeleton file + Qnn context cache binary file

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str()));
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options, &so)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::OK);

  // Clean up
  ASSERT_EQ(std::remove(context_model_file.c_str()), 0);
  ASSERT_EQ(std::remove(context_bin.string().c_str()), 0);
}

// Model has 2 EPContext nodes, both with main_context=1 and embedded context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextEmbeded) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx"), so);
}

// Model has 2 EPContext nodes, both with main_context=1 and external context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextExternal) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_external.onnx"), so);
}

static void CreateQdqModel(const std::string& model_file_name, const Logger& logger) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  onnxruntime::Model model(model_file_name, false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildAddTestCase()(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ASSERT_STATUS_OK(onnxruntime::Model::Save(model, ToPathString(model_file_name)));
}

static void DumpModelWithSharedCtx(ProviderOptions provider_options,
                                   const std::string& onnx_model_path1,
                                   const std::string& onnx_model_path2) {
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
  // enable ep.share_ep_contexts so that QNNEP share the QnnBackendManager across sessions
  so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

#ifndef __aarch64__
#ifndef _M_ARM64
  // weight sharing only available for v73 and higher
  provider_options["soc_model"] = "60";
#endif  // !_M_ARM64
#endif  // !__aarch64__

  so.AppendExecutionProvider("QNN", provider_options);

  // Create 2 sessions to generate context binary models, the 1st session will share the QnnBackendManager
  // to the 2nd session, so graphs from these 2 models are all included in the 2nd context binary
  Ort::Session session1(*ort_env, ToPathString(onnx_model_path1).c_str(), so);

  so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
  Ort::Session session2(*ort_env, ToPathString(onnx_model_path2).c_str(), so);
}

static void GetModelInputNames(const std::string& model_path,
                               std::vector<std::string>& input_names,
                               std::vector<std::string>& output_names,
                               const Logger& logger) {
  std::shared_ptr<Model> ctx_model;
  auto path_str = ToPathString(model_path);
  ASSERT_STATUS_OK(Model::Load(path_str, ctx_model, nullptr, logger));
  auto& ctx_graph = ctx_model->MainGraph();

  auto& inputs = ctx_graph.GetInputs();
  for (auto input : inputs) {
    input_names.push_back(input->Name());
  }

  auto& outputs = ctx_graph.GetOutputs();
  for (auto output : outputs) {
    output_names.push_back(output->Name());
  }
}

// 1. Create 2 QDQ models
// 2. Initialize 2 Ort sessions which share the same QNN EP from these 2 QDQ models
// with EpContextEnable = 1, to dump the context binary
// so, the 2nd context binary contains the graph from the 1st model
// 3. Start 2 ort session from the dumped context model,
// The 2nd session uses graph from 1st session
// 4. Run the 2nd session
TEST_F(QnnHTPBackendTests, QnnContextShareAcrossSessions) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so1.AppendExecutionProvider("QNN", provider_options);
  Ort::SessionOptions so2;
  so2.SetLogId("so2");
  so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so2.AppendExecutionProvider("QNN", provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  Ort::Session session1(*ort_env, ctx_model_file1.c_str(), so1);
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = session1.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);
#endif

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
}

TEST_F(QnnHTPBackendTests, VTCMBackupBufferSharing) {
  ProviderOptions provider_options;
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["backend_type"] = "htp";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  provider_options["enable_vtcm_backup_buffer_sharing"] = "1";
  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AppendExecutionProvider("QNN", provider_options);
  Ort::SessionOptions so2;
  so2.SetLogId("so2");
  so2.AppendExecutionProvider("QNN", provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  Ort::Session session1(*ort_env, ctx_model_file1.c_str(), so1);
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = session1.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);
#endif

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
}

TEST_F(QnnHTPBackendTests, FileMapping_Off) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["disable_file_mapped_weights"] = "1";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so1.AppendExecutionProvider("QNN", provider_options);
  Ort::SessionOptions so2;

  // Test CreateFromBinaryListAsync path
  provider_options["enable_vtcm_backup_buffer_sharing"] = "1";
  so2.SetLogId("so2");
  so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so2.AppendExecutionProvider("QNN", provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  Ort::Session session1(*ort_env, ctx_model_file1.c_str(), so1);
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = session1.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);
  auto ort_outputs2 = session2.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);
#endif

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
}

// For Ort sessions to generate the context binary, with session option ep.share_ep_contexts enabled
// Ort sessions will share the QnnBackendManager, so that all graphs from all models compile into the same Qnn context
TEST_F(QnnHTPBackendTests, QnnContextGenWeightSharingSessionAPI) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2,
                           DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());

  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // clean up
  for (auto model_path : onnx_model_paths) {
    ASSERT_EQ(std::remove(model_path.c_str()), 0);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    ASSERT_EQ(std::remove(ctx_model_path.c_str()), 0);
  }
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_name1.c_str()), 0);
}

// Session created from array wth ep.context_enable enabled without ep.context_file_path
// Error message expected
TEST_F(QnnHTPBackendTests, LoadFromArrayWithQnnEpContextGenPathValidation) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);
  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./qnn_context_binary_multi_partition_test.onnx";
  std::remove(context_model_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AppendExecutionProvider("QNN", provider_options);

  ORT_TRY {
    Ort::Session session1(*ort_env, model_data_span.data(), model_data_span.size(), so);
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&e]() {
      std::string e_message1(std::string(e.what()));
      ASSERT_TRUE(e_message1.find("Please specify a valid ep.context_file_path") != std::string::npos);
    });
  }

  ORT_TRY {
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");
    Ort::Session session2(*ort_env, model_data_span.data(), model_data_span.size(), so);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&ex]() {
      std::string e_message2(std::string(ex.what()));
      ASSERT_TRUE(e_message2.find("Please specify a valid ep.context_file_path") != std::string::npos);
    });
  }
}

TEST_F(QnnHTPBackendTests, QnnEpDynamicOptions) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.AppendExecutionProvider("QNN", provider_options);
  so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx"), so);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx", input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{3, 4};
  std::vector<float> input_value(3 * 4, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                output_names_c.data(), 1);

  const char* const workload_type[] = {"ep.dynamic.workload_type"};
  const char* const efficient_type[] = {"Efficient"};
  const char* const default_type[] = {"Default"};

  // Test Efficient & Default options
  session.SetEpDynamicOptions(workload_type, efficient_type, 1);
  ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                           output_names_c.data(), 1);

  session.SetEpDynamicOptions(workload_type, default_type, 1);
  ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                           output_names_c.data(), 1);

  // Test invalid EP dynamic option and invalid workload type
  const char* const dne[] = {"DNE"};
  try {
    session.SetEpDynamicOptions(workload_type, dne, 1);
    FAIL() << "Expected exception to be thrown for workload type DNE but was set successfully";
  } catch (const std::exception& e) {
    EXPECT_STREQ("Invalid EP Workload Type.", e.what());
  }

  try {
    session.SetEpDynamicOptions(dne, efficient_type, 1);
    FAIL() << "Expected exception to be thrown for dynamic option DNE but was set successfully";
  } catch (const std::exception& e) {
    EXPECT_STREQ("Unsupported EP Dynamic Option", e.what());
  }

  const char* const htp_perf_mode_type[] = {"ep.dynamic.qnn_htp_performance_mode"};
  const char* const eps_type[] = {"extreme_power_saver"};
  const char* const shp_type[] = {"sustained_high_performance"};
  session.SetEpDynamicOptions(htp_perf_mode_type, shp_type, 1);
  ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                           output_names_c.data(), 1);

  session.SetEpDynamicOptions(htp_perf_mode_type, eps_type, 1);
  ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                           output_names_c.data(), 1);

  session.SetEpDynamicOptions(htp_perf_mode_type, shp_type, 1);
  ort_output = session.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                           output_names_c.data(), 1);
}

// Implementation of OrtOutStreamWriteFunc that writes the compiled model to a file.
static OrtStatus* ORT_API_CALL TestWriteToStream(void* stream_state, const void* buffer, size_t buffer_num_bytes) {
  std::ofstream* outfile = reinterpret_cast<std::ofstream*>(stream_state);
  outfile->write(reinterpret_cast<const char*>(buffer), buffer_num_bytes);
  return nullptr;  // No error
}

// Implementation of OrtOutStreamWriteFunc that directly returns an OrtStatus indicating an error.
static OrtStatus* ORT_API_CALL ReturnStatusFromStream(void* stream_state, const void* buffer, size_t buffer_num_bytes) {
  ORT_UNUSED_PARAMETER(stream_state);
  ORT_UNUSED_PARAMETER(buffer);
  ORT_UNUSED_PARAMETER(buffer_num_bytes);
  return Ort::GetApi().CreateStatus(ORT_FAIL, "Error from OrtOutStreamWriteFunc callback");
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to custom write stream
TEST_F(QnnHTPBackendTests, CompileApi_InputFile_WriteOutputModelBytes) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_inputfile_writeoutputmodelbytes.onnx");
  std::filesystem::remove(input_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  const ORTCHAR_T* output_model_file = ORT_TSTR("compileapi_inputfile_writeoutputmodelbytes_ctx.onnx");
  std::filesystem::remove(output_model_file);

  // Open an output file. Test will incrementally write the output model to file
  // via calls to our OrtOutStreamWriteFunc callback.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
  std::ofstream outfile(output_model_file, std::ios::binary);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelWriteFunc(TestWriteToStream, reinterpret_cast<void*>(&outfile));
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  outfile.flush();
  outfile.close();

  // Check that the compiled model has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);
}

// Tests using an OrtOutStreamFunc function that returns an error.
TEST_F(QnnHTPBackendTests, CompileApi_OutputStream_ReturnStatus) {
  // Create a test model (in memory).
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  std::string model_data = test_model.Serialize();

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
  compile_options.SetOutputModelWriteFunc(ReturnStatusFromStream, nullptr);  // Set output stream that returns error
  compile_options.SetEpContextEmbedMode(true);

  // Compile the model. Expect a specific error status.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_EQ(status.GetErrorMessage(), "Error from OrtOutStreamWriteFunc callback");
}

struct CustomInitializerHandlerState {
  const ORTCHAR_T* external_file_path = nullptr;
  std::ofstream* outfile = nullptr;
};

static OrtStatus* ORT_API_CALL TestHandleInitializerDataFunc(void* state,
                                                             const char* initializer_name,
                                                             const OrtValue* c_initializer_value,
                                                             const OrtExternalInitializerInfo* /*c_external_info*/,
                                                             OrtExternalInitializerInfo** c_new_external_info) {
  Ort::Status final_status{nullptr};

  ORT_TRY {
    CustomInitializerHandlerState* custom_state = reinterpret_cast<CustomInitializerHandlerState*>(state);

    if (std::string("constant") == initializer_name) {
      // Keep a specific initializer in the model just to test both scenarios.
      // A real implementation may check the byte size and keep small initializers in the model.
      *c_new_external_info = nullptr;
      return nullptr;
    }

    //
    // Store other initializers in an external file.
    //
    Ort::ConstValue value{c_initializer_value};
    size_t byte_size = value.GetTensorSizeInBytes();
    int64_t offset = custom_state->outfile->tellp();
    const ORTCHAR_T* location = custom_state->external_file_path;

    custom_state->outfile->write(static_cast<const char*>(value.GetTensorRawData()), byte_size);
    custom_state->outfile->flush();

    // Provide caller (ORT) with the new external info.
    Ort::ExternalInitializerInfo new_external_info{nullptr};
    if (Ort::Status status = Ort::ExternalInitializerInfo::Create(location, offset, byte_size, new_external_info);
        !status.IsOK()) {
      return status.release();
    }

    *c_new_external_info = new_external_info.release();
  }
  ORT_CATCH(const Ort::Exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status{ex};
    }));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status(ex.what(), ORT_FAIL);
    }));
  }

  return final_status.release();
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to a file
//   - Use callback to specify where each initializer is stored (i.e., external file or within model).
TEST_F(QnnHTPBackendTests, CompileApi_InputFile_OutputFile_InitializerHandler) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler_ctx.onnx");
  const ORTCHAR_T* initializer_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler.bin");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);
  std::filesystem::remove(initializer_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, logging::Severity::kERROR, test_model);
  ASSERT_STATUS_OK(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  so.AppendExecutionProvider("QNN", provider_options);

  // Open a file to store external initializers. ORT will call our handler function for every initializer.
  ASSERT_FALSE(std::filesystem::exists(initializer_file));
  std::ofstream outfile(initializer_file, std::ios::binary);
  CustomInitializerHandlerState custom_state = {initializer_file, &outfile};

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetOutputModelGetInitializerLocationFunc(TestHandleInitializerDataFunc,
                                                           reinterpret_cast<void*>(&custom_state));
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  outfile.flush();
  outfile.close();

  ASSERT_TRUE(std::filesystem::exists(initializer_file));
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);
}

static OrtStatus* ORT_API_CALL ReuseExternalInitializers(void* state,
                                                         const char* /*initializer_name*/,
                                                         const OrtValue* /*initializer_value*/,
                                                         const OrtExternalInitializerInfo* external_info,
                                                         OrtExternalInitializerInfo** new_external_info) {
  Ort::Status final_status{nullptr};

  ORT_TRY {
    // If the original initializer was stored in an external file, keep it there (just for testing).
    if (external_info != nullptr) {
      Ort::ConstExternalInitializerInfo info(external_info);
      auto location = info.GetFilePath();
      int64_t offset = info.GetFileOffset();
      size_t byte_size = info.GetByteSize();

      Ort::ExternalInitializerInfo new_info(nullptr);
      Ort::Status status = Ort::ExternalInitializerInfo::Create(location.c_str(), offset, byte_size, new_info);
      if (!status.IsOK()) {
        return status.release();
      }

      *new_external_info = new_info.release();

      // Keep track of number of reused external initializers so that we can assert
      // that we reused the expected number of initializers.
      // THIS IS TEST CODE. An application would not do this.
      size_t* num_reused_ext_initializers = reinterpret_cast<size_t*>(state);
      *num_reused_ext_initializers += 1;

      return nullptr;
    }

    // If not originally external, save it within the generated compiled model
    *new_external_info = nullptr;
  }
  ORT_CATCH(const Ort::Exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status{ex};
    }));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status(ex.what(), ORT_FAIL);
    }));
  }

  return final_status.release();
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to a file
//   - Use callback to specify where each initializer is stored. We'll reuse external initializers
//     from original model!
TEST_F(QnnHTPBackendTests, CompileApi_InitializerHandler_ReuseExternalInitializers) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/conv_qdq_external_ini.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("testdata/conv_qdq_external_ini_reuse_ctx.onnx");
  std::filesystem::remove(output_model_file);

  size_t num_reused_ext_initializers = 0;

  // Create model compilation options from the session options.
  Ort::SessionOptions so;
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetOutputModelGetInitializerLocationFunc(ReuseExternalInitializers,
                                                           reinterpret_cast<void*>(&num_reused_ext_initializers));
  compile_options.SetEpContextEmbedMode(true);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  std::filesystem::remove(output_model_file);

  ASSERT_EQ(num_reused_ext_initializers, 2);  // Reused external conv weight and bias.
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
