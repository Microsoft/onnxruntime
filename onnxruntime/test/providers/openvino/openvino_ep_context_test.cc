// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/common/float16.h"
#include "core/common/cpuid_info.h"

#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gsl/gsl"
#include "nlohmann/json.hpp"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

using onnxruntime::MLFloat16;

constexpr size_t kOpenVINOEpContextTestMaxDataSize = size_t{1} << 30;

// Returns true only on Intel CPUs.
//
// The OVIR EP context tests are gated on this because that path is currently
// validated only on Intel silicon. In particular, embed_mode = 0 dumps the
// compiled model to a separate .bin file and memory-maps it back on reload;
// this round-trip is unsupported on non-Intel CPUs (e.g. AMD), where it can
// crash. On those CPUs the OVIR EP context tests are skipped.
bool IsIntelCPU() {
  return onnxruntime::CPUIDInfo::GetCPUIDInfo().GetCPUVendor() == "Intel";
}

struct OpenVINOEpContextCallbackState {
  bool write_called = false;
  bool read_called = false;
  size_t write_count = 0;
  size_t read_count = 0;
  size_t allocation_count = 0;
  size_t max_data_size = kOpenVINOEpContextTestMaxDataSize;
  bool fail_write = false;
  bool fail_read = false;
  bool return_empty = false;
  std::string name;
  std::vector<char> payload;
  std::unordered_map<std::string, std::vector<char>> payloads_by_name;
  std::vector<std::string> read_names;
};

OrtStatus* ORT_API_CALL StoreOpenVINOEpContextData(void* state, const char* name, const void* buffer,
                                                   size_t buffer_size) {
  auto& callback_state = *static_cast<OpenVINOEpContextCallbackState*>(state);
  callback_state.write_called = true;
  ++callback_state.write_count;
  callback_state.name = name;
  if (callback_state.fail_write) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "synthetic OpenVINO EPContext write callback failure");
  }
  callback_state.payload.clear();
  if (buffer_size != 0) {
    if (buffer == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "OpenVINO EPContext write callback received a null non-empty payload");
    }
    callback_state.payload.assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + buffer_size);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL LoadOpenVINOEpContextData(void* state, const char* name, OrtAllocator* allocator,
                                                  void** buffer, size_t* buffer_size) {
  auto& callback_state = *static_cast<OpenVINOEpContextCallbackState*>(state);
  callback_state.read_called = true;
  ++callback_state.read_count;
  callback_state.read_names.emplace_back(name);
  *buffer = nullptr;
  *buffer_size = 0;
  if (callback_state.fail_read) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "synthetic OpenVINO EPContext read callback failure");
  }

  const std::vector<char>* payload = &callback_state.payload;
  if (!callback_state.payloads_by_name.empty()) {
    const auto payload_it = callback_state.payloads_by_name.find(name);
    if (payload_it == callback_state.payloads_by_name.end()) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected OpenVINO EPContext data name");
    }
    payload = &payload_it->second;
  } else if (callback_state.name != name) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected OpenVINO EPContext data name");
  }

  if (callback_state.return_empty) {
    return nullptr;
  }
  if (payload->size() > callback_state.max_data_size) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "OpenVINO application callback rejected payload exceeding its allocation limit");
  }
  *buffer_size = payload->size();
  if (payload->empty()) {
    return nullptr;
  }

  ++callback_state.allocation_count;
  OrtStatus* status = Ort::GetApi().AllocatorAlloc(allocator, payload->size(), buffer);
  if (status == nullptr) {
    std::copy(payload->begin(), payload->end(), static_cast<char*>(*buffer));
  }
  return status;
}

TEST(OpenVINOEpContextCallbackTest, ApplicationReadLimitIsCheckedBeforeAllocation) {
  OpenVINOEpContextCallbackState callback_state;
  callback_state.name = "context.bin";
  callback_state.payload = {'o', 'v'};
  callback_state.max_data_size = callback_state.payload.size() - 1;
  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = nullptr;
  size_t buffer_size = 0;
  Ort::Status status{LoadOpenVINOEpContextData(&callback_state, "context.bin", allocator, &buffer, &buffer_size)};
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("application callback"));
  EXPECT_EQ(callback_state.allocation_count, 0u);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(buffer_size, 0u);

  callback_state.max_data_size = callback_state.payload.size();
  status = Ort::Status{LoadOpenVINOEpContextData(&callback_state, "context.bin", allocator, &buffer, &buffer_size)};
  std::unique_ptr<void, Ort::detail::AllocatedFree> buffer_guard{buffer, Ort::detail::AllocatedFree{allocator}};
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(callback_state.allocation_count, 1u);
  ASSERT_EQ(buffer_size, callback_state.payload.size());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(std::vector<char>(static_cast<char*>(buffer), static_cast<char*>(buffer) + buffer_size),
            callback_state.payload);
}

std::string GetExternalEpContextDataName(const std::filesystem::path& model_path) {
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_THROW_IF_ERROR(onnxruntime::Model::Load(model_path, model_proto));
  for (const auto& node : model_proto.graph().node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }

    bool is_external = false;
    std::string name;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "embed_mode") {
        is_external = attr.i() == 0;
      } else if (attr.name() == "ep_cache_context") {
        name = attr.s();
      }
    }
    if (is_external && !name.empty()) {
      return name;
    }
  }
  ORT_THROW("External EPContext data name not found in ", model_path.string());
}

// Runs a mul_1-style model (X[3,2] -> Y[3,2], Y = X * {1,2,3,4,5,6}) with
// X = all 2.0f and validates that Y == {2,4,6,8,10,12}.
void RunAndValidate(Ort::Session& session) {
  const std::array<int64_t, 2> input_shape = {3, 2};
  std::vector<float> input_data(6, 2.0f);
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  const std::array<const char*, 1> input_names = {"X"};
  const std::array<const char*, 1> output_names = {"Y"};
  std::vector<Ort::Value> output_tensors(1);

  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
              output_names.data(), output_tensors.data(), 1);

  ASSERT_TRUE(output_tensors[0].IsTensor());
  ASSERT_EQ(output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(), 6u);

  const float* out_data = output_tensors[0].GetTensorData<float>();
  EXPECT_THAT(std::vector<float>(out_data, out_data + 6),
              ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

void AddFloat16Initializer(ONNX_NAMESPACE::GraphProto* graph,
                           const std::string& name,
                           std::span<const int64_t> shape,
                           std::span<const float> values) {
  auto* init = graph->add_initializer();
  init->set_name(name);
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t d : shape) {
    init->add_dims(d);
  }
  std::vector<uint16_t> bits;
  bits.reserve(values.size());
  for (float v : values) {
    bits.push_back(MLFloat16(v).val);
  }
  init->set_raw_data(bits.data(), bits.size() * sizeof(uint16_t));
}

void AddValueInfo(ONNX_NAMESPACE::ValueInfoProto* vi,
                  const std::string& name,
                  ONNX_NAMESPACE::TensorProto_DataType elem_type,
                  std::span<const int64_t> shape) {
  vi->set_name(name);
  auto* type = vi->mutable_type()->mutable_tensor_type();
  type->set_elem_type(elem_type);
  auto* vi_shape = type->mutable_shape();
  for (int64_t d : shape) {
    vi_shape->add_dim()->set_dim_value(d);
  }
}

ONNX_NAMESPACE::NodeProto* AddNode(ONNX_NAMESPACE::GraphProto* graph,
                                   const std::string& op_type,
                                   const std::string& name,
                                   const std::vector<std::string>& inputs,
                                   const std::string& output) {
  auto* n = graph->add_node();
  n->set_op_type(op_type);
  n->set_name(name);
  for (const auto& in : inputs) {
    n->add_input(in);
  }
  n->add_output(output);
  return n;
}

// Adds a RandomUniformLike sink so the graph is not wholly supported by OVEP;
// this forces partitioning, which is what runs the constant folding pass that
// populates the constant-output map exercised by tests.
void AddUnsupportedSink(ONNX_NAMESPACE::GraphProto* graph, const std::string& input) {
  auto* n = AddNode(graph, "RandomUniformLike", "unsupported_sink", {input}, "Z");
  auto* attr = n->add_attribute();
  attr->set_name("dtype");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

// Runs an OVEP constant-output model and returns the requested f16 output as
// floats. Uses "AUTO:CPU" (not plain "CPU"): plain "CPU" takes OVEP's unified-
// compile fast path and never reaches CreateOVModel where the folding lives.
// ORT-side optimizations are disabled so OVEP (not ORT) folds the constants,
// which is what routes them through the constant-output fill path under test.
// The runtime input's values never affect the constant outputs, so it is fed
// as an all-zero tensor of the given shape. Asserts the output is f16.
std::vector<float> RunConstantOutput(const std::string& model_data,
                                     const std::string& input_name,
                                     std::span<const int64_t> input_shape,
                                     const char* output_name,
                                     size_t& out_element_count) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", "AUTO:CPU"}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), session_options);

  int64_t input_count = 1;
  for (int64_t d : input_shape) {
    input_count *= d;
  }
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<uint16_t> input_bits(static_cast<size_t>(input_count), MLFloat16(0.0f).val);
  Ort::Value input_tensor = Ort::Value::CreateTensor(
      mem_info, input_bits.data(), input_bits.size() * sizeof(uint16_t),
      input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  const std::array<const char*, 1> input_names = {input_name.c_str()};
  const std::array<const char*, 1> output_names = {output_name};
  std::vector<Ort::Value> output_tensors(1);
  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
              output_names.data(), output_tensors.data(), 1);

  EXPECT_TRUE(output_tensors[0].IsTensor());
  auto type_shape = output_tensors[0].GetTensorTypeAndShapeInfo();
  EXPECT_EQ(type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  out_element_count = type_shape.GetElementCount();
  const MLFloat16* out_data = output_tensors[0].GetTensorData<MLFloat16>();
  std::vector<float> result;
  result.reserve(out_element_count);
  for (size_t i = 0; i < out_element_count; ++i) {
    result.push_back(out_data[i].ToFloat());
  }
  return result;
}

}  // namespace

class OVEPEPContextTests : public ::testing::Test {
};

namespace onnxruntime {
namespace test {

// Test if folder path given to ep_context_file_path throws an error
TEST_F(OVEPEPContextTests, OVEPEPContextFolderPath) {
  Ort::SessionOptions sessionOptions;
  std::unordered_map<std::string, std::string> ov_options;

  // The below line could fail the test in non NPU platforms.Commenting it out so that the device used for building OVEP will be used.
  // ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("OVEP_Test_Model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());

  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_file_path = "./ep_context_folder_path/";

  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_file_path.c_str());
  sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), sessionOptions);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Runs an existing OVIR-encapsulated EP context model: "mul_1_ep_ctx_ovir.onnx"
// wraps a single EPContext node whose "ep_cache_context" points to a sibling
// OpenVINO IR (".xml" + ".bin"), so OVEP imports it via read_model()/
// compile_model() instead of a pre-compiled blob.
//
// OVIR detection is filename-based (".onnx" -> ".xml"), so the model must be
// loaded from a path with the ".xml"/".bin" siblings next to it.
//
// CPU only.
class OVEPEPContextOVIRTests : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!IsIntelCPU()) {
      GTEST_SKIP() << "OVIR EP context is only validated on Intel CPUs; skipping on non-Intel silicon.";
    }
  }

  static constexpr const char* kDevice = "CPU";
  static constexpr const ORTCHAR_T* kOvirModelPath = ORT_TSTR("testdata/mul_1_ep_ctx_ovir.onnx");
};

TEST_F(OVEPEPContextOVIRTests, RunEpCtxOvirModel) {
  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Set up session options targeting CPU.
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  // Load the OVIR EP context model from disk (path-based load is required for
  // OVIR encapsulation detection).
  Ort::Session session(*ort_env, kOvirModelPath, session_options);

  RunAndValidate(session);
}

TEST_F(OVEPEPContextOVIRTests, RunEpCtxOvirModelFromArray) {
  std::ifstream model_stream(kOvirModelPath, std::ios::binary);
  ASSERT_TRUE(model_stream);
  const std::string model_data{std::istreambuf_iterator<char>{model_stream},
                               std::istreambuf_iterator<char>{}};

  Ort::SessionOptions session_options;
  const std::filesystem::path absolute_model_path = std::filesystem::absolute(kOvirModelPath);
  session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, absolute_model_path.string().c_str());
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  Ort::Session session(*ort_env, model_data.data(), model_data.size(), session_options);
  RunAndValidate(session);
}

TEST_F(OVEPEPContextOVIRTests, RunEpCtxOvirModelWithMixedCaseExtension) {
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_mixed_case";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(out_dir); });

  const std::filesystem::path model_path = out_dir / "mixed_case.onnx";
  const std::filesystem::path xml_path = out_dir / "mixed_case.XML";
  const std::filesystem::path bin_path = out_dir / "mixed_case.bin";
  std::filesystem::copy_file("testdata/mul_1_ep_ctx_ovir.xml", xml_path);
  std::filesystem::copy_file("testdata/mul_1_ep_ctx_ovir.bin", bin_path);

  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(kOvirModelPath, model_proto));
  bool updated_context_name = false;
  for (auto& node : *model_proto.mutable_graph()->mutable_node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }
    for (auto& attribute : *node.mutable_attribute()) {
      if (attribute.name() == "ep_cache_context") {
        attribute.set_s(xml_path.filename().string());
        updated_context_name = true;
      }
    }
  }
  ASSERT_TRUE(updated_context_name);
  {
    std::ofstream output_stream(model_path, std::ios::binary);
    ASSERT_TRUE(output_stream);
    ASSERT_TRUE(model_proto.SerializeToOstream(&output_stream));
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  Ort::Session session(*ort_env, model_path.c_str(), session_options);
  RunAndValidate(session);
}

TEST_F(OVEPEPContextOVIRTests, ReadCallbackIsRejectedBeforeFilesystemFallback) {
  OpenVINOEpContextCallbackState callback_state;
  Ort::SessionOptions session_options;
  session_options.SetEpContextDataReadFunc(LoadOpenVINOEpContextData, &callback_state);
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  std::string error;
  try {
    Ort::Session session(*ort_env, kOvirModelPath, session_options);
    FAIL() << "Expected OVIR EPContext read callback rejection";
  } catch (const Ort::Exception& ex) {
    error = ex.what();
  }

  EXPECT_THAT(error,
              testing::HasSubstr("OpenVINO OVIR EPContext does not support EPContext data read callbacks"));
  EXPECT_FALSE(callback_state.read_called);
}

// Negative / security test: an OVIR-encapsulated EP context model whose
// "ep_cache_context" attribute points outside the model directory via "../"
// traversal (e.g. "../../../etc/evil.xml") must be rejected at session-creation
// time rather than silently reading an arbitrary file off disk.

TEST_F(OVEPEPContextOVIRTests, RejectsEpCacheContextPathTraversal) {
  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Load the known-good OVIR EP context model and rewrite its EPContext node so
  // that ep_cache_context escapes the model directory.
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(kOvirModelPath, model_proto));

  // Malicious relative path that escapes the model directory. The ".xml"
  // extension routes validation through the OVIR ".xml" branch in
  // EPCtxHandler::Initialize() (validated against the input model's directory),
  // and "evil.xml" matches the "evil.onnx" output stem below so the node is also
  // recognized as OVIR-encapsulated.
  const std::string malicious_xml_path = "../../../etc/evil.xml";

  bool patched = false;
  for (auto& node : *model_proto.mutable_graph()->mutable_node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.name() == "embed_mode") {
        attr.set_i(0);  // force non-embed so the path (not an inline blob) is validated
      } else if (attr.name() == "ep_cache_context") {
        attr.set_s(malicious_xml_path);
        patched = true;
      }
    }
  }
  ASSERT_TRUE(patched) << "Test model did not contain an EPContext ep_cache_context attribute to patch.";

  // Write the tampered model to a dedicated subfolder. The malicious ".xml" is
  // intentionally never created on disk: validation must reject the path before
  // any attempt to read it.
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_path_traversal";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path malicious_model = out_dir / "evil.onnx";
  {
    std::ofstream ofs(malicious_model, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << malicious_model;
    ASSERT_TRUE(model_proto.SerializeToOstream(&ofs)) << "Failed to serialize tampered model.";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  bool threw = false;
  std::string error_message;
  try {
    Ort::Session session(*ort_env, malicious_model.c_str(), session_options);
  } catch (const Ort::Exception& ex) {
    threw = true;
    error_message = ex.what();
  }

  std::filesystem::remove_all(out_dir);

  ASSERT_TRUE(threw)
      << "Session creation should have rejected the path-traversal ep_cache_context, but it succeeded.";
  EXPECT_THAT(error_message, ::testing::HasSubstr("escapes model directory"))
      << "Expected a path-escape rejection. Actual error: " << error_message;
}

// Generates an EP context model from the OVIR-encapsulated source model and
// then loads + runs the generated model, covering both EP context embed modes:
//   embed_mode = 1: the compiled context is serialized INLINE into the .onnx.
//   embed_mode = 0: the compiled context is dumped to a separate file and only
//                   its filename is stored in the .onnx EPContext node.
//
// embed_mode is only honored during generation (it is written into the
// EPContext node), so it must be exercised via a generate-then-run flow rather
// than by setting the option on a run-only session.
//
// CPU only. Parameter: embed_mode_enabled.
class OVEPOVIRModelsExportEPContextTests : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!IsIntelCPU()) {
      GTEST_SKIP() << "OVIR EP context export is only validated on Intel CPUs; skipping on non-Intel silicon.";
    }
  }

  static constexpr const char* kDevice = "CPU";
  static constexpr const ORTCHAR_T* kOvirModelPath = ORT_TSTR("testdata/mul_1_ep_ctx_ovir.onnx");
};

TEST_P(OVEPOVIRModelsExportEPContextTests, ExportEpCtxFromOVIRModel) {
  const bool embed_mode = GetParam();

  ASSERT_TRUE(std::filesystem::exists(kOvirModelPath))
      << "Missing OVIR EP context model. Expected testdata/mul_1_ep_ctx_ovir.onnx "
         "(with sibling .xml and .bin files).";

  // Generate the EP context model into a dedicated subfolder so that the
  // separately-dumped blob (embed_mode = 0) doesn't collide with testdata.
  const std::filesystem::path out_dir =
      std::filesystem::path("testdata") / (std::string("ovir_epctx_export_embed_") + (embed_mode ? "on" : "off"));
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path epctx_model = out_dir / "mul_1_ovir_epctx_export.onnx";

  // --- Generate EP context model ---
  {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx_model.string().c_str());
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, embed_mode ? "1" : "0");
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    // Creating the session triggers EP context export to epctx_model.
    Ort::Session session(*ort_env, kOvirModelPath, session_options);
  }

  ASSERT_TRUE(std::filesystem::exists(epctx_model))
      << "EP context model was not generated at " << epctx_model;

  OpenVINOEpContextCallbackState callback_state;
  if (!embed_mode) {
    callback_state.name = GetExternalEpContextDataName(epctx_model);
    const std::filesystem::path sidecar_path = epctx_model.parent_path() / callback_state.name;
    std::ifstream sidecar_stream(sidecar_path, std::ios::binary);
    ASSERT_TRUE(sidecar_stream);
    callback_state.payload.assign(std::istreambuf_iterator<char>{sidecar_stream}, std::istreambuf_iterator<char>{});
    sidecar_stream.close();
    ASSERT_FALSE(callback_state.payload.empty());
    ASSERT_TRUE(std::filesystem::remove(sidecar_path));
  }

  // --- Load + run the generated EP context model ---
  {
    Ort::SessionOptions session_options;
    session_options.SetEpContextDataReadFunc(LoadOpenVINOEpContextData, &callback_state);
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    Ort::Session session(*ort_env, epctx_model.c_str(), session_options);

    RunAndValidate(session);
  }

  EXPECT_EQ(callback_state.read_called, !embed_mode);
  EXPECT_EQ(callback_state.read_count, embed_mode ? 0u : 1u);

  if (!embed_mode) {
    // Restore the sidecar to exercise the independent filesystem-path tests below.
    {
      std::ofstream sidecar_stream(epctx_model.parent_path() / callback_state.name, std::ios::binary);
      ASSERT_TRUE(sidecar_stream);
      sidecar_stream.write(callback_state.payload.data(), gsl::narrow<std::streamsize>(callback_state.payload.size()));
      sidecar_stream.close();
      ASSERT_TRUE(sidecar_stream);
    }

    const std::filesystem::path external_initializers_dir = out_dir / "external_initializers";
    std::filesystem::create_directories(external_initializers_dir);

    {
      Ort::SessionOptions session_options;
      session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                     external_initializers_dir.string().c_str());
      std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
      session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

      try {
        Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
        FAIL() << "Session creation should fail when the EP context binary is resolved from the initializer folder.";
      } catch (const Ort::Exception& ex) {
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("External data path does not exist"));
        EXPECT_THAT(ex.what(), ::testing::Not(::testing::HasSubstr("validate_status.IsOK()")));
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("session.model_external_initializers_file_folder_path"));
        EXPECT_THAT(ex.what(), ::testing::HasSubstr("ep.context_file_path"));
      }
    }

    {
      Ort::SessionOptions session_options;
      session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                     external_initializers_dir.string().c_str());
      session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx_model.string().c_str());
      std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
      session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

      Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
      RunAndValidate(session);
    }
  }

  std::filesystem::remove_all(out_dir);
}

TEST_F(OVEPOVIRModelsExportEPContextTests, CompileApiExternalDataUsesCallbacks) {
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_compile_callbacks";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path epctx_model = out_dir / "mul_1_ovir_epctx_compile.onnx";
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(out_dir); });

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  OpenVINOEpContextCallbackState callback_state;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelPath(kOvirModelPath);
  compile_options.SetOutputModelPath(epctx_model.c_str());
  compile_options.SetEpContextEmbedMode(false);
  compile_options.SetEpContextDataWriteFunc(StoreOpenVINOEpContextData, &callback_state);

  Ort::Status compile_status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(compile_status.IsOK()) << compile_status.GetErrorMessage();
  ASSERT_TRUE(std::filesystem::exists(epctx_model));
  ASSERT_TRUE(callback_state.write_called);
  ASSERT_EQ(callback_state.write_count, 1u);
  ASSERT_FALSE(callback_state.name.empty());
  ASSERT_FALSE(callback_state.payload.empty());
  ASSERT_FALSE(std::filesystem::exists(epctx_model.parent_path() / callback_state.name));

  callback_state.max_data_size = callback_state.payload.size() - 1;
  session_options.SetEpContextDataReadFunc(LoadOpenVINOEpContextData, &callback_state);
  std::string oversized_error;
  try {
    Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
    FAIL() << "Expected application callback allocation-limit rejection";
  } catch (const Ort::Exception& ex) {
    oversized_error = ex.what();
    EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
  }
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_EQ(callback_state.read_count, 1u);
  EXPECT_EQ(callback_state.allocation_count, 0u);
  EXPECT_THAT(oversized_error,
              testing::HasSubstr("OpenVINO application callback rejected payload exceeding its allocation limit"));
  EXPECT_FALSE(std::filesystem::exists(epctx_model.parent_path() / callback_state.name));

  callback_state.read_called = false;
  callback_state.max_data_size = callback_state.payload.size();
  callback_state.return_empty = true;
  std::string empty_error;
  try {
    Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
    FAIL() << "Expected empty OpenVINO EPContext callback payload rejection";
  } catch (const Ort::Exception& ex) {
    empty_error = ex.what();
    EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
  }
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_EQ(callback_state.read_count, 2u);
  EXPECT_THAT(empty_error, testing::HasSubstr("OpenVINO EPContext read callback returned an empty payload"));
  EXPECT_FALSE(std::filesystem::exists(epctx_model.parent_path() / callback_state.name));

  callback_state.return_empty = false;
  callback_state.read_called = false;
  callback_state.fail_read = true;
  std::string read_error;
  try {
    Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
    FAIL() << "Expected OpenVINO EPContext read callback failure";
  } catch (const Ort::Exception& ex) {
    read_error = ex.what();
  }
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_EQ(callback_state.read_count, 3u);
  EXPECT_THAT(read_error, testing::HasSubstr("synthetic OpenVINO EPContext read callback failure"));
  EXPECT_FALSE(std::filesystem::exists(epctx_model.parent_path() / callback_state.name));

  callback_state.fail_read = false;
  callback_state.read_called = false;
  Ort::Session session(*ort_env, epctx_model.c_str(), session_options);
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_EQ(callback_state.read_count, 4u);
  EXPECT_EQ(callback_state.allocation_count, 1u);
  RunAndValidate(session);

  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(epctx_model, model_proto));
  const std::filesystem::path substring_model = out_dir / "native_callback_name.onnx";
  const std::string substring_name = "native_callback_name.xml_native.bin";
  bool updated_context_name = false;
  for (auto& node : *model_proto.mutable_graph()->mutable_node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }
    for (auto& attribute : *node.mutable_attribute()) {
      if (attribute.name() == "ep_cache_context") {
        attribute.set_s(substring_name);
        updated_context_name = true;
      }
    }
  }
  ASSERT_TRUE(updated_context_name);
  {
    std::ofstream model_stream(substring_model, std::ios::binary);
    ASSERT_TRUE(model_stream);
    ASSERT_TRUE(model_proto.SerializeToOstream(&model_stream));
  }

  callback_state.name = substring_name;
  callback_state.read_called = false;
  ASSERT_NO_THROW((Ort::Session(*ort_env, substring_model.c_str(), session_options)));
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_FALSE(std::filesystem::exists(substring_model.parent_path() / substring_name));

  const auto valid_payload = callback_state.payload;
  const auto expect_invalid_payload = [&](const std::vector<char>& payload, const char* error) {
    callback_state.payload = payload;
    callback_state.max_data_size = payload.size();
    try {
      Ort::Session invalid_session(*ort_env, substring_model.c_str(), session_options);
      ADD_FAILURE() << "Expected malformed callback payload rejection";
    } catch (const Ort::Exception& ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr(error));
      EXPECT_THAT(ex.what(), testing::Not(testing::HasSubstr("bad_alloc")));
    }
  };
  std::array<uint64_t, 5> malformed_header{};
  ASSERT_GE(valid_payload.size(), sizeof(malformed_header));
  std::memcpy(malformed_header.data(), valid_payload.data(), sizeof(malformed_header));
  for (const size_t field : {size_t{3}, size_t{4}}) {
    auto header = malformed_header;
    header[field] = std::numeric_limits<uint64_t>::max();
    auto payload = valid_payload;
    std::memcpy(payload.data(), header.data(), sizeof(header));
    expect_invalid_payload(payload, "Binary data range exceeds stream size");
  }
  expect_invalid_payload(std::vector<char>(valid_payload.begin(), valid_payload.begin() + 1),
                         "Binary data range exceeds stream size");
  for (const auto& [offset, size] : std::array<std::pair<int64_t, int64_t>, 4>{{
           {0, std::numeric_limits<int64_t>::max()},
           {std::numeric_limits<int64_t>::max(), 1},
           {-1, 1},
           {0, -1},
       }}) {
    const nlohmann::json metadata = {
        {"version", "1.0.0"},
        {"blob_metadata_map", {{"blob", {{"data_offset", offset}, {"size", size}}}}}};
    const auto bson = nlohmann::json::to_bson(metadata);
    auto header = malformed_header;
    header[3] = sizeof(header);
    header[4] = bson.size();
    std::vector<char> payload(sizeof(header) + bson.size());
    std::memcpy(payload.data(), header.data(), sizeof(header));
    std::memcpy(payload.data() + sizeof(header), bson.data(), bson.size());
    expect_invalid_payload(payload, offset < 0 || size < 0 ? "non-negative integers"
                                                           : "Binary data range exceeds stream size");
  }
  {
    nlohmann::json metadata = {
        {"version", "1.0.0"},
        {"blob_metadata_map", {
                                  {"first", {{"data_offset", 0}, {"size", 1}}},
                                  {"second", {{"data_offset", 0}, {"size", 1}}},
                              }}};
    const auto payload_size = sizeof(malformed_header) + nlohmann::json::to_bson(metadata).size();
    for (auto& blob : metadata["blob_metadata_map"]) {
      blob["size"] = payload_size;
    }
    const auto bson = nlohmann::json::to_bson(metadata);
    auto header = malformed_header;
    header[3] = sizeof(header);
    header[4] = bson.size();
    std::vector<char> payload(sizeof(header) + bson.size());
    std::memcpy(payload.data(), header.data(), sizeof(header));
    std::memcpy(payload.data() + sizeof(header), bson.data(), bson.size());
    expect_invalid_payload(payload, "Total blob data size exceeds stream size");
  }
  callback_state.payload = valid_payload;

  // Keep the native-format header but supply shared-weight metadata without enabling session sharing.
  std::array<uint64_t, 5> header{};
  ASSERT_GE(callback_state.payload.size(), sizeof(header));
  std::memcpy(header.data(), callback_state.payload.data(), sizeof(header));
  const nlohmann::json metadata = {
      {"version", "1.0.0"},
      {"weights_metadata_map", {{"weight", {{"location", "missing_shared_weights.bin"}, {"data_offset", 0}, {"size", 1}}}}},
      {"blob_metadata_map", nlohmann::json::object()}};
  const auto bson = nlohmann::json::to_bson(metadata);
  header[3] = sizeof(header);
  header[4] = bson.size();
  callback_state.payload.resize(sizeof(header) + bson.size());
  std::memcpy(callback_state.payload.data(), header.data(), sizeof(header));
  std::memcpy(callback_state.payload.data() + sizeof(header), bson.data(), bson.size());
  callback_state.max_data_size = callback_state.payload.size();
  callback_state.read_called = false;
  try {
    Ort::Session shared_weight_session(*ort_env, substring_model.c_str(), session_options);
    FAIL() << "Expected callback-backed external shared weights to be rejected";
  } catch (const Ort::Exception& ex) {
    EXPECT_THAT(ex.what(),
                testing::HasSubstr("OpenVINO callback-backed EPContext does not support external shared weights"));
    EXPECT_THAT(ex.what(), testing::Not(testing::HasSubstr("Failed to open weight file")));
  }
  EXPECT_TRUE(callback_state.read_called);
  EXPECT_FALSE(std::filesystem::exists(out_dir / "missing_shared_weights.bin"));
}

TEST_F(OVEPOVIRModelsExportEPContextTests, ReadCallbackDeduplicatesNamesWithinOneSession) {
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_distinct_callbacks";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(out_dir); });

  ONNX_NAMESPACE::ModelProto source_model;
  ASSERT_STATUS_OK(Model::Load(kOvirModelPath, source_model));
  auto combined_model = source_model;
  auto* combined_graph = combined_model.mutable_graph();
  combined_graph->clear_node();
  combined_graph->clear_input();
  combined_graph->clear_output();
  combined_graph->clear_value_info();
  std::ifstream xml_stream(std::filesystem::path(kOvirModelPath).replace_extension(".xml"));
  ASSERT_TRUE(xml_stream);
  const std::string source_xml{std::istreambuf_iterator<char>(xml_stream), std::istreambuf_iterator<char>()};
  std::array<uint64_t, 5> combined_header{};
  std::vector<char> combined_payload(sizeof(combined_header));
  nlohmann::json combined_metadata = {{"version", "1.0.0"}, {"blob_metadata_map", nlohmann::json::object()}};
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
  // Rename the OVIR tensor names as well: native imports retain their compiled input/output bindings.
  for (size_t i = 0; i < 3; ++i) {
    auto branch_model = source_model;
    auto* branch_graph = branch_model.mutable_graph();
    const auto prefix = "branch_" + std::to_string(i) + "_";
    auto xml = source_xml;
    for (const auto& name : {"X", "Y"}) {
      const auto old_name = "\"" + std::string(name) + "\"";
      const auto new_name = "\"" + prefix + name + "\"";
      size_t position = 0;
      while ((position = xml.find(old_name, position)) != std::string::npos) {
        xml.replace(position, old_name.size(), new_name);
        position += new_name.size();
      }
    }
    const auto xml_path = out_dir / (prefix + "model.xml");
    std::ofstream renamed_xml(xml_path);
    ASSERT_TRUE(renamed_xml);
    renamed_xml << xml;
    renamed_xml.close();
    std::filesystem::copy_file(std::filesystem::path(kOvirModelPath).replace_extension(".bin"),
                               out_dir / (prefix + "model.bin"));
    for (auto& branch_node : *branch_graph->mutable_node()) {
      auto* node = &branch_node;
      node->set_name(prefix + node->name());
      for (auto& input : *node->mutable_input()) {
        if (!input.empty()) input = prefix + input;
      }
      for (auto& output : *node->mutable_output()) {
        if (!output.empty()) output = prefix + output;
      }
      for (auto& attribute : *node->mutable_attribute()) {
        if (attribute.name() == "ep_cache_context") attribute.set_s(xml_path.filename().string());
      }
    }
    for (auto& input : *branch_graph->mutable_input()) {
      input.set_name(prefix + input.name());
    }
    for (auto& output : *branch_graph->mutable_output()) {
      output.set_name(prefix + output.name());
    }
    const auto source_path = out_dir / (prefix + "model.onnx");
    {
      std::ofstream model_stream(source_path, std::ios::binary);
      ASSERT_TRUE(branch_model.SerializeToOstream(&model_stream));
    }
    const auto native_path = out_dir / (prefix + "native.onnx");
    OpenVINOEpContextCallbackState write_state;
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(source_path.c_str());
    compile_options.SetOutputModelPath(native_path.c_str());
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetEpContextDataWriteFunc(StoreOpenVINOEpContextData, &write_state);
    Ort::Status compile_status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(compile_status.IsOK()) << compile_status.GetErrorMessage();
    ASSERT_EQ(write_state.write_count, 1u);
    const auto& payload = write_state.payload;
    ASSERT_GE(payload.size(), sizeof(combined_header));
    std::array<uint64_t, 5> header{};
    std::memcpy(header.data(), payload.data(), sizeof(header));
    const auto metadata = nlohmann::json::from_bson(payload.begin() + header[3],
                                                    payload.begin() + header[3] + header[4]);
    for (const auto& [name, entry] : metadata.at("blob_metadata_map").items()) {
      const auto offset = entry.at("data_offset").get<size_t>();
      const auto size = entry.at("size").get<size_t>();
      ASSERT_LE(offset, payload.size());
      ASSERT_LE(size, payload.size() - offset);
      combined_metadata["blob_metadata_map"][prefix + name] =
          {{"data_offset", combined_payload.size()}, {"size", size}};
      combined_payload.insert(combined_payload.end(), payload.begin() + offset, payload.begin() + offset + size);
    }
    combined_header = header;
    ONNX_NAMESPACE::ModelProto native_model;
    ASSERT_STATUS_OK(Model::Load(native_path, native_model));
    for (const auto& source_node : native_model.graph().node()) {
      auto* node = combined_graph->add_node();
      *node = source_node;
      node->set_name(prefix + node->name());
      for (auto& attribute : *node->mutable_attribute()) {
        if (attribute.name() == "partition_name") attribute.set_s(prefix + attribute.s());
        if (attribute.name() == "ep_cache_context") attribute.set_s("first_native.bin");
      }
    }
    for (const auto& input : native_model.graph().input()) *combined_graph->add_input() = input;
    for (const auto& output : native_model.graph().output()) *combined_graph->add_output() = output;
  }
  combined_header[3] = combined_payload.size();
  const auto bson = nlohmann::json::to_bson(combined_metadata);
  combined_header[4] = bson.size();
  combined_payload.insert(combined_payload.end(), bson.begin(), bson.end());
  std::memcpy(combined_payload.data(), combined_header.data(), sizeof(combined_header));
  ASSERT_EQ(combined_graph->node_size(), 3);
  const std::string second_name = "second_native.bin";
  bool updated_name = false;
  for (auto& attribute : *combined_graph->mutable_node(2)->mutable_attribute()) {
    if (attribute.name() == "ep_cache_context") {
      attribute.set_s(second_name);
      updated_name = true;
    }
  }
  ASSERT_TRUE(updated_name);
  const auto combined_path = out_dir / "combined_epctx.onnx";
  {
    std::ofstream model_stream(combined_path, std::ios::binary);
    ASSERT_TRUE(combined_model.SerializeToOstream(&model_stream));
  }
  OpenVINOEpContextCallbackState read_state;
  read_state.payloads_by_name.emplace("first_native.bin", combined_payload);
  read_state.payloads_by_name.emplace(second_name, combined_payload);
  session_options.SetEpContextDataReadFunc(LoadOpenVINOEpContextData, &read_state);
  ASSERT_NO_THROW((Ort::Session(*ort_env, combined_path.c_str(), session_options)));
  EXPECT_EQ(read_state.read_count, 2u);
  EXPECT_THAT(read_state.read_names,
              testing::UnorderedElementsAre("first_native.bin", second_name));
}

TEST_F(OVEPOVIRModelsExportEPContextTests, CompileApiWriteCallbackErrorDoesNotFallbackToDisk) {
  const std::filesystem::path out_dir = std::filesystem::path("testdata") / "ovir_epctx_callback_error";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path epctx_model = out_dir / "mul_1_ovir_epctx_error.onnx";
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(out_dir); });

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
  session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

  OpenVINOEpContextCallbackState callback_state;
  callback_state.fail_write = true;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelPath(kOvirModelPath);
  compile_options.SetOutputModelPath(epctx_model.c_str());
  compile_options.SetEpContextEmbedMode(false);
  compile_options.SetEpContextDataWriteFunc(StoreOpenVINOEpContextData, &callback_state);

  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(),
              testing::HasSubstr("synthetic OpenVINO EPContext write callback failure"));
  EXPECT_TRUE(callback_state.write_called);
  if (!callback_state.name.empty()) {
    EXPECT_FALSE(std::filesystem::exists(epctx_model.parent_path() / callback_state.name));
  }
}

TEST_F(OVEPOVIRModelsExportEPContextTests, EpContextCallbacksRejectSharedContexts) {
  OpenVINOEpContextCallbackState callback_state;

  {
    Ort::SessionOptions session_options;
    session_options.SetEpContextDataReadFunc(LoadOpenVINOEpContextData, &callback_state);
    session_options.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    std::string error;
    try {
      Ort::Session session(*ort_env, kOvirModelPath, session_options);
      FAIL() << "Expected OpenVINO shared-context read callback rejection";
    } catch (const Ort::Exception& ex) {
      error = ex.what();
    }
    EXPECT_THAT(error,
                testing::HasSubstr("OpenVINO shared EP contexts do not support EPContext data callbacks"));
  }

  {
    const std::filesystem::path output_model =
        std::filesystem::path("testdata") / "openvino_shared_context_callback_rejection.onnx";
    std::filesystem::remove(output_model);
    auto cleanup = gsl::finally([&]() { std::filesystem::remove(output_model); });

    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    std::unordered_map<std::string, std::string> ov_options = {{"device_type", kDevice}};
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(kOvirModelPath);
    compile_options.SetOutputModelPath(output_model.c_str());
    compile_options.SetEpContextDataWriteFunc(StoreOpenVINOEpContextData, &callback_state);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.GetErrorMessage(),
                testing::HasSubstr("OpenVINO shared EP contexts do not support EPContext data callbacks"));
    EXPECT_FALSE(std::filesystem::exists(output_model));
  }
}

INSTANTIATE_TEST_SUITE_P(
    OVEP_Tests,
    OVEPOVIRModelsExportEPContextTests,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<OVEPOVIRModelsExportEPContextTests::ParamType>& info) {
      return std::string("embed_") + (info.param ? "on" : "off");
    });

// Regression test for the float16 branch of FillOutputsWithConstantData
// (backend_utils.cc), which used to call FillOutputHelper<float> and so wrote
// 4-byte floats into a 2-byte-per-element f16 output tensor, corrupting the
// values and overrunning the buffer by 2x.
//
// That path (OVEP filling a constant graph output) only runs when all of:
//   - device is "AUTO:CPU": a plain "CPU" device takes OVEP's unified-compile
//     fast path and never reaches CreateOVModel, where the folding lives;
//   - the graph is not wholly supported, so partitioning runs the folding pass
//     (RandomUniformLike is the unsupported op that forces this);
//   - the constant branch C = Add(A, B) shares a cluster with a runtime input,
//     so it isn't dropped as an all-initializer cluster (via Y = Add(X, C));
//   - ORT optimizations are off, so OpenVINO (not ORT) folds Add(A, B).
// C is then emitted as an f16 Constant output filled by the code under test.
// Only C is asserted on; Y and Z exist only to shape the partition.
TEST(OVEPConstantOutputTests, FillsFloat16ConstantOutput) {
  const std::vector<int64_t> shape = {3, 2};
  const std::vector<float> a_f = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b_f = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  std::vector<float> expected;
  for (size_t i = 0; i < a_f.size(); ++i) {
    expected.push_back(a_f[i] + b_f[i]);
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_f16_add");

  AddFloat16Initializer(graph, "A", shape, a_f);
  AddFloat16Initializer(graph, "B", shape, b_f);

  // Runtime input; only exists so the constant branch shares a cluster with a
  // non-initializer input. Its values do not affect C.
  AddValueInfo(graph->add_input(), "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);

  // Constant branch: OpenVINO folds Add(A, B) into the f16 Constant output C.
  AddNode(graph, "Add", "add_const", {"A", "B"}, "C");
  // Pulls C into a cluster that has a runtime input (X), so it isn't dropped.
  AddNode(graph, "Add", "add_runtime", {"X", "C"}, "Y");
  // Unsupported by OVEP -> forces partitioning so the folding pass runs.
  AddUnsupportedSink(graph, "X");

  AddValueInfo(graph->add_output(), "C", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_c = 0;
  std::vector<float> actual = RunConstantOutput(model_data, "X", shape, "C", count_c);
  EXPECT_EQ(count_c, expected.size());
  EXPECT_THAT(actual, ::testing::ElementsAreArray(expected));
}

// Two same-shape constant outputs "D" and "D/x" whose names collide under
// '/'-truncation. Before the fix, "D/x"'s constant landed in "D"'s slot, so "D"
// returned Mul(A, B) instead of Add(A, B).
TEST(OVEPConstantOutputTests, RoutesConstantToCorrectOutputOnNameCollision) {
  const std::vector<int64_t> shape = {3};
  const std::vector<float> a = {1.0f, 2.0f, 3.0f};
  const std::vector<float> b = {10.0f, 20.0f, 30.0f};
  const std::vector<float> expected_d = {11.0f, 22.0f, 33.0f};   // Add(A, B)
  const std::vector<float> expected_dx = {10.0f, 40.0f, 90.0f};  // Mul(A, B)

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_name_collision");

  AddFloat16Initializer(graph, "A", shape, a);
  AddFloat16Initializer(graph, "B", shape, b);
  AddValueInfo(graph->add_input(), "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);

  // Two constants whose output names collide once truncated at the first '/'.
  AddNode(graph, "Add", "add_const", {"A", "B"}, "D");
  AddNode(graph, "Mul", "mul_const", {"A", "B"}, "D/x");
  // Pull both constants into a cluster that has a runtime input (X) so they are
  // not dropped as an all-initializer cluster.
  AddNode(graph, "Add", "add_runtime_d", {"X", "D"}, "XD");
  AddNode(graph, "Add", "add_runtime_dx", {"XD", "D/x"}, "Y");
  AddUnsupportedSink(graph, "X");

  AddValueInfo(graph->add_output(), "D", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "D/x", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_d = 0;
  std::vector<float> actual_d = RunConstantOutput(model_data, "X", shape, "D", count_d);
  EXPECT_EQ(count_d, expected_d.size());
  // This is the assertion that fails before the fix: output "D" carries the
  // "D/x" (Mul) constant instead of its own (Add) constant.
  EXPECT_THAT(actual_d, ::testing::ElementsAreArray(expected_d));

  size_t count_dx = 0;
  std::vector<float> actual_dx = RunConstantOutput(model_data, "X", shape, "D/x", count_dx);
  EXPECT_EQ(count_dx, expected_dx.size());
  EXPECT_THAT(actual_dx, ::testing::ElementsAreArray(expected_dx));
}

// Same name collision but with DIFFERENT shapes ("D" is [1], "D/x" is [8]).
// Before the fix, the 8-element "D/x" constant was routed to the 1-element "D"
// slot; with a pre-bound fixed-size buffer that is a heap overrun, caught by
// ASan. With the fix, "D" keeps its own [1] value. The FillOutputHelper size
// check is the in-EP backstop for any residual mis-route, turning an overrun
// into a thrown error.
TEST(OVEPConstantOutputTests, KeepsConstantOutputShapeOnNameCollision) {
  const std::vector<int64_t> small_shape = {1};
  const std::vector<int64_t> large_shape = {8};
  const std::vector<float> a1 = {1.0f};
  const std::vector<float> b1 = {10.0f};
  const std::vector<float> expected_d = {11.0f};  // Add(A1, B1)

  const std::vector<float> a8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> b8 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  std::vector<float> expected_dx;  // Mul(A8, B8)
  for (size_t i = 0; i < a8.size(); ++i) {
    expected_dx.push_back(a8[i] * b8[i]);
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("const_name_collision_mismatched");

  AddFloat16Initializer(graph, "A1", small_shape, a1);
  AddFloat16Initializer(graph, "B1", small_shape, b1);
  AddFloat16Initializer(graph, "A8", large_shape, a8);
  AddFloat16Initializer(graph, "B8", large_shape, b8);
  AddValueInfo(graph->add_input(), "X8", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);

  // Small [1] constant named "D"; large [8] constant named "D/x".
  AddNode(graph, "Add", "add_const", {"A1", "B1"}, "D");
  AddNode(graph, "Mul", "mul_const", {"A8", "B8"}, "D/x");
  // Keep both constants in a cluster that has the runtime input X8. "D" is
  // broadcast-added to X8 only to anchor it to the runtime cluster; its own
  // output value is unaffected.
  AddNode(graph, "Add", "add_runtime_dx", {"X8", "D/x"}, "Y8");
  AddNode(graph, "Add", "add_runtime_d", {"Y8", "D"}, "Y");
  AddUnsupportedSink(graph, "X8");

  AddValueInfo(graph->add_output(), "D", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, small_shape);
  AddValueInfo(graph->add_output(), "D/x", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);
  AddValueInfo(graph->add_output(), "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, large_shape);
  AddValueInfo(graph->add_output(), "Z", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, large_shape);

  std::string model_data;
  model.SerializeToString(&model_data);

  size_t count_d = 0;
  std::vector<float> actual_d = RunConstantOutput(model_data, "X8", large_shape, "D", count_d);
  // Before the fix, output "D" is overrun / reshaped by the [8] "D/x" constant.
  EXPECT_EQ(count_d, expected_d.size());
  EXPECT_THAT(actual_d, ::testing::ElementsAreArray(expected_d));

  size_t count_dx = 0;
  std::vector<float> actual_dx = RunConstantOutput(model_data, "X8", large_shape, "D/x", count_dx);
  EXPECT_EQ(count_dx, expected_dx.size());
  EXPECT_THAT(actual_dx, ::testing::ElementsAreArray(expected_dx));
}

}  // namespace test
}  // namespace onnxruntime
