// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"
#include "core/providers/webgpu/webgpu_context.h"
#endif
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/graph_capture_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Independent reference for ONNX MatMul, including vector promotion and
// broadcasting of leading batch dimensions.
static void ComputeExpectedResult(std::initializer_list<int64_t> a_dims,
                                  std::initializer_list<int64_t> b_dims,
                                  const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                                  TensorShapeVector& output_dims, std::vector<float>& out_vals) {
  ASSERT_GT(a_dims.size(), 0u);
  ASSERT_GT(b_dims.size(), 0u);
  TensorShapeVector a_shape(a_dims), b_shape(b_dims);
  const bool a_is_vector = a_shape.size() == 1;
  const bool b_is_vector = b_shape.size() == 1;
  if (a_is_vector) a_shape.insert(a_shape.begin(), 1);
  if (b_is_vector) b_shape.push_back(1);
  const size_t rank = std::max(a_shape.size(), b_shape.size());
  a_shape.insert(a_shape.begin(), rank - a_shape.size(), 1);
  b_shape.insert(b_shape.begin(), rank - b_shape.size(), 1);
  const int64_t M = a_shape[rank - 2];
  const int64_t K = a_shape.back();
  const int64_t N = b_shape.back();
  ASSERT_EQ(K, b_shape[rank - 2]);
  output_dims.clear();
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    ASSERT_TRUE(a_shape[axis] == b_shape[axis] || a_shape[axis] == 1 || b_shape[axis] == 1);
    output_dims.push_back(a_shape[axis] == 1 ? b_shape[axis] : a_shape[axis]);
  }
  const int64_t num_batches = std::accumulate(output_dims.begin(), output_dims.end(), int64_t{1},
                                              std::multiplies<int64_t>());
  out_vals.assign(static_cast<size_t>(num_batches * M * N), 0.0f);
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    int64_t remaining = batch;
    int64_t a_offset = 0, b_offset = 0;
    int64_t a_stride = M * K, b_stride = K * N;
    for (size_t i = output_dims.size(); i > 0; --i) {
      const size_t axis = i - 1;
      const int64_t coordinate = remaining % output_dims[axis];
      remaining /= output_dims[axis];
      if (a_shape[axis] != 1) a_offset += coordinate * a_stride;
      if (b_shape[axis] != 1) b_offset += coordinate * b_stride;
      a_stride *= a_shape[axis];
      b_stride *= b_shape[axis];
    }
    const float* a = a_vals.data() + a_offset;
    const float* b = b_vals.data() + b_offset;
    float* out = out_vals.data() + batch * M * N;
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a[m * K + k] * b[k * N + n];
        }
        out[m * N + n] = sum;
      }
    }
  }
  if (!a_is_vector) output_dims.push_back(M);
  if (!b_is_vector) output_dims.push_back(N);
}

static std::optional<std::string> GetForcedAlgorithmUnsupportedReason(
    const IExecutionProvider& ep,
    webgpu::MatMulAlgorithm algorithm) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  ORT_UNUSED_PARAMETER(ep);
  switch (algorithm) {
    case webgpu::MatMulAlgorithm::Subgroup:
    case webgpu::MatMulAlgorithm::PackedSplitK:
    case webgpu::MatMulAlgorithm::SubgroupMatrix:
      return "hardware-specific forced MatMul tests require direct adapter capability inspection.";
    case webgpu::MatMulAlgorithm::Gemv:
    case webgpu::MatMulAlgorithm::Naive:
    case webgpu::MatMulAlgorithm::Packed:
      return std::nullopt;
  }
  return std::nullopt;
#else
  auto& context = webgpu::WebGpuContextFactory::GetContext(ep.GetDeviceId());

  switch (algorithm) {
    case webgpu::MatMulAlgorithm::Subgroup: {
      if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups)) {
        return "subgroup requires the WebGPU Subgroups feature.";
      }
      const auto& adapter_info = context.AdapterInfo();
      if (!webgpu::intel::SelectMatMulSubgroupSize(
               adapter_info.subgroupMinSize,
               adapter_info.subgroupMaxSize,
               context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl))
               .has_value()) {
        return "subgroup requires a supported subgroup size (8, 16, or 32).";
      }
      break;
    }
    case webgpu::MatMulAlgorithm::PackedSplitK:
      if (context.GetSplitKConfig().GetSplitDimInner() == 0) {
        return "packed_split_k is not configured for the selected adapter.";
      }
      break;
    case webgpu::MatMulAlgorithm::SubgroupMatrix: {
      if (!context.DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        return "subgroup_matrix requires the WebGPU subgroup-matrix feature.";
      }

      const auto& adapter_info = context.AdapterInfo();
      const auto& device_configs = context.SubgroupMatrixConfigs();
      constexpr auto kF16 = wgpu::SubgroupMatrixComponentType::F16;
      if (!webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
              {device_configs.configs, device_configs.configCount},
              adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize,
              context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl),
              {{kF16, kF16, 8, 16, 16, 32, false},
               {kF16, kF16, 16, 16, 16, 32, false}})) {
        return "subgroup_matrix requires an 8x16x16 or 16x16x16 F16 configuration with subgroup size 32.";
      }
      break;
    }
    case webgpu::MatMulAlgorithm::Gemv:
    case webgpu::MatMulAlgorithm::Naive:
    case webgpu::MatMulAlgorithm::Packed:
      break;
  }

  return std::nullopt;
#endif
}

template <typename T, int version = 13>
void RunTestTyped(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims,
                  bool b_is_constant = false,
                  std::optional<webgpu::MatMulAlgorithm> forced_algorithm = std::nullopt,
                  OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
                  const char* expected_error = nullptr) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  std::unique_ptr<IExecutionProvider> webgpu_ep;
  if (forced_algorithm.has_value()) {
    WebGpuExecutionProviderTestOptions test_options{};
    test_options.forced_matmul_algorithm = forced_algorithm;
    webgpu_ep = WebGpuExecutionProviderWithTestOptions(test_options);
  } else {
    webgpu_ep = DefaultWebGpuExecutionProvider();
  }
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  if (forced_algorithm.has_value()) {
    if (const auto reason = GetForcedAlgorithmUnsupportedReason(*webgpu_ep, *forced_algorithm);
        reason.has_value()) {
      GTEST_SKIP() << *reason;
    }
  }

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  TensorShapeVector output_dims;
  std::vector<float> expected_vals;
  ASSERT_NO_FATAL_FAILURE(ComputeExpectedResult(a_dims, b_dims, a_vals, b_vals, output_dims, expected_vals));

  OpTester test("MatMul", version);
  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("A", a_dims, a_vals);
    test.AddInput<T>("B", b_dims, b_vals, b_is_constant);
    test.AddOutput<T>("Y", output_dims, expected_vals);
  } else {
    test.AddInput<T>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals), b_is_constant);
    test.AddOutput<T>("Y", output_dims, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options)
      .ConfigEp(std::move(webgpu_ep))
      .Config(expected_result, expected_error == nullptr ? "" : expected_error)
      .RunWithConfig();
}

template <int version = 13>
void RunBothTypes(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims) {
  RunTestTyped<float, version>(a_dims, b_dims);
  RunTestTyped<MLFloat16, version>(a_dims, b_dims);
}

template <typename T>
void RunZeroKTest(std::initializer_list<int64_t> a_dims,
                  std::initializer_list<int64_t> b_dims, bool graph_capture = false) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(
      webgpu::options::kEnableGraphCapture,
      graph_capture ? webgpu::options::kEnableGraphCapture_ON : webgpu::options::kEnableGraphCapture_OFF));
  ASSERT_STATUS_OK(config.AddConfigEntry(
      webgpu::options::kValidationMode, webgpu::options::kValidationMode_full));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  MatMulComputeHelper helper;
  ASSERT_STATUS_OK(helper.Compute(TensorShape(a_dims), TensorShape(b_dims)));
  ASSERT_EQ(helper.K(), 0);
  TensorShapeVector output_dims;
  std::vector<float> expected;
  ComputeExpectedResult(a_dims, b_dims, {}, {}, output_dims, expected);
  ASSERT_EQ(helper.OutputShape(), TensorShape(output_dims));
  WebGpuGraphCaptureTester test("MatMul", 13);
  test.AddInput<T>("A", a_dims, std::vector<T>{});
  test.AddInput<T>("B", b_dims, std::vector<T>{});
  const std::vector<T> expected_output(expected.size(), T{0.0f});
  test.AddOutput<T>("Y", output_dims, expected_output);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  if (graph_capture) {
    test.RunWithBoundCapture<T>(std::move(ep), options, TensorShape(output_dims), expected_output);
  } else {
    test.Config(options).ConfigEp(std::move(ep)).RunWithConfig();
  }
}

TEST(MatMulZeroKTest, Float32MatrixAndBatch) {
  RunZeroKTest<float>({2, 0}, {0, 3});
  RunZeroKTest<float>({1, 35, 0}, {0, 1024});
  RunZeroKTest<float>({2, 1, 3, 0}, {1, 4, 0, 5});
}

TEST(MatMulZeroKTest, HelperPreservesBroadcastBatchesWithZeroK) {
  MatMulComputeHelper helper;
  ASSERT_STATUS_OK(helper.Compute(TensorShape({2, 1, 3, 0}), TensorShape({1, 4, 0, 5})));
  EXPECT_EQ(helper.OutputShape(), TensorShape({2, 4, 3, 5}));
  ASSERT_STATUS_OK(helper.Compute(TensorShape({2, 1, 3, 4}), TensorShape({1, 4, 4, 5})));
  EXPECT_EQ(helper.OutputShape(), TensorShape({2, 4, 3, 5}));
}

TEST(MatMulZeroKTest, Float16MatrixAndBatch) {
  RunZeroKTest<MLFloat16>({2, 0}, {0, 3});
  RunZeroKTest<MLFloat16>({1, 35, 0}, {0, 1024});
  RunZeroKTest<MLFloat16>({2, 1, 3, 0}, {1, 4, 0, 5});
}

TEST(MatMulZeroKTest, VectorAndScalarOutputs) {
  RunZeroKTest<float>({0}, {0, 5});
  RunZeroKTest<float>({3, 0}, {0});
  RunZeroKTest<float>({0}, {0});
  RunZeroKTest<MLFloat16>({0}, {0, 5});
  RunZeroKTest<MLFloat16>({0}, {0});
}

TEST(MatMulZeroKTest, EmptyOutput) {
  RunZeroKTest<float>({0, 0}, {0, 3});
  RunZeroKTest<float>({1, 3, 0}, {0, 0, 5});
  RunZeroKTest<float>({2, 1, 3, 0}, {1, 0, 0, 5});
  RunZeroKTest<float>({0, 1, 3, 0}, {1, 2, 0, 5});
  RunZeroKTest<MLFloat16>({0, 0}, {0, 3});
  RunZeroKTest<MLFloat16>({1, 3, 0}, {0, 0, 5});
  RunZeroKTest<MLFloat16>({2, 1, 3, 0}, {1, 0, 0, 5});
  RunZeroKTest<MLFloat16>({0, 1, 3, 0}, {1, 2, 0, 5});
}

TEST(MatMulZeroKTest, GraphCaptureReplay) {
  RunZeroKTest<float>({1, 35, 0}, {0, 1024}, true);
  RunZeroKTest<MLFloat16>({1, 35, 0}, {0, 1024}, true);
}

template <typename T>
void RunConvMatMulTest(bool channels_last, bool initializer, bool empty_output = false) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                         webgpu::options::kValidationMode_full));
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                         webgpu::options::kEnableGraphCapture_ON));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  WebGpuGraphCaptureTester test("Conv", 11, channels_last ? kMSInternalNHWCDomain : kOnnxDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  if (channels_last) {
    test.AddAttribute("activation", std::string{"Clip"});
    test.AddAttribute("activation_params", std::vector<float>{-0.5f, 2.0f});
  }
  const int64_t batch = empty_output ? 0 : 2;
  const int64_t channels = empty_output ? 4 : 0;
  const std::vector<int64_t> input_shape = channels_last ? std::vector<int64_t>{batch, 2, 3, channels}
                                                         : std::vector<int64_t>{batch, channels, 2, 3};
  const std::vector<int64_t> output_shape = channels_last ? std::vector<int64_t>{batch, 2, 3, 3}
                                                          : std::vector<int64_t>{batch, 3, 2, 3};
  test.AddInput<T>("X", input_shape, std::vector<T>{});
  test.AddInput<T>("W", {3, channels, 1, 1}, std::vector<T>(3 * channels, T{1.0f}), initializer);
  test.AddInput<T>("B", {3}, std::vector<T>{T{-2.0f}, T{-0.25f}, T{3.5f}});
  std::vector<T> expected;
  for (int64_t index = 0; index < batch * 18; ++index) {
    const int64_t channel = channels_last ? index % 3 : (index / 6) % 3;
    const float value = channel == 0 ? -2.0f : channel == 1 ? -0.25f
                                                            : 3.5f;
    expected.push_back(T{channels_last ? std::clamp(value, -0.5f, 2.0f) : value});
  }
  test.AddOutput<T>("Y", output_shape, expected);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.RunWithBoundCapture<T>(std::move(ep), options, TensorShape(output_shape), expected);
}

TEST(MatMulZeroKTest, ChannelsFirstConvBias) {
  for (bool initializer : {false, true}) {
    RunConvMatMulTest<float>(false, initializer);
    RunConvMatMulTest<MLFloat16>(false, initializer);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(MatMulZeroKTest, ChannelsLastConvBiasAndActivation) {
  for (bool initializer : {false, true}) {
    RunConvMatMulTest<float>(true, initializer);
    RunConvMatMulTest<MLFloat16>(true, initializer);
  }
}

TEST(MatMulZeroKTest, SharedHelperEmptyOutputChannelsLastConv) {
  for (bool initializer : {false, true}) {
    RunConvMatMulTest<float>(true, initializer, true);
    RunConvMatMulTest<MLFloat16>(true, initializer, true);
  }
}
#endif

TEST(MatMulNaiveProgramTest, Broadcast4DExecution) {
  RunTestTyped<float>({3, 1, 1, 2}, {2, 2, 2});
  RunTestTyped<float>({2, 2, 3, 2}, {2, 1});
  RunTestTyped<float>({2, 3, 2}, {3, 2, 2, 1});
}

TEST(MatMulNaiveProgramTest, VectorExecution) {
  RunTestTyped<float>({2}, {2, 3});
  RunTestTyped<float>({3}, {3});
}

TEST(MatMulNaiveProgramTest, ZeroContractionDimensionExecution) {
  RunTestTyped<float>({1, 0}, {0, 8});
}

TEST(MatMulProgramTest, VectorFallbackExecution) {
  RunTestTyped<float>({8}, {8, 3});
  RunTestTyped<float>({2, 8}, {8});
  RunTestTyped<float>({8}, {8});
  RunTestTyped<float>({8}, {2, 8, 3});
  RunTestTyped<float>({2, 2, 8}, {8});
}

static std::string BuildDynamicMatMulModelBytes() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("dynamic_matmul");

  auto set_float_shape = [](ONNX_NAMESPACE::ValueInfoProto* value_info,
                            const char* first_dimension,
                            const char* second_dimension) {
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* shape = tensor_type->mutable_shape();
    shape->add_dim()->set_dim_param(first_dimension);
    if (second_dimension != nullptr) {
      shape->add_dim()->set_dim_param(second_dimension);
    } else {
      shape->add_dim()->set_dim_value(8);
    }
  };

  auto* a = graph->add_input();
  a->set_name("A");
  set_float_shape(a, "M", "K");

  auto* b = graph->add_input();
  b->set_name("B");
  set_float_shape(b, "K", nullptr);

  auto* y = graph->add_output();
  y->set_name("Y");
  set_float_shape(y, "M", nullptr);

  auto* node = graph->add_node();
  node->set_name("MatMul");
  node->set_op_type("MatMul");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

TEST(WebGpuMatMulAlgorithmTest, ForcedNaive) {
  RunTestTyped<float>({8, 8}, {8, 8}, false, webgpu::MatMulAlgorithm::Naive);
  RunTestTyped<float>({1, 0}, {0, 8}, false, webgpu::MatMulAlgorithm::Naive);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedGemv) {
  RunTestTyped<MLFloat16>({1, 128}, {128, 4}, false, webgpu::MatMulAlgorithm::Gemv);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedGemvRejectsFloatInputs) {
  RunTestTyped<float>({1, 2048}, {2048, 16}, false, webgpu::MatMulAlgorithm::Gemv,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm gemv");
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPacked) {
  RunTestTyped<float>({2, 2}, {2, 2}, false, webgpu::MatMulAlgorithm::Packed);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPackedRejectsZeroContractionDimension) {
  RunTestTyped<float>({1, 0}, {0, 1}, false, webgpu::MatMulAlgorithm::Packed,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm packed");
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroup) {
  RunTestTyped<float>({8, 32}, {32, 64}, false, webgpu::MatMulAlgorithm::Subgroup);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupRejectsZeroContractionDimension) {
  RunTestTyped<float>({1, 0}, {0, 1}, false, webgpu::MatMulAlgorithm::Subgroup,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm subgroup");
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPackedSplitK) {
  RunTestTyped<float>({1, 1024}, {1024, 16}, false, webgpu::MatMulAlgorithm::PackedSplitK);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPackedSplitKHandlesBatchedPartialSplit) {
  RunTestTyped<float>({2, 1, 1028}, {2, 1028, 16}, false,
                      webgpu::MatMulAlgorithm::PackedSplitK);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupMatrix) {
  RunTestTyped<MLFloat16>({32, 16}, {16, 32}, false, webgpu::MatMulAlgorithm::SubgroupMatrix);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupMatrixUnalignedConstantB) {
  RunTestTyped<MLFloat16>({32, 16}, {16, 36}, true, webgpu::MatMulAlgorithm::SubgroupMatrix);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupMatrixRejectsFloatInputs) {
  RunTestTyped<float>({32, 16}, {16, 32}, false, webgpu::MatMulAlgorithm::SubgroupMatrix,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm subgroup_matrix");
}

TEST(WebGpuMatMulAlgorithmTest, ReselectsAlgorithmForEachDynamicShape) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SessionOptions session_options;
  session_options.session_logid = "WebGpuDynamicMatMulAlgorithmSelection";
  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(webgpu_ep)));

  const std::string model_bytes = BuildDynamicMatMulModelBytes();
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const std::vector<std::string> output_names{"Y"};
  auto run = [&](const TensorShape& a_shape, std::vector<float> a_data,
                 const TensorShape& b_shape, std::vector<float> b_data,
                 const TensorShape& expected_shape, float expected_value) {
    OrtValue a_value;
    OrtValue b_value;
    CreateMLValue<float>(a_shape.GetDims(), a_data.data(), OrtMemoryInfo(), &a_value);
    CreateMLValue<float>(b_shape.GetDims(), b_data.data(), OrtMemoryInfo(), &b_value);

    NameMLValMap feeds;
    feeds.emplace("A", std::move(a_value));
    feeds.emplace("B", std::move(b_value));
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const Tensor& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), expected_shape);
    for (float value : output.DataAsSpan<float>()) {
      EXPECT_EQ(value, expected_value);
    }
  };

  // The first invocation uses Packed. The second has K=0 and must reselect Naive on the same
  // session and kernel; reusing the first execution plan would fail Packed's nonzero-K prerequisite.
  run(TensorShape({8, 8}), std::vector<float>(64, 1.0f),
      TensorShape({8, 8}), std::vector<float>(64, 1.0f),
      TensorShape({8, 8}), 8.0f);

  const float unused_storage = 0.0f;
  run(TensorShape({1, 0}), std::vector<float>{unused_storage},
      TensorShape({0, 8}), std::vector<float>{unused_storage},
      TensorShape({1, 8}), 0.0f);
}

// 2D aligned baseline shapes.
TEST(MatMul_Large, DISABLED_Aligned) {
  RunBothTypes({128, 64}, {64, 1024});
}

// 2D unaligned edge shapes.
TEST(MatMul_Large, DISABLED_Unaligned) {
  RunBothTypes({127, 64}, {64, 1024});
  RunBothTypes({127, 63}, {63, 1023});
  RunBothTypes({128, 36}, {36, 1024});
  RunBothTypes({128, 68}, {68, 1024});
}

// 3D broadcast and non-broadcast cases.
TEST(MatMul_Large, DISABLED_Broadcast3D) {
  RunBothTypes({2, 128, 64}, {64, 1024});
  RunBothTypes({2, 128, 64}, {2, 64, 1024});
  RunBothTypes({2, 128, 64}, {64, 1023});
  RunBothTypes({2, 128, 64}, {2, 64, 1023});
}

// 4D broadcast cases.
TEST(MatMul_Large, DISABLED_Broadcast4D) {
  RunBothTypes({2, 2, 128, 64}, {2, 64, 1024});
  RunBothTypes({2, 2, 128, 64}, {2, 64, 1023});
}

// Batched B (true bmm): A [..., M, K] x B [..., K, N] with matching batch. On the
// Subgroup path each (A, B) slice is dispatched on z. Covers small and
// larger batch counts with tile-aligned per-slice shapes.
TEST(MatMul_Large, DISABLED_BatchedB) {
  RunBothTypes({2, 128, 64}, {2, 64, 1024});
  RunBothTypes({4, 64, 128}, {4, 128, 256});
  RunBothTypes({8, 32, 64}, {8, 64, 64});
  RunBothTypes({16, 64, 64}, {16, 64, 128});
}

// Batched B with per-slice shapes that are not tile multiples. The odd-N slices
// (1023, 33) fall back to the generic path (the Intel f16 subgroup kernel needs
// an even B row stride); the even-N odd-M slice (130 x 65) still exercises the
// kernel's bounds-checked partial-M stores under z-dispatch. All must be correct.
TEST(MatMul_Large, DISABLED_BatchedB_Unaligned) {
  RunBothTypes({3, 127, 64}, {3, 64, 1023});
  RunBothTypes({5, 65, 96}, {5, 96, 130});
  RunBothTypes({2, 129, 80}, {2, 80, 33});
}

// Multi-dimensional batch: leading A/B dims collapse into the z grid.
TEST(MatMul_Large, DISABLED_BatchedB_4D) {
  RunBothTypes({2, 2, 64, 128}, {2, 2, 128, 256});
  RunBothTypes({2, 3, 32, 64}, {2, 3, 64, 96});
}

// Constant f16 weight with odd N. The Intel f16 subgroup-matrix load needs an
// even B row stride, so a non-constant odd-N B falls back to the generic path. When
// B is a constant initializer, the first Compute lazily pads it to an even stride
// (N+1) and the subgroup kernel consumes the cached copy via the N_b uniform (output
// is still written at the real, odd N). Marking B constant here exercises that
// padded path for both shared 2D and batched weights across several odd N (1023,
// 33, 65), with even K and both aligned and partial M. Results must match the
// reference. f16 only: the subgroup kernel is f16, so a float B would take the
// generic path.
TEST(MatMul_Large, DISABLED_ConstantWeightOddN) {
  RunTestTyped<MLFloat16>({128, 64}, {64, 1023}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({127, 64}, {64, 1023}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({64, 96}, {96, 33}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({130, 80}, {80, 65}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({2, 127, 64}, {2, 64, 1023}, /*b_is_constant=*/true);
}

// Broadcasted batch dims that are NOT identical but share the same batch
// *product* (A=[2,1,...], B=[1,2,...] -> [2,2,...]; A=[1,4,...], B=[4,1,...] ->
// [4,4,...]). A product-only batch check would wrongly route these onto the
// Subgroup path, which pairs slice i of A with slice i of B and copies A's
// shape to the output - producing the wrong output shape and mismatched pairing.
// N is even and every per-slice shape is tile-aligned, so only the
// identical-batch-dims guard (not the odd-N or partial-tile fallbacks) keeps them
// on the generic broadcasting MatMul. Results must match the broadcast reference.
TEST(MatMul_Large, DISABLED_BatchedB_BroadcastEqualProduct) {
  RunBothTypes({2, 1, 128, 64}, {1, 2, 64, 256});
  RunBothTypes({1, 4, 64, 128}, {4, 1, 128, 96});
}

}  // namespace test
}  // namespace onnxruntime
