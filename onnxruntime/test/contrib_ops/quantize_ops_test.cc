// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_environment.h"

#include <sstream>

namespace onnxruntime {
namespace test {

TEST(DequantizeLinearOpTest, CpuScaleTypeConstraints) {
  for (int scale_type : {ONNX_NAMESPACE::TensorProto::FLOAT, ONNX_NAMESPACE::TensorProto::FLOAT16,
                         ONNX_NAMESPACE::TensorProto::BFLOAT16}) {
    std::unordered_map<std::string, int> versions{{kOnnxDomain, 19}, {kMSDomain, 1}};
    Model model("DequantizeLinearCpuScales", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), versions, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    ModelTestBuilder builder(graph);
    auto* input = builder.MakeInput<int8_t>({2}, {-2, 4});
    NodeArg* scale;
    if (scale_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
      scale = builder.MakeScalarInitializer<float>(0.5f);
    } else if (scale_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
      scale = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(0.5f));
    } else {
      scale = builder.MakeScalarInitializer<BFloat16>(BFloat16(0.5f));
    }
    builder.AddNode("DequantizeLinear", {input, scale}, {builder.MakeOutput()}, kMSDomain)
        .AddAttribute("output_dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto::FLOAT));
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());
    std::string serialized;
    ASSERT_TRUE(model.ToProto().SerializeToString(&serialized));
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCpuExecutionProvider()));
    std::istringstream stream(serialized);
    ASSERT_STATUS_OK(session.Load(stream));
    const auto status = session.Initialize();
    if (scale_type == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
      EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED) << status.ErrorMessage();
      EXPECT_NE(status.ErrorMessage().find("DequantizeLinear"), std::string::npos);
    } else {
      ASSERT_STATUS_OK(status);
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(session.Run(RunOptions{}, builder.feeds_, builder.output_names_, &fetches));
      ASSERT_EQ(fetches.size(), 1u);
      const auto actual = fetches[0].Get<Tensor>().DataAsSpan<float>();
      ASSERT_EQ(actual.size(), 2u);
      EXPECT_EQ(actual[0], -1.0f);
      EXPECT_EQ(actual[1], 2.0f);
    }
  }
}

// scalar zero & scale with uint8
void TestDequantizeLinearPerTensorFloatUint8(bool use_initializer_except_x) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f}, use_initializer_except_x);
  test.AddInput<uint8_t>("x_zero_point", {}, {128}, use_initializer_except_x);
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  test.Run();
}

TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_uint8) {
  TestDequantizeLinearPerTensorFloatUint8(false);
}

// NNAPI EP requires weight to be an initializer
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_uint8_use_initializer_except_x) {
  TestDequantizeLinearPerTensorFloatUint8(true);
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_int8) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  // Disable Tensorrt EP due to error: node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int16 com.microsoft.DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_int16_cpu) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("x", dims, {-300, -30, -1025, 1270});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int16_t>("zero_point", {}, {-1024}, true);
  test.AddOutput<float>("y", dims, {1448.0f, 1988.0f, -2.0f, 4588.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint16 com.microsoft.DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_uint16_cpu) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("x", dims, {30000, 31000, 32768, 33000});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<float>("y", dims, {-5534.0f, -3534.0f, 2.0f, 466.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int32 DequantizeLinear with scalar zero-point & scale.
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_float_int32_cpu) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-300, -30, -1025, 1270});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int32_t>("zero_point", {}, {0}, true);
  test.AddOutput<float>("y", dims, {-600.0f, -60.0f, -2050.0f, 2540.0f});
  test.Run();
}

TEST(DequantizeLinearOpTest, DequantizeLinearOpTest_BroadcastTensorOfOne) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);

  test.AddInput<int32_t>("x", {4}, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {1}, {2.0f}, true);
  test.AddInput<int32_t>("zero_point", {1}, {0}, true);
  test.AddOutput<float>("y", {4}, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

#ifdef USE_CUDA
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_half_uint8) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<MLFloat16>("x_scale", {}, ToFloat16({2.0f}));
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<MLFloat16>("y", dims, ToFloat16({-256.0f, -250.0f, 0.0f, 254.0f}));
  test.Run();
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, DequantizeLinear_per_tensor_half_int8) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<MLFloat16>("x_scale", {}, ToFloat16({2.0f}));
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<MLFloat16>("y", dims, ToFloat16({-40.0f, 14.0f, 220.0f, 274.0f}));
  // Disable Tensorrt EP due to error: node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

// 1d zero & scale with uint8 broadcast axis 0
TEST(DequantizeLinearContribOpTest, DequantizeLinear_0) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("scale", {3},
                       {1.0f,
                        2.0f,
                        4.0f});
  test.AddInput<uint8_t>("zero_point", {3},
                         {0,
                          0,
                          0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 40, 80, 120});
  test.Run();
}

// 1d zero & scale with int8 broadcast axis 1
TEST(DequantizeLinearContribOpTest, DequantizeLinear_1) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<int8_t>("X", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("scale", {4}, {1, 2, 4, 8});
  test.AddInput<int8_t>("zero_point", {4}, {0, -10, -20, -30});
  test.AddOutput<float>("Y", dims,
                        {0, 22, 88, 264,
                         0, 24, 96, 288,
                         0, 40, 160, 480});
  // Disable Tensorrt EP because only zero zero_point is supported.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with int8 broadcast axis 0 with 4d tensor input/output
TEST(DequantizeLinearContribOpTest, DequantizeLinear_2) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{2, 3, 2, 4};
  test.AddInput<int8_t>("X", dims,
                        {7, 9, 10, 10,
                         5, 8, 9, 1,

                         8, 6, 7, 9,
                         10, 0, 7, 10,

                         8, 2, 6, 0,
                         5, 9, 8, 1,

                         2, 7, 5, 3,
                         2, 4, 1, 3,

                         8, 7, 4, 8,
                         10, 1, 5, 5,

                         7, 7, 0, 2,
                         4, 4, 0, 5});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("scale", {3}, {1, 10, 7});
  test.AddInput<int8_t>("zero_point", {3}, {10, 2, 1});
  test.AddOutput<float>("Y", dims,
                        {-3, -1, 0, 0,
                         -5, -2, -1, -9,

                         60, 40, 50, 70,
                         80, -20, 50, 80,

                         49, 7, 35, -7,
                         28, 56, 49, 0,

                         -8, -3, -5, -7,
                         -8, -6, -9, -7,

                         60, 50, 20, 60,
                         80, -10, 30, 30,

                         42, 42, -7, 7,
                         21, 21, -7, 28});
  // Disable Tensorrt EP because only zero zero_point is supported.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with uint8 broadcast axis -2 (-2 resolves to axis 0)
TEST(DequantizeLinearContribOpTest, DequantizeLinear_3) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", -2);
  test.AddInput<float>("scale", {3},
                       {1.0f,
                        2.0f,
                        4.0f});
  test.AddInput<uint8_t>("zero_point", {3},
                         {0,
                          0,
                          0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 40, 80, 120});
  test.Run();
}

// quantize with scalar zero point and scale
void TestQuantizeLinearPerTensorFloatUint8(bool use_initializer_except_x) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<float>("x", dims, {
                                      0.f, 2.f,        //
                                      3.f, -3.f,       // rounding half to even
                                      2.9f, -2.9f,     // low case
                                      3.1f, -3.1f,     // up case
                                      254.f, -256.f,   // critical point
                                      255.f, -257.f,   // critical point
                                      256.f, -258.f,   // critical point
                                      1000.f, -1000.f  // saturate case
                                  });
  test.AddInput<float>("y_scale", {}, {2.0f}, use_initializer_except_x);
  test.AddInput<uint8_t>("y_zero_point", {}, {128}, use_initializer_except_x);
  test.AddOutput<uint8_t>("y", dims,
                          {128, 129,
                           130, 126,
                           129, 127,
                           130, 126,
                           255, 0,
                           255, 0,
                           255, 0,
                           255, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_float_uint8) {
  TestQuantizeLinearPerTensorFloatUint8(false);
}

// Only NNAPI EP requires weight to be an initializer
#ifdef USE_NNAPI
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_float_uint8_use_initializer_except_x) {
  TestQuantizeLinearPerTensorFloatUint8(true);
}
#endif

TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_float_int8) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<float>("x", dims, {
                                      0.f, 2.f,        //
                                      3.f, -3.f,       // rounding half to even
                                      2.9f, -2.9f,     // low case
                                      3.1f, -3.1f,     // up case
                                      254.f, -256.f,   // critical point
                                      255.f, -257.f,   // critical point
                                      256.f, -258.f,   // critical point
                                      1000.f, -1000.f  // saturate case
                                  });
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {1});
  test.AddOutput<int8_t>("y", dims,
                         {1, 2,
                          3, -1,
                          2, 0,
                          3, -1,
                          127, -127,
                          127, -127,
                          127, -128,
                          127, -128});
  std::unordered_set<std::string> excluded_providers;
  // Disable Tensorrt EP due to error: node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

#ifdef USE_OPENVINO
TEST(QuantizeLinearContribOpTest, OVEPQuantizeLinear_per_tensor_float_int8) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<float>("x", dims, {
                                      0.f, 2.f,        //
                                      3.f, -3.f,       // rounding half to even
                                      2.9f, -2.9f,     // low case
                                      3.1f, -3.1f,     // up case
                                      254.f, -256.f,   // critical point
                                      255.f, -257.f,   // critical point
                                      256.f, -258.f,   // critical point
                                      1000.f, -1000.f  // saturate case
                                  });
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {1});
  test.AddOutput<int8_t>("y", dims,
                         {1, 2,
                          2, 0,
                          2, 0,
                          3, -1,
                          127, -127,
                          127, -128,
                          127, -128,
                          127, -128});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultOpenVINOExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_OPENVINO

// Test uint16 com.microsoft.QuantizeLinear (per tensor)
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_float_uint16) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{12};
  test.AddInput<float>("x", dims, {
                                      0.f, -128.f, 3.f, -3.f,  // rounding half to even
                                      2.9f, -2.9f,             // round < .5
                                      3.1f, -3.1f,             // round > .5
                                      65536.f, -65534.f,       // critical point
                                      70000.f, -70000.f        // saturate case
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<uint16_t>("y", dims,
                           {32767, 32703,
                            32769, 32765,
                            32768, 32766,
                            32769, 32765,
                            65535, 0,
                            65535, 0});
  std::unordered_set<std::string> excluded_providers;
  // Disable Tensorrt EP due to error: unsupported data type
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

#ifdef USE_OPENVINO
TEST(QuantizeLinearContribOpTest, OVEPQuantizeLinear_per_tensor_float_uint16) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{12};
  test.AddInput<float>("x", dims, {
                                      0.f, -128.f, 3.f, -3.f,  // rounding half to even
                                      2.9f, -2.9f,             // round < .5
                                      3.1f, -3.1f,             // round > .5
                                      65536.f, -65534.f,       // critical point
                                      70000.f, -70000.f        // saturate case
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<uint16_t>("y", dims,
                           {32767, 32703,
                            32768, 32766,
                            32768, 32766,
                            32769, 32765,
                            65535, 0,
                            65535, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultOpenVINOExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_OPENVINO

// Test int16 com.microsoft.QuantizeLinear (per tensor)
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_float_int16) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<float>("x", dims, {
                                      0.f, -514.f, 3.f, -3.f,  // rounding half to even
                                      2.9f, -2.9f,             // round < .5
                                      3.1f, -3.1f,             // round > .5
                                      65022.f, -66046.f,       // critical point
                                      65023.f, -66047.f,       // critical point
                                      65024.f, -66048.f,       // critical point
                                      70000.f, -70000.f        // saturate case
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int16_t>("zero_point", {}, {256}, true);
  test.AddOutput<int16_t>("y", dims,
                          {256, -1,
                           258, 254,
                           257, 255,
                           258, 254,
                           32767, -32767,
                           32767, -32768,
                           32767, -32768,
                           32767, -32768});

  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#ifdef USE_CUDA
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_half_uint8) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({
                                          0.f, 2.f,        //
                                          3.f, -3.f,       // rounding half to even
                                          2.9f, -2.9f,     // low case
                                          3.1f, -3.1f,     // up case
                                          254.f, -256.f,   // critical point
                                          255.f, -257.f,   // critical point
                                          256.f, -258.f,   // critical point
                                          1000.f, -1000.f  // saturate case
                                      }));
  test.AddInput<MLFloat16>("y_scale", {}, ToFloat16({2.0f}));
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims,
                          {128, 129,
                           130, 126,
                           129, 127,
                           130, 126,
                           255, 0,
                           255, 0,
                           255, 0,
                           255, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_tensor_half_int8) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{16};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({
                                          0.f, 2.f,        //
                                          3.f, -3.f,       // rounding half to even
                                          2.9f, -2.9f,     // low case
                                          3.1f, -3.1f,     // up case
                                          254.f, -256.f,   // critical point
                                          255.f, -257.f,   // critical point
                                          256.f, -258.f,   // critical point
                                          1000.f, -1000.f  // saturate case
                                      }));
  test.AddInput<MLFloat16>("y_scale", {}, ToFloat16({2.0f}));
  test.AddInput<int8_t>("y_zero_point", {}, {1});
  test.AddOutput<int8_t>("y", dims,
                         {1, 2,
                          3, -1,
                          2, 0,
                          3, -1,
                          127, -127,
                          127, -127,
                          127, -128,
                          127, -128});
  // Disable Tensorrt EP due to error: node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

// quantize with broadcasting
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_channel) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("scale", {3}, {1, 2, 4});
  test.AddInput<uint8_t>("zero_point", {3}, {0, 0, 0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 2, 3, 255,
                           0, 1, 2, 255,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with broadcasting and negative axis (-2 resolves to axis 0)
TEST(QuantizeLinearContribOpTest, QuantizeLinear_per_channel_negative_axis) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddAttribute<int64_t>("axis", -2);
  test.AddInput<float>("scale", {3}, {1, 2, 4});
  test.AddInput<uint8_t>("zero_point", {3}, {0, 0, 0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 2, 3, 255,
                           0, 1, 2, 255,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}
}  // namespace test
}  // namespace onnxruntime
