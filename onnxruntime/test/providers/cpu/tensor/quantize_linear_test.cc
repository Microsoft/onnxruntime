// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/framework/float6.h"
#include "core/framework/int4.h"
#include "core/framework/int2.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#ifdef USE_CUDA
#include "core/providers/cuda/tensor/quantize_linear_common.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
TEST(QuantizeLinearOpTest, CudaElementCountRange) {
  EXPECT_TRUE(cuda::IsQDQElementCountSupported(0));
  EXPECT_TRUE(cuda::IsQDQElementCountSupported(std::numeric_limits<int32_t>::max()));
  EXPECT_FALSE(cuda::IsQDQElementCountSupported(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1));
  EXPECT_FALSE(cuda::IsQDQElementCountSupported(-1));
  EXPECT_NE(cuda::QDQElementCountErrorMessage().find("INT32_MAX"), std::string::npos);
}

static void RunQDQOp25CudaOnly(OpTester& test) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (cuda_ep == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  SessionOptions so;
  auto status = so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_ep));
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Keep backward-compatible alias used by existing QuantizeLinear tests.
static void RunQuantizeLinearOp25CudaOnly(OpTester& test) {
  RunQDQOp25CudaOnly(test);
}
#endif  // USE_CUDA

template <typename QuantT, typename ScaleT, typename OutT>
void TestDequantizeLinearOutputType(int opset, int granularity, bool has_zero_point, int output_dtype_mode) {
  OpTester test("DequantizeLinear", opset);
  const std::vector<int64_t> dims{2, 3};
  const std::vector<QuantT> x{0, 1, 2, 3, 4, 5};
  std::vector<int64_t> scale_dims;
  std::vector<ScaleT> scales;
  InlinedVector<float> expected;
  const float zp = has_zero_point ? 2.0f : 0.0f;
  if (granularity == 0) {
    scales = {ScaleT(0.5f)};
    expected = {-zp * 0.5f, (1 - zp) * 0.5f, (2 - zp) * 0.5f,
                (3 - zp) * 0.5f, (4 - zp) * 0.5f, (5 - zp) * 0.5f};
  } else if (granularity == 1) {
    test.AddAttribute<int64_t>("axis", -1);
    scale_dims = {3};
    scales = {ScaleT(0.25f), ScaleT(0.5f), ScaleT(2.0f)};
    expected = {-zp * 0.25f, (1 - zp) * 0.5f, (2 - zp) * 2.0f,
                (3 - zp) * 0.25f, (4 - zp) * 0.5f, (5 - zp) * 2.0f};
  } else if (granularity == 2) {
    test.AddAttribute<int64_t>("axis", -1);
    test.AddAttribute<int64_t>("block_size", 2);
    scale_dims = {2, 2};
    scales = {ScaleT(0.25f), ScaleT(0.5f), ScaleT(1.0f), ScaleT(2.0f)};
    expected = {-zp * 0.25f, (1 - zp) * 0.25f, (2 - zp) * 0.5f,
                3 - zp, 4 - zp, (5 - zp) * 2.0f};
  } else {
    test.AddAttribute<int64_t>("axis", 0);
    test.AddAttribute<int64_t>("block_size", 2);
    scale_dims = {1, 3};
    scales = {ScaleT(0.25f), ScaleT(0.5f), ScaleT(2.0f)};
    expected = {-zp * 0.25f, (1 - zp) * 0.5f, (2 - zp) * 2.0f,
                (3 - zp) * 0.25f, (4 - zp) * 0.5f, (5 - zp) * 2.0f};
  }
  if (output_dtype_mode >= 0) {
    const int64_t output_type = std::is_same_v<OutT, float> ? ONNX_NAMESPACE::TensorProto::FLOAT
                                                            : ONNX_NAMESPACE::TensorProto::FLOAT16;
    test.AddAttribute<int64_t>("output_dtype", output_dtype_mode == 0 ? 0 : output_type);
  }
  test.AddInput<QuantT>("x", dims, x);
  test.AddInput<ScaleT>("x_scale", scale_dims, scales);
  if (has_zero_point) {
    test.AddInput<QuantT>("x_zero_point", scale_dims, std::vector<QuantT>(scales.size(), QuantT(2)));
  }
  std::vector<OutT> output(expected.begin(), expected.end());
  test.AddOutput<OutT>("y", dims, output);
  test.SetOutputTolerance(0.0f, 0.0f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

class DequantizeLinearOutputTypeTest : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(DequantizeLinearOutputTypeTest, Cpu) {
  const auto [opset, granularity, zero_point] = GetParam();
  TestDequantizeLinearOutputType<int8_t, float, MLFloat16>(opset, granularity, zero_point, 1);
  TestDequantizeLinearOutputType<uint8_t, MLFloat16, float>(opset, granularity, zero_point, 1);
  TestDequantizeLinearOutputType<uint8_t, float, MLFloat16>(opset, granularity, zero_point, 1);
  TestDequantizeLinearOutputType<int8_t, MLFloat16, float>(opset, granularity, zero_point, 1);
  for (int output_dtype : {-1, 0, 1}) {
    TestDequantizeLinearOutputType<int8_t, float, float>(opset, granularity, zero_point, output_dtype);
    TestDequantizeLinearOutputType<uint8_t, MLFloat16, MLFloat16>(opset, granularity, zero_point, output_dtype);
  }
}

INSTANTIATE_TEST_SUITE_P(DequantizeLinear, DequantizeLinearOutputTypeTest,
                         ::testing::Combine(::testing::Values(23, 24, 25),
                                            ::testing::Values(0, 1, 2, 3), ::testing::Bool()));

TEST(DequantizeLinearOpTest, FloatScaleHalfOutputRounding) {
  OpTester test("DequantizeLinear", 23);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto::FLOAT16);
  test.AddInput<int8_t>("x", {4}, {-128, -1, 0, 127});
  test.AddInput<float>("x_scale", {}, {1.0003f});
  test.AddOutput<MLFloat16>("y", {4}, {MLFloat16(-128.0f), MLFloat16(-1.0f), MLFloat16(0.0f), MLFloat16(127.0625f)});
  test.SetOutputTolerance(0.0f, 0.0f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DequantizeLinearOpTest, MixedTypesInt4Blocked) {
  OpTester test("DequantizeLinear", 23);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto::FLOAT);
  test.AddAttribute<int64_t>("block_size", 2);
  test.AddInput<Int4x2>("x", {2, 3}, {Int4x2(-8, -1), Int4x2(0, 1), Int4x2(6, 7)});
  test.AddInput<MLFloat16>("x_scale", {2, 2}, {MLFloat16(0.25f), MLFloat16(0.5f), MLFloat16(1.0f), MLFloat16(2.0f)});
  test.AddInput<Int4x2>("x_zero_point", {2, 2}, {Int4x2(1, 2), Int4x2(3, 4)});
  test.AddOutput<float>("y", {2, 3}, {-2.25f, -0.5f, -1.0f, -2.0f, 3.0f, 6.0f});
  test.SetOutputTolerance(0.0f, 0.0f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOpTest, MixedTypesFloat8) {
  OpTester test("DequantizeLinear", 23);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto::FLOAT16);
  test.AddInput<Float8E4M3FN>("x", {4}, {Float8E4M3FN(-2.0f, true), Float8E4M3FN(-0.5f, true), Float8E4M3FN(0.5f, true), Float8E4M3FN(2.0f, true)});
  test.AddInput<float>("x_scale", {}, {0.25f});
  test.AddOutput<MLFloat16>("y", {4}, {MLFloat16(-0.5f), MLFloat16(-0.125f), MLFloat16(0.125f), MLFloat16(0.5f)});
  test.SetOutputTolerance(0.0f, 0.0f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

template <typename InT, typename ScaleT>
void TestQuantizeLinearPrecision(int opset, const std::vector<InT>& x, ScaleT scale,
                                 int precision, const std::vector<int8_t>& expected) {
  OpTester test("QuantizeLinear", opset);
  if (precision >= 0) {
    test.AddAttribute<int64_t>("precision", precision);
  }
  test.AddInput<InT>("x", {static_cast<int64_t>(x.size())}, x);
  test.AddInput<ScaleT>("y_scale", {}, {scale});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", {static_cast<int64_t>(x.size())}, expected);
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(QuantizeLinearPrecisionTest, FloatScaleIsNotRoundedToHalf) {
  const std::vector<MLFloat16> x{MLFloat16(-15.5f), MLFloat16(15.5f)};
  for (int precision : {-1, 0, 1}) {
    TestQuantizeLinearPrecision<MLFloat16, float>(23, x, 1.0003f, precision, {-15, 15});
  }
  TestQuantizeLinearPrecision<MLFloat16, float>(23, x, 1.0003f, 10, {-16, 16});
}

TEST(QuantizeLinearPrecisionTest, HalfPrecisionRoundsInputAndQuotient) {
  for (int precision : {-1, 0, 10}) {
    TestQuantizeLinearPrecision<float, MLFloat16>(23, {-1.4998f, 1.4998f}, MLFloat16(1.0f), precision, {-2, 2});
    TestQuantizeLinearPrecision<MLFloat16, MLFloat16>(23, {MLFloat16(-1.0f), MLFloat16(1.0f)},
                                                      MLFloat16(0.4f), precision, {-2, 2});
  }
  TestQuantizeLinearPrecision<float, MLFloat16>(23, {-1.4998f, 1.4998f}, MLFloat16(1.0f), 1, {-1, 1});
  TestQuantizeLinearPrecision<MLFloat16, MLFloat16>(23, {MLFloat16(-1.0f), MLFloat16(1.0f)}, MLFloat16(0.4f), 1, {-3, 3});
  TestQuantizeLinearPrecision<float, float>(23, {-1.4998f, 1.4998f}, 1.0f, 10, {-2, 2});
  // Preserve the existing computation for opsets without the precision attribute.
  TestQuantizeLinearPrecision<MLFloat16, MLFloat16>(21, {MLFloat16(-1.0f), MLFloat16(1.0f)}, MLFloat16(0.4f), -1, {-3, 3});
}

template <typename T>
void TestQuantizeLinearMixedLimits() {
  for (int precision : {1, 10}) {
    OpTester test("QuantizeLinear", 23);
    test.AddAttribute<int64_t>("precision", precision);
    test.AddInput<float>("x", {5}, {-1.e30f, -0.5f, 0.5f, 1.5f, 1.e30f});
    test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(1.0f)});
    test.AddInput<T>("y_zero_point", {}, {T(1)});
    test.AddOutput<T>("y", {5}, {std::numeric_limits<T>::lowest(), T(1), T(1), T(3), std::numeric_limits<T>::max()});
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(QuantizeLinearPrecisionTest, IntegerLimitsAndNonzeroZeroPoint) {
  TestQuantizeLinearMixedLimits<int8_t>();
  TestQuantizeLinearMixedLimits<uint8_t>();
  TestQuantizeLinearMixedLimits<int16_t>();
  TestQuantizeLinearMixedLimits<uint16_t>();
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename T>
void TestQuantizeLinearMixedFloat8(bool saturate) {
  OpTester test("QuantizeLinear", 23);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddAttribute<int64_t>("block_size", 2);
  test.AddAttribute<int64_t>("saturate", saturate);
  test.AddAttribute<int64_t>("output_dtype", utils::ToTensorProtoElementType<T>());
  test.AddInput<MLFloat16>("x", {5}, {MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(4.0f), MLFloat16(8.0f), MLFloat16(65504.0f)});
  test.AddInput<float>("y_scale", {3}, {2.0f, 4.0f, 0.5f});
  test.AddOutput<T>("y", {5}, {T(0.0f, saturate), T(0.5f, saturate), T(1.0f, saturate), T(2.0f, saturate), T(131008.0f, saturate)});
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(QuantizeLinearPrecisionTest, Float8MixedTypes) {
  for (bool saturate : {false, true}) {
    TestQuantizeLinearMixedFloat8<Float8E4M3FN>(saturate);
  }
}
#endif

template <typename T, size_t elements_per_byte = 1, typename ScaleT>
void TestQuantizeLinearBlockedPrecision(int granularity, int precision) {
  OpTester test("QuantizeLinear", 25);
  const std::vector<int64_t> dims{3, 5};
  // a and b are exactly representable in FLOAT16, so only the rounding of the quotient differs.
  const ScaleT a(0.39990234375f);  // fp16(0.4)
  const ScaleT b(0.7998046875f);   // fp16(0.8)
  std::vector<int64_t> scale_dims;
  std::vector<ScaleT> scales;
  std::vector<int> values;
  switch (granularity) {
    case 0:
      test.AddAttribute<int64_t>("axis", 1);
      scale_dims = {5};
      scales = {a, b, a, b, a};
      values = {2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2};
      break;
    case 1:
      test.AddAttribute<int64_t>("axis", 1);
      test.AddAttribute<int64_t>("block_size", 3);
      scale_dims = {3, 2};
      scales = {a, b, b, a, a, b};
      values = {2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1};
      break;
    case 2:
      test.AddAttribute<int64_t>("axis", 0);
      test.AddAttribute<int64_t>("block_size", 2);
      scale_dims = {2, 5};
      scales = {a, b, a, b, a, b, a, b, a, b};
      values = {2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
      break;
    default:
      FAIL() << "Unexpected quantization granularity: " << granularity;
  }
  // 1.0 / fp16(0.4) is about 2.50061 and rounds to 3 in FLOAT precision. In FLOAT16 the quotient rounds
  // to 2.5 and then to even, 2. An omitted precision uses the y_scale type. 1.0 / fp16(0.8) gives 1 either way.
  const bool half = precision == ONNX_NAMESPACE::TensorProto::FLOAT16 ||
                    (precision < 0 && std::is_same_v<ScaleT, MLFloat16>);
  if (!half) {
    std::replace(values.begin(), values.end(), 2, 3);
  }
  std::vector<T> expected((values.size() + elements_per_byte - 1) / elements_per_byte);
  for (size_t i = 0; i < values.size(); ++i) {
    if constexpr (elements_per_byte > 1) {
      expected[i / elements_per_byte].SetElem(i % elements_per_byte,
                                              static_cast<typename T::UnpackedType>(values[i]));
    } else {
      expected[i] = static_cast<T>(values[i]);
    }
  }
  if (precision >= 0) {
    test.AddAttribute<int64_t>("precision", precision);
  }
  test.AddAttribute<int64_t>("output_dtype", utils::ToTensorProtoElementType<T>());
  test.AddInput<MLFloat16>("x", dims, std::vector<MLFloat16>(15, MLFloat16(1.0f)));
  test.AddInput<ScaleT>("y_scale", scale_dims, scales);
  test.AddOutput<T>("y", dims, expected);
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename ScaleT>
void TestQuantizeLinearBlockedPrecisionForScaleType() {
  for (int granularity : {0, 1, 2}) {
    for (int precision : {-1, static_cast<int>(ONNX_NAMESPACE::TensorProto::FLOAT),
                          static_cast<int>(ONNX_NAMESPACE::TensorProto::FLOAT16)}) {
      TestQuantizeLinearBlockedPrecision<int8_t, 1, ScaleT>(granularity, precision);
      TestQuantizeLinearBlockedPrecision<Int4x2, 2, ScaleT>(granularity, precision);
      TestQuantizeLinearBlockedPrecision<UInt2x4, 4, ScaleT>(granularity, precision);
    }
  }
}

TEST(QuantizeLinearPrecisionTest, PerAxisAndBlockedPrecision) {
  TestQuantizeLinearBlockedPrecisionForScaleType<float>();
  TestQuantizeLinearBlockedPrecisionForScaleType<MLFloat16>();
}

// scalar zero & scale with uint8
TEST(DequantizeLinearOpTest, Uint8) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, Int8) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with uint8 (large enough input to execute MLAS vectorized loop)
TEST(DequantizeLinearOpTest, Uint8_Large) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{1, 1039};  // not evenly divisible by 16 (loop unroll amount) to test handling of leftover inputs
  test.AddInput<uint8_t>("x", dims, std::vector<uint8_t>(1039, 1));
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddOutput<float>("y", dims, std::vector<float>(1039, 0.0f));
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  // Disable WebGPU EP because it requires dims.Size() to be multiple of 4. Fails with error: needs at least component size 4.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

// scalar zero & scale with int8 (large enough input to execute MLAS vectorized loop)
TEST(DequantizeLinearOpTest, Int8_Large) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{1, 1039};  // not evenly divisible by 16 (loop unroll amount) to test handling of leftover inputs
  test.AddInput<int8_t>("x", dims, std::vector<int8_t>(1039, 1));
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {1});
  test.AddOutput<float>("y", dims, std::vector<float>(1039, 0.0f));
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  // Disable WebGPU EP because it requires dims.Size() to be multiple of 4. Fails with error: needs at least component size 4.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(DequantizeLinearOpTest, Int4_LargeInitializerInput) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{1024};

  std::vector<Int4x2> x_vals(Int4x2::CalcNumInt4Pairs(static_cast<size_t>(dims[0])), Int4x2{});
  std::vector<float> expected_y_vals(static_cast<size_t>(dims[0]), 0.f);

  test.AddInput<Int4x2>("x", dims, x_vals, true);
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<Int4x2>("x_zero_point", {}, {Int4x2(0, 0)});
  test.AddOutput<float>("y", dims, expected_y_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Regression test: int8 tensor whose byte size is not a multiple of 4.
// DML graph fusion rounds tensor sizes to a multiple of 4 via AlignToPow2.
// If the original buffer is not padded, the subsequent memcpy reads past the
// allocation boundary (heap-buffer-overflow detectable with ASan).
// Mirrors the WebNN PoC: dequantizeLinear with int8[135] (135 % 4 != 0).
TEST(DequantizeLinearOpTest, Int8_NonAlignedSize_Initializer) {
  OpTester test("DequantizeLinear", 10);
  constexpr int64_t kNumElements = 135;  // 135 bytes, AlignToPow2(135,4)=136

  std::vector<int8_t> x_data(kNumElements);
  std::vector<float> y_expected(kNumElements);
  const float scale = 0.5f;
  const int8_t zero_point = 0;
  for (int64_t i = 0; i < kNumElements; ++i) {
    x_data[i] = static_cast<int8_t>(i % 127);
    y_expected[i] = (x_data[i] - zero_point) * scale;
  }

  // Mark all inputs as initializers so they go through DML's ProcessInputData
  // → UnpackInitializer → AlignToPow2 code path during graph fusion.
  test.AddInput<int8_t>("x", {kNumElements}, x_data, /*is_initializer=*/true);
  test.AddInput<float>("x_scale", {1}, {scale}, /*is_initializer=*/true);
  test.AddInput<int8_t>("x_zero_point", {1}, {zero_point}, /*is_initializer=*/true);
  test.AddOutput<float>("y", {kNumElements}, y_expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int4
TEST(DequantizeLinearOpTest, Int4) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of int4 values to test packing/unpacking
  test.AddInput<Int4x2>("x", dims, {Int4x2(-8, -3), Int4x2(1, 7), Int4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<Int4x2>("x_zero_point", {}, {Int4x2(-1, unused_val)});
  test.AddOutput<float>("y", dims, {-14.0f, -4.0f, 4.0f, 16.0f, 6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with int4
TEST(DequantizeLinearOpTest, Int4NoZeroPoint) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of int4 values to test packing/unpacking
  test.AddInput<Int4x2>("x", dims, {Int4x2(-8, -3), Int4x2(1, 7), Int4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {-16.0f, -6.0f, 2.0f, 14.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with uint4
TEST(DequantizeLinearOpTest, UInt4) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of uint4 values to test packing/unpacking
  test.AddInput<UInt4x2>("x", dims, {UInt4x2(0, 1), UInt4x2(3, 15), UInt4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<UInt4x2>("x_zero_point", {}, {UInt4x2(1, unused_val)});
  test.AddOutput<float>("y", dims, {-2.0f, 0.0f, 4.0f, 28.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with uint4
TEST(DequantizeLinearOpTest, UInt4NoZeroPoint) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of uint4 values to test packing/unpacking
  test.AddInput<UInt4x2>("x", dims, {UInt4x2(0, 1), UInt4x2(3, 15), UInt4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {0.0f, 2.0f, 6.0f, 30.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int2
// INT2 range: [-2, 1] (2-bit signed two's complement)
TEST(DequantizeLinearOpTest, Int2) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // 5 int2 values: -2, 1, 0, -1, 1 (requires 2 Int2x4 packed values)
  // Pack: (-2, 1, 0, -1) and (1, unused, unused, unused)
  test.AddInput<Int2x4>("x", dims, {Int2x4(-2, 1, 0, -1), Int2x4(1, unused_val, unused_val, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<Int2x4>("x_zero_point", {}, {Int2x4(-1, unused_val, unused_val, unused_val)});
  // y = (x - zp) * scale = ([-2, 1, 0, -1, 1] - (-1)) * 2 = [-1, 2, 1, 0, 2] * 2 = [-2, 4, 2, 0, 4]
  test.AddOutput<float>("y", dims, {-2.0f, 4.0f, 2.0f, 0.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DequantizeLinearOpTest, Int2_LargeInitializerInput) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{4096};
  std::vector<Int2x4> x_vals(Int2x4::CalcNumInt2Quads(static_cast<size_t>(dims[0])), Int2x4());

  test.AddInput<Int2x4>("x", dims, x_vals, true);
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<Int2x4>("x_zero_point", {}, {Int2x4()});
  test.AddOutput<float>("y", dims, std::vector<float>(static_cast<size_t>(dims[0]), 0.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with int2 (no zero point)
TEST(DequantizeLinearOpTest, Int2NoZeroPoint) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // 5 int2 values: -2, 1, 0, -1, 1
  test.AddInput<Int2x4>("x", dims, {Int2x4(-2, 1, 0, -1), Int2x4(1, unused_val, unused_val, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  // y = x * scale = [-2, 1, 0, -1, 1] * 2 = [-4, 2, 0, -2, 2]
  test.AddOutput<float>("y", dims, {-4.0f, 2.0f, 0.0f, -2.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with uint2
// UINT2 range: [0, 3]
TEST(DequantizeLinearOpTest, UInt2) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // 5 uint2 values: 0, 1, 2, 3, 1
  test.AddInput<UInt2x4>("x", dims, {UInt2x4(0, 1, 2, 3), UInt2x4(1, unused_val, unused_val, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<UInt2x4>("x_zero_point", {}, {UInt2x4(1, unused_val, unused_val, unused_val)});
  // y = (x - zp) * scale = ([0, 1, 2, 3, 1] - 1) * 2 = [-1, 0, 1, 2, 0] * 2 = [-2, 0, 2, 4, 0]
  test.AddOutput<float>("y", dims, {-2.0f, 0.0f, 2.0f, 4.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with uint2 (no zero point)
TEST(DequantizeLinearOpTest, UInt2NoZeroPoint) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // 5 uint2 values: 0, 1, 2, 3, 1
  test.AddInput<UInt2x4>("x", dims, {UInt2x4(0, 1, 2, 3), UInt2x4(1, unused_val, unused_val, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  // y = x * scale = [0, 1, 2, 3, 1] * 2 = [0, 2, 4, 6, 2]
  test.AddOutput<float>("y", dims, {0.0f, 2.0f, 4.0f, 6.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int16 DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, Int16) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("x", dims, {-300, -30, -1025, 1270});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int16_t>("zero_point", {}, {-1024}, true);
  test.AddOutput<float>("y", dims, {1448.0f, 1988.0f, -2.0f, 4588.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint16 DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, Uint16) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("x", dims, {30000, 31000, 32768, 33000});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<float>("y", dims, {-5534.0f, -3534.0f, 2.0f, 466.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int32
TEST(DequantizeLinearOpTest, Int32) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

// non-zero zero point with int32
TEST(DequantizeLinearOpTest, Int32_Non_Zero_Zero_Point) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f}, true);
  test.AddInput<int32_t>("x_zero_point", {}, {1}, true);
  test.AddOutput<float>("y", dims, {-62.f, -8.f, 198.f, 252.f});
  test.Run();
}

TEST(DequantizeLinearOpTest_BroadcastTensor, Int32) {
  OpTester test("DequantizeLinear", 13);
  test.AddInput<int32_t>("x", {4}, {-30, -3, 100, 127});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("x_scale", {1}, {2.0f});
  test.AddInput<int32_t>("x_zero_point", {1}, {0});
  test.AddOutput<float>("y", {4}, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

// 2d inputs
TEST(DequantizeLinearOpTest, 2D) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddInput<float>("scale", {}, {1.0f});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 10, 20, 30});
  test.Run();
}

// dequantize with scalar data
TEST(DequantizeLinearOpTest, Scalar) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", {}, {220.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 0.
  // Disable WebGPU EP due to error: needs at least component size 4
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

// dequantize with scalar data
TEST(DequantizeLinearOpMLFloat16Test, Scalar) {
  OpTester test("DequantizeLinear", 19);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<MLFloat16>("x_scale", {}, {MLFloat16(2.0f)});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<MLFloat16>("y", {}, {MLFloat16(220.0f)});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 0.
  // Disable WebGPU EP due to error: needs at least component size 4
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

// dequantize without zero point
TEST(DequantizeLinearOpTest, Without_Zero_Point) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {}, {200.0f});
  // No DQ allowed without corresponding Q. Skip since TRT10
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

// dequantize without zero point int8 (testing 8 elements for webgpu)
TEST(DequantizeLinearOpTest, No_Zero_Point_int8) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {1, 8}, {-10, 50, 100, 120, -9, 49, 99, 119});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {1, 8}, {-20.0f, 100.0f, 200.0f, 240.0f, -18.0f, 98.0f, 198.0f, 238.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // No DQ allowed without corresponding Q. Skip since TRT10
}

// dequantize without zero point uint8 (testing 8 elements for webgpu)
TEST(DequantizeLinearOpTest, No_Zero_Point_uint8) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<uint8_t>("x", {1, 8}, {10, 50, 100, 180, 9, 49, 99, 179});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {1, 8}, {20.0f, 100.0f, 200.0f, 360.0f, 18.0f, 98.0f, 198.0f, 358.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // No DQ allowed without corresponding Q. Skip since TRT10
}

// dequantize zero point int8 (testing 8 elements for webgpu)
TEST(DequantizeLinearOpTest, Zero_Point_int8) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {1, 8}, {-10, 50, 100, 120, -9, 49, 99, 119});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("zero_point", {}, {-10});
  test.AddOutput<float>("y", {1, 8}, {0.0f, 120.0f, 220.0f, 260.0f, 2.0f, 118.0f, 218.0f, 258.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // No DQ allowed without corresponding Q. Skip since TRT10
}

// dequantize zero point uint8 (testing 8 elements for webgpu)
TEST(DequantizeLinearOpTest, Zero_Point_uint8) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<uint8_t>("x", {1, 8}, {10, 50, 100, 180, 9, 49, 99, 119});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("zero_point", {}, {10});
  test.AddOutput<float>("y", {1, 8}, {0.0f, 80.0f, 180.0f, 340.0f, -2.0f, 78.0f, 178.0f, 218.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // No DQ allowed without corresponding Q. Skip since TRT10
}

// 1d zero & scale with default axis
TEST(DequantizeLinearOpTest, Per_Channel_Axis_Default) {
  OpTester test("DequantizeLinear", 13);
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
  // Disable Tensorrt EP due to the non-zero zero_point.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with uint8 broadcast axis 0
TEST(DequantizeLinearOpTest, Per_Channel_Axis_0) {
  OpTester test("DequantizeLinear", 13);
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
TEST(DequantizeLinearOpTest, Per_Channel_Axis_1_int8) {
  OpTester test("DequantizeLinear", 13);
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
  // Disable Tensorrt EP due to the non-zero zero_point.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with int32 broadcast axis 1
TEST(DequantizeLinearOpTest, Per_Channel_Axis_1_int32) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<int32_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 2, 4, 6,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("scale", {4}, {1, 2, 4, 8});
  test.AddInput<int32_t>("zero_point", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", dims,
                        {0, 2, 8, 24,
                         0, 4, 16, 48,
                         0, 20, 80, 240});
  // Disable Tensorrt EP due to error, only activation types allowed as input to this layer.
  // Disable CUDA EP, there is no implementation for int32_t.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kCudaExecutionProvider});
}

// 1d zero & scale with uint8 broadcast axis -2 (-2 resolves to axis 0)
TEST(DequantizeLinearOpTest, Per_Channel_Neg_2) {
  OpTester test("DequantizeLinear", 13);
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
TEST(QuantizeLinearOpTest, Uint8) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 1000, -254, -1000});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpMLFloat16Test, Uint8) {
  OpTester test("QuantizeLinear", 19);
  std::vector<int64_t> dims{6};
  test.AddInput<MLFloat16>("x", dims, {MLFloat16(0.0f), MLFloat16(2.0f), MLFloat16(4.0f), MLFloat16(1000.0f), MLFloat16(-254.0f), MLFloat16(-1000.0f)});
  test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(2.0f)});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpMLFloat16Test, Int8RoundsFractionalValues) {
  OpTester test("QuantizeLinear", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<MLFloat16>("x", dims,
                           {MLFloat16(0.050018310546875f), MLFloat16(-0.050018310546875f),
                            MLFloat16(0.04998779296875f), MLFloat16(-0.04998779296875f)});
  test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(0.0999755859375f)});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {1, -1, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: -127 and -128";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, -2, -5});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {0, 51, 76, 127, -51, -127});
  // Disable Tensorrt EP due to the error, out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Repro for new-delete-type-mismatch in DML EP during graph fusion.
// QuantizeLinear float32→int8 with 5D input triggers a type-size
// mismatch (192 bytes allocated, 1 byte deallocated) visible under ASan.
TEST(QuantizeLinearOpTest, Int8_5D_DML_TypeMismatch) {
  auto dml_ep = DefaultDmlExecutionProvider();
  if (!dml_ep) {
    GTEST_SKIP() << "Skipping because DML EP is not available.";
  }

  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{6, 1, 1, 1, 1};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {1, 2, 3, 4, 5, 6});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(dml_ep));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}

// Same as above but with per-axis quantization along axis 0 to exercise
// the DML graph fusion path with per-channel int8 quantization.
TEST(QuantizeLinearOpTest, Int8_5D_PerAxis_DML_TypeMismatch) {
  auto dml_ep = DefaultDmlExecutionProvider();
  if (!dml_ep) {
    GTEST_SKIP() << "Skipping because DML EP is not available.";
  }

  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{6, 1, 1, 1, 1};
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("y_scale", {6}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  test.AddInput<int8_t>("y_zero_point", {6}, {0, 0, 0, 0, 0, 0});
  test.AddOutput<int8_t>("y", dims, {1, 2, 3, 4, 5, 6});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(dml_ep));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}

// Opset 21 QuantizeLinear float32→uint8 WITHOUT zero_point.
// Without zero_point, the output type defaults to uint8.
TEST(QuantizeLinearOpTest, Uint8_5D_NoZeroPoint_Opset21_DML) {
  auto dml_ep = DefaultDmlExecutionProvider();
  if (!dml_ep) {
    GTEST_SKIP() << "Skipping because DML EP is not available.";
  }

  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{6, 1, 1, 1, 1};
  test.AddInput<float>("x", dims, {0.0f, 51.0f, 102.0f, 153.0f, 204.0f, 255.0f});
  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddOutput<uint8_t>("y", dims, {0, 51, 102, 153, 204, 255});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(dml_ep));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}

// Opset 21 QuantizeLinear float32→int8 with zero_point (the customer's exact scenario).
TEST(QuantizeLinearOpTest, Int8_5D_WithZeroPoint_Opset21_DML) {
  auto dml_ep = DefaultDmlExecutionProvider();
  if (!dml_ep) {
    GTEST_SKIP() << "Skipping because DML EP is not available.";
  }

  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{6, 1, 1, 1, 1};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {1, 2, 3, 4, 5, 6});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(dml_ep));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}

// Test uint16 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Uint16) {
  OpTester test("QuantizeLinear", 21);
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
TEST(QuantizeLinearOpTest, OVEP_Uint16) {
  OpTester test("QuantizeLinear", 21);
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

// Test int16 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Int16) {
  OpTester test("QuantizeLinear", 21);
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

// Test int4 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Int4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{7};
  constexpr int8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -20.0f,  // Clamp to qmin
                                      -16.0f,  // Close to qmin
                                      -3.0f,   // round
                                      0.0f,    // Zero-point
                                      2.9f,    // round
                                      12.0f,   // qmax
                                      20.0f,   // Clamp to qmax
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<Int4x2>("zero_point", {}, {Int4x2(1, unused_val)}, true);
  test.AddOutput<Int4x2>("y", dims,
                         {Int4x2(-8, -7), Int4x2(-1, 1), Int4x2(2, 7),
                          Int4x2(7, unused_val)});

  std::unordered_set<std::string> excluded_providers;
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

#ifdef USE_OPENVINO
TEST(QuantizeLinearOpTest, OVEP_Int4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{7};
  constexpr int8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -20.0f,  // Clamp to qmin
                                      -16.0f,  // Close to qmin
                                      -3.0f,   // round
                                      0.0f,    // Zero-point
                                      2.9f,    // round
                                      12.0f,   // qmax
                                      20.0f,   // Clamp to qmax
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<Int4x2>("zero_point", {}, {Int4x2(1, unused_val)}, true);
  test.AddOutput<Int4x2>("y", dims,
                         {Int4x2(-8, -7), Int4x2(0, 1), Int4x2(2, 7),
                          Int4x2(7, unused_val)});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultOpenVINOExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_OPENVINO

// Test uint4 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, UInt4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{7};
  constexpr uint8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -20.0f,  // Clamp to qmin
                                      -8.0f,   // qmin
                                      -3.0f,   // round
                                      0.0f,    // Zero-point
                                      2.9f,    // round
                                      22.0f,   // qmax
                                      30.0f,   // Clamp to qmax
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<UInt4x2>("zero_point", {}, {UInt4x2(4, unused_val)}, true);
  test.AddOutput<UInt4x2>("y", dims,
                          {UInt4x2(0, 0), UInt4x2(2, 4), UInt4x2(5, 15),
                           UInt4x2(15, unused_val)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#ifdef USE_CUDA
TEST(QuantizeLinearOpTest, Opset25_Uint8_Cuda) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 1000, -254, -1000});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});

  RunQuantizeLinearOp25CudaOnly(test);
}

TEST(QuantizeLinearOpMLFloat16Test, Opset25_PerAxisInt8_Cuda) {
  constexpr int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    GTEST_SKIP() << "CUDA compute capability " << min_cuda_architecture << " or higher is required.";
  }

  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<MLFloat16>("x", dims,
                           {MLFloat16(-4.0f), MLFloat16(-2.0f), MLFloat16(0.0f), MLFloat16(2.0f),
                            MLFloat16(4.0f), MLFloat16(6.0f), MLFloat16(8.0f), MLFloat16(10.0f)});
  test.AddInput<MLFloat16>("y_scale", {4},
                           {MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(4.0f), MLFloat16(4.0f)});
  test.AddInput<int8_t>("y_zero_point", {4}, {0, 0, 0, 0});
  test.AddOutput<int8_t>("y", dims, {-2, -1, 0, 0, 2, 3, 2, 2});

  RunQuantizeLinearOp25CudaOnly(test);
}

TEST(QuantizeLinearOpTest, Opset25_BlockedUInt4_Cuda) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", 2);
  test.AddInput<float>("x", dims, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f});
  test.AddInput<float>("y_scale", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  test.AddInput<UInt4x2>("y_zero_point", {2, 2}, {UInt4x2(0, 0), UInt4x2(0, 0)});
  test.AddOutput<UInt4x2>("y", dims,
                          {UInt4x2(0, 1), UInt4x2(2, 3), UInt4x2(4, 5), UInt4x2(6, 7)});

  RunQuantizeLinearOp25CudaOnly(test);
}

// DequantizeLinear opset 25 CUDA tests (exercises T1/T2/T3 type constraints)

TEST(DequantizeLinearOpTest, Opset25_Uint8_Cuda) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});

  RunQDQOp25CudaOnly(test);
}

TEST(DequantizeLinearOpTest, Opset25_Int8_Cuda) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});

  RunQDQOp25CudaOnly(test);
}

TEST(DequantizeLinearOpMLFloat16Test, Opset25_PerAxisInt8_Cuda) {
  constexpr int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    GTEST_SKIP() << "CUDA compute capability " << min_cuda_architecture << " or higher is required.";
  }

  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<int8_t>("x", dims, {-2, -1, 0, 1, 2, 3, 4, 5});
  test.AddInput<MLFloat16>("x_scale", {4},
                           {MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(4.0f), MLFloat16(4.0f)});
  test.AddInput<int8_t>("x_zero_point", {4}, {0, 0, 0, 0});
  // y = (x - zp) * scale
  test.AddOutput<MLFloat16>("y", dims,
                            {MLFloat16(-4.0f), MLFloat16(-2.0f), MLFloat16(0.0f), MLFloat16(4.0f),
                             MLFloat16(4.0f), MLFloat16(6.0f), MLFloat16(16.0f), MLFloat16(20.0f)});

  RunQDQOp25CudaOnly(test);
}

TEST(DequantizeLinearOpTest, Opset25_BlockedInt4_Cuda) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", 2);
  // int4 values: 0,1,2,3, 4,5,6,7
  test.AddInput<Int4x2>("x", dims,
                        {Int4x2(0, 1), Int4x2(2, 3), Int4x2(4, 5), Int4x2(6, 7)});
  test.AddInput<float>("x_scale", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  test.AddInput<Int4x2>("x_zero_point", {2, 2}, {Int4x2(0, 0), Int4x2(0, 0)});
  // y = (x - 0) * 2
  test.AddOutput<float>("y", dims, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f});

  RunQDQOp25CudaOnly(test);
}

TEST(DequantizeLinearOpTest, Opset25_BlockedUInt4_Cuda) {
  OpTester test("DequantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", 2);
  test.AddInput<UInt4x2>("x", dims,
                         {UInt4x2(0, 1), UInt4x2(2, 3), UInt4x2(4, 5), UInt4x2(6, 7)});
  test.AddInput<float>("x_scale", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  test.AddInput<UInt4x2>("x_zero_point", {2, 2}, {UInt4x2(0, 0), UInt4x2(0, 0)});
  test.AddOutput<float>("y", dims, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f});

  RunQDQOp25CudaOnly(test);
}
#endif  // USE_CUDA

template <bool Signed>
static void GetExpectedInt4Quant(const float* input, Int4x2Base<Signed>* output, size_t num_elems, float scale,
                                 int8_t zero_point) {
  using UnpackedType = typename Int4x2Base<Signed>::UnpackedType;

  for (size_t n = 0; n < num_elems; n++) {
    float float_val = std::nearbyintf(input[n] / scale) + static_cast<float>(zero_point);
    float_val = std::max(float_val, static_cast<float>(Int4x2Base<Signed>::min_val));
    float_val = std::min(float_val, static_cast<float>(Int4x2Base<Signed>::max_val));

    UnpackedType int_val = static_cast<UnpackedType>(float_val);

    size_t i = n >> 1;
    size_t j = n & 0x1;
    output[i].SetElem(j, int_val);
  }
}

template <bool Signed>
static void GetExpectedInt2Quant(const float* input, Int2x4Base<Signed>* output, size_t num_elems, float scale,
                                 int8_t zero_point) {
  using UnpackedType = typename Int2x4Base<Signed>::UnpackedType;

  for (size_t n = 0; n < num_elems; n++) {
    float float_val = std::nearbyintf(input[n] / scale) + static_cast<float>(zero_point);
    float_val = std::max(float_val, static_cast<float>(Int2x4Base<Signed>::min_val));
    float_val = std::min(float_val, static_cast<float>(Int2x4Base<Signed>::max_val));

    UnpackedType int_val = static_cast<UnpackedType>(float_val);

    size_t i = n >> 2;   // n / 4
    size_t j = n & 0x3;  // n % 4
    output[i].SetElem(j, int_val);
  }
}

// Test int2 QuantizeLinear (per tensor)
// INT2 range: [-2, 1]
TEST(QuantizeLinearOpTest, Int2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr int8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -6.0f,  // Clamp to qmin (-2)
                                      -4.0f,  // qmin
                                      -1.0f,  // round to 0 with zp=0
                                      0.0f,   // Zero-point
                                      4.0f,   // Clamp to qmax (1)
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<Int2x4>("zero_point", {}, {Int2x4(0, unused_val, unused_val, unused_val)}, true);
  // y = clamp(round(x / scale) + zp, -2, 1)
  // = clamp([-3, -2, -0.5, 0, 2] + 0, -2, 1) = [-2, -2, 0, 0, 1]
  test.AddOutput<Int2x4>("y", dims,
                         {Int2x4(-2, -2, 0, 0), Int2x4(1, unused_val, unused_val, unused_val)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint2 QuantizeLinear (per tensor)
// UINT2 range: [0, 3]
TEST(QuantizeLinearOpTest, UInt2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{5};
  constexpr uint8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -4.0f,  // Clamp to qmin (0)
                                      -2.0f,  // zp - 1 = 0
                                      0.0f,   // zp
                                      2.0f,   // zp + 1
                                      8.0f,   // Clamp to qmax (3)
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<UInt2x4>("zero_point", {}, {UInt2x4(1, unused_val, unused_val, unused_val)}, true);
  // y = clamp(round(x / scale) + zp, 0, 3)
  // = clamp([-2, -1, 0, 1, 4] + 1, 0, 3) = clamp([-1, 0, 1, 2, 5], 0, 3) = [0, 0, 1, 2, 3]
  test.AddOutput<UInt2x4>("y", dims,
                          {UInt2x4(0, 0, 1, 2), UInt2x4(3, unused_val, unused_val, unused_val)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int2 QuantizeLinear (per tensor) with a "large" number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks.
TEST(QuantizeLinearOpTest, Large_Int2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{1017};
  constexpr int8_t unused_val = 0;
  constexpr std::array<float, 4> pattern = {-4.0f, -2.0f, 0.0f, 2.0f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<Int2x4> output(Int2x4::CalcNumInt2Quads(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  int8_t zp = 0;
  GetExpectedInt2Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<Int2x4>("zero_point", {}, {Int2x4(zp, unused_val, unused_val, unused_val)}, true);
  test.AddOutput<Int2x4>("y", dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint2 QuantizeLinear (per tensor) with a "large" number of input elements.
TEST(QuantizeLinearOpTest, Large_UInt2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{1017};
  constexpr uint8_t unused_val = 0;
  constexpr std::array<float, 4> pattern = {-2.0f, 0.0f, 2.0f, 4.0f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<UInt2x4> output(UInt2x4::CalcNumInt2Quads(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  uint8_t zp = 1;
  GetExpectedInt2Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<UInt2x4>("zero_point", {}, {UInt2x4(zp, unused_val, unused_val, unused_val)}, true);
  test.AddOutput<UInt2x4>("y", dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int2 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks.
TEST(QuantizeLinearOpTest, OddLarge_Int2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{1019};  // Odd number, not multiple of 4
  constexpr int8_t unused_val = 0;
  constexpr std::array<float, 4> pattern = {-4.0f, -2.0f, 0.0f, 2.0f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<Int2x4> output(Int2x4::CalcNumInt2Quads(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  int8_t zp = 0;
  GetExpectedInt2Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<Int2x4>("zero_point", {}, {Int2x4(zp, unused_val, unused_val, unused_val)}, true);
  test.AddOutput<Int2x4>("y", dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint2 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
TEST(QuantizeLinearOpTest, OddLarge_UInt2) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{1019};  // Odd number, not multiple of 4
  constexpr uint8_t unused_val = 0;
  constexpr std::array<float, 4> pattern = {-2.0f, 0.0f, 2.0f, 4.0f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<UInt2x4> output(UInt2x4::CalcNumInt2Quads(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  uint8_t zp = 1;
  GetExpectedInt2Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<UInt2x4>("zero_point", {}, {UInt2x4(zp, unused_val, unused_val, unused_val)}, true);
  test.AddOutput<UInt2x4>("y", dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int2 QuantizeLinear per axis on the last axis. Each scale covers a single element, so every
// scale interval starts and ends inside a packed byte and must not overwrite its neighbors.
TEST(QuantizeLinearOpTest, Int2_PerAxis_LastAxis) {
  OpTester test("QuantizeLinear", 25);
  std::vector<int64_t> dims{2, 4};
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("x", dims, {1.0f, 1.0f, 1.0f, 1.0f, -2.0f, -2.0f, -2.0f, -2.0f});
  test.AddInput<float>("scale", {4}, {1.0f, 1.0f, 4.0f, 1.0f}, true);
  test.AddInput<Int2x4>("zero_point", {4}, {Int2x4(0, 0, 0, 0)}, true);
  // y = clamp(round(x / scale), -2, 1)
  // row 0: [1, 1, round(0.25), 1] = [1, 1, 0, 1]
  // row 1: [-2, -2, round(-0.5), -2] = [-2, -2, 0, -2]
  test.AddOutput<Int2x4>("y", dims, {Int2x4(1, 1, 0, 1), Int2x4(-2, -2, 0, -2)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int4 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks of even size.
TEST(QuantizeLinearOpTest, OddLarge_Int4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{1017};
  constexpr int8_t unused_val = 0;
  constexpr std::array<float, 6> pattern = {-20.0f, -14.0f, -4.1f, -0.0f, 3.0f, 3.3f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<Int4x2> output(Int4x2::CalcNumInt4Pairs(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  int8_t zp = 1;
  GetExpectedInt4Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<Int4x2>("zero_point", {}, {Int4x2(zp, unused_val)}, true);
  test.AddOutput<Int4x2>("y", dims, output);

  std::unordered_set<std::string> excluded_providers;
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

// Test uint4 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks of even size.
TEST(QuantizeLinearOpTest, OddLarge_UInt4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{1017};
  constexpr uint8_t unused_val = 0;
  constexpr std::array<float, 6> pattern = {-20.0f, -14.0f, -4.1f, -0.0f, 3.0f, 3.3f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<UInt4x2> output(UInt4x2::CalcNumInt4Pairs(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  uint8_t zp = 1;
  GetExpectedInt4Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<UInt4x2>("zero_point", {}, {UInt4x2(zp, unused_val)}, true);
  test.AddOutput<UInt4x2>("y", dims, output);

  std::unordered_set<std::string> excluded_providers;
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8_NegativeZeroPoint) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: 104 and 105";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {-23});
  test.AddOutput<int8_t>("y", dims, {-23, 28, 53, 104, 127, -74, -128, -128});
  std::unordered_set<std::string> excluded_providers;
  // Disable Tensorrt EP due to the error, node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

#ifdef USE_OPENVINO
TEST(QuantizeLinearOpTest, OVEP_Int8_NegativeZeroPoint) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {-23});
  test.AddOutput<int8_t>("y", dims, {-23, 28, 54, 105, 127, -74, -128, -128});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultOpenVINOExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_OPENVINO

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8_PositiveZeroPoint) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: -104 and -105";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {23});
  test.AddOutput<int8_t>("y", dims, {23, 74, 99, 127, 127, -28, -104, -128});
  std::unordered_set<std::string> excluded_providers;
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  excluded_providers.insert(kTensorrtExecutionProvider);
  // Disable OV EP due to different formulation for QuantizeLinear
  excluded_providers.insert(kOpenVINOExecutionProvider);
  test.ConfigExcludeEps(excluded_providers)
      .RunWithConfig();
}

#ifdef USE_OPENVINO
TEST(QuantizeLinearOpTest, OVEP_Int8_PositiveZeroPoint) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {23});
  test.AddOutput<int8_t>("y", dims, {23, 74, 100, 127, 127, -28, -104, -128});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultOpenVINOExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_OPENVINO

// quantize with 2D data
TEST(QuantizeLinearOpTest, 2D) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddInput<float>("scale", {}, {4});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 0, 1, 250,
                           0, 0, 1, 250,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, Scalar) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", {}, {130});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, QuantizeLinear_Without_Zero_Point_Opset10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_Without_Zero_Point_Opset13) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 13);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_With_Zero_Point0) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {0});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_With_Zero_Dim1) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {1}, {3});
  test.AddInput<float>("y_scale", {1}, {2.0f});
  test.AddOutput<uint8_t>("y", {1}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, Per_Channel_Axis_Default) {
  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 1, 1001,
                        1, 1, 2, 1100,
                        2, 4.2f, 3, 1200});
  test.AddInput<float>("scale", {4}, {1, 2, 3, 20});
  test.AddInput<uint8_t>("zero_point", {4}, {64, 100, 127, 127});
  test.AddOutput<uint8_t>("Y", dims,
                          {64, 101, 127, 177,
                           65, 100, 128, 182,
                           66, 102, 128, 187});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, Per_Channel_Axis_0) {
  OpTester test("QuantizeLinear", 13);
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

// quantize with per-channel and negative axis (-2 resolves to axis 0)
TEST(QuantizeLinearOpTest, Per_Channel_Axis_neg) {
  OpTester test("QuantizeLinear", 13);
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

template <typename F6>
void QuantizeDequantizeLinearOp28Float6Test() {
  const std::vector<int64_t> dims{6};
  const std::vector<float> input{-4.0f, -1.0f, -0.25f, 0.0f, 0.25f, 4.0f};
  constexpr float scale = 0.5f;
  const F6 zero_point(0.0f);
  std::vector<F6> quantized;
  std::vector<float> dequantized;
  quantized.reserve(input.size());
  dequantized.reserve(input.size());

  for (float value : input) {
    const F6 quantized_value(value / scale);
    quantized.push_back(quantized_value);
    dequantized.push_back(static_cast<float>(quantized_value) * scale);
  }

  OpTester quantize_test("QuantizeLinear", 28);
  quantize_test.AddAttribute<int64_t>("saturate", 0);
  quantize_test.AddInput<float>("x", dims, input);
  quantize_test.AddInput<float>("y_scale", {}, {scale});
  quantize_test.AddInput<F6>("y_zero_point", {}, {zero_point});
  quantize_test.AddOutput<F6>("y", dims, quantized);
  quantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                    {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

  OpTester dequantize_test("DequantizeLinear", 28);
  dequantize_test.AddInput<F6>("x", dims, quantized);
  dequantize_test.AddInput<float>("x_scale", {}, {scale});
  dequantize_test.AddInput<F6>("x_zero_point", {}, {zero_point});
  dequantize_test.AddOutput<float>("y", dims, dequantized);
  dequantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(QuantizeLinearOpTest, Float6E2M3) {
  QuantizeDequantizeLinearOp28Float6Test<Float6E2M3>();
}

TEST(QuantizeLinearOpTest, Float6E3M2) {
  QuantizeDequantizeLinearOp28Float6Test<Float6E3M2>();
}

template <typename F6>
void QuantizeDequantizeLinearOp28Float6ScalarZeroPointTest() {
  OpTester quantize_test("QuantizeLinear", 28);
  quantize_test.AddInput<float>("x", {1}, {1.0f});
  quantize_test.AddInput<float>("y_scale", {}, {0.5f});
  quantize_test.AddInput<F6>("y_zero_point", {}, {F6(1.0f)});
  quantize_test.AddOutput<F6>("y", {1}, {F6(3.0f)});
  quantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                    {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

  OpTester dequantize_test("DequantizeLinear", 28);
  dequantize_test.AddInput<F6>("x", {1}, {F6(1.0f)});
  dequantize_test.AddInput<float>("x_scale", {}, {0.5f});
  dequantize_test.AddInput<F6>("x_zero_point", {}, {F6(1.0f)});
  dequantize_test.AddOutput<float>("y", {1}, {0.0f});
  dequantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

template <typename F6, typename ScaleT>
void QuantizeDequantizeLinearOp28Float6ScaleTest(bool blocked) {
  const std::vector<int64_t> dims = blocked ? std::vector<int64_t>{2, 5} : std::vector<int64_t>{2, 3};
  const std::vector<int64_t> scale_dims = blocked ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{3};
  const std::vector<float> scale_values = blocked ? std::vector<float>{0.5f, 1.0f, 2.0f, 1.0f, 2.0f, 0.5f}
                                                  : std::vector<float>{0.5f, 1.0f, 2.0f};
  const std::vector<float> input_values = blocked ? std::vector<float>{0.5f, -1.0f, 1.0f, -2.0f, 2.0f,
                                                                       -1.0f, 2.0f, -2.0f, 4.0f, -1.0f}
                                                  : std::vector<float>{0.5f, -1.0f, 2.0f, -1.0f, 2.0f, -4.0f};
  const std::vector<float> expected_quantized = blocked ? std::vector<float>{1, -2, 1, -2, 1,
                                                                             -1, 2, -1, 2, -2}
                                                        : std::vector<float>{1, -1, 1, -2, 2, -2};
  std::vector<ScaleT> scales, input, expected_output;
  std::vector<F6> quantized;
  std::vector<F6> zero_points;
  for (float value : scale_values) {
    scales.emplace_back(value);
  }
  for (size_t i = 0; i < scales.size(); ++i) {
    zero_points.emplace_back(i % 2 == 0 ? 1.0f : -1.0f);
  }
  for (float value : input_values) {
    input.emplace_back(value);
  }
  for (size_t i = 0; i < expected_quantized.size(); ++i) {
    const size_t scale_index = blocked ? (i / 5) * 3 + (i % 5) / 2 : i % 3;
    quantized.emplace_back(expected_quantized[i] + static_cast<float>(zero_points[scale_index]));
    expected_output.emplace_back(expected_quantized[i] * scale_values[scale_index]);
  }

  OpTester quantize_test("QuantizeLinear", 28);
  quantize_test.AddAttribute<int64_t>("axis", blocked ? 1 : -1);
  if (blocked) {
    quantize_test.AddAttribute<int64_t>("block_size", 2);
  }
  quantize_test.AddInput<ScaleT>("x", dims, input);
  quantize_test.AddInput<ScaleT>("y_scale", scale_dims, scales);
  quantize_test.AddInput<F6>("y_zero_point", scale_dims, zero_points);
  quantize_test.AddOutput<F6>("y", dims, quantized);
  quantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                    {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

  OpTester dequantize_test("DequantizeLinear", 28);
  dequantize_test.AddAttribute<int64_t>("axis", blocked ? 1 : -1);
  if (blocked) {
    dequantize_test.AddAttribute<int64_t>("block_size", 2);
  }
  dequantize_test.AddInput<F6>("x", dims, quantized);
  dequantize_test.AddInput<ScaleT>("x_scale", scale_dims, scales);
  dequantize_test.AddInput<F6>("x_zero_point", scale_dims, zero_points);
  dequantize_test.AddOutput<ScaleT>("y", dims, expected_output);
  dequantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

template <typename F6>
void QuantizeDequantizeLinearOp28Float6NonzeroZeroPointTest() {
  const std::vector<int64_t> dims{2, 3};
  const std::vector<F6> zero_points{F6(0.0f), F6(1.0f), F6(-1.0f)};
  const std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<F6> quantized{F6(0.0f), F6(2.0f), F6(1.0f),
                                  F6(3.0f), F6(5.0f), F6(4.0f)};

  OpTester quantize_test("QuantizeLinear", 28);
  quantize_test.AddAttribute<int64_t>("axis", 1);
  quantize_test.AddInput<float>("x", dims, values);
  quantize_test.AddInput<float>("y_scale", {3}, {1.0f, 1.0f, 1.0f});
  quantize_test.AddInput<F6>("y_zero_point", {3}, zero_points);
  quantize_test.AddOutput<F6>("y", dims, quantized);
  quantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                    {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

  OpTester dequantize_test("DequantizeLinear", 28);
  dequantize_test.AddAttribute<int64_t>("axis", 1);
  dequantize_test.AddInput<F6>("x", dims, quantized);
  dequantize_test.AddInput<float>("x_scale", {3}, {1.0f, 1.0f, 1.0f});
  dequantize_test.AddInput<F6>("x_zero_point", {3}, zero_points);
  dequantize_test.AddOutput<float>("y", dims, values);
  dequantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

template <typename F6>
void QuantizeDequantizeLinearOp28Float6BlockedNonLastAxisTest() {
  const std::vector<int64_t> dims{3, 2};
  const std::vector<int64_t> scale_dims{2, 2};
  const std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<F6> zero_points{F6(1.0f), F6(-1.0f), F6(-1.0f), F6(1.0f)};
  const std::vector<F6> quantized{F6(1.0f), F6(0.0f), F6(3.0f),
                                  F6(2.0f), F6(3.0f), F6(6.0f)};

  OpTester quantize_test("QuantizeLinear", 28);
  quantize_test.AddAttribute<int64_t>("axis", 0);
  quantize_test.AddAttribute<int64_t>("block_size", 2);
  quantize_test.AddInput<float>("x", dims, values);
  quantize_test.AddInput<float>("y_scale", scale_dims, std::vector<float>(4, 1.0f));
  quantize_test.AddInput<F6>("y_zero_point", scale_dims, zero_points);
  quantize_test.AddOutput<F6>("y", dims, quantized);
  quantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                    {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

  OpTester dequantize_test("DequantizeLinear", 28);
  dequantize_test.AddAttribute<int64_t>("axis", 0);
  dequantize_test.AddAttribute<int64_t>("block_size", 2);
  dequantize_test.AddInput<F6>("x", dims, quantized);
  dequantize_test.AddInput<float>("x_scale", scale_dims, std::vector<float>(4, 1.0f));
  dequantize_test.AddInput<F6>("x_zero_point", scale_dims, zero_points);
  dequantize_test.AddOutput<float>("y", dims, values);
  dequantize_test.Run(OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(QuantizeLinearOpTest, Float6E2M3PerAxisFloat16AndBlocked) {
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E2M3, MLFloat16>(false);
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E2M3, float>(true);
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E2M3, MLFloat16>(true);
  QuantizeDequantizeLinearOp28Float6NonzeroZeroPointTest<Float6E2M3>();
  QuantizeDequantizeLinearOp28Float6ScalarZeroPointTest<Float6E2M3>();
  QuantizeDequantizeLinearOp28Float6BlockedNonLastAxisTest<Float6E2M3>();
}

TEST(QuantizeLinearOpTest, Float6E3M2PerAxisFloat16AndBlocked) {
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E3M2, MLFloat16>(false);
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E3M2, float>(true);
  QuantizeDequantizeLinearOp28Float6ScaleTest<Float6E3M2, MLFloat16>(true);
  QuantizeDequantizeLinearOp28Float6NonzeroZeroPointTest<Float6E3M2>();
  QuantizeDequantizeLinearOp28Float6ScalarZeroPointTest<Float6E3M2>();
  QuantizeDequantizeLinearOp28Float6BlockedNonLastAxisTest<Float6E3M2>();
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename InT, typename OutT>
void DequantizeLinearOp19Test() {
  OpTester test("DequantizeLinear", 19);
  std::vector<int64_t> dims{4};
  std::vector<InT> x;
  x.push_back(InT(0.0f, true));
  x.push_back(InT(1.0f, true));
  x.push_back(InT(2.0f, true));
  x.push_back(InT(3.0f, true));
  test.AddInput<InT>("x", dims, x);
  test.AddInput<OutT>("x_scale", {}, {static_cast<OutT>(1.0f)});
  test.AddInput<InT>("x_zero_point", {}, {InT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(static_cast<OutT>(it.ToFloat()));
  }
  test.AddOutput<OutT>("y", dims, y);
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DequantizeLinearOpTest, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E4M3FN, float>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E4M3FNUZ, float>();
  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E5M2, float>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E5M2FNUZ, float>();
}

TEST(DequantizeLinearOpMLFloat16Test, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E4M3FN, MLFloat16>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E4M3FNUZ, MLFloat16>();
  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E5M2, MLFloat16>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E5M2FNUZ, MLFloat16>();
}

template <typename InT, typename OutT>
void QuantizeLinearOp19Test(bool saturate) {
  OpTester test("QuantizeLinear", 19);
  if (!saturate) {
    test.AddAttribute<int64_t>("saturate", 0);
  }
  std::vector<int64_t> dims{6};
  std::vector<InT> x{0, 2, 3, 1000, -254, -1000};
  test.AddInput<InT>("x", dims, x);
  test.AddInput<InT>("y_scale", {}, {1.0f});
  test.AddInput<OutT>("y_zero_point", {}, {OutT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(OutT(it, saturate));
  }
  test.AddOutput<OutT>("y", dims, y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(QuantizeLinearOpTest, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E4M3FN>(true);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E4M3FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E5M2>(true);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E5M2FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E4M3FN>(false);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E4M3FNUZ>(false);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E5M2>(false);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E5M2FNUZ>(false);
}

template <typename InT, typename OutT>
void QuantizeLinearOp19F16Test(bool saturate) {
  OpTester test("QuantizeLinear", 19);
  if (!saturate) {
    test.AddAttribute<int64_t>("saturate", 0);
  }
  std::vector<int64_t> dims{6};
  std::vector<InT> x{InT(0.0f), InT(2.0f), InT(3.0f), InT(1000.0f), InT(-254.0f), InT(-1000.0f)};
  test.AddInput<InT>("x", dims, x);
  test.AddInput<InT>("y_scale", {}, {InT(1.0f)});
  test.AddInput<OutT>("y_zero_point", {}, {OutT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(OutT(it, saturate));
  }
  test.AddOutput<OutT>("y", dims, y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(QuantizeLinearOpMLFloat16Test, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FN>(true);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2>(true);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FN>(false);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FNUZ>(false);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2>(false);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2FNUZ>(false);
}

#endif

namespace blocked_dequantization {

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int(int64_t block_size,
                                                          int64_t scale_block_count,
                                                          int64_t zero_point_block_count) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep) {
    eps.push_back(std::move(webgpu_ep));
  }

  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; ++i) x_zero_point.push_back(0);
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) {
    x.push_back(static_cast<Tin>(i));
    y.push_back(Tout(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4(int64_t block_size,
                                                           int64_t scale_block_count,
                                                           int64_t zero_point_block_count,
                                                           std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  if (!ep) {
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (webgpu_ep) {
      eps.push_back(std::move(webgpu_ep));
    }
  }
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  using UnpackedType = typename Tin::UnpackedType;
  for (int64_t i = 0, n = zero_point_block_count; i < n; ++i) x_zero_point.push_back(Tin(0, 0));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) {
    if (i & 1) x.push_back(Tin(static_cast<UnpackedType>(i - 1), static_cast<UnpackedType>(i)));
    y.push_back(Tout(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8(int64_t block_size,
                                                             int64_t scale_block_count,
                                                             int64_t zero_point_block_count) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep) {
    eps.push_back(std::move(webgpu_ep));
  }
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; i++) x_zero_point.push_back(Tin(0.0f));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) x.push_back(Tin(static_cast<float>(i)));
  for (int i = 0; i < 8; ++i) y.push_back(Tout(static_cast<float>(i) * 2.0f));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

// test negative block size fail
TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(-3, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(-3, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(-4, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(-4, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(-5, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(-5, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(-6, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(-1, 2, 2);
}

TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(-1, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(-2, 2, 2);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(-3, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(-4, 2, 2);
  }
  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(-5, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(-6, 2, 2);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(-1, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(-1, 2, 2);
  }
}
#endif

// test block size incompatible with x_scale shape fail
TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 1, 1);
}

TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 3, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 3, 3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 3, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 3, 3);
  }
}
#endif

// test x_scale vs. x_zero_point shape incompatible fail
TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 2, 1);
}

TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 2, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 2, 3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 2, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 2, 3);
  }
}
#endif

// test DQ with blocked quantization succeed
template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Int4_Succeed(std::vector<int64_t>&& dims,
                                                  int64_t axis,
                                                  int64_t block_size,
                                                  std::vector<int>& x_,
                                                  std::vector<float>& x_scale_,
                                                  std::vector<int>& x_zero_point_,
                                                  std::vector<float>& y_,
                                                  std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  if (!ep) {
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (webgpu_ep) {
      eps.push_back(std::move(webgpu_ep));
    }
  }
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  using UnpackedType = typename Tin::UnpackedType;
  size_t i = 0, n = x_.size();
  for (; i < n - 1; i += 2) x.push_back(Tin(static_cast<UnpackedType>(x_[i]), static_cast<UnpackedType>(x_[i + 1])));
  if (i < n) x.push_back(Tin(static_cast<UnpackedType>(x_[i]), 0xF));

  if (use_zero_point) {
    i = 0, n = x_zero_point_.size();
    for (; i < n - 1; i += 2) x_zero_point.push_back(Tin(static_cast<UnpackedType>(x_zero_point_[i]),
                                                         static_cast<UnpackedType>(x_zero_point_[i + 1])));
    if (i < n) x_zero_point.push_back(Tin(static_cast<UnpackedType>(x_zero_point_[i]), 0xF));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Int_Succeed(std::vector<int64_t>&& dims,
                                                 int64_t axis,
                                                 int64_t block_size,
                                                 std::vector<int>& x_,
                                                 std::vector<float>& x_scale_,
                                                 std::vector<int>& x_zero_point_,
                                                 std::vector<float>& y_) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep) {
    eps.push_back(std::move(webgpu_ep));
  }
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }
  for (auto v : x_) x.push_back(static_cast<Tin>(v));
  if (use_zero_point)
    for (auto v : x_zero_point_) x_zero_point.push_back(static_cast<Tin>(v));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Float8_Succeed(std::vector<int64_t>&& dims,
                                                    int64_t axis,
                                                    int64_t block_size,
                                                    std::vector<int>& x_,
                                                    std::vector<float>& x_scale_,
                                                    std::vector<int>& x_zero_point_,
                                                    std::vector<float>& y_) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep) {
    eps.push_back(std::move(webgpu_ep));
  }
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  for (auto v : x_) x.push_back(Tin(static_cast<float>(v)));
  if (use_zero_point) {
    for (auto v : x_zero_point_) x_zero_point.push_back(Tin(static_cast<float>(v)));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_OneDimension) {
  std::vector<float> x_scale{0.5f, 2.0f};
  std::vector<int> zero_point{0, 10};
  std::vector<int> x{10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<float> y{5.0f, 10.0f, 15.0f, 20.0f, 80.0f, 100.0f, 120.0f, 140.0f};

  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({8}, 0, 4, x, x_scale, zero_point, y);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point;
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, 8.0};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point;
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, 8.0};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, 2.0, 0, 3.5, 3.0, 10.0, 16.0, -10.5, 15};
  std::vector<float> y_3{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, -14.0, -24, 21, 5, 10.0, 16.0, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, 2.0, 0, 3.5, 3.0, 10.0, 16.0, -10.5, 15};
  std::vector<float> y_3{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, -14.0, -24, 21, 5, 10.0, 16.0, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 8, -2, 0, 0, -1, 7, 1, 2, 0, 6, 8, -3.5, 1, -10.5, 15};
  std::vector<float> y_3{2, 8, -2, 0, -6, -8, 7, 1, 2, 0, 6, 8, 10, 16, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 8, -2, 0, 0, -1, 7, 1, 2, 0, 6, 8, -3.5, 1, -10.5, 15};
  std::vector<float> y_3{2, 8, -2, 0, -6, -8, 7, 1, 2, 0, 6, 8, 10, 16, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis_Cuda) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 0, 4, 0, 0, 3.5, 0, 1, 2, 4, 4, 8, -3.5, -7, 0, 15};
  std::vector<float> y_3{2, 0, -2, 0, 0, 3.5, 7, 1, 2, 4, 6, 8, -3.5, -7, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 0, 4, 0, 0, 3.5, 0, 1, 2, 4, 4, 8, -3.5, -7, 0, 15};
  std::vector<float> y_3{2, 0, -2, 0, 0, 3.5, 7, 1, 2, 4, 6, 8, -3.5, -7, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, 7, 3, -8, -20, 21, 7, 16, 36, -35, -11, 24, 52, -49, -15};
  std::vector<float> y_3{0, -4, 7, 3, -8, -20, 21, 7, -16, -36, 35, 11, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, 7, 3, -8, -20, 21, 7, 16, 36, -35, -11, 24, 52, -49, -15};
  std::vector<float> y_3{0, -4, 7, 3, -8, -20, 21, 7, -16, -36, 35, 11, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 3.5, -6, -4, -20, 17.5, -2, -10, 16, 3.5, -5, -2, 32, -10.5, -9};
  std::vector<float> y_3{4, -4, 3.5, -6, -4, -20, 17.5, -2, -12, -36, 31.5, 2, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 3.5, -6, -4, -20, 17.5, -2, -10, 16, 3.5, -5, -2, 32, -10.5, -9};
  std::vector<float> y_3{4, -4, 3.5, -6, -4, -20, 17.5, -2, -12, -36, 31.5, 2, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<float> y_3{0, -4, -4, -12, -8, -20, 21, 7, 16, 36, 20, 44, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<float> y_3{0, -4, -4, -12, -8, -20, 21, 7, 16, 36, 20, 44, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<float> y_3{4, -4, 0, -12, -4, -20, 17.5, -2, -10, 16, -6, 24, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<float> y_3{4, -4, 0, -12, -4, -20, 17.5, -2, -10, 16, -6, 24, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -2, -8, -12, 14, 17.5, 6, 7, 16, 18, 40, 44, -42, -45.5, -14, -15};
  std::vector<float> y_3{0, -2, -4, -12, 14, 17.5, 21, 7, 16, 18, 20, 44, -42, -45.5, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -2, -8, -12, 14, 17.5, 6, 7, 16, 18, 40, 44, -42, -45.5, -14, -15};
  std::vector<float> y_3{0, -2, -4, -12, 14, 17.5, 21, 7, 16, 18, 20, 44, -42, -45.5, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, 2, -8, -12, 10.5, 14, -3, -2, -10, -8, 20, 24, -3.5, -7, -8, -9};
  std::vector<float> y_3{4, 2, 0, -12, 10.5, 14, 17.5, -2, -10, -8, -6, 24, -3.5, -7, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, 2, -8, -12, 10.5, 14, -3, -2, -10, -8, 20, 24, -3.5, -7, -8, -9};
  std::vector<float> y_3{4, 2, 0, -12, 10.5, 14, 17.5, -2, -10, -8, -6, 24, -3.5, -7, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_FirstAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, -8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, -8.0};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  }
}

TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_MiddleAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, -8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, -8};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  }
}

TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_LastAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, -8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, -8};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  }
}
#endif
}  // namespace blocked_dequantization

namespace blocked_quantization {

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int(int64_t block_size,
                                                        int64_t scale_block_count,
                                                        int64_t zero_point_block_count) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> y, x_zero_point;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; ++i) x_zero_point.push_back(0);
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) {
    x.push_back(Tin(static_cast<float>(i) * 2.0f));
    y.push_back(static_cast<Tout>(i));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4(int64_t block_size,
                                                         int64_t scale_block_count,
                                                         int64_t zero_point_block_count,
                                                         std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_zero_point, y;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  using UnpackedType = typename Tout::UnpackedType;
  for (int64_t i = 0, n = zero_point_block_count; i < n; ++i) x_zero_point.push_back(Tout(0, 0));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) {
    if (i & 1) y.push_back(Tout(static_cast<UnpackedType>(i - 1), static_cast<UnpackedType>(i)));
    x.push_back(Tin(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8(int64_t block_size,
                                                           int64_t scale_block_count,
                                                           int64_t zero_point_block_count) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_zero_point, y;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; i++) x_zero_point.push_back(Tout(0.0f));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) y.push_back(Tout(static_cast<float>(i)));
  for (int i = 0; i < 8; ++i) x.push_back(Tin(static_cast<float>(i) * 2.0f));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

// test negative block size fail
TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(-3, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(-3, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(-4, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(-4, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(-5, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(-5, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(-6, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(-1, 2, 2);
}

TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(-1, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(-2, 2, 2);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(-3, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(-4, 2, 2);
  }
  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(-5, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(-6, 2, 2);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(-1, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(-1, 2, 2);
  }
}
#endif

// test block size incompatible with x_scale shape fail
TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 1, 1);
}

TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 3, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 3, 3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 3, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 3, 3);
  }
}
#endif

// test x_scale vs. x_zero_point shape incompatible fail
TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 2, 1);
}

TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 2, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 2, 3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 2, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 2, 3);
  }
}
#endif

// test DQ with blocked quantization succeed
template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Int4_Succeed(std::vector<int64_t>&& dims,
                                                int64_t axis,
                                                int64_t block_size,
                                                std::vector<float>& x_,
                                                std::vector<float>& scale_,
                                                std::vector<int>& zero_point_,
                                                std::vector<int>& y_,
                                                std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  using UnpackedType = typename Tout::UnpackedType;
  size_t i = 0, n = y_.size();
  for (; i < n - 1; i += 2) y.push_back(Tout(static_cast<UnpackedType>(y_[i]),
                                             static_cast<UnpackedType>(y_[i + 1])));
  if (i < n) y.push_back(Tout(static_cast<UnpackedType>(y_[i]), 0xF));

  if (use_zero_point) {
    i = 0, n = zero_point_.size();
    for (; i < n - 1; i += 2) zero_point.push_back(Tout(static_cast<UnpackedType>(zero_point_[i]),
                                                        static_cast<UnpackedType>(zero_point_[i + 1])));
    if (i < n) zero_point.push_back(Tout(static_cast<UnpackedType>(zero_point_[i]), 0xF));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("y_scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("y_zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Int_Succeed(std::vector<int64_t>&& dims,
                                               int64_t axis,
                                               int64_t block_size,
                                               std::vector<float>& x_,
                                               std::vector<float>& scale_,
                                               std::vector<int>& zero_point_,
                                               std::vector<int>& y_) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }
  for (auto v : y_) y.push_back(static_cast<Tout>(v));
  if (use_zero_point)
    for (auto v : zero_point_) zero_point.push_back(static_cast<Tout>(v));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Float8_Succeed(std::vector<int64_t>&& dims,
                                                  int64_t axis,
                                                  int64_t block_size,
                                                  std::vector<float>& x_,
                                                  std::vector<float>& scale_,
                                                  std::vector<float>& zero_point_,
                                                  std::vector<float>& y_) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  for (auto v : y_) y.push_back(Tout(static_cast<float>(v)));
  if (use_zero_point) {
    for (auto v : zero_point_) zero_point.push_back(Tout(static_cast<float>(v)));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -8, -2, -8, -2, -8, -2, -8, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -8, -2, -8, -2, -8, -2, -8, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -9, -2, -9, -2, -9, -2, -9, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_LastAxis_Cuda) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -10, -4, -5, -4, -7, -6, -7, -6, -10, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 20, 11, 10, 11, 8, 9, 8, 9, 20, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 23, 11, 10, 11, 8, 9, 8, 9, 23, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_FirstAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                         -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                         1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                         5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                         -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                         -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                         5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  }
}

TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_MiddleAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_2{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  }
}

TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_LastAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_2{-7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
  }
}
#endif
}  // namespace blocked_quantization
}  // namespace test
}  // namespace onnxruntime
