// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "boost/mp11.hpp"

#include <gsl/gsl>

#include "gtest/gtest.h"

#include "core/framework/data_types_internal.h"
#include "core/framework/float6.h"
#include "core/framework/int2.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/utils.h"

#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

template <typename T>
int GetMinRequiredCudaComputeCapability() {
  return 0;
}

template <>
int GetMinRequiredCudaComputeCapability<MLFloat16>() {
  return 530;
}

template <>
int GetMinRequiredCudaComputeCapability<BFloat16>() {
  return 800;
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
int GetMinRequiredCudaComputeCapability<Float8E4M3FN>() {
  return 800;
}

template <>
int GetMinRequiredCudaComputeCapability<Float8E5M2>() {
  return 800;
}

#endif

enum Saturate { True,
                False,
                None };

template <typename SrcType,
          typename DstType>
void TestCastOp(gsl::span<const SrcType> input,
                gsl::span<const DstType> output,
                const BaseTester::DimsVariant& dimensions,
                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                const std::string& expected_failure_string = "",
                int opset = 21,
                Saturate saturate = Saturate::None,
                bool cuda_only = false,
                const std::unordered_set<std::string>& additional_excluded_providers = {}) {
  OpTester test("Cast", opset);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
  test.AddInput<SrcType>("input", dimensions, input.data(), input.size());
  test.AddOutput<DstType>("output", dimensions, output.data(), output.size());
  if (saturate != Saturate::None) {
    test.AddAttribute<int64_t>("saturate", saturate == Saturate::True ? 1 : 0);
  }

  std::unordered_set<std::string> excluded_provider_types{kTensorrtExecutionProvider};
  const auto min_required_cuda_compute_capability =
      std::max(GetMinRequiredCudaComputeCapability<SrcType>(), GetMinRequiredCudaComputeCapability<DstType>());
  if (!HasCudaEnvironment(min_required_cuda_compute_capability)) {
    excluded_provider_types.insert(kCudaExecutionProvider);
  }

  if (input.size() == 0) {
    // The OpenVINO doesn't support 0 size input
    excluded_provider_types.insert(kOpenVINOExecutionProvider);
  }

  // Add any additional excluded providers
  excluded_provider_types.insert(additional_excluded_providers.begin(), additional_excluded_providers.end());

  if (cuda_only && (excluded_provider_types.count(kCudaExecutionProvider) > 0)) {
    return;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (cuda_only) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(expect_result, expected_failure_string, {}, nullptr, &execution_providers);
    return;
  }

  test.Run(expect_result, expected_failure_string, excluded_provider_types);
}

// INT2 types were introduced in opset 25 (IR13)
constexpr int kInt2Opset = 25;

// Helper for INT2 cast tests that uses opset 25 by default
template <typename SrcType, typename DstType>
void TestCastOpInt2(gsl::span<const SrcType> input,
                    gsl::span<const DstType> output,
                    const BaseTester::DimsVariant& dimensions,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& expected_failure_string = "",
                    Saturate saturate = Saturate::None) {
  TestCastOp(input, output, dimensions, expect_result, expected_failure_string, kInt2Opset, saturate);
}

template <typename T>
using RequiresCastThroughFloat =
    boost::mp11::mp_any<
        std::is_same<T, MLFloat16>,
        std::is_same<T, BFloat16>>;

template <typename... T>
using AnyRequireCastThroughFloat = boost::mp11::mp_any<RequiresCastThroughFloat<T>...>;

template <typename SrcType, typename DstType>
typename std::enable_if<AnyRequireCastThroughFloat<SrcType, DstType>::value>::type
CastSpan(gsl::span<const SrcType> src, gsl::span<DstType> dst) {
  std::transform(
      src.begin(), src.end(), dst.begin(),
      [](SrcType s) {
        return static_cast<DstType>(static_cast<float>(s));
      });
}

template <typename SrcType, typename DstType>
typename std::enable_if<!AnyRequireCastThroughFloat<SrcType, DstType>::value>::type
CastSpan(gsl::span<const SrcType> src, gsl::span<DstType> dst) {
  std::transform(
      src.begin(), src.end(), dst.begin(),
      [](SrcType s) {
        return static_cast<DstType>(s);
      });
}

template <typename SrcType, typename DstType>
std::vector<DstType> CastedValues(gsl::span<const SrcType> src) {
  std::vector<DstType> result(src.size());
  CastSpan<SrcType, DstType>(src, gsl::make_span(result));
  return result;
}

struct CastNonStringTester {
  int opset{21};
  bool cuda_only{false};

  template <typename SrcType, typename DstType>
  void operator()(const std::pair<SrcType, DstType>&) {
    SCOPED_TRACE(
        onnxruntime::MakeString(
            "Cast from type ", utils::ToTensorProtoElementType<SrcType>(),
            " to type ", utils::ToTensorProtoElementType<DstType>()));

    const std::vector<int> input_int_values{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const TensorShape shape{2, 3, 2, 2};
    const size_t size = gsl::narrow<size_t>(shape.Size());
    ASSERT_EQ(input_int_values.size(), size);

    auto input_buffer = std::make_unique<SrcType[]>(size);
    auto input_span = gsl::make_span<SrcType>(input_buffer.get(), size);
    CastSpan<int, SrcType>(gsl::make_span(input_int_values), input_span);

    auto output_buffer = std::make_unique<DstType[]>(size);
    auto output_span = gsl::make_span<DstType>(output_buffer.get(), size);
    CastSpan<SrcType, DstType>(input_span, output_span);

    TestCastOp<SrcType, DstType>(input_span, output_span, shape.AsShapeVector(),
                                 OpTester::ExpectResult::kExpectSuccess, "", opset, Saturate::None, cuda_only);
  }
};

using CastNonStringTypes =
    boost::mp11::mp_list<
        bool,
        float, double,
        uint8_t, uint16_t, uint32_t, uint64_t,
        int8_t, int16_t, int32_t, int64_t,
        MLFloat16, BFloat16>;

TEST(CastOpTest, NonStringTypes) {
  boost::mp11::mp_for_each<boost::mp11::mp_product<std::pair, CastNonStringTypes, CastNonStringTypes>>(
      CastNonStringTester{});
}

TEST(CastOpTest, NonStringTypes_Opset25To28) {
  for (int opset : {25, 27, 28}) {
    SCOPED_TRACE(opset);
    boost::mp11::mp_for_each<boost::mp11::mp_product<std::pair, CastNonStringTypes, CastNonStringTypes>>(
        CastNonStringTester{opset});
  }
}

#if defined(USE_CUDA)
TEST(CastOpTest, NonStringTypes_Opset28_Cuda) {
  if (!HasCudaEnvironment(530)) {
    GTEST_SKIP() << "CUDA Cast tests require a CUDA device with compute capability >= 5.3.";
  }

  boost::mp11::mp_for_each<boost::mp11::mp_product<std::pair, CastNonStringTypes, CastNonStringTypes>>(
      CastNonStringTester{28, true});
}
#endif

TEST(CastOpTest, FromString) {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_data = {"-inf", "+INF", "0.9767611", "0.28280696",
                                                "-0.12019656", "5.0", "NaN", "nan"};
  const std::vector<float> float_output = {-(std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity(),
                                           0.9767611f, 0.28280696f,
                                           -0.12019656f, 5.0f, NAN, NAN};
  TestCastOp(gsl::make_span(string_data), gsl::make_span(float_output), shape);

  const std::vector<std::string> float16_string_data = {"-inf", "+INF", "0.5", "0.25",
                                                        "0.0", "-1.0", "-1.5", "NaN"};
  const std::vector<MLFloat16> float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0.5f, 0.25f,
                  0.0f, -1.0f, -1.5f, NAN}));
  TestCastOp(gsl::make_span(float16_string_data), gsl::make_span(float16_output), shape);

  const std::vector<std::string> int_16_string_data = {"0", "1", "2", "3", "4", "5", "-32768", "32767"};
  const std::vector<int16_t> int_16_output = {0, 1, 2, 3, 4, 5, SHRT_MIN, SHRT_MAX};
  TestCastOp(gsl::make_span(int_16_string_data), gsl::make_span(int_16_output), shape);

  const std::vector<std::string> int_64_string_data = {"0", "1", "2", "3", "4", "5", "-9223372036854775808", "9223372036854775807"};
  const std::vector<int64_t> int_64_output = {0, 1, 2, 3, 4, 5, LLONG_MIN, LLONG_MAX};
  TestCastOp(gsl::make_span(int_64_string_data), gsl::make_span(int_64_output), shape);
}

TEST(CastOpTest, ToString) {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f, -0.120196559f, 5.0f,
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()};

  // float output precision is 8, so the expected output differs slightly from the input due to that
  const std::vector<std::string> string_output = {"NaN", "-1", "0.039187793", "0.29614019",
                                                  "-0.12019656", "5", "-INF", "INF"};
  TestCastOp(gsl::make_span(float_input), gsl::make_span(string_output), shape);

  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0.5f, 0.25f,
                  0.0f, -1.0f, -1.5f, NAN}));
  const std::vector<std::string> float16_string_output = {"-INF", "INF", "0.5", "0.25",
                                                          "0", "-1", "-1.5", "NaN"};
  TestCastOp(gsl::make_span(float16_input), gsl::make_span(float16_string_output), shape);

  const std::vector<std::string> int_string_data = {"0", "1", "2", "3", "4", "5", "6", "7"};
  const std::vector<int16_t> int_16_input = {0, 1, 2, 3, 4, 5, 6, 7};
  TestCastOp(gsl::make_span(int_16_input), gsl::make_span(int_string_data), shape);
}

TEST(CastOpTest, Int4x2ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int8_t> expected_int8_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint8_t> expected_uint8_output = {248, 7, 0, UINT8_MAX, 3, 251, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, Int4x2ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int16_t> expected_int16_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint16_t> expected_uint16_output = {65528, 7, 0, UINT16_MAX, 3, 65531, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, Int4x2ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int32_t> expected_int32_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint32_t> expected_uint32_output = {4294967288, 7, 0, UINT32_MAX, 3, 4294967291, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, Int4x2ToInt32OddNumberOfElements) {
  // GIVEN
  const std::vector<int64_t> odd_shape{5};
  const std::vector<Int4x2> odd_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, 0),
  };

  const std::vector<int32_t> expected_odd_output = {-8, 7, 0, -1, 3};

  // WHEN, THEN
  TestCastOp(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, Int4x2ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int64_t> expected_int64_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint64_t> expected_uint64_output = {18446744073709551608ULL, 7, 0, UINT64_MAX, 3, 18446744073709551611ULL, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint8_t> expected_uint8_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int8_t> expected_int8_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint16_t> expected_uint16_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int16_t> expected_int16_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint32_t> expected_uint32_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int32_t> expected_int32_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint64_t> expected_uint64_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int64_t> expected_int64_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, Int4x2ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(0, -1),  // zero and non-zero
      Int4x2(7, 0),   // non-zero and zero
      Int4x2(-8, 3),  // both non-zero
      Int4x2(0, 0)    // both zero
  };

  const bool bool_output[] = {false, true, true, false, true, true, false, false};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, UInt4x2ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 1),   // zero and non-zero
      UInt4x2(15, 0),  // non-zero and zero
      UInt4x2(8, 7),   // both non-zero
      UInt4x2(0, 0)    // both zero
  };

  const bool bool_output[] = {false, true, true, false, true, true, false, false};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, Int4x2ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(1, 2),  // two 4-bit int elements: lower = 1, upper = 2
      Int4x2(-3, -4),
      Int4x2(5, -6),
      Int4x2(-8, 7)};

  const std::vector<float> expected_float_output = {1.0f, 2.0f, -3.0f, -4.0f, 5.0f, -6.0f, -8.0f, 7.0f};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, UInt4x2ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 1),
      UInt4x2(2, 3),
      UInt4x2(7, 8),
      UInt4x2(14, 15)};

  const std::vector<float> expected_float_output = {0.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 14.0f, 15.0f};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, Int4x2ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -3),  // zero and negative
      Int4x2(4, -2),  // positive and negative
      Int4x2(1, 6)    // both positive
  };

  const std::vector<double> expected_double_output = {-8.0, 7.0, 0.0, -3.0, 4.0, -2.0, 1.0, 6.0};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, UInt4x2ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<double> expected_double_output = {0.0, 15.0, 1.0, 14.0, 7.0, 8.0, 3.0, 12.0};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, Int4x2ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, UInt4x2ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, Int4x2ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, UInt4x2ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, Int4x2ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // mixed values
      Int4x2(6, 2)    // positive values
  };

  // Each Int4x2 becomes two string values
  const std::vector<std::string> expected_output = {
      "-8", "7",  // from first Int4x2
      "0", "-1",  // from second Int4x2
      "3", "-5",  // from third Int4x2
      "6", "2"    // from fourth Int4x2
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const Int4x2>(int4x2_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, UInt4x2ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(8, 7),   // mid-range values
      UInt4x2(3, 12),  // mixed values
      UInt4x2(10, 5)   // other values
  };

  // Each UInt4x2 becomes two string values
  const std::vector<std::string> expected_output = {
      "0", "15",  // from first UInt4x2
      "8", "7",   // from second UInt4x2
      "3", "12",  // from third UInt4x2
      "10", "5"   // from fourth UInt4x2
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const UInt4x2>(uint4x2_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // min and max values
      Int4x2(0, -1),  // -1 becomes max unsigned value
      Int4x2(3, -5),  // positive and negative values
      Int4x2(6, 2)    // positive values
  };

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(8, 7),   // -8 becomes 8
      UInt4x2(0, 15),  // -1 becomes 15
      UInt4x2(3, 11),  // -5 becomes 11
      UInt4x2(6, 2)    // unchanged
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // 15 is out of int4 range
      UInt4x2(1, 14),  // 14 is out of int4 range
      UInt4x2(7, 8),   // 8 is out of int4 range
      UInt4x2(3, 6)    // both within range
  };

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(0, -1),  // 15 becomes -1
      Int4x2(1, -2),  // 14 becomes -2
      Int4x2(7, -8),  // 8 becomes -8
      Int4x2(3, 6)    // unchanged
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int8ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int8_t> int8_input = {-10, 15, 0, -1, 7, -8, -128, 127};

  const std::vector<Int4x2> expected_int4x2_output = {
      // 10 in binary is 00001010.
      // Invert all bits -> 11110101, add 1 -> 11110110
      // So -10 in binary is 11110110.
      // Truncate to 4 least significant bits -> 0110.
      // In 4-bit two's complement, 0110 = 0 * -8 + 1 * 4 + 1 * 2 = 6.
      Int4x2(6, -1),  // -10 truncated to 6, 15 truncated to -1
      Int4x2(0, -1),  // 0 unchanged, -1 unchanged
      Int4x2(7, -8),  // 7 unchanged, -8 unchanged
      Int4x2(0, -1)   // -128 truncated to 0, 127 truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt8ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint8_t> uint8_input = {20, 255, 0, 17, 7, 240, 15, 31};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 (0x14) truncated to 4, 255 (0xFF) truncated to 15
      UInt4x2(0, 1),   // 0 (0x00) truncated to 0, 17 (0x11) truncated to 1
      UInt4x2(7, 0),   // 7 (0x07) truncated to 7, 240 (0xF0) truncated to 0
      UInt4x2(15, 15)  // 15 (0x0F) truncated to 15, 31 (0x1F) truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint8_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int16_t> int16_input = {-10, 32767, 0, -32768, 7, -8, 240, 31};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 (0xFFF6) truncated to 6, 32767 (0x7FFF) truncated to -1
      Int4x2(0, 0),   // 0 (0x0000) truncated to 0, -32768 (0x8000) truncated to 0
      Int4x2(7, -8),  // 7 (0x0007) truncated to 7, -8 (0xFFF8) truncated to -8
      Int4x2(0, -1)   // 240 (0x00F0) truncated to 0, 31 (0x001F) truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int16_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt16ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint16_t> uint16_input = {20, 65535, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 (0x0014) truncated to 4, 65535 (0xFFFF) truncated to 15
      UInt4x2(0, 0),   // 0 (0x0000) truncated to 0, 256 (0x0100) truncated to 0
      UInt4x2(7, 0),   // 7 (0x0007) truncated to 7, 240 (0x00F0) truncated to 0
      UInt4x2(15, 15)  // 15 (0x000F) truncated to 15, 4095 (0x0FFF) truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint16_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int32ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int32_t> int32_input = {-10, INT32_MAX, 0, INT32_MIN, 3, -5, 4080, 287};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 (0xFFFFFFF6) truncated to 6, 2147483647 (0x7FFFFFFF) truncated to -1
      Int4x2(0, 0),   // 0 (0x00000000) truncated to 0, -2147483648 (0x80000000) truncated to 0
      Int4x2(3, -5),  // 3 (0x00000003) truncated to 3, -5 (0xFFFFFFFB) truncated to -5
      Int4x2(0, -1)   // 4080 (0x00000FF0) truncated to 0, 287 (0x0000011F) truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int32_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int32ToInt4x2OddNumberOfElements) {
  // GIVEN
  const std::vector<int64_t> odd_shape{5};
  const std::vector<int32_t> odd_input = {-10, INT32_MAX, 0, INT32_MIN, 4095};

  const std::vector<Int4x2> expected_odd_output = {
      Int4x2(6, -1),  // -10 truncated to 6, 2147483647 truncated to -1
      Int4x2(0, 0),   // 0 truncated to 0, -2147483648 truncated to 0
      Int4x2(-1, 0)   // 4095 truncated to -1, paired with 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, Int32ToInt4x2EmptyTensor) {
  // GIVEN
  const std::vector<int64_t> empty_shape{0};
  const std::vector<int32_t> empty_input = {};
  const std::vector<Int4x2> empty_output = {};

  // WHEN, THEN
  TestCastOp(gsl::make_span(empty_input), gsl::make_span(empty_output), empty_shape);
}

TEST(CastOpTest, UInt32ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint32_t> uint32_input = {20, UINT32_MAX, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 truncated to 4, 4294967295 truncated to 15
      UInt4x2(0, 0),   // 0 truncated to 0, 256 truncated to 0
      UInt4x2(7, 0),   // 7 truncated to 7, 240 truncated to 0
      UInt4x2(15, 15)  // 15 truncated to 15, 4095 truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint32_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int64ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int64_t> int64_input = {-10, INT64_MAX, 0, INT64_MIN, 7, -8, 65520, 4111};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 truncated to 6, 9223372036854775807 truncated to -1
      Int4x2(0, 0),   // 0 truncated to 0, -9223372036854775808 truncated to 0
      Int4x2(7, -8),  // 7 truncated to 7, -8 truncated to -8
      Int4x2(0, -1)   // 65520 truncated to 0, 4111 truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int64_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt64ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint64_t> uint64_input = {20, UINT64_MAX, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 truncated to 4, 18446744073709551615 truncated to 15
      UInt4x2(0, 0),   // 0 truncated to 0, 256 truncated to 0
      UInt4x2(7, 0),   // 7 truncated to 7, 240 truncated to 0
      UInt4x2(15, 15)  // 15 truncated to 15, 4095 truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint64_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, FloatToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {-10.7f, 15.3f, 0.4f, -1.6f, 7.0f, -8.0f, 240.1f, 31.9f};

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5, sign-extended to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(7, -8),  // 7.0 converted to 7; -8.0 converted to -8
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(float_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, DoubleToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<double> double_input = {20.7, 255.3, 0.4, 1.6, 7.8, 240.2, 15.1, 31.9};

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(double_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, MLFloat16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const MLFloat16 mlfloat16_array[8] = {
      MLFloat16(static_cast<float>(-10.7f)),
      MLFloat16(static_cast<float>(15.3f)),
      MLFloat16(static_cast<float>(0.4f)),
      MLFloat16(static_cast<float>(-1.6f)),
      MLFloat16(static_cast<float>(3.8f)),
      MLFloat16(static_cast<float>(-5.2f)),
      MLFloat16(static_cast<float>(240.1f)),
      MLFloat16(static_cast<float>(31.9f))};

  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(4, -5),  // 3.8 rounded to 4; -5.2 rounded to -5 (0xFB), truncated to 11 (0x0B), sign-extended to -5
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 8),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToUInt4x2) {
  // GIVEN
  // 8 MLFloat16 values will compress to 4 UInt4x2 values
  const std::vector<int64_t> shape{2, 4};  // Shape that contains 8 elements

  // MLFloat16 values with edge cases and truncation scenarios
  const MLFloat16 mlfloat16_array[8] = {
      MLFloat16(static_cast<float>(20.7f)),
      MLFloat16(static_cast<float>(255.3f)),
      MLFloat16(static_cast<float>(0.4f)),
      MLFloat16(static_cast<float>(1.6f)),
      MLFloat16(static_cast<float>(7.8f)),
      MLFloat16(static_cast<float>(240.2f)),
      MLFloat16(static_cast<float>(15.1f)),
      MLFloat16(static_cast<float>(31.9f))};

  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 8),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToInt4x2BoundaryValues) {
  // GIVEN
  // Test MLFloat16 values that need truncation to Int4x2 range
  const std::vector<int64_t> shape{3, 2};
  const MLFloat16 mlfloat16_array[6] = {
      MLFloat16(static_cast<float>(-10)),    // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(9)),      // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(-8)),     // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(7)),      // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(-0.6f)),  // Should round to -1
      MLFloat16(static_cast<float>(1.7f))    // Should round to 2
  };

  // Values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(6, -7),  // -10 (0xFFFFFFF6) truncated to 6, 9 (0x00000009) truncated to -7
      Int4x2(-8, 7),  // -8 (0xFFFFFFF8) truncated to -8, 7 (0x00000007) truncated to 7
      Int4x2(-1, 2)   // -0.6 rounds to -1, 1.7 rounds to 2
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 6),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToUInt4x2BoundaryValues) {
  // GIVEN
  // Test MLFloat16 values that need truncation to UInt4x2 range
  const std::vector<int64_t> shape{3, 2};  // Shape that contains 6 elements
  const MLFloat16 mlfloat16_array[6] = {
      MLFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 4 bits
      MLFloat16(static_cast<float>(20)),    // Above max, truncated to lower 4 bits
      MLFloat16(static_cast<float>(0)),     // At min, should remain 0
      MLFloat16(static_cast<float>(15)),    // At max, should remain 15
      MLFloat16(static_cast<float>(3.4f)),  // Should round to 3
      MLFloat16(static_cast<float>(5.7f))   // Should round to 6
  };

  // Values get truncated to lower 4 bits (no sign extension for unsigned)
  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 15),  // 0 and 15 already within range
      UInt4x2(3, 6)    // 3.4 rounds to 3, 5.7 rounds to 6
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 6),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const BFloat16 bfloat16_array[8] = {
      BFloat16(static_cast<float>(-10.7f)),
      BFloat16(static_cast<float>(15.3f)),
      BFloat16(static_cast<float>(0.4f)),
      BFloat16(static_cast<float>(-1.6f)),
      BFloat16(static_cast<float>(3.8f)),
      BFloat16(static_cast<float>(-5.2f)),
      BFloat16(static_cast<float>(240.1f)),
      BFloat16(static_cast<float>(31.9f))};

  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(4, -5),  // 3.8 rounded to 4; -5.2 rounded to -5 (0xFB), truncated to 11 (0x0B), sign-extended to -5
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 8),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const BFloat16 bfloat16_array[8] = {
      BFloat16(static_cast<float>(20.7f)),
      BFloat16(static_cast<float>(255.3f)),
      BFloat16(static_cast<float>(0.4f)),
      BFloat16(static_cast<float>(1.6f)),
      BFloat16(static_cast<float>(7.8f)),
      BFloat16(static_cast<float>(240.2f)),
      BFloat16(static_cast<float>(15.1f)),
      BFloat16(static_cast<float>(31.9f))};

  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 8),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToUInt4x2BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{3, 2};
  const BFloat16 bfloat16_array[6] = {
      BFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 4 bits
      BFloat16(static_cast<float>(20)),    // Above max, truncated to lower 4 bits
      BFloat16(static_cast<float>(0)),     // At min, should remain 0
      BFloat16(static_cast<float>(15)),    // At max, should remain 15
      BFloat16(static_cast<float>(3.4f)),  // Should round to 3
      BFloat16(static_cast<float>(5.7f))   // Should round to 6
  };

  // Values get truncated to lower 4 bits (no clamping for consistency)
  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 15),  // 0 and 15 already within range
      UInt4x2(3, 6)    // 3.4 rounds to 3, 5.7 rounds to 6
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 6),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BoolToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(0, 1),
      Int4x2(1, 0),
      Int4x2(0, 1),
      Int4x2(1, 1)};

  // WHEN, THEN
  TestCastOp(bool_input_span, gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, BoolToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(0, 1),
      UInt4x2(1, 0),
      UInt4x2(0, 1),
      UInt4x2(1, 1)};

  // WHEN, THEN
  TestCastOp(bool_input_span, gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, StringToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-8", "7",  // boundary values
      "0", "-1",  // zero and negative
      "3", "-5",  // mixed values
      "6", "2"    // positive values
  };

  const std::vector<Int4x2> expected_output{
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const Int4x2>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "0", "15",  // boundary values
      "8", "7",   // mid-range values
      "3", "12",  // mixed values
      "10", "5"   // other values
  };

  const std::vector<UInt4x2> expected_output{
      UInt4x2(0, 15),
      UInt4x2(8, 7),
      UInt4x2(3, 12),
      UInt4x2(10, 5)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const UInt4x2>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt4x2BoundaryValues) {
  // GIVEN
  // Test string values that need truncation to UInt4x2 range
  const std::vector<int64_t> shape{3, 2};
  const std::vector<std::string> string_input = {
      "-5", "20",   // out of range values that get truncated
      "16", "100",  // out of range values that get truncated
      "0", "15"     // boundary values that are in range
  };

  // Each pair of strings becomes one UInt4x2
  // Values get truncated to lower 4 bits (no sign extension for unsigned)
  const std::vector<UInt4x2> expected_output{
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 4),   // 16 (0x00000010) truncated to 0, 100 (0x00000064) truncated to 4
      UInt4x2(0, 15)   // 0 and 15 already in range
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const UInt4x2>(expected_output), shape);
}

TEST(CastOpTest, FloatStringToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-10.7", "255.3",
      "0.4", "2",
      "6.8", "240.2",
      "15.0", "-8"};

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(5, -1),  // -11 -> 5, 255 -> -1
      Int4x2(0, 2),
      Int4x2(7, 0),
      Int4x2(-1, -8)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const Int4x2>(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int2x4ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),  // boundary and zero values
      Int2x4(1, -2, -1, 0)   // mixed values
  };

  const std::vector<int8_t> expected_int8_output = {-2, 1, 0, -1, 1, -2, -1, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Negative values will be cast to their unsigned representation
  const std::vector<uint8_t> expected_uint8_output = {254, 1, 0, 255, 1, 254, 255, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, Int2x4ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<int16_t> expected_int16_output = {-2, 1, 0, -1, 1, -2, -1, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, Int2x4ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<int32_t> expected_int32_output = {-2, 1, 0, -1, 1, -2, -1, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, Int2x4ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<int64_t> expected_int64_output = {-2, 1, 0, -1, 1, -2, -1, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, Int2x4ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<float> expected_float_output = {-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, Int2x4ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<double> expected_double_output = {-2.0, 1.0, 0.0, -1.0, 1.0, -2.0, -1.0, 0.0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, Int2x4ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(0, -1, 1, 0),
      Int2x4(-2, 0, 1, -1)};

  const bool bool_output[] = {false, true, true, false, true, false, true, true};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, Int2x4ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f}));

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, Int2x4ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<std::string> expected_output = {
      "-2", "1", "0", "-1",
      "1", "-2", "-1", "0"};

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const Int2x4>(int2x4_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, Int2x4ToInt32OddNumberOfElements) {
  // GIVEN - Test with 5 elements (not a multiple of 4)
  const std::vector<int64_t> odd_shape{5};
  const std::vector<Int2x4> odd_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, 0, 0, 0),  // last 3 values are padding
  };

  const std::vector<int32_t> expected_odd_output = {-2, 1, 0, -1, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, UInt2x4ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),  // boundary and mid values
      UInt2x4(3, 0, 2, 1)   // reversed order
  };

  const std::vector<uint8_t> expected_uint8_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<int8_t> expected_int8_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<int32_t> expected_int32_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, UInt2x4ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<float> expected_float_output = {0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, UInt2x4ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 1, 2, 0),
      UInt2x4(3, 0, 0, 1)};

  const bool bool_output[] = {false, true, true, false, true, false, false, true};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, UInt2x4ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<std::string> expected_output = {
      "0", "3", "1", "2",
      "3", "0", "2", "1"};

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const UInt2x4>(uint2x4_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Reinterpret: -2 becomes 2, -1 becomes 3 (mask to 2 bits)
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(2, 1, 0, 3),
      UInt2x4(1, 2, 3, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  // Sign-extend: 2 becomes -2, 3 becomes -1 (values >= 2 are negative in 2-bit signed)
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(0, -1, 1, -2),
      Int2x4(-1, 0, -2, 1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, Int4x2ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // Truncate to 2 bits and sign-extend: -8 -> 0, 7 -> -1, 0 -> 0, -1 -> -1, 3 -> -1, -5 -> -1, 6 -> -2, 2 -> -2
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(0, -1, 0, -1),
      Int2x4(-1, -1, -2, -2)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int4x2_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // Truncate to 2 bits: -8 -> 0, 7 -> 3, 0 -> 0, -1 -> 3, 3 -> 3, -5 -> 3, 6 -> 2, 2 -> 2
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 0, 3),
      UInt2x4(3, 3, 2, 2)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int4x2_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  // Truncate to 2 bits and sign-extend: 0 -> 0, 15 -> -1, 1 -> 1, 14 -> -2, 7 -> -1, 8 -> 0, 3 -> -1, 12 -> 0
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(0, -1, 1, -2),
      Int2x4(-1, 0, -1, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint4x2_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  // Truncate to 2 bits: 0 -> 0, 15 -> 3, 1 -> 1, 14 -> 2, 7 -> 3, 8 -> 0, 3 -> 3, 12 -> 0
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 3, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, Int2x4ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Values fit directly: -2 -> -2, 1 -> 1, 0 -> 0, -1 -> -1
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(-2, 1),
      Int4x2(0, -1),
      Int4x2(1, -2),
      Int4x2(-1, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Mask to 4 bits: -2 -> 14, 1 -> 1, 0 -> 0, -1 -> 15
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(14, 1),
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(15, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  // Values fit directly: 0-3 all fit in int4
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(0, 3),
      Int4x2(1, 2),
      Int4x2(3, 0),
      Int4x2(2, 1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt2x4ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  // Values fit directly: 0-3 all fit in uint4
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(0, 3),
      UInt4x2(1, 2),
      UInt4x2(3, 0),
      UInt4x2(2, 1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int8ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int8_t> int8_input = {-10, 15, 0, -1, 7, -8, -128, 127};

  // Truncate to 2 bits and sign-extend
  // -10 = 0xF6, truncate to 0x02 = 2, sign-extend to -2
  // 15 = 0x0F, truncate to 0x03 = 3, sign-extend to -1
  // 0 = 0x00, truncate to 0x00 = 0
  // -1 = 0xFF, truncate to 0x03 = 3, sign-extend to -1
  // 7 = 0x07, truncate to 0x03 = 3, sign-extend to -1
  // -8 = 0xF8, truncate to 0x00 = 0
  // -128 = 0x80, truncate to 0x00 = 0
  // 127 = 0x7F, truncate to 0x03 = 3, sign-extend to -1
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -1, 0, -1),
      Int2x4(-1, 0, 0, -1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int8_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, UInt8ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint8_t> uint8_input = {20, 255, 0, 17, 7, 240, 15, 31};

  // values get truncated to lower 2 bits
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 0, 1),  // 20 (0x14) truncate to 0, 255 (0xFF) truncate to 3, etc.
      UInt2x4(3, 0, 3, 3)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint8_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, Int32ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int32_t> int32_input = {-10, INT32_MAX, 0, INT32_MIN, 3, -5, 4080, 287};

  // Truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -1, 0, 0),  // -10 -> -2, INT32_MAX -> -1, 0 -> 0, INT32_MIN -> 0
      Int2x4(-1, -1, 0, -1)  // 3 -> -1, -5 -> -1, 4080 -> 0, 287 -> -1
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int32_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, Int32ToInt2x4OddNumberOfElements) {
  // GIVEN
  const std::vector<int64_t> odd_shape{5};
  const std::vector<int32_t> odd_input = {-10, INT32_MAX, 0, INT32_MIN, 3};

  // Truncate to 2 bits and sign-extend; INT2 packs 4 per byte
  const std::vector<Int2x4> expected_odd_output = {
      Int2x4(-2, -1, 0, 0),  // -10 -> -2, INT32_MAX -> -1, 0 -> 0, INT32_MIN -> 0
      Int2x4(-1, 0, 0, 0)    // 3 -> -1, padded with 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, UInt32ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint32_t> uint32_input = {20, UINT32_MAX, 0, 256, 7, 240, 15, 4095};

  // Truncate to 2 bits (no sign extension for unsigned)
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 0, 0),  // 20 -> 0, UINT32_MAX -> 3, 0 -> 0, 256 -> 0
      UInt2x4(3, 0, 3, 3)   // 7 -> 3, 240 -> 0, 15 -> 3, 4095 -> 3
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint32_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, FloatToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {-2.3f, 1.7f, 0.4f, -1.6f, 3.0f, -5.2f, 240.1f, 31.9f};

  // Round then truncate to 2 bits and sign-extend
  // -2.3 rounds to -2 -> -2
  // 1.7 rounds to 2 -> -2 (truncate and sign-extend)
  // 0.4 rounds to 0 -> 0
  // -1.6 rounds to -2 -> -2
  // 3.0 -> 3, truncate to -1
  // -5.2 rounds to -5 -> -1 (truncate 0x03)
  // 240.1 rounds to 240 -> 0 (truncate)
  // 31.9 rounds to 32 -> 0 (truncate)
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -2, 0, -2),
      Int2x4(-1, -1, 0, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(float_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, FloatToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {0.4f, 3.7f, 1.0f, 2.5f, 4.0f, -1.0f, 15.1f, 31.9f};

  // Round then truncate to 2 bits (round-half-to-even rounding)
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 0, 1, 3),  // 0.4->0, 3.7->4->0, 1.0->1, 2.5->2 (rounds to even)->3 truncated
      UInt2x4(0, 3, 3, 0)   // 4.0->4->0, -1->-1->3, 15.1->15->3, 31.9->32->0
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(float_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, BoolToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(0, 1, 1, 0),
      Int2x4(0, 1, 1, 1)};

  // WHEN, THEN
  TestCastOpInt2(bool_input_span, gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, BoolToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 1, 1, 0),
      UInt2x4(0, 1, 1, 1)};

  // WHEN, THEN
  TestCastOpInt2(bool_input_span, gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, StringToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-2", "1", "0", "-1",
      "1", "-2", "-1", "0"};

  const std::vector<Int2x4> expected_output{
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const std::string>(string_input), gsl::span<const Int2x4>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "0", "3", "1", "2",
      "3", "0", "2", "1"};

  const std::vector<UInt2x4> expected_output{
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const std::string>(string_input), gsl::span<const UInt2x4>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt2x4BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2};
  const std::vector<std::string> string_input = {
      "-5", "20",  // out of range values that get truncated
      "0", "3"     // boundary values that are in range
  };

  // Values get truncated to lower 2 bits (no sign extension for unsigned)
  const std::vector<UInt2x4> expected_output{
      UInt2x4(3, 0, 0, 3)  // -5 -> 3, 20 -> 0, 0 -> 0, 3 -> 3
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const std::string>(string_input), gsl::span<const UInt2x4>(expected_output), shape);
}

TEST(CastOpTest, FloatStringToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-2.7", "1.3",
      "0.4", "-1.6",
      "3.8", "-5.2",
      "15.0", "-2"};

  // Round then truncate to 2 bits and sign-extend
  // -2.7 rounds to -3, -3 & 0x3 = 1, sign-extended = 1
  // 1.3 rounds to 1, 0.4 rounds to 0, -1.6 rounds to -2
  // 3.8 rounds to 4 -> 0, -5.2 rounds to -5 -> -1, 15.0 -> -1, -2 -> -2
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(1, 1, 0, -2),   // -2.7 -> -3 -> 1 (truncate & sign-extend), 1.3 -> 1, 0.4 -> 0, -1.6 -> -2
      Int2x4(0, -1, -1, -2)  // 3.8 -> 4 -> 0, -5.2 -> -5 -> -1, 15.0 -> -1, -2 -> -2
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const std::string>(string_input), gsl::span<const Int2x4>(expected_int2x4_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Negative values will be cast to their unsigned representation
  const std::vector<uint16_t> expected_uint16_output = {65534, 1, 0, UINT16_MAX, 1, 65534, UINT16_MAX, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Negative values will be cast to their unsigned representation
  const std::vector<uint32_t> expected_uint32_output = {4294967294, 1, 0, UINT32_MAX, 1, 4294967294, UINT32_MAX, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, Int2x4ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // Negative values will be cast to their unsigned representation
  const std::vector<uint64_t> expected_uint64_output = {18446744073709551614ULL, 1, 0, UINT64_MAX,
                                                        1, 18446744073709551614ULL, UINT64_MAX, 0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt2x4ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<uint16_t> expected_uint16_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<int16_t> expected_int16_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, UInt2x4ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<uint32_t> expected_uint32_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, UInt2x4ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<uint64_t> expected_uint64_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt2x4ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<int64_t> expected_int64_output = {0, 3, 1, 2, 3, 0, 2, 1};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, UInt2x4ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<double> expected_double_output = {0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, UInt2x4ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f}));

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, Int2x4ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f}));

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int2x4_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, UInt2x4ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f}));

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint2x4_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, Int16ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int16_t> int16_input = {-10, INT16_MAX, 0, INT16_MIN, 3, -5, 4080, 287};

  // Truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -1, 0, 0),
      Int2x4(-1, -1, 0, -1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int16_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, UInt16ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint16_t> uint16_input = {20, UINT16_MAX, 0, 17, 7, 240, 15, 31};

  // Truncate to 2 bits
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 0, 1),
      UInt2x4(3, 0, 3, 3)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint16_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, Int64ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int64_t> int64_input = {-10, INT64_MAX, 0, INT64_MIN, 3, -5, 4080, 287};

  // Truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -1, 0, 0),
      Int2x4(-1, -1, 0, -1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(int64_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, UInt64ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint64_t> uint64_input = {20, UINT64_MAX, 0, 17, 7, 240, 15, 31};

  // Truncate to 2 bits
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 0, 1),
      UInt2x4(3, 0, 3, 3)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(uint64_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, DoubleToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<double> double_input = {-2.3, 1.7, 0.4, -1.6, 3.0, -5.2, 240.1, 31.9};

  // Round then truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, -2, 0, -2),
      Int2x4(-1, -1, 0, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(double_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, DoubleToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<double> double_input = {0.4, 3.7, 1.0, 2.5, 4.0, -1.0, 15.1, 31.9};

  // Round then truncate to 2 bits (round-half-to-even rounding)
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 0, 1, 3),  // 2.5 rounds to 2 (even), truncated to 3 bits -> becomes 3 after truncation
      UInt2x4(0, 3, 3, 0)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(double_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, MLFloat16ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{-2.0f, 1.0f, 0.0f, -1.0f, 3.0f, -5.0f, 15.0f, 31.0f}));

  // Truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(-1, -1, -1, -1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(float16_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, MLFloat16ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 3.0f, 1.0f, 2.0f, 4.0f, 15.0f, 7.0f, 31.0f}));

  // Truncate to 2 bits
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(0, 3, 3, 3)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(float16_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, BFloat16ToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<BFloat16> bfloat16_input =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{-2.0f, 1.0f, 0.0f, -1.0f, 3.0f, -5.0f, 15.0f, 31.0f}));

  // Truncate to 2 bits and sign-extend
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(-1, -1, -1, -1)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(bfloat16_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, BFloat16ToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<BFloat16> bfloat16_input =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 3.0f, 1.0f, 2.0f, 4.0f, 15.0f, 7.0f, 31.0f}));

  // Truncate to 2 bits
  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(0, 3, 3, 3)};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(bfloat16_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, MLFloat16ToInt2x4BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2};
  const MLFloat16 mlfloat16_array[4] = {
      MLFloat16(static_cast<float>(-5)),     // Truncated to lower 2 bits
      MLFloat16(static_cast<float>(4)),      // Truncated to lower 2 bits
      MLFloat16(static_cast<float>(-0.6f)),  // Should round to -1
      MLFloat16(static_cast<float>(1.7f))    // Should round to 2 -> -2 (truncated)
  };

  // Values get truncated to lower 2 bits and sign-extended
  const std::vector<Int2x4> expected_int2x4 = {
      Int2x4(-1, 0, -1, -2)  // -5 -> -1, 4 -> 0, -0.6 -> -1, 1.7 -> 2 -> -2
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const MLFloat16>(mlfloat16_array, 4), gsl::span<const Int2x4>(expected_int2x4),
                 shape);
}

TEST(CastOpTest, MLFloat16ToUInt2x4BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2};
  const MLFloat16 mlfloat16_array[4] = {
      MLFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 2 bits
      MLFloat16(static_cast<float>(20)),    // Above max, truncated to lower 2 bits
      MLFloat16(static_cast<float>(3.4f)),  // Should round to 3
      MLFloat16(static_cast<float>(5.7f))   // Should round to 6 -> 2 (truncated)
  };

  // Values get truncated to lower 2 bits (no sign extension for unsigned)
  const std::vector<UInt2x4> expected_uint2x4 = {
      UInt2x4(3, 0, 3, 2)  // -5 -> 3, 20 -> 0, 3.4 -> 3, 5.7 -> 6 -> 2
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const MLFloat16>(mlfloat16_array, 4), gsl::span<const UInt2x4>(expected_uint2x4),
                 shape);
}

TEST(CastOpTest, BFloat16ToUInt2x4BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2};
  const BFloat16 bfloat16_array[4] = {
      BFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 2 bits
      BFloat16(static_cast<float>(20)),    // Above max, truncated to lower 2 bits
      BFloat16(static_cast<float>(3.4f)),  // Should round to 3
      BFloat16(static_cast<float>(5.7f))   // Should round to 6 -> 2 (truncated)
  };

  // Values get truncated to lower 2 bits (no sign extension for unsigned)
  const std::vector<UInt2x4> expected_uint2x4 = {
      UInt2x4(3, 0, 3, 2)  // -5 -> 3, 20 -> 0, 3.4 -> 3, 5.7 -> 6 -> 2
  };

  // WHEN, THEN
  TestCastOpInt2(gsl::span<const BFloat16>(bfloat16_array, 4), gsl::span<const UInt2x4>(expected_uint2x4),
                 shape);
}

TEST(CastOpTest, Int32ToInt2x4EmptyTensor) {
  // GIVEN
  const std::vector<int64_t> empty_shape{0};
  const std::vector<int32_t> empty_input{};
  const std::vector<Int2x4> expected_empty_output{};

  // WHEN, THEN
  TestCastOpInt2(gsl::make_span(empty_input), gsl::make_span(expected_empty_output), empty_shape);
}

template <typename F6>
void CastOpTestFloat6() {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {-8.0f, -1.0f, -0.125f, 0.0f, 0.125f, 1.0f, 3.5f, 8.0f};
  std::vector<F6> float6_output;
  std::vector<float> expected_float_output;
  float6_output.reserve(float_input.size());
  expected_float_output.reserve(float_input.size());

  for (float value : float_input) {
    const F6 quantized(value);
    float6_output.push_back(quantized);
    expected_float_output.push_back(static_cast<float>(quantized));
  }

  TestCastOp<float, F6>(gsl::make_span(float_input), gsl::make_span(float6_output), shape,
                        OpTester::ExpectResult::kExpectSuccess, "", 28);
  TestCastOp<F6, float>(gsl::make_span(float6_output), gsl::make_span(expected_float_output), shape,
                        OpTester::ExpectResult::kExpectSuccess, "", 28);
}

TEST(CastOpTest, Float6E2M3) {
  CastOpTestFloat6<Float6E2M3>();
}

TEST(CastOpTest, Float6E3M2) {
  CastOpTestFloat6<Float6E3M2>();
}

template <typename F6>
void TestFloat6RoundingAtEveryMidpoint() {
  for (uint8_t bits = 0; bits < 0x1F; ++bits) {
    const float lower = static_cast<float>(F6(bits, F6::FromBits()));
    const float upper = static_cast<float>(F6(bits + 1, F6::FromBits()));
    const float midpoint = (lower + upper) / 2;
    const uint8_t nearest_even = (bits & 1) == 0 ? bits : bits + 1;
    EXPECT_EQ(F6(std::nextafter(midpoint, lower)).ToBits(), bits);
    EXPECT_EQ(F6(midpoint).ToBits(), nearest_even);
    EXPECT_EQ(F6(std::nextafter(midpoint, upper)).ToBits(), bits + 1);
    EXPECT_EQ(F6(-midpoint).ToBits(), static_cast<uint8_t>(0x20 | nearest_even));
  }
}

TEST(CastOpTest, Float6RoundingAtEveryMidpoint) {
  TestFloat6RoundingAtEveryMidpoint<Float6E2M3>();
  TestFloat6RoundingAtEveryMidpoint<Float6E3M2>();
}

TEST(CastOpTest, DoubleToFloat6RoundingNearMidpoints) {
  const std::vector<int64_t> shape{3};
  const double e2_midpoint = 1.1875;
  const double e3_midpoint = 1.375;
  const std::vector<double> e2_values{std::nextafter(e2_midpoint, 0.0), e2_midpoint,
                                      std::nextafter(e2_midpoint, 2.0)};
  const std::vector<double> e3_values{std::nextafter(e3_midpoint, 0.0), e3_midpoint,
                                      std::nextafter(e3_midpoint, 2.0)};
  const std::vector<Float6E2M3> e2_expected{
      Float6E2M3(0x09, Float6E2M3::FromBits()), Float6E2M3(0x0A, Float6E2M3::FromBits()),
      Float6E2M3(0x0A, Float6E2M3::FromBits())};
  const std::vector<Float6E3M2> e3_expected{
      Float6E3M2(0x0D, Float6E3M2::FromBits()), Float6E3M2(0x0E, Float6E3M2::FromBits()),
      Float6E3M2(0x0E, Float6E3M2::FromBits())};

  TestCastOp<double, Float6E2M3>(gsl::make_span(e2_values), gsl::make_span(e2_expected), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 28);
  TestCastOp<double, Float6E3M2>(gsl::make_span(e3_values), gsl::make_span(e3_expected), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 28);
}

TEST(CastOpTest, StringToFloat6RoundingNearMidpoints) {
  const std::vector<int64_t> shape{2};
  const std::vector<std::string> e2_values{"1.1874999999999998", "1.1875"};
  const std::vector<std::string> e3_values{"1.3749999999999998", "1.375"};
  const std::vector<Float6E2M3> e2_expected{
      Float6E2M3(0x09, Float6E2M3::FromBits()), Float6E2M3(0x0A, Float6E2M3::FromBits())};
  const std::vector<Float6E3M2> e3_expected{
      Float6E3M2(0x0D, Float6E3M2::FromBits()), Float6E3M2(0x0E, Float6E3M2::FromBits())};

  TestCastOp<std::string, Float6E2M3>(gsl::make_span(e2_values), gsl::make_span(e2_expected), shape,
                                      OpTester::ExpectResult::kExpectSuccess, "", 28);
  TestCastOp<std::string, Float6E3M2>(gsl::make_span(e3_values), gsl::make_span(e3_expected), shape,
                                      OpTester::ExpectResult::kExpectSuccess, "", 28);
}

TEST(CastOpTest, Float6EncodingBoundaries) {
  const float infinity = std::numeric_limits<float>::infinity();

  EXPECT_EQ(Float6E2M3(-0.0f).ToBits(), 0x20);
  EXPECT_EQ(Float6E3M2(-0.0f).ToBits(), 0x20);
  EXPECT_EQ(Float6E2M3(infinity).ToBits(), 0x1F);
  EXPECT_EQ(Float6E2M3(-infinity).ToBits(), 0x3F);
  EXPECT_EQ(Float6E3M2(infinity).ToBits(), 0x1F);
  EXPECT_EQ(Float6E3M2(-infinity).ToBits(), 0x3F);
  EXPECT_EQ(Float6E2M3(std::numeric_limits<float>::max()).ToBits(), 0x1F);
  EXPECT_EQ(Float6E2M3(-std::numeric_limits<float>::max()).ToBits(), 0x3F);
  EXPECT_EQ(Float6E3M2(std::numeric_limits<float>::max()).ToBits(), 0x1F);
  EXPECT_EQ(Float6E3M2(-std::numeric_limits<float>::max()).ToBits(), 0x3F);
  EXPECT_EQ(Float6E2M3(std::numeric_limits<float>::quiet_NaN()).ToBits(), 0x20);
  EXPECT_EQ(Float6E3M2(std::numeric_limits<float>::quiet_NaN()).ToBits(), 0x20);

  // Round-to-nearest-even at adjacent normalized E2M3 and E3M2 values.
  EXPECT_EQ(Float6E2M3(1.0625f).ToBits(), 0x08);
  EXPECT_EQ(Float6E2M3(1.1875f).ToBits(), 0x0A);
  EXPECT_EQ(Float6E3M2(1.125f).ToBits(), 0x0C);
  EXPECT_EQ(Float6E3M2(1.375f).ToBits(), 0x0E);

  EXPECT_EQ(static_cast<float>(Float6E2M3(0x01, Float6E2M3::FromBits())), 0.125f);
  EXPECT_EQ(static_cast<float>(Float6E2M3(0x1F, Float6E2M3::FromBits())), 7.5f);
  EXPECT_EQ(static_cast<float>(Float6E3M2(0x01, Float6E3M2::FromBits())), 0.0625f);
  EXPECT_EQ(static_cast<float>(Float6E3M2(0x1F, Float6E3M2::FromBits())), 28.0f);
}

template <typename F6>
void TestFloat6ToPackedIntegerTruncation() {
  const std::vector<int64_t> shape{2};
  const uint8_t positive_fractional_bits = std::is_same_v<F6, Float6E2M3> ? 0x0C : 0x0E;
  const uint8_t negative_fractional_bits = std::is_same_v<F6, Float6E2M3> ? 0x2C : 0x2E;
  const std::vector<F6> input{F6(positive_fractional_bits, F6::FromBits()),
                              F6(negative_fractional_bits, F6::FromBits())};  // +1.5, -1.5

  const std::vector<Int4x2> int4_output{Int4x2(std::byte{0xF1})};
  TestCastOp(gsl::make_span(input), gsl::make_span(int4_output), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 28);

  const std::vector<UInt4x2> uint4_output{UInt4x2(std::byte{0xF1})};
  TestCastOp(gsl::make_span(input), gsl::make_span(uint4_output), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 28);

  const std::vector<Int2x4> int2_output{Int2x4(std::byte{0x0D})};
  TestCastOp(gsl::make_span(input), gsl::make_span(int2_output), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 28);

  const std::vector<UInt2x4> uint2_output{UInt2x4(std::byte{0x0D})};
  TestCastOp(gsl::make_span(input), gsl::make_span(uint2_output), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 28);
}

TEST(CastOpTest, Float6E2M3ToPackedIntegerTruncation) {
  TestFloat6ToPackedIntegerTruncation<Float6E2M3>();
}

TEST(CastOpTest, Float6E3M2ToPackedIntegerTruncation) {
  TestFloat6ToPackedIntegerTruncation<Float6E3M2>();
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename F8>
void CastOpTestFloat8(Saturate saturate) {
  ASSERT_NE(saturate, Saturate::None);
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f, -0.120196559f, 5.0f,
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()};

  // float output precision is 8, so the expected output differs slightly from the input due to that
  std::vector<F8> output;
  output.reserve(float_input.size());
  for (size_t i = 0; i < float_input.size(); ++i) {
    output.emplace_back(F8(float_input[i], saturate == Saturate::True));
  }
  TestCastOp<float, F8>(gsl::make_span(float_input), gsl::make_span(output), shape, OpTester::ExpectResult::kExpectSuccess, "", 19, saturate);

  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(gsl::make_span(float_input));

  TestCastOp<MLFloat16, F8>(gsl::make_span(float16_input), gsl::make_span(output), shape, OpTester::ExpectResult::kExpectSuccess, "", 19, saturate);
}

TEST(CastOpTest, ToFloat8E4M3FN) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCudaExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    CastOpTestFloat8<Float8E4M3FN>(Saturate::True);
    CastOpTestFloat8<Float8E4M3FN>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E4M3FNUZ) {
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());
  if (enable_cpu) {
    CastOpTestFloat8<Float8E4M3FNUZ>(Saturate::True);
    CastOpTestFloat8<Float8E4M3FNUZ>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E5M2) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCudaExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    CastOpTestFloat8<Float8E5M2>(Saturate::True);
    CastOpTestFloat8<Float8E5M2>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E5M2FNUZ) {
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());
  if (enable_cpu) {
    CastOpTestFloat8<Float8E5M2FNUZ>(Saturate::True);
    CastOpTestFloat8<Float8E5M2FNUZ>(Saturate::False);
  }
}

TEST(CastOpTest, Int4x2ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  std::vector<Float8E4M3FN> expected_float8_output;
  expected_float8_output.reserve(8);
  const std::vector<float> float_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : float_values) {
    expected_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<Int4x2, Float8E4M3FN>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<Int4x2, Float8E4M3FN>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8_output), shape,
                                   OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, UInt4x2ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  std::vector<Float8E4M3FN> expected_uint_float8_output;
  expected_uint_float8_output.reserve(8);
  const std::vector<float> uint_float_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_float_values) {
    expected_uint_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<UInt4x2, Float8E4M3FN>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<UInt4x2, Float8E4M3FN>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8_output), shape,
                                    OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, Int4x2ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  std::vector<Float8E5M2> expected_float8e5m2_output;
  expected_float8e5m2_output.reserve(8);
  const std::vector<float> float_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : float_values) {
    expected_float8e5m2_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<Int4x2, Float8E5M2>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8e5m2_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<Int4x2, Float8E5M2>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8e5m2_output), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, UInt4x2ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  std::vector<Float8E5M2> expected_uint_float8e5m2_output;
  expected_uint_float8e5m2_output.reserve(8);
  const std::vector<float> uint_float_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_float_values) {
    expected_uint_float8e5m2_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<UInt4x2, Float8E5M2>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8e5m2_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<UInt4x2, Float8E5M2>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8e5m2_output), shape,
                                  OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, Float8E4M3FNToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, Int4x2>(gsl::make_span(float8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToInt4x2_OddShape) {
  // GIVEN
  const std::vector<int64_t> shape{1, 2, 3};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, Int4x2>(gsl::make_span(float8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> uint_float8_input;
  const std::vector<float> uint_input_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_input_values) {
    uint_float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, UInt4x2>(gsl::make_span(uint_float8_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int2x4ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  std::vector<Float8E4M3FN> expected_float8_output;
  const std::vector<float> expected_values = {-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f};
  for (float val : expected_values) {
    expected_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  TestCastOpInt2<Int2x4, Float8E4M3FN>(gsl::make_span(int2x4_input), gsl::make_span(expected_float8_output), shape);
}

TEST(CastOpTest, UInt2x4ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  std::vector<Float8E4M3FN> expected_float8_output;
  const std::vector<float> expected_values = {0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f};
  for (float val : expected_values) {
    expected_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  TestCastOpInt2<UInt2x4, Float8E4M3FN>(gsl::make_span(uint2x4_input), gsl::make_span(expected_float8_output), shape);
}

TEST(CastOpTest, Int2x4ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int2x4> int2x4_input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  std::vector<Float8E5M2> expected_float8_output;
  const std::vector<float> expected_values = {-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f};
  for (float val : expected_values) {
    expected_float8_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  TestCastOpInt2<Int2x4, Float8E5M2>(gsl::make_span(int2x4_input), gsl::make_span(expected_float8_output), shape);
}

TEST(CastOpTest, UInt2x4ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt2x4> uint2x4_input = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  std::vector<Float8E5M2> expected_float8_output;
  const std::vector<float> expected_values = {0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f};
  for (float val : expected_values) {
    expected_float8_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  TestCastOpInt2<UInt2x4, Float8E5M2>(gsl::make_span(uint2x4_input), gsl::make_span(expected_float8_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-2.0f, 1.0f, 0.0f, -1.0f, 1.0f, -2.0f, -1.0f, 0.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, -2, -1, 0)};

  // WHEN, THEN
  TestCastOpInt2<Float8E4M3FN, Int2x4>(gsl::make_span(float8_input), gsl::make_span(expected_int2x4_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToUInt2x4) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {0.0f, 3.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<UInt2x4> expected_uint2x4_output = {
      UInt2x4(0, 3, 1, 2),
      UInt2x4(3, 0, 2, 1)};

  // WHEN, THEN
  TestCastOpInt2<Float8E4M3FN, UInt2x4>(gsl::make_span(float8_input), gsl::make_span(expected_uint2x4_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToInt2x4_OddShape) {
  // GIVEN
  const std::vector<int64_t> shape{5};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-2.0f, 1.0f, 0.0f, -1.0f, 1.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  // 5 elements padded to 8 (2 Int2x4 values)
  const std::vector<Int2x4> expected_int2x4_output = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, 0, 0, 0)  // padded with 0
  };

  // WHEN, THEN
  TestCastOpInt2<Float8E4M3FN, Int2x4>(gsl::make_span(float8_input), gsl::make_span(expected_int2x4_output), shape);
}

#endif

#if !defined(DISABLE_FLOAT4_TYPES) && defined(USE_CUDA)

template <typename F4>
void CastOpTestFloatFloat4(std::vector<int64_t> shape,
                           std::vector<float> float_data,
                           bool is_fp4_input = false,
                           int opset = 23) {
  int num_pairs = static_cast<int>(float_data.size()) / 2;
  int num_fp4_elements = static_cast<int>((float_data.size() + 1) / 2);
  bool is_odd_count = (float_data.size() % 2 != 0);

  std::vector<F4> fp4_data;
  fp4_data.reserve(num_fp4_elements);

  for (int i = 0; i < num_pairs; ++i) {
    fp4_data.emplace_back(F4(float_data[i * 2], float_data[i * 2 + 1]));
  }

  if (is_odd_count) {
    fp4_data.emplace_back(F4(float_data.back(), 0));  // Padding zero
  }

  if (!is_fp4_input) {
    TestCastOp<float, F4>(gsl::make_span(float_data), gsl::make_span(fp4_data), shape,
                          OpTester::ExpectResult::kExpectSuccess, "", opset, Saturate::None, true);

  } else {
    std::vector<float> casted_back_float;
    for (int i = 0; i < num_pairs; ++i) {
      auto pair = fp4_data[i].ToFloat2();
      casted_back_float.push_back(pair.first);
      casted_back_float.push_back(pair.second);
    }

    if (is_odd_count) {
      casted_back_float.push_back(fp4_data[num_pairs].ToFloat2().first);
    }

    TestCastOp<F4, float>(gsl::make_span(fp4_data), gsl::make_span(casted_back_float), shape,
                          OpTester::ExpectResult::kExpectSuccess, "", opset, Saturate::None, true);
  }
}

static std::vector<float> GenerateRandomFloatVector(size_t count) {
  std::vector<float> ret;
  ret.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    int sign = (((rand() % 2) == 0) ? 1 : -1);
    float random = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 7.f;  // let some values be outside the range of FP4
    ret.push_back(sign * random);
  }

  return ret;
}

TEST(CastOpTest, FloatToFloat4E2M1x2) {
  // Even count test (with some special values)
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       7.f, -7.f,
                                       0.5f, -0.5f,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       -std::numeric_limits<float>::quiet_NaN()});

  // Odd count test
  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f,
                                       0.987f,
                                       43.8f});

  // Arbitrary sized tests
  std::vector<int64_t> counts = {1, 5, 256, 512, 1024, 1025, 2048, 2049, 127, 89, 53, 42};

  for (auto s : counts) {
    CastOpTestFloatFloat4<Float4E2M1x2>({s, 1, 1}, GenerateRandomFloatVector(s));
  }
}

TEST(CastOpTest, Float4E2M1x2ToFloat) {
  // Even count test (with some special values)
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {0.5f, 7.34f,
                                       1.f, 1.5f,
                                       2.f, 3.f,
                                       4.f, 6.f},
                                      true);

  // Odd count test
  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f,
                                       0.987f,
                                       43.8f},
                                      true);

  // Arbitrary sized tests
  std::vector<int64_t> counts = {1, 5, 256, 512, 1024, 1025, 2048, 2049, 127, 89, 53, 42};

  for (auto s : counts) {
    CastOpTestFloatFloat4<Float4E2M1x2>({s, 1, 1}, GenerateRandomFloatVector(s), true);
  }
}

// Opset 25 tests for Float4 types on CUDA
TEST(CastOpTest, FloatToFloat4E2M1x2_Opset25) {
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       7.f, -7.f,
                                       0.5f, -0.5f,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       -std::numeric_limits<float>::quiet_NaN()},
                                      false, 25);

  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f, 0.987f, 43.8f},
                                      false, 25);
}

TEST(CastOpTest, Float4E2M1x2ToFloat_Opset25) {
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {0.5f, 7.34f,
                                       1.f, 1.5f,
                                       2.f, 3.f,
                                       4.f, 6.f},
                                      true, 25);

  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f, 0.987f, 43.8f},
                                      true, 25);
}

#endif

// Opset 25 tests for standard types on CUDA.
// Verifies CUDA Cast kernel registration at opset 25 works for common type conversions.
#if defined(USE_CUDA)

TEST(CastOpTest, StandardTypes_Opset25_Cuda) {
  const std::vector<int64_t> shape{2, 3};

  // float -> double
  {
    const std::vector<float> input = {1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f};
    const std::vector<double> expected = {1.0, 2.5, -3.0, 0.0, 100.0, -0.5};
    TestCastOp<float, double>(gsl::make_span(input), gsl::make_span(expected), shape,
                              OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // double -> float
  {
    const std::vector<double> input = {1.0, 2.5, -3.0, 0.0, 100.0, -0.5};
    const std::vector<float> expected = {1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f};
    TestCastOp<double, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                              OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // float -> int32_t
  {
    const std::vector<float> input = {1.0f, 2.9f, -3.0f, 0.0f, 100.0f, -0.5f};
    const std::vector<int32_t> expected = {1, 2, -3, 0, 100, 0};
    TestCastOp<float, int32_t>(gsl::make_span(input), gsl::make_span(expected), shape,
                               OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // int32_t -> float
  {
    const std::vector<int32_t> input = {1, 2, -3, 0, 100, -7};
    const std::vector<float> expected = {1.0f, 2.0f, -3.0f, 0.0f, 100.0f, -7.0f};
    TestCastOp<int32_t, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                               OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // float -> MLFloat16
  if (HasCudaEnvironment(530)) {
    const std::vector<float> input = {1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f};
    const std::vector<MLFloat16> expected = CastedValues<float, MLFloat16>(gsl::make_span(input));
    TestCastOp<float, MLFloat16>(gsl::make_span(input), gsl::make_span(expected), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // MLFloat16 -> float
  if (HasCudaEnvironment(530)) {
    const std::vector<MLFloat16> input = CastedValues<float, MLFloat16>(
        gsl::make_span(std::vector<float>{1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f}));
    const std::vector<float> expected = {1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f};
    TestCastOp<MLFloat16, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // BFloat16 -> float
  if (HasCudaEnvironment(800)) {
    const std::vector<BFloat16> input = CastedValues<float, BFloat16>(
        gsl::make_span(std::vector<float>{1.0f, 2.5f, -3.0f, 0.0f, 100.0f, -0.5f}));
    const std::vector<float> expected = CastedValues<BFloat16, float>(gsl::make_span(input));
    TestCastOp<BFloat16, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                                OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // bool -> float
  {
    const bool input[] = {true, false, true, true, false, false};
    const gsl::span<const bool> input_span(input);
    const std::vector<float> expected = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    TestCastOp<bool, float>(input_span, gsl::make_span(expected), shape,
                            OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

TEST(CastOpTest, Float8_Opset25_Cuda) {
  constexpr int min_cuda_architecture = 11080;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    return;
  }

  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f,
                                          -0.120196559f, 5.0f,
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()};

  // Float8E4M3FN: float -> Float8E4M3FN at opset 25
  {
    std::vector<Float8E4M3FN> output;
    output.reserve(float_input.size());
    for (size_t i = 0; i < float_input.size(); ++i) {
      output.emplace_back(Float8E4M3FN(float_input[i], true));
    }
    TestCastOp<float, Float8E4M3FN>(gsl::make_span(float_input), gsl::make_span(output), shape,
                                    OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::True, /*cuda_only=*/true);
  }

  // Float8E5M2: float -> Float8E5M2 at opset 25
  {
    std::vector<Float8E5M2> output;
    output.reserve(float_input.size());
    for (size_t i = 0; i < float_input.size(); ++i) {
      output.emplace_back(Float8E5M2(float_input[i], true));
    }
    TestCastOp<float, Float8E5M2>(gsl::make_span(float_input), gsl::make_span(output), shape,
                                  OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::True, /*cuda_only=*/true);
  }

  // Float8E4M3FN -> float at opset 25
  {
    std::vector<Float8E4M3FN> input;
    input.reserve(float_input.size());
    for (size_t i = 0; i < float_input.size(); ++i) {
      input.emplace_back(Float8E4M3FN(float_input[i], true));
    }
    std::vector<float> expected;
    expected.reserve(input.size());
    for (const auto& v : input) {
      expected.push_back(v.ToFloat());
    }
    TestCastOp<Float8E4M3FN, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                                    OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }

  // Float8E5M2 -> float at opset 25
  {
    std::vector<Float8E5M2> input;
    input.reserve(float_input.size());
    for (size_t i = 0; i < float_input.size(); ++i) {
      input.emplace_back(Float8E5M2(float_input[i], true));
    }
    std::vector<float> expected;
    expected.reserve(input.size());
    for (const auto& v : input) {
      expected.push_back(v.ToFloat());
    }
    TestCastOp<Float8E5M2, float>(gsl::make_span(input), gsl::make_span(expected), shape,
                                  OpTester::ExpectResult::kExpectSuccess, "", 25, Saturate::None, /*cuda_only=*/true);
  }
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)

#endif  // defined(USE_CUDA)

// Regression tests for sub-byte same-type cast (CopyCpuTensor heap overflow fix).
// When src and dst types are the same, Cast::Compute calls CopyCpuTensor which must
// use SizeInBytes() (not shape.Size() * DataType()->Size()) for the memcpy byte count.

TEST(CastOpTest, Int4x2ToInt4x2_SameType) {
  const std::vector<int64_t> shape{3, 3};  // 9 elements (odd, tests ceil-division)
  const std::vector<Int4x2> input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2),
      Int4x2(1, 0)  // 9th element in low nibble of 5th byte (padding in high nibble)
  };

  TestCastOp(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, UInt4x2ToUInt4x2_SameType) {
  const std::vector<int64_t> shape{2, 5};  // 10 elements (even)
  const std::vector<UInt4x2> input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 6),
      UInt4x2(9, 11)};

  TestCastOp(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, Int4x2ToInt4x2_LargeShape) {
  // Large shape (16464 elements)
  const std::vector<int64_t> shape{28, 6, 14, 7};
  const int64_t num_elements = 28 * 6 * 14 * 7;  // 16464
  const size_t num_storage = static_cast<size_t>((num_elements + 1) / 2);

  std::vector<Int4x2> input_vec(num_storage);
  for (size_t i = 0; i < num_storage; ++i) {
    input_vec[i] = Int4x2(static_cast<int8_t>(i % 8), static_cast<int8_t>(-(static_cast<int8_t>(i % 7))));
  }
  const auto& input = input_vec;

  TestCastOp(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, Int2x4ToInt2x4_SameType) {
  const std::vector<int64_t> shape{5};  // 5 elements (not multiple of 4, tests ceil-division)
  const std::vector<Int2x4> input = {
      Int2x4(-2, 1, 0, -1),
      Int2x4(1, 0, 0, 0)  // 5th element in first position (padding in positions 2-4)
  };

  TestCastOpInt2(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, UInt4x2ToUInt4x2_LargeShape) {
  const std::vector<int64_t> shape{28, 6, 14, 7};
  const int64_t num_elements = 28 * 6 * 14 * 7;  // 16464
  const size_t num_storage = static_cast<size_t>((num_elements + 1) / 2);

  std::vector<UInt4x2> input_vec(num_storage);
  for (size_t i = 0; i < num_storage; ++i) {
    input_vec[i] = UInt4x2(static_cast<uint8_t>(i % 16), static_cast<uint8_t>((i + 3) % 16));
  }
  const auto& input = input_vec;

  TestCastOp(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, Int2x4ToInt2x4_LargeShape) {
  const std::vector<int64_t> shape{100, 100};  // 10000 elements (not multiple of 4)
  const int64_t num_elements = 100 * 100;
  const size_t num_storage = static_cast<size_t>((num_elements + 3) / 4);

  std::vector<Int2x4> input_vec(num_storage);
  for (size_t i = 0; i < num_storage; ++i) {
    input_vec[i] = Int2x4(static_cast<int8_t>(i % 2), static_cast<int8_t>(-(static_cast<int8_t>(i % 2))),
                          static_cast<int8_t>((i + 1) % 2), static_cast<int8_t>(0));
  }
  const auto& input = input_vec;

  TestCastOpInt2(gsl::make_span(input), gsl::make_span(input), shape);
}

TEST(CastOpTest, UInt2x4ToUInt2x4_LargeShape) {
  const std::vector<int64_t> shape{100, 101};  // 10100 elements (not multiple of 4)
  const int64_t num_elements = 100 * 101;
  const size_t num_storage = static_cast<size_t>((num_elements + 3) / 4);

  std::vector<UInt2x4> input_vec(num_storage);
  for (size_t i = 0; i < num_storage; ++i) {
    input_vec[i] = UInt2x4(static_cast<uint8_t>(i % 4), static_cast<uint8_t>((i + 1) % 4),
                           static_cast<uint8_t>((i + 2) % 4), static_cast<uint8_t>((i + 3) % 4));
  }
  const auto& input = input_vec;

  TestCastOpInt2(gsl::make_span(input), gsl::make_span(input), shape);
}

// Direct CopyCpuTensor test with guaranteed distinct buffers to exercise the memcpy path.
// This bypasses the MayInplace optimization that can alias input/output in OpTester.
// Uses guard bytes after the valid buffer region to detect overflow deterministically
// without relying on ASan; the pre-fix code would overwrite these sentinel bytes.
TEST(CastOpTest, CopyCpuTensor_SubByteTypes_DistinctBuffers) {
  constexpr uint8_t kGuardByte = 0xCD;
  constexpr size_t kGuardSize = 64;

  // Helper: allocate a buffer of `valid_bytes` + guard region, fill guard with sentinel,
  // then construct a non-owning Tensor over the valid portion.
  auto make_guarded_tensor = [&](MLDataType dtype, const TensorShape& shape,
                                 size_t valid_bytes, std::vector<uint8_t>& backing) {
    backing.resize(valid_bytes + kGuardSize);
    std::memset(backing.data() + valid_bytes, kGuardByte, kGuardSize);
    return Tensor(dtype, shape, backing.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  };

  auto check_guard = [&](const std::vector<uint8_t>& backing, size_t valid_bytes,
                         const char* label) {
    for (size_t i = 0; i < kGuardSize; ++i) {
      EXPECT_EQ(backing[valid_bytes + i], kGuardByte)
          << label << ": guard byte at offset " << i << " was overwritten (heap overflow detected)";
    }
  };

  // Test Int4x2 with odd element count (ceil-division edge case)
  {
    const int64_t num_logical_elements = 17;  // odd: requires ceil(17/2) = 9 storage bytes
    TensorShape shape({num_logical_elements});
    auto int4_type = DataTypeImpl::GetType<Int4x2>();
    constexpr size_t expected_bytes = 9;

    std::vector<uint8_t> src_backing, dst_backing;
    Tensor src = make_guarded_tensor(int4_type, shape, expected_bytes, src_backing);
    Tensor dst = make_guarded_tensor(int4_type, shape, expected_bytes, dst_backing);

    ASSERT_EQ(src.SizeInBytes(), expected_bytes);

    // Fill source with known pattern
    for (size_t i = 0; i < expected_bytes; ++i) {
      src_backing[i] = static_cast<uint8_t>(0xA0 + i);
    }
    // Fill destination valid region with different pattern
    std::memset(dst_backing.data(), 0xFF, expected_bytes);

    ASSERT_NE(src.DataRaw(), dst.MutableDataRaw());

    CopyCpuTensor(&src, &dst);

    // Verify copy correctness
    for (size_t i = 0; i < expected_bytes; ++i) {
      EXPECT_EQ(dst_backing[i], src_backing[i]) << "Int4x2: mismatch at byte " << i;
    }
    // Verify no overflow past the valid region
    check_guard(src_backing, expected_bytes, "Int4x2 src");
    check_guard(dst_backing, expected_bytes, "Int4x2 dst");
  }

  // Test UInt4x2 with large even element count (matches PoC shape)
  {
    const int64_t num_logical_elements = 16464;  // from PoC: ceil(16464/2) = 8232 bytes
    TensorShape shape({num_logical_elements});
    auto uint4_type = DataTypeImpl::GetType<UInt4x2>();
    constexpr size_t expected_bytes = 8232;

    std::vector<uint8_t> src_backing, dst_backing;
    Tensor src = make_guarded_tensor(uint4_type, shape, expected_bytes, src_backing);
    Tensor dst = make_guarded_tensor(uint4_type, shape, expected_bytes, dst_backing);

    ASSERT_EQ(src.SizeInBytes(), expected_bytes);

    for (size_t i = 0; i < expected_bytes; ++i) {
      src_backing[i] = static_cast<uint8_t>(i & 0xFF);
    }
    std::memset(dst_backing.data(), 0xFF, expected_bytes);

    ASSERT_NE(src.DataRaw(), dst.MutableDataRaw());

    CopyCpuTensor(&src, &dst);

    for (size_t i = 0; i < expected_bytes; ++i) {
      EXPECT_EQ(dst_backing[i], src_backing[i]) << "UInt4x2: mismatch at byte " << i;
    }
    check_guard(src_backing, expected_bytes, "UInt4x2 src");
    check_guard(dst_backing, expected_bytes, "UInt4x2 dst");
  }

  // Test Int2x4 (4 elements per byte — would be 4x overflow with old code)
  {
    const int64_t num_logical_elements = 7;  // ceil(7/4) = 2 storage bytes
    TensorShape shape({num_logical_elements});
    auto int2_type = DataTypeImpl::GetType<Int2x4>();
    constexpr size_t expected_bytes = 2;

    std::vector<uint8_t> src_backing, dst_backing;
    Tensor src = make_guarded_tensor(int2_type, shape, expected_bytes, src_backing);
    Tensor dst = make_guarded_tensor(int2_type, shape, expected_bytes, dst_backing);

    ASSERT_EQ(src.SizeInBytes(), expected_bytes);

    src_backing[0] = 0xAB;
    src_backing[1] = 0xCD;
    std::memset(dst_backing.data(), 0xFF, expected_bytes);

    ASSERT_NE(src.DataRaw(), dst.MutableDataRaw());

    CopyCpuTensor(&src, &dst);

    for (size_t i = 0; i < expected_bytes; ++i) {
      EXPECT_EQ(dst_backing[i], src_backing[i]) << "Int2x4: mismatch at byte " << i;
    }
    check_guard(src_backing, expected_bytes, "Int2x4 src");
    check_guard(dst_backing, expected_bytes, "Int2x4 dst");
  }
}

// Correctness test for Cast kernel with a moderately large tensor.
// Exercises the same kernel code path as tensors > 2^31 elements but stays within
// CI GPU memory limits. For the actual overflow scenario, see the host-side test below.
TEST(CastOpTest, CastKernelCorrectness_ModerateSize) {
  constexpr int64_t num_elements = 1 << 24;  // 16M elements
  const std::vector<int64_t> shape = {num_elements};

  std::vector<float> input(num_elements);
  std::vector<int32_t> expected(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    input[i] = static_cast<float>(i % 1000);
    expected[i] = static_cast<int32_t>(i % 1000);
  }

  TestCastOp<float, int32_t>(gsl::make_span(input), gsl::make_span(expected), shape);
}

// Host-side regression test that verifies the grid launch arithmetic uses 64-bit
// types for element counts exceeding INT32_MAX. This validates the fix without
// needing to allocate > 8 GB of GPU memory.
// The fix changed:
//   CUDA_LONG N = static_cast<CUDA_LONG>(count)  // was int32 truncation
// to:
//   int64_t N = static_cast<int64_t>(count)       // correct 64-bit
TEST(CastOpTest, CastKernel_Int64IndexArithmetic_NoOverflow) {
  // Simulate the grid launch calculation from UnaryElementWiseImpl / CudaCastStd
  // with a count that exceeds INT32_MAX.
  constexpr size_t count = static_cast<size_t>(INT32_MAX) + 65536;  // 2^31 + 65536
  constexpr int maxThreadsPerBlock = 256;
  constexpr int maxElementsPerThread = 4;

  // Verify N is correctly represented (not truncated to int32)
  int64_t N = static_cast<int64_t>(count);
  ASSERT_GT(N, static_cast<int64_t>(INT32_MAX));
  ASSERT_EQ(N, static_cast<int64_t>(count));

  // Verify blocksPerGrid calculation doesn't overflow
  // (uses size_t arithmetic for the divisor)
  size_t elements_per_block = static_cast<size_t>(maxThreadsPerBlock) * maxElementsPerThread;
  int blocksPerGrid = static_cast<int>((count + elements_per_block - 1) / elements_per_block);
  ASSERT_GT(blocksPerGrid, 0);
  // For count = 2^31 + 65536, elements_per_block = 1024, we expect ~2M blocks
  ASSERT_EQ(blocksPerGrid, static_cast<int>((count + 1023) / 1024));

  // Verify that the per-thread index calculation doesn't overflow in int64_t
  // Simulate the last block's thread 0: id = NumElementsPerThread * NumThreadsPerBlock * (blocksPerGrid-1) + 0
  int64_t last_block_start = static_cast<int64_t>(maxElementsPerThread) * maxThreadsPerBlock *
                             (blocksPerGrid - 1);
  ASSERT_GT(last_block_start, 0);  // Positive (no overflow)
  ASSERT_LE(last_block_start, N);  // Within bounds

  // Verify the old int32 code would have failed:
  // static_cast<int32_t>(count) would silently wrap
  int32_t truncated_N = static_cast<int32_t>(count);
  ASSERT_LT(truncated_N, 0);  // Proves the old code was broken (wraps negative)
}

#if !defined(DISABLE_FLOAT8_TYPES)

float FloatFromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

template <typename SrcType>
void TestCastToFloat8E8M0(gsl::span<const SrcType> input,
                          gsl::span<const Float8E8M0> output,
                          const std::vector<int64_t>& shape,
                          Saturate saturate = Saturate::None,
                          const std::string& round_mode = "",
                          int opset = 24,
                          OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                          const std::string& expected_failure_string = "") {
  OpTester test("Cast", opset);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<Float8E8M0>());
  test.AddInput<SrcType>("input", shape, input.data(), input.size());
  test.AddOutput<Float8E8M0>("output", shape, output.data(), output.size());
  if (saturate != Saturate::None) {
    test.AddAttribute<int64_t>("saturate", saturate == Saturate::True ? 1 : 0);
  }
  if (!round_mode.empty()) {
    test.AddAttribute<std::string>("round_mode", round_mode);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .Config(expect_result, expected_failure_string)
      .RunWithConfig();
}

template <typename DstType>
void TestCastFromFloat8E8M0(gsl::span<const Float8E8M0> input,
                            gsl::span<const DstType> output,
                            const std::vector<int64_t>& shape,
                            int opset = 24) {
  OpTester test("Cast", opset);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
  test.AddInput<Float8E8M0>("input", shape, input.data(), input.size());
  test.AddOutput<DstType>("output", shape, output.data(), output.size());

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}

TEST(CastOpTest, FloatToFloat8E8M0_Saturate) {
  const std::vector<int64_t> shape{8};
  // Test values: NaN, -1, positive, 1.5 (tie), -Inf, +Inf, 0, very small
  const std::vector<float> input = {NAN, -1.0f, 4.0f, 1.5f,
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    0.0f, 1e-39f};

  std::vector<Float8E8M0> expected;
  expected.reserve(input.size());
  for (float v : input) {
    expected.emplace_back(Float8E8M0(v, true));
  }

  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True);
}

TEST(CastOpTest, FloatToFloat8E8M0_NoSaturate) {
  const std::vector<int64_t> shape{6};
  const std::vector<float> input = {NAN, -1.0f, 4.0f, 0.0f,
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity()};

  std::vector<Float8E8M0> expected;
  expected.reserve(input.size());
  for (float v : input) {
    expected.emplace_back(Float8E8M0(v, false));
  }

  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::False);
}

TEST(CastOpTest, FloatToFloat8E8M0_RoundModeUp) {
  const std::vector<int64_t> shape{4};
  // "up" mode is ceiling: always round up to the next power of 2 when not exact.
  // Exact powers of 2 (mantissa == 0) are unchanged; all others round up.
  //   1.5 (mantissa != 0) -> 2^1 = 2.0 (val=128)
  //   3.0 (mantissa != 0) -> 2^2 = 4.0 (val=129)
  //   1.3 (mantissa != 0) -> 2^1 = 2.0 (val=128)  [ceiling, not round-half-up]
  //   2.5 (mantissa != 0) -> 2^2 = 4.0 (val=129)  [ceiling, not round-half-up]
  const std::vector<float> input = {1.5f, 3.0f, 1.3f, 2.5f};
  const std::vector<Float8E8M0> expected = {
      Float8E8M0(128, Float8E8M0::FromBits()),  // 1.5  -> 2.0
      Float8E8M0(129, Float8E8M0::FromBits()),  // 3.0  -> 4.0
      Float8E8M0(128, Float8E8M0::FromBits()),  // 1.3  -> 2.0
      Float8E8M0(129, Float8E8M0::FromBits()),  // 2.5  -> 4.0
  };
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "up");
}

TEST(CastOpTest, FloatToFloat8E8M0_RoundModeDown) {
  const std::vector<int64_t> shape{4};
  // "down" mode is floor: always truncate to the lower power of 2, never increment.
  // All non-power-of-2 values keep the lower exponent regardless of their fractional part.
  //   1.5 -> 2^0 = 1.0 (val=127)  [floor, not round-half-down]
  //   3.0 -> 2^1 = 2.0 (val=128)  [floor]
  //   1.7 -> 2^0 = 1.0 (val=127)  [floor, not round-half-down -- 1.7 > midpoint but still floors]
  //   2.5 -> 2^1 = 2.0 (val=128)  [floor]
  const std::vector<float> input = {1.5f, 3.0f, 1.7f, 2.5f};
  const std::vector<Float8E8M0> expected = {
      Float8E8M0(127, Float8E8M0::FromBits()),  // 1.5  -> 1.0
      Float8E8M0(128, Float8E8M0::FromBits()),  // 3.0  -> 2.0
      Float8E8M0(127, Float8E8M0::FromBits()),  // 1.7  -> 1.0
      Float8E8M0(128, Float8E8M0::FromBits()),  // 2.5  -> 2.0
  };
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "down");
}

TEST(CastOpTest, FloatToFloat8E8M0_RoundModeNearest) {
  const std::vector<int64_t> shape{4};
  // "nearest" mode: round to nearest power of 2; ties (exactly halfway) round up.
  // Decision threshold: guard bit (bit 22 of mantissa, representing 0.5 of fractional part).
  //   1.5 -> midpoint (mantissa=0x400000) -> round up -> 2^1 = 2.0 (val=128)
  //   3.0 -> midpoint (mantissa=0x400000) -> round up -> 2^2 = 4.0 (val=129)
  //   1.3 -> closer to 1.0 (mantissa=0x266666 < 0x400000) -> 2^0 = 1.0 (val=127)
  //   2.5 -> closer to 2.0 (mantissa=0x200000 < 0x400000) -> 2^1 = 2.0 (val=128)
  // Note: "nearest" differs from "up" for 1.3 and 2.5 (ceiling would give val=128/129).
  const std::vector<float> input = {1.5f, 3.0f, 1.3f, 2.5f};
  const std::vector<Float8E8M0> expected = {
      Float8E8M0(128, Float8E8M0::FromBits()),  // 1.5  -> 2.0 (tie, rounds up)
      Float8E8M0(129, Float8E8M0::FromBits()),  // 3.0  -> 4.0 (tie, rounds up)
      Float8E8M0(127, Float8E8M0::FromBits()),  // 1.3  -> 1.0 (nearer to 1.0)
      Float8E8M0(128, Float8E8M0::FromBits()),  // 2.5  -> 2.0 (nearer to 2.0)
  };
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "nearest");
}

TEST(CastOpTest, FloatToFloat8E8M0_RoundModeNearestSubnormal) {
  const std::vector<int64_t> shape{4};
  const std::vector<float> input = {
      FloatFromBits(0x00400000),  // 2^-127, exact E8M0 minimum
      FloatFromBits(0x00500000),  // below midpoint, differs from round_mode="up"
      FloatFromBits(0x00600000),  // exact midpoint, ties upward
      FloatFromBits(0x007FFFFF),  // largest float32 subnormal
  };
  const std::vector<Float8E8M0> expected = {
      Float8E8M0(0x00, Float8E8M0::FromBits()),
      Float8E8M0(0x00, Float8E8M0::FromBits()),
      Float8E8M0(0x01, Float8E8M0::FromBits()),
      Float8E8M0(0x01, Float8E8M0::FromBits()),
  };
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "nearest");
}

TEST(CastOpTest, Float8E8M0ToFloat) {
  const std::vector<int64_t> shape{4};
  const std::vector<Float8E8M0> input = {
      Float8E8M0(127, Float8E8M0::FromBits()),  // 2^0 = 1.0
      Float8E8M0(128, Float8E8M0::FromBits()),  // 2^1 = 2.0
      Float8E8M0(0, Float8E8M0::FromBits()),    // 2^-127 (smallest)
      Float8E8M0(254, Float8E8M0::FromBits()),  // 2^127 (largest finite)
  };

  std::vector<float> expected;
  expected.reserve(input.size());
  for (const auto& v : input) {
    expected.emplace_back(v.ToFloat());
  }

  TestCastFromFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, MLFloat16ToFloat8E8M0) {
  const std::vector<int64_t> shape{4};
  const std::vector<float> float_values = {1.0f, 2.0f, 4.0f, 0.5f};
  std::vector<MLFloat16> input;
  input.reserve(float_values.size());
  for (float v : float_values) {
    input.emplace_back(MLFloat16(v));
  }

  std::vector<Float8E8M0> expected;
  expected.reserve(float_values.size());
  for (float v : float_values) {
    expected.emplace_back(Float8E8M0(v, true));
  }

  TestCastToFloat8E8M0<MLFloat16>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True);
}

TEST(CastOpTest, DoubleToFloat8E8M0) {
  const std::vector<int64_t> shape{4};
  const std::vector<double> input = {1.0, 2.0, 4.0, 0.5};

  std::vector<Float8E8M0> expected;
  expected.reserve(input.size());
  for (double v : input) {
    expected.emplace_back(Float8E8M0(static_cast<float>(v), true));
  }

  TestCastToFloat8E8M0<double>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True);
}

TEST(CastOpTest, Int32ToFloat8E8M0) {
  const std::vector<int64_t> shape{4};
  const std::vector<int32_t> input = {1, 2, 4, 8};

  std::vector<Float8E8M0> expected;
  expected.reserve(input.size());
  for (int32_t v : input) {
    expected.emplace_back(Float8E8M0(static_cast<float>(v), true));
  }

  TestCastToFloat8E8M0<int32_t>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True);
}

TEST(CastOpTest, Float8E8M0ToDouble) {
  const std::vector<int64_t> shape{3};
  const std::vector<Float8E8M0> input = {
      Float8E8M0(127, Float8E8M0::FromBits()),  // 1.0
      Float8E8M0(128, Float8E8M0::FromBits()),  // 2.0
      Float8E8M0(126, Float8E8M0::FromBits()),  // 0.5
  };

  std::vector<double> expected;
  expected.reserve(input.size());
  for (const auto& v : input) {
    expected.emplace_back(static_cast<double>(v.ToFloat()));
  }

  TestCastFromFloat8E8M0<double>(gsl::make_span(input), gsl::make_span(expected), shape);
}

// Edge-case tests verifying E8M0 conversion table.
// E8M0 format: val 0 = 2^(-127) (E8M0_MIN), val 254 = 2^127 (E8M0_MAX), val 255 = NaN.
// E8M0 cannot represent zero, negative values, or infinity.

TEST(CastOpTest, FloatToFloat8E8M0_SaturateUp_EdgeCases) {
  // With saturate=true and round_mode="up", out-of-range values clamp to the nearest boundary.
  const std::vector<int64_t> shape{8};
  const std::vector<float> input = {
      0.0f,                                     // x = 0 (below E8M0 range)
      -0.0f,                                    // x = -0 (behavior unspecified per spec)
      NAN,                                      // x = NaN
      std::numeric_limits<float>::infinity(),   // x = +Inf
      -std::numeric_limits<float>::infinity(),  // x = -Inf
      3e38f,                                    // x > E8M0_MAX (~1.76 * 2^127)
      1e-39f,                                   // x < E8M0_MIN (positive subnormal)
      -1.0f,                                    // x < 0
  };
  const std::vector<Float8E8M0> expected = {
      Float8E8M0(0, Float8E8M0::FromBits()),    // 0 → E8M0_MIN (val 0)
      Float8E8M0(0, Float8E8M0::FromBits()),    // -0 → E8M0_MIN (val 0, treated same as +0)
      Float8E8M0(255, Float8E8M0::FromBits()),  // NaN → NaN (val 255)
      Float8E8M0(254, Float8E8M0::FromBits()),  // +Inf → E8M0_MAX (val 254)
      Float8E8M0(0, Float8E8M0::FromBits()),    // -Inf → E8M0_MIN (val 0, negative saturated)
      Float8E8M0(254, Float8E8M0::FromBits()),  // 3e38 → E8M0_MAX (val 254, overflow saturated)
      Float8E8M0(0, Float8E8M0::FromBits()),    // 1e-39 -> E8M0_MIN (val 0): ceiling of x < E8M0_MIN is E8M0_MIN
      Float8E8M0(0, Float8E8M0::FromBits()),    // -1 → E8M0_MIN (val 0, negative saturated)
  };
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "up");
}

TEST(CastOpTest, FloatToFloat8E8M0_NonSaturateNearest_EdgeCases) {
  // With saturate=false and round_mode="nearest", all values outside [E8M0_MIN, E8M0_MAX] become NaN.
  const std::vector<int64_t> shape{8};
  const std::vector<float> input = {
      0.0f,                                     // x = 0 (not representable)
      -0.0f,                                    // x = -0 (not representable)
      NAN,                                      // x = NaN
      std::numeric_limits<float>::infinity(),   // x = +Inf (not representable)
      -std::numeric_limits<float>::infinity(),  // x = -Inf (not representable)
      3e38f,                                    // x > E8M0_MAX (not representable)
      1e-39f,                                   // x < E8M0_MIN: subnormal below 2^(-127) -> NaN
      -1.0f,                                    // x < 0 (not representable)
  };
  const std::vector<Float8E8M0> expected(8, Float8E8M0(255, Float8E8M0::FromBits()));  // all -> NaN
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::False, "nearest");
}

TEST(CastOpTest, FloatToFloat8E8M0_NonSaturate_AboveMax) {
  // With saturate=false, any value strictly above E8M0_MAX (2^127) gives NaN,
  // since 2^128 is not representable in E8M0 (val 255 = NaN).
  const std::vector<int64_t> shape{2};
  const std::vector<float> input = {
      2e38f,  // ~1.18 * 2^127, strictly above E8M0_MAX -> NaN
      3e38f,  // ~1.76 * 2^127, strictly above E8M0_MAX -> NaN
  };
  const std::vector<Float8E8M0> expected(2, Float8E8M0(255, Float8E8M0::FromBits()));  // all -> NaN
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::False, "nearest");
}

TEST(CastOpTest, FloatToFloat8E8M0_Saturate_AboveMax) {
  // With saturate=true, values above E8M0_MAX clamp to E8M0_MAX.
  const std::vector<int64_t> shape{2};
  const std::vector<float> input = {2e38f, 3e38f};
  const std::vector<Float8E8M0> expected(2, Float8E8M0(254, Float8E8M0::FromBits()));  // E8M0_MAX
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::True, "up");
}

TEST(CastOpTest, FloatToFloat8E8M0_ExactMax) {
  // Exactly E8M0_MAX (2^127) is representable in all modes.
  const std::vector<int64_t> shape{1};
  // 2^127 = 1.7014118e+38
  const float e8m0_max = Float8E8M0(254, Float8E8M0::FromBits()).ToFloat();
  const std::vector<float> input = {e8m0_max};
  const std::vector<Float8E8M0> expected = {Float8E8M0(254, Float8E8M0::FromBits())};
  TestCastToFloat8E8M0<float>(gsl::make_span(input), gsl::make_span(expected), shape, Saturate::False, "nearest");
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)

TEST(CastOpTest, Float32ToInt64_LargeValues) {
  const std::vector<int64_t> shape{8};
  const std::vector<float> input = {
      1099511627776.0f,     // 2^40, exact in f32
      -1099511627776.0f,    // -2^40, exact in f32
      140737488355328.0f,   // 2^47, exact in f32
      -140737488355328.0f,  // -2^47, exact in f32
      9007199254740992.0f,  // 2^53
      0.0f,                 // zero
      1.5f,                 // truncates to 1
      -1.5f,                // truncates to -1
  };
  const std::vector<int64_t> expected = {
      1099511627776LL,
      -1099511627776LL,
      140737488355328LL,
      -140737488355328LL,
      static_cast<int64_t>(9007199254740992.0f),
      0LL,
      1LL,
      -1LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_PowersOfTwo) {
  const std::vector<int64_t> shape{4};
  const std::vector<float> input = {
      8589934592.0f,   // 2^33
      17179869184.0f,  // 2^34
      4294967296.0f,   // 2^32
      -4294967296.0f,  // -2^32
  };
  const std::vector<int64_t> expected = {
      8589934592LL,
      17179869184LL,
      4294967296LL,
      -4294967296LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_VeryLargeValues) {
  const std::vector<int64_t> shape{8};
  const std::vector<float> input = {
      36028797018963968.0f,     // 2^55, boundary into the exp >= 32 branch
      -36028797018963968.0f,    // -2^55
      1152921504606846976.0f,   // 2^60
      -1152921504606846976.0f,  // -2^60
      4611686018427387904.0f,   // 2^62, largest power of two < 2^63
      -4611686018427387904.0f,  // -2^62
      18014398509481984.0f,     // 2^54, last value still in the exp > 0 branch
      -18014398509481984.0f,    // -2^54
  };
  const std::vector<int64_t> expected = {
      36028797018963968LL,
      -36028797018963968LL,
      1152921504606846976LL,
      -1152921504606846976LL,
      4611686018427387904LL,
      -4611686018427387904LL,
      18014398509481984LL,
      -18014398509481984LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_MantissaBoundary) {
  const std::vector<int64_t> shape{4};
  const std::vector<float> input = {
      8388608.0f,   // 2^23, boundary into the exp == 0 branch
      8388609.0f,   // 2^23 + 1
      -8388609.0f,  // -(2^23 + 1)
      16777215.0f,  // 2^24 - 1, last value before exp > 0
  };
  const std::vector<int64_t> expected = {
      8388608LL,
      8388609LL,
      -8388609LL,
      16777215LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_TinyValues) {
  const std::vector<int64_t> shape{4};
  const std::vector<float> input = {
      0.5f,
      -0.5f,
      0.999f,
      1e-20f,  // far below the exp > -24 cutoff -> 0
  };
  const std::vector<int64_t> expected = {
      0LL,
      0LL,
      0LL,
      0LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float16ToInt64_LargeValues) {
  const std::vector<int64_t> shape{4};
  const std::vector<MLFloat16> input = {
      MLFloat16(32768.0f),   // 2^15, exact in f16
      MLFloat16(-32768.0f),  // -2^15
      MLFloat16(65504.0f),   // f16 max finite
      MLFloat16(-65504.0f),  // f16 min finite
  };
  const std::vector<int64_t> expected = {
      32768LL,
      -32768LL,
      65504LL,
      -65504LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_SpecialValues) {
  // float infinity/NaN -> int64 is undefined behavior in C++.
  // Only test well-defined conversions here (zero, small values).
  const std::vector<int64_t> shape{4};
  const std::vector<float> input = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
  };
  const std::vector<int64_t> expected = {
      0LL,
      0LL,
      1LL,
      -1LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Int32ToInt64) {
  // int32 -> int64 must sign-extend
  const std::vector<int64_t> shape{8};
  const std::vector<int32_t> input = {
      0,
      1,
      -1,
      std::numeric_limits<int32_t>::max(),  // 2^31 - 1
      std::numeric_limits<int32_t>::min(),  // -2^31
      -12345,
      32767,
      -32768,
  };
  const std::vector<int64_t> expected = {
      0LL,
      1LL,
      -1LL,
      2147483647LL,
      -2147483648LL,
      -12345LL,
      32767LL,
      -32768LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, UInt32ToInt64) {
  // uint32 -> int64 must zero-extend
  const std::vector<int64_t> shape{8};
  const std::vector<uint32_t> input = {
      0u,
      1u,
      0x80000000u,                           // 2^31, top bit set
      std::numeric_limits<uint32_t>::max(),  // 2^32 - 1
      123456u,
      0x7FFFFFFFu,  // 2^31 - 1
      0xFFFFFFFEu,  // 2^32 - 2
      0xC0000000u,  // 3 * 2^30
  };
  const std::vector<int64_t> expected = {
      0LL,
      1LL,
      2147483648LL,
      4294967295LL,
      123456LL,
      2147483647LL,
      4294967294LL,
      3221225472LL,
  };
  // QNN EP doesn't correctly handle UINT_32 to INT_64 cast (sign-extends instead of zero-extends).
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::None, false, {kQnnExecutionProvider});
}

TEST(CastOpTest, BoolToInt64) {
  // bool -> int64 zero-extends to 0 or 1.
  const std::vector<int64_t> shape{4};
  const bool bool_input[] = {false, true, true, false};
  const gsl::span<const bool> bool_input_span(bool_input);
  const std::vector<int64_t> expected = {
      0LL,
      1LL,
      1LL,
      0LL,
  };
  TestCastOp(bool_input_span, gsl::make_span(expected), shape);
}

// Regression tests for int64 non-multiple-of-4 element counts.
// size % 4 == 2
TEST(CastOpTest, Float32ToInt64_SizeMod4Eq2) {
  const std::vector<int64_t> shape{2};
  const std::vector<float> input = {1.5f, -2.5f};
  const std::vector<int64_t> expected = {1LL, -2LL};
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Float32ToInt64_SizeMod4Eq2_Large) {
  const std::vector<int64_t> shape{6};
  const std::vector<float> input = {
      4294967296.0f,   // 2^32
      -4294967296.0f,  // -2^32
      8589934592.0f,   // 2^33
      1.0f,
      -1.0f,
      1099511627776.0f,  // 2^40
  };
  const std::vector<int64_t> expected = {
      4294967296LL,
      -4294967296LL,
      8589934592LL,
      1LL,
      -1LL,
      1099511627776LL,
  };
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, Int32ToInt64_SizeMod4Eq2) {
  const std::vector<int64_t> shape{2};
  const std::vector<int32_t> input = {-1, std::numeric_limits<int32_t>::min()};
  const std::vector<int64_t> expected = {-1LL, -2147483648LL};
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

TEST(CastOpTest, UInt32ToInt64_SizeMod4Eq2) {
  const std::vector<int64_t> shape{2};
  const std::vector<uint32_t> input = {0x80000000u, std::numeric_limits<uint32_t>::max()};
  const std::vector<int64_t> expected = {2147483648LL, 4294967295LL};

  // QNN EP doesn't correctly handle UINT_32 to INT_64 cast (sign-extends instead of zero-extends).
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape,
             OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::None, false, {kQnnExecutionProvider});
}

// size % 4 == 1
TEST(CastOpTest, Int32ToInt64_SizeMod4Eq1) {
  const std::vector<int64_t> shape{5};
  const std::vector<int32_t> input = {0, -1, 1, std::numeric_limits<int32_t>::min(), 42};
  const std::vector<int64_t> expected = {0LL, -1LL, 1LL, -2147483648LL, 42LL};
  TestCastOp(gsl::make_span(input), gsl::make_span(expected), shape);
}

}  // namespace test
}  // namespace onnxruntime
