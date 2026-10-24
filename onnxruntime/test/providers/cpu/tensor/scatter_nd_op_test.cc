// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"
#include "core/providers/cpu/tensor/scatter_nd.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// The reductions below all previously threw for MLFloat16 and BFloat16. Index 0 appears twice,
// so slices 0 and 2 both land on output slice 0 and the reduction is applied more than once.
// The values are small powers of two, exact in both half formats, so the expected result does
// not depend on rounding.
template <typename T>
void RunScatterNDHalfReductionTest(const char* reduction, const std::vector<float>& expected_slice0,
                                   const std::vector<float>& expected_slice1) {
  OpTester test("ScatterND", 18);
  test.AddAttribute("reduction", reduction);

  const std::vector<float> data(12, 1.f);
  std::vector<float> updates;
  for (float v : {2.f, 4.f, 8.f}) updates.insert(updates.end(), 6, v);

  std::vector<float> expected;
  expected.insert(expected.end(), expected_slice0.begin(), expected_slice0.end());
  expected.insert(expected.end(), expected_slice1.begin(), expected_slice1.end());

  if constexpr (std::is_same_v<T, MLFloat16>) {
    test.AddInput<MLFloat16>("data", {2, 2, 3}, ToFloat16(data));
    test.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
    test.AddInput<MLFloat16>("updates", {3, 2, 3}, ToFloat16(updates));
    test.AddOutput<MLFloat16>("output", {2, 2, 3}, ToFloat16(expected));
  } else {
    test.AddInput<BFloat16>("data", {2, 2, 3}, ToBFloat16(data));
    test.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
    test.AddInput<BFloat16>("updates", {3, 2, 3}, ToBFloat16(updates));
    test.AddOutput<BFloat16>("output", {2, 2, 3}, ToBFloat16(expected));
  }

  if constexpr (std::is_same_v<T, BFloat16>) {
    // bfloat16 runs on CPU only for now: the CUDA kernel selects its compute type by element
    // size, so it treats bfloat16 as float16 and reduces misread bits
    // (microsoft/onnxruntime#32061). Widen this once that is fixed.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.emplace_back(DefaultCpuExecutionProvider());
    test.ConfigEps(std::move(execution_providers)).RunWithConfig();
  } else {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  }
}

}  // namespace

TEST(ScatterNDOpTest, ScatterND_18_add_MLFloat16) {
  RunScatterNDHalfReductionTest<MLFloat16>("add", std::vector<float>(6, 11.f), std::vector<float>(6, 5.f));
}

TEST(ScatterNDOpTest, ScatterND_18_add_BFloat16) {
  RunScatterNDHalfReductionTest<BFloat16>("add", std::vector<float>(6, 11.f), std::vector<float>(6, 5.f));
}

TEST(ScatterNDOpTest, ScatterND_18_mul_MLFloat16) {
  RunScatterNDHalfReductionTest<MLFloat16>("mul", std::vector<float>(6, 16.f), std::vector<float>(6, 4.f));
}

TEST(ScatterNDOpTest, ScatterND_18_mul_BFloat16) {
  RunScatterNDHalfReductionTest<BFloat16>("mul", std::vector<float>(6, 16.f), std::vector<float>(6, 4.f));
}

TEST(ScatterNDOpTest, ScatterND_18_min_MLFloat16) {
  RunScatterNDHalfReductionTest<MLFloat16>("min", std::vector<float>(6, 1.f), std::vector<float>(6, 1.f));
}

TEST(ScatterNDOpTest, ScatterND_18_min_BFloat16) {
  RunScatterNDHalfReductionTest<BFloat16>("min", std::vector<float>(6, 1.f), std::vector<float>(6, 1.f));
}

TEST(ScatterNDOpTest, ScatterND_18_max_MLFloat16) {
  RunScatterNDHalfReductionTest<MLFloat16>("max", std::vector<float>(6, 8.f), std::vector<float>(6, 4.f));
}

TEST(ScatterNDOpTest, ScatterND_18_max_BFloat16) {
  RunScatterNDHalfReductionTest<BFloat16>("max", std::vector<float>(6, 8.f), std::vector<float>(6, 4.f));
}

// Pins the accumulation precision: the reduction is carried out in half and rounded after every
// update. One ULP at 1024 is 1.0 in binary16, so each 0.25 update rounds away and the total stays
// 1024, where accumulating in float and rounding once at the end would give 1026.
//
// ONNX does not specify the intermediate precision, so this is a property of the CPU kernel
// rather than of the operator, and it runs on CPU only.
TEST(ScatterNDOpTest, ScatterND_18_add_MLFloat16_RoundsAfterEachUpdate) {
  OpTester test("ScatterND", 18);
  test.AddAttribute("reduction", "add");

  test.AddInput<MLFloat16>("data", {2, 1}, ToFloat16({1024.f, 0.f}));
  test.AddInput<int64_t>("indices", {8, 1}, {0, 0, 0, 0, 0, 0, 0, 0});
  test.AddInput<MLFloat16>("updates", {8, 1}, ToFloat16(std::vector<float>(8, 0.25f)));
  test.AddOutput<MLFloat16>("output", {2, 1}, ToFloat16({1024.f, 0.f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultCpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers)).RunWithConfig();
}

TEST(ScatterNDOpTest, ScatterND_scaler_string_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<std::string>("data", {2, 2}, {"h", "h", "o", "z"});
  test1.AddInput<int64_t>("indices", {2}, {0, 1});
  test1.AddInput<std::string>("updates", {}, {"k"});
  test1.AddOutput<std::string>("output", {2, 2}, {"h", "k", "o", "z"});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<std::string>("data", {6}, {"h", "k", "o", "o", "l", "t"});
  test2.AddInput<int64_t>("indices", {1}, {3});
  test2.AddInput<std::string>("updates", {}, {"z"});
  test2.AddOutput<std::string>("output", {6}, {"h", "k", "o", "z", "l", "t"});
  test2.Run();

  OpTester test3("ScatterND", 11);
  test3.AddInput<std::string>("data", {3, 2}, {"h", "k", "o", "z", "l", "z"});
  test3.AddInput<int64_t>("indices", {2}, {2, 1});
  test3.AddInput<std::string>("updates", {}, {"t"});
  test3.AddOutput<std::string>("output", {3, 2}, {"h", "k", "o", "z", "l", "t"});
  test3.Run();
}

TEST(ScatterNDOpTest, ScatterND_matrice_int64_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2, 2}, {1LL, 1LL, 2LL, 2LL});
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 0LL, 1LL, 1LL});
  test.AddInput<int64_t>("updates", {2}, {0LL, 3LL});
  test.AddOutput<int64_t>("output", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_matrice_int64_int64_neg_indices) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2, 2}, {1LL, 1LL, 2LL, 2LL});
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 0LL, -1LL, -1LL});
  test.AddInput<int64_t>("updates", {2}, {0LL, 3LL});
  test.AddOutput<int64_t>("output", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // Output mismatch with OpenVINO EP
}

TEST(ScatterNDOpTest, ScatterND_matrice_string_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<std::string>("data", {2, 2, 2}, {"egg", "dance", "bob", "air", "smart", "terry", "laugh", "kite"});
  test1.AddInput<int64_t>("indices", {2, 1, 2}, {0, 1, 1, 0});
  test1.AddInput<std::string>("updates", {2, 1, 2}, {"air", "bob", "terry", "smart"});
  test1.AddOutput<std::string>("output", {2, 2, 2}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite"});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<std::string>("data", {3, 3}, {"egg", "", "air", "", "terry", "smart", "laugh", "", "hop"});
  test2.AddInput<int64_t>("indices", {3, 2}, {2, 1, 1, 0, 0, 1});
  test2.AddInput<std::string>("updates", {3}, {"kite", "bob", "dance"});
  test2.AddOutput<std::string>("output", {3, 3}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite", "hop"});
  test2.Run();
}

TEST(ScatterNDOpTest, ScatterND_matrice_string_int64_neg_indices) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<std::string>("data", {2, 2, 2}, {"egg", "dance", "bob", "air", "smart", "terry", "laugh", "kite"});
  test1.AddInput<int64_t>("indices", {2, 1, 2}, {0, -1, -1, 0});
  test1.AddInput<std::string>("updates", {2, 1, 2}, {"air", "bob", "terry", "smart"});
  test1.AddOutput<std::string>("output", {2, 2, 2}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite"});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<std::string>("data", {3, 3}, {"egg", "", "air", "", "terry", "smart", "laugh", "", "hop"});
  test2.AddInput<int64_t>("indices", {3, 2}, {-1, -2, 1, 0, 0, -2});
  test2.AddInput<std::string>("updates", {3}, {"kite", "bob", "dance"});
  test2.AddOutput<std::string>("output", {3, 3}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite", "hop"});
  test2.Run();
}

TEST(ScatterNDOpTest, ScatterND_slice_float_int64_t) {
  OpTester test("ScatterND", 11);
  test.AddInput<float>("data", {2, 2}, {0.0f, 0.1f, 0.1f, 0.1f});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddInput<float>("updates", {2, 2}, {0.2f, 0.3f, 0.0f, 0.1f});
  test.AddOutput<float>("output", {2, 2}, {0.0f, 0.1f, 0.2f, 0.3f});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_slice_double_int64_t) {
  OpTester test("ScatterND", 11);
  test.AddInput<double>("data", {2, 2}, {0.0f, 0.1f, 0.1f, 0.1f});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddInput<double>("updates", {2, 2}, {0.2f, 0.3f, 0.0f, 0.1f});
  test.AddOutput<double>("output", {2, 2}, {0.0f, 0.1f, 0.2f, 0.3f});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_3tensor_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<int64_t>("data", {2, 2, 2}, {0LL, 1LL, 1LL, 1LL, 1LL, 1LL, 6LL, 7LL});
  test1.AddInput<int64_t>("indices", {2, 2}, {0LL, 1LL, -1LL, 0LL});
  test1.AddInput<int64_t>("updates", {2, 2}, {2LL, 3LL, 4LL, 5LL});
  test1.AddOutput<int64_t>("output", {2, 2, 2}, {0LL, 1LL, 2LL, 3LL, 4LL, 5LL, 6LL, 7LL});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});

  OpTester test2("ScatterND", 11);
  test2.AddInput<int8_t>("data", {2, 2, 2}, {0, 0, 2, 3, 4, 0, 6, 7});
  test2.AddInput<int64_t>("indices", {2, 3}, {0, 0, 1, -1, 0, -1});
  test2.AddInput<int8_t>("updates", {2}, {1, 5});
  test2.AddOutput<int8_t>("output", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // Exclude TensorRT from INT8 tests

  OpTester test3("ScatterND", 11);
  test3.AddInput<int16_t>("data", {2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3});
  test3.AddInput<int64_t>("indices", {1, 1}, {1LL});
  test3.AddInput<int16_t>("updates", {1, 2, 2}, {4, 5, 6, 7});
  test3.AddOutput<int16_t>("output", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test3.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_batched_index_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2, 2}, {2LL, 3LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2, 1, 2}, {0LL, 0LL, 0LL, 1LL});
  test.AddInput<int64_t>("updates", {2, 1}, {0LL, 1LL});
  test.AddOutput<int64_t>("output", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_batched_index_bool_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<bool>("data", {2, 2}, {false, true, false, true});
  test.AddInput<int64_t>("indices", {2, 1, 2}, {0LL, 0LL, 0LL, 1LL});
  test.AddInput<bool>("updates", {2, 1}, {true, false});
  test.AddOutput<bool>("output", {2, 2}, {true, false, false, true});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_sliced_index_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 0LL, 0LL, 0LL});
  test.AddInput<int64_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddInput<int64_t>("updates", {2, 1, 2}, {2LL, 3LL, 0LL, 1LL});
  test.AddOutput<int64_t>("output", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_sliced_index_string_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<std::string>("data", {2, 2}, {"", "", "", ""});
  test.AddInput<int64_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddInput<std::string>("updates", {2, 1, 2}, {"f", "ghi", "ab", "cde"});
  test.AddOutput<std::string>("output", {2, 2}, {"ab", "cde", "f", "ghi"});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_string_duplicate_indices) {
  OpTester test("ScatterND", 18);
  test.AddInput<std::string>("data", {2}, {"original", "untouched"});
  test.AddInput<int64_t>("indices", {3, 1}, {0, 0, 0});
  test.AddInput<std::string>("updates", {3},
                             {"first value longer than the small string optimization",
                              "second value longer than the small string optimization",
                              "last value longer than the small string optimization"});
  test.AddOutput<std::string>("output", {2},
                              {"last value longer than the small string optimization", "untouched"});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_string_add_duplicate_indices) {
  OpTester test("ScatterND", 18);
  test.AddAttribute("reduction", "add");
  test.AddInput<std::string>("data", {2}, {"base", "untouched"});
  test.AddInput<int64_t>("indices", {3, 1}, {0, 0, 0});
  test.AddInput<std::string>("updates", {3}, {"-first", "-second", "-last"});
  test.AddOutput<std::string>("output", {2}, {"base-first-second-last", "untouched"});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_batched_3tensor_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<uint32_t>("data", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0});
  test1.AddInput<int64_t>("indices", {2, 2, 2}, {0LL, 1LL, 1LL, 0LL, 0LL, 0LL, 1LL, 1LL});
  test1.AddInput<uint32_t>("updates", {2, 2, 2}, {2, 3, 4, 5, 0, 1, 6, 7});
  test1.AddOutput<uint32_t>("output", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<uint32_t>("data", {2, 2, 2}, {0, 0, 2, 0, 4, 0, 0, 7});
  test2.AddInput<int64_t>("indices", {2, 2, 3}, {0, 0, -1, -1, 0, -1, 0, 1, -1, 1, -1, 0});
  test2.AddInput<uint32_t>("updates", {2, 2}, {1, 5, 3, 6});
  test2.AddOutput<uint32_t>("output", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test2.Run();

  OpTester test3("ScatterND", 11);
  test3.AddInput<int64_t>("data", {2, 2, 2}, {1LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL});
  test3.AddInput<int64_t>("indices", {2, 1, 1}, {1, 0});
  test3.AddInput<int64_t>("updates", {2, 1, 2, 2}, {4LL, 5LL, 6LL, 7LL, 0LL, 1LL, 2LL, 3LL});
  test3.AddOutput<int64_t>("output", {2, 2, 2}, {0LL, 1LL, 2LL, 3LL, 4LL, 5LL, 6LL, 7LL});
  test3.Run();
}

TEST(ScatterNDOpTest, ScatterND_18_add) {
  OpTester test1("ScatterND", 18);
  test1.AddAttribute("reduction", "add");
  test1.AddInput<float>("data", {2, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
  // The linter complains if the line is split into multiple lines.
  test1.AddInput<float>("updates", {3, 2, 3}, {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 32768.0f, 65536.0f, 131072.0f, 262144.0f});
  test1.AddOutput<float>("output", {2, 2, 3}, {8194.1f, 16388.1f, 32776.10f, 65552.10f, 131104.1f, 262208.1f, 128.1f, 256.1f, 512.1f, 1024.1f, 2048.1f, 4096.1f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_18_mul) {
  OpTester test1("ScatterND", 18);
  test1.AddAttribute("reduction", "mul");
  test1.AddInput<float>("data", {2, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
  // The linter complains if the line is split into multiple lines.
  test1.AddInput<float>("updates", {3, 2, 3}, {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 32768.0f, 65536.0f, 131072.0f, 262144.0f});
  test1.AddOutput<float>("output", {2, 2, 3}, {1638.4f, 6553.6f, 26214.4f, 104857.6f, 419430.4f, 1677721.625f, 12.8f, 25.6f, 51.2f, 102.4f, 204.8f, 409.6f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_18_mul_long_shape) {
  OpTester test1("ScatterND", 18);
  test1.AddAttribute("reduction", "mul");
  test1.AddInput<float>("data", {2, 2, 3, 1, 1, 1, 1}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
  // The linter complains if the line is split into multiple lines.
  test1.AddInput<float>("updates", {3, 2, 3, 1, 1, 1, 1}, {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 32768.0f, 65536.0f, 131072.0f, 262144.0f});
  test1.AddOutput<float>("output", {2, 2, 3, 1, 1, 1, 1}, {1638.4f, 6553.6f, 26214.4f, 104857.6f, 419430.4f, 1677721.625f, 12.8f, 25.6f, 51.2f, 102.4f, 204.8f, 409.6f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_18_min) {
  OpTester test1("ScatterND", 18);
  test1.AddAttribute("reduction", "min");
  test1.AddInput<float>("data", {2, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
  // The linter complains if the line is split into multiple lines.
  test1.AddInput<float>("updates", {3, 2, 3}, {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 32768.0f, 65536.0f, 131072.0f, 262144.0f});
  test1.AddOutput<float>("output", {2, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_18_max) {
  OpTester test1("ScatterND", 18);
  test1.AddAttribute("reduction", "max");
  test1.AddInput<float>("data", {2, 2, 3}, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
  test1.AddInput<int64_t>("indices", {3, 1}, {0, 1, 0});
  // The linter complains if the line is split into multiple lines.
  test1.AddInput<float>("updates", {3, 2, 3}, {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 32768.0f, 65536.0f, 131072.0f, 262144.0f});
  test1.AddOutput<float>("output", {2, 2, 3}, {8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Test for ScatterND with empty indices - output should be same as input
TEST(ScatterNDOpTest, ScatterND_empty_indices) {
  // Test with float data type and minimal empty case
  OpTester test1("ScatterND", 11);
  test1.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test1.AddInput<int64_t>("indices", {0, 1}, {});                                  // Empty indices tensor - no indices to process
  test1.AddInput<float>("updates", {0, 3}, {});                                    // Empty updates tensor
  test1.AddOutput<float>("output", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // Same as input
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_zero_index_depth_updates_entire_tensor) {
  OpTester test("ScatterND", 18);
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {1, 0}, {});
  test.AddInput<float>("updates", {1, 2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  test.AddOutput<float>("output", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_zero_index_depth_adds_multiple_updates) {
  OpTester test("ScatterND", 18);
  test.AddAttribute("reduction", "add");
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddInput<int64_t>("indices", {2, 0}, {});
  test.AddInput<float>("updates", {2, 2}, {10.0f, 20.0f, 100.0f, 200.0f});
  test.AddOutput<float>("output", {2}, {111.0f, 222.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_zero_index_depth_empty_data) {
  OpTester test("ScatterND", 18);
  test.AddInput<float>("data", {0, 3}, {});
  test.AddInput<int64_t>("indices", {1, 0}, {});
  test.AddInput<float>("updates", {1, 0, 3}, {});
  test.AddOutput<float>("output", {0, 3}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ScatterNDOpTest, ScatterND_nonempty_indices_reject_indexed_zero_dimension) {
  const auto status = scatter_nd_internal::ValidateShapes(
      TensorShape{1, 0, 2}, TensorShape{1, 2}, TensorShape{1, 2});
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(
                                         "indices must be empty when an indexed data dimension has size 0"));
}

}  // namespace test
}  // namespace onnxruntime
