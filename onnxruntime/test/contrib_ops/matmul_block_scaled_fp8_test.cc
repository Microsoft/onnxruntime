// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <utility>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"
#include "test/util/include/scoped_env_vars.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#if defined(USE_CUDA)
// CUDA_VERSION comes from cuda.h. Without this include the guard below silently
// evaluates to false and every test in this file is compiled out.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8_tiling.h"
#include "core/providers/cuda/cuda_provider_options.h"
#endif

namespace onnxruntime::test {

#if defined(USE_CUDA) && !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

// MatMulBlockQuantizedFp8Weight: weight-only block-scaled FP8 (E4M3) matmul.
//   A       : [..., K] FP16/BF16 activation
//   B       : [N, K]   FP8 E4M3 weight (one byte per value, not packed)
//   b_scale : [N, ceil(K/block_size)] FP32 per-block weight scales
//   a_scale : optional scalar FP32 -> statically quantize A to FP8 (W8A8)
//   bias    : optional [N]
//   Y       : [..., N]
// Dequantized weight value is fp8_e4m3(B[n, k]) * b_scale[n, k / block_size].

namespace {
std::string CurrentExecutablePath() {
#ifdef _WIN32
  std::string path(MAX_PATH, '\0');
  const DWORD length = GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size()));
  ORT_ENFORCE(length != 0 && length < path.size(), "GetModuleFileNameA failed.");
  path.resize(length);
  return path;
#else
  std::string path(PATH_MAX, '\0');
  const ssize_t length = readlink("/proc/self/exe", path.data(), path.size());
  ORT_ENFORCE(length > 0 && static_cast<size_t>(length) < path.size(), "readlink(/proc/self/exe) failed.");
  path.resize(static_cast<size_t>(length));
  return path;
#endif
}

// Builds a [N, K] FP8 E4M3 weight where every element of row r equals row_value[r].
std::vector<Float8E4M3FN> MakeConstRowWeight(const std::vector<float>& row_value, int64_t k) {
  std::vector<Float8E4M3FN> b(static_cast<size_t>(row_value.size()) * static_cast<size_t>(k));
  for (size_t r = 0; r < row_value.size(); ++r) {
    for (int64_t c = 0; c < k; ++c) {
      b[r * static_cast<size_t>(k) + static_cast<size_t>(c)] = Float8E4M3FN(row_value[r]);
    }
  }
  return b;
}
}  // namespace

template <typename T>
void TestFp8SplitOutputs() {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for split-output FP8 matmul.";
  }
  for (const auto& [m, k] : {std::pair<int64_t, int64_t>{0, 256}, {1, 256}, {2, 256}, {2, 320}, {2, 1088}, {9, 256}, {32, 256}, {33, 256}, {65, 256}, {2, 16}, {2, 30}, {2, 0}}) {
    for (const bool activation_scale : {false, true}) {
      SCOPED_TRACE("M=" + std::to_string(m) + " K=" + std::to_string(k) +
                   " W8A8=" + std::to_string(activation_scale));
      constexpr int64_t n = 17;
      constexpr int64_t block_size = 64;
      const int64_t k_blocks = (k + block_size - 1) / block_size;
      const std::vector<int64_t> widths = m == 2 && k == 256
                                              ? std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 10}
                                              : (m == 33 ? std::vector<int64_t>{17}
                                                         : std::vector<int64_t>{1, 7, 9});
      std::vector<T> a;
      std::vector<Float8E4M3FN> b;
      std::vector<float> scales;
      std::vector<T> bias;
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t i = 0; i < k; ++i) {
          a.emplace_back(static_cast<float>((row + i) % 5 - 2) / 4.0f);
        }
      }
      for (int64_t col = 0; col < n; ++col) {
        bias.emplace_back(static_cast<float>(col - 8) / 8.0f);
        for (int64_t i = 0; i < k; ++i) {
          b.emplace_back(static_cast<float>((col + i) % 3 - 1));
        }
        for (int64_t kb = 0; kb < k_blocks; ++kb) {
          scales.push_back(static_cast<float>(1 + (col + kb) % 2) / 4.0f);
        }
      }
      TensorShapeVector a_shape = m == 1 ? TensorShapeVector{k} : TensorShapeVector{m, k};
      if (m == 9) {
        a_shape = {3, 3, k};
      }
      OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
      test.AddAttribute<int64_t>("block_size", block_size);
      test.AddAttribute<std::vector<int64_t>>("split_sizes", widths);
      test.AddInput<T>("A", a_shape, a);
      test.AddInput<Float8E4M3FN>("B", {n, k}, b);
      test.AddInput<float>("b_scale", {n, k_blocks}, scales);
      if (activation_scale) {
        test.AddInput<float>("a_scale", {}, {0.5f});
      } else {
        test.AddOptionalInputEdge<float>();
      }
      if (activation_scale) {
        test.AddInput<T>("bias", {n}, bias);
      }
      int64_t start = 0;
      for (size_t output = 0; output < widths.size(); ++output) {
        std::vector<T> expected;
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = start; col < start + widths[output]; ++col) {
            float sum = activation_scale ? static_cast<float>(bias[col]) : 0.0f;
            for (int64_t i = 0; i < k; ++i) {
              sum += static_cast<float>(a[row * k + i]) * b[col * k + i].ToFloat() *
                     scales[col * k_blocks + i / block_size];
            }
            expected.emplace_back(sum);
          }
        }
        auto shape = a_shape;
        shape.back() = widths[output];
        test.AddOutput<T>(("Y" + std::to_string(output)).c_str(), shape, expected);
        start += widths[output];
      }
      test.SetOutputTolerance(0.0f, 0.0f);
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, SharedExpertNumericsAndFallbacks) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "SM80+ CUDA device required.";
  const auto round = [](float x) { return static_cast<float>(BFloat16(x)); };
  for (const int64_t m : {0, 1, 2, 3, 65}) {
    for (const int64_t k : {64, 256, 640}) {
      for (const int32_t prefill : {0, 1}) {
        SCOPED_TRACE("M=" + std::to_string(m) + " K=" + std::to_string(k) +
                     " prefill=" + std::to_string(prefill));
        constexpr int64_t n = 17;
        constexpr int64_t gate_width = 32;
        const int64_t block_size = k == 256 ? 128 : k;
        const int64_t k_blocks = k / block_size;
        const std::vector<int64_t> input_shape = m == 3 ? std::vector<int64_t>{1, m, k} : std::vector<int64_t>{m, k};
        const std::vector<int64_t> output_shape = m == 3 ? std::vector<int64_t>{1, m, n} : std::vector<int64_t>{m, n};
        const std::vector<int64_t> activation_shape = m == 3 ? std::vector<int64_t>{1, m, gate_width} : std::vector<int64_t>{m, gate_width};
        std::vector<BFloat16> up(m * k), gate(m * k), prepared(m * k), residual(m * n), expected(m * n);
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            const float g = static_cast<float>(col % 3 - 1);
            const float u = static_cast<float>((col + row) % 5 - 2) / 16.f;
            up[row * k + col] = BFloat16(u);
            gate[row * k + col] = BFloat16(g);
            const float sig = g > 0 ? round(1.f / round(1.f + round(std::exp(-g))))
                                    : round(1.f - round(1.f / round(1.f + round(std::exp(g)))));
            prepared[row * k + col] = BFloat16(round(u * round(g * sig)));
          }
          for (int64_t col = 0; col < n; ++col) {
            residual[row * n + col] = BFloat16(static_cast<float>(col % 5 - 2) / 8.f);
          }
        }
        const auto run = [](OpTester& op) {
          std::vector<std::unique_ptr<IExecutionProvider>> providers;
          providers.push_back(DefaultCudaExecutionProvider());
          op.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
        };
        OpTester silu("SigmoidGateMul", 1, kMSDomain);
        silu.AddAttribute<int64_t>("silu_gate", 1);
        silu.AddAttribute<int64_t>("sigmoid_in_fp32", 0);
        silu.AddInput<BFloat16>("up", input_shape, up);
        silu.AddInput<BFloat16>("gate", input_shape, gate);
        silu.AddOptionalInputEdge<BFloat16>();
        silu.AddInput<int32_t>("prefill", {1}, {prefill});
        silu.AddOutput<BFloat16>("prepared", input_shape, prepared);
        silu.SetOutputTolerance(0.f, 0.f);
        run(silu);

        std::vector<BFloat16> projected(m * n);
        OpTester projection("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
        projection.AddAttribute<int64_t>("block_size", block_size);
        projection.AddInput<BFloat16>("A", input_shape, prepared);
        projection.AddInput<Float8E4M3FN>("B", {n, k}, MakeConstRowWeight(std::vector<float>(n, 1.f), k));
        projection.AddInput<float>("scale", {n, k_blocks}, std::vector<float>(n * k_blocks, .125f));
        projection.AddOptionalInputEdge<float>();
        projection.AddOptionalInputEdge<BFloat16>();
        projection.AddInput<int32_t>("prefill", {1}, {prefill});
        projection.AddOutput<BFloat16>("down", output_shape, projected);
        projection.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
          ASSERT_EQ(fetches.size(), 1u);
          const auto& output = fetches[0].Get<Tensor>();
          projected.assign(output.Data<BFloat16>(), output.Data<BFloat16>() + m * n);
        });
        run(projection);

        std::vector<BFloat16> gate_activation(m * gate_width);
        const std::vector<BFloat16> gate_weight(gate_width, BFloat16(.125f));
        for (int64_t i = 0; i < m * gate_width; ++i) {
          gate_activation[i] = BFloat16(static_cast<float>((i / gate_width) % 3 - 1));
        }
        OpTester post("SigmoidGateMul", 1, kMSDomain);
        post.AddAttribute<int64_t>("broadcast_gate", 1);
        post.AddAttribute<int64_t>("sigmoid_in_fp32", 0);
        post.AddInput<BFloat16>("down", output_shape, projected);
        post.AddInput<BFloat16>("activation", activation_shape, gate_activation);
        post.AddInput<BFloat16>("residual", output_shape, residual);
        post.AddInput<int32_t>("prefill", {1}, {prefill});
        post.AddInput<BFloat16>("weight", {gate_width, 1}, gate_weight);
        post.AddOutput<BFloat16>("Y", output_shape, expected);
        post.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
          ASSERT_EQ(fetches.size(), 1u);
          const auto& output = fetches[0].Get<Tensor>();
          expected.assign(output.Data<BFloat16>(), output.Data<BFloat16>() + m * n);
        });
        run(post);

        OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
        test.AddAttribute<int64_t>("block_size", block_size);
        test.AddAttribute<int64_t>("shared_expert", 1);
        test.AddInput<BFloat16>("up", input_shape, up);
        test.AddInput<Float8E4M3FN>("B", {n, k}, MakeConstRowWeight(std::vector<float>(n, 1.f), k));
        test.AddInput<float>("scale", {n, k_blocks}, std::vector<float>(n * k_blocks, .125f));
        test.AddOptionalInputEdge<float>();
        test.AddOptionalInputEdge<BFloat16>();
        test.AddInput<int32_t>("prefill", {1}, {prefill});
        for (int i = 0; i < 3; ++i) test.AddOptionalInputEdge<BFloat16>();
        test.AddInput<BFloat16>("gate", input_shape, gate);
        test.AddInput<BFloat16>("gate_activation", activation_shape, gate_activation);
        test.AddInput<BFloat16>("gate_weight", {gate_width, 1}, gate_weight);
        test.AddInput<BFloat16>("residual", output_shape, residual);
        test.AddOutput<BFloat16>("Y", output_shape, expected);
        test.SetOutputTolerance(0.f, 0.f);
        run(test);
      }
    }
  }
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, SharedExpertRejectsMissingInputs) {
  OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
  test.AddAttribute<int64_t>("shared_expert", 1);
  test.AddInput<BFloat16>("A", {1, 256}, std::vector<BFloat16>(256));
  test.AddInput<Float8E4M3FN>("B", {1, 256}, MakeConstRowWeight({1.f}, 256));
  test.AddInput<float>("scale", {1, 2}, {1.f, 1.f});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
  test.Run(OpTester::ExpectResult::kExpectFailure, "shared_expert requires all shared inputs");
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, SharedExpertRejectsInvalidContracts) {
  for (int kind = 0; kind < 7; ++kind) {
    SCOPED_TRACE(kind);
    OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
    test.AddAttribute<int64_t>("block_size", 256);
    test.AddAttribute<int64_t>("shared_expert", kind == 5 ? 2 : kind == 6 ? 0
                                                                          : 1);
    if (kind == 4) test.AddAttribute<std::vector<int64_t>>("split_sizes", {8});
    test.AddInput<BFloat16>("A", {2, 256}, std::vector<BFloat16>(512));
    test.AddInput<Float8E4M3FN>("B", {8, 256}, MakeConstRowWeight(std::vector<float>(8, 1.f), 256));
    test.AddInput<float>("scale", {8, 1}, std::vector<float>(8, 1.f));
    if (kind == 3) {
      test.AddInput<float>("a_scale", {1}, {1.f});
    } else {
      test.AddOptionalInputEdge<float>();
    }
    test.AddOptionalInputEdge<BFloat16>();
    test.AddInput<int32_t>("prefill", {1}, {0});
    for (int i = 0; i < 3; ++i) test.AddOptionalInputEdge<BFloat16>();
    test.AddInput<BFloat16>("gate", {kind == 0 ? 3 : 2, 256}, std::vector<BFloat16>((kind == 0 ? 3 : 2) * 256));
    test.AddInput<BFloat16>("activation", {2, 8}, std::vector<BFloat16>(16));
    test.AddInput<BFloat16>("gate_weight", {kind == 1 ? 9 : 8, 1}, std::vector<BFloat16>(kind == 1 ? 9 : 8));
    test.AddInput<BFloat16>("residual", {kind == 2 ? 3 : 2, 8}, std::vector<BFloat16>((kind == 2 ? 3 : 2) * 8));
    test.AddOutput<BFloat16>("Y", {2, 8}, std::vector<BFloat16>(16));
    const std::string error = kind <= 2 ? "shared_expert input shapes are incompatible" : kind <= 4 ? "shared_expert excludes"
                                                                                      : kind == 5   ? "shared_expert must be 0 or 1"
                                                                                                    : "Shared expert inputs require shared_expert=1";
    test.Run(OpTester::ExpectResult::kExpectFailure, error);
  }
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, SplitOutputsFp16) {
  TestFp8SplitOutputs<MLFloat16>();
}

template <typename T>
void TestPerColumnS2ExactOutputs() {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for per-column FP8 matmul.";
  }
  constexpr int64_t m = 2, n = 33;
  for (int64_t k : {320, 640, 2560, 6144}) {
    std::vector<T> a(m * k), bias(n), expected(m * n);
    std::vector<Float8E4M3FN> b(n * k);
    std::vector<float> scales(n);
    for (int64_t i = 0; i < m * k; ++i) a[i] = T(float(i % 5 - 2) / 8.f);
    for (int64_t col = 0; col < n; ++col) {
      scales[col] = float(col % 3 + 1) / 8.f;
      bias[col] = T(float(col % 7 - 3) / 32.f);
      for (int64_t i = 0; i < k; ++i) b[col * k + i] = Float8E4M3FN(float((col + i) % 7 - 3) / 16.f);
      for (int64_t row = 0; row < m; ++row) {
        float sum = 0.f;
        for (int64_t i = 0; i < k; ++i) {
          sum += float(a[row * k + i]) * b[col * k + i].ToFloat();
        }
        expected[row * n + col] = T(sum * scales[col] + float(bias[col]));
      }
    }
    for (bool split : {false, true}) {
      OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
      test.AddAttribute<int64_t>("block_size", k);
      const std::vector<int64_t> widths = split ? std::vector<int64_t>{1, 7, 25}
                                                : std::vector<int64_t>{n};
      if (split) test.AddAttribute<std::vector<int64_t>>("split_sizes", widths);
      test.AddInput<T>("A", {m, k}, a);
      test.AddInput<Float8E4M3FN>("B", {n, k}, b);
      test.AddInput<float>("b_scale", {n, 1}, scales);
      test.AddOptionalInputEdge<float>();
      test.AddInput<T>("bias", {n}, bias);
      int64_t offset = 0;
      for (size_t index = 0; index < widths.size(); ++index) {
        std::vector<T> values;
        for (int64_t row = 0; row < m; ++row) {
          values.insert(values.end(), expected.begin() + row * n + offset,
                        expected.begin() + row * n + offset + widths[index]);
        }
        test.AddOutput<T>(("Y" + std::to_string(index)).c_str(), {m, widths[index]}, values);
        offset += widths[index];
      }
      test.SetOutputTolerance(0.f, 0.f);
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, PerColumnS2ExactOutputsFp16) {
  TestPerColumnS2ExactOutputs<MLFloat16>();
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, PerColumnS2ExactOutputsBf16) {
  TestPerColumnS2ExactOutputs<BFloat16>();
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, SplitOutputsBFloat16) {
  TestFp8SplitOutputs<BFloat16>();
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, RejectsInvalidSplitSizes) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for split-output FP8 matmul.";
  }
  for (const std::vector<int64_t>& widths : {std::vector<int64_t>{}, {0, 17}, {-1, 18}, {8, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 9}}) {
    OpTester test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
    test.AddAttribute<std::vector<int64_t>>("split_sizes", widths);
    test.AddInput<BFloat16>("A", {1, 16}, std::vector<BFloat16>(16));
    test.AddInput<Float8E4M3FN>("B", {17, 16}, std::vector<Float8E4M3FN>(17 * 16));
    test.AddInput<float>("b_scale", {17, 1}, std::vector<float>(17, 1.0f));
    for (size_t i = 0; i < std::max<size_t>(1, widths.size()); ++i) {
      const int64_t width = i < widths.size() ? std::max<int64_t>(0, widths[i]) : 17;
      test.AddOutput<BFloat16>(("Y" + std::to_string(i)).c_str(), {1, width}, std::vector<BFloat16>(width));
    }
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure, "split_sizes", {}, nullptr, &providers);
  }
  OpTester count_test("MatMulBlockQuantizedFp8Weight", 1, kMSDomain);
  count_test.AddAttribute<std::vector<int64_t>>("split_sizes", {8, 9});
  count_test.AddInput<BFloat16>("A", {1, 16}, std::vector<BFloat16>(16));
  count_test.AddInput<Float8E4M3FN>("B", {17, 16}, std::vector<Float8E4M3FN>(17 * 16));
  count_test.AddInput<float>("b_scale", {17, 1}, std::vector<float>(17, 1.0f));
  count_test.AddOutput<BFloat16>("Y", {1, 17}, std::vector<BFloat16>(17));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  count_test.Run(OpTester::ExpectResult::kExpectFailure, "split_sizes", {}, nullptr, &providers);
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCoreKSplitSelection) {
  struct Case {
    int n;
    int m;
    int windows;
    int sm_count;
    int compute_capability_major;
    int compute_capability_minor;
    int expected;
  };
  const Case cases[] = {
      {17408, 1, 80, 48, 12, 1, 32},
      {16384, 8, 80, 48, 12, 1, 32},
      {5120, 8, 128, 48, 12, 1, 32},
      {16369, 1, 80, 48, 12, 1, 32},
      {16368, 1, 80, 48, 12, 1, 8},
      {16384, 1, 79, 48, 12, 1, 8},
      {5105, 1, 128, 48, 12, 1, 32},
      {5104, 1, 128, 48, 12, 1, 16},
      {5120, 1, 127, 48, 12, 1, 16},
      {32768, 1, 80, 48, 12, 1, 32},
      {32769, 1, 80, 48, 12, 1, 32},
      {65536, 1, 80, 48, 12, 1, 32},
      {131072, 1, 80, 48, 12, 1, 32},
      {248320, 1, 80, 48, 12, 1, 32},
      {1024, 4, 80, 48, 12, 1, 16},
      {7168, 8, 80, 48, 12, 1, 16},
      {16384, 9, 80, 48, 12, 1, 8},
      {5120, 9, 128, 48, 12, 1, 16},
      {16384, 16, 80, 48, 12, 1, 8},
      {16384, 1, 80, 47, 12, 1, 8},
      {16384, 1, 80, 49, 12, 1, 8},
      {16384, 1, 80, 48, 12, 0, 8},
      {16384, 1, 80, 48, 9, 0, 8},
  };

  for (const Case& c : cases) {
    SCOPED_TRACE("N = " + std::to_string(c.n) +
                 ", M = " + std::to_string(c.m) +
                 ", windows = " + std::to_string(c.windows) +
                 ", SMs = " + std::to_string(c.sm_count) +
                 ", CC = " + std::to_string(c.compute_capability_major) + "." +
                 std::to_string(c.compute_capability_minor));
    EXPECT_EQ(onnxruntime::contrib::cuda::PickFp8MmaKSplit(
                  c.n, c.m, c.windows, c.sm_count,
                  c.compute_capability_major, c.compute_capability_minor),
              c.expected);
  }
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCoreForcedKSplit32) {
  constexpr const char* kChildProcessVariable = "ORT_FP8_GEMV_KSPLIT_TEST_CHILD";
  const bool is_child_process = !Env::Default().GetEnvironmentVar(kChildProcessVariable).empty();
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_FP8_GEMV_MMA", "1"},
      {"ORT_FP8_GEMV_MAX_M", "32"},
      {"ORT_FP8_GEMV_KSPLIT", "32"},
      {"ORT_FP8_GEMV_MATCH_N", "17"},
      {"ORT_FP8_GEMV_MATCH_K", "2112"},
      {kChildProcessVariable, "1"},
  }};
  if (!is_child_process) {
    const std::string command =
        "\"" + CurrentExecutablePath() +
        "\" --gtest_filter=MatMulBlockQuantizedFp8WeightOpTest.GemvTensorCoreForcedKSplit32 --gtest_color=no";
    ASSERT_EQ(std::system(command.c_str()), 0);
    return;
  }

  constexpr int64_t m = 8;
  constexpr int64_t n = 17;
  constexpr int64_t k = 2112;  // 33 windows exercise a ragged KSplit32 reduction.
  constexpr int64_t block_size = 64;
  constexpr int64_t k_blocks = k / block_size;

  static const float kWeightValues[] = {1.0f, 2.0f, -1.0f};
  static const float kActValues[] = {1.0f, -1.0f, 0.5f, -0.5f};
  std::vector<Float8E4M3FN> b(static_cast<size_t>(n * k));
  std::vector<float> b_ref(static_cast<size_t>(n * k));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t i = 0; i < k; ++i) {
      const float value = kWeightValues[(col + i) % 3];
      b[static_cast<size_t>(col * k + i)] = Float8E4M3FN(value);
      b_ref[static_cast<size_t>(col * k + i)] = value;
    }
  }
  std::vector<float> b_scale(static_cast<size_t>(n * k_blocks));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      b_scale[static_cast<size_t>(col * k_blocks + kb)] =
          static_cast<float>(1 + (col + kb) % 3) / 4.0f;
    }
  }
  std::vector<float> a(static_cast<size_t>(m * k));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t i = 0; i < k; ++i) {
      a[static_cast<size_t>(row * k + i)] = kActValues[(row + i) % 4];
    }
  }
  std::vector<float> expected(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int64_t i = 0; i < k; ++i) {
        acc += a[static_cast<size_t>(row * k + i)] * b_ref[static_cast<size_t>(col * k + i)] *
               b_scale[static_cast<size_t>(col * k_blocks + i / block_size)];
      }
      expected[static_cast<size_t>(row * n + col)] = acc;
    }
  }

  {
    OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
    test.AddInput<Float8E4M3FN>("B", {n, k}, b);
    test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
    test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
    test.SetOutputTolerance(0.005f);
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  {
    OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
    test.AddInput<Float8E4M3FN>("B", {n, k}, b);
    test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
    test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
    test.SetOutputTolerance(0.05f);
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// GEMM path (K not a multiple of 16 forces the cuBLAS dequant path), FP16 activations.
// Weights are constant per row, so Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp8WeightOpTest, WeightOnlyGemmFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 24;  // K % 16 != 0 -> cuBLAS GEMM path

  // Weight row 0 = +1.0, row 1 = +2.0 (both exact in E4M3).
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, 2.0f}, k);
  // One FP32 scale per row (single K block), both 1.0.
  std::vector<float> b_scale = {1.0f, 1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // sum_k A[0, :] = 24, sum_k A[1, :] = 48. Y = {{24, 48}, {48, 96}}.
  std::vector<float> expected = {24.0f, 48.0f, 48.0f, 96.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 128);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, 1}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// GEMM path with non-unit per-block scales, negative weights and bias, BF16 activations/output.
TEST(MatMulBlockQuantizedFp8WeightOpTest, WeightOnlyGemmScalesBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 24;  // K % 16 != 0 -> cuBLAS GEMM path

  // Weight row 0 = +1.0, row 1 = -1.0 (both exact in E4M3).
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, -1.0f}, k);
  // Row 0 scale = 2.0, row 1 scale = 1.0.
  std::vector<float> b_scale = {2.0f, 1.0f};

  std::vector<float> a(m * k, 1.0f);
  // W[0] = 1.0 * 2.0 = 2.0; W[1] = -1.0 * 1.0 = -1.0; sum_k A = 24.
  // Y = {2*24, -1*24} + bias{1, 2} = {49, -22}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {49.0f, -22.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 128);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, 1}, b_scale);
  test.AddOptionalInputEdge<float>();  // a_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Fused decode GEMV fast path (small M, K % 16 == 0, block_size % 16 == 0) with a multi-block K
// and per-block scales, FP16. Weights are constant per row so Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvDecodeMultiBlockFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 32;  // K % 16 == 0 -> GEMV path
  constexpr int64_t block_size = 16;
  constexpr int64_t k_blocks = k / block_size;  // 2 blocks

  // Weight row 0 = +1.0, row 1 = +2.0.
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, 2.0f}, k);
  // Unit FP32 scale for every block of every row.
  std::vector<float> b_scale(n * k_blocks, 1.0f);

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // sum_k A[0, :] = 32, sum_k A[1, :] = 64. Y = {{32, 64}, {64, 128}}.
  std::vector<float> expected = {32.0f, 64.0f, 64.0f, 128.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the optional a_scale (W8A8) activation path: A is statically quantized to FP8 E4M3 and
// dequantized before the matmul. Activation values are exactly representable in E4M3 so the result
// is exact. Uses the GEMV path with per-block scales and bias, BF16.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvW8A8ActivationScaleBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 32;  // K % 16 == 0 -> GEMV path
  constexpr int64_t block_size = 16;
  constexpr int64_t k_blocks = k / block_size;  // 2 blocks

  // Weight row 0 = +1.0, row 1 = -1.0.
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, -1.0f}, k);
  // Row 0 scale = 2.0 for both blocks, row 1 scale = 1.0 for both blocks.
  std::vector<float> b_scale = {2.0f, 2.0f, 1.0f, 1.0f};

  // A = 1.0 everywhere; a_scale = 1.0 -> fp8(1.0) * 1.0 = 1.0 (exact).
  std::vector<float> a(m * k, 1.0f);
  std::vector<float> a_scale = {1.0f};
  // W[0] = 1.0 * 2.0 = 2.0; W[1] = -1.0 * 1.0 = -1.0; sum_k A = 32.
  // Y = {2*32, -1*32} + bias{1, 2} = {65, -30}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {65.0f, -30.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddInput<float>("a_scale", {}, a_scale);
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Covers the M == 1 wide-tile GEMV dispatch, where one warp computes several output columns
// (ColsPerWarp 2 for N >= 4096 and 4 for N >= 8192) and pre-issues loads. N is deliberately not a
// multiple of the block's column tile so the per-column bounds predication is exercised, and the
// weights vary per row so a mis-mapped column cannot pass by accident.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvDecodeWideTilesFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  // n = 4098 selects ColsPerWarp 2, n = 8194 selects ColsPerWarp 4; both leave a ragged tail.
  for (const int64_t n : {4098, 8194}) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 64;  // K % 16 == 0 -> GEMV path; two 32-element K blocks
    constexpr int64_t block_size = 32;
    constexpr int64_t k_blocks = k / block_size;

    // Row r weight value cycles through {1, 2, -1, -2} so each column has a distinct expectation.
    static const float kRowValues[] = {1.0f, 2.0f, -1.0f, -2.0f};
    std::vector<float> row_value(static_cast<size_t>(n));
    for (int64_t r = 0; r < n; ++r) {
      row_value[static_cast<size_t>(r)] = kRowValues[r % 4];
    }
    std::vector<Float8E4M3FN> b = MakeConstRowWeight(row_value, k);
    std::vector<float> b_scale(static_cast<size_t>(n * k_blocks), 1.0f);

    // A = 1.0 everywhere -> sum_k A = k, so Y[0, col] = row_value[col] * k.
    std::vector<float> a(static_cast<size_t>(m * k), 1.0f);
    std::vector<float> expected(static_cast<size_t>(n));
    for (int64_t col = 0; col < n; ++col) {
      expected[static_cast<size_t>(col)] = row_value[static_cast<size_t>(col)] * static_cast<float>(k);
    }

    OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
    test.AddInput<Float8E4M3FN>("B", {n, k}, b);
    test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
    test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
    test.SetOutputTolerance(0.5f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// Covers the M > 1 (speculative decode / MTP verify) GEMV dispatch. There RowsPerWarp is 2 or 4
// and ColsPerWarp is chosen from N (1 / 2 / 4), which selects the hoisted-widening code path in
// the kernel. Every (RowsPerWarp, ColsPerWarp, Unroll) combination is exercised with a ragged N so
// the per-column bounds predication is hit, and both the row and the column vary so a mis-mapped
// row or column cannot pass by accident.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvSpeculativeDecodeTilesFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  // m = 2 -> RowsPerWarp 2, m = 3/4 -> RowsPerWarp 4, m = 5/8 -> RowsPerWarp 8 (m = 3 and m = 5
  // also leave a ragged row tail). n picks ColsPerWarp 1 (n < 2048), 2 and 4; all leave a ragged
  // column tail.
  for (const int64_t m : {2, 3, 4, 5, 8}) {
    for (const int64_t n : {1026, 2050, 4098, 8194}) {
      constexpr int64_t k = 64;  // K % 16 == 0 -> GEMV path; two 32-element K blocks
      constexpr int64_t block_size = 32;
      constexpr int64_t k_blocks = k / block_size;

      static const float kRowValues[] = {1.0f, 2.0f, -1.0f, -2.0f};
      std::vector<float> row_value(static_cast<size_t>(n));
      for (int64_t r = 0; r < n; ++r) {
        row_value[static_cast<size_t>(r)] = kRowValues[r % 4];
      }
      std::vector<Float8E4M3FN> b = MakeConstRowWeight(row_value, k);
      std::vector<float> b_scale(static_cast<size_t>(n * k_blocks), 1.0f);

      // A[row, :] = row + 1 -> Y[row, col] = row_value[col] * k * (row + 1).
      std::vector<float> a(static_cast<size_t>(m * k));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t i = 0; i < k; ++i) {
          a[static_cast<size_t>(row * k + i)] = static_cast<float>(row + 1);
        }
      }
      std::vector<float> expected(static_cast<size_t>(m * n));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
          expected[static_cast<size_t>(row * n + col)] =
              row_value[static_cast<size_t>(col)] * static_cast<float>(k) * static_cast<float>(row + 1);
        }
      }

      OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
      test.AddAttribute<int64_t>("block_size", block_size);
      test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
      test.AddInput<Float8E4M3FN>("B", {n, k}, b);
      test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
      test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
      test.SetOutputTolerance(0.5f);

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

// Covers the SM80+ tensor-core (mma.m16n8k16) GEMV dispatch, taken when K is a multiple of 64
// with at least 4 K windows and block_size is a multiple of 64. It shares no code with the FMA
// kernel: the K axis is permuted, one warp owns a 16-column tile (each lane holding two columns
// eight apart and two rows), and the K split is reduced across warps through shared memory. So
// it needs its own coverage - ragged N and M, several K blocks with per-block scales that vary by
// column, and inputs that vary along every one of M, N and K so a mis-mapped index cannot pass.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCoreTilesFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  struct Case {
    int64_t n, k, block_size;
  };
  // k = 256/320 give 4/5 K windows, which selects the small-KSplit fallback; k = 1024 gives the
  // full KSplit (16 below N = 8192, 8 at or above it). Every N is ragged modulo the 16-column tile.
  const Case cases[] = {{18, 256, 64}, {1026, 320, 64}, {4098, 1024, 256}, {8194, 1024, 512}};
  static const float kWeightValues[] = {1.0f, 2.0f, -1.0f};      // exact in E4M3
  static const float kActValues[] = {1.0f, -1.0f, 0.5f, -0.5f};  // exact in FP16
  for (const Case& c : cases) {
    const int64_t k_blocks = c.k / c.block_size;
    // 1-8 use one row tile, 9-16 two and 17-32 four. M=33 and 64 exercise a partial and full
    // second launch, respectively; odd values leave masked rows that must not be written.
    for (const int64_t m : {1, 3, 4, 8, 9, 16, 17, 32, 33, 64}) {
      // Periods 3 (weight) and 4 (activation) are coprime, so no (row, col) pair sums to zero by
      // symmetry. Even so the signed terms cancel heavily, so the scales are kept in [0.25, 0.75]
      // rather than scaled down: every product is a multiple of 1/8 and the reference stays exact
      // in FP16, while |expected| stays well above the tolerance below (an all-zero output must
      // not pass).
      std::vector<Float8E4M3FN> b(static_cast<size_t>(c.n * c.k));
      std::vector<float> b_ref(static_cast<size_t>(c.n * c.k));
      for (int64_t col = 0; col < c.n; ++col) {
        for (int64_t i = 0; i < c.k; ++i) {
          const float v = kWeightValues[(col + i) % 3];
          b[static_cast<size_t>(col * c.k + i)] = Float8E4M3FN(v);
          b_ref[static_cast<size_t>(col * c.k + i)] = v;
        }
      }
      std::vector<float> b_scale(static_cast<size_t>(c.n * k_blocks));
      for (int64_t col = 0; col < c.n; ++col) {
        for (int64_t kb = 0; kb < k_blocks; ++kb) {
          b_scale[static_cast<size_t>(col * k_blocks + kb)] = static_cast<float>(1 + (col + kb) % 3) / 4.0f;
        }
      }
      std::vector<float> a(static_cast<size_t>(m * c.k));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t i = 0; i < c.k; ++i) {
          a[static_cast<size_t>(row * c.k + i)] = kActValues[(row + i) % 4];
        }
      }
      std::vector<float> expected(static_cast<size_t>(m * c.n));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < c.n; ++col) {
          float acc = 0.0f;
          for (int64_t i = 0; i < c.k; ++i) {
            acc += a[static_cast<size_t>(row * c.k + i)] * b_ref[static_cast<size_t>(col * c.k + i)] *
                   b_scale[static_cast<size_t>(col * k_blocks + i / c.block_size)];
          }
          expected[static_cast<size_t>(row * c.n + col)] = acc;
        }
      }

      OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
      test.AddAttribute<int64_t>("block_size", c.block_size);
      test.AddInput<MLFloat16>("A", {m, c.k}, FloatsToMLFloat16s(a));
      test.AddInput<Float8E4M3FN>("B", {c.n, c.k}, b);
      test.AddInput<float>("b_scale", {c.n, k_blocks}, b_scale);
      test.AddOutput<MLFloat16>("Y", {m, c.n}, FloatsToMLFloat16s(expected));
      test.SetOutputTolerance(0.005f);

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

// BF16 companion to GemvTensorCoreTilesFp16. BF16 has its own mma instruction and its own FP8
// converter (E4M3 -> FP16 -> FP32 -> BF16), neither shared with the FP16 instantiation. Covers
// the multi-tile and split-launch paths with activation QDQ and bias.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCoreTilesBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t n = 1026;  // ragged modulo the 16-column tile
  constexpr int64_t k = 256;
  constexpr int64_t block_size = 64;
  constexpr int64_t k_blocks = k / block_size;
  static const float kWeightValues[] = {1.0f, 2.0f, -1.0f};
  static const float kActValues[] = {1.0f, -1.0f, 0.5f, -0.5f};

  std::vector<Float8E4M3FN> b(static_cast<size_t>(n * k));
  std::vector<float> b_ref(static_cast<size_t>(n * k));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t i = 0; i < k; ++i) {
      const float v = kWeightValues[(col + i) % 3];
      b[static_cast<size_t>(col * k + i)] = Float8E4M3FN(v);
      b_ref[static_cast<size_t>(col * k + i)] = v;
    }
  }
  std::vector<float> b_scale(static_cast<size_t>(n * k_blocks));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      b_scale[static_cast<size_t>(col * k_blocks + kb)] = static_cast<float>(1 + (col + kb) % 3) / 4.0f;
    }
  }
  const std::vector<float> a_scale = {1.0f};
  for (const int64_t m : {4, 9, 33}) {
    std::vector<float> a(static_cast<size_t>(m * k));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t i = 0; i < k; ++i) {
        a[static_cast<size_t>(row * k + i)] = kActValues[(row + i) % 4];
      }
    }
    std::vector<float> bias(static_cast<size_t>(n));
    for (int64_t col = 0; col < n; ++col) {
      bias[static_cast<size_t>(col)] = static_cast<float>(col % 5) - 2.0f;
    }
    std::vector<float> expected(static_cast<size_t>(m * n));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
          acc += a[static_cast<size_t>(row * k + i)] * b_ref[static_cast<size_t>(col * k + i)] *
                 b_scale[static_cast<size_t>(col * k_blocks + i / block_size)];
        }
        expected[static_cast<size_t>(row * n + col)] = acc + bias[static_cast<size_t>(col)];
      }
    }

    OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
    test.AddInput<Float8E4M3FN>("B", {n, k}, b);
    test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
    test.AddInput<float>("a_scale", {}, a_scale);
    test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
    test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
    test.SetOutputTolerance(0.02f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// Selection boundaries for the residency-hinted entry point, at a fixed device size so the
// expectations do not move with the test machine.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCorePinnedResidencyBoundaries) {
  constexpr int sm_count = 132;
  constexpr int compute_capability_major = 9;
  constexpr int compute_capability_minor = 0;
  using onnxruntime::contrib::cuda::Fp8MmaGemvPinsResidency;

  // ceil(N / 16) has to land in (2 * sm_count, 3 * sm_count] == (264, 396].
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 264, 16, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_TRUE(Fp8MmaGemvPinsResidency(
      16 * 264 + 1, 16, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_TRUE(Fp8MmaGemvPinsResidency(
      16 * 396, 16, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 396 + 1, 16, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(16 * 300, 16, 1, sm_count, 8, 6));
  EXPECT_TRUE(Fp8MmaGemvPinsResidency(16 * 300, 16, 1, sm_count, 8, 9));
  // 8-warp blocks regress under any explicit bounds, 32-warp blocks cannot host 3 blocks per SM,
  // and 2 or 4 row tiles spill at the register cap that 3 resident blocks imply.
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 300, 8, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 300, 32, 1, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 300, 16, 2, sm_count, compute_capability_major, compute_capability_minor));
  EXPECT_FALSE(Fp8MmaGemvPinsResidency(
      16 * 300, 16, 4, sm_count, compute_capability_major, compute_capability_minor));
}

// Runs the residency-hinted kernel. It is a second instantiation of the same body, so what is
// under test is the dispatch: nothing above reaches it, because which N selects it depends on the
// device's SM count.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCorePinnedResidency) {
  constexpr const char* kChildProcessVariable = "ORT_FP8_GEMV_PINNED_TEST_CHILD";
  const bool is_child_process = !Env::Default().GetEnvironmentVar(kChildProcessVariable).empty();
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_DISABLE_FUSED_FP8_ACT_QDQ", "0"},
      {"ORT_FP8_GEMV_MMA", "1"},
      {"ORT_FP8_GEMV_MAX_M", "32"},
      {"ORT_FP8_GEMV_KSPLIT", "0"},
      {"ORT_FP8_GEMV_MATCH_N", "0"},
      {"ORT_FP8_GEMV_MATCH_K", "0"},
      {"ORT_FP8_GEMV_DISABLE_GB10_TUNING", "0"},
      {kChildProcessVariable, "1"},
  }};
  if (!is_child_process) {
    const std::string command =
        "\"" + CurrentExecutablePath() +
        "\" --gtest_filter=MatMulBlockQuantizedFp8WeightOpTest.GemvTensorCorePinnedResidency --gtest_color=no";
    ASSERT_EQ(std::system(command.c_str()), 0);
    return;
  }

  cudaDeviceProp device_prop{};
  int device_id = 0;
  ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&device_prop, device_id), cudaSuccess);
  if (device_prop.major < 8 || (device_prop.major == 8 && device_prop.minor < 9)) {
    GTEST_SKIP() << "The residency hint requires native FP8 tensor-core support on SM89 or newer devices.";
  }
  const int sm_count = device_prop.multiProcessorCount;

  constexpr int64_t k = 1024;  // 16 K windows, so KSplit stays at its full 16
  constexpr int64_t block_size = 256;
  constexpr int64_t k_blocks = k / block_size;
  // Narrowest N above 2 blocks per SM. Past N = 8192 the launcher drops to 8 warps per block and
  // stops hinting at all, so a device that large has no shape to test here.
  const int64_t n_pinned = 16 * (2 * sm_count + 1);
  if (n_pinned >= 8192) {
    GTEST_SKIP() << "Device has " << sm_count << " SMs; the hinted window is above N = 8192.";
  }

  static const float kWeightValues[] = {1.0f, 2.0f, -1.0f};      // exact in E4M3
  static const float kActValues[] = {1.0f, -1.0f, 0.5f, -0.5f};  // exact in FP16
  // A ragged width in the same window leaves the last 16-column tile partly out of range.
  for (const int64_t n : {n_pinned, n_pinned + 5}) {
    const int k_split = onnxruntime::contrib::cuda::PickFp8MmaKSplit(
        static_cast<int>(n), 1, static_cast<int>(k / 64), sm_count, device_prop.major, device_prop.minor);
    ASSERT_TRUE(onnxruntime::contrib::cuda::Fp8MmaGemvPinsResidency(
        static_cast<int>(n), k_split, 1, sm_count, device_prop.major, device_prop.minor))
        << "N = " << n << " should take the hinted entry point on this device";

    std::vector<Float8E4M3FN> b(static_cast<size_t>(n * k));
    std::vector<float> b_scale(static_cast<size_t>(n * k_blocks));
    for (int64_t col = 0; col < n; ++col) {
      for (int64_t i = 0; i < k; ++i) {
        b[static_cast<size_t>(col * k + i)] = Float8E4M3FN(kWeightValues[(col + i) % 3]);
      }
      for (int64_t kb = 0; kb < k_blocks; ++kb) {
        b_scale[static_cast<size_t>(col * k_blocks + kb)] = static_cast<float>(1 + (col + kb) % 3) / 4.0f;
      }
    }

    // Only one row tile is hinted, so M stops at 8.
    for (const int64_t m : {1, 3, 8}) {
      SCOPED_TRACE("N = " + std::to_string(n) + ", M = " + std::to_string(m));
      std::vector<float> a(static_cast<size_t>(m * k));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t i = 0; i < k; ++i) {
          a[static_cast<size_t>(row * k + i)] = kActValues[(row + i) % 4];
        }
      }
      std::vector<float> expected(static_cast<size_t>(m * n));
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
          float acc = 0.0f;
          for (int64_t i = 0; i < k; ++i) {
            acc += a[static_cast<size_t>(row * k + i)] * kWeightValues[(col + i) % 3] *
                   b_scale[static_cast<size_t>(col * k_blocks + i / block_size)];
          }
          expected[static_cast<size_t>(row * n + col)] = acc;
        }
      }

      std::vector<float> bias(static_cast<size_t>(n));
      for (int64_t col = 0; col < n; ++col) {
        bias[static_cast<size_t>(col)] = static_cast<float>(col % 5) - 2.0f;
      }
      for (const bool with_optional_inputs : {false, true}) {
        SCOPED_TRACE("with_optional_inputs = " + std::to_string(with_optional_inputs));
        std::vector<float> expected_output = expected;
        if (with_optional_inputs) {
          for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
              expected_output[static_cast<size_t>(row * n + col)] += bias[static_cast<size_t>(col)];
            }
          }
        }
        for (const bool is_bf16 : {false, true}) {
          SCOPED_TRACE("is_bf16 = " + std::to_string(is_bf16));
          OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
          test.AddAttribute<int64_t>("block_size", block_size);
          if (is_bf16) {
            test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
            test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected_output));
          } else {
            test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
            test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected_output));
          }
          test.AddInput<Float8E4M3FN>("B", {n, k}, b);
          test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
          if (with_optional_inputs) {
            test.AddInput<float>("a_scale", {}, {1.0f});
            if (is_bf16) {
              test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
            } else {
              test.AddInput<MLFloat16>("bias", {n}, FloatsToMLFloat16s(bias));
            }
          }
          test.SetOutputTolerance(is_bf16 ? 0.02f : 0.005f);

          std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
          execution_providers.push_back(DefaultCudaExecutionProvider());
          test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
        }
      }
    }
  }
}

// Lane-ownership probe for the tensor-core path.
//
// The tests above sum over the whole K axis, so a wrong lane -> (row, column) mapping could in
// principle still land on a plausible value. Here the activation is one-hot, which collapses
// Y[row, col] to a single weight times its block scale, and the probe offset differs per row.
// Both the weight and the block scale cycle modulo 3 along N, so the two columns a lane owns
// (g and g + 8, which differ by 8 == 2 mod 3) never carry the same value, and the 3 x 3 possible
// weight-times-scale products are all distinct, so the eight rows give eight different values in
// any given column. Any swap of the mma ownership -- output rows 2t / 2t + 1 against columns
// g / g + 8, or the activation row against the output column -- shows up as a mismatch instead of
// cancelling out.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvTensorCoreLaneOwnershipFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 8;   // full mma N extent: output rows 2t and 2t + 1 for every t
  constexpr int64_t n = 40;  // ragged: the last 16-column tile has only its low half in range
  constexpr int64_t k = 256;
  constexpr int64_t block_size = 64;
  constexpr int64_t k_blocks = k / block_size;

  static const float kProbeWeights[3] = {1.0f, -2.0f, 0.5f};  // exact in E4M3
  static const float kProbeScales[3] = {1.0f, 4.0f, 16.0f};   // all 9 products are distinct

  std::vector<Float8E4M3FN> b(static_cast<size_t>(n * k));
  std::vector<float> b_scale(static_cast<size_t>(n * k_blocks));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t i = 0; i < k; ++i) {
      b[static_cast<size_t>(col * k + i)] = Float8E4M3FN(kProbeWeights[(col + i) % 3]);
    }
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      b_scale[static_cast<size_t>(col * k_blocks + kb)] = kProbeScales[(col + kb) % 3];
    }
  }

  // Row `row` probes K offset 23 * row. The eight (weight index, scale-block index) pairs that
  // produces are all different, and each row lands in a different 16-element lane slice of its
  // 64-element K window, so no two rows can be confused for one another.
  std::vector<float> a(static_cast<size_t>(m * k), 0.0f);
  std::vector<float> expected(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    const int64_t i = 23 * row;
    a[static_cast<size_t>(row * k + i)] = 1.0f;
    for (int64_t col = 0; col < n; ++col) {
      expected[static_cast<size_t>(row * n + col)] =
          kProbeWeights[(col + i) % 3] * kProbeScales[(col + i / block_size) % 3];
    }
  }

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// The weight dequantization scratch is capped and tiled over N. ORT_FP8_DEQUANT_SCRATCH_MIB
// shrinks the cap so that a small shape still splits into several tiles: 1 MiB / (4096 * 2 B) is
// 128 rows, so N = 769 takes six full passes and one ragged pass. The CUDA arena is capped at
// 8 MiB: the 3 MiB FP8 weight plus an untiled 6 MiB dequant scratch cannot fit, while the 1 MiB
// tiled scratch can. The weight pattern does not repeat at a tile boundary and every N row carries
// a distinct scale, so a tile that picked up the wrong weight or scale offset, or wrote to the
// wrong output column, changes Y.
TEST(MatMulBlockQuantizedFp8WeightOpTest, WeightDequantScratchTilingFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FP8_DEQUANT_SCRATCH_MIB", "1"}}};

  constexpr int64_t m = 65;  // past the chunked GEMV limit, so the dequant + cuBLAS path runs
  constexpr int64_t n = 769;
  constexpr int64_t k = 4096;
  constexpr int64_t block_size = 128;
  constexpr int64_t k_blocks = k / block_size;
  constexpr int64_t tile_rows = 1024 * 1024 / (k * sizeof(MLFloat16));
  static_assert(tile_rows == 128);
  static_assert((n + tile_rows - 1) / tile_rows == 7);
  static_assert(n % tile_rows == 1);

  std::vector<float> row_weights(static_cast<size_t>(n));
  std::vector<float> b_scale(static_cast<size_t>(n) * static_cast<size_t>(k_blocks));
  for (int64_t row = 0; row < n; ++row) {
    row_weights[static_cast<size_t>(row)] = row % 3 == 0 ? 2.0f : 1.0f;
    const float scale = static_cast<float>(row + 1) / 256.0f;
    for (int64_t blk = 0; blk < k_blocks; ++blk) {
      b_scale[static_cast<size_t>(row * k_blocks + blk)] = scale;
    }
  }
  std::vector<Float8E4M3FN> b = MakeConstRowWeight(row_weights, k);

  std::vector<float> a(static_cast<size_t>(m) * static_cast<size_t>(k), 1.0f);
  std::vector<float> expected(static_cast<size_t>(m) * static_cast<size_t>(n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      expected[static_cast<size_t>(row * n + col)] =
          16.0f * static_cast<float>(col + 1) * row_weights[static_cast<size_t>(col)];
    }
  }

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.gpu_mem_limit = 8 * 1024 * 1024;
  provider_options.arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kSameAsRequested;
  provider_options.use_tf32 = false;
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(CudaExecutionProviderWithOptions(&provider_options));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulBlockQuantizedFp8WeightOpTest, ZeroKWithBiasFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;

  const std::vector<float> bias = {1.0f, -2.0f, 0.5f};
  const std::vector<float> expected = {1.0f, -2.0f, 0.5f, 1.0f, -2.0f, 0.5f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 128);
  test.AddInput<MLFloat16>("A", {m, 0}, std::vector<MLFloat16>{});
  test.AddInput<Float8E4M3FN>("B", {n, 0}, std::vector<Float8E4M3FN>{});
  test.AddInput<float>("b_scale", {n, 0}, std::vector<float>{});
  test.AddOptionalInputEdge<float>();  // a_scale (skipped)
  test.AddInput<MLFloat16>("bias", {n}, FloatsToMLFloat16s(bias));
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // USE_CUDA && !DISABLE_FLOAT8_TYPES && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

}  // namespace onnxruntime::test
