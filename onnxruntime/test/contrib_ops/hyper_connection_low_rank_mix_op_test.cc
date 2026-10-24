// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

template <typename T>
float RoundToType(float value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(value).ToFloat();
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(value).ToFloat();
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

float Sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

struct Attrs {
  float epsilon = 1e-6f;
  float reduction_scale = 0.25f;
  float silu_alpha = 0.25f;
  float gate_div = 4.0f;
  float gate_mul = 2.0f;
};

struct Problem {
  int64_t rows, C, H, R;
  std::vector<float> streams, norm, down, up, inject, prev_out, prev_gate;
};

template <typename T>
Problem MakeProblem(int64_t rows, int64_t C, int64_t H, int64_t R, bool prologue, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int64_t K = C * H;
  auto fill = [&](size_t n, float scale, float shift) {
    std::vector<float> v(n);
    for (auto& x : v) x = RoundToType<T>(normal(gen) * scale + shift);
    return v;
  };
  Problem p{rows, C, H, R, {}, {}, {}, {}, {}, {}, {}};
  p.streams = fill(rows * K, 1.0f, 0.0f);
  p.norm = fill(K, 0.1f, 1.0f);
  p.down = fill(K * R, 1.0f / std::sqrt(static_cast<float>(K)), 0.0f);
  p.up = fill(R * K, 2.0f / std::sqrt(static_cast<float>(R)), 0.0f);
  p.inject = fill(K * C, 1.0f / std::sqrt(static_cast<float>(K)), 0.0f);
  if (prologue) {
    p.prev_out = fill(rows * H, 1.0f, 0.0f);
    p.prev_gate = fill(rows * C, 0.5f, 1.0f);
  }
  return p;
}

struct Expected {
  std::vector<float> pre, gate, streams;
};

// float32 math with the rounding points listed in the schema documentation.
template <typename T>
Expected Reference(const Problem& p, const Attrs& a) {
  const int64_t K = p.C * p.H;
  auto rd = [](float v) { return RoundToType<T>(v); };
  Expected e;
  e.streams = p.streams;
  const bool prologue = !p.prev_out.empty();
  for (int64_t r = 0; r < p.rows; ++r) {
    std::vector<float> hn(K);
    for (int64_t c = 0; c < p.C; ++c) {
      float sum_sq = 0.0f;
      for (int64_t h = 0; h < p.H; ++h) {
        float v = p.streams[r * K + c * p.H + h];
        if (prologue) {
          v = rd(v + p.prev_gate[r * p.C + c] * p.prev_out[r * p.H + h]);
          e.streams[r * K + c * p.H + h] = v;
        }
        sum_sq += v * v;
      }
      const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(p.H) + a.epsilon);
      for (int64_t h = 0; h < p.H; ++h) {
        hn[c * p.H + h] = rd(e.streams[r * K + c * p.H + h] * inv * p.norm[c * p.H + h]);
      }
    }
    std::vector<float> hidden(p.R);
    for (int64_t j = 0; j < p.R; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(hn[k]) * p.down[k * p.R + j];
      const float z = rd(rd(static_cast<float>(acc)) * a.silu_alpha);
      hidden[j] = rd(z * rd(Sigmoid(z)));
    }
    std::vector<float> gates(K);
    for (int64_t k = 0; k < K; ++k) {
      double acc = 0.0;
      for (int64_t j = 0; j < p.R; ++j) acc += static_cast<double>(hidden[j]) * p.up[j * K + k];
      gates[k] = rd(Sigmoid(rd(static_cast<float>(acc))));
    }
    for (int64_t h = 0; h < p.H; ++h) {
      float sum = 0.0f;
      for (int64_t c = 0; c < p.C; ++c) sum += hn[c * p.H + h] * gates[c * p.H + h];
      e.pre.push_back(rd(sum * a.reduction_scale));
    }
    for (int64_t c = 0; c < p.C; ++c) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(hn[k]) * p.inject[k * p.C + c];
      const float t = rd(rd(static_cast<float>(acc)) / a.gate_div);
      e.gate.push_back(rd(rd(Sigmoid(t)) * a.gate_mul));
    }
  }
  return e;
}

template <typename T>
void RunProblem(const Problem& p, const Attrs& a, bool request_streams_out, float tolerance,
                bool constant_weights = true) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  if constexpr (std::is_same_v<T, BFloat16>) {
    if (!CudaHasBF16Support()) {
      GTEST_SKIP() << "BFloat16 requires CUDA compute capability 8.0 or higher";
    }
  }
  const Expected e = Reference<T>(p, a);
  const int64_t K = p.C * p.H;
  const bool prologue = !p.prev_out.empty();

  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", p.C);
  test.AddAttribute<float>("epsilon", a.epsilon);
  test.AddAttribute<float>("reduction_scale", a.reduction_scale);
  test.AddAttribute<float>("silu_alpha", a.silu_alpha);
  test.AddAttribute<float>("gate_div", a.gate_div);
  test.AddAttribute<float>("gate_mul", a.gate_mul);
  test.AddInput<T>("streams", {p.rows, K}, ToTensorType<T>(p.streams));
  test.AddInput<T>("norm_scale", {K}, ToTensorType<T>(p.norm), constant_weights);
  test.AddInput<T>("down_weight", {K, p.R}, ToTensorType<T>(p.down), constant_weights);
  test.AddInput<T>("up_weight", {p.R, K}, ToTensorType<T>(p.up), constant_weights);
  test.AddInput<T>("inject_weight", {K, p.C}, ToTensorType<T>(p.inject), constant_weights);
  if (prologue) {
    test.AddInput<T>("prev_block_output", {p.rows, p.H}, ToTensorType<T>(p.prev_out));
    test.AddInput<T>("prev_post_gate", {p.rows, p.C}, ToTensorType<T>(p.prev_gate));
  } else {
    test.AddOptionalInputEdge<T>();
    test.AddOptionalInputEdge<T>();
  }
  test.AddOutput<T>("pre_output", {p.rows, p.H}, ToTensorType<T>(e.pre));
  test.AddOutput<T>("post_gate", {p.rows, p.C}, ToTensorType<T>(e.gate));
  if (request_streams_out) {
    test.AddOutput<T>("streams_out", {p.rows, K}, ToTensorType<T>(e.streams));
  } else {
    test.AddOptionalOutputEdge<T>();
  }
  test.SetOutputTolerance(tolerance, tolerance);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void RunTest(int64_t rows, int64_t C, int64_t H, int64_t R, bool prologue, bool request_streams_out,
             float tolerance, unsigned seed = 1, bool constant_weights = true) {
  const Attrs a;
  const Problem p = MakeProblem<T>(rows, C, H, R, prologue, seed);
  RunProblem<T>(p, a, request_streams_out, tolerance, constant_weights);
}

constexpr float kFp32Tol = 2e-4f;
constexpr float kFp16Tol = 8e-3f;
constexpr float kBf16Tol = 6e-2f;

}  // namespace

TEST(HyperConnectionLowRankMixTest, FusedFloat16WithPrologue) {
  RunTest<MLFloat16>(2, 4, 16, 8, true, true, kFp16Tol);
  RunTest<MLFloat16>(2, 4, 16, 8, true, true, kFp16Tol, 1, false);
}

TEST(HyperConnectionLowRankMixTest, FusedFloat16WithoutPrologue) {
  RunTest<MLFloat16>(1, 4, 64, 16, false, false, kFp16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedBFloat16WithPrologueWithoutStreamsOut) {
  RunTest<BFloat16>(3, 4, 64, 16, true, false, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedBFloat16TwoBranches) {
  RunTest<BFloat16>(2, 2, 32, 8, true, true, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedMultipleKSlicesAndWeightChunks) {
  RunTest<BFloat16>(2, 4, 1288, 40, true, true, kBf16Tol);
  RunTest<MLFloat16>(1, 4, 2560, 320, false, true, kFp16Tol);
}

TEST(HyperConnectionLowRankMixTest, FusedDecodeShapeRank320) {
  for (int rows : {1, 2, 4}) {
    for (bool prologue : {false, true}) {
      RunTest<BFloat16>(rows, 4, 2560, 320, prologue, true, kBf16Tol, 37);
      RunTest<MLFloat16>(rows, 4, 2560, 320, prologue, true, kFp16Tol, 53);
    }
  }
}

TEST(HyperConnectionLowRankMixTest, UnfusedManyRows) {
  RunTest<MLFloat16>(9, 4, 16, 8, true, true, kFp16Tol);
  RunTest<MLFloat16>(9, 4, 16, 8, true, false, kFp16Tol);
  RunTest<BFloat16>(17, 4, 64, 16, false, true, kBf16Tol);
  RunTest<BFloat16>(17, 4, 64, 16, true, false, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, UnfusedFloat16CancellingDotProducts) {
  // At K = C * H = 20480, cuBLAS selects a split-K kernel for the general-path GEMMs; without
  // CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION, the per-split partial sums exceed FP16 max (65504)
  // and overflow to +/-Inf (producing NaN on reduction) instead of cancelling in FP32 to small finite values.
  for (const int64_t R : {int64_t{6}, int64_t{8}}) {
    const int64_t rows = 9;
    const int64_t C = 4;
    const int64_t H = 5120;
    const int64_t K = C * H;
    const int64_t half_k = K / 2;

    Attrs a;
    a.epsilon = 0.0f;
    Problem p{rows, C, H, R, {}, {}, {}, {}, {}, {}, {}};
    p.streams.assign(static_cast<size_t>(rows * K), 1.0f);
    p.norm.assign(static_cast<size_t>(K), 16.0f);  // hn == 16.0 everywhere
    p.down.assign(static_cast<size_t>(K * R), 0.0f);
    for (int64_t j = 0; j < R; ++j) {
      for (int64_t k = 0; k < half_k - 1; ++k) {
        p.down[static_cast<size_t>(k * R + j)] = 32.0f;
        p.down[static_cast<size_t>((half_k + k) * R + j)] = -32.0f;
      }
      p.down[static_cast<size_t>((half_k - 1) * R + j)] = 0.0625f * static_cast<float>((j % 4) - 1);
    }
    p.up.assign(static_cast<size_t>(R * K), 0.125f);
    p.inject.assign(static_cast<size_t>(K * C), 0.0f);
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t k = 0; k < half_k - 1; ++k) {
        p.inject[static_cast<size_t>(k * C + c)] = 32.0f;
        p.inject[static_cast<size_t>((half_k + k) * C + c)] = -32.0f;
      }
      p.inject[static_cast<size_t>((half_k - 1) * C + c)] = 0.0625f * static_cast<float>(c - 1);
    }

    RunProblem<MLFloat16>(p, a, false, 1e-3f);
  }
}

TEST(HyperConnectionLowRankMixTest, UnfusedUnsupportedShape) {
  RunTest<MLFloat16>(2, 3, 10, 5, true, true, kFp16Tol);
  RunTest<BFloat16>(1, 4, 12, 16, false, false, kBf16Tol);
}

TEST(HyperConnectionLowRankMixTest, Float32) {
  RunTest<float>(1, 4, 64, 16, true, true, kFp32Tol);
  RunTest<float>(5, 2, 24, 8, false, true, kFp32Tol);
}

TEST(HyperConnectionLowRankMixTest, PrologueInputsMustBeProvidedTogether) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  constexpr int64_t C = 2, H = 8, R = 8, K = C * H;
  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", C);
  test.AddInput<float>("streams", {1, K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("norm_scale", {K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("down_weight", {K, R}, std::vector<float>(K * R, 0.1f));
  test.AddInput<float>("up_weight", {R, K}, std::vector<float>(R * K, 0.1f));
  test.AddInput<float>("inject_weight", {K, C}, std::vector<float>(K * C, 0.1f));
  test.AddInput<float>("prev_block_output", {1, H}, std::vector<float>(H, 1.0f));
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("pre_output", {1, H}, std::vector<float>(H, 0.0f));
  test.AddOutput<float>("post_gate", {1, C}, std::vector<float>(C, 0.0f));
  test.AddOptionalOutputEdge<float>();
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be provided together", {}, nullptr, &providers);
}

TEST(HyperConnectionLowRankMixTest, WeightShapeMismatchFails) {
  auto cuda = DefaultCudaExecutionProvider();
  if (!cuda) {
    GTEST_SKIP() << "HyperConnectionLowRankMix is only implemented for CUDA";
  }
  constexpr int64_t C = 2, H = 8, R = 8, K = C * H;
  OpTester test("HyperConnectionLowRankMix", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_branches", C);
  test.AddInput<float>("streams", {1, K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("norm_scale", {K}, std::vector<float>(K, 1.0f));
  test.AddInput<float>("down_weight", {K, R}, std::vector<float>(K * R, 0.1f));
  test.AddInput<float>("up_weight", {R + 1, K}, std::vector<float>((R + 1) * K, 0.1f));
  test.AddInput<float>("inject_weight", {K, C}, std::vector<float>(K * C, 0.1f));
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("pre_output", {1, H}, std::vector<float>(H, 0.0f));
  test.AddOutput<float>("post_gate", {1, C}, std::vector<float>(C, 0.0f));
  test.AddOptionalOutputEdge<float>();
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectFailure, "up_weight must have shape", {}, nullptr, &providers);
}

}  // namespace onnxruntime::test
