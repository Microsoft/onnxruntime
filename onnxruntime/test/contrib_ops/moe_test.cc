// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "nlohmann/json.hpp"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

// Note: QMoE CPU implementation now always applies softmax normalization to top-k selected experts
// regardless of the normalize_routing_weights parameter value for mathematical correctness.

#ifndef ENABLE_TRAINING

// The CUTLASS SIMT kernel (128x128x8 tile) used on the CUDA MoE path requires minimum
// problem dimensions. For float on SM80+, both hidden_size and inter_size must be >= 128.
// For fp16/bf16 on SM90 TMA WS path, K (hidden_size) must be >= 64.
// Use a conservative threshold here.
static constexpr int kMoEMinCudaDim = 128;

static void RunMoETest(const std::vector<float>& input, const std::vector<float>& router_probs,
                       const std::vector<float>& fc1_experts_weights, const std::vector<float>& fc2_experts_weights,
                       const std::vector<float>& fc3_experts_weights, const std::vector<float>& fc1_experts_bias,
                       const std::vector<float>& fc2_experts_bias, const std::vector<float>& output_data, int num_rows,
                       int num_experts, int hidden_size, int inter_size, std::string activation_type,
                       int normalize_routing_weights = 1, int top_k = 1, bool use_float16 = false) {
  constexpr int min_cuda_arch = 700;

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size, hidden_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size};
  std::vector<int64_t> fc3_experts_weights_dims = fc1_experts_weights_dims;
  std::vector<int64_t> fc1_experts_bias_dims = {num_experts, inter_size};
  std::vector<int64_t> fc2_experts_bias_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  // CUDA path: only run when dimensions are large enough for CUTLASS kernels.
  bool enable_cuda = HasCudaEnvironment(min_cuda_arch) &&
                     hidden_size >= kMoEMinCudaDim && inter_size >= kMoEMinCudaDim;
  if (enable_cuda) {
    OpTester tester("MoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
    tester.AddAttribute<std::string>("activation_type", activation_type);
    tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
      tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
      tester.AddInput<MLFloat16>("fc1_experts_weights", fc1_experts_weights_dims, ToFloat16(fc1_experts_weights));
      if (!fc1_experts_bias.empty()) {
        tester.AddInput<MLFloat16>("fc1_experts_bias", fc1_experts_bias_dims, ToFloat16(fc1_experts_bias));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
      tester.AddInput<MLFloat16>("fc2_experts_weights", fc2_experts_weights_dims, ToFloat16(fc2_experts_weights));
      if (!fc2_experts_bias.empty()) {
        tester.AddInput<MLFloat16>("fc2_experts_bias", fc2_experts_bias_dims, ToFloat16(fc2_experts_bias));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
      if (!fc3_experts_weights.empty()) {
        tester.AddInput<MLFloat16>("fc3_experts_weights", fc3_experts_weights_dims, ToFloat16(fc3_experts_weights));
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
      tester.SetOutputTolerance(0.005f);
    } else {
      tester.AddInput<float>("input", input_dims, input);
      tester.AddInput<float>("router_probs", router_probs_dims, router_probs);
      tester.AddInput<float>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
      if (!fc1_experts_bias.empty()) {
        tester.AddInput<float>("fc1_experts_bias", fc1_experts_bias_dims, fc1_experts_bias);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
      tester.AddInput<float>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
      if (!fc2_experts_bias.empty()) {
        tester.AddInput<float>("fc2_experts_bias", fc2_experts_bias_dims, fc2_experts_bias);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
      if (!fc3_experts_weights.empty()) {
        tester.AddInput<float>("fc3_experts_weights", fc3_experts_weights_dims, fc3_experts_weights);
      }
      tester.AddOutput<float>("output", output_dims, output_data);
      tester.SetOutputTolerance(0.001f);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  // CPU path: run when FC3 is not used (CPU MoE does not support FC3).
  if (fc3_experts_weights.empty()) {
    OpTester cpu_tester("MoE", 1, onnxruntime::kMSDomain);
    cpu_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
    cpu_tester.AddAttribute<std::string>("activation_type", activation_type);
    cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));

    cpu_tester.AddInput<float>("input", input_dims, input);
    cpu_tester.AddInput<float>("router_probs", router_probs_dims, router_probs);
    cpu_tester.AddInput<float>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
    if (!fc1_experts_bias.empty()) {
      cpu_tester.AddInput<float>("fc1_experts_bias", fc1_experts_bias_dims, fc1_experts_bias);
    } else {
      cpu_tester.AddOptionalInputEdge<float>();
    }
    cpu_tester.AddInput<float>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
    if (!fc2_experts_bias.empty()) {
      cpu_tester.AddInput<float>("fc2_experts_bias", fc2_experts_bias_dims, fc2_experts_bias);
    } else {
      cpu_tester.AddOptionalInputEdge<float>();
    }
    cpu_tester.AddOutput<float>("output", output_dims, output_data);
    cpu_tester.SetOutputTolerance(0.001f);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
    cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
    cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
  }
}

// TODO(wy): Add python parity tests that can serve as examples. Need cutlass upgrade to build cutlass extensions to
// add weights preprocesser to onnxruntime_pybind_quant.cc
template <typename ElementType>
static void RunCudaReluSquaredMoETest(int minimum_cuda_arch, int expert_weight_bits = 0) {
  if (!HasCudaEnvironment(minimum_cuda_arch)) {
    GTEST_SKIP() << "CUDA architecture does not support the test element type";
  }

  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 128;
  constexpr int inter_size = 128;
  std::vector<ElementType> input(num_rows * hidden_size);
  std::vector<ElementType> expected(input.size());
  std::vector<ElementType> fc1(num_experts * inter_size * hidden_size, ElementType(0.0f));
  std::vector<ElementType> fc2(num_experts * hidden_size * inter_size, ElementType(0.0f));
  std::vector<ElementType> fc1_bias(num_experts * inter_size);
  std::vector<ElementType> fc2_bias(num_experts * hidden_size);

  for (int expert = 0; expert < num_experts; ++expert) {
    for (int channel = 0; channel < hidden_size; ++channel) {
      const int diagonal = expert * hidden_size * inter_size + channel * hidden_size + channel;
      fc1[diagonal] = ElementType(static_cast<float>(expert + 1));
      fc2[diagonal] = ElementType(1.0f);
      fc1_bias[expert * inter_size + channel] = ElementType(expert == 0 ? 0.5f : -0.5f);
      fc2_bias[expert * hidden_size + channel] = ElementType(expert == 0 ? 0.25f : -0.25f);
    }
  }
  for (int row = 0; row < num_rows; ++row) {
    for (int channel = 0; channel < hidden_size; ++channel) {
      const float value = static_cast<float>((channel + row) % 8 - 3) * 0.5f;
      input[row * hidden_size + channel] = ElementType(value);
      const float first = std::max(0.0f, value + 0.5f);
      const float second = std::max(0.0f, 2.0f * value - 0.5f);
      expected[row * hidden_size + channel] = ElementType(0.5f * (first * first + second * second));
    }
  }

  OpTester tester(expert_weight_bits == 0 ? "MoE" : "QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "relu2");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddInput<ElementType>("input", {num_rows, hidden_size}, input);
  tester.AddInput<ElementType>("router_probs", {num_rows, num_experts},
                               std::vector<ElementType>(num_rows * num_experts, ElementType(0.5f)));
  if (expert_weight_bits == 0) {
    tester.AddInput<ElementType>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, fc1);
    tester.AddInput<ElementType>("fc1_experts_bias", {num_experts, inter_size}, fc1_bias);
    tester.AddInput<ElementType>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, fc2);
    tester.AddInput<ElementType>("fc2_experts_bias", {num_experts, hidden_size}, fc2_bias);
  } else {
    ASSERT_TRUE(expert_weight_bits == 4 || expert_weight_bits == 8);
    const int pack_size = 8 / expert_weight_bits;
    const int zero_point = 1 << (expert_weight_bits - 1);
    const uint8_t packed_zero = expert_weight_bits == 4 ? 0x88 : 0x80;
    std::vector<uint8_t> quantized_fc1(num_experts * hidden_size * inter_size / pack_size, packed_zero);
    std::vector<uint8_t> quantized_fc2(quantized_fc1.size(), packed_zero);
    for (int expert = 0; expert < num_experts; ++expert) {
      for (int channel = 0; channel < hidden_size; ++channel) {
        const int diagonal = expert * hidden_size * inter_size / pack_size +
                             channel * inter_size / pack_size + channel / pack_size;
        const int shift = (channel % pack_size) * expert_weight_bits;
        const uint8_t mask = static_cast<uint8_t>(((1 << expert_weight_bits) - 1) << shift);
        quantized_fc1[diagonal] = static_cast<uint8_t>(
            (quantized_fc1[diagonal] & ~mask) | ((zero_point + expert + 1) << shift));
        quantized_fc2[diagonal] = static_cast<uint8_t>(
            (quantized_fc2[diagonal] & ~mask) | ((zero_point + 1) << shift));
      }
    }
    tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);
    tester.AddAttribute<int64_t>("weights_prepacked", 0);
    tester.AddInput<uint8_t>("fc1_experts_weights",
                             {num_experts, inter_size, hidden_size / pack_size}, quantized_fc1, true);
    tester.AddInput<ElementType>("fc1_scales", {num_experts, inter_size},
                                 std::vector<ElementType>(num_experts * inter_size, ElementType(1.0f)), true);
    tester.AddInput<ElementType>("fc1_experts_bias", {num_experts, inter_size}, fc1_bias);
    tester.AddInput<uint8_t>("fc2_experts_weights",
                             {num_experts, hidden_size, inter_size / pack_size}, quantized_fc2, true);
    tester.AddInput<ElementType>("fc2_scales", {num_experts, hidden_size},
                                 std::vector<ElementType>(num_experts * hidden_size, ElementType(1.0f)), true);
    tester.AddInput<ElementType>("fc2_experts_bias", {num_experts, hidden_size}, fc2_bias);
  }
  tester.AddOutput<ElementType>("output", {num_rows, hidden_size}, expected);
  tester.SetOutputAbsErr("output", 0.001f);
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MoETest, CudaReluSquaredFloat32) {
  RunCudaReluSquaredMoETest<float>(700);
}

TEST(MoETest, CudaReluSquaredFloat16) {
  RunCudaReluSquaredMoETest<MLFloat16>(700);
}

TEST(MoETest, CudaReluSquaredBFloat16) {
  RunCudaReluSquaredMoETest<BFloat16>(800);
}

TEST(MoETest, CudaReluSquaredQuantizedInt4) {
  RunCudaReluSquaredMoETest<MLFloat16>(750, 4);
}

TEST(MoETest, CudaReluSquaredQuantizedInt8) {
  RunCudaReluSquaredMoETest<MLFloat16>(750, 8);
}

static void RunQMoETest(const std::vector<float>& input, const std::vector<float>& router_probs,
                        const std::vector<uint8_t>& fc1_experts_weights,
                        const std::vector<uint8_t>& fc2_experts_weights,
                        const std::vector<uint8_t>& fc3_experts_weights, const std::vector<float>& fc1_scales,
                        const std::vector<float>& fc2_scales, const std::vector<float>& fc3_scales,
                        const std::vector<float>& output_data, int num_rows, int num_experts, int hidden_size,
                        int inter_size, std::string activation_type, int normalize_routing_weights = 1, int top_k = 1, int expert_weight_bits = 4) {
  ORT_ENFORCE(expert_weight_bits == 2 || expert_weight_bits == 4 || expert_weight_bits == 8,
              "Unsupported expert_weight_bits: ", expert_weight_bits);
  constexpr int min_cuda_arch = 700;
  const int64_t pack_size = 8 / expert_weight_bits;

  // Test CUDA execution provider (skip when dimensions are too small for CUTLASS).
  bool enable_cuda = HasCudaEnvironment(min_cuda_arch) &&
                     hidden_size >= kMoEMinCudaDim && inter_size >= kMoEMinCudaDim;
  if (enable_cuda) {
    OpTester cuda_tester("QMoE", 1, onnxruntime::kMSDomain);
    cuda_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
    cuda_tester.AddAttribute<std::string>("activation_type", activation_type);
    cuda_tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));
    cuda_tester.AddAttribute<int64_t>("expert_weight_bits", static_cast<int64_t>(expert_weight_bits));

    std::vector<int64_t> input_dims = {num_rows, hidden_size};
    std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
    std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
    std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
    std::vector<int64_t> fc3_experts_weights_dims = fc1_experts_weights_dims;
    std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
    std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
    std::vector<int64_t> fc3_scales_dims = fc1_scales_dims;
    std::vector<int64_t> output_dims = {num_rows, hidden_size};

    cuda_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
    cuda_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));

    cuda_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
    cuda_tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims, ToFloat16(fc1_scales));
    cuda_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
    cuda_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
    cuda_tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims, ToFloat16(fc2_scales));
    cuda_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias

    // Only add FC3 inputs if fc3_experts_weights is not empty
    if (!fc3_experts_weights.empty()) {
      cuda_tester.AddInput<uint8_t>("fc3_experts_weights", fc3_experts_weights_dims, fc3_experts_weights);
      cuda_tester.AddInput<MLFloat16>("fc3_scales", fc3_scales_dims, ToFloat16(fc3_scales));
    } else {
      cuda_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
      cuda_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_scales
    }
    cuda_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
    cuda_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    cuda_tester.SetOutputTolerance(0.005f);

    std::vector<std::unique_ptr<IExecutionProvider>> cuda_execution_providers;
    cuda_execution_providers.push_back(DefaultCudaExecutionProvider());
    cuda_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_execution_providers);
  }

  // Test CPU execution provider (always available)
  // Skip CPU test if FC3 weights are provided since CPU doesn't support FC3
  if (fc3_experts_weights.empty()) {
    // Ensure CPU EP is available before running CPU tests
    auto cpu_ep = DefaultCpuExecutionProvider();
    if (!cpu_ep) {
      return;  // Skip CPU test if CPU EP is not available
    }

    OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
    cpu_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
    cpu_tester.AddAttribute<std::string>("activation_type", activation_type);
    cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));
    cpu_tester.AddAttribute<int64_t>("expert_weight_bits", static_cast<int64_t>(expert_weight_bits));

    std::vector<int64_t> input_dims = {num_rows, hidden_size};
    std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
    std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
    std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
    std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
    std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
    std::vector<int64_t> output_dims = {num_rows, hidden_size};

    cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
    cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));

    cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
    cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);  // Use float, not MLFloat16
    cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc1_experts_bias
    cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
    cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);  // Use float, not MLFloat16
    cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc2_experts_bias

    // CPU doesn't support FC3, so always skip it
    cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights (skip FC3 for CPU - not implemented)
    cpu_tester.AddOptionalInputEdge<float>();      // fc3_scales (use float, not MLFloat16)
    cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
    cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    cpu_tester.SetOutputTolerance(0.01f);  // Slightly higher tolerance for CPU vs CUDA differences

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
    cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
    cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
  }

#if defined(USE_WEBGPU)
  // Test WebGPU execution provider (skip if FC3 weights provided — not yet supported)
  if (fc3_experts_weights.empty()) {
    OpTester webgpu_tester("QMoE", 1, onnxruntime::kMSDomain);
    webgpu_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
    webgpu_tester.AddAttribute<std::string>("activation_type", activation_type);
    webgpu_tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));
    webgpu_tester.AddAttribute<int64_t>("expert_weight_bits", static_cast<int64_t>(expert_weight_bits));

    std::vector<int64_t> input_dims = {num_rows, hidden_size};
    std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
    std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
    std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
    std::vector<int64_t> fc3_experts_weights_dims = fc1_experts_weights_dims;
    std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
    std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
    std::vector<int64_t> fc3_scales_dims = fc1_scales_dims;
    std::vector<int64_t> output_dims = {num_rows, hidden_size};

    webgpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
    webgpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));

    webgpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
    webgpu_tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims, ToFloat16(fc1_scales));
    webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
    webgpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
    webgpu_tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims, ToFloat16(fc2_scales));
    webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias

    if (!fc3_experts_weights.empty()) {
      webgpu_tester.AddInput<uint8_t>("fc3_experts_weights", fc3_experts_weights_dims, fc3_experts_weights);
      webgpu_tester.AddInput<MLFloat16>("fc3_scales", fc3_scales_dims, ToFloat16(fc3_scales));
    } else {
      webgpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
      webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_scales
    }
    webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
    webgpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    webgpu_tester.SetOutputTolerance(0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> webgpu_execution_providers;
    webgpu_execution_providers.push_back(DefaultWebGpuExecutionProvider());
    webgpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &webgpu_execution_providers);
  }
#endif
}

TEST(MoETest, MoETest_Gelu) {
  int num_rows = 4;
  int num_experts = 4;
  int hidden_size = 8;
  int inter_size = 16;

  const std::vector<float> input = {
      -1.1200173f, -0.45884353f, -1.2929888f, 1.0784022f, 0.116372705f, 0.26902613f, -1.8818876f, -0.5457026f,
      0.22222236f, -0.28868636f, 0.6692926f, 1.4944887f, 0.02431708f, -0.49781424f, 0.7378293f, 1.276276f,
      -0.15469065f, -0.28456813f, -0.6296439f, -0.24855971f, 0.80565417f, -1.1018785f, -0.74082595f, 0.82407707f,
      -0.95033455f, 0.659333f, -0.68629056f, -0.2916592f, 1.869919f, -1.1053563f, -0.14417848f, -0.34625578f};
  const std::vector<float> router_probs = {
      -0.84837115f, 0.100507565f, -0.10548311f, 0.40957215f, 1.0159845f, 0.26919764f, 0.021741152f, -0.34184334f,
      -0.71324956f, 0.29018253f, -0.18227568f, 0.31496462f, -0.48426327f, -1.006643f, -0.100081146f, -0.07692295f};
  const std::vector<float> fc1_experts_weights = {
      0.14731085f, 0.6808038f, 0.066695035f, 0.66509604f, 0.0044258237f, 0.02726549f, 0.3856619f, 0.5981904f,
      0.52229995f, 0.563313f, 0.74768895f, 0.875114f, 0.725697f, 0.608024f, 0.47776443f, 0.12875527f,
      0.14753294f, 0.496278f, 0.14385962f, 0.33904207f, 0.25986814f, 0.21940875f, 0.1954779f, 0.5832493f,
      0.22475791f, 0.40115923f, 0.35806787f, 0.50080043f, 0.16632986f, 0.054212093f, 0.66910046f, 0.7129646f,
      0.20864725f, 0.5627332f, 0.33224183f, 0.7574118f, 0.21194929f, 0.93843824f, 0.65808296f, 0.6979155f,
      0.6708725f, 0.38582766f, 0.4259563f, 0.016453922f, 0.787478f, 0.1752944f, 0.4896857f, 0.43706065f,
      0.20204341f, 0.49648678f, 0.50546914f, 0.8614903f, 0.76478684f, 0.44311923f, 0.38754892f, 0.09010619f,
      0.4890914f, 0.5637965f, 0.91240376f, 0.08653879f, 0.8837609f, 0.64324677f, 0.1917851f, 0.42292297f,
      0.52103406f, 0.10889745f, 0.5624194f, 0.50689125f, 0.68136156f, 0.51592916f, 0.8457724f, 0.67365384f,
      0.8223115f, 0.23793429f, 0.9478464f, 0.41499162f, 0.33302015f, 0.16355914f, 0.12778795f, 0.31756145f,
      0.122039974f, 0.90374637f, 0.8058562f, 0.23666352f, 0.36027592f, 0.09583914f, 0.70483273f, 0.68979055f,
      0.15674388f, 0.09422666f, 0.18389302f, 0.5660855f, 0.647715f, 0.8985412f, 0.33187324f, 0.8329813f,
      0.20966923f, 0.4640969f, 0.72425205f, 0.91345936f, 0.91101736f, 0.58141935f, 0.258766f, 0.2389242f,
      0.8499667f, 0.99461937f, 0.14655197f, 0.35384023f, 0.6359461f, 0.91481227f, 0.58982253f, 0.5049309f,
      0.3202675f, 0.6806185f, 0.28808743f, 0.20315295f, 0.26342732f, 0.3323797f, 0.24027151f, 0.7067495f,
      0.92174435f, 0.5141565f, 0.64706135f, 0.31508058f, 0.2649613f, 0.6472777f, 0.6152024f, 0.5391889f,
      0.54176575f, 0.8775838f, 0.4100297f, 0.15878445f, 0.6654094f, 0.81974447f, 0.36427742f, 0.92438906f,
      0.5624327f, 0.9998074f, 0.69465625f, 0.4192087f, 0.9998382f, 0.537167f, 0.4035356f, 0.5327942f,
      0.10692614f, 0.93719745f, 0.5888109f, 0.17528385f, 0.75933146f, 0.26393235f, 0.5478503f, 0.16675615f,
      0.5392941f, 0.8873769f, 0.7127341f, 0.84719825f, 0.81082416f, 0.9595239f, 0.9624148f, 0.32085752f,
      0.8462349f, 0.38537037f, 0.33008623f, 0.121996105f, 0.32500392f, 0.70447034f, 0.5267702f, 0.60918206f,
      0.9505569f, 0.32452917f, 0.7437857f, 0.25604928f, 0.73991376f, 0.12042731f, 0.19128f, 0.11884099f,
      0.79387546f, 0.9105244f, 0.15076452f, 0.016954303f, 0.5574533f, 0.97854143f, 0.52562714f, 0.74840516f,
      0.5670015f, 0.7801898f, 0.6129275f, 0.21612722f, 0.38059133f, 0.8796869f, 0.7397436f, 0.04606521f,
      0.7335071f, 0.19911051f, 0.16170406f, 0.91123873f, 0.21814507f, 0.31775457f, 0.7480201f, 0.01935333f,
      0.25676018f, 0.9495086f, 0.006731212f, 0.90938f, 0.21944171f, 0.78107727f, 0.04303074f, 0.014169693f,
      0.08565581f, 0.7415793f, 0.09847212f, 0.85791886f, 0.11525959f, 0.21590549f, 0.41052878f, 0.39856833f,
      0.07003945f, 0.77256775f, 0.89473504f, 0.88606364f, 0.83566517f, 0.42164284f, 0.12842774f, 0.83621645f,
      0.99880487f, 0.18661183f, 0.7705178f, 0.94459325f, 0.8554656f, 0.9245506f, 0.2866572f, 0.026760519f,
      0.8173947f, 0.6434499f, 0.96910787f, 0.3719685f, 0.44309366f, 0.52065957f, 0.6801467f, 0.91559356f,
      0.15438312f, 0.32471877f, 0.9005606f, 0.72000104f, 0.210657f, 0.14639091f, 0.1449349f, 0.29998857f,
      0.6956213f, 0.8906783f, 0.053477287f, 0.9454652f, 0.88645273f, 0.33288354f, 0.68586344f, 0.64644206f,
      0.52280146f, 0.6593821f, 0.14703393f, 0.08102721f, 0.3612727f, 0.49863482f, 0.56869274f, 0.013192713f,
      0.049140453f, 0.7684034f, 0.25234455f, 0.22947109f, 0.31360215f, 0.25775224f, 0.9111291f, 0.20481992f,
      0.9146645f, 0.56965464f, 0.08815551f, 0.94240886f, 0.6258745f, 0.99914503f, 0.16196036f, 0.32954985f,
      0.7692217f, 0.16545832f, 0.76092035f, 0.9572636f, 0.6773468f, 0.9883351f, 0.5232172f, 0.7516082f,
      0.99699783f, 0.11234015f, 0.44905245f, 0.036860168f, 0.25571418f, 0.122906685f, 0.31561613f, 0.17643315f,
      0.7526061f, 0.3457417f, 0.88480055f, 0.85264915f, 0.54419917f, 0.09466505f, 0.99065316f, 0.9714598f,
      0.1699655f, 0.7194791f, 0.8094361f, 0.7505796f, 0.78976786f, 0.12100351f, 0.025618374f, 0.38863534f,
      0.9172919f, 0.9931982f, 0.7766713f, 0.79595923f, 0.45025164f, 0.49758863f, 0.0206694f, 0.410219f,
      0.5268722f, 0.7875145f, 0.51607805f, 0.9232646f, 0.65216696f, 0.37254804f, 0.9926925f, 0.891779f,
      0.73710823f, 0.44369537f, 0.345411f, 0.23052484f, 0.3794065f, 0.17272717f, 0.18365502f, 0.75130385f,
      0.09908545f, 0.6753082f, 0.39128417f, 0.6578879f, 0.6752498f, 0.32066393f, 0.5958617f, 0.92406017f,
      0.35618675f, 0.009468555f, 0.5664503f, 0.7046166f, 0.1378029f, 0.59446543f, 0.45684695f, 0.7892222f,
      0.009061217f, 0.07294935f, 0.74785477f, 0.35225332f, 0.2059856f, 0.23875463f, 0.3946715f, 0.34832305f,
      0.30525374f, 0.73330396f, 0.14970505f, 0.66732657f, 0.24620473f, 0.61079127f, 0.3883261f, 0.1682638f,
      0.6078656f, 0.2167924f, 0.91963893f, 0.3561433f, 0.95950544f, 0.38534206f, 0.8177203f, 0.46279848f,
      0.10741913f, 0.74054784f, 0.44563496f, 0.80913067f, 0.36545795f, 0.25771832f, 0.5238985f, 0.9138188f,
      0.3321901f, 0.048188448f, 0.25373894f, 0.3576542f, 0.20394331f, 0.5808252f, 0.4121768f, 0.7477461f,
      0.036315024f, 0.9668448f, 0.45626426f, 0.15941626f, 0.57173324f, 0.96037793f, 0.6620425f, 0.6495659f,
      0.7049642f, 0.7210276f, 0.45194024f, 0.9383765f, 0.6595957f, 0.26129955f, 0.6961637f, 0.70084965f,
      0.9867357f, 0.7179537f, 0.11051077f, 0.41742706f, 0.53540194f, 0.6788141f, 0.88390845f, 0.19156617f,
      0.3576584f, 0.06738949f, 0.91683406f, 0.044237554f, 0.17582512f, 0.37464648f, 0.425507f, 0.8217978f,
      0.08598822f, 0.96300787f, 0.27943915f, 0.46856833f, 0.9781642f, 0.39156884f, 0.48017246f, 0.9735775f,
      0.046470165f, 0.97367156f, 0.67735744f, 0.81400645f, 0.20925027f, 0.8676517f, 0.8424056f, 0.5433857f,
      0.6252997f, 0.95143014f, 0.9348918f, 0.6299002f, 0.9112503f, 0.112507045f, 0.36471343f, 0.032975793f,
      0.46214014f, 0.07820749f, 0.7521582f, 0.6581022f, 0.10224587f, 0.55310667f, 0.9383168f, 0.85099494f,
      0.24750638f, 0.3113383f, 0.57078993f, 0.5464366f, 0.37972575f, 0.9702046f, 0.16709393f, 0.12927437f,
      0.60106593f, 0.1561181f, 0.9254285f, 0.68640935f, 0.7719844f, 0.4312939f, 0.44589508f, 0.61493605f,
      0.6898794f, 0.9734828f, 0.5672131f, 0.378174f, 0.29570967f, 0.88821906f, 0.47314453f, 0.5726589f,
      0.8976595f, 0.28516f, 0.2686717f, 0.3010999f, 0.9200215f, 0.3460216f, 0.72310495f, 0.26598173f,
      0.8881911f, 0.27172273f, 0.97299975f, 0.032645762f, 0.15592176f, 0.9024811f, 0.84183806f, 0.6740978f,
      0.42515814f, 0.76195645f, 0.61834025f, 0.12333155f, 0.080114245f, 0.016334832f, 0.4207481f, 0.052783668f,
      0.059116423f, 0.26870382f, 0.012159586f, 0.71670127f, 0.27454042f, 0.42793816f, 0.0857597f, 0.61387974f};
  const std::vector<float> fc2_experts_weights = {
      0.18302453f, 0.06460017f, 0.9055086f, 0.80260223f, 0.43536508f, 0.73533344f, 0.87621754f,
      0.57699674f, 0.11638266f, 0.9980005f, 0.22312081f, 0.9744255f, 0.72243124f, 0.5890645f,
      0.30296493f, 0.4269635f, 0.44593316f, 0.04124081f, 0.12955844f, 0.24076664f, 0.13430476f,
      0.9296606f, 0.9212578f, 0.7996927f, 0.3538614f, 0.20361924f, 0.29026228f, 0.632942f,
      0.2862816f, 0.55906135f, 0.9153206f, 0.41526443f, 0.5643144f, 0.98830533f, 0.4198916f,
      0.40261286f, 0.64640516f, 0.83163047f, 0.54665035f, 0.04920423f, 0.66399014f, 0.99917024f,
      0.35380626f, 0.3169638f, 0.89860183f, 0.26557416f, 0.46709478f, 0.4968937f, 0.9259722f,
      0.37530023f, 0.20413119f, 0.89688545f, 0.14449131f, 0.23771948f, 0.6135615f, 0.95198375f,
      0.06195748f, 0.020154774f, 0.012922287f, 0.29422665f, 0.14915991f, 0.32725257f, 0.72685635f,
      0.3111158f, 0.26143986f, 0.5249744f, 0.21432412f, 0.38691485f, 0.10324633f, 0.44519007f,
      0.28353918f, 0.67986554f, 0.7740991f, 0.054515004f, 0.52598435f, 0.18009722f, 0.5014211f,
      0.635427f, 0.9951532f, 0.61719346f, 0.82031804f, 0.63555616f, 0.6186035f, 0.5455279f,
      0.5304596f, 0.34265757f, 0.8774212f, 0.14969361f, 0.7602738f, 0.80709815f, 0.58842945f,
      0.15339059f, 0.94945997f, 0.1523174f, 0.34716582f, 0.5188402f, 0.4364831f, 0.8398661f,
      0.969324f, 0.15048373f, 0.8964218f, 0.09793854f, 0.29194576f, 0.39229625f, 0.81010276f,
      0.55225646f, 0.4995767f, 0.41947508f, 0.99719256f, 0.58249784f, 0.7717041f, 0.8169449f,
      0.2625361f, 0.92673707f, 0.099448025f, 0.92562044f, 0.358508f, 0.5002066f, 0.1526736f,
      0.93378997f, 0.18122643f, 0.52884465f, 0.66146517f, 0.4115672f, 0.21036887f, 0.71636236f,
      0.3569854f, 0.39879733f, 0.5501401f, 0.40959156f, 0.5230092f, 0.5080162f, 0.9921749f,
      7.969141e-05f, 0.2996632f, 0.5525096f, 0.05675614f, 0.851531f, 0.58067596f, 0.7438458f,
      0.8786539f, 0.8324149f, 0.30341f, 0.09121573f, 0.31400484f, 0.82673794f, 0.59784645f,
      0.27005446f, 0.28632385f, 0.52560204f, 0.03917718f, 0.9653573f, 0.46825224f, 0.43548512f,
      0.62078035f, 0.0006400347f, 0.6568952f, 0.078864336f, 0.42088854f, 0.100299895f, 0.08127314f,
      0.52832156f, 0.5181678f, 0.8300168f, 0.45993322f, 0.48518902f, 0.13617045f, 0.22095704f,
      0.6667756f, 0.111976564f, 0.93922615f, 0.038565338f, 0.99439347f, 0.8619068f, 0.02608782f,
      0.79918617f, 0.7023336f, 0.68886834f, 0.8461837f, 0.034480453f, 0.28104752f, 0.5752184f,
      0.96928054f, 0.94479716f, 0.6499472f, 0.31438643f, 0.6821157f, 0.8066132f, 0.0070040226f,
      0.027926445f, 0.02856338f, 0.75412107f, 0.56397897f, 0.7178481f, 0.28890234f, 0.8031663f,
      0.43097937f, 0.8831251f, 0.79153055f, 0.63141924f, 0.91840404f, 0.44386774f, 0.061530292f,
      0.81982285f, 0.018661916f, 0.61170083f, 0.69368154f, 0.56660175f, 0.29975814f, 0.37731683f,
      0.23421508f, 0.9946784f, 0.60866946f, 0.9860047f, 0.76857555f, 0.8517718f, 0.99132854f,
      0.22980672f, 0.13705963f, 0.047644496f, 0.838051f, 0.17248261f, 0.28836077f, 0.22687018f,
      0.33094752f, 0.71633244f, 0.7178768f, 0.60117006f, 0.5667807f, 0.20335877f, 0.7778716f,
      0.28580618f, 0.9548785f, 0.75558543f, 0.7203748f, 0.68979263f, 0.94391155f, 0.30140227f,
      0.19580519f, 0.6663505f, 0.63076067f, 0.86896664f, 0.06484294f, 0.46668667f, 0.40556252f,
      0.46882278f, 0.102760494f, 0.73050886f, 0.8356961f, 0.6755136f, 0.5681396f, 0.109577894f,
      0.80634f, 0.5885344f, 0.30270886f, 0.5224626f, 0.1426636f, 0.61006856f, 0.7554362f,
      0.5708561f, 0.5937015f, 0.45749134f, 0.37791175f, 0.93248576f, 0.464967f, 0.46046263f,
      0.18158185f, 0.018330276f, 0.9106033f, 0.2911517f, 0.74790317f, 0.69202286f, 0.6075949f,
      0.85149264f, 0.94095474f, 0.17537653f, 0.33921933f, 0.110060334f, 0.36293298f, 0.46352726f,
      0.02775067f, 0.34128642f, 0.9567719f, 0.8162452f, 0.6936434f, 0.17896658f, 0.18262434f,
      0.13349658f, 0.95527196f, 0.6541582f, 0.002533555f, 0.12373549f, 0.8386668f, 0.49692506f,
      0.5846437f, 0.2106899f, 0.21170747f, 0.47471708f, 0.3711096f, 0.40584308f, 0.92123467f,
      0.44139355f, 0.98912156f, 0.6823429f, 0.008196831f, 0.6082054f, 0.77075267f, 0.4575745f,
      0.52732253f, 0.084977865f, 0.5830078f, 0.6437313f, 0.7027664f, 0.59101045f, 0.21356237f,
      0.27901447f, 0.32865065f, 0.029210031f, 0.0028281808f, 0.44412827f, 0.0028039217f, 0.998863f,
      0.35483408f, 0.33203715f, 0.59384584f, 0.3924346f, 0.75354826f, 0.78473866f, 0.33041215f,
      0.11807448f, 0.74022657f, 0.81752694f, 0.22417879f, 0.6959876f, 0.021027386f, 0.7004118f,
      0.5640625f, 0.82573634f, 0.010389388f, 0.79243237f, 0.3391887f, 0.7897731f, 0.15555972f,
      0.18762505f, 0.8451145f, 0.38050216f, 0.57453954f, 0.86466515f, 0.9299548f, 0.8687942f,
      0.38184917f, 0.8531213f, 0.87651116f, 0.21878648f, 0.610987f, 0.54310906f, 0.5017944f,
      0.15232486f, 0.4647144f, 0.20985329f, 0.89526993f, 0.08409953f, 0.846228f, 0.060534656f,
      0.70841146f, 0.62002844f, 0.90842223f, 0.7173401f, 0.19575512f, 0.54321826f, 0.22779316f,
      0.8902793f, 0.40386343f, 0.16297412f, 0.7796456f, 0.7209991f, 0.57273495f, 0.80687743f,
      0.70873487f, 0.45478833f, 0.97999126f, 0.45059657f, 0.5500629f, 0.4574679f, 0.46920043f,
      0.5106411f, 0.16112137f, 0.30643195f, 0.41949958f, 0.6965155f, 0.45372677f, 0.7416339f,
      0.07971662f, 0.019000888f, 0.29456508f, 0.63610595f, 0.1810705f, 0.7203987f, 0.71254706f,
      0.4558406f, 0.18640989f, 0.04116726f, 0.7322634f, 0.66699976f, 0.045647383f, 0.11515516f,
      0.5025297f, 0.020907402f, 0.50736845f, 0.34275955f, 0.21446168f, 0.51500404f, 0.093204916f,
      0.9095519f, 0.949713f, 0.16832972f, 0.31914896f, 0.49500751f, 0.10780871f, 0.03719926f,
      0.16613615f, 0.1210829f, 0.88132215f, 0.6184779f, 0.8939942f, 0.24608392f, 0.3340807f,
      0.37963718f, 0.4086216f, 0.4861309f, 0.96492773f, 0.27750528f, 0.82491904f, 0.44960213f,
      0.07433933f, 0.014267266f, 0.5327839f, 0.78514075f, 0.0019600391f, 0.112906754f, 0.0718022f,
      0.9215306f, 0.89411324f, 0.19682491f, 0.8896367f, 0.16405362f, 0.044967473f, 0.2238177f,
      0.29950172f, 0.45508182f, 0.652841f, 0.56343627f, 0.028514147f, 0.59348315f, 0.46439964f,
      0.7485309f, 0.27697015f, 0.70751995f, 0.14157385f, 0.41028368f, 0.9382696f, 0.83076525f,
      0.16925985f, 0.7039947f, 0.8105199f, 0.4000479f, 0.50734824f, 0.13979793f, 0.4297228f,
      0.9553564f, 0.7211761f, 0.072479784f, 0.9391413f, 0.13516217f, 0.14382833f, 0.29872334f,
      0.06472236f, 0.3285427f, 0.2932666f, 0.06387442f, 0.30173403f, 0.89418864f, 0.26674682f,
      0.061959863f, 0.17714864f, 0.36635226f, 0.22465849f, 0.8074024f, 0.4161513f, 0.5806135f,
      0.34574044f, 0.9039644f, 0.47833657f, 0.7588931f, 0.18114948f, 0.033798456f, 0.38355058f,
      0.8689936f, 0.017748117f, 0.3073569f, 0.15338135f, 0.25849247f, 0.2459296f, 0.026683688f,
      0.7071423f, 0.49425328f, 0.50869817f, 0.013295233f, 0.7934831f, 0.37678176f, 0.97975004f,
      0.007119417f};
  const std::vector<float> fc1_experts_bias = {
      0.71526206f, 0.7472273f, 0.18946046f, 0.6239893f, 0.86909235f, 0.5726507f, 0.3942092f, 0.5369412f,
      0.44638616f, 0.7517496f, 0.16049433f, 0.75355124f, 0.7818118f, 0.19706267f, 0.9082818f, 0.9910924f,
      0.30288565f, 0.3599528f, 0.74917775f, 0.10828978f, 0.697729f, 0.61665237f, 0.81516486f, 0.0656966f,
      0.0846076f, 0.72456455f, 0.6801054f, 0.034616888f, 0.22117025f, 0.042510748f, 0.14178854f, 0.27440017f,
      0.91376925f, 0.40047455f, 0.7871756f, 0.97484046f, 0.7278661f, 0.052394807f, 0.75161135f, 0.6907173f,
      0.8875328f, 0.0067828894f, 0.807508f, 0.9092707f, 0.034817636f, 0.55231315f, 0.92683655f, 0.13634592f,
      0.66405964f, 0.7209387f, 0.63104504f, 0.9971379f, 0.9093898f, 0.9289774f, 0.4376766f, 0.9193563f,
      0.03404367f, 0.23018533f, 0.39305943f, 0.3514716f, 0.96184736f, 0.73583263f, 0.8219065f, 0.8401047f};
  const std::vector<float> fc2_experts_bias = {
      0.12649822f, 0.4420895f, 0.5730123f, 0.63004625f, 0.7571163f, 0.3010466f, 0.3492328f, 0.91837066f,
      0.36580783f, 0.15267932f, 0.8390199f, 0.83857775f, 0.34321654f, 0.40003997f, 0.13106f, 0.08245313f,
      0.68802476f, 0.28640372f, 0.89804775f, 0.09964341f, 0.43088746f, 0.5107959f, 0.75697356f, 0.90466535f,
      0.83860224f, 0.720098f, 0.2705031f, 0.14292616f, 0.052693605f, 0.5248023f, 0.9849401f, 0.40502876f};
  const std::vector<float> output = {
      0.2552814f, 0.17651685f, 0.0034551744f, -0.123282805f, 0.0073816925f, 0.004265253f, 0.16927283f, -0.05276826f,
      9.555821f, 7.6907287f, 10.626425f, 7.0543795f, 8.10093f, 10.3664465f, 10.925815f, 8.737018f,
      0.565234f, 0.17098689f, 0.10810414f, 0.43916586f, 0.3535297f, 0.45673048f, 0.3853893f, 0.18613164f,
      1.3354061f, 0.5049282f, 0.72775036f, 0.90331376f, 1.2945517f, 0.9123066f, 1.1995136f, 0.7708638f};

  RunMoETest(input, router_probs, fc1_experts_weights, fc2_experts_weights, {}, fc1_experts_bias, fc2_experts_bias,
             output, num_rows, num_experts, hidden_size, inter_size, "gelu", 0);
}

TEST(MoETest, MoETest_Relu) {
  int num_rows = 4;
  int num_experts = 4;
  int hidden_size = 8;
  int inter_size = 16;

  const std::vector<float> input = {
      0.7670296f, -0.93721074f, -2.330477f, -0.78088343f, 0.8250065f, 1.2206652f, -0.06297584f, 1.1463639f,
      1.2215378f, -0.31372663f, -0.7234253f, -0.3627346f, 0.44249064f, 0.19418247f, -0.49998695f, -0.55005103f,
      0.023851749f, -1.5203826f, 0.52939993f, -0.39082858f, -1.9291036f, 0.034976702f, -0.48336256f, -1.226073f,
      -0.33963847f, 0.0073261578f, -0.0521804f, 1.16749f, 1.7302082f, 2.0561688f, -0.2347232f, -1.3456243f};
  const std::vector<float> router_probs = {
      -0.08146476f, -0.40439552f, 1.0100367f, -0.7724162f, -0.08113786f, -0.36328858f, 0.3688482f, -0.013465762f,
      -0.32420647f, -0.3815508f, 0.79585606f, 0.14430691f, -0.21869831f, 0.11483674f, -0.11992836f, 0.35216537f};
  const std::vector<float> fc1_experts_weights = {
      0.81960344f, 0.3370188f, 0.0025978684f, 0.704681f, 0.77974546f, 0.81039995f, 0.9191656f,
      0.0930863f, 0.9296998f, 0.6688521f, 0.8346353f, 0.254516f, 0.15189773f, 0.87411255f,
      0.42156547f, 0.9192536f, 0.45050132f, 0.8188108f, 0.8788173f, 0.3993737f, 0.75130886f,
      0.9728532f, 0.44305897f, 0.29990643f, 0.38805157f, 0.73084867f, 0.6822241f, 0.21224737f,
      0.7268921f, 0.38206023f, 0.29594004f, 0.63248974f, 0.50729614f, 0.058027983f, 0.1513629f,
      0.40888822f, 0.85721636f, 0.08917904f, 0.04846859f, 0.32651705f, 0.47014588f, 0.19931877f,
      0.0065300465f, 0.14808255f, 0.11647397f, 0.61241513f, 0.013427794f, 0.54063064f, 0.62020564f,
      0.42109168f, 0.093910515f, 0.17329216f, 0.8595984f, 0.77621365f, 0.6858292f, 0.9661502f,
      0.6401168f, 0.98367476f, 0.8728501f, 0.6658554f, 0.2636242f, 0.0023456216f, 0.22547692f,
      0.73036134f, 0.045871615f, 0.57232875f, 0.7400529f, 0.3514018f, 0.6855346f, 0.38650817f,
      0.17856151f, 0.06670016f, 0.31548113f, 0.37051463f, 0.9207522f, 0.8086716f, 0.96955734f,
      0.20027226f, 0.4609884f, 0.6984514f, 0.92106473f, 0.7068576f, 0.76193494f, 0.33959562f,
      0.42948407f, 0.45626813f, 0.33349442f, 0.9746214f, 0.6947775f, 0.30955923f, 0.6265461f,
      0.13321638f, 0.49613327f, 0.25389326f, 0.3382396f, 0.63154167f, 0.4751312f, 0.17637217f,
      0.49510366f, 0.41178054f, 0.38488472f, 0.2956162f, 0.5160656f, 0.83521235f, 0.19854712f,
      0.8649436f, 0.11974698f, 0.2576263f, 0.08250773f, 0.34127057f, 0.3939438f, 0.99294376f,
      0.19409746f, 0.2726491f, 0.07161391f, 0.3470292f, 0.73995143f, 0.024847984f, 0.3278438f,
      0.4233855f, 0.052116573f, 0.39976662f, 0.032325685f, 0.024002194f, 0.003641069f, 0.91025376f,
      0.26059705f, 0.6037772f, 0.15248245f, 0.43843627f, 0.8800602f, 0.03305638f, 0.5070212f,
      0.36010116f, 0.2957666f, 0.30232358f, 0.39696145f, 0.739853f, 0.7784371f, 0.5088047f,
      0.9549045f, 0.83768386f, 0.68377024f, 0.77861303f, 0.8702919f, 0.26859397f, 0.004249513f,
      0.95591706f, 0.07397425f, 0.53982985f, 0.16686243f, 0.10180014f, 0.7563229f, 0.44548005f,
      0.5443443f, 0.7884607f, 0.3090204f, 0.52255917f, 0.17314798f, 0.816009f, 0.18360549f,
      0.45647776f, 0.80287653f, 0.20888287f, 0.79162645f, 0.37694973f, 0.47585016f, 0.30602258f,
      0.099057496f, 0.38170832f, 0.45378727f, 0.43509573f, 0.39106607f, 0.04720515f, 0.31711966f,
      0.5076527f, 0.15831816f, 0.24648392f, 0.20536041f, 0.13140821f, 0.39764988f, 0.029871285f,
      0.125171f, 0.40119207f, 0.00656116f, 0.054280818f, 0.9766699f, 0.2587883f, 0.29160416f,
      0.26099247f, 0.7965795f, 0.5606195f, 0.114180505f, 0.0958215f, 0.31298608f, 0.5905492f,
      0.84465307f, 0.2458393f, 0.90208143f, 0.3489008f, 0.3763513f, 0.23226917f, 0.21532774f,
      0.77226925f, 0.7452516f, 0.6557768f, 0.58111167f, 0.8635635f, 0.8374386f, 0.98291886f,
      0.04922247f, 0.91418463f, 0.66022503f, 0.35444462f, 0.41294336f, 0.48700142f, 0.5836911f,
      0.25849265f, 0.52233416f, 0.04094696f, 0.21901816f, 0.30438894f, 0.036863506f, 0.89029974f,
      0.11969727f, 0.16423601f, 0.72156656f, 0.8343076f, 0.09412521f, 0.9767149f, 0.31788063f,
      0.98074025f, 0.09888804f, 0.6211971f, 0.6106814f, 0.14735395f, 0.5540803f, 0.67416143f,
      0.6272928f, 0.25640452f, 0.74873763f, 0.63780516f, 0.59887487f, 0.6872336f, 0.6481394f,
      0.85645115f, 0.73576546f, 0.13524544f, 0.12807935f, 0.77395487f, 0.12080628f, 0.92312264f,
      0.26914406f, 0.25794363f, 0.43679124f, 0.901151f, 0.89180696f, 0.6873948f, 0.7573983f,
      0.6867124f, 0.65562236f, 0.60750586f, 0.9705138f, 0.84136647f, 0.11822635f, 0.24455917f,
      0.3057955f, 0.9648854f, 0.11856395f, 0.5729957f, 0.5475096f, 0.97399056f, 0.46134835f,
      0.08452982f, 0.20571685f, 0.37042046f, 0.18362767f, 0.6367655f, 0.789582f, 0.5267614f,
      0.006936848f, 0.22689629f, 0.56744653f, 0.2886421f, 0.08430874f, 0.2594666f, 0.8881108f,
      0.06989372f, 0.09070045f, 0.9822047f, 0.2052834f, 0.37891757f, 0.9356598f, 0.43602943f,
      0.9036556f, 0.14923638f, 0.59657127f, 0.9274289f, 0.17446929f, 0.25843787f, 0.026530087f,
      0.97506f, 0.32732427f, 0.18941313f, 0.6330173f, 0.9477422f, 0.76062596f, 0.58501935f,
      0.8771834f, 0.83592474f, 0.38817167f, 0.059375823f, 0.6059905f, 0.7935056f, 0.4160077f,
      0.8732242f, 0.48319155f, 0.48121578f, 0.7409689f, 0.24937624f, 0.36391765f, 0.87772477f,
      0.9568925f, 0.8909887f, 0.4418506f, 0.029734552f, 0.36356616f, 0.039716125f, 0.96128887f,
      0.43307513f, 0.9863913f, 0.72956276f, 0.81273925f, 0.5219139f, 0.734132f, 0.038692355f,
      0.571489f, 0.22488606f, 0.64955276f, 0.13203424f, 0.4537862f, 0.15951324f, 0.39076614f,
      0.20122272f, 0.2049576f, 0.7498283f, 0.67207885f, 0.23164761f, 0.81357706f, 0.90659577f,
      0.16087383f, 0.0070830584f, 0.4716931f, 0.24090862f, 0.61514187f, 0.3901443f, 0.8615075f,
      0.19645631f, 0.70352167f, 0.19309378f, 0.6200726f, 0.16256708f, 0.50783044f, 0.40783793f,
      0.06589496f, 0.4638992f, 0.576659f, 0.69065434f, 0.67509633f, 0.34033298f, 0.46363378f,
      0.54112387f, 0.692392f, 0.38902867f, 0.7229242f, 0.9170264f, 0.14645958f, 0.6049296f,
      0.50687206f, 0.041014254f, 0.5943895f, 0.5889769f, 0.996743f, 0.3512686f, 0.3545606f,
      0.11835027f, 0.8793823f, 0.4897902f, 0.6020277f, 0.4176172f, 0.9979084f, 0.017213225f,
      0.76697665f, 0.51498765f, 0.9828271f, 0.18942028f, 0.03156215f, 0.21532834f, 0.7546957f,
      0.3030849f, 0.25331455f, 0.25018185f, 0.8181312f, 0.4379952f, 0.93655676f, 0.4191329f,
      0.1364128f, 0.6576238f, 0.26358372f, 0.10446805f, 0.20143801f, 0.7034796f, 0.81369543f,
      0.9055267f, 0.8845484f, 0.9813073f, 0.80806476f, 0.45996118f, 0.17288941f, 0.0109113455f,
      0.010527074f, 0.12900633f, 0.38850087f, 0.58397317f, 0.064349174f, 0.059881568f, 0.9363466f,
      0.64850605f, 0.2611835f, 0.6134902f, 0.39324278f, 0.99017924f, 0.5611374f, 0.8489496f,
      0.6768587f, 0.16939592f, 0.6630776f, 0.008604288f, 0.04554516f, 0.59782606f, 0.941691f,
      0.5579074f, 0.51328385f, 0.25597447f, 0.3972702f, 0.76215106f, 0.42129284f, 0.788768f,
      0.58574325f, 0.23052096f, 0.56766605f, 0.69195485f, 0.44551176f, 0.68473387f, 0.8536634f,
      0.9008311f, 0.6359719f, 0.76128954f, 0.098151624f, 0.8975601f, 0.27424216f, 0.5211961f,
      0.5697224f, 0.91796166f, 0.20880443f, 0.02678603f, 0.33305728f, 0.36334568f, 0.9016098f,
      0.71459657f, 0.20877302f, 0.22013813f, 0.49310172f, 0.3066004f, 0.98130906f, 0.29471546f,
      0.22050089f, 0.50056237f, 0.65390605f, 0.959695f, 0.5274922f, 0.40259063f, 0.3766839f,
      0.04788208f, 0.9146384f, 0.7766764f, 0.3396778f, 0.80288273f, 0.62271714f, 0.07512486f,
      0.47491795f, 0.24217117f, 0.53226113f, 0.10418975f, 0.956497f, 0.2662105f, 0.694273f,
      0.18205583f, 0.08483446f, 0.062181532f, 0.6005109f, 0.42657375f, 0.066022694f, 0.26139832f,
      0.9344639f, 0.4183907f, 0.22029644f, 0.38556474f, 0.8900659f, 0.7218073f, 0.34206223f,
      0.080626905f};
  const std::vector<float> fc2_experts_weights = {
      0.6255686f, 0.7868948f, 0.37150103f, 0.014990091f, 0.12253064f, 0.17739785f, 0.2880798f, 0.25019473f,
      0.1781866f, 0.15737581f, 0.2366305f, 0.75366426f, 0.39971018f, 0.93269855f, 0.7777848f, 0.35543054f,
      0.09472537f, 0.0029274821f, 0.78196156f, 0.24055547f, 0.13405013f, 0.2729941f, 0.49309975f, 0.379932f,
      0.994184f, 0.5411925f, 0.004210472f, 0.080795944f, 0.69859487f, 0.19460368f, 0.8686458f, 0.52536315f,
      0.71121234f, 0.95854944f, 0.6817853f, 0.84227383f, 0.75649333f, 0.8497335f, 0.9576164f, 0.36471486f,
      0.6911175f, 0.549694f, 0.7617172f, 0.82247066f, 0.053544044f, 0.25394785f, 0.36938924f, 0.8000995f,
      0.65789884f, 0.919321f, 0.89608955f, 0.029270172f, 0.9348151f, 0.15788019f, 0.19988996f, 0.17417055f,
      0.7006223f, 0.4366895f, 0.61926836f, 0.040263534f, 0.7878332f, 0.59613144f, 0.8557286f, 0.21456867f,
      0.065598905f, 0.6989418f, 0.31273842f, 0.0647831f, 0.7991694f, 0.22429371f, 0.5039311f, 0.009367704f,
      0.20085096f, 0.5693159f, 0.24570602f, 0.22299266f, 0.34460813f, 0.6356306f, 0.74428976f, 0.750327f,
      0.63625044f, 0.043019474f, 0.6682699f, 0.7801003f, 0.57832605f, 0.86499554f, 0.73779976f, 0.7819258f,
      0.28080195f, 0.3018247f, 0.981851f, 0.41664255f, 0.11966437f, 0.6922361f, 0.9410264f, 0.3208093f,
      0.45933473f, 0.32138962f, 0.6778976f, 0.7697645f, 0.66478735f, 0.65776104f, 0.15482187f, 0.63283706f,
      0.42452294f, 0.63012594f, 0.273876f, 0.16297674f, 0.5731115f, 0.7744376f, 0.21586305f, 0.80205464f,
      0.7284089f, 0.35509557f, 0.08370459f, 0.91119635f, 0.97456336f, 0.661535f, 0.98558843f, 0.031699598f,
      0.40856004f, 0.6885702f, 0.8378734f, 0.98845494f, 0.7422309f, 0.38662314f, 0.2530955f, 0.47626138f,
      0.061956525f, 0.644235f, 0.735041f, 0.83424246f, 0.2638232f, 0.6273294f, 0.7184546f, 0.0766899f,
      0.32245284f, 0.60083544f, 0.906533f, 0.086719275f, 0.31617337f, 0.24053413f, 0.048938513f, 0.8919203f,
      0.22487706f, 0.011788309f, 0.9583977f, 0.69286025f, 0.9183893f, 0.542809f, 0.70638496f, 0.5914087f,
      0.29438752f, 0.67338234f, 0.73173434f, 0.63433063f, 0.4570613f, 0.70197225f, 0.60771453f, 0.7654821f,
      0.13812399f, 0.14224577f, 0.67939556f, 0.9691554f, 0.88740516f, 0.3961745f, 0.57973206f, 0.49563587f,
      0.37618434f, 0.026723444f, 0.97712487f, 0.1978364f, 0.96686924f, 0.58496886f, 0.13263822f, 0.7866956f,
      0.74798125f, 0.0469383f, 0.30301625f, 0.9748982f, 0.65107566f, 0.32560885f, 0.8141865f, 0.3695873f,
      0.3067485f, 0.5446483f, 0.29709607f, 0.5181831f, 0.29501313f, 0.33996922f, 0.11060041f, 0.02524674f,
      0.1647259f, 0.34876132f, 0.031807184f, 0.60586995f, 0.5313419f, 0.79801136f, 0.81332296f, 0.41627264f,
      0.9496114f, 0.466555f, 0.41363865f, 0.9874878f, 0.14229488f, 0.11154431f, 0.091483414f, 0.1414501f,
      0.45834088f, 0.3178513f, 0.68110096f, 0.13568163f, 0.07941705f, 0.53083426f, 0.96346164f, 0.5235164f,
      0.76482266f, 0.21967298f, 0.6893093f, 0.34609234f, 0.22017813f, 0.34257007f, 0.70869184f, 0.3112445f,
      0.6078779f, 0.5715967f, 0.25227106f, 0.94672066f, 0.44809794f, 0.8252871f, 0.88438797f, 0.86481494f,
      0.95148784f, 0.112026334f, 0.4173867f, 0.34240413f, 0.36137718f, 0.28898042f, 0.19898665f, 0.9130488f,
      0.22580266f, 0.40754277f, 0.75443816f, 0.26275212f, 0.9795632f, 0.4115007f, 0.37215167f, 0.6558706f,
      0.5015968f, 0.9426372f, 0.4018826f, 0.8016564f, 0.26275063f, 0.33729053f, 0.29362458f, 0.5511502f,
      0.12605143f, 0.5844644f, 0.30534166f, 0.009582937f, 0.41596514f, 0.7294f, 0.29118127f, 0.14877802f,
      0.77849144f, 0.37462044f, 0.09746289f, 0.612149f, 0.4321279f, 0.13902885f, 0.5097318f, 0.23706353f,
      0.5031309f, 0.35566723f, 0.813157f, 0.43805653f, 0.08011025f, 0.27288425f, 0.36515707f, 0.65959847f,
      0.42534047f, 0.06618434f, 0.89204556f, 0.25902748f, 0.2919191f, 0.77043164f, 0.7297412f, 0.3313396f,
      0.11166459f, 0.6964502f, 0.90752834f, 0.15901726f, 0.900068f, 0.2406702f, 0.6299379f, 0.7841949f,
      0.7123557f, 0.50165176f, 0.50806034f, 0.9854489f, 0.3689273f, 0.19051671f, 0.32111454f, 0.18070674f,
      0.39045036f, 0.6977819f, 0.9975799f, 0.5931799f, 0.24834275f, 0.6194577f, 0.095391035f, 0.7776301f,
      0.20649683f, 0.9747401f, 0.6052985f, 0.4263978f, 0.078885734f, 0.7982801f, 0.7177174f, 0.05027789f,
      0.36251247f, 0.63427305f, 0.64812917f, 0.7067924f, 0.44507074f, 0.23906898f, 0.19735986f, 0.03428924f,
      0.57597995f, 0.7426891f, 0.2980855f, 0.19379246f, 0.10265827f, 0.86077714f, 0.33929902f, 0.53255826f,
      0.9328308f, 0.30511153f, 0.32955307f, 0.39670604f, 0.5471632f, 0.26892018f, 0.5072957f, 0.3091979f,
      0.19757104f, 0.23322952f, 0.26604044f, 0.26614368f, 0.79264474f, 0.8869355f, 0.49160433f, 0.8244896f,
      0.65486836f, 0.92657536f, 0.753946f, 0.45817143f, 0.46995157f, 0.33152503f, 0.56953406f, 0.07021719f,
      0.749946f, 0.50672436f, 0.5824448f, 0.9922104f, 0.9277247f, 0.86002564f, 0.064810574f, 0.9553747f,
      0.41281444f, 0.42781502f, 0.92897725f, 0.7250554f, 0.029657006f, 0.3121612f, 0.77614623f, 0.18359429f,
      0.2813105f, 0.44517577f, 0.68485546f, 0.5000241f, 0.9771502f, 0.81278664f, 0.3692627f, 0.7917771f,
      0.24083132f, 0.19877058f, 0.28815013f, 0.30789846f, 0.24441916f, 0.7504142f, 0.4646262f, 0.5149533f,
      0.6508104f, 0.5387549f, 0.1506182f, 0.77973497f, 0.5613007f, 0.8947809f, 0.24737573f, 0.882782f,
      0.005495131f, 0.90993816f, 0.80128354f, 0.8917693f, 0.091435075f, 0.6704894f, 0.95763975f, 0.27762014f,
      0.5927816f, 0.74048257f, 0.39038336f, 0.7001237f, 0.30437487f, 0.11625093f, 0.3411532f, 0.8124163f,
      0.6896569f, 0.71350795f, 0.6001208f, 0.36836517f, 0.5188247f, 0.01892668f, 0.15339965f, 0.25419557f,
      0.2064318f, 0.88833815f, 0.47727865f, 0.88709056f, 0.49082512f, 0.11388689f, 0.31912774f, 0.9593644f,
      0.78015697f, 0.8311006f, 0.63250524f, 0.6571592f, 0.20667112f, 0.9809466f, 0.14625502f, 0.04218155f,
      0.57540226f, 0.9262546f, 0.44018233f, 0.4769255f, 0.3852706f, 0.09546256f, 0.9905191f, 0.13382024f,
      0.07074398f, 0.16185725f, 0.4233225f, 0.3151368f, 0.9110969f, 0.41447622f, 0.58130866f, 0.7651092f,
      0.9817701f, 0.11023259f, 0.42101204f, 0.5397561f, 0.5778314f, 0.22598988f, 0.31468558f, 0.8214317f,
      0.67929304f, 0.79097277f, 0.7053677f, 0.8750746f, 0.019512117f, 0.032795787f, 0.43307304f, 0.59631824f,
      0.84294224f, 0.93783194f, 0.53943527f, 0.60289854f, 0.8253078f, 0.30536187f, 0.14199954f, 0.9196194f,
      0.9227386f, 0.15846318f, 0.29161406f, 0.7992451f, 0.72343415f, 0.9935814f, 0.6151709f, 0.077278376f,
      0.11056489f, 0.16041255f, 0.99320936f, 0.06393474f, 0.33417904f, 0.46236527f, 0.7078488f, 0.25308424f,
      0.5302883f, 0.99474716f, 0.028710365f, 0.6765068f, 0.998457f, 0.29653466f, 0.08064735f, 0.89677596f,
      0.9564106f, 0.53748304f, 0.79050577f, 0.09722155f, 0.9004303f, 0.3784039f, 0.47111923f, 0.95958996f};
  const std::vector<float> fc1_experts_bias = {
      0.8748215f, 0.5054756f, 0.74107623f, 0.32518923f, 0.0639081f, 0.62639004f, 0.64906263f, 0.17322052f,
      0.7424998f, 0.07288867f, 0.93031204f, 0.9841952f, 0.6361292f, 0.18628561f, 0.7433356f, 0.5852079f,
      0.6359594f, 0.66432667f, 0.88067776f, 0.28508204f, 0.38752747f, 0.63635296f, 0.55448055f, 0.9031888f,
      0.23738074f, 0.48179168f, 0.5934266f, 0.3672055f, 0.84085834f, 0.5546908f, 0.03788501f, 0.44583207f,
      0.27322155f, 0.5485856f, 0.44189203f, 0.00403291f, 0.40888733f, 0.45211035f, 0.35256076f, 0.9593902f,
      0.39090043f, 0.8212086f, 0.62385887f, 0.07793343f, 0.61749303f, 0.9143678f, 0.17294967f, 0.17681253f,
      0.9894245f, 0.901755f, 0.221053f, 0.8008725f, 0.43603396f, 0.007035315f, 0.5375667f, 0.661547f,
      0.35001957f, 0.67394173f, 0.072449565f, 0.84650797f, 0.92626715f, 0.77573335f, 0.58474565f, 0.66467446f};
  const std::vector<float> fc2_experts_bias = {
      0.13822609f, 0.3750633f, 0.45226622f, 0.22175694f, 0.13068998f, 0.8363088f, 0.8393226f, 0.045905888f,
      0.65910596f, 0.7034011f, 0.97498417f, 0.78927684f, 0.95966834f, 0.33630514f, 0.8501932f, 0.9067007f,
      0.027835965f, 0.09864664f, 0.6012027f, 0.7730189f, 0.25159347f, 0.55506724f, 0.49927413f, 0.62655383f,
      0.23132521f, 0.7820195f, 0.8325047f, 0.15307087f, 0.5048437f, 0.5013873f, 0.66055787f, 0.96579224f};
  const std::vector<float> output = {
      1.3775184f, 2.0985768f, 2.091839f, 2.9706357f, 1.9404914f, 1.9915576f, 2.3302228f, 2.3702593f,
      0.51896286f, 0.7936432f, 0.9944805f, 1.3225251f, 0.73894113f, 0.87975955f, 1.0468717f, 1.1585085f,
      0.012911659f, 0.045757107f, 0.27884653f, 0.3585817f, 0.116771236f, 0.25755364f, 0.23161705f, 0.2906256f,
      4.8571277f, 5.649453f, 5.485141f, 5.306299f, 4.767025f, 6.9010167f, 5.3520975f, 6.711155f};

  RunMoETest(input, router_probs, fc1_experts_weights, fc2_experts_weights, {}, fc1_experts_bias, fc2_experts_bias,
             output, num_rows, num_experts, hidden_size, inter_size, "relu", 0);
}

TEST(MoETest, MoETest_Mixtral) {
  // This test uses FC3 (gated SiLU / Mixtral pattern) with dimensions too small for the
  // CUTLASS SIMT kernel (needs hidden_size >= 128, inter_size >= 128). CPU MoE does not
  // support FC3. Skip until test data is regenerated with larger dimensions.
  GTEST_SKIP() << "Dimensions too small for CUTLASS kernel and CPU MoE does not support FC3";
  int num_rows = 6;
  int num_experts = 8;
  int hidden_size = 4;
  int inter_size = 8;

  const std::vector<float> input = {0.9212995f, 0.5282444f, -0.008228387f, -1.449332f, -0.6051824f, -0.17924511f,
                                    0.1995587f, -1.2461947f, 0.86708033f, 0.19191018f, 1.1600108f, -0.008815222f,
                                    0.8504777f, -0.84964496f, -1.4019964f, 0.17225051f, 0.35569248f, 1.2056456f,
                                    1.3690308f, -0.69495815f, 1.4324434f, 0.22761835f, -1.1286871f, 1.124213f};
  const std::vector<float> router_probs = {
      -0.09331456f, -0.47121337f, 0.07311103f, 0.47643483f, 0.21135253f, -0.72226393f, -0.048502743f, 0.39447474f,
      -0.9014899f, -0.36629856f, -0.23088816f, -0.099606544f, -0.45191774f, -0.30394578f, 0.6266495f, 0.67937183f,
      0.27117345f, -0.36059442f, 0.81510246f, 0.61359257f, 0.07649982f, -0.44949868f, -0.54758865f, 0.4736983f,
      0.21584567f, 0.21296778f, 0.093342215f, -0.09353682f, 0.61422515f, 0.19574627f, 0.0063361377f, -0.2465148f,
      0.15675665f, -0.4546509f, 0.24447554f, 0.5921611f, -0.18192923f, -0.66116416f, -0.40265432f, 0.33475468f,
      1.2906091f, 0.4709078f, 0.16256471f, 0.19308007f, 0.97568524f, 0.25876164f, -0.7964541f, -1.0319631f};
  const std::vector<float> fc1_experts_weights = {
      0.3860137f, 0.083209574f, -0.16235226f, 0.30897498f, 0.077925384f, 0.4039817f, 0.054659843f,
      -0.15768659f, 0.13434184f, -0.13558972f, 0.21042877f, 0.44641107f, 0.28902978f, -0.21858627f,
      0.28863233f, 0.089463115f, 0.25391752f, -0.30475253f, -0.49495423f, -0.19318026f, -0.38351142f,
      0.41026944f, 0.14401567f, 0.20710677f, 0.15813059f, -0.008697987f, 0.39130414f, -0.3552568f,
      0.031481862f, -0.3412701f, 0.154176f, -0.17219114f, 0.41923493f, -0.09923202f, 0.43019837f,
      0.15579104f, -0.4233985f, 0.3460176f, -0.13757241f, -0.19166303f, -0.41503525f, -0.49708033f,
      0.14305532f, -0.109221935f, 0.19466156f, -0.41033173f, 0.37121457f, -0.36702687f, -0.08633667f,
      0.10443485f, 0.2581259f, 0.40365517f, 0.45547962f, -0.39646107f, 0.12583363f, -0.21506298f,
      -0.054792404f, -0.37424505f, 0.45542932f, -0.36697525f, 0.26722562f, 0.1757198f, 0.16247797f,
      -0.2703231f, -0.49740213f, 0.33463532f, 0.37881732f, 0.1822241f, -0.3486371f, -0.49346995f,
      -0.40608948f, 0.37285012f, 0.24005288f, 0.42075223f, 0.26193494f, 0.12654608f, -0.0048963428f,
      -0.38025302f, -0.4283861f, -0.46767431f, 0.20468098f, -0.245484f, -0.10062629f, -0.28775263f,
      -0.09111178f, -0.35191745f, -0.32670784f, 0.16585541f, -0.1485982f, 0.3086716f, -0.16040438f,
      -0.36678362f, -0.088219464f, -0.2423737f, -0.15297079f, -0.4759978f, -0.34751755f, -0.10303855f,
      0.3702919f, 0.25632292f, -0.3163945f, -0.4009425f, -0.34168184f, -0.49343884f, -0.3858195f,
      -0.1236487f, 0.33743858f, 0.08369112f, -0.38030273f, -0.40111196f, 0.24873763f, -0.37192065f,
      -0.06156373f, 0.23985302f, -0.23140603f, -0.05451995f, -0.04352224f, -0.118291676f, -0.25351608f,
      -0.44571918f, -0.4041785f, -0.26773083f, 0.48291886f, -0.24150735f, -0.335764f, 0.121197104f,
      0.13780516f, 0.27395487f, -0.20423341f, 0.18377024f, -0.33313757f, -0.32685202f, -0.024149835f,
      -0.18288034f, -0.374829f, 0.29657948f, 0.40208143f, 0.08111167f, -0.08705664f, -0.4631365f,
      -0.18211937f, 0.12729281f, 0.23576546f, -0.06320876f, -0.19767642f, 0.27861303f, -0.39819986f,
      0.31600899f, -0.19397742f, 0.0076527f, -0.09880793f, 0.060619473f, -0.1510992f, 0.36356348f,
      -0.012998581f, 0.39029974f, 0.48074025f, -0.24359548f, -0.36475456f, 0.401151f, 0.15562236f,
      -0.38143605f, -0.31637233f, -0.41569126f, 0.43565983f, -0.4734699f, 0.37718338f, -0.01680845f,
      -0.058149397f, 0.31273925f, -0.046213806f, 0.31357706f, 0.36150748f, -0.43410504f, 0.19239199f,
      0.0943895f, 0.10750586f, 0.07299572f, 0.13676548f, -0.24053341f, -0.063970566f, 0.47506f,
      0.33592474f, -0.018784225f, -0.47026545f, 0.021913886f, -0.34048676f, 0.40659577f, -0.3035437f,
      -0.036100805f, -0.11097133f, 0.08897692f, 0.3793823f, 0.48282713f, 0.3181312f, -0.298562f,
      -0.3271106f, 0.4363466f, 0.17685872f, 0.013283849f, 0.067666054f, -0.40184838f, -0.16694272f,
      0.48130906f, -0.12331611f, -0.025082052f, -0.41516554f, -0.27970356f, -0.010209799f, -0.31057972f,
      -0.062004805f, 0.20347959f, -0.48908865f, 0.14850605f, -0.33060408f, -0.24402553f, 0.19195485f,
      0.39756012f, -0.13665432f, -0.20528454f, -0.45211792f, -0.25782883f, -0.43781847f, -0.114435256f,
      0.12556863f, -0.40527463f, 0.21121234f, 0.15789884f, -0.4344011f, 0.13625044f, -0.04066527f,
      0.22840887f, 0.2868948f, -0.49707252f, 0.45854944f, 0.419321f, 0.19894183f, -0.45698053f,
      -0.17861038f, -0.14490443f, -0.12849897f, 0.28196156f, 0.18178529f, 0.39608955f, -0.18726158f,
      0.16826987f, 0.17789757f, -0.4162954f, -0.4850099f, -0.25944453f, 0.34227383f, -0.47072983f,
      -0.4352169f, 0.2801003f, 0.26976448f, 0.41119635f};
  const std::vector<float> fc2_experts_weights = {
      0.10833451f, -0.07365984f, 0.2932343f, -0.20955177f, -0.21085852f, -0.21087126f, 0.31800103f,
      0.11782206f, 0.34020698f, -0.29177922f, -0.35068116f, -0.27660736f, -0.2378315f, 0.14320332f,
      0.12659892f, 0.29377612f, -0.18258394f, -0.24102151f, 0.1875877f, -0.14290786f, 0.21457997f,
      -0.08389844f, 0.20224877f, -0.27469966f, -0.17842063f, 0.1077901f, 0.07474385f, -0.09014153f,
      0.21074237f, 0.24034885f, -0.2563875f, -0.18875091f, 0.32136288f, 0.0776935f, 0.045481127f,
      -0.3115706f, 0.14841764f, -0.053040292f, -0.16197139f, 0.30368346f, 0.0788243f, -0.19561274f,
      -0.17894714f, -0.016884197f, 0.19741052f, -0.090344846f, -0.20172869f, -0.12107184f, -0.26413083f,
      0.12608862f, 0.27366453f, -0.3328494f, 0.08211302f, 0.18264277f, 0.064109616f, -0.12590908f,
      0.18453442f, 0.18579696f, 0.13220324f, -0.062126897f, -0.09362138f, 0.037823465f, -0.062456656f,
      -0.10535928f, 0.1978099f, -0.24614547f, 0.17770219f, 0.16043694f, 0.2525901f, -0.27119386f,
      0.25427446f, -0.16714293f, 0.13119277f, 0.33202717f, -0.04986229f, -0.0027341924f, -0.0813988f,
      -0.2952116f, 0.16967128f, -0.35097876f, 0.21948591f, 0.2645375f, 0.33435768f, -0.08339601f,
      -0.2904943f, 0.0794895f, 0.19531254f, -0.35189477f, -0.080250844f, -0.21193951f, -0.0309231f,
      -0.17402375f, -0.14452116f, -0.11223866f, -0.33598322f, 0.2900932f, 0.26874313f, 0.19688474f,
      -0.35054854f, 0.03135616f, 0.21416605f, -0.032677338f, -0.20834164f, 0.33705652f, -0.1322388f,
      -0.20129368f, -0.31874785f, 0.015792634f, 0.15667121f, 0.07826358f, 0.069915086f, -0.26813045f,
      -0.330179f, 0.006225848f, 0.32238203f, 0.20397249f, -0.2058509f, -0.04589425f, -0.26063374f,
      -0.17056243f, 0.064027935f, 0.19252343f, 0.29287276f, -0.3245995f, 0.23639117f, -0.24935842f,
      0.13239416f, 0.29919288f, 0.27704936f, -0.26995474f, -0.02733084f, -0.34864828f, -0.2894185f,
      0.068286195f, 0.09405743f, 0.07494662f, -0.096224755f, 0.3261805f, 0.050550338f, -0.20862648f,
      -0.020016002f, 0.08490415f, 0.123811804f, -0.24999082f, 0.13250813f, -0.18062393f, -0.2937818f,
      -0.19311349f, 0.3409702f, 0.30223787f, 0.31660154f, 0.2075398f, 0.26709175f, -0.04732303f,
      -0.19453493f, 0.17665526f, -0.18320526f, -0.2386011f, -0.11290163f, 0.07419645f, 0.3327035f,
      0.033594366f, 0.20476541f, 0.2744358f, 0.2854276f, -0.122100174f, -0.07907457f, 0.17039073f,
      -0.09647329f, 0.16555631f, -0.07724f, -0.23979841f, 0.14391156f, 0.054206114f, 0.15763119f,
      0.35125035f, 0.24138254f, 0.33516192f, 0.018923176f, -0.30413106f, -0.24802732f, -0.21961808f,
      -0.31156835f, -0.17721775f, -0.32546985f, -0.32619375f, -0.21126744f, -0.3485449f, -0.21701548f,
      0.13481297f, 0.29488218f, -0.10516899f, 0.072144486f, -0.3312356f, 0.30869225f, 0.22181617f,
      -0.34610963f, -0.16886877f, 0.11531327f, -0.07264093f, -0.038529005f, -0.15963489f, 0.283981f,
      -0.19763571f, 0.2931936f, 0.022812065f, 0.07107194f, 0.27581826f, -0.058253434f, -0.20129326f,
      -0.057181682f, 0.28675067f, -0.26233214f, 0.080249704f, -0.34746924f, 0.18536879f, 0.13062657f,
      0.014987925f, 0.15174268f, 0.0003976555f, 0.19563977f, -0.2798801f, -0.051920194f, 0.15684144f,
      -0.26691115f, -0.25876564f, 0.18136817f, 0.3074607f, 0.21154472f, 0.055384878f, 0.11652225f,
      0.33556697f, -0.22811417f, -0.16051741f, 0.24729891f, -0.24191524f, -0.19495378f, 0.25809082f,
      0.1115539f, 0.11422251f, -0.1498502f, -0.0048792143f, 0.32358363f, -0.21220984f, 0.002779711f,
      0.16814983f, -0.24407779f, 0.34336287f, -0.176639f, -0.08490091f, -0.09566104f, -0.23039621f,
      -0.34692943f, 0.19935164f, 0.09392998f, -0.33113837f};
  const std::vector<float> fc3_experts_weights = {
      0.45783097f, -0.16871625f, -0.1772582f, -0.4837973f, -0.2863351f, 0.12490183f, -0.06599659f,
      -0.362943f, 0.011728346f, -0.34154075f, -0.42419833f, -0.27533132f, -0.43760604f, -0.31836903f,
      0.49980444f, 0.09443748f, 0.15407985f, -0.46634215f, -0.3283869f, -0.16642791f, 0.07818556f,
      -0.43996066f, -0.21543652f, -0.2993343f, 0.0013856292f, -0.1860516f, -0.034647882f, -0.33881485f,
      -0.34319758f, -0.2917009f, -0.17114872f, -0.39464045f, 0.31960344f, 0.42969978f, -0.049498677f,
      -0.11194843f, 0.007296145f, -0.029854119f, 0.12020564f, 0.14011681f, -0.45412838f, -0.18451887f,
      0.42106473f, 0.19477749f, -0.024868786f, -0.30145288f, -0.30590254f, -0.44788343f, -0.16298121f,
      0.16885209f, 0.31881082f, 0.23084867f, -0.44197202f, -0.30068123f, -0.078908324f, 0.48367476f,
      0.07232875f, -0.12948537f, 0.20685762f, -0.19044077f, -0.32362783f, 0.36494362f, -0.22735089f,
      -0.100233376f, 0.4191656f, -0.07843453f, -0.056941032f, -0.20405996f, -0.4515314f, -0.4865722f,
      0.18582922f, -0.27452308f, -0.3214385f, -0.039011598f, -0.16650558f, -0.16176039f, 0.016065598f,
      -0.10605621f, -0.17215621f, -0.23940295f, -0.4069137f, 0.4192536f, -0.20009357f, 0.13248974f,
      -0.17348295f, 0.04063064f, 0.46615022f, 0.23036134f, -0.43329984f, 0.1984514f, 0.47462142f,
      0.13154167f, 0.33521235f, 0.49294376f, -0.0766145f, 0.10377723f, 0.0070211887f, 0.4549045f,
      -0.42602575f, -0.1909796f, 0.29162645f, -0.108933926f, -0.102350116f, -0.20839584f, 0.34465307f,
      0.2452516f, 0.16022503f, -0.28098184f, -0.4058748f, 0.054080307f, 0.14813942f, -0.23085594f,
      -0.13989884f, 0.33768386f, 0.03982985f, 0.022559166f, -0.12305027f, -0.45279485f, -0.47012872f,
      -0.23900753f, -0.2541607f, 0.1557768f, -0.14555538f, -0.19561106f, 0.4767149f, 0.17416143f,
      0.35645115f, -0.24205637f, 0.2573983f, -0.19420451f, -0.29428315f, 0.06744653f, -0.2947166f,
      -0.3255307f, 0.26062596f, -0.0839923f, 0.4568925f, 0.4863913f, 0.14955276f, 0.17207885f,
      0.11514187f, 0.007830441f, -0.036366224f, 0.006872058f, 0.18671238f, 0.4648854f, -0.12957954f,
      -0.21135789f, -0.121082425f, -0.24156213f, 0.08501935f, 0.3732242f, 0.3909887f, 0.22956276f,
      -0.36796576f, -0.2683524f, -0.10985571f, -0.09216207f, 0.041123867f, -0.45898575f, -0.14543939f,
      0.26697665f, -0.24668545f, -0.23641628f, 0.30806476f, -0.43565083f, 0.06113738f, 0.44169098f,
      0.08574325f, 0.1359719f, -0.29119557f, -0.006898284f, 0.027492225f, 0.12271714f, 0.194273f,
      0.43446392f, -0.38164973f, 0.0149876475f, -0.24981815f, -0.39553195f, -0.040038824f, -0.44011843f,
      0.3489496f, 0.057907403f, -0.26947904f, 0.26128954f, -0.47321397f, -0.19339961f, -0.09740937f,
      -0.42487514f, -0.31794417f, -0.08160931f, 0.4979084f, 0.2546957f, -0.3635872f, 0.38454843f,
      -0.11149913f, -0.10675722f, -0.45445484f, -0.07870716f, 0.35366338f, 0.069722414f, -0.29122698f,
      0.15390605f, -0.16032219f, 0.456497f, -0.4339773f, -0.15793777f, -0.48278677f, -0.19691509f,
      0.15762383f, 0.48130733f, 0.08397317f, 0.49017924f, 0.09782606f, 0.288768f, 0.4008311f,
      0.41796166f, -0.27986187f, 0.45969498f, 0.30288273f, -0.2337895f, -0.23860168f, -0.4193731f,
      -0.3218134f, 0.49418402f, 0.19111753f, 0.20062232f, -0.29914904f, -0.21919805f, -0.07547706f,
      -0.09143996f, -0.3426242f, 0.041192472f, 0.049694f, -0.063310504f, 0.06931591f, -0.19817531f,
      0.13012594f, 0.1885702f, -0.2633695f, -0.49578953f, 0.2617172f, 0.11926836f, -0.25429398f,
      0.48185098f, -0.22612399f, 0.3378734f, 0.25366426f, -0.41920406f, 0.32247066f, -0.45973647f,
      -0.27700734f, -0.08335745f, -0.33702326f, 0.48845494f};
  const std::vector<float> output = {0.026516449f, 0.04061616f, 0.04403834f, -0.13644142f, 0.038774252f,
                                     0.024002096f, -0.061423667f, 0.034824893f, -0.022858473f, 0.04693405f,
                                     -0.0120724365f, -0.028846134f, -0.0168579f, -0.07958221f, 0.048179876f,
                                     0.053492386f, -0.026292695f, -0.009724421f, -0.026503641f, 0.031220898f,
                                     0.04189077f, 0.11775493f, -0.037770163f, -0.0790936f};

  RunMoETest(input, router_probs, fc1_experts_weights, fc2_experts_weights, fc3_experts_weights, {}, {}, output,
             num_rows, num_experts, hidden_size, inter_size, "silu", 1, /*normalize_routing_weights*/
             2 /*top_k*/);
}

TEST(MoETest, QMoETest_CUDA_Int4_DisablePrepackingFailsLoudly) {
  constexpr int min_cuda_arch = 700;
  if (!HasCudaEnvironment(min_cuda_arch)) {
    GTEST_SKIP() << "CUDA execution provider not available";
  }

  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
  }

  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 128;
  constexpr int64_t expert_weight_bits = 4;
  constexpr int64_t pack_size = 8 / expert_weight_bits;

  const std::vector<float> input(num_rows * hidden_size, 0.0f);
  const std::vector<float> router_probs(num_rows * num_experts, 1.0f);
  const std::vector<uint8_t> fc1_experts_weights(num_experts * inter_size * (hidden_size / pack_size), 0);
  const std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * (inter_size / pack_size), 0);
  const std::vector<float> fc1_scales(num_experts * inter_size, 1.0f);
  const std::vector<float> fc2_scales(num_experts * hidden_size, 1.0f);
  const std::vector<float> dummy_output(num_rows * hidden_size, 0.0f);

  OpTester cuda_tester("QMoE", 1, onnxruntime::kMSDomain);
  cuda_tester.AddAttribute<int64_t>("k", 1);
  cuda_tester.AddAttribute<std::string>("activation_type", "identity");
  cuda_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cuda_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);
  cuda_tester.AddAttribute<std::string>("quant_type", "int");
  cuda_tester.AddAttribute<int64_t>("weights_prepacked", 0);

  const std::vector<int64_t> input_dims = {num_rows, hidden_size};
  const std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  const std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
  const std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  const std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
  const std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  const std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cuda_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cuda_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cuda_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cuda_tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims, ToFloat16(fc1_scales));
  cuda_tester.AddOptionalInputEdge<MLFloat16>();
  cuda_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cuda_tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims, ToFloat16(fc2_scales));
  cuda_tester.AddOptionalInputEdge<MLFloat16>();
  cuda_tester.AddOptionalInputEdge<uint8_t>();
  cuda_tester.AddOptionalInputEdge<MLFloat16>();
  cuda_tester.AddOptionalInputEdge<MLFloat16>();
  cuda_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(dummy_output));

  SessionOptions session_options;
  session_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "1";

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_execution_providers;
  cuda_execution_providers.push_back(std::move(cuda_ep));
  cuda_tester.Run(session_options,
                  OpTester::ExpectResult::kExpectFailure,
                  "QMoE weights_prepacked=0 requires PrePack to run",
                  {},
                  nullptr,
                  &cuda_execution_providers);
}

TEST(MoETest, QMoETest_Mixtral_Int4) {
  // This test uses FC3 (gated SiLU / Mixtral pattern) with dimensions too small for the
  // CUTLASS kernel (needs hidden_size >= 128, inter_size >= 128). CPU QMoE does not
  // support FC3. Skip until test data is regenerated with larger dimensions.
  GTEST_SKIP() << "Dimensions too small for CUTLASS kernel and CPU QMoE does not support FC3";
  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 64;
  int inter_size = 64;

  const std::vector<float> input = {
      -0.8477f, -0.0746f, 1.606f, -0.3242f, 0.4028f, 0.2384f, -0.0359f, -1.667f, -1.265f, -0.3035f, 0.5327f,
      1.109f, 1.111f, 0.533f, -0.5947f, -0.2009f, 0.4224f, -0.576f, 0.825f, 1.038f, -0.2722f, 0.0497f,
      1.963f, -1.075f, -0.8374f, 1.055f, 0.448f, -0.602f, -0.2874f, -1.311f, -0.0609f, -1.991f, -0.0732f,
      -1.49f, 0.6636f, -0.4053f, -1.603f, -1.088f, 0.09534f, -0.6807f, -0.3958f, 1.205f, -0.4275f, 0.82f,
      1.029f, 0.2693f, 1.229f, 1.116f, 0.718f, -0.827f, 2.527f, -1.041f, 1.042f, -2.771f, -0.654f,
      0.7144f, 0.6255f, -0.00957f, -0.2313f, 0.4663f, 2.803f, 0.0655f, 1.232f, 1.557f, -1.238f, -1.337f,
      0.1522f, -0.2783f, 0.2252f, 2.252f, 0.557f, -0.6885f, 1.16f, -0.5244f, -1.424f, -0.02344f, -1.09f,
      -0.749f, -1.118f, -2.6f, -1.308f, -0.742f, 0.3064f, 1.892f, 1.573f, -0.3843f, 0.6475f, 0.38f,
      -1.423f, -2.04f, 0.005592f, -0.5977f, 1.063f, -1.626f, -0.04883f, -2.041f, -0.502f, -0.8906f, -0.3987f,
      0.387f, 0.4644f, -1.419f, -1.35f, -0.9634f, -0.871f, -0.53f, 0.495f, -0.6157f, 0.6523f, -1.036f,
      -1.234f, 0.11566f, 0.2035f, -1.782f, 0.837f, -0.8955f, -1.392f, -0.4f, 0.6533f, -0.289f, -1.328f,
      0.528f, -1.269f, -0.581f, -0.4805f, -1.539f, -0.554f, 0.478f};
  const std::vector<float> router_probs = {-0.579f, -0.07007f, 0.0784f, 0.5327f};
  const std::vector<uint8_t> fc1_experts_weights = {
      31, 119, 6, 42, 175, 252, 107, 46, 177, 207, 30, 178, 230, 186, 37, 69, 175, 194, 74, 203, 73, 190, 129,
      112, 203, 106, 103, 156, 52, 121, 95, 101, 29, 149, 95, 107, 247, 189, 182, 92, 136, 49, 56, 227, 58, 71,
      26, 111, 192, 107, 253, 212, 206, 86, 171, 35, 130, 119, 32, 66, 17, 99, 14, 11, 188, 109, 242, 62, 124,
      127, 140, 70, 110, 81, 18, 90, 206, 12, 4, 240, 63, 17, 119, 191, 87, 245, 85, 170, 129, 213, 96, 249,
      91, 127, 68, 172, 34, 23, 102, 76, 222, 244, 156, 71, 35, 55, 34, 166, 194, 164, 32, 193, 94, 144, 60,
      200, 16, 255, 137, 29, 205, 211, 167, 39, 163, 178, 47, 244, 232, 70, 207, 190, 177, 105, 53, 88, 29, 246,
      150, 176, 166, 224, 69, 68, 245, 85, 155, 237, 114, 129, 3, 237, 88, 245, 165, 72, 194, 38, 125, 249, 110,
      235, 242, 116, 151, 65, 48, 196, 129, 79, 170, 63, 186, 95, 42, 173, 252, 235, 245, 207, 163, 38, 185, 69,
      163, 102, 131, 116, 124, 153, 135, 30, 123, 10, 210, 137, 115, 58, 209, 244, 239, 55, 16, 127, 190, 109, 168,
      152, 111, 29, 201, 31, 109, 56, 62, 1, 191, 50, 149, 179, 198, 241, 252, 100, 150, 66, 172, 200, 52, 4,
      132, 82, 141, 103, 161, 17, 185, 62, 255, 121, 158, 184, 31, 243, 93, 103, 243, 91, 99, 16, 143, 74, 27,
      19, 56, 100, 122, 183, 64, 131, 239, 183, 77, 143, 240, 194, 148, 42, 171, 112, 230, 204, 239, 224, 156, 245,
      50, 75, 174, 255, 86, 217, 246, 79, 17, 74, 87, 29, 218, 6, 1, 77, 251, 235, 251, 14, 177, 21, 26,
      219, 87, 203, 108, 241, 50, 69, 66, 104, 236, 111, 47, 51, 77, 8, 233, 220, 187, 162, 191, 15, 35, 196,
      159, 16, 155, 220, 123, 26, 147, 68, 84, 127, 167, 52, 183, 253, 246, 108, 59, 142, 65, 214, 173, 62, 165,
      204, 178, 95, 201, 211, 108, 132, 161, 220, 3, 191, 34, 251, 28, 245, 99, 160, 87, 43, 154, 244, 100, 76,
      164, 28, 159, 155, 110, 137, 90, 109, 217, 125, 145, 143, 215, 238, 127, 182, 165, 31, 181, 172, 6, 63, 149,
      133, 127, 247, 60, 91, 47, 87, 248, 214, 69, 225, 183, 202, 159, 210, 85, 2, 34, 33, 83, 91, 86, 243,
      84, 149, 146, 5, 104, 125, 178, 63, 182, 159, 166, 186, 95, 101, 250, 124, 119, 207, 199, 37, 65, 182, 78,
      218, 164, 193, 71, 19, 73, 88, 88, 25, 80, 60, 109, 161, 23, 78, 139, 248, 243, 122, 201, 53, 67, 68,
      140, 249, 116, 139, 39, 72, 197, 92, 54, 53, 209, 28, 226, 149, 237, 216, 239, 241, 223, 214, 52, 85, 240,
      237, 187, 106, 220, 186, 49, 87, 219, 235, 63, 213, 248, 176, 196, 135, 177, 79, 8, 66, 243, 18, 127, 201,
      16, 167, 252, 26, 95, 225, 154, 210, 202, 182, 227, 232, 249, 135, 47, 151, 254, 169, 34, 31, 159, 29, 233,
      228, 37, 213, 40, 245, 22, 47, 73, 213, 93, 242, 117, 75, 110, 248, 206, 74, 24, 139, 252, 174, 245, 101,
      161, 214, 58, 154, 202, 25, 147, 127, 100, 111, 217, 190, 167, 20, 16, 237, 167, 247, 247, 192, 54, 153, 63,
      187, 178, 64, 182, 209, 247, 218, 211, 45, 242, 132, 94, 137, 238, 184, 240, 250, 42, 193, 150, 139, 159, 242,
      149, 89, 22, 111, 9, 78, 194, 146, 249, 173, 185, 243, 147, 250, 228, 144, 123, 250, 49, 92, 231, 1, 152,
      179, 101, 178, 255, 94, 136, 6, 30, 77, 173, 137, 110, 56, 16, 90, 95, 115, 145, 113, 51, 172, 152, 242,
      119, 7, 186, 149, 168, 213, 228, 229, 133, 31, 40, 189, 19, 74, 88, 75, 134, 255, 17, 116, 208, 224, 242,
      252, 156, 153, 44, 165, 119, 23, 206, 175, 33, 213, 59, 243, 103, 244, 92, 130, 184, 162, 229, 49, 203, 157,
      208, 106, 156, 237, 218, 223, 235, 58, 203, 117, 228, 119, 127, 58, 169, 171, 166, 203, 180, 254, 149, 90, 37,
      117, 63, 98, 107, 130, 143, 179, 72, 168, 184, 61, 137, 185, 123, 57, 70, 90, 115, 78, 26, 36, 167, 237,
      145, 247, 220, 103, 129, 207, 183, 6, 56, 178, 46, 198, 245, 220, 42, 59, 134, 217, 186, 33, 50, 200, 145,
      84, 96, 139, 72, 115, 253, 221, 42, 177, 49, 22, 201, 183, 194, 46, 222, 94, 169, 233, 22, 252, 228, 48,
      91, 72, 201, 150, 203, 219, 247, 30, 117, 19, 131, 74, 235, 214, 223, 221, 244, 36, 170, 94, 84, 253, 52,
      186, 223, 127, 156, 248, 182, 74, 89, 223, 202, 255, 194, 151, 195, 115, 176, 180, 55, 194, 33, 14, 133, 39,
      250, 129, 142, 25, 49, 126, 47, 67, 215, 56, 116, 242, 117, 36, 98, 207, 78, 168, 150, 175, 109, 229, 54,
      45, 221, 205, 130, 52, 133, 208, 174, 234, 234, 188, 71, 250, 3, 43, 225, 57, 144, 225, 157, 202, 251, 194,
      242, 106, 188, 121, 239, 104, 206, 238, 116, 28, 253, 1, 62, 153, 193, 147, 24, 120, 70, 241, 148, 54, 227,
      159, 242, 208, 23, 14, 102, 41, 29, 254, 184, 52, 27, 45, 69, 137, 149, 23, 151, 123, 99, 190, 107, 247,
      81, 164, 57, 163, 77, 213, 250, 203, 235, 134, 85, 83, 1, 188, 70, 100, 70, 56, 217, 34, 55, 103, 123,
      9, 191, 147, 132, 36, 21, 38, 89, 82, 170, 166, 167, 129, 238, 171, 227, 254, 188, 14, 202, 249, 54, 158,
      12, 146, 139, 203, 113, 153, 163, 180, 104, 220, 210, 108, 77, 200, 183, 135, 118, 241, 170, 143, 238, 120, 68,
      215, 67, 133, 182, 222, 155, 12, 254, 157, 250, 87, 190, 226, 141, 62, 250, 212, 43, 218, 109, 229, 79, 213,
      249, 190, 225, 107, 238, 15, 233, 72, 56, 193, 64, 86, 156, 215, 170, 70, 69, 28, 221, 97, 21, 73, 8,
      184, 53, 138, 80, 239, 59, 117, 23, 39, 104, 167, 202, 47, 231, 253, 174, 226, 114, 40, 140, 3, 200, 198,
      134, 85, 10, 104, 50, 1, 75, 243, 1, 248, 148, 143, 180, 213, 47, 198, 76, 78, 12, 15, 128, 23, 144,
      82, 50, 132, 219, 73, 81, 45, 169, 39, 178, 85, 130, 111, 17, 111, 54, 13, 47, 184, 139, 245, 23, 74,
      202, 138, 183, 30, 136, 134, 120, 105, 95, 253, 89, 252, 67, 169, 149, 251, 4, 57, 175, 30, 79, 15, 104,
      34, 65, 252, 101, 211, 6, 81, 72, 40, 18, 240, 163, 255, 53, 132, 173, 124, 251, 145, 69, 98, 46, 137,
      179, 11, 113, 247, 158, 232, 179, 215, 14, 47, 185, 237, 58, 121, 159, 227, 227, 244, 73, 214, 181, 149, 47,
      93, 20, 245, 237, 166, 192, 82, 133, 84, 33, 124, 12, 229, 59, 44, 30, 62, 136, 243, 143, 119, 101, 253,
      36, 159, 245, 87, 175, 250, 223, 86, 185, 146, 51, 39, 248, 128, 212, 189, 62, 190, 232, 134, 148, 88, 158,
      113, 157, 130, 201, 122, 132, 250, 250, 87, 200, 79, 47, 148, 30, 232, 74, 199, 188, 175, 234, 8, 95, 141,
      253, 49, 158, 20, 135, 153, 195, 212, 255, 104, 68, 78, 75, 155, 253, 163, 146, 58, 124, 34, 231, 4, 138,
      241, 19, 36, 25, 114, 230, 167, 147, 72, 69, 242, 130, 46, 228, 130, 210, 149, 74, 248, 79, 251, 31, 180,
      249, 233, 148, 243, 47, 170, 153, 176, 185, 207, 111, 191, 191, 13, 180, 247, 72, 91, 196, 244, 187, 245, 16,
      188, 239, 66, 253, 181, 45, 52, 245, 26, 63, 112, 42, 68, 29, 216, 166, 57, 87, 252, 27, 159, 177, 143,
      142, 34, 244, 38, 214, 144, 243, 72, 165, 69, 30, 241, 164, 126, 164, 228, 142, 184, 251, 172, 51, 49, 202,
      241, 76, 203, 169, 210, 46, 204, 49, 108, 138, 164, 116, 188, 23, 163, 251, 107, 156, 159, 69, 125, 163, 93,
      80, 31, 97, 200, 83, 212, 111, 248, 154, 82, 128, 187, 232, 254, 195, 174, 195, 69, 204, 39, 67, 34, 78,
      253, 107, 2, 219, 245, 222, 22, 200, 83, 39, 220, 206, 47, 76, 118, 63, 67, 142, 24, 115, 97, 56, 233,
      199, 230, 206, 186, 91, 18, 141, 148, 43, 94, 25, 96, 140, 233, 165, 202, 242, 71, 209, 235, 173, 105, 40,
      82, 63, 165, 218, 95, 48, 89, 103, 89, 24, 100, 195, 208, 251, 28, 240, 235, 157, 207, 191, 52, 249, 203,
      134, 40, 153, 219, 102, 29, 32, 196, 71, 149, 124, 189, 183, 14, 244, 90, 123, 152, 206, 232, 70, 138, 111,
      231, 6, 237, 243, 69, 101, 118, 41, 70, 171, 97, 133, 207, 234, 21, 24, 223, 131, 55, 18, 253, 128, 63,
      179, 85, 174, 62, 36, 217, 191, 90, 142, 215, 79, 247, 16, 53, 200, 241, 85, 141, 194, 125, 75, 30, 237,
      4, 255, 247, 80, 69, 127, 204, 238, 106, 255, 98, 24, 206, 110, 57, 103, 90, 175, 100, 253, 151, 206, 90,
      33, 243, 52, 13, 51, 212, 99, 182, 10, 105, 41, 111, 22, 182, 207, 205, 81, 138, 203, 205, 99, 204, 125,
      152, 55, 3, 130, 116, 116, 106, 178, 74, 234, 56, 51, 20, 223, 180, 49, 155, 79, 53, 194, 227, 118, 116,
      165, 100, 196, 255, 91, 74, 250, 19, 177, 79, 188, 141, 91, 149, 65, 28, 76, 53, 185, 123, 164, 127, 74,
      168, 73, 65, 202, 112, 41, 111, 233, 240, 200, 241, 155, 181, 78, 136, 222, 55, 161, 1, 252, 241, 35, 54,
      223, 122, 32, 31, 59, 46, 177, 168, 246, 226, 25, 71, 24, 221, 120, 27, 118, 242, 216, 200, 252, 208, 174,
      165, 169, 177, 84, 249, 19, 236, 146, 17, 15, 29, 117, 31, 24, 67, 208, 105, 117, 100, 109, 19, 34, 159,
      82, 33, 194, 122, 211, 167, 239, 207, 153, 109, 211, 29, 209, 111, 224, 178, 249, 244, 35, 127, 72, 18, 36,
      211, 211, 251, 98, 142, 200, 131, 114, 93, 84, 51, 75, 87, 252, 180, 27, 111, 16, 238, 233, 186, 168, 181,
      99, 31, 43, 101, 0, 127, 202, 206, 219, 93, 42, 206, 142, 209, 28, 58, 252, 168, 119, 101, 236, 157, 53,
      61, 31, 129, 120, 81, 116, 129, 20, 20, 170, 181, 173, 33, 40, 67, 24, 78, 159, 158, 255, 210, 197, 193,
      73, 127, 254, 239, 216, 130, 121, 97, 249, 22, 136, 179, 221, 10, 40, 246, 89, 48, 36, 57, 186, 236, 58,
      110, 45, 217, 173, 46, 210, 225, 223, 235, 91, 55, 79, 82, 162, 17, 22, 223, 11, 1, 189, 11, 180, 47,
      191, 225, 173, 175, 46, 212, 158, 75, 68, 62, 99, 203, 115, 174, 139, 141, 81, 135, 75, 131, 221, 138, 247,
      155, 154, 56, 220, 123, 32, 28, 151, 47, 241, 14, 14, 84, 147, 39, 247, 99, 249, 187, 111, 26, 60, 140,
      194, 5, 130, 101, 114, 74, 229, 127, 52, 252, 233, 79, 13, 213, 46, 196, 171, 215, 250, 166, 165, 37, 171,
      143, 244, 108, 131, 87, 95, 248, 223, 169, 98, 195, 211, 227, 201, 122, 202, 170, 220, 98, 246, 166, 182, 115,
      177, 67, 189, 94, 175, 28, 9, 225, 100, 173, 217, 150, 32, 90, 72, 190, 222, 65, 169, 170, 175, 179, 194,
      42, 102, 122, 218, 174, 217, 177, 200, 249, 111, 195, 246, 172, 60, 60, 160, 113, 205, 27, 87, 233, 189, 95,
      10, 176, 72, 54, 206, 131, 114, 245, 31, 123, 31, 55, 15, 189, 165, 122, 93, 45, 65, 154, 252, 102, 37,
      39, 132, 85, 148, 140, 37, 110, 201, 35, 31, 54, 243, 174, 164, 230, 255, 55, 7, 220, 123, 246, 98, 31,
      12, 137, 63, 95, 44, 255, 165, 241, 78, 225, 172, 72, 119, 209, 63, 209, 116, 7, 152, 47, 38, 191, 210,
      155, 175, 100, 95, 147, 210, 30, 137, 13, 226, 201, 232, 63, 169, 236, 164, 29, 225, 254, 225, 181, 253, 155,
      211, 100, 33, 255, 75, 14, 86, 238, 0, 124, 199, 102, 67, 230, 84, 161, 90, 169, 181, 140, 11, 133, 41,
      201, 177, 136, 234, 155, 31, 67, 254, 100, 21, 224, 17, 164, 218, 193, 147, 191, 64, 21, 125, 43, 63, 221,
      90, 160, 95, 68, 244, 34, 115, 79, 241, 171, 151, 21, 167, 122, 221, 1, 236, 85, 41, 156, 73, 72, 218,
      146, 171, 44, 62, 233, 174, 75, 237, 232, 23, 207, 64, 128, 165, 209, 135, 170, 49, 143, 238, 109, 228, 200,
      67, 114, 135, 72, 51, 253, 107, 74, 202, 58, 160, 186, 194, 63, 97, 175, 241, 180, 58, 187, 19, 212, 175,
      225, 184, 224, 21, 190, 67, 154, 106, 170, 129, 195, 92, 4, 122, 15, 252, 88, 182, 218, 127, 20, 137, 226,
      143, 205, 70, 126, 137, 190, 120, 186, 175, 44, 169, 76, 26, 45, 213, 99, 171, 144, 90, 100, 175, 104, 66,
      159, 155, 218, 31, 54, 242, 98, 50, 204, 82, 65, 8, 138, 143, 109, 222, 40, 65, 244, 4, 165, 153, 195,
      253, 217, 11, 186, 54, 53, 37, 66, 35, 95, 241, 185, 18, 18, 37, 245, 63, 157, 139, 109, 250, 235, 110,
      6, 85, 234, 85, 1, 211, 82, 159, 245, 71, 92, 52, 233, 124, 250, 244, 174, 218, 47, 37, 232, 203, 199,
      243, 175, 244, 166, 44, 108, 93, 79, 51, 241, 241, 64, 124, 61, 69, 246, 241, 140, 255, 79, 235, 58, 196,
      255, 216, 207, 193, 143, 86, 89, 30, 158, 57, 120, 55, 60, 130, 220, 148, 89, 138, 253, 210, 86, 125, 104,
      83, 244, 158, 100, 56, 216, 103, 13, 77, 106, 41, 163, 164, 237, 2, 95, 141, 76, 250, 238, 199, 129, 41,
      69, 128, 127, 87, 22, 124, 27, 95, 172, 233, 47, 29, 198, 57, 234, 251, 207, 107, 215, 49, 147, 20, 116,
      146, 77, 48, 138, 156, 137, 218, 160, 168, 31, 175, 101, 61, 169, 170, 6, 113, 255, 228, 156, 107, 248, 243,
      109, 201, 245, 134, 240, 37, 96, 165, 63, 94, 56, 191, 97, 58, 57, 125, 247, 21, 249, 124, 143, 140, 235,
      234, 45, 188, 27, 239, 149, 235, 122, 76, 82, 124, 242, 163, 65, 95, 1, 64, 227, 26, 55, 211, 151, 138,
      124, 191, 100, 88, 55, 80, 16, 31, 101, 51, 138, 141, 235, 99, 60, 118, 40, 168, 141, 138, 25, 97, 167,
      37, 174, 98, 213, 26, 76, 121, 235, 113, 247, 131, 114, 133, 252, 245, 244, 33, 108, 253, 122, 201, 143, 92,
      140, 29, 201, 197, 144, 65, 189, 251, 147, 211, 43, 234, 170, 44, 101, 36, 40, 29, 19, 216, 134, 82, 71,
      147, 110, 170, 208, 184, 3, 212, 1, 121, 63, 188, 135, 53, 31, 89, 232, 63, 158, 70, 30, 231, 102, 65,
      22, 150, 244, 121, 148, 90, 103, 155, 242, 48, 185, 245, 59, 105, 182, 110, 59, 127, 33, 248, 52, 192, 125,
      218, 50, 169, 252, 255, 122, 71, 158, 255, 158, 178, 65, 164, 223, 137, 182, 78, 253, 118, 66, 130, 238, 90,
      81, 214, 160, 24, 40, 113, 112, 63, 111, 218, 234, 203, 30, 49, 97, 137, 97, 30, 212, 158, 89, 156, 224,
      56, 70, 203, 48, 119, 148, 255, 18, 4, 35, 117, 42, 79, 248, 178, 226, 250, 67, 250, 71, 215, 170, 197,
      176, 124, 207, 83, 34, 72, 73, 46, 133, 1, 15, 117, 18, 225, 95, 60, 186, 169, 53, 176, 136, 63, 104,
      75, 219, 115, 78, 239, 211, 44, 198, 130, 156, 166, 98, 215, 144, 240, 93, 209, 254, 255, 251, 150, 124, 172,
      228, 6, 183, 79, 127, 241, 62, 121, 129, 183, 228, 30, 237, 244, 108, 246, 65, 75, 241, 145, 82, 217, 210,
      122, 79, 79, 244, 167, 167, 119, 20, 72, 202, 139, 57, 212, 141, 246, 239, 50, 204, 179, 122, 156, 146, 216,
      243, 73, 159, 31, 182, 227, 67, 191, 130, 248, 227, 191, 85, 85, 241, 37, 147, 37, 171, 50, 38, 240, 241,
      140, 61, 83, 236, 116, 62, 245, 44, 91, 227, 189, 220, 246, 243, 186, 175, 145, 88, 23, 61, 175, 6, 175,
      148, 255, 118, 53, 237, 38, 109, 110, 254, 191, 236, 58, 3, 73, 194, 159, 83, 80, 145, 251, 129, 101, 216,
      241, 156, 178, 255, 46, 230, 198, 160, 190, 236, 187, 91, 48, 127, 111, 250, 168, 218, 87, 111, 95, 101, 207,
      168, 98, 64, 199, 207, 170, 37, 101, 125, 190, 166, 139, 157, 39, 246, 39, 21, 137, 250, 172, 253, 80, 113,
      5, 18, 155, 92, 18, 180, 111, 239, 165, 246, 91, 87, 57, 223, 206, 62, 245, 227, 121, 84, 180, 162, 25,
      240, 223, 225, 111, 28, 203, 8, 20, 203, 17, 235, 56, 233, 198, 158, 250, 181, 57, 191, 173, 72, 136, 169,
      232, 179, 51, 222, 165, 152, 85, 125, 199, 118, 82, 182, 77, 240, 249, 255, 209, 127, 49, 190, 235, 18, 143,
      125, 52, 60, 12, 111, 95, 115, 117, 111, 220, 90, 159, 147, 26, 152, 84, 137, 70, 8, 182, 12, 249, 252,
      191, 247, 49, 180, 130, 239, 87, 201, 99, 86, 80, 122, 26, 104, 243, 250, 245, 225, 46, 242, 102, 143, 15,
      154, 27, 134, 236, 86, 50, 31, 152, 254, 47, 127, 249, 204, 245, 53, 207, 100, 84, 47, 88, 210, 213, 161,
      227, 71, 182, 30, 141, 127, 78, 83, 33, 213, 131, 33, 169, 222, 202, 80, 179, 164, 227, 222, 218, 65, 254,
      228, 75, 140, 218, 190, 214, 157, 224, 118, 74, 82, 134, 148, 70, 143, 225, 35, 43, 53, 91, 177, 240, 239,
      132, 139, 226, 245, 39, 97, 45, 74, 251, 87, 175, 185, 187, 205, 83, 33, 45, 155, 164, 132, 61, 146, 56,
      229, 175, 110, 94, 204, 177, 111, 126, 206, 105, 67, 215, 205, 253, 237, 46, 89, 116, 14, 35, 92, 123, 43,
      242, 249, 35, 227, 233, 16, 251, 175, 245, 103, 127, 196, 233, 43, 23, 135, 27, 189, 97, 29, 5, 178, 37,
      52, 27, 140, 229, 63, 86, 138, 32, 34, 87, 38, 206, 32, 155, 191, 48, 61, 63, 113, 27, 43, 92, 149,
      143, 47, 137, 116, 150, 127, 49, 82, 25, 22, 99, 7, 61, 255, 246, 203, 83, 132, 166, 44, 75, 211, 131,
      94, 26, 127, 105, 73, 189, 163, 182, 57, 95, 248, 34, 97, 244, 255, 47, 131, 55, 188, 97, 80, 205, 124,
      184, 111, 50, 168, 44, 159, 249, 74, 164, 200, 227, 33, 253, 24, 217, 38, 135, 130, 87, 114, 90, 199, 216,
      27, 127, 252, 246, 134, 80, 49, 147, 186, 122, 107, 70, 249, 30, 95, 72, 34, 205, 229, 95, 79, 142, 71,
      194, 133, 36, 215, 188, 100, 111, 129, 190, 93, 113, 227, 167, 29, 254, 212, 38, 141, 182, 105, 164, 121, 147,
      225, 252, 137, 32, 50, 31, 64, 35, 168, 78, 185, 31, 38, 73, 255, 33, 60, 38, 125, 49, 12, 242, 248,
      94, 104, 114, 78, 255, 101, 76, 158, 95, 13, 212, 169, 242, 116, 250, 154, 184, 86, 68, 172, 166, 95, 182,
      21, 191, 24, 145, 207, 26, 119, 201, 58, 126, 203, 10, 147, 111, 165, 251, 99, 64, 26, 77, 218, 208, 244,
      239, 93, 226, 254, 68, 198, 82, 188, 47, 200, 15, 44, 100, 33, 90, 55, 71, 249, 148, 175, 219, 59, 159,
      191, 127, 8, 125, 206, 27, 67, 136, 234, 89, 12, 203, 163, 52, 78, 167, 144, 68, 112, 127, 176, 235, 113,
      116, 39, 217, 143, 26, 23, 232, 23, 54, 203, 231, 43, 41, 22, 56, 137, 150, 111, 47, 107, 242, 146, 190,
      63, 154, 248, 73, 25, 251, 161, 77, 187, 158, 34, 87, 21, 224, 133, 0, 239, 190, 152, 87, 128, 229, 24,
      130, 40, 95, 111, 241, 251, 251, 12, 26, 254, 81, 111, 47, 91, 137, 22, 61, 202, 85, 175, 174, 169, 196,
      218, 253, 1, 192, 54, 46, 81, 190, 33, 17, 248, 66, 248, 240, 226, 49, 52, 148, 177, 246, 20, 41, 211,
      128, 12, 247, 221, 130, 118, 113, 183, 47, 230, 233, 57, 162, 183, 74, 28, 195, 132, 210, 124, 248, 255, 166,
      51, 207, 135, 157, 115, 248, 238, 126, 195, 175, 177, 36, 68, 225, 216, 205, 214, 200, 243, 248, 22, 231, 138,
      168, 34, 210, 145, 236, 23, 175, 25, 248, 208, 77, 215, 160, 242, 209, 168, 25, 216, 191, 84, 182, 191, 143,
      243, 240, 182, 210, 241, 170, 216, 182, 29, 87, 168, 99, 128, 216, 169, 245, 68, 33, 153, 164, 28, 56, 79,
      157, 145, 17, 158, 124, 171, 255, 225, 225, 252, 43, 61, 249, 165, 40, 197, 111, 7, 8, 166, 211, 241, 84,
      185, 240, 35, 48, 146, 167, 163, 110, 247, 119, 50, 112, 101, 239, 87, 66, 34, 107, 106, 138, 29, 234, 43,
      151, 206, 254, 95, 142, 85, 46, 74, 46, 89, 200, 199, 210, 198, 156, 126, 247, 17, 44, 140, 84, 159, 187,
      213, 202, 151, 46, 234, 165, 78, 247, 210, 250, 228, 223, 85, 27, 137, 127, 131, 91, 42, 124, 97, 237, 139,
      24, 175, 47, 104, 81, 163, 210, 56, 193, 107, 115, 203, 250, 184, 140, 206, 51, 13, 118, 156, 164, 132, 226,
      127, 213, 32, 23, 66, 55, 137, 154, 29, 125, 226, 85, 38, 207, 180, 91, 24, 35, 167, 224, 138, 93, 66,
      40, 85, 185, 43, 88, 184, 142, 37, 3, 234, 220, 165, 127, 219, 130, 48, 49, 175, 55, 79, 196, 50, 197,
      141, 111, 217, 44, 209, 150, 57, 253, 208, 107, 222, 146, 130, 22, 161, 118, 96, 177, 219, 128, 67, 114, 228,
      143, 247, 49, 204, 69, 25, 7, 169, 52, 35, 168, 255, 181, 86, 79, 169, 135, 151, 21, 74, 119, 106, 15,
      26, 68, 16, 118, 42, 93, 207, 189, 59, 138, 190, 206, 204, 123, 53, 197, 83, 61, 149, 123, 54, 26, 247,
      22, 63, 254, 255, 24, 154, 63, 232, 29, 97, 233, 128, 207, 79, 169, 239, 251, 57, 38, 51, 102, 63, 72,
      226, 234, 250, 254, 190, 141, 96, 137, 54, 234, 111, 111, 190, 106, 95, 233, 147, 226, 15, 63, 251, 41, 53,
      58, 132, 97, 73, 38, 80, 213, 162, 250, 125, 179, 154, 27, 229, 171, 159, 237, 145, 128, 207, 211, 74, 134,
      249, 155, 25, 47, 163, 247, 183, 184, 94, 59, 212, 227, 188, 162, 253, 114, 79, 250, 92, 130, 111, 42, 75,
      45, 136, 209, 108, 30, 100, 93, 221, 142, 125, 228, 143, 255, 219, 238, 67, 182, 40, 158, 133, 21, 248, 143,
      195, 220, 193, 96, 111, 120, 208, 6, 249, 42, 225, 30, 159, 223, 116, 203, 35, 243, 9, 147, 251, 29, 217,
      184, 125, 131, 165, 79, 85, 53, 127, 47, 146, 229, 245, 185, 173, 37, 230, 209, 118, 108, 78, 114, 246, 208,
      200, 186, 239, 244, 2, 188, 114, 255, 76, 89, 86, 157, 237, 225, 41, 43, 43, 86, 89, 52, 132, 26, 148,
      253, 31, 185, 174, 248, 131, 92, 34, 173, 11, 72, 198, 145, 215, 180, 106, 173, 44, 99, 46, 116, 31, 51,
      71, 77, 203, 138, 17, 55, 70, 183, 181, 155, 151, 229, 81, 204, 249, 19, 158, 237, 134, 96, 126, 88, 139,
      111, 138, 247, 110, 253, 217, 107, 111, 77, 234, 130, 199, 177, 143, 199, 47, 123, 155, 32, 99, 117, 16, 181,
      155, 87, 166, 244, 42, 36, 251, 108, 15, 45, 108, 33, 85, 97, 117, 203, 142, 45, 102, 225, 249, 255, 191,
      19, 17, 255, 229, 214, 84, 89, 201, 41, 249, 242, 208, 188, 220, 51, 140, 233, 224, 97, 192, 114, 242, 46,
      23, 241, 29, 59, 150, 179, 21, 239, 109, 77, 205, 63, 81, 102, 116, 33, 87, 84, 190, 255, 249, 71, 40,
      97, 117};
  const std::vector<uint8_t> fc2_experts_weights = {
      194, 252, 114, 86, 142, 245, 201, 173, 104, 137, 96, 176, 255, 143, 35, 68, 92, 181, 221, 16, 7, 116, 143,
      99, 41, 223, 180, 116, 244, 139, 190, 255, 138, 100, 204, 207, 5, 23, 77, 101, 217, 92, 134, 241, 138, 183,
      146, 41, 171, 194, 248, 212, 175, 20, 12, 202, 83, 43, 255, 239, 87, 77, 70, 168, 40, 227, 76, 38, 121,
      67, 146, 21, 139, 160, 109, 169, 212, 66, 131, 219, 235, 113, 1, 246, 88, 5, 180, 106, 247, 179, 253, 170,
      103, 204, 178, 62, 173, 37, 80, 214, 130, 113, 23, 137, 0, 183, 32, 115, 1, 253, 222, 223, 19, 114, 243,
      57, 208, 115, 251, 89, 66, 63, 81, 9, 83, 94, 128, 146, 23, 85, 208, 96, 142, 87, 190, 157, 83, 0,
      169, 183, 221, 110, 96, 118, 39, 215, 61, 44, 19, 42, 64, 207, 130, 184, 50, 255, 132, 126, 191, 108, 110,
      226, 232, 44, 175, 234, 228, 67, 85, 191, 56, 90, 44, 219, 49, 21, 246, 113, 70, 69, 67, 118, 132, 241,
      118, 75, 185, 159, 254, 44, 183, 104, 40, 86, 18, 95, 106, 13, 65, 216, 245, 242, 46, 230, 223, 59, 167,
      239, 230, 159, 252, 41, 141, 193, 65, 187, 74, 95, 187, 86, 99, 68, 255, 230, 93, 241, 138, 74, 56, 162,
      138, 150, 143, 220, 59, 152, 212, 218, 251, 165, 13, 207, 184, 78, 11, 49, 79, 245, 181, 136, 197, 239, 110,
      7, 251, 90, 78, 151, 23, 74, 27, 159, 97, 82, 79, 181, 228, 111, 136, 86, 227, 86, 170, 147, 172, 245,
      65, 136, 162, 119, 224, 94, 252, 1, 190, 25, 96, 181, 83, 95, 237, 132, 22, 19, 251, 63, 94, 99, 137,
      62, 219, 223, 159, 206, 246, 96, 71, 85, 255, 119, 55, 190, 175, 108, 175, 189, 101, 172, 224, 175, 191, 204,
      137, 52, 46, 107, 6, 177, 160, 96, 253, 131, 123, 215, 95, 107, 220, 123, 143, 9, 251, 101, 87, 241, 163,
      162, 66, 97, 166, 206, 251, 244, 206, 125, 242, 246, 9, 110, 49, 168, 82, 249, 238, 214, 209, 91, 212, 76,
      122, 12, 88, 247, 41, 241, 253, 29, 203, 226, 51, 60, 17, 0, 255, 193, 237, 213, 45, 96, 201, 175, 111,
      148, 209, 250, 140, 237, 197, 55, 248, 158, 236, 245, 78, 111, 27, 103, 235, 191, 23, 250, 216, 95, 2, 45,
      79, 46, 116, 153, 169, 43, 183, 41, 86, 119, 228, 158, 138, 62, 145, 30, 15, 199, 59, 223, 67, 124, 237,
      186, 98, 138, 94, 184, 156, 249, 117, 40, 223, 158, 38, 91, 176, 94, 160, 254, 23, 175, 142, 235, 3, 146,
      251, 255, 205, 42, 122, 255, 26, 149, 136, 207, 79, 31, 218, 150, 248, 161, 14, 220, 55, 76, 135, 3, 195,
      108, 35, 239, 171, 120, 201, 250, 219, 16, 72, 181, 232, 245, 63, 243, 197, 68, 177, 83, 53, 106, 254, 28,
      89, 45, 31, 243, 162, 150, 122, 82, 155, 74, 77, 220, 46, 49, 254, 105, 43, 18, 62, 234, 13, 189, 85,
      189, 222, 206, 63, 99, 171, 247, 242, 136, 166, 167, 31, 166, 174, 85, 232, 25, 7, 179, 91, 218, 212, 129,
      68, 31, 208, 101, 133, 45, 11, 163, 41, 104, 190, 169, 239, 254, 196, 162, 158, 220, 62, 239, 25, 225, 48,
      20, 147, 119, 151, 178, 216, 238, 185, 119, 191, 34, 191, 87, 230, 94, 212, 148, 130, 130, 15, 101, 26, 99,
      130, 17, 252, 229, 56, 247, 175, 12, 40, 12, 224, 210, 247, 23, 152, 63, 67, 131, 220, 86, 217, 69, 116,
      5, 240, 239, 173, 165, 138, 254, 182, 208, 192, 27, 39, 97, 171, 251, 22, 154, 110, 149, 89, 107, 245, 100,
      230, 64, 137, 85, 62, 152, 166, 223, 246, 250, 186, 183, 146, 212, 207, 169, 191, 247, 246, 19, 248, 49, 47,
      180, 213, 25, 108, 253, 153, 189, 149, 193, 191, 87, 44, 88, 63, 62, 254, 166, 55, 191, 207, 81, 143, 161,
      247, 15, 55, 174, 158, 214, 233, 181, 156, 134, 109, 159, 179, 245, 28, 197, 86, 249, 145, 113, 14, 44, 101,
      41, 221, 15, 230, 151, 232, 47, 69, 199, 172, 147, 143, 113, 70, 70, 222, 162, 71, 29, 38, 71, 183, 252,
      22, 233, 175, 247, 29, 95, 118, 153, 17, 83, 31, 125, 137, 238, 255, 47, 92, 104, 211, 106, 104, 102, 39,
      86, 228, 92, 105, 108, 84, 175, 159, 185, 190, 121, 144, 74, 222, 159, 38, 35, 192, 209, 205, 1, 211, 73,
      84, 162, 192, 236, 250, 155, 245, 223, 52, 242, 14, 242, 37, 245, 170, 110, 181, 82, 251, 244, 143, 71, 56,
      231, 86, 236, 55, 130, 21, 232, 156, 40, 96, 113, 17, 253, 35, 168, 107, 247, 254, 253, 147, 252, 85, 222,
      211, 238, 231, 70, 149, 174, 246, 194, 212, 163, 254, 59, 233, 45, 182, 244, 132, 173, 33, 64, 79, 71, 118,
      42, 134, 0, 202, 30, 47, 103, 212, 40, 232, 124, 116, 75, 29, 254, 253, 124, 47, 217, 35, 87, 114, 228,
      75, 47, 54, 212, 222, 225, 72, 99, 109, 198, 88, 63, 1, 98, 76, 242, 18, 193, 24, 47, 1, 103, 153,
      41, 11, 140, 100, 217, 110, 96, 24, 136, 97, 116, 138, 42, 48, 114, 232, 164, 201, 62, 236, 255, 202, 249,
      253, 66, 246, 68, 21, 220, 15, 47, 239, 205, 150, 127, 104, 154, 20, 176, 48, 41, 90, 149, 181, 251, 19,
      232, 41, 64, 55, 91, 155, 23, 33, 100, 102, 238, 79, 19, 169, 49, 152, 105, 98, 137, 250, 166, 38, 33,
      254, 251, 160, 142, 200, 171, 236, 247, 153, 177, 36, 178, 219, 228, 253, 145, 12, 152, 68, 34, 89, 163, 34,
      47, 247, 140, 228, 45, 179, 100, 3, 134, 4, 151, 67, 144, 180, 40, 245, 234, 113, 29, 149, 201, 44, 132,
      86, 90, 228, 121, 64, 66, 78, 194, 200, 235, 92, 133, 3, 157, 126, 152, 191, 85, 222, 85, 168, 83, 202,
      178, 216, 154, 101, 95, 186, 106, 183, 53, 103, 202, 110, 72, 37, 81, 15, 194, 228, 251, 255, 246, 219, 65,
      168, 83, 30, 11, 63, 23, 28, 87, 106, 185, 138, 246, 89, 132, 45, 255, 107, 85, 18, 90, 244, 158, 214,
      76, 81, 90, 249, 151, 58, 232, 91, 231, 99, 214, 141, 165, 69, 140, 234, 193, 72, 73, 158, 23, 38, 244,
      213, 247, 250, 8, 88, 241, 159, 91, 111, 253, 99, 66, 232, 155, 17, 167, 61, 199, 236, 255, 204, 253, 118,
      34, 194, 8, 136, 237, 159, 117, 63, 247, 30, 83, 229, 226, 10, 253, 235, 148, 85, 85, 127, 221, 153, 165,
      245, 141, 239, 227, 216, 24, 91, 50, 246, 171, 215, 113, 184, 17, 153, 164, 60, 29, 105, 190, 123, 143, 111,
      177, 225, 68, 78, 210, 217, 28, 191, 26, 255, 187, 255, 166, 69, 239, 79, 236, 85, 155, 34, 6, 247, 172,
      61, 41, 68, 159, 247, 54, 159, 103, 98, 112, 209, 188, 252, 148, 125, 113, 230, 167, 154, 53, 143, 168, 172,
      66, 253, 52, 209, 191, 253, 238, 9, 46, 240, 23, 175, 148, 85, 54, 116, 55, 214, 253, 152, 175, 6, 98,
      158, 109, 204, 219, 107, 107, 109, 206, 150, 57, 250, 125, 170, 229, 20, 175, 104, 90, 250, 54, 112, 46, 92,
      250, 156, 181, 177, 177, 166, 167, 250, 181, 209, 208, 127, 235, 151, 133, 86, 147, 42, 42, 84, 133, 136, 170,
      202, 172, 28, 115, 160, 106, 251, 79, 205, 177, 67, 169, 69, 184, 109, 207, 164, 38, 247, 16, 245, 251, 129,
      246, 85, 156, 255, 218, 74, 212, 246, 81, 196, 161, 196, 23, 174, 234, 115, 35, 56, 126, 133, 16, 65, 139,
      113, 63, 243, 23, 18, 211, 30, 236, 167, 189, 129, 111, 15, 228, 254, 120, 127, 141, 215, 232, 119, 119, 113,
      127, 254, 111, 241, 86, 14, 154, 50, 201, 146, 87, 25, 239, 39, 24, 232, 94, 185, 226, 130, 95, 0, 127,
      6, 142, 209, 31, 203, 85, 215, 81, 22, 189, 109, 109, 252, 219, 175, 70, 246, 194, 198, 21, 227, 52, 247,
      149, 32, 242, 126, 5, 236, 140, 73, 117, 177, 169, 162, 68, 44, 132, 68, 77, 110, 54, 130, 161, 235, 118,
      154, 250, 47, 75, 249, 8, 169, 195, 231, 244, 192, 33, 155, 133, 27, 103, 173, 213, 231, 168, 44, 225, 30,
      9, 29, 19, 172, 211, 253, 226, 31, 191, 0, 237, 92, 149, 224, 121, 198, 142, 200, 255, 28, 33, 22, 196,
      66, 213, 73, 104, 64, 133, 221, 181, 147, 69, 105, 234, 243, 97, 169, 130, 242, 229, 171, 202, 210, 53, 29,
      97, 103, 20, 53, 224, 37, 74, 155, 98, 73, 255, 124, 188, 118, 166, 188, 103, 236, 19, 39, 106, 205, 254,
      161, 251, 181, 191, 223, 211, 79, 153, 209, 128, 160, 205, 36, 186, 39, 22, 187, 92, 237, 130, 212, 8, 190,
      91, 166, 145, 223, 232, 83, 88, 94, 242, 164, 94, 97, 120, 246, 31, 243, 57, 59, 100, 221, 60, 1, 39,
      178, 111, 61, 50, 115, 127, 72, 103, 226, 15, 221, 46, 237, 91, 111, 230, 43, 181, 183, 23, 146, 24, 67,
      248, 191, 168, 74, 170, 95, 163, 177, 171, 47, 80, 175, 11, 13, 111, 125, 254, 55, 202, 56, 216, 103, 216,
      134, 50, 222, 169, 211, 118, 107, 135, 221, 90, 102, 69, 142, 108, 40, 223, 86, 15, 157, 79, 110, 198, 12,
      70, 54, 184, 118, 159, 87, 150, 78, 25, 254, 251, 196, 205, 153, 157, 43, 17, 54, 214, 210, 174, 22, 238,
      161, 175, 239, 239, 28, 128, 234, 25, 67, 97, 136, 171, 177, 12, 95, 212, 195, 251, 50, 246, 175, 61, 3,
      89, 68, 142, 41, 43, 114, 67, 130, 48, 53, 106, 47, 119, 106, 253, 129, 242, 254, 223, 90, 77, 56, 221,
      31, 140, 194, 65, 80, 73, 228, 3, 94, 181, 72, 206, 223, 81, 171, 103, 203, 225, 152, 9, 204, 170, 247,
      26, 155, 197, 65, 239, 112, 5, 22, 72, 142, 69, 172, 127, 13, 131, 246, 189, 175, 242, 157, 130, 212, 59,
      175, 254, 55, 144, 163, 60, 59, 194, 103, 144, 95, 123, 18, 234, 183, 239, 45, 52, 212, 242, 172, 164, 128,
      50, 69, 183, 85, 18, 116, 120, 242, 151, 179, 4, 246, 161, 74, 69, 170, 39, 16, 84, 192, 27, 178, 122,
      30, 248, 79, 135, 98, 118, 26, 212, 52, 0, 175, 51, 29, 222, 134, 114, 206, 37, 167, 13, 110, 143, 136,
      164, 249, 59, 209, 230, 181, 205, 237, 203, 28, 177, 224, 228, 102, 39, 156, 22, 204, 142, 160, 195, 51, 207,
      56, 255, 63, 24, 123, 153, 92, 90, 244, 117, 187, 115, 21, 85, 129, 166, 122, 31, 19, 214, 173, 76, 255,
      204, 133, 127, 106, 94, 133, 225, 222, 97, 100, 170, 103, 11, 227, 2, 144, 174, 103, 177, 168, 133, 19, 83,
      255, 100, 55, 128, 230, 142, 163, 136, 206, 45, 122, 78, 99, 236, 150, 133, 34, 71, 93, 245, 246, 251, 149,
      65, 14, 240, 34, 234, 100, 250, 142, 80, 39, 97, 61, 76, 75, 85, 49, 113, 29, 99, 223, 156, 243, 90,
      240, 121, 208, 51, 204, 72, 17, 255, 105, 228, 128, 249, 147, 253, 127, 160, 38, 207, 22, 79, 137, 226, 207,
      218, 213, 166, 154, 177, 4, 204, 235, 182, 59, 245, 206, 213, 244, 135, 71, 25, 92, 216, 142, 81, 129, 255,
      100, 99, 36, 38, 79, 47, 168, 252, 7, 146, 46, 191, 72, 131, 83, 98, 121, 94, 178, 31, 41, 229, 220,
      210, 255, 209, 16, 40, 159, 126, 189, 250, 10, 89, 245, 39, 222, 208, 124, 201, 127, 84, 69, 125, 186, 65,
      250, 52, 79, 165, 45, 232, 234, 250, 162, 85, 220, 186, 165, 229, 136, 255, 72, 34, 47, 239, 15, 213, 129,
      25, 247, 146, 116, 222, 242, 57, 229, 28, 34, 180, 222, 205, 162, 167, 237, 191, 141, 142, 164, 253, 44, 254,
      44, 183, 245, 250, 227, 65, 192, 165, 238, 39, 33, 210, 236, 204, 139, 95, 88, 140, 115, 116, 46, 131, 119,
      2, 141, 243, 37, 247, 49, 36, 56, 63, 218, 242, 28, 149, 100, 49, 87, 183, 210, 250, 168, 148, 244, 117,
      68, 159, 53, 247, 76, 57, 82, 245, 63, 105, 3, 236, 94, 205, 232, 36, 248, 156, 254, 185, 7, 139, 218,
      99, 207, 166, 54, 148, 52, 97, 102, 62, 230, 143, 115, 170, 72, 140, 38, 202, 147, 49, 242, 245, 165, 219,
      158, 240, 111, 177, 193, 81, 176, 181, 14, 18, 110, 70, 55, 99, 80, 249, 127, 161, 29, 51, 220, 143, 44,
      204, 79, 168, 245, 244, 58, 254, 108, 184, 34, 198, 152, 196, 218, 239, 245, 61, 138, 145, 221, 65, 4, 68,
      104, 6, 39, 40, 241, 99, 104, 203, 108, 255, 127, 123, 246, 118, 93, 230, 70, 242, 114, 117, 189, 127, 84,
      233, 2, 183, 88, 47, 243, 217, 98, 89, 225, 89, 31, 138, 2, 221, 5, 79, 247, 63, 120, 99, 71, 184,
      62, 131, 142, 217, 209, 215, 251, 125, 126, 54, 109, 89, 193, 234, 51, 215, 72, 85, 140, 245, 39, 243, 24,
      70, 4, 18, 120, 151, 166, 170, 154, 105, 160, 65, 185, 132, 227, 238, 52, 188, 219, 71, 233, 130, 91, 93,
      239, 218, 72, 195, 199, 157, 199, 211, 98, 202, 118, 163, 28, 178, 64, 181, 169, 2, 191, 74, 161, 254, 19,
      33, 154, 163, 191, 187, 158, 232, 245, 255, 250, 205, 62, 70, 111, 63, 158, 94, 136, 253, 46, 220, 41, 174,
      183, 211, 91, 15, 136, 2, 75, 124, 246, 236, 204, 196, 66, 238, 79, 122, 225, 219, 108, 228, 156, 172, 39,
      107, 163, 191, 196, 182, 29, 69, 147, 250, 36, 42, 188, 116, 196, 163, 146, 161, 255, 228, 4, 125, 253, 124,
      72, 248, 19, 108, 223, 250, 103, 242, 78, 245, 90, 53, 58, 16, 42, 31, 177, 173, 245, 126, 84, 52, 55,
      243, 148, 239, 57, 126, 220, 252, 172, 72, 175, 65, 119, 38, 188, 178, 7, 180, 76, 95, 31, 230, 253, 123,
      81, 187, 184, 39, 111, 218, 94, 251, 187, 175, 24, 158, 90, 128, 130, 145, 33, 86, 46, 143, 33, 175, 134,
      215, 99, 54, 214, 70, 28, 204, 205, 238, 252, 172, 166, 210, 42, 11, 213, 67, 143, 48, 145, 152, 19, 147,
      18, 31, 17, 174, 233, 186, 132, 215, 24, 205, 87, 250, 100, 128, 249, 20, 66, 3, 79, 50, 88, 42, 145,
      156, 45, 246, 169, 230, 163, 130, 13, 164, 127, 76, 150, 60, 101, 209, 112, 34, 247, 117, 197, 23, 114, 164,
      27, 142, 56, 227, 79, 49, 159, 21, 177, 124, 212, 89, 250, 113, 6, 16, 159, 33, 231, 238, 80, 245, 195,
      207, 248, 205, 253, 145, 154, 220, 96, 164, 140, 231, 89, 161, 6, 213, 163, 30, 123, 232, 102, 188, 217, 159,
      128, 16, 29, 27, 28, 199, 114, 212, 95, 111, 255, 109, 215, 5, 101, 191, 216, 134, 194, 108, 98, 167, 61,
      218, 49, 159, 10, 95, 159, 158, 166, 37, 129, 62, 151, 144, 55, 36, 54, 107, 156, 216, 219, 203, 138, 36,
      250, 220, 215, 237, 226, 88, 159, 136, 107, 117, 153, 239, 155, 18, 156, 69, 49, 2, 118, 142, 241, 41, 168,
      56, 155, 250, 105, 112, 217, 189, 214, 164, 162, 244, 52, 95, 36, 35, 191, 66, 51, 75, 192, 171, 103, 118,
      93, 237, 124, 70, 141, 10, 62, 84, 150, 120, 228, 85, 156, 140, 144, 68, 189, 239, 222, 203, 119, 197, 132,
      209, 162, 135, 179, 163, 212, 117, 79, 47, 104, 239, 19, 41, 166, 95, 129, 187, 67, 44, 159, 141, 167, 248,
      43, 179, 42, 15, 94, 249, 234, 104, 118, 225, 254, 236, 204, 92, 13, 26, 204, 141, 195, 227, 2, 104, 251,
      22, 146, 225, 179, 250, 49, 224, 65, 253, 5, 64, 245, 122, 155, 253, 156, 100, 114, 239, 111, 97, 199, 251,
      133, 253, 25, 27, 143, 251, 212, 169, 66, 111, 131, 58, 23, 117, 31, 123, 61, 148, 166, 127, 179, 37, 163,
      3, 82, 94, 113, 41, 80, 56, 207, 67, 173, 41, 28, 249, 157, 10, 42, 98, 136, 201, 213, 127, 102, 34,
      44, 228, 230, 63, 23, 158, 15, 75, 69, 241, 72, 131, 6, 199, 253, 165, 175, 217, 39, 101, 29, 84, 213,
      51, 132, 21, 195, 127, 246, 254, 191, 165, 96, 71, 190, 91, 124, 252, 76, 97, 188, 153, 47, 210, 193, 105,
      71, 206, 209, 146, 102, 254, 251, 43, 152, 117, 188, 195, 196, 88, 246, 223, 102, 100, 159, 7, 252, 43, 240,
      167, 106, 228, 175, 127, 237, 159, 249, 211, 203, 193, 191, 135, 109, 12, 190, 241, 67, 93, 130, 72, 251, 84,
      77, 243, 169, 112, 243, 219, 129, 23, 91, 251, 43, 188, 141, 32, 170, 91, 173, 231, 60, 198, 42, 159, 117,
      218, 197, 63, 226, 96, 127, 201, 179, 206, 186, 144, 219, 84, 145, 216, 51, 47, 2, 113, 241, 68, 149, 98,
      200, 203, 232, 43, 239, 21, 129, 255, 213, 175, 119, 246, 229, 22, 159, 49, 54, 246, 93, 2, 130, 116, 49,
      212, 38, 51, 244, 99, 63, 24, 220, 4, 60, 33, 250, 190, 121, 197, 254, 50, 120, 181, 65, 124, 0, 30,
      249, 209, 209, 124, 136, 241, 134, 4, 116, 174, 87, 11, 113, 177, 239, 82, 66, 125, 215, 84, 105, 153, 227,
      166, 132, 214, 76, 209, 163, 69, 58, 111, 167, 253, 217, 27, 20, 189, 196, 49, 180, 181, 0, 78, 99, 103,
      175, 67, 39, 202, 161, 48, 125, 182, 236, 196, 98, 69, 93, 246, 174, 179, 195, 35, 233, 196, 105, 95, 127,
      25, 7, 153, 69, 42, 94, 19, 113, 40, 171, 113, 33, 191, 176, 215, 71, 230, 134, 187, 213, 233, 107, 225,
      254, 33, 249, 248, 165, 74, 227, 56, 153, 191, 195, 249, 57, 223, 184, 218, 58, 246, 83, 149, 108, 135, 21,
      187, 33, 12, 179, 72, 152, 59, 2, 92, 18, 244, 185, 76, 243, 166, 247, 159, 185, 169, 116, 174, 67, 98,
      241, 248, 137, 239, 17, 0, 224, 0, 191, 230, 117, 248, 58, 170, 27, 251, 202, 176, 148, 57, 2, 220, 130,
      86, 242, 217, 193, 120, 89, 173, 143, 55, 5, 94, 41, 63, 75, 33, 221, 126, 16, 95, 166, 209, 253, 151,
      131, 17, 245, 250, 24, 129, 69, 161, 105, 67, 100, 222, 34, 67, 226, 68, 20, 151, 12, 117, 213, 29, 193,
      241, 90, 238, 15, 162, 60, 146, 66, 155, 52, 196, 197, 65, 138, 25, 44, 216, 56, 241, 191, 200, 76, 86,
      47, 214, 250, 46, 177, 66, 186, 21, 98, 46, 176, 216, 221, 140, 172, 42, 124, 105, 253, 239, 125, 13, 29,
      147, 84, 165, 142, 87, 5, 35, 116, 212, 17, 54, 193, 35, 92, 16, 88, 172, 184, 172, 39, 245, 149, 60,
      115, 76, 95, 185, 254, 16, 177, 116, 160, 191, 43, 191, 84, 204, 56, 77, 109, 91, 224, 120, 244, 95, 165,
      156, 109, 255, 226, 223, 142, 170, 220, 55, 99, 112, 27, 108, 223, 185, 107, 155, 104, 191, 78, 70, 81, 19,
      40, 242, 10, 56, 244, 79, 245, 204, 23, 33, 161, 108, 63, 239, 46, 174, 214, 105, 235, 180, 148, 88, 84,
      30, 27, 164, 84, 58, 30, 73, 220, 135, 108, 3, 117, 204, 51, 24, 236, 192, 115, 254, 74, 123, 58, 28,
      200, 228, 192, 61, 229, 214, 34, 51, 251, 89, 95, 240, 67, 162, 83, 90, 159, 79, 26, 115, 109, 129, 142,
      218, 249, 19, 164, 245, 60, 244, 193, 2, 136, 211, 95, 73, 194, 17, 44, 204, 251, 223, 92, 146, 213, 187,
      84, 49, 120, 216, 49, 63, 102, 160, 85, 185, 148, 233, 207, 135, 254, 245, 236, 4, 21, 61, 177, 109, 24,
      151, 195, 36, 109, 159, 20, 86, 153, 79, 159, 237, 176, 176, 223, 127, 134, 143, 150, 65, 243, 222, 249, 52,
      254, 244, 23, 19, 149, 180, 211, 181, 147, 100, 106, 175, 230, 90, 147, 239, 232, 253, 178, 56, 147, 97, 210,
      177, 108, 185, 98, 207, 98, 102, 18, 90, 161, 240, 94, 85, 165, 127, 226, 74, 46, 249, 21, 19, 165, 20,
      51, 43, 10, 209, 210, 23, 111, 215, 31, 98, 239, 191, 175, 86, 217, 76, 113, 61, 26, 7, 251, 102, 7,
      76, 75, 107, 72, 233, 115, 123, 178, 36, 23, 29, 187, 131, 216, 36, 9, 251, 79, 47, 93, 105, 210, 236,
      102, 228, 110, 156, 202, 228, 73, 95, 119, 118, 126, 247, 248, 118, 250, 227, 29, 113, 89, 66, 115, 154, 217,
      12, 104, 183, 138, 158, 252, 87, 156, 43, 159, 59, 19, 205, 128, 91, 41, 113, 100, 167, 253, 95, 44, 235,
      60, 51, 223, 27, 57, 40, 196, 182, 60, 16, 167, 3, 136, 249, 80, 187, 250, 94, 39, 231, 15, 181, 181,
      181, 97, 214, 211, 145, 119, 234, 78, 24, 75, 190, 30, 218, 221, 215, 211, 93, 54, 173, 14, 28, 159, 69,
      73, 108, 46, 255, 186, 178, 20, 246, 189, 23, 45, 65, 132, 145, 98, 54, 220, 199, 136, 181, 64, 205, 240,
      72, 105, 157, 91, 89, 120, 93, 231, 20, 39, 216, 92, 109, 95, 177, 134, 57, 15, 126, 12, 218, 17, 58,
      210, 252, 71, 28, 28, 128, 147, 70, 217, 27, 69, 76, 185, 250, 25, 18, 46, 109, 127, 161, 211, 68, 202,
      251, 5, 153, 176, 102, 129, 20, 183, 249, 50, 61, 241, 252, 139, 220, 52, 235, 254, 188, 186, 230, 227, 214,
      225, 116, 64, 241, 251, 102, 213, 191, 115, 183, 46, 59, 79, 168, 93, 49, 53, 207, 242, 249, 40, 214, 197,
      193, 49, 210, 201, 63, 182, 136, 247, 29, 72, 113, 255, 164, 204, 109, 141, 120, 3, 73, 129, 230, 214, 107,
      190, 246, 94, 13, 157, 202, 235, 83, 239, 106, 101, 143, 82, 207, 107, 28, 66, 153, 191, 236, 11, 133, 242,
      125, 163, 39, 49, 69, 49, 222, 129, 241, 215, 170, 85, 177, 93, 103, 215, 221, 88, 31, 191, 61, 221, 105,
      117, 55, 26, 41, 27, 197, 220, 160, 188, 36, 162, 144, 43, 239, 28, 121, 250, 43, 148, 78, 244, 136, 110,
      26, 73, 81, 243, 183, 25, 19, 101, 94, 8, 211, 45, 225, 81, 247, 255, 236, 111, 19, 230, 60, 132, 4,
      182, 154, 154, 96, 185, 131, 51, 191, 58, 78, 135, 170, 201, 210, 81, 135, 94, 250, 49, 47, 180, 5, 37,
      145, 237, 223, 111, 40, 24, 255, 39, 101, 76, 94, 125, 106, 113, 33, 20, 254, 69, 180, 67, 137, 223, 61,
      128, 78, 34, 48, 75, 108, 31, 207, 180, 254, 205, 40, 182, 205, 233, 35, 149, 76, 37, 179, 134, 43, 184,
      219, 25, 203, 175, 188, 83, 202, 58, 90, 254, 39, 139, 203, 118, 159, 143, 140, 77, 159, 107, 104, 168, 203,
      44, 18, 191, 164, 39, 62, 150, 71, 52, 173, 13, 212, 244, 190, 57, 68, 119, 235, 226, 12, 87, 102, 41,
      156, 68, 47, 36, 39, 188, 231, 97, 65, 72, 129, 223, 19, 145, 227, 191, 14, 254, 213, 118, 119, 101, 214,
      182, 121, 2, 170, 21, 9, 14, 28, 255, 9, 205, 182, 69, 58, 172, 245, 120, 237, 11, 53, 49, 90, 79,
      235, 241, 95, 166, 90, 215, 119, 77, 111, 145, 13, 157, 33, 57, 39, 252, 127, 2, 153, 151, 12, 110, 81,
      136, 107, 239, 74, 38, 156, 179, 53, 89, 106, 169, 158, 72, 190, 38, 125, 187, 253, 113, 212, 139, 191, 147,
      155, 65, 159, 249, 251, 63, 67, 201, 173, 24, 76, 207, 101, 109, 31, 163, 136, 252, 202, 37, 52, 126, 120,
      81, 225};
  const std::vector<uint8_t> fc3_experts_weights = {
      123, 186, 42, 165, 140, 44, 223, 124, 136, 165, 213, 231, 84, 236, 233, 49, 19, 130, 60, 166, 63, 110, 31,
      193, 171, 238, 175, 234, 180, 113, 216, 26, 159, 185, 138, 72, 239, 132, 203, 94, 120, 242, 252, 33, 253, 154,
      247, 128, 96, 218, 30, 47, 218, 24, 82, 105, 64, 231, 188, 35, 255, 89, 198, 116, 84, 157, 44, 160, 175,
      53, 93, 90, 166, 198, 49, 152, 113, 238, 86, 121, 159, 234, 205, 166, 70, 130, 244, 136, 58, 124, 65, 174,
      228, 142, 107, 216, 135, 94, 196, 77, 42, 195, 99, 2, 61, 31, 31, 170, 41, 108, 54, 169, 204, 106, 67,
      204, 147, 111, 35, 27, 113, 55, 167, 76, 95, 191, 248, 208, 231, 172, 193, 31, 121, 140, 250, 187, 117, 93,
      27, 82, 229, 56, 251, 164, 33, 238, 22, 124, 152, 80, 154, 78, 241, 217, 250, 54, 229, 38, 183, 63, 74,
      161, 62, 69, 90, 232, 59, 49, 162, 140, 130, 210, 131, 253, 179, 159, 87, 120, 231, 123, 96, 166, 105, 95,
      18, 220, 249, 7, 150, 79, 29, 206, 112, 20, 181, 3, 52, 171, 124, 187, 88, 136, 98, 25, 94, 47, 212,
      53, 29, 239, 154, 164, 212, 61, 208, 171, 253, 243, 216, 71, 207, 231, 49, 174, 29, 244, 43, 168, 77, 203,
      129, 233, 123, 131, 33, 254, 151, 138, 110, 245, 89, 173, 220, 53, 60, 80, 147, 195, 255, 42, 8, 18, 139,
      97, 146, 248, 54, 220, 163, 22, 242, 58, 198, 57, 126, 122, 203, 118, 149, 13, 145, 193, 154, 237, 207, 99,
      82, 122, 2, 207, 26, 88, 62, 71, 173, 148, 251, 231, 41, 35, 168, 110, 226, 13, 109, 159, 216, 39, 100,
      43, 65, 167, 20, 255, 15, 37, 110, 220, 54, 175, 40, 88, 49, 5, 189, 146, 218, 110, 207, 118, 213, 76,
      191, 145, 12, 214, 238, 32, 123, 62, 58, 106, 223, 63, 24, 122, 127, 192, 189, 215, 129, 2, 212, 242, 207,
      161, 183, 246, 181, 194, 106, 153, 134, 105, 255, 242, 131, 252, 110, 153, 79, 240, 153, 191, 197, 205, 161, 136,
      55, 243, 141, 212, 116, 169, 162, 14, 45, 222, 22, 54, 142, 136, 247, 194, 72, 198, 20, 219, 125, 238, 45,
      255, 217, 134, 19, 234, 36, 57, 49, 108, 38, 54, 166, 101, 200, 31, 64, 90, 147, 144, 71, 41, 138, 12,
      78, 52, 66, 241, 216, 30, 187, 179, 205, 162, 63, 60, 236, 223, 20, 247, 199, 235, 194, 94, 223, 239, 127,
      169, 251, 238, 199, 172, 206, 164, 227, 220, 243, 105, 125, 113, 116, 245, 247, 194, 249, 28, 174, 151, 28, 114,
      47, 205, 148, 241, 95, 55, 55, 197, 188, 152, 96, 85, 190, 114, 108, 144, 226, 237, 86, 25, 130, 200, 17,
      199, 242, 100, 248, 154, 140, 44, 154, 209, 155, 27, 249, 26, 226, 214, 211, 172, 184, 122, 61, 17, 247, 255,
      14, 111, 39, 28, 63, 224, 27, 132, 42, 79, 45, 171, 155, 35, 230, 163, 215, 225, 175, 198, 184, 110, 21,
      247, 251, 83, 77, 223, 230, 135, 216, 199, 119, 131, 71, 121, 129, 185, 244, 155, 125, 205, 17, 249, 114, 133,
      79, 117, 53, 115, 65, 178, 226, 113, 69, 96, 246, 8, 16, 223, 210, 108, 68, 61, 89, 170, 3, 133, 11,
      197, 243, 89, 15, 201, 85, 125, 219, 193, 47, 106, 248, 52, 145, 191, 176, 207, 159, 219, 43, 242, 52, 250,
      250, 31, 248, 131, 175, 249, 43, 250, 239, 101, 45, 114, 62, 95, 167, 237, 190, 169, 109, 76, 119, 193, 229,
      157, 148, 90, 62, 23, 207, 40, 136, 131, 133, 119, 84, 101, 20, 217, 28, 144, 237, 169, 75, 100, 104, 110,
      113, 1, 101, 153, 253, 243, 13, 43, 171, 90, 255, 170, 204, 118, 251, 129, 253, 239, 233, 122, 134, 239, 225,
      167, 176, 94, 50, 76, 164, 234, 125, 109, 238, 114, 164, 164, 49, 163, 125, 193, 38, 32, 193, 242, 119, 38,
      116, 84, 189, 242, 161, 159, 179, 191, 253, 14, 43, 179, 37, 215, 181, 230, 94, 44, 140, 61, 82, 182, 210,
      221, 248, 104, 224, 249, 18, 126, 36, 40, 119, 72, 49, 171, 203, 86, 208, 15, 15, 201, 18, 164, 157, 171,
      155, 225, 176, 137, 228, 246, 228, 160, 159, 116, 245, 94, 195, 219, 197, 207, 254, 188, 135, 104, 64, 223, 233,
      212, 137, 31, 115, 115, 202, 40, 247, 111, 39, 142, 213, 171, 229, 10, 92, 234, 56, 50, 192, 69, 9, 253,
      208, 111, 36, 110, 137, 128, 94, 136, 191, 243, 41, 197, 61, 102, 46, 220, 39, 45, 155, 112, 110, 47, 84,
      21, 43, 68, 172, 154, 1, 76, 26, 110, 94, 181, 58, 63, 173, 127, 12, 19, 124, 6, 212, 182, 76, 21,
      209, 235, 123, 48, 151, 172, 189, 241, 240, 147, 144, 147, 216, 236, 53, 56, 252, 241, 123, 63, 25, 202, 131,
      40, 163, 119, 254, 98, 86, 245, 86, 229, 62, 42, 175, 76, 226, 8, 246, 251, 17, 76, 204, 59, 132, 73,
      196, 194, 88, 51, 117, 19, 146, 154, 90, 79, 72, 207, 177, 182, 99, 185, 113, 179, 91, 130, 156, 47, 209,
      187, 191, 130, 62, 22, 66, 189, 174, 137, 176, 174, 253, 161, 142, 35, 67, 243, 1, 96, 158, 67, 36, 255,
      223, 214, 181, 200, 234, 230, 127, 52, 155, 241, 6, 100, 56, 227, 28, 172, 149, 47, 63, 107, 40, 35, 173,
      174, 76, 173, 64, 78, 101, 103, 104, 150, 95, 240, 148, 226, 120, 56, 72, 118, 183, 142, 40, 246, 75, 34,
      37, 95, 230, 179, 96, 99, 245, 140, 64, 87, 103, 161, 158, 130, 87, 164, 11, 249, 254, 65, 14, 82, 60,
      179, 94, 242, 43, 238, 167, 108, 171, 21, 128, 238, 98, 136, 133, 196, 209, 111, 34, 202, 98, 90, 104, 9,
      222, 99, 70, 246, 47, 77, 219, 35, 28, 125, 79, 231, 54, 31, 207, 119, 159, 95, 239, 54, 3, 39, 18,
      5, 96, 101, 166, 21, 15, 138, 198, 44, 154, 148, 50, 219, 73, 218, 240, 78, 76, 176, 252, 37, 221, 230,
      17, 67, 113, 11, 155, 201, 7, 53, 250, 188, 223, 102, 79, 217, 24, 170, 148, 25, 82, 11, 45, 75, 211,
      72, 10, 166, 78, 79, 17, 152, 143, 58, 118, 135, 90, 156, 24, 157, 253, 205, 60, 241, 8, 251, 77, 172,
      173, 148, 172, 66, 19, 59, 126, 50, 87, 172, 251, 25, 2, 212, 111, 215, 239, 28, 63, 79, 179, 167, 20,
      18, 222, 66, 167, 253, 226, 115, 132, 81, 156, 61, 225, 188, 178, 225, 185, 9, 27, 242, 0, 196, 191, 122,
      133, 8, 101, 21, 250, 28, 133, 39, 69, 126, 149, 43, 239, 152, 127, 155, 2, 231, 72, 156, 169, 20, 171,
      210, 230, 14, 116, 110, 128, 245, 224, 89, 153, 49, 63, 90, 52, 75, 106, 77, 188, 26, 186, 120, 146, 223,
      223, 158, 252, 93, 231, 155, 50, 99, 131, 42, 9, 88, 131, 247, 41, 245, 255, 77, 26, 151, 121, 79, 47,
      250, 41, 29, 120, 133, 198, 177, 202, 100, 80, 185, 112, 134, 179, 39, 164, 190, 68, 72, 48, 104, 253, 117,
      223, 190, 244, 134, 232, 153, 147, 56, 242, 197, 75, 77, 69, 148, 161, 155, 101, 173, 224, 66, 154, 129, 126,
      4, 188, 79, 90, 119, 146, 255, 140, 202, 133, 153, 190, 5, 251, 228, 174, 183, 164, 171, 251, 67, 209, 255,
      20, 67, 86, 226, 209, 178, 7, 87, 222, 69, 203, 57, 225, 24, 172, 250, 127, 241, 114, 215, 38, 28, 209,
      130, 89, 108, 28, 28, 138, 196, 195, 127, 56, 164, 178, 206, 236, 146, 29, 190, 129, 213, 193, 222, 84, 41,
      148, 249, 112, 172, 201, 0, 119, 252, 182, 238, 23, 210, 63, 94, 217, 146, 222, 238, 33, 51, 64, 80, 138,
      218, 136, 244, 105, 22, 126, 205, 221, 143, 237, 111, 152, 218, 223, 126, 229, 178, 201, 202, 84, 244, 33, 234,
      89, 196, 147, 77, 51, 156, 28, 50, 10, 154, 6, 245, 214, 69, 131, 141, 100, 86, 179, 157, 124, 104, 179,
      48, 174, 183, 189, 21, 98, 55, 87, 117, 119, 200, 16, 233, 204, 151, 169, 119, 236, 151, 8, 202, 202, 255,
      99, 41, 33, 207, 124, 239, 212, 147, 235, 18, 129, 37, 125, 151, 58, 79, 26, 75, 13, 169, 205, 167, 161,
      235, 76, 41, 29, 203, 4, 26, 25, 232, 17, 41, 68, 111, 147, 17, 31, 178, 111, 220, 148, 42, 136, 79,
      141, 186, 138, 191, 35, 61, 248, 121, 130, 218, 139, 234, 154, 56, 10, 188, 194, 33, 138, 246, 59, 13, 179,
      27, 78, 23, 174, 61, 78, 46, 74, 132, 100, 127, 78, 207, 97, 167, 82, 249, 208, 84, 65, 87, 74, 9,
      210, 195, 244, 249, 208, 223, 89, 246, 229, 62, 35, 74, 253, 82, 115, 50, 76, 111, 139, 21, 249, 165, 205,
      48, 54, 31, 159, 53, 74, 37, 173, 232, 62, 184, 63, 109, 49, 221, 193, 196, 139, 214, 212, 139, 245, 23,
      211, 243, 17, 135, 235, 31, 179, 184, 152, 210, 202, 245, 85, 31, 228, 193, 202, 234, 14, 133, 94, 118, 255,
      68, 210, 46, 204, 199, 106, 16, 25, 19, 126, 112, 104, 220, 241, 218, 67, 216, 219, 40, 192, 159, 205, 49,
      108, 58, 60, 25, 63, 184, 247, 213, 211, 54, 198, 46, 103, 186, 155, 224, 159, 246, 131, 190, 215, 221, 194,
      234, 26, 201, 71, 170, 40, 185, 236, 169, 210, 66, 249, 78, 245, 67, 35, 221, 252, 180, 25, 43, 200, 53,
      250, 108, 207, 118, 135, 99, 116, 213, 153, 255, 22, 94, 248, 58, 204, 4, 2, 190, 208, 191, 130, 87, 156,
      2, 174, 15, 248, 164, 159, 41, 39, 29, 47, 42, 102, 248, 116, 59, 77, 228, 157, 61, 121, 4, 163, 165,
      33, 156, 242, 247, 45, 31, 51, 170, 23, 183, 252, 245, 124, 4, 87, 103, 144, 118, 182, 237, 159, 140, 242,
      190, 131, 126, 16, 179, 105, 31, 72, 254, 243, 25, 207, 45, 194, 234, 241, 55, 18, 69, 118, 30, 1, 252,
      40, 164, 231, 225, 6, 23, 104, 157, 51, 249, 247, 4, 208, 149, 17, 58, 180, 248, 215, 140, 236, 178, 21,
      133, 110, 155, 79, 245, 69, 35, 255, 189, 245, 87, 216, 123, 3, 155, 202, 253, 32, 237, 154, 120, 20, 232,
      47, 178, 109, 200, 177, 43, 8, 97, 82, 115, 166, 106, 161, 120, 28, 44, 227, 84, 165, 86, 229, 168, 9,
      234, 233, 80, 215, 118, 220, 176, 138, 218, 127, 251, 169, 236, 121, 215, 98, 72, 27, 221, 203, 67, 59, 194,
      79, 167, 118, 50, 98, 141, 162, 224, 181, 124, 57, 57, 191, 230, 201, 213, 15, 84, 6, 28, 112, 228, 53,
      196, 61, 143, 154, 249, 110, 47, 236, 33, 191, 95, 102, 22, 189, 73, 108, 112, 122, 23, 147, 216, 229, 147,
      255, 63, 41, 93, 129, 131, 251, 88, 168, 75, 39, 252, 249, 227, 52, 175, 93, 254, 96, 196, 121, 155, 36,
      95, 252, 88, 9, 74, 50, 254, 40, 64, 75, 121, 88, 185, 98, 15, 51, 87, 163, 253, 122, 132, 35, 196,
      194, 250, 163, 100, 70, 55, 39, 184, 4, 171, 216, 95, 204, 243, 111, 47, 254, 95, 47, 90, 182, 101, 46,
      140, 56, 97, 201, 83, 225, 128, 147, 66, 29, 222, 54, 133, 44, 97, 249, 177, 222, 158, 76, 59, 164, 195,
      230, 151, 58, 68, 117, 55, 8, 94, 107, 233, 48, 49, 214, 230, 114, 239, 48, 97, 92, 5, 101, 95, 58,
      245, 213, 148, 33, 38, 49, 232, 92, 24, 110, 188, 149, 243, 8, 152, 252, 202, 254, 203, 220, 72, 198, 176,
      218, 156, 63, 227, 106, 145, 178, 255, 86, 33, 191, 117, 175, 161, 249, 78, 146, 56, 136, 217, 158, 87, 98,
      212, 131, 81, 58, 6, 70, 59, 239, 155, 247, 169, 63, 92, 5, 228, 162, 69, 40, 221, 95, 111, 13, 182,
      79, 180, 10, 165, 242, 161, 149, 88, 246, 201, 96, 107, 89, 250, 220, 212, 253, 10, 108, 31, 167, 130, 126,
      119, 96, 225, 70, 149, 46, 151, 131, 84, 246, 188, 184, 146, 36, 160, 72, 194, 214, 161, 223, 235, 222, 233,
      243, 70, 158, 131, 103, 22, 120, 58, 89, 190, 17, 187, 92, 104, 191, 103, 187, 218, 244, 111, 246, 178, 73,
      95, 188, 254, 52, 116, 31, 195, 66, 148, 54, 231, 109, 151, 251, 35, 100, 146, 49, 96, 194, 213, 250, 143,
      95, 111, 193, 114, 212, 250, 225, 46, 249, 179, 211, 75, 149, 221, 133, 74, 138, 230, 61, 87, 215, 106, 199,
      246, 239, 31, 63, 81, 172, 247, 206, 87, 118, 1, 38, 125, 196, 78, 138, 99, 58, 81, 157, 82, 252, 59,
      118, 117, 83, 172, 39, 222, 163, 181, 121, 204, 142, 99, 101, 248, 55, 17, 182, 75, 71, 170, 77, 70, 154,
      242, 159, 178, 243, 201, 235, 165, 129, 127, 149, 158, 20, 52, 193, 8, 231, 210, 53, 47, 47, 220, 127, 101,
      243, 220, 219, 188, 221, 166, 84, 173, 140, 111, 106, 42, 88, 15, 200, 59, 248, 214, 246, 202, 242, 77, 238,
      210, 113, 10, 217, 241, 191, 201, 132, 122, 248, 173, 236, 214, 18, 201, 205, 198, 218, 36, 127, 95, 56, 251,
      233, 169, 218, 173, 243, 251, 84, 38, 133, 178, 108, 150, 181, 244, 78, 143, 34, 167, 87, 12, 255, 138, 242,
      194, 174, 198, 243, 100, 250, 227, 136, 35, 222, 53, 144, 249, 60, 105, 28, 111, 0, 53, 46, 239, 81, 21,
      137, 143, 84, 4, 91, 52, 233, 158, 55, 181, 38, 125, 25, 20, 198, 103, 17, 190, 248, 20, 186, 22, 86,
      165, 42, 175, 85, 85, 75, 171, 41, 51, 14, 207, 23, 250, 196, 249, 230, 50, 118, 29, 255, 191, 3, 55,
      58, 237, 106, 172, 15, 75, 117, 89, 122, 108, 248, 227, 79, 106, 245, 61, 211, 190, 49, 165, 202, 29, 82,
      94, 6, 141, 157, 173, 71, 69, 64, 69, 57, 181, 106, 234, 9, 69, 15, 203, 135, 60, 143, 197, 117, 117,
      175, 202, 251, 115, 238, 130, 72, 111, 93, 161, 138, 251, 23, 78, 205, 1, 211, 30, 22, 239, 25, 162, 140,
      10, 245, 22, 172, 181, 244, 50, 3, 85, 101, 120, 101, 68, 67, 70, 180, 195, 73, 121, 91, 138, 136, 241,
      248, 246, 98, 252, 169, 57, 159, 255, 223, 50, 78, 76, 205, 164, 136, 180, 44, 135, 175, 81, 215, 120, 154,
      38, 200, 129, 173, 201, 8, 129, 153, 247, 223, 172, 143, 80, 207, 197, 47, 84, 140, 253, 248, 103, 95, 156,
      230, 161, 60, 71, 95, 207, 101, 134, 247, 244, 196, 212, 140, 30, 143, 60, 115, 145, 181, 89, 105, 61, 146,
      175, 136, 244, 84, 223, 22, 180, 140, 25, 143, 76, 36, 58, 127, 46, 106, 169, 94, 45, 240, 44, 26, 83,
      48, 253, 35, 237, 214, 251, 249, 233, 172, 102, 175, 93, 182, 170, 227, 221, 49, 23, 202, 231, 44, 247, 4,
      188, 212, 182, 132, 73, 159, 179, 216, 180, 127, 45, 145, 104, 99, 63, 228, 185, 239, 176, 24, 69, 78, 174,
      213, 230, 142, 172, 217, 197, 219, 103, 133, 76, 50, 192, 28, 30, 218, 40, 15, 141, 191, 173, 10, 93, 196,
      231, 205, 17, 60, 117, 219, 13, 255, 194, 190, 159, 77, 24, 116, 41, 206, 245, 224, 91, 255, 162, 171, 89,
      106, 168, 18, 18, 140, 89, 190, 38, 120, 107, 200, 236, 225, 170, 35, 81, 50, 243, 111, 124, 255, 83, 95,
      175, 242, 219, 166, 102, 254, 175, 176, 194, 142, 110, 15, 63, 59, 37, 241, 172, 113, 29, 120, 63, 68, 128,
      186, 187, 186, 197, 177, 214, 80, 204, 68, 250, 169, 188, 162, 47, 244, 27, 229, 247, 108, 96, 124, 100, 70,
      103, 205, 126, 239, 229, 155, 255, 221, 54, 15, 168, 96, 137, 149, 15, 209, 250, 54, 248, 246, 198, 105, 118,
      244, 46, 90, 111, 159, 15, 153, 211, 206, 199, 251, 96, 240, 55, 75, 191, 228, 152, 228, 178, 162, 143, 15,
      247, 81, 144, 18, 178, 145, 107, 182, 216, 90, 255, 149, 126, 246, 178, 212, 1, 71, 105, 114, 123, 223, 69,
      171, 91, 140, 137, 133, 143, 44, 107, 123, 80, 132, 240, 75, 175, 43, 102, 89, 98, 174, 187, 153, 250, 191,
      99, 182, 248, 69, 196, 142, 123, 25, 248, 37, 221, 127, 68, 110, 101, 134, 196, 64, 35, 223, 173, 217, 79,
      234, 96, 255, 134, 65, 253, 133, 50, 164, 119, 99, 109, 175, 251, 158, 242, 247, 223, 68, 177, 239, 57, 199,
      236, 54, 140, 254, 113, 101, 195, 41, 60, 38, 187, 210, 252, 83, 178, 85, 143, 135, 12, 148, 90, 115, 73,
      207, 165, 247, 178, 53, 159, 176, 219, 145, 177, 21, 112, 107, 170, 150, 91, 196, 27, 128, 210, 219, 65, 206,
      182, 214, 188, 253, 126, 94, 245, 158, 223, 151, 81, 232, 194, 157, 166, 147, 253, 161, 69, 77, 254, 4, 59,
      210, 70, 31, 118, 108, 68, 253, 200, 212, 52, 159, 68, 118, 190, 191, 157, 218, 110, 103, 247, 207, 183, 23,
      223, 223, 206, 105, 105, 95, 22, 83, 90, 117, 242, 123, 120, 91, 253, 63, 74, 109, 125, 207, 127, 154, 240,
      242, 17, 167, 246, 107, 59, 226, 135, 72, 196, 28, 197, 153, 150, 221, 53, 251, 60, 193, 20, 176, 81, 177,
      178, 116, 175, 106, 24, 95, 255, 5, 253, 95, 187, 51, 220, 252, 94, 45, 16, 188, 108, 132, 231, 136, 10,
      162, 230, 36, 138, 212, 111, 217, 53, 172, 97, 241, 123, 113, 181, 128, 159, 189, 71, 32, 46, 32, 246, 31,
      245, 138, 84, 120, 198, 62, 86, 218, 244, 121, 111, 179, 46, 244, 1, 120, 214, 209, 138, 249, 146, 40, 31,
      193, 19, 239, 37, 150, 12, 111, 82, 197, 55, 234, 31, 121, 213, 148, 90, 114, 253, 33, 242, 235, 138, 143,
      87, 230, 42, 173, 239, 104, 161, 115, 237, 212, 223, 67, 79, 250, 219, 249, 116, 65, 25, 105, 52, 116, 51,
      229, 214, 197, 34, 80, 117, 131, 219, 129, 251, 169, 247, 43, 46, 255, 43, 171, 41, 229, 43, 175, 205, 23,
      219, 169, 79, 166, 58, 81, 147, 240, 195, 138, 32, 138, 71, 59, 188, 232, 228, 253, 252, 182, 39, 183, 62,
      243, 24, 107, 212, 142, 152, 168, 63, 227, 81, 254, 105, 239, 203, 232, 74, 154, 73, 101, 106, 235, 247, 42,
      211, 17, 65, 231, 97, 128, 71, 218, 148, 196, 223, 155, 242, 122, 131, 41, 59, 136, 130, 42, 104, 250, 175,
      238, 42, 149, 245, 143, 252, 65, 243, 60, 170, 132, 221, 126, 251, 135, 102, 120, 113, 241, 249, 220, 37, 248,
      166, 19, 77, 30, 55, 242, 61, 191, 217, 211, 43, 15, 218, 194, 151, 255, 153, 33, 203, 97, 77, 9, 179,
      55, 140, 185, 90, 62, 163, 86, 171, 173, 118, 173, 247, 151, 89, 48, 86, 195, 131, 42, 8, 47, 254, 97,
      38, 182, 80, 53, 111, 93, 147, 111, 240, 124, 102, 94, 127, 141, 247, 86, 11, 37, 11, 73, 218, 224, 62,
      75, 235, 197, 123, 98, 62, 127, 246, 27, 1, 132, 244, 217, 26, 182, 110, 220, 209, 153, 188, 207, 118, 72,
      109, 16, 23, 45, 244, 86, 240, 47, 49, 169, 233, 49, 151, 242, 76, 216, 202, 243, 101, 16, 176, 196, 51,
      225, 204, 68, 217, 185, 59, 42, 36, 255, 117, 191, 219, 145, 31, 220, 37, 17, 237, 35, 7, 194, 57, 168,
      163, 175, 3, 93, 54, 12, 203, 153, 35, 116, 191, 244, 196, 132, 226, 217, 171, 25, 92, 207, 51, 250, 245,
      217, 152, 244, 71, 34, 74, 194, 176, 127, 253, 83, 41, 48, 199, 82, 72, 244, 228, 136, 146, 145, 179, 33,
      103, 202, 149, 92, 159, 26, 233, 154, 133, 183, 194, 36, 2, 206, 31, 209, 131, 230, 109, 0, 109, 92, 49,
      150, 19, 112, 19, 68, 126, 120, 207, 138, 93, 11, 22, 197, 140, 188, 85, 173, 117, 194, 200, 31, 221, 152,
      138, 251, 192, 185, 47, 116, 118, 13, 189, 34, 142, 103, 47, 69, 151, 134, 122, 13, 34, 66, 211, 25, 180,
      151, 227, 104, 201, 175, 96, 248, 114, 140, 90, 145, 203, 90, 171, 129, 187, 219, 162, 245, 119, 107, 84, 171,
      255, 218, 26, 46, 132, 203, 79, 191, 87, 10, 124, 85, 164, 105, 115, 82, 158, 175, 182, 105, 246, 29, 97,
      86, 2, 246, 234, 81, 29, 36, 68, 197, 37, 172, 30, 45, 44, 111, 44, 249, 63, 146, 187, 14, 224, 254,
      52, 218, 75, 54, 185, 165, 92, 214, 70, 247, 253, 79, 232, 112, 230, 42, 58, 53, 15, 98, 135, 139, 76,
      47, 244, 68, 7, 48, 75, 211, 87, 107, 203, 94, 67, 35, 241, 248, 146, 213, 53, 47, 98, 45, 173, 206,
      82, 65, 229, 182, 96, 175, 234, 252, 131, 78, 120, 104, 105, 150, 119, 123, 122, 61, 65, 185, 39, 199, 163,
      127, 193, 222, 149, 184, 233, 75, 98, 170, 150, 196, 226, 235, 242, 26, 192, 17, 84, 218, 119, 62, 58, 59,
      164, 80, 54, 117, 164, 212, 6, 219, 212, 203, 175, 242, 162, 87, 170, 19, 90, 212, 37, 108, 141, 83, 157,
      77, 155, 25, 236, 8, 216, 21, 6, 95, 88, 239, 19, 31, 188, 103, 29, 186, 92, 129, 6, 76, 254, 130,
      242, 209, 181, 53, 29, 251, 164, 165, 111, 47, 89, 122, 196, 103, 252, 112, 127, 133, 76, 110, 172, 82, 119,
      201, 222, 85, 188, 238, 131, 110, 187, 204, 175, 185, 68, 85, 31, 222, 35, 119, 138, 88, 223, 71, 154, 152,
      251, 161, 216, 192, 252, 133, 120, 177, 212, 172, 147, 122, 184, 154, 42, 179, 56, 255, 191, 235, 172, 134, 60,
      98, 187, 243, 199, 253, 250, 84, 72, 228, 121, 168, 20, 168, 99, 120, 93, 242, 177, 110, 195, 143, 217, 54,
      104, 223, 14, 11, 95, 226, 43, 129, 23, 135, 152, 95, 147, 3, 131, 18, 127, 187, 67, 250, 240, 206, 40,
      140, 218, 67, 161, 252, 85, 12, 27, 193, 43, 242, 106, 220, 158, 164, 222, 236, 198, 240, 202, 201, 108, 38,
      52, 96, 238, 24, 175, 118, 65, 194, 116, 236, 248, 29, 1, 106, 176, 24, 229, 192, 172, 95, 83, 180, 253,
      254, 245, 105, 88, 154, 53, 183, 41, 57, 245, 127, 98, 27, 250, 251, 207, 48, 148, 251, 89, 12, 227, 214,
      141, 37, 133, 160, 251, 112, 230, 191, 106, 244, 74, 98, 178, 221, 118, 102, 127, 7, 27, 179, 110, 61, 252,
      133, 107, 97, 23, 98, 244, 211, 98, 89, 191, 44, 170, 197, 43, 240, 29, 199, 248, 210, 239, 148, 106, 209,
      195, 252, 178, 79, 140, 234, 75, 108, 78, 194, 175, 251, 246, 146, 26, 242, 212, 60, 235, 225, 76, 140, 68,
      185, 219, 190, 137, 159, 32, 237, 188, 101, 65, 177, 28, 238, 152, 161, 137, 117, 245, 3, 149, 126, 114, 199,
      39, 49, 255, 13, 15, 53, 186, 74, 245, 25, 245, 197, 251, 129, 47, 25, 153, 198, 133, 226, 167, 88, 94,
      245, 245, 74, 129, 255, 233, 121, 145, 219, 243, 157, 239, 152, 121, 161, 190, 223, 197, 240, 230, 55, 25, 246,
      156, 255, 197, 160, 239, 136, 214, 13, 203, 163, 208, 79, 246, 181, 213, 167, 56, 104, 245, 33, 48, 191, 251,
      33, 127, 100, 71, 66, 54, 104, 224, 85, 34, 255, 52, 247, 83, 68, 227, 120, 232, 117, 105, 66, 237, 217,
      169, 175, 191, 17, 72, 214, 5, 99, 191, 227, 121, 171, 67, 226, 190, 150, 152, 81, 255, 3, 156, 119, 228,
      98, 215};
  const std::vector<float> fc1_scales = {
      0.01553376f, 0.015543817f, 0.015551699f, 0.015492203f, 0.015023133f, 0.0154082235f, 0.0155232195f,
      0.01528402f, 0.015559638f, 0.015533516f, 0.015493423f, 0.015256615f, 0.0152339935f, 0.015549371f,
      0.015381575f, 0.015576782f, 0.015412793f, 0.015498972f, 0.0151363555f, 0.015505189f, 0.014904913f,
      0.015218727f, 0.015376769f, 0.015279377f, 0.015432924f, 0.015483502f, 0.015457189f, 0.015407557f,
      0.0156120695f, 0.014825948f, 0.015501786f, 0.015303297f, 0.015532501f, 0.0152144935f, 0.015333908f,
      0.01479763f, 0.015206473f, 0.01543629f, 0.015437368f, 0.01513233f, 0.015589874f, 0.015567031f,
      0.015393224f, 0.014935784f, 0.015579218f, 0.015432265f, 0.015484579f, 0.015261326f, 0.015371274f,
      0.015189547f, 0.015558099f, 0.014714118f, 0.015086958f, 0.015577158f, 0.014815275f, 0.01525769f,
      0.015569633f, 0.014951542f, 0.015491992f, 0.015379513f, 0.015588352f, 0.015455488f, 0.015094815f,
      0.015585413f, 0.0151954815f, 0.015539678f, 0.015179157f, 0.015570812f, 0.015453467f, 0.015222808f,
      0.015130177f, 0.015514964f, 0.015050512f, 0.013596393f, 0.015181009f, 0.014813691f, 0.015430912f,
      0.015623035f, 0.015465939f, 0.0155621655f, 0.015619047f, 0.015616288f, 0.015411615f, 0.015294425f,
      0.015334727f, 0.01536013f, 0.015485667f, 0.015279645f, 0.015232291f, 0.015200818f, 0.014945071f,
      0.015612004f, 0.015533011f, 0.01562017f, 0.015604494f, 0.015526485f, 0.014934285f, 0.015624931f,
      0.015617797f, 0.0155350845f, 0.015362147f, 0.015408119f, 0.01547795f, 0.014903402f, 0.0154722165f,
      0.015608951f, 0.015536772f, 0.015497636f, 0.01543246f, 0.015433108f, 0.015222307f, 0.0156019665f,
      0.0154854f, 0.014986996f, 0.015555747f, 0.015378246f, 0.015050007f, 0.015395556f, 0.0154241435f,
      0.015317103f, 0.015418313f, 0.015221456f, 0.015339879f, 0.015616156f, 0.01556934f, 0.015396217f,
      0.015617745f, 0.015584825f};
  const std::vector<float> fc2_scales = {
      0.015234984f, 0.015523607f, 0.015164727f, 0.01548125f, 0.015093872f, 0.015315635f, 0.015266418f,
      0.015527874f, 0.015592782f, 0.015093137f, 0.014813861f, 0.015202709f, 0.0153913535f, 0.01537223f,
      0.015511734f, 0.015440272f, 0.015092988f, 0.015597204f, 0.015287647f, 0.015497316f, 0.015502119f,
      0.015546441f, 0.015100006f, 0.015404332f, 0.015531912f, 0.015555983f, 0.01507354f, 0.015588721f,
      0.01545357f, 0.015513655f, 0.015537361f, 0.015617292f, 0.015471501f, 0.015559636f, 0.015541913f,
      0.015565485f, 0.015380409f, 0.015168384f, 0.0155151095f, 0.015469871f, 0.015443675f, 0.015554659f,
      0.015623292f, 0.014806481f, 0.015374577f, 0.015407367f, 0.015303424f, 0.015412778f, 0.015173398f,
      0.015220221f, 0.015319703f, 0.015124975f, 0.015372854f, 0.015297962f, 0.015397722f, 0.015355343f,
      0.015466366f, 0.01507015f, 0.015495513f, 0.015593667f, 0.015281979f, 0.015336113f, 0.015525f,
      0.01537925f, 0.015516909f, 0.015614616f, 0.015543677f, 0.015600901f, 0.0153762605f, 0.015399329f,
      0.015290953f, 0.015491776f, 0.015287561f, 0.015271302f, 0.015343454f, 0.015566604f, 0.015624354f,
      0.01533857f, 0.015119089f, 0.015481008f, 0.015398314f, 0.015596798f, 0.0153150465f, 0.015608612f,
      0.015555618f, 0.015332868f, 0.015389856f, 0.015581448f, 0.015621847f, 0.015410677f, 0.01556886f,
      0.015614897f, 0.01547879f, 0.015478665f, 0.015515525f, 0.01555785f, 0.01561863f, 0.015433328f,
      0.015305866f, 0.015573423f, 0.015373498f, 0.0155666135f, 0.015396729f, 0.015547626f, 0.014429122f,
      0.015496805f, 0.015291028f, 0.015550148f, 0.015425619f, 0.0155315865f, 0.015438886f, 0.015576545f,
      0.015619017f, 0.01515908f, 0.015479961f, 0.015447514f, 0.015065838f, 0.015309097f, 0.015131723f,
      0.014979966f, 0.014841583f, 0.015531611f, 0.015469328f, 0.015101345f, 0.015491165f, 0.0155728385f,
      0.015560919f, 0.015370855f};
  const std::vector<float> fc3_scales = {
      0.015415549f, 0.015507627f, 0.014678219f, 0.015550405f, 0.015007719f, 0.015621224f, 0.0155345425f,
      0.015270567f, 0.015584674f, 0.015545895f, 0.015420519f, 0.015511904f, 0.015497334f, 0.015613152f,
      0.015344387f, 0.015462939f, 0.015408138f, 0.015263364f, 0.015522234f, 0.015557403f, 0.015617529f,
      0.0155323f, 0.015070785f, 0.0154183265f, 0.015569469f, 0.014966013f, 0.015585924f, 0.0155711975f,
      0.01525447f, 0.015368329f, 0.015493156f, 0.015439328f, 0.015451316f, 0.015313955f, 0.015007403f,
      0.015397709f, 0.015486734f, 0.01554385f, 0.015589319f, 0.015365845f, 0.0152554605f, 0.015575631f,
      0.015524423f, 0.015446551f, 0.01492084f, 0.015455352f, 0.014697226f, 0.015101928f, 0.01525531f,
      0.01557962f, 0.015178623f, 0.015425265f, 0.015473807f, 0.015434511f, 0.015518608f, 0.015348455f,
      0.014946166f, 0.0153529495f, 0.015595689f, 0.015601011f, 0.015585726f, 0.0155280195f, 0.014892634f,
      0.015474405f, 0.015582396f, 0.01517096f, 0.015513012f, 0.015467694f, 0.015459979f, 0.015562061f,
      0.015136767f, 0.015591653f, 0.015295904f, 0.014878606f, 0.015608272f, 0.015360581f, 0.015440369f,
      0.015552597f, 0.0153689645f, 0.015544422f, 0.015161956f, 0.015341356f, 0.015590522f, 0.0155716f,
      0.0153000355f, 0.015417134f, 0.015441434f, 0.015425701f, 0.015540993f, 0.015532201f, 0.015549095f,
      0.015335085f, 0.01554049f, 0.015028752f, 0.015245372f, 0.01556482f, 0.015607696f, 0.015421748f,
      0.0154471155f, 0.015398482f, 0.015602099f, 0.015455678f, 0.015591139f, 0.01557602f, 0.015448909f,
      0.0153864585f, 0.015211966f, 0.015580256f, 0.015525388f, 0.015311712f, 0.015527213f, 0.015249299f,
      0.015606547f, 0.0154935885f, 0.015555864f, 0.01537651f, 0.015581995f, 0.015337018f, 0.01547428f,
      0.015216509f, 0.015208464f, 0.015577957f, 0.015380967f, 0.015528679f, 0.015578562f, 0.015344413f,
      0.015526013f, 0.015194058f};
  const std::vector<float> output = {
      0.04828f, -0.05322f, -0.11176f, 0.09344f, -0.02678f, 0.09827f, 0.06616f, -0.04233f, -0.03937f, 0.1582f,
      -0.0437f, 0.04413f, 0.0931f, 0.11127f, -0.0747f, -0.10297f, -0.06226f, 0.02866f, -0.1395f, -0.008934f,
      -0.0385f, 0.1564f, 0.1207f, -0.104f, 0.131f, -0.01776f, -0.00962f, 0.05615f, -0.0129f, -0.01724f,
      -0.06555f, 0.00729f, -0.02585f, 0.01662f, 0.1351f, -0.02095f, 0.1703f, -0.0237f, -0.1381f, 0.10895f,
      -0.0724f, 0.04358f, 0.1371f, -0.0707f, 0.02188f, -0.06122f, -0.03586f, -0.01924f, 0.01304f, -0.039f,
      0.12317f, -0.2336f, 0.0972f, -0.0862f, 0.05716f, 0.05075f, 0.1477f, 0.1316f, -0.05365f, -0.1301f,
      0.01836f, -0.09186f, 0.0641f, -0.10913f, -0.1576f, 0.0441f, 0.03537f, -0.062f, 0.06915f, 0.02954f,
      0.1605f, -0.05975f, -0.08435f, 0.1779f, -0.01181f, 0.001026f, 0.1284f, 0.1531f, 0.0571f, -0.1577f,
      0.05838f, 0.1444f, -0.02432f, 0.10065f, -0.04343f, -0.09296f, 0.0335f, -0.00582f, 0.004944f, -0.013054f,
      -0.049f, 0.0776f, 0.04633f, 0.0746f, -0.1191f, -0.1118f, -0.209f, -0.09753f, -0.02882f, -0.01466f,
      -0.08655f, 0.1167f, -0.02155f, 0.05896f, 0.0117f, -0.05618f, 0.0908f, 0.1324f, -0.04462f, 0.04077f,
      -0.02385f, 0.01863f, 0.0729f, 0.1226f, -0.1261f, -0.0583f, 0.0774f, -0.1523f, 0.2018f, 0.1119f,
      -0.04095f, -0.01188f, 0.1113f, 0.0502f, 0.00584f, -0.02325f, 0.02837f, 0.04144f};

  RunQMoETest(input, router_probs, fc1_experts_weights, fc2_experts_weights, fc3_experts_weights, fc1_scales,
              fc2_scales, fc3_scales, output, num_rows, num_experts, hidden_size, inter_size, "silu",
              1, /*normalize_routing_weights*/
              2, /*top_k*/
              4 /*expert_weight_bits*/);
}

// CPU-specific QMoE tests
TEST(MoETest, QMoETest_CPU_Int2_MLAS) {
#ifdef USE_MLAS
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 16;
  int inter_size = 16;
  constexpr int64_t expert_weight_bits = 2;
  constexpr int64_t pack_size = 8 / expert_weight_bits;
  constexpr uint8_t packed_zero = 0xAA;  // Four 2-bit zero points of value 2 (= 2^(2-1)).

  const std::vector<float> input = {
      -0.5f, 0.2f, 1.1f, -0.3f, 0.8f, -0.1f, 0.4f, -0.7f, 0.9f, -0.2f, 0.6f, 0.1f, -0.4f, 0.3f, -0.8f, 0.7f,
      0.1f, 0.7f, -0.4f, 0.2f, 0.8f, -0.3f, 0.5f, -0.1f, 0.6f, 0.4f, -0.7f, 0.3f, 0.9f, -0.2f, 0.1f, 0.8f};
  const std::vector<float> router_probs = {0.3f, 0.7f, 0.6f, 0.4f};

  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * (hidden_size / pack_size), packed_zero);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * (inter_size / pack_size), packed_zero);
  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.05f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.05f);
  std::vector<float> expected_output(num_rows * hidden_size, 0.0f);

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 2);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");
  cpu_tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<float>();
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  cpu_tester.SetOutputTolerance(0.05f);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_Int2_NonZeroOutput) {
#ifdef USE_MLAS
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 4;
  constexpr int64_t inter_size = 4;
  constexpr int64_t expert_weight_bits = 2;
  constexpr int64_t pack_size = 8 / expert_weight_bits;
  auto pack_2bit = [](uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) -> uint8_t {
    return static_cast<uint8_t>((v0 & 0x03u) |
                                ((v1 & 0x03u) << 2) |
                                ((v2 & 0x03u) << 4) |
                                ((v3 & 0x03u) << 6));
  };

  const std::vector<float> input = {1.0f, -2.0f, 0.5f, 3.0f};
  const std::vector<float> router_probs = {1.0f};

  // Symmetric 2-bit quantization uses storage zero point 2, so stored values [0, 1, 2, 3]
  // dequantize to signed weights [-2, -1, 0, 1] when scale is 1.0.
  //
  // FC1 rows:
  //   [ 1,  0,  0,  0] -> 1
  //   [ 0, -1,  0,  0] -> 2
  //   [ 0,  0,  1,  1] -> 3.5
  //   [-1,  0,  0,  1] -> 2
  const std::vector<uint8_t> fc1_experts_weights = {
      pack_2bit(3, 2, 2, 2),
      pack_2bit(2, 1, 2, 2),
      pack_2bit(2, 2, 3, 3),
      pack_2bit(1, 2, 2, 3),
  };

  // FC2 rows:
  //   [ 1,  1,  0,  0] -> 3
  //   [ 0,  0, -1,  0] -> -3.5
  //   [ 0,  0,  0,  1] -> 2
  //   [ 1,  0,  0, -1] -> -1
  const std::vector<uint8_t> fc2_experts_weights = {
      pack_2bit(3, 3, 2, 2),
      pack_2bit(2, 2, 1, 2),
      pack_2bit(2, 2, 2, 3),
      pack_2bit(3, 2, 2, 1),
  };

  const std::vector<float> fc1_scales(num_experts * inter_size, 1.0f);
  const std::vector<float> fc2_scales(num_experts * hidden_size, 1.0f);
  const std::vector<float> expected_output = {3.0f, -3.5f, 2.0f, -1.0f};

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "identity");
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<float>();
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  cpu_tester.SetOutputTolerance(0.001f);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_Int2_BlockWiseLutIdentity) {
#ifdef USE_MLAS
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 128;
  constexpr int64_t block_size = 32;
  constexpr int64_t expert_weight_bits = 2;
  constexpr int64_t pack_size = 8 / expert_weight_bits;
  constexpr int64_t blocks_per_row = hidden_size / block_size;
  auto pack_2bit = [](uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) -> uint8_t {
    return static_cast<uint8_t>((v0 & 0x03u) |
                                ((v1 & 0x03u) << 2) |
                                ((v2 & 0x03u) << 4) |
                                ((v3 & 0x03u) << 6));
  };

  std::vector<float> input(static_cast<size_t>(hidden_size));
  for (int64_t i = 0; i < hidden_size; ++i) {
    input[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.25f;
  }

  const std::vector<float> router_probs = {1.0f};
  std::vector<uint8_t> fc1_experts_weights(static_cast<size_t>(num_experts * inter_size * (hidden_size / pack_size)), pack_2bit(2, 2, 2, 2));
  std::vector<uint8_t> fc2_experts_weights(static_cast<size_t>(num_experts * hidden_size * (inter_size / pack_size)), pack_2bit(2, 2, 2, 2));
  std::vector<float> fc1_scales(static_cast<size_t>(num_experts * inter_size * blocks_per_row), 1.0f);
  std::vector<float> fc2_scales(static_cast<size_t>(num_experts * hidden_size * blocks_per_row), 1.0f);

  for (int64_t row = 0; row < inter_size; ++row) {
    const int64_t col = row;
    const int64_t packed_col = col / pack_size;
    const int64_t lane = col % pack_size;
    const uint8_t one_storage_value = static_cast<uint8_t>(3u << (lane * expert_weight_bits));
    const uint8_t clear_mask = static_cast<uint8_t>(0x03u << (lane * expert_weight_bits));
    const size_t offset = static_cast<size_t>(row * (hidden_size / pack_size) + packed_col);
    fc1_experts_weights[offset] = static_cast<uint8_t>((fc1_experts_weights[offset] & ~clear_mask) | one_storage_value);
    fc2_experts_weights[offset] = static_cast<uint8_t>((fc2_experts_weights[offset] & ~clear_mask) | one_storage_value);
  }

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "identity");
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);
  cpu_tester.AddAttribute<int64_t>("block_size", block_size);

  const std::vector<int64_t> input_dims = {num_rows, hidden_size};
  const std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  const std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
  const std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  const std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size, blocks_per_row};
  const std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size, blocks_per_row};
  const std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<float>();
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(input));
  cpu_tester.SetOutputTolerance(0.05f);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_Int2_InvalidHiddenSize) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 6;
  constexpr int64_t inter_size = 8;
  constexpr int64_t expert_weight_bits = 2;
  constexpr int64_t pack_size = 8 / expert_weight_bits;

  const std::vector<float> input = {0.5f, -0.25f, 0.75f, -0.5f, 0.25f, 1.0f};
  const std::vector<float> router_probs = {1.0f};
  std::vector<uint8_t> fc1_experts_weights(num_experts * inter_size * (hidden_size / pack_size), 0xAA);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * (inter_size / pack_size), 0xAA);
  std::vector<float> fc1_scales(num_experts * inter_size, 0.05f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.05f);
  std::vector<float> dummy_output(num_rows * hidden_size, 0.0f);

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "identity");
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOptionalInputEdge<uint8_t>();
  cpu_tester.AddOptionalInputEdge<float>();
  cpu_tester.AddOptionalInputEdge<MLFloat16>();
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(dummy_output));

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectFailure,
                 "hidden_size (6) must be divisible by pack_size (4)",
                 {},
                 nullptr,
                 &cpu_execution_providers);
}

static void RunQMoEMixedWidthContractTest(bool invalid_fc1_shape,
                                          std::unique_ptr<IExecutionProvider> execution_provider,
                                          const char* provider_name,
                                          bool use_raw_weights = false,
                                          bool use_float16_scales = false,
                                          int64_t hidden_size = 8,
                                          int64_t inter_size = 8,
                                          bool legacy_layout = false,
                                          const char* execution_error = nullptr) {
  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t fc1_bits = 2;
  constexpr int64_t fc2_bits = 4;
  constexpr int64_t fc1_pack_size = 8 / fc1_bits;
  constexpr int64_t fc2_pack_size = 8 / fc2_bits;

  const std::vector<int64_t> input_dims = {num_rows, hidden_size};
  const std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_weights_dims = legacy_layout
                                              ? std::vector<int64_t>{num_experts, hidden_size,
                                                                     inter_size / fc1_pack_size}
                                              : std::vector<int64_t>{num_experts, inter_size,
                                                                     hidden_size / fc1_pack_size};
  if (invalid_fc1_shape) {
    ++fc1_weights_dims[2];
  }
  const std::vector<int64_t> fc2_weights_dims = legacy_layout
                                                    ? std::vector<int64_t>{num_experts, inter_size,
                                                                           hidden_size / fc2_pack_size}
                                                    : std::vector<int64_t>{num_experts, hidden_size,
                                                                           inter_size / fc2_pack_size};
  const std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size};
  const std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
  if (use_raw_weights) {
    tester.AddAttribute<int64_t>("weights_prepacked", 0);
  }
  tester.AddInput<MLFloat16>("input", input_dims, std::vector<MLFloat16>(num_rows * hidden_size));
  tester.AddInput<MLFloat16>("router_probs", router_probs_dims, std::vector<MLFloat16>(num_rows * num_experts));
  tester.AddInput<uint8_t>("fc1_experts_weights", fc1_weights_dims,
                           std::vector<uint8_t>(static_cast<size_t>(fc1_weights_dims[1] * fc1_weights_dims[2])));
  if (use_float16_scales) {
    tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims,
                               std::vector<MLFloat16>(num_experts * inter_size, MLFloat16(1.0f)));
  } else {
    tester.AddInput<float>("fc1_scales", fc1_scales_dims, std::vector<float>(num_experts * inter_size, 1.0f));
  }
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", fc2_weights_dims,
                           std::vector<uint8_t>(static_cast<size_t>(fc2_weights_dims[1] * fc2_weights_dims[2])));
  if (use_float16_scales) {
    tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims,
                               std::vector<MLFloat16>(num_experts * hidden_size, MLFloat16(1.0f)));
  } else {
    tester.AddInput<float>("fc2_scales", fc2_scales_dims, std::vector<float>(num_experts * hidden_size, 1.0f));
  }
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", input_dims, std::vector<MLFloat16>(num_rows * hidden_size));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  const std::string expected_error =
      invalid_fc1_shape ? "Input 'fc1_experts_weights' is expected to have shape"
      : execution_error != nullptr
          ? execution_error
          : MakeString("Mixed-width QMoE execution is not yet implemented on ", provider_name, ".");
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             expected_error, {}, nullptr, &execution_providers);
}

TEST(MoETest, QMoETest_MixedWidthContract) {
  RunQMoEMixedWidthContractTest(false, DefaultCpuExecutionProvider(), "CPU");
}

TEST(MoETest, QMoETest_MixedWidthInvalidFC1Shape) {
  RunQMoEMixedWidthContractTest(true, DefaultCpuExecutionProvider(), "CPU");
}

TEST(MoETest, QMoETest_MixedWidthNonSquareLayouts) {
  RunQMoEMixedWidthContractTest(false, DefaultCpuExecutionProvider(), "CPU", false, false, 8, 16, false);
  RunQMoEMixedWidthContractTest(false, DefaultCpuExecutionProvider(), "CPU", false, false, 8, 16, true);
}

#if defined(USE_CUDA)
TEST(MoETest, QMoETest_MixedWidthContract_CUDA) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthContractTest(false, DefaultCudaExecutionProvider(), "CUDA", true, true, 8, 8, false,
                                "requires block_size to be a power of two in [16, 256]");
}

static void RunQMoEMixedWidthCudaIdentityTest(int64_t fc1_bits, int64_t fc2_bits,
                                              int64_t max_scratch_bytes = 0,
                                              bool fused_swiglu = false,
                                              bool with_zero_points = false,
                                              bool use_bf16 = false,
                                              int64_t block_size = 32,
                                              int64_t hidden_size = 64,
                                              int64_t inter_size = 64,
                                              bool expect_scratch_failure = true,
                                              bool use_initializers = false,
                                              int64_t num_rows = 1,
                                              int64_t row_tile_size = 0) {
  constexpr int64_t num_experts = 1;

  auto make_identity = [block_size](int64_t bits, int64_t rows, int64_t columns,
                                    bool interleaved_rows, bool asymmetric) {
    const int64_t pack_size = 8 / bits;
    const int64_t blocks_per_row = columns / block_size;
    const int64_t packed_blocks_per_row = (blocks_per_row + pack_size - 1) / pack_size;
    const uint8_t default_zero = static_cast<uint8_t>(1u << (bits - 1));
    const uint8_t max_asymmetric_zero = static_cast<uint8_t>((1u << bits) - 2u);
    std::vector<uint8_t> weights(static_cast<size_t>(rows * columns / pack_size), 0);
    std::vector<uint8_t> zero_points(static_cast<size_t>(rows * packed_blocks_per_row), 0);
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t identity_column = interleaved_rows ? row / 2 : row;
      for (int64_t column = 0; column < columns; ++column) {
        const int64_t block = column / block_size;
        const uint8_t zero = asymmetric
                                 ? static_cast<uint8_t>(1 + ((row + block) % max_asymmetric_zero))
                                 : default_zero;
        const uint8_t code = static_cast<uint8_t>(zero + (column == identity_column ? 1 : 0));
        const int64_t byte_index = row * (columns / pack_size) + column / pack_size;
        weights[static_cast<size_t>(byte_index)] |=
            static_cast<uint8_t>(code << ((column % pack_size) * bits));
        if (asymmetric && column % block_size == 0) {
          zero_points[static_cast<size_t>(row * packed_blocks_per_row + block / pack_size)] |=
              static_cast<uint8_t>(zero << ((block % pack_size) * bits));
        }
      }
    }
    return std::make_pair(std::move(weights), std::move(zero_points));
  };

  std::vector<float> input(static_cast<size_t>(num_rows * hidden_size));
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
  }
  const std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts), 1.0f);
  const int64_t fc1_rows = fused_swiglu ? 2 * inter_size : inter_size;
  auto [fc1_weights, fc1_zero_points] =
      make_identity(fc1_bits, fc1_rows, hidden_size, fused_swiglu, with_zero_points);
  auto [fc2_weights, fc2_zero_points] =
      make_identity(fc2_bits, hidden_size, inter_size, false, with_zero_points);
  const std::vector<float> fc1_scales(static_cast<size_t>(fc1_rows * (hidden_size / block_size)), 1.0f);
  const std::vector<float> fc2_scales(static_cast<size_t>(hidden_size * (inter_size / block_size)), 1.0f);
  std::vector<float> expected = input;
  if (fused_swiglu) {
    for (float& value : expected) {
      const float rounded = use_bf16 ? BFloat16(value).ToFloat() : MLFloat16(value).ToFloat();
      value = rounded / (1.0f + std::exp(-rounded)) * rounded;
    }
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  const int64_t expert_bits = fc1_bits == 2 && fc2_bits == 2 ? 2 : 4;
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", fused_swiglu ? "swiglu" : "identity");
  tester.AddAttribute<int64_t>("swiglu_fusion", fused_swiglu ? 1 : 0);
  tester.AddAttribute<int64_t>("expert_weight_bits", expert_bits);
  if (fc1_bits != expert_bits) {
    tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
    tester.AddAttribute<int64_t>("fc3_expert_weight_bits", fc1_bits);
  }
  if (fc2_bits != expert_bits) {
    tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
  }
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddAttribute<int64_t>("block_size", block_size);
  if (use_bf16) {
    tester.AddInput<BFloat16>("input", {num_rows, hidden_size}, ToBFloat16(input));
    tester.AddInput<BFloat16>("router_probs", {num_rows, num_experts}, ToBFloat16(router_probs));
    tester.AddInput<uint8_t>("fc1_experts_weights",
                             {num_experts, fc1_rows, hidden_size / (8 / fc1_bits)}, fc1_weights, use_initializers);
    tester.AddInput<BFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                              ToBFloat16(fc1_scales), use_initializers);
    tester.AddOptionalInputEdge<BFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights",
                             {num_experts, hidden_size, inter_size / (8 / fc2_bits)}, fc2_weights, use_initializers);
    tester.AddInput<BFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                              ToBFloat16(fc2_scales), use_initializers);
    tester.AddOptionalInputEdge<BFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<BFloat16>();
    tester.AddOptionalInputEdge<BFloat16>();
  } else {
    tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
    tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
    tester.AddInput<uint8_t>("fc1_experts_weights",
                             {num_experts, fc1_rows, hidden_size / (8 / fc1_bits)}, fc1_weights, use_initializers);
    tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                               ToFloat16(fc1_scales), use_initializers);
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights",
                             {num_experts, hidden_size, inter_size / (8 / fc2_bits)}, fc2_weights, use_initializers);
    tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                               ToFloat16(fc2_scales), use_initializers);
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<MLFloat16>();
  }
  if (with_zero_points) {
    tester.AddInput<uint8_t>("fc1_zero_points",
                             {num_experts, fc1_rows, (hidden_size / block_size + (8 / fc1_bits) - 1) / (8 / fc1_bits)},
                             fc1_zero_points);
    tester.AddInput<uint8_t>("fc2_zero_points",
                             {num_experts, hidden_size, (inter_size / block_size + (8 / fc2_bits) - 1) / (8 / fc2_bits)},
                             fc2_zero_points);
  }
  if (use_bf16) {
    tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, ToBFloat16(expected));
  } else {
    tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  }
  tester.SetOutputTolerance(0.02f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsDisableCPUEPFallback, "1"));
  if (max_scratch_bytes > 0) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        "ep.cuda.qmoe_int_dequant_max_scratch_bytes", std::to_string(max_scratch_bytes).c_str()));
  }
  if (row_tile_size > 0) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        "ep.cuda.qmoe_row_tile_size", std::to_string(row_tile_size).c_str()));
  }
  tester.Run(session_options,
             max_scratch_bytes > 0 && expect_scratch_failure ? OpTester::ExpectResult::kExpectFailure
                                                             : OpTester::ExpectResult::kExpectSuccess,
             max_scratch_bytes > 0 && expect_scratch_failure ? "exceeding the configured limit" : "",
             {}, nullptr, &execution_providers);
}

TEST(MoETest, QMoETest_MixedWidthCudaBlockWise) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(2, 4);
  RunQMoEMixedWidthCudaIdentityTest(4, 2);
  RunQMoEMixedWidthCudaIdentityTest(2, 2);
  RunQMoEMixedWidthCudaIdentityTest(2, 8);
  RunQMoEMixedWidthCudaIdentityTest(8, 2);
  RunQMoEMixedWidthCudaIdentityTest(4, 8);
  RunQMoEMixedWidthCudaIdentityTest(8, 4);
}

TEST(MoETest, QMoETest_MixedWidthCudaCanonicalNonSquare) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 4, /*max_scratch_bytes=*/0, /*fused_swiglu=*/false, /*with_zero_points=*/false,
      /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/64, /*inter_size=*/128);
}

TEST(MoETest, QMoETest_MixedWidthCudaSingleBlock) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 4, /*max_scratch_bytes=*/0, /*fused_swiglu=*/false, /*with_zero_points=*/true,
      /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/32, /*inter_size=*/64);
}

#if !defined(ORT_QUICK_BUILD) && defined(ENABLE_BF16)
TEST(MoETest, QMoETest_MixedWidthCudaBlockWiseBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 0, false, false, true);
}
#endif

TEST(MoETest, QMoETest_MixedWidthCudaScratchLimit) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1);
}

TEST(MoETest, QMoETest_MixedWidthCudaFusedSwiGLU) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 0, true);
}

TEST(MoETest, QMoETest_Int2CudaPackedDecode) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 2, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true, /*with_zero_points=*/false,
      /*use_bf16=*/false, /*block_size=*/64, /*hidden_size=*/512, /*inter_size=*/512,
      /*expect_scratch_failure=*/false, /*use_initializers=*/true);
}

TEST(MoETest, QMoETest_Int2CudaPackedDecodeBlock128) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 2, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true, /*with_zero_points=*/false,
      /*use_bf16=*/false, /*block_size=*/128, /*hidden_size=*/512, /*inter_size=*/512,
      /*expect_scratch_failure=*/false, /*use_initializers=*/true);
}

// Runs the same multi-row, multi-expert, k=2 INT2 SwiGLU model once with the packed GEMV
// path selected and once with it disabled (forcing the already-validated dense fallback from
// #32743), so a stride/permutation/top-k bug in the new packed pipeline shows up as a numeric
// mismatch instead of only a shape or gate mismatch.
static std::vector<float> RunQMoEPackedIntGemvParityCase(bool disable_packed_gemv) {
  constexpr int64_t num_rows = 4;
  constexpr int64_t num_experts = 4;
  constexpr int64_t k = 2;
  constexpr int64_t hidden_size = 512;
  constexpr int64_t inter_size = 512;
  constexpr int64_t block_size = 64;
  constexpr int64_t bits = 2;
  constexpr int64_t pack_size = 8 / bits;
  constexpr int64_t fc1_rows = 2 * inter_size;  // Fused SwiGLU doubles FC1's output rows.

  uint32_t rng_state = 12345u;
  auto next_rand = [&rng_state]() {
    rng_state = rng_state * 1103515245u + 12345u;
    return rng_state;
  };

  std::vector<float> input(static_cast<size_t>(num_rows * hidden_size));
  for (float& value : input) {
    value = static_cast<float>(next_rand() % 17) * 0.05f - 0.4f;
  }

  // Give each row a distinct preferred expert pair so k=2 top-k selection and the resulting
  // expert permutation differ across rows.
  constexpr int64_t preferred_pairs[num_rows][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts));
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t e = 0; e < num_experts; ++e) {
      router_probs[static_cast<size_t>(row * num_experts + e)] = 0.1f * static_cast<float>(e);
    }
    router_probs[static_cast<size_t>(row * num_experts + preferred_pairs[row][0])] = 3.0f;
    router_probs[static_cast<size_t>(row * num_experts + preferred_pairs[row][1])] = 2.0f;
  }

  // Each expert's weight is a "shifted identity": row r picks out input column
  // (r + expert_shift) % columns, so a bug that mixes up per-expert strides or permutation
  // changes the numeric result rather than only the output shape.
  auto make_expert_weights = [](int64_t rows, int64_t columns, int64_t expert_shift) {
    std::vector<uint8_t> weights(static_cast<size_t>(rows * columns / pack_size), 0);
    constexpr uint8_t base_code = static_cast<uint8_t>(1u << (bits - 1));
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t identity_column = (row + expert_shift) % columns;
      for (int64_t column = 0; column < columns; ++column) {
        const uint8_t code = static_cast<uint8_t>(base_code + (column == identity_column ? 1 : 0));
        const int64_t byte_index = row * (columns / pack_size) + column / pack_size;
        weights[static_cast<size_t>(byte_index)] |=
            static_cast<uint8_t>(code << ((column % pack_size) * bits));
      }
    }
    return weights;
  };

  std::vector<uint8_t> fc1_weights(static_cast<size_t>(num_experts * fc1_rows * hidden_size / pack_size));
  std::vector<uint8_t> fc2_weights(static_cast<size_t>(num_experts * hidden_size * inter_size / pack_size));
  for (int64_t e = 0; e < num_experts; ++e) {
    const auto fc1_e = make_expert_weights(fc1_rows, hidden_size, e * 7);
    const auto fc2_e = make_expert_weights(hidden_size, inter_size, e * 11);
    std::copy(fc1_e.begin(), fc1_e.end(), fc1_weights.begin() + static_cast<int64_t>(fc1_e.size()) * e);
    std::copy(fc2_e.begin(), fc2_e.end(), fc2_weights.begin() + static_cast<int64_t>(fc2_e.size()) * e);
  }
  const std::vector<float> fc1_scales(static_cast<size_t>(num_experts * fc1_rows * (hidden_size / block_size)), 1.0f);
  const std::vector<float> fc2_scales(static_cast<size_t>(num_experts * hidden_size * (inter_size / block_size)), 1.0f);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", bits);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddAttribute<int64_t>("block_size", block_size);

  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, fc1_rows, hidden_size / pack_size},
                           fc1_weights, /*is_initializer=*/true);
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                             ToFloat16(fc1_scales), /*is_initializer=*/true);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / pack_size},
                           fc2_weights, /*is_initializer=*/true);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                             ToFloat16(fc2_scales), /*is_initializer=*/true);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<MLFloat16>();

  // The real assertion is the packed-vs-dense parity check the caller performs on GetFetches();
  // use a placeholder expected output and a huge tolerance so the framework's own comparison
  // never fails here.
  const std::vector<float> placeholder(static_cast<size_t>(num_rows * hidden_size), 0.0f);
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(placeholder));
  tester.SetOutputTolerance(1e6f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  SessionOptions session_options;
  // ASSERT_STATUS_OK contains a bare `return;`, which only compiles in void-returning
  // functions; use EXPECT_TRUE here since this helper returns a vector.
  EXPECT_TRUE(session_options.config_options.AddConfigEntry(
                                                kOrtSessionOptionsDisableCPUEPFallback, "1")
                  .IsOK());
  if (!disable_packed_gemv) {
    // A one-byte dense-dequant scratch cap proves the packed path actually ran: if it silently
    // fell back to dense, this run would fail loudly instead of producing a plausible-looking
    // but wrong result.
    EXPECT_TRUE(session_options.config_options.AddConfigEntry(
                                                  "ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1")
                    .IsOK());
  }

  EnvVarMap env_overrides;
  if (disable_packed_gemv) {
    env_overrides["ORT_DISABLE_MOE_GEMV"] = "1";
  }
  ScopedEnvironmentVariables scoped_env_vars{env_overrides};

  tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  const auto fetches = tester.GetFetches();
  const Tensor& out = fetches[0].Get<Tensor>();
  const MLFloat16* data = out.Data<MLFloat16>();
  std::vector<float> result(static_cast<size_t>(out.Shape().Size()));
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = data[i].ToFloat();
  }
  return result;
}

TEST(MoETest, QMoETest_Int2CudaPackedDecodeMultiRowExpertParity) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  const std::vector<float> packed_output = RunQMoEPackedIntGemvParityCase(/*disable_packed_gemv=*/false);
  const std::vector<float> dense_output = RunQMoEPackedIntGemvParityCase(/*disable_packed_gemv=*/true);
  ASSERT_EQ(packed_output.size(), dense_output.size());
  for (size_t i = 0; i < packed_output.size(); ++i) {
    EXPECT_NEAR(packed_output[i], dense_output[i], 0.02f) << "mismatch at index " << i;
  }
}

#ifdef USE_CUDA
static std::string BuildQMoEInt2DynamicRowsModel() {
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 512;
  constexpr int64_t inter_size = 512;
  constexpr int64_t block_size = 64;
  constexpr int64_t pack_size = 4;
  constexpr int64_t fc1_rows = 2 * inter_size;

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);
  auto* graph = model.mutable_graph();
  graph->set_name("qmoe_int2_dynamic_rows");

  auto add_fp16_value_info = [graph](const char* name, int64_t width, bool is_input) {
    auto* value_info = is_input ? graph->add_input() : graph->add_output();
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    tensor_type->mutable_shape()->add_dim()->set_dim_param("num_rows");
    tensor_type->mutable_shape()->add_dim()->set_dim_value(width);
  };
  add_fp16_value_info("input", hidden_size, true);
  add_fp16_value_info("router_probs", num_experts, true);
  add_fp16_value_info("output", hidden_size, false);

  auto make_identity_weights = [](int64_t rows, int64_t columns) {
    std::vector<uint8_t> weights(static_cast<size_t>(rows * columns / pack_size), 0xAA);
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t column = row % columns;
      const size_t byte_index = static_cast<size_t>(row * (columns / pack_size) + column / pack_size);
      weights[byte_index] |= static_cast<uint8_t>(1u << ((column % pack_size) * 2));
    }
    return weights;
  };
  const auto fc1_weights = make_identity_weights(fc1_rows, hidden_size);
  const auto fc2_weights = make_identity_weights(hidden_size, inter_size);

  auto make_scales = [](int64_t rows, int64_t blocks) {
    std::vector<MLFloat16> scales(static_cast<size_t>(rows * blocks));
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t block = 0; block < blocks; ++block) {
        scales[static_cast<size_t>(row * blocks + block)] = MLFloat16(0.5f + 0.0625f * block);
      }
    }
    return scales;
  };
  const auto fc1_scales = make_scales(fc1_rows, hidden_size / block_size);
  const auto fc2_scales = make_scales(hidden_size, inter_size / block_size);

  auto add_initializer = [graph](const char* name, int32_t data_type, const std::vector<int64_t>& dims,
                                 const void* data, size_t bytes) {
    auto* initializer = graph->add_initializer();
    initializer->set_name(name);
    initializer->set_data_type(data_type);
    for (int64_t dim : dims) {
      initializer->add_dims(dim);
    }
    initializer->mutable_raw_data()->assign(static_cast<const char*>(data), bytes);
  };
  add_initializer("fc1_weights", ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                  {num_experts, fc1_rows, hidden_size / pack_size}, fc1_weights.data(), fc1_weights.size());
  add_initializer("fc1_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                  {num_experts, fc1_rows, hidden_size / block_size}, fc1_scales.data(),
                  fc1_scales.size() * sizeof(MLFloat16));
  add_initializer("fc2_weights", ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                  {num_experts, hidden_size, inter_size / pack_size}, fc2_weights.data(), fc2_weights.size());
  add_initializer("fc2_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                  {num_experts, hidden_size, inter_size / block_size}, fc2_scales.data(),
                  fc2_scales.size() * sizeof(MLFloat16));

  auto* node = graph->add_node();
  node->set_op_type("QMoE");
  node->set_domain(kMSDomain);
  for (const char* input_name : {"input", "router_probs", "fc1_weights", "fc1_scales", "",
                                 "fc2_weights", "fc2_scales", "", "", "", ""}) {
    node->add_input(input_name);
  }
  node->add_output("output");
  auto add_int_attribute = [node](const char* name, int64_t value) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  };
  add_int_attribute("k", 1);
  add_int_attribute("swiglu_fusion", 1);
  add_int_attribute("normalize_routing_weights", 1);
  add_int_attribute("expert_weight_bits", 2);
  add_int_attribute("weights_prepacked", 0);
  add_int_attribute("block_size", block_size);
  auto* activation = node->add_attribute();
  activation->set_name("activation_type");
  activation->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  activation->set_s("swiglu");

  std::string model_bytes;
  ORT_ENFORCE(model.SerializeToString(&model_bytes));
  return model_bytes;
}

TEST(MoETest, QMoETest_Int2CudaCachedScalesDecodeThenPrefill) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }

  constexpr int64_t hidden_size = 512;
  const std::string model_bytes = BuildQMoEInt2DynamicRowsModel();
#if defined(ENABLE_CUDA_PROFILING)
  const auto profile_prefix = std::filesystem::temp_directory_path() / "qmoe_int2_cached_scales";
#endif
  SessionOptions session_options;
#if defined(ENABLE_CUDA_PROFILING)
  session_options.enable_profiling = true;
  session_options.profile_file_prefix = profile_prefix.native();
#endif
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

#if defined(ENABLE_CUDA_PROFILING)
  const std::string setup_profile_path = session.EndProfiling();
  std::error_code setup_profile_error;
  std::filesystem::remove(setup_profile_path, setup_profile_error);
  session.StartProfiling(profile_prefix.string());
#endif

  const auto run_rows = [&session](int64_t num_rows) {
    std::vector<MLFloat16> input(static_cast<size_t>(num_rows * hidden_size), MLFloat16(0.1f));
    std::vector<MLFloat16> router_probs(static_cast<size_t>(num_rows), MLFloat16(1.0f));
    OrtValue input_value;
    OrtValue router_value;
    Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), TensorShape({num_rows, hidden_size}),
                         input.data(), OrtMemoryInfo(), input_value);
    Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), TensorShape({num_rows, 1}),
                         router_probs.data(), OrtMemoryInfo(), router_value);
    NameMLValMap feeds{{"input", input_value}, {"router_probs", router_value}};
    const std::vector<std::string> output_names{"output"};
    std::vector<OrtValue> fetches;
    EXPECT_STATUS_OK(session.Run(feeds, output_names, &fetches));
    if (fetches.empty()) {
      return std::vector<float>{};
    }
    const Tensor& output = fetches[0].Get<Tensor>();
    const MLFloat16* output_data = output.Data<MLFloat16>();
    std::vector<float> result(static_cast<size_t>(output.Shape().Size()));
    std::transform(output_data, output_data + result.size(), result.begin(),
                   [](MLFloat16 value) { return value.ToFloat(); });
    return result;
  };

  const std::vector<float> first_decode = run_rows(1);
  const std::vector<float> second_decode = run_rows(1);
  ASSERT_EQ(first_decode.size(), static_cast<size_t>(hidden_size));
  ASSERT_EQ(first_decode, second_decode);
#if defined(ENABLE_CUDA_PROFILING)
  const std::string profile_path = session.EndProfiling();
  auto remove_profile = gsl::finally([&profile_path] {
    std::error_code error;
    std::filesystem::remove(profile_path, error);
  });

  if (session.GetProfiling().HasEpProfilers() && session.GetProfiling().GetEpProfilingStatus().IsOK()) {
    std::ifstream profile_stream(profile_path);
    ASSERT_TRUE(profile_stream.is_open());
    const auto profile = nlohmann::json::parse(profile_stream);
    bool saw_kernel_event = false;
    for (const auto& event : profile) {
      if (event.value("cat", "") == "Kernel") {
        saw_kernel_event = true;
        EXPECT_EQ(event.value("name", "").find("QMoETranspose2DKernel"), std::string::npos)
            << "initializer scales were transposed during repeated decode";
      }
    }
    EXPECT_TRUE(saw_kernel_event);
  }
#endif

  const std::vector<float> prefill = run_rows(257);
  ASSERT_EQ(prefill.size(), static_cast<size_t>(257 * hidden_size));
  for (size_t column = 0; column < static_cast<size_t>(hidden_size); ++column) {
    EXPECT_NEAR(prefill[column], first_decode[column], 0.02f) << "mismatch at column " << column;
  }
}
#endif

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillDefaultEnabled) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", std::nullopt}, {"ORT_ENABLE_QMOE_INT2_PREFILL", std::nullopt}}};
  for (int64_t num_rows : {1, 33, 257}) {
    SCOPED_TRACE(num_rows);
    RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, false, true, num_rows);
#if defined(ENABLE_BF16)
    RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, true, 64, 512, 512, false, true, num_rows);
#endif
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillWithoutDecode) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  for (int64_t num_rows : {1, 33, 257}) {
    SCOPED_TRACE(num_rows);
    RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, false, true, num_rows);
  }
}

#if defined(ENABLE_BF16)
TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillBFloat16WithoutDecode) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  for (int64_t num_rows : {1, 33, 257}) {
    SCOPED_TRACE(num_rows);
    RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, true, 64, 512, 512, false, true, num_rows);
  }
}
#endif

template <typename ElementType>
static void RunQMoEPackedPrefillRoutingTest(int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                                            int64_t top_k, bool skewed, bool scale_initializers = true,
                                            int64_t row_tile_size = 0, int64_t block_size = 64,
                                            int64_t fc1_bits = 2, int64_t fc2_bits = 4) {
  constexpr int64_t num_experts = 8;
  constexpr float alpha = 1.5f;
  constexpr float beta = 0.125f;
  constexpr float limit = 0.5f;
  const auto round_value = [](float value) { return ElementType(value).ToFloat(); };
  const auto fc1_scale = [](int64_t expert, int64_t column, int64_t block) {
    return 0.5f + 0.125f * static_cast<float>(expert) +
           0.015625f * static_cast<float>((column / 2) % 7) + 0.03125f * static_cast<float>(block);
  };
  const auto fc2_scale = [](int64_t expert, int64_t column, int64_t block) {
    return 0.125f * static_cast<float>(expert + 1) +
           0.015625f * static_cast<float>(column % 5) + 0.03125f * static_cast<float>(block);
  };
  const auto make_weights = [](int64_t rows, int64_t columns, int64_t bits, bool interleaved) {
    const int64_t pack_size = 8 / bits;
    const uint8_t zero = static_cast<uint8_t>(1u << (bits - 1));
    std::vector<uint8_t> weights(static_cast<size_t>(num_experts * rows * columns / pack_size), 0);
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      for (int64_t row = 0; row < rows; ++row) {
        const int64_t selected_column = (interleaved ? row / 2 : row) % columns;
        for (int64_t column = 0; column < columns; ++column) {
          const uint8_t code = static_cast<uint8_t>(zero + (column == selected_column));
          weights[static_cast<size_t>(((expert * rows + row) * columns + column) / pack_size)] |=
              static_cast<uint8_t>(code << ((column % pack_size) * bits));
        }
      }
    }
    return weights;
  };
  const auto fc1_weights = make_weights(2 * inter_size, hidden_size, fc1_bits, true);
  const auto fc2_weights = make_weights(hidden_size, inter_size, fc2_bits, false);
  std::vector<ElementType> input(static_cast<size_t>(num_rows * hidden_size));
  std::vector<ElementType> router(static_cast<size_t>(num_rows * num_experts), ElementType(-20.0f));
  std::vector<ElementType> fc1_scales(static_cast<size_t>(num_experts * 2 * inter_size * hidden_size / block_size));
  std::vector<ElementType> fc2_scales(static_cast<size_t>(num_experts * hidden_size * inter_size / block_size));
  std::vector<ElementType> fc1_bias(static_cast<size_t>(num_experts * 2 * inter_size));
  std::vector<ElementType> fc2_bias(static_cast<size_t>(num_experts * hidden_size));
  std::vector<ElementType> expected(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = ElementType(static_cast<float>(static_cast<int>(index % 29) - 14) * 0.125f);
  }
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    for (int64_t column = 0; column < 2 * inter_size; ++column) {
      for (int64_t block = 0; block < hidden_size / block_size; ++block) {
        fc1_scales[static_cast<size_t>((expert * 2 * inter_size + column) * hidden_size / block_size + block)] =
            ElementType(fc1_scale(expert, column, block));
      }
    }
    for (int64_t column = 0; column < hidden_size; ++column) {
      for (int64_t block = 0; block < inter_size / block_size; ++block) {
        fc2_scales[static_cast<size_t>((expert * hidden_size + column) * inter_size / block_size + block)] =
            ElementType(fc2_scale(expert, column, block));
      }
    }
    for (int64_t column = 0; column < inter_size; ++column) {
      fc1_bias[static_cast<size_t>(expert * 2 * inter_size + 2 * column)] =
          ElementType(0.0625f * static_cast<float>(expert - 2) +
                      0.015625f * static_cast<float>(column % 11 - 5));
      fc1_bias[static_cast<size_t>(expert * 2 * inter_size + 2 * column + 1)] =
          ElementType(-0.03125f * static_cast<float>(expert + 1) +
                      0.015625f * static_cast<float>(column % 7 - 3));
    }
    for (int64_t column = 0; column < hidden_size; ++column) {
      fc2_bias[static_cast<size_t>(expert * hidden_size + column)] =
          ElementType(0.03125f * static_cast<float>(expert - 3) +
                      0.015625f * static_cast<float>(column % 13 - 6));
    }
  }
  float denominator = 0.0f;
  for (int64_t rank = 0; rank < top_k; ++rank) {
    denominator += std::exp(-static_cast<float>(rank));
  }
  for (int64_t row = 0; row < num_rows; ++row) {
    const int64_t start_expert = skewed && row + 2 < num_rows ? 0 : row % (num_experts - 2);
    for (int64_t rank = 0; rank < top_k; ++rank) {
      const int64_t expert = 1 + (start_expert + rank) % (num_experts - 2);
      router[static_cast<size_t>(row * num_experts + expert)] = ElementType(4.0f - static_cast<float>(rank));
    }
    for (int64_t column = 0; column < hidden_size; ++column) {
      float sum = 0.0f;
      for (int64_t rank = 0; rank < top_k; ++rank) {
        const int64_t expert = 1 + (start_expert + rank) % (num_experts - 2);
        const int64_t input_column = (column % inter_size) % hidden_size;
        const float fc1 = round_value(input[static_cast<size_t>(row * hidden_size + input_column)].ToFloat() *
                                      round_value(fc1_scale(expert, 2 * (column % inter_size), input_column / block_size)));
        const size_t gate_bias_index = static_cast<size_t>(expert * 2 * inter_size + 2 * (column % inter_size));
        const float gate = std::min(fc1 + fc1_bias[gate_bias_index].ToFloat(), limit);
        const float linear = std::clamp(
                                 fc1 + fc1_bias[gate_bias_index + 1].ToFloat(), -limit, limit) +
                             beta;
        const float activated = round_value(gate / (1.0f + std::exp(-alpha * gate)) * linear);
        const float fc2 = round_value(activated *
                                      round_value(fc2_scale(expert, column, (column % inter_size) / block_size)));
        sum += (fc2 + fc2_bias[static_cast<size_t>(expert * hidden_size + column)].ToFloat()) *
               std::exp(-static_cast<float>(rank)) / denominator;
      }
      expected[static_cast<size_t>(row * hidden_size + column)] = ElementType(sum);
    }
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<float>("activation_alpha", alpha);
  tester.AddAttribute<float>("activation_beta", beta);
  tester.AddAttribute<float>("swiglu_limit", limit);
  tester.AddAttribute<int64_t>("expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
  tester.AddAttribute<int64_t>("fc3_expert_weight_bits", fc1_bits);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddAttribute<int64_t>("block_size", block_size);
  tester.AddInput<ElementType>("input", {num_rows, hidden_size}, input);
  tester.AddInput<ElementType>("router_probs", {num_rows, num_experts}, router);
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, 2 * inter_size, hidden_size / (8 / fc1_bits)}, fc1_weights, true);
  tester.AddInput<ElementType>("fc1_scales", {num_experts, 2 * inter_size, hidden_size / block_size},
                               fc1_scales, scale_initializers);
  tester.AddInput<ElementType>("fc1_experts_bias", {num_experts, 2 * inter_size}, fc1_bias, true);
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / (8 / fc2_bits)}, fc2_weights, true);
  tester.AddInput<ElementType>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                               fc2_scales, scale_initializers);
  tester.AddInput<ElementType>("fc2_experts_bias", {num_experts, hidden_size}, fc2_bias, true);
  tester.AddOutput<ElementType>("output", {num_rows, hidden_size}, expected);
  tester.SetOutputTolerance(std::is_same_v<ElementType, BFloat16> ? 0.004f : 0.001f);
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1"));
  if (row_tile_size > 0) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        "ep.cuda.qmoe_row_tile_size", std::to_string(row_tile_size).c_str()));
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillRouting) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  for (int64_t top_k : {1, 2, 4}) {
    for (bool skewed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "top_k=" << top_k << " skewed=" << skewed);
      RunQMoEPackedPrefillRoutingTest<MLFloat16>(33, 512, 256, top_k, skewed);
#if defined(ENABLE_BF16)
      RunQMoEPackedPrefillRoutingTest<BFloat16>(257, 512, 256, top_k, skewed);
#endif
    }
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillRuntimeScales) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  RunQMoEPackedPrefillRoutingTest<MLFloat16>(33, 128, 64, 2, false, false);
  RunQMoEPackedPrefillRoutingTest<MLFloat16>(33, 64, 128, 4, true, false);
#if defined(ENABLE_BF16)
  RunQMoEPackedPrefillRoutingTest<BFloat16>(33, 128, 64, 2, false, false);
  RunQMoEPackedPrefillRoutingTest<BFloat16>(33, 64, 128, 4, true, false);
#endif
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillBlockSizes) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  for (int64_t block_size : {32, 64, 128}) {
    SCOPED_TRACE(block_size);
    for (const auto& weight_bits : {std::pair<int64_t, int64_t>{2, 4}, {2, 2}, {4, 2}}) {
      SCOPED_TRACE(testing::Message() << "fc1_bits=" << weight_bits.first << " fc2_bits=" << weight_bits.second);
      for (int64_t num_rows : {1, 33, 257}) {
        for (bool scale_initializers : {false, true}) {
          for (int64_t row_tile_size : {0, 16}) {
            SCOPED_TRACE(testing::Message() << "rows=" << num_rows
                                            << " scale_initializers=" << scale_initializers
                                            << " row_tile_size=" << row_tile_size);
            RunQMoEPackedPrefillRoutingTest<MLFloat16>(num_rows, 512, 256, 2, false,
                                                       scale_initializers, row_tile_size, block_size,
                                                       weight_bits.first, weight_bits.second);
#if defined(ENABLE_BF16)
            RunQMoEPackedPrefillRoutingTest<BFloat16>(num_rows, 512, 256, 4, true,
                                                      scale_initializers, row_tile_size, block_size,
                                                      weight_bits.first, weight_bits.second);
#endif
          }
        }
      }
    }
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillFallback) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, true, false, 64, 512, 512, true, true, 33);
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 16, 512, 512, true, true, 33);
  RunQMoEMixedWidthCudaIdentityTest(2, 8, 1, true, false, false, 64, 512, 512, true, true, 33);
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, false, false, false, 64, 512, 512, true, true, 33);
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, true, false, 33);
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillWorkspaceLimit) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 64, 8192, false, true, 8193);
#if defined(ENABLE_BF16)
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, true, 64, 64, 8192, false, true, 8193);
#endif
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillRespectsRowTiling) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, false, true, 33, 33);
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, false, true, 33, 16);
#if defined(ENABLE_BF16)
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, true, 64, 512, 512, false, true, 257, 16);
#endif
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillTiledRouting) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "SM80 or later is required for packed INT2 prefill.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "1"}}};
  for (int64_t top_k : {1, 2, 4}) {
    for (bool scale_initializers : {false, true}) {
      SCOPED_TRACE(testing::Message() << "top_k=" << top_k << " scale_initializers=" << scale_initializers);
      RunQMoEPackedPrefillRoutingTest<MLFloat16>(33, 128, 64, top_k, false, scale_initializers, 16);
#if defined(ENABLE_BF16)
      RunQMoEPackedPrefillRoutingTest<BFloat16>(257, 64, 128, top_k, true, scale_initializers, 16);
#endif
    }
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedPrefillDisabled) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      {{"ORT_DISABLE_MOE_GEMV", "1"}, {"ORT_ENABLE_QMOE_INT2_PREFILL", "0"}}};
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, true, true, 1);
  RunQMoEMixedWidthCudaIdentityTest(2, 4, 1, true, false, false, 64, 512, 512, true, true, 33);
}

TEST(MoETest, QMoETest_MixedWidthCudaBlock32DenseFallbackWithoutPrefill) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_QMOE_INT2_PREFILL", "0"}}};
  for (const auto& weight_bits : {std::pair<int64_t, int64_t>{2, 4}, {2, 2}, {4, 2}}) {
    SCOPED_TRACE(testing::Message() << "fc1_bits=" << weight_bits.first << " fc2_bits=" << weight_bits.second);
    for (int64_t num_rows : {1, 33}) {
      SCOPED_TRACE(num_rows);
      RunQMoEMixedWidthCudaIdentityTest(
          weight_bits.first, weight_bits.second, /*max_scratch_bytes=*/0, /*fused_swiglu=*/true,
          /*with_zero_points=*/false, /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/512,
          /*inter_size=*/512, /*expect_scratch_failure=*/false, /*use_initializers=*/true, num_rows);
    }
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedDecode) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  for (const auto& bits : {std::pair<int64_t, int64_t>{2, 4},
                           std::pair<int64_t, int64_t>{4, 2}}) {
    RunQMoEMixedWidthCudaIdentityTest(
        bits.first, bits.second, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true,
        /*with_zero_points=*/false, /*use_bf16=*/false, /*block_size=*/64, /*hidden_size=*/512,
        /*inter_size=*/512, /*expect_scratch_failure=*/false, /*use_initializers=*/true);
  }
}

TEST(MoETest, QMoETest_Int2CudaPackedDecodeFallback) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  ScopedEnvironmentVariables scoped_env_vars{{{"ORT_ENABLE_QMOE_INT2_PREFILL", "0"}}};
  RunQMoEMixedWidthCudaIdentityTest(
      2, 2, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true, /*with_zero_points=*/false,
      /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/64, /*inter_size=*/64,
      /*expect_scratch_failure=*/true, /*use_initializers=*/true);
}

#if !defined(ORT_QUICK_BUILD) && defined(ENABLE_BF16)
TEST(MoETest, QMoETest_Int2CudaPackedDecodeBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 2, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true, /*with_zero_points=*/false,
      /*use_bf16=*/true, /*block_size=*/64, /*hidden_size=*/512, /*inter_size=*/512,
      /*expect_scratch_failure=*/false, /*use_initializers=*/true);
}

TEST(MoETest, QMoETest_MixedWidthCudaPackedDecodeBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  for (const auto& bits : {std::pair<int64_t, int64_t>{2, 4},
                           std::pair<int64_t, int64_t>{4, 2}}) {
    RunQMoEMixedWidthCudaIdentityTest(
        bits.first, bits.second, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true,
        /*with_zero_points=*/false, /*use_bf16=*/true, /*block_size=*/64, /*hidden_size=*/512,
        /*inter_size=*/512, /*expect_scratch_failure=*/false, /*use_initializers=*/true);
  }
}
#endif

TEST(MoETest, QMoETest_MixedWidthCudaPackedDecodeBlock32) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  for (const auto& bits : {std::pair<int64_t, int64_t>{2, 4},
                           std::pair<int64_t, int64_t>{4, 2},
                           std::pair<int64_t, int64_t>{2, 2}}) {
    for (int64_t num_rows : {1, 4}) {
      RunQMoEMixedWidthCudaIdentityTest(
          bits.first, bits.second, /*max_scratch_bytes=*/1, /*fused_swiglu=*/true,
          /*with_zero_points=*/false, /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/512,
          /*inter_size=*/512, /*expect_scratch_failure=*/false, /*use_initializers=*/true, num_rows);
    }
  }
}

static std::vector<uint8_t> PackLsbBits(const std::vector<uint8_t>& codes, int64_t rows, int64_t cols, int64_t bits) {
  const int64_t row_bytes = (cols * bits + 7) / 8;
  std::vector<uint8_t> out(static_cast<size_t>(rows * row_bytes), 0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const uint8_t code = static_cast<uint8_t>(codes[static_cast<size_t>(r * cols + c)] & ((1u << bits) - 1u));
      const int64_t bit_offset = c * bits;
      const size_t byte_idx = static_cast<size_t>(r * row_bytes + bit_offset / 8);
      const int shift = static_cast<int>(bit_offset % 8);
      out[byte_idx] |= static_cast<uint8_t>(code << shift);
      if (shift + bits > 8) {
        out[byte_idx + 1] |= static_cast<uint8_t>(code >> (8 - shift));
      }
    }
  }
  return out;
}

TEST(MoETest, QMoETest_Int3AndMixedWidthCudaSwiGLU) {
  if (!HasCudaEnvironment(750)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.5 or newer is required.";
  }

  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 64;
  constexpr int64_t fc1_rows = 2 * inter_size;

  for (const auto& bits : {std::pair<int64_t, int64_t>{3, 3},
                           std::pair<int64_t, int64_t>{3, 4},
                           std::pair<int64_t, int64_t>{4, 3},
                           std::pair<int64_t, int64_t>{2, 3}}) {
    for (int64_t block_size : {32, 64}) {
      for (int64_t num_rows : {1, 4}) {
        const int64_t fc1_bits = bits.first;
        const int64_t fc2_bits = bits.second;
        const uint8_t z1 = static_cast<uint8_t>(1u << (fc1_bits - 1));
        const uint8_t z2 = static_cast<uint8_t>(1u << (fc2_bits - 1));

        std::vector<uint8_t> fc1_codes(static_cast<size_t>(num_experts * fc1_rows * hidden_size), z1);
        std::vector<uint8_t> fc2_codes(static_cast<size_t>(num_experts * hidden_size * inter_size), z2);
        for (int64_t e = 0; e < num_experts; ++e) {
          for (int64_t r = 0; r < fc1_rows; ++r) {
            const int64_t col = r / 2;
            fc1_codes[static_cast<size_t>((e * fc1_rows + r) * hidden_size + col)] = static_cast<uint8_t>(z1 + 1);
          }
          for (int64_t r = 0; r < hidden_size; ++r) {
            if (r < inter_size) {
              fc2_codes[static_cast<size_t>((e * hidden_size + r) * inter_size + r)] = static_cast<uint8_t>(z2 + 1);
            }
          }
        }
        auto fc1_weights = PackLsbBits(fc1_codes, num_experts * fc1_rows, hidden_size, fc1_bits);
        auto fc2_weights = PackLsbBits(fc2_codes, num_experts * hidden_size, inter_size, fc2_bits);
        const std::vector<float> fc1_scales(
            static_cast<size_t>(num_experts * fc1_rows * (hidden_size / block_size)), 1.0f);
        const std::vector<float> fc2_scales(
            static_cast<size_t>(num_experts * hidden_size * (inter_size / block_size)), 1.0f);

        std::vector<float> input(static_cast<size_t>(num_rows * hidden_size), 0.0f);
        std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), 0.0f);
        for (int64_t r = 0; r < num_rows; ++r) {
          for (int64_t c = 0; c < inter_size; ++c) {
            const float val = static_cast<float>(static_cast<int>((r * 7 + c) % 9) - 4) * 0.25f;
            input[static_cast<size_t>(r * hidden_size + c)] = val;
            const float h = MLFloat16(val).ToFloat();
            expected[static_cast<size_t>(r * hidden_size + c)] = h / (1.0f + std::exp(-h)) * h;
          }
        }
        const std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts), 1.0f);

        OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
        tester.AddAttribute<int64_t>("k", 2);
        tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
        tester.AddAttribute<std::string>("activation_type", "swiglu");
        tester.AddAttribute<int64_t>("swiglu_fusion", 1);
        tester.AddAttribute<int64_t>("expert_weight_bits", fc1_bits);
        tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
        tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
        tester.AddAttribute<int64_t>("fc3_expert_weight_bits", fc1_bits);
        tester.AddAttribute<int64_t>("weights_prepacked", 0);
        tester.AddAttribute<int64_t>("block_size", block_size);
        tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
        tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
        tester.AddInput<uint8_t>("fc1_experts_weights",
                                 {num_experts, fc1_rows, (hidden_size * fc1_bits + 7) / 8}, fc1_weights, true);
        tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                                   ToFloat16(fc1_scales), true);
        tester.AddOptionalInputEdge<MLFloat16>();
        tester.AddInput<uint8_t>("fc2_experts_weights",
                                 {num_experts, hidden_size, (inter_size * fc2_bits + 7) / 8}, fc2_weights, true);
        tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                                   ToFloat16(fc2_scales), true);
        tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
        tester.SetOutputTolerance(0.02f);

        std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
        execution_providers.push_back(DefaultCudaExecutionProvider());
        SessionOptions session_options;
        ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
            kOrtSessionOptionsDisableCPUEPFallback, "1"));
        tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
      }
    }
  }
}

// Aligned shapes (hidden % 256 == 0, inter % 128 == 0) select the fast INT3 kernels when no zero points are
// supplied; explicit zero points equal to the symmetric default select the scalar kernels. Both must match an
// independent reference that rounds each dequantized weight to FP16 before accumulation.
TEST(MoETest, QMoETest_Int3AndMixedWidthCudaFastPathParity) {
  if (!HasCudaEnvironment(750)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.5 or newer is required.";
  }
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 256;
  constexpr int64_t inter_size = 128;
  constexpr int64_t fc1_rows = 2 * inter_size;

  uint32_t state = 12345u;
  auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
  };

  struct Projection {
    std::vector<uint8_t> codes;
    std::vector<float> scales;  // Already rounded to FP16.
    std::vector<uint8_t> zero_codes;
  };
  auto make_projection = [&](int64_t rows, int64_t cols, int64_t bits, int64_t block_size) {
    Projection projection;
    const int64_t blocks = cols / block_size;
    projection.codes.resize(static_cast<size_t>(rows * cols));
    for (auto& code : projection.codes) {
      code = static_cast<uint8_t>(next() & ((1u << bits) - 1u));
    }
    projection.scales.resize(static_cast<size_t>(rows * blocks));
    for (auto& scale : projection.scales) {
      scale = MLFloat16(0.01f * static_cast<float>(1 + next() % 7)).ToFloat();
    }
    projection.zero_codes.assign(static_cast<size_t>(rows * blocks), static_cast<uint8_t>(1u << (bits - 1)));
    return projection;
  };
  auto dequantize = [](const Projection& projection, int64_t row, int64_t cols, int64_t block_size,
                       int64_t bits, int64_t col) {
    const int64_t blocks = cols / block_size;
    const int zero = 1 << (bits - 1);
    const float scale = projection.scales[static_cast<size_t>(row * blocks + col / block_size)];
    const int code = projection.codes[static_cast<size_t>(row * cols + col)];
    return MLFloat16(static_cast<float>(code - zero) * scale).ToFloat();
  };

  for (const auto& bits : {std::pair<int64_t, int64_t>{3, 3},
                           std::pair<int64_t, int64_t>{3, 4},
                           std::pair<int64_t, int64_t>{4, 3},
                           std::pair<int64_t, int64_t>{2, 3},
                           std::pair<int64_t, int64_t>{3, 2}}) {
    for (int64_t block_size : {32, 128}) {
      for (int64_t num_rows : {1, 3}) {
        for (bool with_zero_points : {false, true}) {
          const int64_t fc1_bits = bits.first;
          const int64_t fc2_bits = bits.second;
          const auto fc1 = make_projection(num_experts * fc1_rows, hidden_size, fc1_bits, block_size);
          const auto fc2 = make_projection(num_experts * hidden_size, inter_size, fc2_bits, block_size);

          std::vector<float> input(static_cast<size_t>(num_rows * hidden_size));
          for (auto& value : input) {
            value = MLFloat16(static_cast<float>(static_cast<int>(next() % 17) - 8) * 0.125f).ToFloat();
          }

          // Equal router logits select both experts with a normalized weight of 0.5 each.
          std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), 0.0f);
          for (int64_t r = 0; r < num_rows; ++r) {
            std::vector<float> accumulated(static_cast<size_t>(hidden_size), 0.0f);
            for (int64_t e = 0; e < num_experts; ++e) {
              std::vector<float> activation(static_cast<size_t>(inter_size));
              for (int64_t j = 0; j < inter_size; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int64_t c = 0; c < hidden_size; ++c) {
                  const float x = input[static_cast<size_t>(r * hidden_size + c)];
                  gate += x * dequantize(fc1, e * fc1_rows + 2 * j, hidden_size, block_size, fc1_bits, c);
                  up += x * dequantize(fc1, e * fc1_rows + 2 * j + 1, hidden_size, block_size, fc1_bits, c);
                }
                activation[static_cast<size_t>(j)] = MLFloat16(gate / (1.0f + std::exp(-gate)) * up).ToFloat();
              }
              for (int64_t c = 0; c < hidden_size; ++c) {
                float value = 0.0f;
                for (int64_t j = 0; j < inter_size; ++j) {
                  value += activation[static_cast<size_t>(j)] *
                           dequantize(fc2, e * hidden_size + c, inter_size, block_size, fc2_bits, j);
                }
                accumulated[static_cast<size_t>(c)] += MLFloat16(value).ToFloat() * 0.5f;
              }
            }
            for (int64_t c = 0; c < hidden_size; ++c) {
              expected[static_cast<size_t>(r * hidden_size + c)] = accumulated[static_cast<size_t>(c)];
            }
          }

          const std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts), 1.0f);
          OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
          tester.AddAttribute<int64_t>("k", 2);
          tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
          tester.AddAttribute<std::string>("activation_type", "swiglu");
          tester.AddAttribute<int64_t>("swiglu_fusion", 1);
          tester.AddAttribute<int64_t>("expert_weight_bits", fc1_bits);
          tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
          tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
          tester.AddAttribute<int64_t>("fc3_expert_weight_bits", fc1_bits);
          tester.AddAttribute<int64_t>("weights_prepacked", 0);
          tester.AddAttribute<int64_t>("block_size", block_size);
          tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
          tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
          tester.AddInput<uint8_t>("fc1_experts_weights",
                                   {num_experts, fc1_rows, (hidden_size * fc1_bits + 7) / 8},
                                   PackLsbBits(fc1.codes, num_experts * fc1_rows, hidden_size, fc1_bits), true);
          tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                                     ToFloat16(fc1.scales), true);
          tester.AddOptionalInputEdge<MLFloat16>();
          tester.AddInput<uint8_t>("fc2_experts_weights",
                                   {num_experts, hidden_size, (inter_size * fc2_bits + 7) / 8},
                                   PackLsbBits(fc2.codes, num_experts * hidden_size, inter_size, fc2_bits), true);
          tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                                     ToFloat16(fc2.scales), true);
          if (with_zero_points) {
            tester.AddOptionalInputEdge<MLFloat16>();
            tester.AddOptionalInputEdge<uint8_t>();
            tester.AddOptionalInputEdge<MLFloat16>();
            tester.AddOptionalInputEdge<MLFloat16>();
            const int64_t fc1_blocks = hidden_size / block_size;
            const int64_t fc2_blocks = inter_size / block_size;
            tester.AddInput<uint8_t>("fc1_zero_points",
                                     {num_experts, fc1_rows, (fc1_blocks * fc1_bits + 7) / 8},
                                     PackLsbBits(fc1.zero_codes, num_experts * fc1_rows, fc1_blocks, fc1_bits), true);
            tester.AddInput<uint8_t>("fc2_zero_points",
                                     {num_experts, hidden_size, (fc2_blocks * fc2_bits + 7) / 8},
                                     PackLsbBits(fc2.zero_codes, num_experts * hidden_size, fc2_blocks, fc2_bits), true);
          }
          tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
          tester.SetOutputTolerance(0.05f, 0.02f);

          std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
          execution_providers.push_back(DefaultCudaExecutionProvider());
          SessionOptions session_options;
          ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
              kOrtSessionOptionsDisableCPUEPFallback, "1"));
          SCOPED_TRACE(MakeString("fc1_bits=", fc1_bits, " fc2_bits=", fc2_bits, " block_size=", block_size,
                                  " rows=", num_rows, " zero_points=", with_zero_points));
          tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
        }
      }
    }
  }
}

TEST(MoETest, QMoETest_Int3CudaRejectsInvalidZeroPoints) {
  if (!HasCudaEnvironment(750)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.5 or newer is required.";
  }
  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 64;
  constexpr int64_t block_size = 32;
  constexpr int64_t bits = 3;
  constexpr int64_t fc1_rows = 2 * inter_size;

  // The INT3 profile only accepts zero-point code 4. Both constant and runtime-fed zero points are checked.
  for (bool zero_points_are_initializers : {true, false}) {
    const std::vector<uint8_t> fc1_codes(static_cast<size_t>(num_experts * fc1_rows * hidden_size), 4);
    const std::vector<uint8_t> fc2_codes(static_cast<size_t>(num_experts * hidden_size * inter_size), 4);
    const std::vector<uint8_t> fc1_zero_codes(
        static_cast<size_t>(num_experts * fc1_rows * (hidden_size / block_size)), 5);
    const std::vector<uint8_t> fc2_zero_codes(
        static_cast<size_t>(num_experts * hidden_size * (inter_size / block_size)), 4);

    OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", 2);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("swiglu_fusion", 1);
    tester.AddAttribute<int64_t>("expert_weight_bits", bits);
    tester.AddAttribute<int64_t>("weights_prepacked", 0);
    tester.AddAttribute<int64_t>("block_size", block_size);
    tester.AddInput<MLFloat16>("input", {num_rows, hidden_size},
                               ToFloat16(std::vector<float>(num_rows * hidden_size, 1.0f)));
    tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16({1.0f, 1.0f}));
    tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, fc1_rows, hidden_size * bits / 8},
                             PackLsbBits(fc1_codes, num_experts * fc1_rows, hidden_size, bits), true);
    tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                               ToFloat16(std::vector<float>(num_experts * fc1_rows * hidden_size / block_size, 1.0f)),
                               true);
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size * bits / 8},
                             PackLsbBits(fc2_codes, num_experts * hidden_size, inter_size, bits), true);
    tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                               ToFloat16(std::vector<float>(num_experts * hidden_size * inter_size / block_size, 1.0f)),
                               true);
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc1_zero_points",
                             {num_experts, fc1_rows, (hidden_size / block_size * bits + 7) / 8},
                             PackLsbBits(fc1_zero_codes, num_experts * fc1_rows, hidden_size / block_size, bits),
                             zero_points_are_initializers);
    tester.AddInput<uint8_t>("fc2_zero_points",
                             {num_experts, hidden_size, (inter_size / block_size * bits + 7) / 8},
                             PackLsbBits(fc2_zero_codes, num_experts * hidden_size, inter_size / block_size, bits),
                             zero_points_are_initializers);
    tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size},
                                ToFloat16(std::vector<float>(num_rows * hidden_size, 0.0f)));

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    SCOPED_TRACE(MakeString("zero_points_are_initializers=", zero_points_are_initializers));
    tester.Run(session_options, OpTester::ExpectResult::kExpectFailure,
               "INT3 zero points equal to 4", {}, nullptr, &execution_providers);
  }
}

TEST(MoETest, QMoETest_MixedWidthCudaAsymmetricZeroPoints) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaIdentityTest(
      2, 4, /*max_scratch_bytes=*/0, /*fused_swiglu=*/false, /*with_zero_points=*/true,
      /*use_bf16=*/false, /*block_size=*/32, /*hidden_size=*/128, /*inter_size=*/128);
}

static void RunQMoEMixedWidthCudaInvalidFallbackInputTest(
    int64_t hidden_size, int64_t inter_size, int64_t block_size,
    bool row_wise_scales, bool legacy_weights, const char* expected_error,
    bool use_float8_scales = false) {
  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 1;
  constexpr int64_t fc1_bits = 2;
  constexpr int64_t fc2_bits = 4;
  const int64_t fc1_pack = 8 / fc1_bits;
  const int64_t fc2_pack = 8 / fc2_bits;
  const std::vector<int64_t> fc1_weight_shape =
      legacy_weights ? std::vector<int64_t>{num_experts, hidden_size, inter_size / fc1_pack}
                     : std::vector<int64_t>{num_experts, inter_size, hidden_size / fc1_pack};
  const std::vector<int64_t> fc2_weight_shape =
      legacy_weights ? std::vector<int64_t>{num_experts, inter_size, hidden_size / fc2_pack}
                     : std::vector<int64_t>{num_experts, hidden_size, inter_size / fc2_pack};
  const int64_t fc1_blocks = (hidden_size + block_size - 1) / block_size;
  const int64_t fc2_blocks = (inter_size + block_size - 1) / block_size;
  const std::vector<int64_t> fc1_scale_shape =
      row_wise_scales ? std::vector<int64_t>{num_experts, inter_size}
                      : std::vector<int64_t>{num_experts, inter_size, fc1_blocks};
  const std::vector<int64_t> fc2_scale_shape =
      row_wise_scales ? std::vector<int64_t>{num_experts, hidden_size}
                      : std::vector<int64_t>{num_experts, hidden_size, fc2_blocks};

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddAttribute<int64_t>("block_size", block_size);
  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size},
                             std::vector<MLFloat16>(static_cast<size_t>(num_rows * hidden_size)));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, {MLFloat16(1.0f)});
  tester.AddInput<uint8_t>("fc1_experts_weights", fc1_weight_shape,
                           std::vector<uint8_t>(static_cast<size_t>(fc1_weight_shape[1] * fc1_weight_shape[2])));
  const size_t fc1_scale_count = static_cast<size_t>(inter_size * (row_wise_scales ? 1 : fc1_blocks));
  if (use_float8_scales) {
    tester.AddInput<Float8E4M3FN>("fc1_scales", fc1_scale_shape,
                                  std::vector<Float8E4M3FN>(fc1_scale_count, Float8E4M3FN(1.0f)));
  } else {
    tester.AddInput<MLFloat16>("fc1_scales", fc1_scale_shape,
                               std::vector<MLFloat16>(fc1_scale_count, MLFloat16(1.0f)));
  }
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", fc2_weight_shape,
                           std::vector<uint8_t>(static_cast<size_t>(fc2_weight_shape[1] * fc2_weight_shape[2])));
  const size_t fc2_scale_count = static_cast<size_t>(hidden_size * (row_wise_scales ? 1 : fc2_blocks));
  if (use_float8_scales) {
    tester.AddInput<Float8E4M3FN>("fc2_scales", fc2_scale_shape,
                                  std::vector<Float8E4M3FN>(fc2_scale_count, Float8E4M3FN(1.0f)));
  } else {
    tester.AddInput<MLFloat16>("fc2_scales", fc2_scale_shape,
                               std::vector<MLFloat16>(fc2_scale_count, MLFloat16(1.0f)));
  }
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size},
                              std::vector<MLFloat16>(static_cast<size_t>(num_rows * hidden_size)));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  tester.Run(session_options, OpTester::ExpectResult::kExpectFailure, expected_error,
             {}, nullptr, &execution_providers);
}

TEST(MoETest, QMoETest_MixedWidthCudaRejectsUnsafeFallbackInputs) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      64, 64, 32, true, false, "requires block-wise fc1_scales shape");
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      64, 128, 32, false, true, "requires canonical fc1_experts_weights shape");
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      96, 96, 24, false, false, "requires block_size to be a power of two");
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      80, 64, 32, false, false, "requires hidden_size to be divisible by block_size");
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      64, 80, 32, false, false, "requires inter_size to be divisible by block_size");
  RunQMoEMixedWidthCudaInvalidFallbackInputTest(
      64, 64, 32, false, false, "fc1_scales must match the activation type", true);
}
#endif

static void RunQMoEMixedWidthZeroPointTest(bool block_wise, bool invalid_fc2_shape) {
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 48;
  constexpr int64_t inter_size = 48;
  constexpr int64_t fc1_bits = 2;
  constexpr int64_t fc2_bits = 4;
  constexpr int64_t block_size = 16;
  constexpr int64_t blocks_per_row = hidden_size / block_size;

  const std::vector<int64_t> fc1_scales_dims =
      block_wise ? std::vector<int64_t>{num_experts, inter_size, blocks_per_row}
                 : std::vector<int64_t>{num_experts, inter_size};
  const std::vector<int64_t> fc2_scales_dims =
      block_wise ? std::vector<int64_t>{num_experts, hidden_size, blocks_per_row}
                 : std::vector<int64_t>{num_experts, hidden_size};
  const std::vector<int64_t> fc1_zero_points_dims =
      block_wise ? std::vector<int64_t>{num_experts, inter_size, 1}
                 : std::vector<int64_t>{num_experts, 12};
  std::vector<int64_t> fc2_zero_points_dims =
      block_wise ? std::vector<int64_t>{num_experts, hidden_size, 2}
                 : std::vector<int64_t>{num_experts, 24};
  if (invalid_fc2_shape) {
    --fc2_zero_points_dims.back();
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
  if (block_wise) {
    tester.AddAttribute<int64_t>("block_size", block_size);
  }
  tester.AddInput<MLFloat16>("input", {1, hidden_size}, std::vector<MLFloat16>(hidden_size));
  tester.AddInput<MLFloat16>("router_probs", {1, num_experts}, std::vector<MLFloat16>(num_experts));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, inter_size, 12},
                           std::vector<uint8_t>(num_experts * inter_size * 12));
  tester.AddInput<float>("fc1_scales", fc1_scales_dims,
                         std::vector<float>(static_cast<size_t>(TensorShape(fc1_scales_dims).Size()), 1.0f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, 24},
                           std::vector<uint8_t>(num_experts * hidden_size * 24));
  tester.AddInput<float>("fc2_scales", fc2_scales_dims,
                         std::vector<float>(static_cast<size_t>(TensorShape(fc2_scales_dims).Size()), 1.0f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc1_zero_points", fc1_zero_points_dims,
                           std::vector<uint8_t>(static_cast<size_t>(TensorShape(fc1_zero_points_dims).Size())));
  tester.AddInput<uint8_t>("fc2_zero_points", fc2_zero_points_dims,
                           std::vector<uint8_t>(static_cast<size_t>(TensorShape(fc2_zero_points_dims).Size())));
  tester.AddOutput<MLFloat16>("output", {1, hidden_size}, std::vector<MLFloat16>(hidden_size));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             invalid_fc2_shape ? "Input 'fc2_zero_points' is expected to have shape"
                               : "Mixed-width QMoE execution is not yet implemented on CPU.",
             {}, nullptr, &execution_providers);
}

TEST(MoETest, QMoETest_MixedWidthRowWiseZeroPoints) {
  RunQMoEMixedWidthZeroPointTest(false, false);
  RunQMoEMixedWidthZeroPointTest(false, true);
}

TEST(MoETest, QMoETest_MixedWidthBlockWiseZeroPoints) {
  RunQMoEMixedWidthZeroPointTest(true, false);
  RunQMoEMixedWidthZeroPointTest(true, true);
}

TEST(MoETest, QMoETest_MixedWidthFusedSwiGLURequiresMatchingFC1AndFC3) {
  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddAttribute<int64_t>("fc1_expert_weight_bits", 2);
  tester.AddAttribute<int64_t>("fc3_expert_weight_bits", 4);
  tester.AddInput<MLFloat16>("input", {1, 8}, std::vector<MLFloat16>(8));
  tester.AddInput<MLFloat16>("router_probs", {1, 1}, std::vector<MLFloat16>(1));
  tester.AddInput<uint8_t>("fc1_experts_weights", {1, 16, 2}, std::vector<uint8_t>(32));
  tester.AddInput<float>("fc1_scales", {1, 16}, std::vector<float>(16, 1.0f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {1, 8, 4}, std::vector<uint8_t>(32));
  tester.AddInput<float>("fc2_scales", {1, 8}, std::vector<float>(8, 1.0f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, 8}, std::vector<MLFloat16>(8));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "Fused SwiGLU requires FC1 and FC3 expert weight bits to match.",
             {}, nullptr, &execution_providers);
}

TEST(MoETest, QMoETest_PackedByteCountSupportsArbitraryBitWidths) {
  EXPECT_EQ(contrib::moe_helper::PackedByteCount(8, 3), 3);
  EXPECT_EQ(contrib::moe_helper::PackedByteCount(8, 5), 5);
  EXPECT_EQ(contrib::moe_helper::PackedByteCount(4, 6), 3);
  EXPECT_EQ(contrib::moe_helper::PackedByteCountWithPadding(1, 3), 1);
  EXPECT_EQ(contrib::moe_helper::PackedByteCountWithPadding(3, 5), 2);
}

// Regression test: row-wise asymmetric 2-bit with dimensions that trigger
// non-4-aligned parallel dequant block size. Without the alignment fix in
// GetDequantBlockSize, zero-point lane indexing is incorrect for shards
// starting at non-4-aligned rows, producing wrong output.
TEST(MoETest, QMoETest_CPU_Int2_RowWiseAsymmetricParallelDequant) {
#ifdef USE_MLAS
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  // Dimensions chosen so GetDequantBlockSize(fc1_out_features=200, num_expert_tokens=100)
  // returns 25 (not divisible by zp_pack_size=4) without the alignment fix.
  constexpr int64_t num_rows = 100;
  constexpr int64_t num_experts = 1;
  constexpr int64_t hidden_size = 32;
  constexpr int64_t inter_size = 100;
  constexpr int64_t expert_weight_bits = 2;
  constexpr int64_t pack_size = 8 / expert_weight_bits;
  constexpr int64_t fc1_out_features = 2 * inter_size;  // swiglu

  // Construct ZP with varying lanes: byte 0xE4 = lanes [0, 1, 2, 3] (LSB first).
  // Then construct weights where each row's 2-bit values equal that row's ZP,
  // so dequantized = scale * (value - zp) = 0 for all elements.
  // If wrong ZP lane is read due to alignment bug, output != 0.
  auto make_zp_byte = [](uint8_t lane0, uint8_t lane1, uint8_t lane2, uint8_t lane3) -> uint8_t {
    return static_cast<uint8_t>((lane0 & 0x3) | ((lane1 & 0x3) << 2) | ((lane2 & 0x3) << 4) | ((lane3 & 0x3) << 6));
  };
  // ZP pattern: lanes [0, 1, 2, 3] repeating → byte 0xE4
  const uint8_t zp_byte = make_zp_byte(0, 1, 2, 3);

  // FC1 ZP: fc1_out_features / 4 = 50 bytes per expert
  const int64_t fc1_zp_size = fc1_out_features / pack_size;
  std::vector<uint8_t> fc1_zp(static_cast<size_t>(num_experts * fc1_zp_size), zp_byte);

  // FC2 ZP: hidden_size / 4 = 8 bytes per expert
  const int64_t fc2_zp_size = hidden_size / pack_size;
  std::vector<uint8_t> fc2_zp(static_cast<size_t>(num_experts * fc2_zp_size), zp_byte);

  // Build weight bytes: for each row, all 2-bit values = that row's ZP
  auto build_weights = [&](int64_t rows, int64_t cols) -> std::vector<uint8_t> {
    const int64_t packed_cols = cols / pack_size;
    std::vector<uint8_t> weights(static_cast<size_t>(num_experts * rows * packed_cols));
    for (int64_t e = 0; e < num_experts; ++e) {
      for (int64_t r = 0; r < rows; ++r) {
        // Get this row's ZP value from the packed ZP byte
        const uint8_t row_zp_byte = zp_byte;  // same pattern for all groups of 4
        const int lane = static_cast<int>(r % pack_size);
        const uint8_t row_zp = (row_zp_byte >> (lane * 2)) & 0x3;
        // Pack all columns with this ZP value
        const uint8_t weight_byte = make_zp_byte(row_zp, row_zp, row_zp, row_zp);
        const size_t row_offset = static_cast<size_t>((e * rows + r) * packed_cols);
        std::fill(weights.begin() + row_offset,
                  weights.begin() + row_offset + static_cast<size_t>(packed_cols),
                  weight_byte);
      }
    }
    return weights;
  };

  std::vector<uint8_t> fc1_weights = build_weights(fc1_out_features, hidden_size);
  std::vector<uint8_t> fc2_weights = build_weights(hidden_size, inter_size);

  // Scales = 1.0 (so dequantized = value - zp = 0 when correct)
  std::vector<float> fc1_scales(static_cast<size_t>(num_experts * fc1_out_features), 1.0f);
  std::vector<float> fc2_scales(static_cast<size_t>(num_experts * hidden_size), 1.0f);

  // Input: random-ish non-zero values
  std::vector<float> input(static_cast<size_t>(num_rows * hidden_size));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
  }

  // Router: all tokens → single expert
  std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts), 1.0f);

  // Expected: all weights dequantize to 0 → output is 0
  std::vector<float> expected_output(static_cast<size_t>(num_rows * hidden_size), 0.0f);

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");
  cpu_tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_weights_dims = {num_experts, fc1_out_features, hidden_size / pack_size};
  std::vector<int64_t> fc2_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, fc1_out_features};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> fc1_zp_dims = {num_experts, fc1_zp_size};
  std::vector<int64_t> fc2_zp_dims = {num_experts, fc2_zp_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_weights_dims, fc1_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_weights_dims, fc2_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  cpu_tester.AddOptionalInputEdge<float>();      // fc3_scales
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  cpu_tester.AddInput<uint8_t>("fc1_zero_points", fc1_zp_dims, fc1_zp);
  cpu_tester.AddInput<uint8_t>("fc2_zero_points", fc2_zp_dims, fc2_zp);
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  cpu_tester.SetOutputTolerance(0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_Int4_MLAS) {
#ifdef USE_MLAS
  // Skip this test if we're not testing CPU execution provider
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 32;
  int inter_size = 32;

  const std::vector<float> input = {
      -0.5f, 0.2f, 1.1f, -0.3f, 0.8f, -0.1f, 0.4f, -0.7f, 0.9f, -0.2f, 0.6f, 0.1f, -0.4f, 0.3f, -0.8f, 0.7f,
      0.2f, -0.5f, 0.1f, 0.9f, -0.3f, 0.6f, -0.1f, 0.4f, -0.7f, 0.8f, 0.3f, -0.2f, 0.5f, 0.1f, -0.6f, 0.9f,
      0.1f, 0.7f, -0.4f, 0.2f, 0.8f, -0.3f, 0.5f, -0.1f, 0.6f, 0.4f, -0.7f, 0.3f, 0.9f, -0.2f, 0.1f, 0.8f,
      -0.5f, 0.6f, 0.3f, -0.1f, 0.4f, 0.7f, -0.8f, 0.2f, 0.9f, 0.1f, -0.3f, 0.5f, 0.6f, -0.4f, 0.8f, 0.2f};

  const std::vector<float> router_probs = {0.3f, 0.7f, 0.6f, 0.4f};

  // Generate simple test weights for 4-bit symmetric quantization with SwiGLU
  // Use 0x88 which unpacks to 8,8 -> 0,0 in signed form (8-8=0) for zero weights
  // For SwiGLU: FC1 outputs 2*inter_size (gate + linear), FC2 takes inter_size input
  std::vector<uint8_t> fc1_experts_weights(num_experts * hidden_size * inter_size, 0x88);      // 2*inter_size for SwiGLU, packed into /2
  std::vector<uint8_t> fc2_experts_weights(num_experts * inter_size * hidden_size / 2, 0x88);  // 8,8 values to produce zero output
  std::vector<uint8_t> fc3_experts_weights;                                                    // Empty for CPU (FC3 not supported)

  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.01f);  // 2x for SwiGLU (gate + linear)
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.01f);     // Smaller scale factor
  std::vector<float> fc3_scales;

  // With zero weights (0x88 -> 8,8 -> 0,0 signed), the implementation will produce all zero outputs
  std::vector<float> output(num_rows * hidden_size, 0.0f);

  // Test CPU execution provider ONLY (don't use RunQMoETest which tests both CUDA and CPU)
  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 2);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");  // CPU only supports swiglu
  cpu_tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);  // Always use 1 - softmax normalization always applied
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 4);         // Test 4-bit quantization

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size / 2};  // SwiGLU: 2*inter_size output, 4-bit packed
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size / 2};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};  // 2x for SwiGLU
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);  // Use float for CPU
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);  // Use float for CPU
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc2_experts_bias
  cpu_tester.AddOptionalInputEdge<uint8_t>();                             // fc3_experts_weights (skip FC3 for CPU)
  cpu_tester.AddOptionalInputEdge<float>();                               // fc3_scales (use float for CPU)
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc3_experts_bias

  // When using 0x88 for 4-bit quantized weights with the current implementation,
  // all dequantized values should be 0.0f (8-8=0), and thus output should be all zeros
  std::vector<float> expected_output(num_rows * hidden_size, 0.0f);

  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  cpu_tester.SetOutputTolerance(0.05f);  // Small tolerance for numerical differences

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

// Block-wise 4/8-bit SwiGLU QMoE on CPU with non-trivial weights. Block-wise experts with constant
// scales (and, if present, constant zero points) take the MLAS QNBit GEMM (MatMulNBits kernel) path;
// the expected output is an fp32 reference computed here from the dequantized weights and the
// kernel's routing (softmax over the top-k router logits). Weights, scales, zero points and biases
// are constant initializers so that PrePack runs; as graph inputs the kernel would fall back to the
// dequantize+SGEMM path.
namespace {
struct QMoEBlockWiseCase {
  int num_rows;
  int num_experts;
  int hidden_size;
  int inter_size;
  int block_size;
  int top_k;
  bool with_bias;
  int bits{4};
  bool with_zero_points{false};
  int accuracy_level{0};
  // Expected number of pre-packed expert weights (0 or 2), or -1 to skip the check. The output
  // check alone cannot tell the QNBit path from the dequantize fallback, so 8-bit cases (where the
  // fallback never pre-packs) also assert this; the 4-bit fallback pre-packs too, so it cannot.
  int expected_prepacked{-1};
  // Intra-op threads, or 0 for the default pool. With at least as many active experts as threads
  // the kernel runs one expert per thread instead of batching them into one MLAS dispatch, so this
  // selects which of the two QNBit dispatch paths a case exercises.
  int intra_op_num_threads{0};
};

// `output_out`, when set, receives the run's output as fp32 so callers can compare the results of
// different kernel policies bitwise.
template <typename T>
void RunQMoECpuBlockWiseSwiGLU(const QMoEBlockWiseCase& c, float tolerance,
                               bool share_prepacked_weights_across_sessions = false,
                               std::vector<float>* output_out = nullptr) {
  const int num_rows = c.num_rows, num_experts = c.num_experts, hidden_size = c.hidden_size;
  const int inter_size = c.inter_size, block_size = c.block_size, top_k = c.top_k, bits = c.bits;
  const int fc1_rows = 2 * inter_size;  // interleaved [gate, up] rows
  const int fc1_blocks = hidden_size / block_size;
  const int fc2_blocks = inter_size / block_size;
  const int pack = 8 / bits;
  const float code_zero = (bits == 4) ? 8.0f : 128.0f;
  const int code_max = (1 << bits) - 1;
  constexpr bool is_fp16 = std::is_same_v<T, MLFloat16>;

  uint32_t state = 0x12345678u;
  auto next_uniform = [&state]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f;  // [0, 1]
  };
  auto next_code = [&state, bits]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<uint8_t>((state >> 12) & ((1u << bits) - 1));
  };
  // The kernel converts T inputs to fp32, so the reference must see the T-rounded values.
  auto round_to_t = [](float v) { return is_fp16 ? MLFloat16(v).ToFloat() : v; };

  std::vector<float> input(static_cast<size_t>(num_rows * hidden_size));
  for (auto& v : input) v = round_to_t(2.0f * next_uniform() - 1.0f);
  std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts));
  for (auto& v : router_probs) v = round_to_t(4.0f * next_uniform() - 2.0f);

  // Weights are stored as [E, N, K/pack] uint8 with the lower k in the low bits; scales as [E, N, K/block].
  std::vector<uint8_t> fc1_codes(static_cast<size_t>(num_experts * fc1_rows * hidden_size));
  std::vector<uint8_t> fc2_codes(static_cast<size_t>(num_experts * hidden_size * inter_size));
  for (auto& v : fc1_codes) v = next_code();
  for (auto& v : fc2_codes) v = next_code();
  auto pack_codes = [pack, bits](const std::vector<uint8_t>& codes) {
    std::vector<uint8_t> packed(codes.size() / pack);
    for (size_t i = 0; i < packed.size(); ++i) {
      uint8_t byte = 0;
      for (int j = 0; j < pack; ++j) byte |= static_cast<uint8_t>(codes[i * pack + j] << (j * bits));
      packed[i] = byte;
    }
    return packed;
  };
  const std::vector<uint8_t> fc1_weights = pack_codes(fc1_codes);
  const std::vector<uint8_t> fc2_weights = pack_codes(fc2_codes);
  // Zero points: one code per block, stored [E, N, ceil(blocks/pack)] with the lower block in the
  // low bits (the MatMulNBits layout); kept away from the code range ends so weights stay signed.
  std::vector<uint8_t> fc1_zp_codes, fc2_zp_codes, fc1_zero_points, fc2_zero_points;
  if (c.with_zero_points) {
    fc1_zp_codes.resize(static_cast<size_t>(num_experts * fc1_rows * fc1_blocks));
    fc2_zp_codes.resize(static_cast<size_t>(num_experts * hidden_size * fc2_blocks));
    for (auto& v : fc1_zp_codes) v = static_cast<uint8_t>(1 + static_cast<int>(next_uniform() * (code_max - 2)));
    for (auto& v : fc2_zp_codes) v = static_cast<uint8_t>(1 + static_cast<int>(next_uniform() * (code_max - 2)));
    auto pack_zero_points = [pack, bits](const std::vector<uint8_t>& codes, int rows_total, int blocks) {
      const int packed_blocks = (blocks + pack - 1) / pack;
      std::vector<uint8_t> packed(static_cast<size_t>(rows_total * packed_blocks), 0);
      for (int r = 0; r < rows_total; ++r) {
        for (int b = 0; b < blocks; ++b) {
          packed[static_cast<size_t>(r * packed_blocks + b / pack)] |=
              static_cast<uint8_t>(codes[static_cast<size_t>(r * blocks + b)] << ((b % pack) * bits));
        }
      }
      return packed;
    };
    fc1_zero_points = pack_zero_points(fc1_zp_codes, num_experts * fc1_rows, fc1_blocks);
    fc2_zero_points = pack_zero_points(fc2_zp_codes, num_experts * hidden_size, fc2_blocks);
  }
  std::vector<float> fc1_scales(static_cast<size_t>(num_experts * fc1_rows * fc1_blocks));
  std::vector<float> fc2_scales(static_cast<size_t>(num_experts * hidden_size * fc2_blocks));
  // Signed scales, as produced by MatMulNBits-style quantizers; scaled so the weight range is
  // comparable for 4 and 8 bit codes.
  const float scale_unit = 8.0f / code_zero;
  for (auto& v : fc1_scales) v = round_to_t((next_uniform() < 0.5f ? -1.0f : 1.0f) * scale_unit * (0.01f + 0.03f * next_uniform()));
  for (auto& v : fc2_scales) v = round_to_t((next_uniform() < 0.5f ? -1.0f : 1.0f) * scale_unit * (0.01f + 0.03f * next_uniform()));
  std::vector<float> fc1_bias, fc2_bias;
  if (c.with_bias) {
    fc1_bias.resize(static_cast<size_t>(num_experts * fc1_rows));
    fc2_bias.resize(static_cast<size_t>(num_experts * hidden_size));
    for (auto& v : fc1_bias) v = round_to_t(0.2f * next_uniform() - 0.1f);
    for (auto& v : fc2_bias) v = round_to_t(0.2f * next_uniform() - 0.1f);
  }

  auto dequant = [&](const std::vector<uint8_t>& codes, const std::vector<float>& scales,
                     const std::vector<uint8_t>& zp_codes, int e, int n, int k, int rows, int cols, int blocks) {
    const size_t block_index = static_cast<size_t>((e * rows + n) * blocks + k / block_size);
    const float scale = scales[block_index];
    const float zero = c.with_zero_points ? static_cast<float>(zp_codes[block_index]) : code_zero;
    return (static_cast<float>(codes[static_cast<size_t>((e * rows + n) * cols + k)]) - zero) * scale;
  };

  std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), 0.0f);
  for (int t = 0; t < num_rows; ++t) {
    const float* logits = router_probs.data() + t * num_experts;
    std::vector<std::pair<float, int>> sorted;
    for (int e = 0; e < num_experts; ++e) sorted.emplace_back(logits[e], e);
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum_exp = 0.0f;
    std::vector<float> weights(static_cast<size_t>(top_k));
    for (int j = 0; j < top_k; ++j) {
      weights[j] = std::exp(sorted[j].first - sorted[0].first);
      sum_exp += weights[j];
    }
    for (int j = 0; j < top_k; ++j) {
      const int e = sorted[j].second;
      const float w = weights[j] / sum_exp;
      const float* x = input.data() + t * hidden_size;
      std::vector<float> h(static_cast<size_t>(fc1_rows));
      for (int n = 0; n < fc1_rows; ++n) {
        float acc = c.with_bias ? fc1_bias[static_cast<size_t>(e * fc1_rows + n)] : 0.0f;
        for (int k = 0; k < hidden_size; ++k) acc += x[k] * dequant(fc1_codes, fc1_scales, fc1_zp_codes, e, n, k, fc1_rows, hidden_size, fc1_blocks);
        h[n] = acc;
      }
      std::vector<float> a(static_cast<size_t>(inter_size));
      for (int i = 0; i < inter_size; ++i) {
        const float gate = h[2 * i], up = h[2 * i + 1];
        a[i] = gate / (1.0f + std::exp(-gate)) * up;
      }
      for (int n = 0; n < hidden_size; ++n) {
        float acc = c.with_bias ? fc2_bias[static_cast<size_t>(e * hidden_size + n)] : 0.0f;
        for (int k = 0; k < inter_size; ++k) acc += a[k] * dequant(fc2_codes, fc2_scales, fc2_zp_codes, e, n, k, hidden_size, inter_size, fc2_blocks);
        expected[static_cast<size_t>(t * hidden_size + n)] += w * acc;
      }
    }
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<float>("activation_alpha", 1.0f);
  tester.AddAttribute<float>("activation_beta", 0.0f);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", bits);
  tester.AddAttribute<int64_t>("block_size", block_size);
  if (c.accuracy_level != 0) {
    tester.AddAttribute<int64_t>("accuracy_level", c.accuracy_level);
  }

  auto add_t_input = [&](const char* name, std::vector<int64_t> dims, const std::vector<float>& values, bool is_initializer) {
    if constexpr (is_fp16) {
      tester.AddInput<MLFloat16>(name, dims, ToFloat16(values), is_initializer);
    } else {
      tester.AddInput<float>(name, dims, values, is_initializer);
    }
  };
  add_t_input("input", {num_rows, hidden_size}, input, false);
  add_t_input("router_probs", {num_rows, num_experts}, router_probs, false);
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, fc1_rows, hidden_size / pack}, fc1_weights, true);
  add_t_input("fc1_scales", {num_experts, fc1_rows, fc1_blocks}, fc1_scales, true);
  if (c.with_bias) {
    add_t_input("fc1_experts_bias", {num_experts, fc1_rows}, fc1_bias, true);
  } else {
    tester.AddOptionalInputEdge<T>();
  }
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / pack}, fc2_weights, true);
  add_t_input("fc2_scales", {num_experts, hidden_size, fc2_blocks}, fc2_scales, true);
  if (c.with_bias) {
    add_t_input("fc2_experts_bias", {num_experts, hidden_size}, fc2_bias, true);
  } else {
    tester.AddOptionalInputEdge<T>();
  }
  if (c.with_zero_points) {
    tester.AddOptionalInputEdge<uint8_t>();  // fc3_experts_weights
    tester.AddOptionalInputEdge<T>();        // fc3_scales
    tester.AddOptionalInputEdge<T>();        // fc3_experts_bias
    tester.AddInput<uint8_t>("fc1_zero_points", {num_experts, fc1_rows, (fc1_blocks + pack - 1) / pack}, fc1_zero_points, true);
    tester.AddInput<uint8_t>("fc2_zero_points", {num_experts, hidden_size, (fc2_blocks + pack - 1) / pack}, fc2_zero_points, true);
  }
  if constexpr (is_fp16) {
    tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  } else {
    tester.AddOutput<float>("output", {num_rows, hidden_size}, expected);
  }
  tester.SetOutputTolerance(tolerance);

  auto cpu_ep = []() {
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());
    return eps;
  };

  if (!share_prepacked_weights_across_sessions) {
    auto eps = cpu_ep();
    SessionOptions so;
    if (c.intra_op_num_threads > 0) {
      so.intra_op_param.thread_pool_size = c.intra_op_num_threads;
      so.use_per_session_threads = true;
    }
    size_t prepacked = 0, shared = 0;
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps, {}, &prepacked, &shared);
    if (c.expected_prepacked >= 0) {
      ASSERT_EQ(prepacked, static_cast<size_t>(c.expected_prepacked));
    }
    if (output_out != nullptr) {
      const auto fetches = tester.GetFetches();
      ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
      const Tensor& out = fetches[0].Get<Tensor>();
      const T* data = out.Data<T>();
      output_out->resize(static_cast<size_t>(out.Shape().Size()));
      for (size_t i = 0; i < output_out->size(); ++i) {
        if constexpr (is_fp16) {
          (*output_out)[i] = data[i].ToFloat();
        } else {
          (*output_out)[i] = data[i];
        }
      }
    }
    return;
  }

  // Two sessions sharing the fc1/fc2 initializers: the second must adopt the first session's packed
  // buffers through UseSharedPrePackedBuffers (the tagged QNBit layout) and still match the reference.
  OrtValue fc1_value, fc2_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({num_experts, fc1_rows, hidden_size / pack}),
                       const_cast<uint8_t*>(fc1_weights.data()), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), fc1_value);
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({num_experts, hidden_size, inter_size / pack}),
                       const_cast<uint8_t*>(fc2_weights.data()), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), fc2_value);
  SessionOptions so;
  ASSERT_STATUS_OK(so.AddInitializer("fc1_experts_weights", &fc1_value));
  ASSERT_STATUS_OK(so.AddInitializer("fc2_experts_weights", &fc2_value));
  tester.EnableSharingOfPrePackedWeightsAcrossSessions();

  size_t prepacked_session_1 = 0, prepacked_session_2 = 0, shared = 0;
  {
    auto eps = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps, {}, &prepacked_session_1, &shared);
    ASSERT_EQ(shared, static_cast<size_t>(0));
  }
  ASSERT_EQ(prepacked_session_1, tester.GetNumPrePackedWeightsShared());
  // Both expert weights pre-pack on every platform (the legacy 4-bit path pre-packs too), so a
  // count of 2 proves the sharing round trip below exercises real buffers.
  ASSERT_EQ(prepacked_session_1, static_cast<size_t>(2));
  {
    auto eps = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps, {}, &prepacked_session_2, &shared);
    ASSERT_EQ(prepacked_session_1, prepacked_session_2);
    ASSERT_EQ(shared, prepacked_session_2);
  }
}
}  // namespace

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Decode) {
  // Single token: the QNBit path runs the selected experts sequentially, each GEMM threaded internally.
  RunQMoECpuBlockWiseSwiGLU<float>({1, 8, 64, 64, 32, 2, false}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Prefill) {
  RunQMoECpuBlockWiseSwiGLU<float>({37, 4, 128, 96, 32, 2, false}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_SingleBlockZeroPoints) {
  RunQMoECpuBlockWiseSwiGLU<float>(
      {5, 4, 32, 32, 32, 2, false, 4, true, 0, 2}, 0.01f);
}

// The two cases below pin the per-expert dispatch path: capping the pool at two threads leaves at
// least as many active experts as threads, so the experts run one per thread instead of being
// batched. Same shapes and reference as the grouped cases above, so the paths must agree.
TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Decode_PerExpertDispatch) {
  RunQMoECpuBlockWiseSwiGLU<float>({1, 8, 64, 64, 32, 2, false, 4, false, 0, -1, /*intra_op*/ 2}, 0.01f);
}

// Uneven token counts per expert: batched into one dispatch per distinct count when grouped.
TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Prefill_PerExpertDispatch) {
  RunQMoECpuBlockWiseSwiGLU<float>({37, 4, 128, 96, 32, 2, false, 4, false, 0, -1, /*intra_op*/ 2}, 0.01f);
}

// A single-threaded session still batches, so the grouped path must also hold with a null pool.
TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Decode_SingleThread) {
  RunQMoECpuBlockWiseSwiGLU<float>({1, 8, 64, 64, 32, 2, false, 4, false, 0, -1, /*intra_op*/ 1}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Bias_Block64) {
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 64, 3, true}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Fp16) {
  RunQMoECpuBlockWiseSwiGLU<MLFloat16>({5, 4, 128, 64, 32, 2, true}, 0.02f);
}

// 8-bit experts pre-pack only on the QNBit path, so the pre-pack count proves which path ran.
static int ExpectedInt8QNBitPrepacks(int block_size) {
  return MlasIsQNBitGemmAvailable(8, static_cast<size_t>(block_size), SQNBIT_CompInt8) ? 2 : 0;
}

TEST(MoETest, QMoETest_CPU_Int8_BlockWise_SwiGLU) {
  // MLAS has no fp32-activation QNBit kernel for 8 bits, so this runs with int8 activations by
  // default and carries their quantization error; the tolerance matches the other int8 cases.
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 8, false, 0, ExpectedInt8QNBitPrepacks(32)}, 0.05f);
}

TEST(MoETest, QMoETest_CPU_Int8_BlockWise_SwiGLU_ForcedFp32FallsBackToDequantize) {
  // Forcing fp32 activations keeps 8-bit experts on the dequantize path, which is exact enough
  // for the tight tolerance and never pre-packs.
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "fp32"}}};
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 8, false, 0, 0}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int8_BlockWise_SwiGLU_Int8Activations) {
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "int8"}}};
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 8, false, 0, ExpectedInt8QNBitPrepacks(32)}, 0.05f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_Int8Activations) {
  // accuracy_level=4 opts into int8 activations, as for MatMulNBits.
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 4, false, 4}, 0.05f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_AccuracyLevel4SelectsInt8) {
  // accuracy_level=4 must pick the int8 kernels even where the fp32 ones are also available. The
  // output check alone cannot tell the two apart (both meet the int8 tolerance), so compare raw
  // outputs: level 4 must be bitwise identical to the forced-int8 run (same kernels, each output
  // column owned by one thread) and differ from the forced-fp32 run (int8 activation quantization
  // perturbs every output).
  if (!MlasIsQNBitGemmAvailable(4, 32, SQNBIT_CompInt8) || !MlasIsQNBitGemmAvailable(4, 32, SQNBIT_CompFp32)) {
    GTEST_SKIP() << "platform lacks one of the 4-bit QNBit compute types";
  }
  const QMoEBlockWiseCase base{5, 4, 128, 64, 32, 2, true};
  std::vector<float> level4, forced_fp32, forced_int8;
  {
    QMoEBlockWiseCase c = base;
    c.accuracy_level = 4;
    RunQMoECpuBlockWiseSwiGLU<float>(c, 0.05f, false, &level4);
  }
  {
    ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "fp32"}}};
    RunQMoECpuBlockWiseSwiGLU<float>(base, 0.01f, false, &forced_fp32);
  }
  {
    ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "int8"}}};
    RunQMoECpuBlockWiseSwiGLU<float>(base, 0.05f, false, &forced_int8);
  }
  ASSERT_FALSE(level4.empty());
  EXPECT_EQ(level4, forced_int8);
  EXPECT_NE(level4, forced_fp32);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_PolicyMatrix) {
  // Cells of the documented policy that the tolerance checks cannot tell apart: runs that must
  // select the same kernels produce bitwise-identical outputs. accuracy_level values other than 4
  // behave as 0, and the fp32 override is the 4-bit default.
  if (!MlasIsQNBitGemmAvailable(4, 32, SQNBIT_CompFp32)) {
    GTEST_SKIP() << "platform lacks the 4-bit fp32 QNBit kernel";
  }
  const QMoEBlockWiseCase base{5, 4, 128, 64, 32, 2, true};
  std::vector<float> by_default, level1, forced_fp32;
  RunQMoECpuBlockWiseSwiGLU<float>(base, 0.01f, false, &by_default);
  {
    QMoEBlockWiseCase c = base;
    c.accuracy_level = 1;
    RunQMoECpuBlockWiseSwiGLU<float>(c, 0.01f, false, &level1);
  }
  {
    ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "fp32"}}};
    RunQMoECpuBlockWiseSwiGLU<float>(base, 0.01f, false, &forced_fp32);
  }
  ASSERT_FALSE(by_default.empty());
  EXPECT_EQ(level1, by_default);
  EXPECT_EQ(forced_fp32, by_default);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_DisabledByEnv) {
  // "0" keeps the previous paths; the output still matches the exact reference.
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "0"}}};
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int8_BlockWise_SwiGLU_DisabledByEnv) {
  // "0" keeps the previous paths, which never pre-pack 8-bit experts.
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "0"}}};
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 8, false, 0, 0}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_ZeroPoints) {
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 4, true}, 0.01f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_ZeroPoints_Int8Activations) {
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 4, true, 4}, 0.05f);
}

TEST(MoETest, QMoETest_CPU_Int8_BlockWise_SwiGLU_ZeroPoints_Int8Activations) {
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_QMOE_CPU_QNBIT_GEMM", "int8"}}};
  RunQMoECpuBlockWiseSwiGLU<float>({5, 4, 128, 64, 32, 2, true, 8, true}, 0.05f);
}

TEST(MoETest, QMoETest_CPU_Int4_BlockWise_SwiGLU_SharedPrePackedWeights) {
  RunQMoECpuBlockWiseSwiGLU<float>({3, 4, 64, 64, 32, 2, true}, 0.01f, /*share_prepacked_weights_across_sessions*/ true);
}

TEST(MoETest, QMoETest_CPU_Int8_MLAS) {
#ifdef USE_MLAS
  // Skip this test if we're not testing CPU execution provider
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  // Test CPU implementation with 8-bit quantization - CPU ONLY
  int num_rows = 1;
  int num_experts = 2;
  int hidden_size = 16;
  int inter_size = 16;

  const std::vector<float> input = {
      0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f};

  const std::vector<float> router_probs = {0.4f, 0.6f};

  // For 8-bit symmetric quantization with SwiGLU
  // Use quantized weights at zero point for zero outputs (128 = 0 in signed)
  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * hidden_size, 128);  // 2*inter_size for SwiGLU, no packing for 8-bit
  std::vector<uint8_t> fc2_experts_weights(num_experts * inter_size * hidden_size, 128);      // 128 = 0 in signed
  std::vector<uint8_t> fc3_experts_weights;                                                   // Empty for CPU

  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.1f);  // 2x for SwiGLU
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.1f);
  std::vector<float> fc3_scales;

  // Expected output should be zero since we're using zero weights (128-128=0)
  std::vector<float> output(num_rows * hidden_size, 0.0f);

  // Test with different attributes for 8-bit
  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");  // CPU only supports swiglu
  cpu_tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);  // Always use 1 - softmax normalization always applied
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 8);         // Test 8-bit quantization

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size * 2, hidden_size};  // SwiGLU: 2*inter_size output, 8-bit no packing
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};  // 2x for SwiGLU
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);  // Use float, not MLFloat16
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);  // Use float, not MLFloat16
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc2_experts_bias
  cpu_tester.AddOptionalInputEdge<uint8_t>();                             // fc3_experts_weights (skip FC3 for CPU)
  cpu_tester.AddOptionalInputEdge<float>();                               // fc3_scales (use float, not MLFloat16)
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc3_experts_bias
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output));
  cpu_tester.SetOutputTolerance(0.05f);  // Small tolerance since we expect near-zero output

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_FC3_Error) {
#ifdef USE_MLAS
  // Skip this test if we're not testing CPU execution provider
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  // Test that CPU throws error when FC3 gating is provided - CPU ONLY
  int num_rows = 1;
  int num_experts = 2;
  int hidden_size = 8;
  int inter_size = 8;

  const std::vector<float> input = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
  const std::vector<float> router_probs = {0.5f, 0.5f};

  // Using new layout: fc1 has fused swiglu doubling (2*inter_size) and 4-bit pack_size=2 so hidden_size packed dimension is hidden_size/2
  const int pack_size = 2;                    // for 4-bit
  const int fc1_inter_size = 2 * inter_size;  // swiglu fused
  std::vector<uint8_t> fc1_experts_weights(num_experts * fc1_inter_size * (hidden_size / pack_size), 0x01);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * (inter_size / pack_size), 0x10);
  std::vector<uint8_t> fc3_experts_weights(num_experts * inter_size * (hidden_size / pack_size), 0x21);  // FC3 provided

  std::vector<float> fc1_scales(num_experts * fc1_inter_size, 0.1f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.05f);
  std::vector<float> fc3_scales(num_experts * inter_size, 0.08f);  // FC3 scales provided

  // Test CPU execution provider ONLY (designed to test CPU-specific error handling)
  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 1);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");  // CPU only supports swiglu
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);   // Use 1 for consistency, though this test focuses on FC3 error
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 4);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, fc1_inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc3_experts_weights_dims = {num_experts, inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, fc1_inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> fc3_scales_dims = {num_experts, inter_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);  // Use float for CPU
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                           // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);                               // Use float for CPU
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                                                        // fc2_experts_bias
  cpu_tester.AddInput<uint8_t>("fc3_experts_weights", fc3_experts_weights_dims, fc3_experts_weights);  // FC3 provided!
  cpu_tester.AddInput<float>("fc3_scales", fc3_scales_dims, fc3_scales);                               // Use float for CPU
  cpu_tester.AddOptionalInputEdge<MLFloat16>();                                                        // fc3_experts_bias

  std::vector<float> dummy_output(num_rows * hidden_size, 0.0f);
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(dummy_output));

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());

  // Expect this to fail with FC3 not implemented error
  cpu_tester.Run(OpTester::ExpectResult::kExpectFailure, "FC3 gating is not yet implemented", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_SwiGLU_Int4) {
#ifdef USE_MLAS
  // Skip this test if we're not testing CPU execution provider
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  // Test CPU implementation with 4-bit quantization and SwiGLU activation
  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 16;
  int inter_size = 16;

  const std::vector<float> input = {
      0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f,
      0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, -0.9f, 1.0f, -1.1f, 1.2f, -1.3f, 1.4f, -1.5f, 1.6f, -1.7f};

  const std::vector<float> router_probs = {0.6f, 0.4f, 0.3f, 0.7f};

  // For SwiGLU, FC1 weights need to be 2x inter_size (concatenated linear + gate weights)
  // 4-bit: each uint8 stores 2 weights, so we need (hidden_size * inter_size * 2) / 2 uint8s per expert
  const int fc1_weight_size_per_expert = hidden_size * inter_size * 2 / 2;  // For 4-bit SwiGLU
  const int fc2_weight_size_per_expert = inter_size * hidden_size / 2;      // For 4-bit FC2

  // Generate test weights for symmetric quantization (zero point is 8 for 4-bit)
  std::vector<uint8_t> fc1_experts_weights(num_experts * fc1_weight_size_per_expert, 0x88);  // 8,8 -> 0,0 signed weights
  std::vector<uint8_t> fc2_experts_weights(num_experts * fc2_weight_size_per_expert, 0x88);  // 8,8 -> 0,0 signed weights
  std::vector<uint8_t> fc3_experts_weights;                                                  // Empty for SwiGLU (gate weights concatenated with FC1)

  // Scales: for SwiGLU, FC1 has 2*inter_size outputs (linear + gate)
  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.05f);  // Small scale for reasonable outputs
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.05f);
  std::vector<float> fc3_scales;

  // For SwiGLU with zero weights (0x88 -> 8,8 -> 0,0 signed):
  // Gate output = 0, Linear output = 0
  // SwiGLU = gate * sigmoid(gate) * (linear + 1) = 0 * sigmoid(0) * (0 + 1) = 0 * 0.5 * 1 = 0
  // So output should be zero
  std::vector<float> output(num_rows * hidden_size, 0.0f);

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 2);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");  // Test SwiGLU activation
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 4);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size / 2};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / 2};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};  // 2x for SwiGLU (linear + gate)
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights (empty for SwiGLU)
  cpu_tester.AddOptionalInputEdge<float>();      // fc3_scales
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output));
  cpu_tester.SetOutputTolerance(0.02f);  // Higher tolerance for SwiGLU nonlinearity

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_SwiGLU_Int8) {
#ifdef USE_MLAS
  // Skip this test if we're not testing CPU execution provider
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }
  // Test CPU implementation with 8-bit quantization and SwiGLU activation
  int num_rows = 1;
  int num_experts = 2;
  int hidden_size = 8;
  int inter_size = 8;

  const std::vector<float> input = {0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, -0.9f};
  const std::vector<float> router_probs = {0.0f, 0.0f};

  // For SwiGLU with 8-bit symmetric quantization: FC1 weights are 2x inter_size (concatenated linear + gate weights)
  const int fc1_weight_size_per_expert = hidden_size * inter_size * 2;  // For 8-bit SwiGLU
  const int fc2_weight_size_per_expert = inter_size * hidden_size;      // For 8-bit FC2

  // Generate test weights at zero (for symmetric quantization storage format: uint8 with zero point 128)
  // Fill with 128 so dequantized value (val - 128) == 0 => zero output
  std::vector<uint8_t> fc1_experts_weights(num_experts * fc1_weight_size_per_expert, 128);
  std::vector<uint8_t> fc2_experts_weights(num_experts * fc2_weight_size_per_expert, 128);
  std::vector<uint8_t> fc3_experts_weights;  // Empty for SwiGLU

  // Scales: for SwiGLU, FC1 has 2*inter_size outputs
  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.1f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.1f);
  std::vector<float> fc3_scales;

  std::vector<float> output(num_rows * hidden_size, 0.0f);

  OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  cpu_tester.AddAttribute<int64_t>("k", 2);
  cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");  // Test SwiGLU activation
  cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 8);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, inter_size * 2, hidden_size};  // 8-bit SwiGLU: explicit 2x
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};  // 2x for SwiGLU
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  cpu_tester.AddOptionalInputEdge<float>();      // fc3_scales
  cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output));
  cpu_tester.SetOutputTolerance(0.02f);

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
  cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}

TEST(MoETest, QMoETest_CPU_RouterWeights) {
  // Test that separate router_weights for aggregation works correctly.
  // router_probs is used only for Top-K expert selection.
  // router_weights is used only for weighting expert outputs.
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 16;
  int inter_size = 16;

  const std::vector<float> input = {
      -0.5f, 0.2f, 1.1f, -0.3f, 0.8f, -0.1f, 0.4f, -0.7f, 0.9f, -0.2f, 0.6f, 0.1f, -0.4f, 0.3f, -0.8f, 0.7f,
      0.1f, 0.7f, -0.4f, 0.2f, 0.8f, -0.3f, 0.5f, -0.1f, 0.6f, 0.4f, -0.7f, 0.3f, 0.9f, -0.2f, 0.1f, 0.8f};

  // router_probs is only used for Top-K selection.
  const std::vector<float> router_probs = {0.1f, 0.9f, 0.8f, 0.2f};

  // router_weights is used for aggregation and intentionally differs from router_probs.
  const std::vector<float> router_weights = {3.0f, 1.0f, 1.0f, 2.0f};

  // Use zero-valued int8 weights and expert-specific FC2 bias so the final output is
  // a simple weighted combination of constant expert outputs.
  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * hidden_size, 128);
  std::vector<uint8_t> fc2_experts_weights(num_experts * inter_size * hidden_size, 128);

  std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.1f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.1f);
  std::vector<float> fc2_experts_bias;
  fc2_experts_bias.insert(fc2_experts_bias.end(), static_cast<size_t>(hidden_size), 1.0f);
  fc2_experts_bias.insert(fc2_experts_bias.end(), static_cast<size_t>(hidden_size), 3.0f);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> router_weights_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, inter_size * 2};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> fc2_experts_bias_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  auto run_test = [&](int64_t normalize_routing_weights, const std::vector<float>& expected_output) {
    OpTester cpu_tester("QMoE", 1, onnxruntime::kMSDomain);
    cpu_tester.AddAttribute<int64_t>("k", 2);
    cpu_tester.AddAttribute<std::string>("activation_type", "swiglu");
    cpu_tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
    cpu_tester.AddAttribute<int64_t>("normalize_routing_weights", normalize_routing_weights);
    cpu_tester.AddAttribute<int64_t>("expert_weight_bits", 8);

    cpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
    cpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
    cpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
    cpu_tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
    cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
    cpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
    cpu_tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
    cpu_tester.AddInput<MLFloat16>("fc2_experts_bias", fc2_experts_bias_dims, ToFloat16(fc2_experts_bias));
    cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
    cpu_tester.AddOptionalInputEdge<float>();      // fc3_scales
    cpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
    cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc1_zero_points
    cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc2_zero_points
    cpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_zero_points
    cpu_tester.AddInput<MLFloat16>("router_weights", router_weights_dims, ToFloat16(router_weights));
    cpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
    cpu_tester.SetOutputTolerance(0.05f);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
    cpu_execution_providers.push_back(DefaultCpuExecutionProvider());
    cpu_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);
  };

  std::vector<float> expected_output_no_norm;
  expected_output_no_norm.insert(expected_output_no_norm.end(), static_cast<size_t>(hidden_size), 6.0f);
  expected_output_no_norm.insert(expected_output_no_norm.end(), static_cast<size_t>(hidden_size), 7.0f);
  run_test(0, expected_output_no_norm);

  std::vector<float> expected_output_norm;
  expected_output_norm.insert(expected_output_norm.end(), static_cast<size_t>(hidden_size), 1.5f);
  expected_output_norm.insert(expected_output_norm.end(), static_cast<size_t>(hidden_size), 7.0f / 3.0f);
  run_test(1, expected_output_norm);
}

// Test for CPU MoE implementation
static void RunMoECpuTest(const std::vector<float>& input, const std::vector<float>& router_probs,
                          const std::vector<float>& fc1_experts_weights, const std::vector<float>& fc2_experts_weights,
                          const std::vector<float>& fc3_experts_weights, const std::vector<float>& fc1_experts_bias,
                          const std::vector<float>& fc2_experts_bias, const std::vector<float>& output_data, int num_rows,
                          int num_experts, int hidden_size, int inter_size, std::string activation_type,
                          int normalize_routing_weights = 1, int top_k = 1) {
  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
  tester.AddAttribute<std::string>("activation_type", activation_type);
  tester.AddAttribute<int64_t>("normalize_routing_weights", static_cast<int64_t>(normalize_routing_weights));

  bool is_swiglu = (activation_type == "swiglu");

  if (is_swiglu) {
    tester.AddAttribute<int64_t>("swiglu_fusion", static_cast<int64_t>(1));
    tester.AddAttribute<float>("activation_beta", 1.0f);
  }

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};

  int64_t fc1_output_size = is_swiglu ? (2 * inter_size) : inter_size;

  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, fc1_output_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size};
  std::vector<int64_t> fc3_experts_weights_dims = fc1_experts_weights_dims;
  std::vector<int64_t> fc1_experts_bias_dims = {num_experts, fc1_output_size};
  std::vector<int64_t> fc2_experts_bias_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  tester.AddInput<float>("input", input_dims, input);
  tester.AddInput<float>("router_probs", router_probs_dims, router_probs);
  tester.AddInput<float>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  if (!fc1_experts_bias.empty()) {
    tester.AddInput<float>("fc1_experts_bias", fc1_experts_bias_dims, fc1_experts_bias);
  } else {
    tester.AddOptionalInputEdge<float>();
  }
  tester.AddInput<float>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  if (!fc2_experts_bias.empty()) {
    tester.AddInput<float>("fc2_experts_bias", fc2_experts_bias_dims, fc2_experts_bias);
  } else {
    tester.AddOptionalInputEdge<float>();
  }
  if (!fc3_experts_weights.empty()) {
    tester.AddInput<float>("fc3_experts_weights", fc3_experts_weights_dims, fc3_experts_weights);
  } else {
    tester.AddOptionalInputEdge<float>();
  }
  tester.AddOptionalInputEdge<float>();  // fc3_experts_bias

  tester.AddOutput<float>("output", output_dims, output_data);
  tester.SetOutputTolerance(0.05f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#if !defined(__wasm__) && !defined(_WIN32) && !defined(__ANDROID__)
static std::vector<nlohmann::json> ParseMoeCounterLogs(const std::string& logs) {
  constexpr std::string_view marker = "moe_expert_counters ";
  std::vector<nlohmann::json> events;
  size_t position = 0;
  while ((position = logs.find(marker, position)) != std::string::npos) {
    position += marker.size();
    const size_t end = logs.find('\n', position);
    events.push_back(nlohmann::json::parse(logs.substr(position, end - position)));
    if (end == std::string::npos) {
      break;
    }
    position = end + 1;
  }
  return events;
}

static std::vector<nlohmann::json> RunMoECpuLoggingTest(
    bool enable_moe_statistics,
    const std::vector<int64_t>& input_shape,
    OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
    std::string_view expected_error = "",
    int64_t top_k = 1) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 4;
  constexpr int inter_size = 8;
  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> router_probs = {
      0.8f, 0.2f,
      0.3f, 0.7f};
  const std::vector<float> fc1_experts_weights(num_experts * hidden_size * (2 * inter_size), 0.1f);
  const std::vector<float> fc2_experts_weights(num_experts * inter_size * hidden_size, 0.1f);
  const std::vector<float> output_data = {
      1.169694f, 1.169694f, 1.169694f, 1.169694f,
      6.970291f, 6.970291f, 6.970291f, 6.970291f};

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<float>("activation_beta", 1.0f);
  tester.AddInput<float>("input_ids", input_shape, input);
  tester.AddInput<float>("router_probs", {num_rows, num_experts}, router_probs);
  tester.AddInput<float>("fc1_experts_weights", {num_experts, hidden_size, 2 * inter_size}, fc1_experts_weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("fc2_experts_weights", {num_experts, inter_size, hidden_size}, fc2_experts_weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", input_shape, output_data);
  tester.SetOutputTolerance(0.05f);

  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kWARNING);
  if (enable_moe_statistics) {
    EXPECT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  }

  RunOptions run_options;
  run_options.run_tag = "{routing \"request\"}";
  run_options.run_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  testing::internal::CaptureStderr();
  tester.Run(session_options, expected_result, std::string(expected_error), {},
             &run_options, &execution_providers);
  return ParseMoeCounterLogs(testing::internal::GetCapturedStderr());
}

TEST(MoETest, MoECpuLogHasCounterUpdateSchema) {
  const auto routing_events = RunMoECpuLoggingTest(
      true, {1, 2, 4}, OpTester::ExpectResult::kExpectSuccess, "", 2);
  ASSERT_EQ(routing_events.size(), 1U);
  const auto& event = routing_events[0];
  EXPECT_EQ(event["request_id"], "{routing \"request\"}");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({0, 1}));
  EXPECT_EQ(event["counters"], nlohmann::json({0.1, 0.1}));
  EXPECT_TRUE(event.contains("node_index"));
  EXPECT_TRUE(event.contains("node_name"));
  EXPECT_EQ(event["node_type"], "MoE");
}

TEST(MoETest, MoECpuRoutingLogDisabledHasNoDecision) {
  EXPECT_TRUE(RunMoECpuLoggingTest(false, {2, 1, 4}).empty());
}

#ifdef USE_MLAS
static std::vector<nlohmann::json> RunQMoECpuLoggingTest(
    const std::vector<int64_t>& input_shape,
    OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
    std::string_view expected_error = "",
    int64_t top_k = 1) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 32;
  constexpr int inter_size = 32;
  constexpr int pack_size = 2;

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddInput<MLFloat16>("input", input_shape,
                             ToFloat16(std::vector<float>(num_rows * hidden_size, 1.0f)));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts},
                             ToFloat16({0.1f, 0.9f, 0.8f, 0.2f}));
  tester.AddInput<uint8_t>("fc1_experts_weights",
                           {num_experts, 2 * inter_size, hidden_size / pack_size},
                           std::vector<uint8_t>(num_experts * inter_size * hidden_size, 0x88));
  tester.AddInput<float>("fc1_scales", {num_experts, 2 * inter_size},
                         std::vector<float>(num_experts * 2 * inter_size, 0.01f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights",
                           {num_experts, inter_size, hidden_size / pack_size},
                           std::vector<uint8_t>(num_experts * inter_size * hidden_size / pack_size, 0x88));
  tester.AddInput<float>("fc2_scales", {num_experts, hidden_size},
                         std::vector<float>(num_experts * hidden_size, 0.01f));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", input_shape,
                              ToFloat16(std::vector<float>(num_rows * hidden_size, 0.0f)));

  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  EXPECT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  RunOptions run_options;
  run_options.run_tag = "cpu qmoe request";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  testing::internal::CaptureStderr();
  tester.Run(session_options, expected_result, std::string(expected_error), {},
             &run_options, &execution_providers);
  return ParseMoeCounterLogs(testing::internal::GetCapturedStderr());
}

TEST(MoETest, QMoECpuLogHasCounterUpdateSchema) {
  const auto routing_events = RunQMoECpuLoggingTest(
      {1, 2, 32}, OpTester::ExpectResult::kExpectSuccess, "", 2);
  ASSERT_EQ(routing_events.size(), 1U);
  const auto& event = routing_events[0];
  EXPECT_EQ(event["request_id"], "cpu qmoe request");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({1, 0}));
  EXPECT_EQ(event["counters"], nlohmann::json({0.1, 0.1}));
  EXPECT_EQ(event["node_type"], "QMoE");
}
#endif

#ifdef USE_CUDA
static void ConfigureCudaMoeRoutingTester(
    OpTester& tester,
    const std::vector<int64_t>& input_shape = {2, kMoEMinCudaDim},
    int64_t top_k = 1) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = kMoEMinCudaDim;
  constexpr int inter_size = kMoEMinCudaDim;

  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "relu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddInput<float>("input_ids", input_shape,
                         std::vector<float>(num_rows * hidden_size, 1.0f));
  tester.AddInput<float>("router_probs", {num_rows, num_experts},
                         {2.0f, 1.0f, 1.0f, 3.0f});
  tester.AddInput<float>("fc1_experts_weights", {num_experts, inter_size, hidden_size},
                         std::vector<float>(num_experts * inter_size * hidden_size, 0.0f));
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("fc2_experts_weights", {num_experts, hidden_size, inter_size},
                         std::vector<float>(num_experts * hidden_size * inter_size, 0.0f));
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", input_shape,
                          std::vector<float>(num_rows * hidden_size, 0.0f));
}

TEST(MoETest, MoECudaLogHasCounterUpdateSchema) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  auto execution_provider = DefaultCudaExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  if (execution_provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE routing statistics are not supported by the CUDA plugin execution provider.";
  }

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  ConfigureCudaMoeRoutingTester(tester, {2, kMoEMinCudaDim}, 2);

  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  RunOptions run_options;
  run_options.run_tag = "cuda request";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  testing::internal::CaptureStderr();
  tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, &run_options, &execution_providers);
  const auto routing_events = ParseMoeCounterLogs(testing::internal::GetCapturedStderr());

  ASSERT_EQ(routing_events.size(), 1U);
  const auto& event = routing_events[0];
  EXPECT_EQ(event["request_id"], "cuda request");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({0, 1}));
  EXPECT_EQ(event["counters"], nlohmann::json({0.1, 0.1}));
  EXPECT_EQ(event["node_type"], "MoE");
}

TEST(MoETest, QMoECudaTiledLogHasOneCounterUpdate) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  auto execution_provider = DefaultCudaExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  if (execution_provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE routing statistics are not supported by the CUDA plugin execution provider.";
  }

  constexpr int num_rows = 3;
  constexpr int num_experts = 2;
  constexpr int hidden_size = kMoEMinCudaDim;
  constexpr int inter_size = kMoEMinCudaDim;
  constexpr int pack_size = 2;

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "relu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddInput<MLFloat16>("input_ids", {num_rows, hidden_size},
                             ToFloat16(std::vector<float>(num_rows * hidden_size, 1.0f)));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts},
                             ToFloat16({2.0f, 1.0f, 1.0f, 3.0f, 4.0f, 0.0f}));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, hidden_size, inter_size / pack_size},
                           std::vector<uint8_t>(num_experts * hidden_size * inter_size / pack_size, 0));
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, inter_size},
                             ToFloat16(std::vector<float>(num_experts * inter_size, 1.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, inter_size, hidden_size / pack_size},
                           std::vector<uint8_t>(num_experts * inter_size * hidden_size / pack_size, 0));
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size},
                             ToFloat16(std::vector<float>(num_experts * hidden_size, 1.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size},
                              ToFloat16(std::vector<float>(num_rows * hidden_size, 0.0f)));

  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      "ep.cuda.qmoe_row_tile_size", "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsDisableCPUEPFallback, "1"));
  RunOptions run_options;
  run_options.run_tag = "qmoe tiled request";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  testing::internal::CaptureStderr();
  tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, &run_options, &execution_providers);
  const auto routing_events = ParseMoeCounterLogs(testing::internal::GetCapturedStderr());

  ASSERT_EQ(routing_events.size(), 1U);
  const auto& event = routing_events[0];
  EXPECT_EQ(event["request_id"], "qmoe tiled request");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({0, 1}));
  EXPECT_EQ(event["counters"], nlohmann::json({0.1, 0.1}));
  EXPECT_EQ(event["node_type"], "QMoE");
}

TEST(MoETest, QMoECudaInt3TiledLogHasOneCounterUpdate) {
  if (!HasCudaEnvironment(750)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.5 or newer is required.";
  }
  auto execution_provider = DefaultCudaExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  if (execution_provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE routing statistics are not supported by the CUDA plugin execution provider.";
  }

  constexpr int64_t num_rows = 3;
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t inter_size = 64;
  constexpr int64_t fc1_rows = 2 * inter_size;
  constexpr int64_t block_size = 32;
  constexpr int64_t bits = 3;

  // Code 4 is the INT3 zero point, so all dequantized weights are zero.
  const std::vector<uint8_t> fc1_codes(static_cast<size_t>(num_experts * fc1_rows * hidden_size), 4);
  const std::vector<uint8_t> fc2_codes(static_cast<size_t>(num_experts * hidden_size * inter_size), 4);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", bits);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddAttribute<int64_t>("block_size", block_size);
  tester.AddInput<MLFloat16>("input_ids", {num_rows, hidden_size},
                             ToFloat16(std::vector<float>(num_rows * hidden_size, 1.0f)));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts},
                             ToFloat16({2.0f, 1.0f, 1.0f, 3.0f, 4.0f, 0.0f}));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, fc1_rows, hidden_size * bits / 8},
                           PackLsbBits(fc1_codes, num_experts * fc1_rows, hidden_size, bits), true);
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_rows, hidden_size / block_size},
                             ToFloat16(std::vector<float>(num_experts * fc1_rows * hidden_size / block_size, 1.0f)),
                             true);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size * bits / 8},
                           PackLsbBits(fc2_codes, num_experts * hidden_size, inter_size, bits), true);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, inter_size / block_size},
                             ToFloat16(std::vector<float>(num_experts * hidden_size * inter_size / block_size, 1.0f)),
                             true);
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size},
                              ToFloat16(std::vector<float>(num_rows * hidden_size, 0.0f)));

  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry("ep.cuda.qmoe_row_tile_size", "1"));
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  RunOptions run_options;
  run_options.run_tag = "qmoe int3 tiled request";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  testing::internal::CaptureStderr();
  tester.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, &run_options, &execution_providers);
  const auto routing_events = ParseMoeCounterLogs(testing::internal::GetCapturedStderr());

  ASSERT_EQ(routing_events.size(), 1U);
  const auto& event = routing_events[0];
  EXPECT_EQ(event["request_id"], "qmoe int3 tiled request");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({0, 1}));
  EXPECT_EQ(event["node_type"], "QMoE");
}

TEST(MoETest, MoeStatisticsRejectsCudaGraphCapture) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.enable_cuda_graph = 1;
  auto execution_provider = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(execution_provider, nullptr);
  if (execution_provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE routing statistics are rejected for CUDA plugins before checking graph capture.";
  }

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  ConfigureCudaMoeRoutingTester(tester);

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  tester.Run(session_options, OpTester::ExpectResult::kExpectFailure,
             "is not supported when graph capture is enabled", {}, nullptr, &execution_providers);
}

TEST(MoETest, MoeStatisticsRejectsCudaPlugin) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  auto execution_provider = DefaultCudaExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  if (execution_provider->GetOrtEp() == nullptr) {
    GTEST_SKIP() << "This test requires the CUDA plugin execution provider.";
  }

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  ConfigureCudaMoeRoutingTester(tester);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  tester.Run(session_options, OpTester::ExpectResult::kExpectFailure,
             "is not supported by the CUDA plugin execution provider", {}, nullptr, &execution_providers);
}
#endif
#endif

TEST(MoETest, MoECpuTest_BasicSwiGLU) {
  int num_rows = 2;
  int num_experts = 2;
  int hidden_size = 4;
  int inter_size = 8;

  // Simple test data
  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  const std::vector<float> router_probs = {
      0.8f, 0.2f,
      0.3f, 0.7f};

  const std::vector<float> fc1_experts_weights(num_experts * hidden_size * (2 * inter_size), 0.1f);

  const std::vector<float> fc2_experts_weights(num_experts * inter_size * hidden_size, 0.1f);

  const std::vector<float> fc3_experts_weights = {};  // No FC3
  const std::vector<float> fc1_experts_bias = {};     // No bias
  const std::vector<float> fc2_experts_bias = {};     // No bias

  const std::vector<float> output_data = {
      1.169694f, 1.169694f, 1.169694f, 1.169694f,
      6.970291f, 6.970291f, 6.970291f, 6.970291f};

  RunMoECpuTest(input, router_probs, fc1_experts_weights, fc2_experts_weights,
                fc3_experts_weights, fc1_experts_bias, fc2_experts_bias, output_data,
                num_rows, num_experts, hidden_size, inter_size, "swiglu");
}

// Regression test: MoE must reject k > num_experts. Without the guard the per-token
// sorted_logits vector (size = num_experts) is passed to std::partial_sort with
// mid = begin + k_, writing past the vector's heap allocation.
TEST(MoETest, MoECpuTest_KExceedsNumExperts) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  constexpr int num_rows = 1;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 4;
  constexpr int inter_size = 8;
  constexpr int64_t k = 3;  // > num_experts: the bug trigger.

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> router_probs = {0.6f, 0.4f};
  const std::vector<float> fc1_experts_weights(num_experts * hidden_size * (2 * inter_size), 0.1f);
  const std::vector<float> fc2_experts_weights(num_experts * inter_size * hidden_size, 0.1f);
  const std::vector<float> dummy_output(num_rows * hidden_size, 0.0f);

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", int64_t{1});
  tester.AddAttribute<int64_t>("swiglu_fusion", int64_t{1});
  tester.AddAttribute<float>("activation_beta", 1.0f);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, 2 * inter_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  tester.AddInput<float>("input", input_dims, input);
  tester.AddInput<float>("router_probs", router_probs_dims, router_probs);
  tester.AddInput<float>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  tester.AddOptionalInputEdge<float>();  // fc1_experts_bias
  tester.AddInput<float>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  tester.AddOptionalInputEdge<float>();  // fc2_experts_bias
  tester.AddOptionalInputEdge<float>();  // fc3_experts_weights
  tester.AddOptionalInputEdge<float>();  // fc3_experts_bias
  tester.AddOutput<float>("output", output_dims, dummy_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "'k' must be <= num_experts",
             {}, nullptr, &execution_providers);
}

// Regression test: QMoE must reject k > num_experts. Same partial_sort OOB pattern
// as MoE above, but on the quantized CPU kernel path.
TEST(MoETest, QMoETest_CPU_KExceedsNumExperts) {
#ifdef USE_MLAS
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (!cpu_ep) {
    GTEST_SKIP() << "CPU execution provider not available";
  }

  constexpr int64_t num_rows = 1;
  constexpr int64_t num_experts = 2;
  constexpr int64_t hidden_size = 8;
  constexpr int64_t inter_size = 8;
  constexpr int64_t expert_weight_bits = 4;
  constexpr int64_t pack_size = 8 / expert_weight_bits;
  constexpr int64_t fc1_inter_size = 2 * inter_size;  // swiglu fused
  constexpr int64_t k = 3;                            // > num_experts: the bug trigger.

  const std::vector<float> input = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
  const std::vector<float> router_probs = {0.5f, 0.5f};
  std::vector<uint8_t> fc1_experts_weights(num_experts * fc1_inter_size * (hidden_size / pack_size), 0x01);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * (inter_size / pack_size), 0x10);
  std::vector<float> fc1_scales(num_experts * fc1_inter_size, 0.1f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.05f);
  std::vector<float> dummy_output(num_rows * hidden_size, 0.0f);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", int64_t{1});
  tester.AddAttribute<int64_t>("expert_weight_bits", expert_weight_bits);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, fc1_inter_size, hidden_size / pack_size};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / pack_size};
  std::vector<int64_t> fc1_scales_dims = {num_experts, fc1_inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  tester.AddInput<float>("fc1_scales", fc1_scales_dims, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  tester.AddInput<float>("fc2_scales", fc2_scales_dims, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  tester.AddOptionalInputEdge<float>();      // fc3_scales
  tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(dummy_output));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "'k' must be <= num_experts",
             {}, nullptr, &execution_providers);
#else
  GTEST_SKIP() << "Skipping CPU QMoE test";
#endif
}
#endif

}  // namespace test
}  // namespace onnxruntime
