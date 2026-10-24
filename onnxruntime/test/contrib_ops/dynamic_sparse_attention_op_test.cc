// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#if defined(USE_CUDA) || defined(USE_WEBGPU)
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/default_providers.h"
#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_environment.h"
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#include "test/common/cuda_op_test_utils.h"
#endif

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_WEBGPU)
namespace {

struct DynamicSparseAttentionCase {
  int64_t batch_size = 1;
  int64_t sequence_length = 1;
  int64_t num_heads = 1;
  int64_t kv_num_heads = 1;
  int64_t head_size = 8;
  int64_t cache_sequence_length = 1;
  int64_t auxiliary_sequence_length = 0;
  int64_t max_selected = 1;
  int64_t rotary_cache_length = 0;
  int64_t rotary_half_dim = 0;

  std::string attention_mode = "selected_only";
  std::string selected_kv_source = "main";
  float scale = 1.0f;
  float qk_norm_epsilon = 1e-6f;
  int64_t local_window_size = -1;
  int64_t auxiliary_kv_shared = 0;
  int64_t do_rotary = 0;
  int64_t rotary_interleaved = 0;
  int64_t rotary_offset = 0;
  int64_t smooth_softmax = 0;
  bool packed_qkv = false;
  bool has_past = true;
  bool has_scale = true;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> past_key;
  std::vector<float> past_value;
  std::vector<float> auxiliary_key;
  std::vector<float> auxiliary_value;
  std::vector<int32_t> selected_indices;
  std::vector<int64_t> selected_indices_shape;
  std::vector<int32_t> selected_counts;
  std::vector<int64_t> selected_counts_shape;
  std::vector<int32_t> seqlens_k;
  int32_t total_sequence_length = 1;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<int64_t> position_ids;
  std::vector<float> q_norm_weight;
  std::vector<float> k_norm_weight;
  std::vector<float> head_sink;
  std::vector<float> key_scale;
  std::vector<float> value_scale;
  std::optional<int32_t> committed_length;
  std::optional<int32_t> expected_validation_status;

  std::vector<float> expected_output;
  std::vector<float> expected_present_key;
  std::vector<float> expected_present_value;
};

template <typename T>
std::vector<T> ToTensorData(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, float>) {
    return values;
  } else {
    std::vector<T> converted;
    converted.reserve(values.size());
    for (float value : values) {
      converted.emplace_back(static_cast<T>(value));
    }
    return converted;
  }
}

template <typename T = MLFloat16, typename CacheT = T>
void RunDynamicSparseAttentionCase(
    const DynamicSparseAttentionCase& c,
    std::unique_ptr<IExecutionProvider> execution_provider,
    OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
    const std::string& expected_error = "") {
  ASSERT_NE(execution_provider, nullptr);

  const int64_t hidden_size = c.num_heads * c.head_size;
  const int64_t kv_hidden_size = c.kv_num_heads * c.head_size;
  const int64_t query_count = c.batch_size * c.sequence_length;
  const int64_t query_input_size =
      c.packed_qkv ? hidden_size + 2 * kv_hidden_size : hidden_size;

  ASSERT_EQ(c.query.size(), static_cast<size_t>(query_count * query_input_size));
  if (!c.packed_qkv) {
    ASSERT_EQ(c.key.size(), static_cast<size_t>(query_count * kv_hidden_size));
    ASSERT_EQ(c.value.size(), static_cast<size_t>(query_count * kv_hidden_size));
  }
  const size_t cache_elements =
      static_cast<size_t>(c.batch_size * c.kv_num_heads * c.cache_sequence_length * c.head_size);
  if (c.has_past) {
    ASSERT_EQ(c.past_key.size(), cache_elements);
    ASSERT_EQ(c.past_value.size(), cache_elements);
  }

  OpTester tester("DynamicSparseAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", c.num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  if (c.has_scale) {
    tester.AddAttribute<float>("scale", c.scale);
  }
  tester.AddAttribute<int64_t>("is_causal", 1);
  tester.AddAttribute<int64_t>("local_window_size", c.local_window_size);
  tester.AddAttribute<std::string>("attention_mode", c.attention_mode);
  tester.AddAttribute<std::string>("selected_kv_source", c.selected_kv_source);
  tester.AddAttribute<int64_t>("do_rotary", c.do_rotary);
  tester.AddAttribute<int64_t>("rotary_interleaved", c.rotary_interleaved);
  tester.AddAttribute<int64_t>("rotary_offset", c.rotary_offset);
  tester.AddAttribute<float>("qk_norm_epsilon", c.qk_norm_epsilon);
  tester.AddAttribute<int64_t>("smooth_softmax", c.smooth_softmax);
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", c.auxiliary_kv_shared);

  tester.AddInput<T>("query", {c.batch_size, c.sequence_length, query_input_size},
                     ToTensorData<T>(c.query));
  if (c.packed_qkv) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("key", {c.batch_size, c.sequence_length, kv_hidden_size},
                       ToTensorData<T>(c.key));
    tester.AddInput<T>("value", {c.batch_size, c.sequence_length, kv_hidden_size},
                       ToTensorData<T>(c.value));
  }
  if (c.has_past) {
    tester.AddInput<CacheT>("past_key",
                            {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                            ToTensorData<CacheT>(c.past_key));
    tester.AddInput<CacheT>("past_value",
                            {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                            ToTensorData<CacheT>(c.past_value));
  } else {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  }

  if (c.auxiliary_sequence_length > 0) {
    tester.AddInput<T>("auxiliary_key",
                       {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                       ToTensorData<T>(c.auxiliary_key));
  } else {
    tester.AddOptionalInputEdge<T>();
  }

  if (!c.auxiliary_value.empty()) {
    tester.AddInput<T>("auxiliary_value",
                       {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                       ToTensorData<T>(c.auxiliary_value));
  } else {
    tester.AddOptionalInputEdge<T>();
  }

  const std::vector<int64_t> selected_indices_shape =
      c.selected_indices_shape.empty()
          ? std::vector<int64_t>{query_count, c.max_selected}
          : c.selected_indices_shape;
  const std::vector<int64_t> selected_counts_shape =
      c.selected_counts_shape.empty()
          ? std::vector<int64_t>{query_count}
          : c.selected_counts_shape;
  tester.AddInput<int32_t>("selected_indices", selected_indices_shape, c.selected_indices);
  tester.AddInput<int32_t>("selected_counts", selected_counts_shape, c.selected_counts);
  tester.AddInput<int32_t>("seqlens_k", {c.batch_size}, c.seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {}, {c.total_sequence_length});

  if (c.cos_cache.empty()) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("cos_cache", {c.rotary_cache_length, c.rotary_half_dim},
                       ToTensorData<T>(c.cos_cache));
    tester.AddInput<T>("sin_cache", {c.rotary_cache_length, c.rotary_half_dim},
                       ToTensorData<T>(c.sin_cache));
  }
  if (c.position_ids.empty()) {
    tester.AddOptionalInputEdge<int64_t>();
  } else {
    tester.AddInput<int64_t>("position_ids", {c.batch_size, c.sequence_length}, c.position_ids);
  }
  if (c.q_norm_weight.empty()) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("q_norm_weight", {c.head_size}, ToTensorData<T>(c.q_norm_weight));
    tester.AddInput<T>("k_norm_weight", {c.head_size}, ToTensorData<T>(c.k_norm_weight));
  }
  if (c.head_sink.empty()) {
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("head_sink", {c.num_heads}, ToTensorData<T>(c.head_sink));
  }
  if (c.committed_length.has_value()) {
    tester.AddInput<int32_t>("committed_length", {1}, {*c.committed_length});
  } else {
    tester.AddOptionalInputEdge<int32_t>();
  }
  if (!c.key_scale.empty() || !c.value_scale.empty()) {
    if (c.key_scale.empty())
      tester.AddOptionalInputEdge<float>();
    else
      tester.AddInput<float>("key_scale", {int64_t(c.key_scale.size())}, c.key_scale);
    if (c.value_scale.empty())
      tester.AddOptionalInputEdge<float>();
    else
      tester.AddInput<float>("value_scale", {int64_t(c.value_scale.size())}, c.value_scale);
  }

  tester.AddOutput<T>("output", {c.batch_size, c.sequence_length, hidden_size},
                      ToTensorData<T>(c.expected_output));
  tester.AddOutput<CacheT>("present_key",
                           {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                           ToTensorData<CacheT>(c.expected_present_key));
  tester.AddOutput<CacheT>("present_value",
                           {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                           ToTensorData<CacheT>(c.expected_present_value));
  if (c.expected_validation_status.has_value()) {
    tester.AddOutput<int32_t>("validation_status", {1}, {*c.expected_validation_status});
  }
  tester.SetOutputTolerance(0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  tester.Run(expected_result, expected_error, {}, nullptr, &execution_providers);
}

DynamicSparseAttentionCase MakeSingleTokenSelectedOnlyCase() {
  DynamicSparseAttentionCase c;
  c.query.assign(8, 1.0f);
  c.key.assign(8, 3.0f);
  c.value.assign(8, 9.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.selected_indices = {-1, -1};
  c.selected_counts = {0};
  c.max_selected = 2;
  c.seqlens_k = {0};
  c.expected_output.assign(8, 0.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;
  return c;
}

DynamicSparseAttentionCase MakeSingleTokenSelectedValueCase(float value = 9.0f) {
  auto c = MakeSingleTokenSelectedOnlyCase();
  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, value);
  c.selected_indices = {0, -1};
  c.selected_counts = {1};
  c.expected_output.assign(8, value);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;
  return c;
}

#ifdef USE_CUDA
DynamicSparseAttentionCase MakeInt8CacheCase() {
  DynamicSparseAttentionCase c;
  c.sequence_length = 2;
  c.num_heads = 4;
  c.kv_num_heads = 2;
  c.cache_sequence_length = 4;
  c.max_selected = 3;
  c.query.assign(64, 0.0f);
  c.key.resize(32);
  c.value.resize(32);
  c.key_scale.resize(16);
  c.value_scale.resize(16);
  c.past_key.resize(64);
  c.past_value.resize(64);
  for (size_t i = 0; i < 64; ++i) {
    c.past_key[i] = float(int(i % 13) - 6);
    c.past_value[i] = float(int(i % 17) - 8);
  }
  for (size_t i = 0; i < 16; ++i) {
    c.key_scale[i] = i < 8 ? 0.5f : 0.25f;
    c.value_scale[i] = i < 8 ? 0.25f : 0.5f;
  }
  for (size_t i = 0; i < 32; ++i) {
    c.key[i] = i % 3 == 0 ? -300.0f : 0.25f * float(i);
    c.value[i] = i % 3 == 0 ? 300.0f : 0.125f * float(i);
  }
  c.key[0] = -1e20f;
  c.value[0] = 1e20f;
  c.selected_indices = {0, 2, -1, 1, 2, 3};
  c.selected_counts = {2, 3};
  c.seqlens_k = {3};
  c.total_sequence_length = 4;
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;
  for (int s = 0; s < 2; ++s) {
    for (int h = 0; h < 2; ++h) {
      for (int d = 0; d < 8; ++d) {
        const int channel = h * 8 + d;
        const int src = s * 16 + channel;
        const int dst = (h * 4 + 2 + s) * 8 + d;
        c.expected_present_key[dst] = std::clamp(std::nearbyint(c.key[src] / c.key_scale[channel]), -128.0f, 127.0f);
        c.expected_present_value[dst] = std::clamp(std::nearbyint(c.value[src] / c.value_scale[channel]), -128.0f, 127.0f);
      }
    }
  }
  c.expected_output.assign(64, 0.0f);
  for (int s = 0; s < 2; ++s) {
    for (int h = 0; h < 4; ++h) {
      const int kv = h / 2;
      for (int d = 0; d < 8; ++d) {
        float sum = 0.0f;
        for (int i = 0; i < c.selected_counts[s]; ++i) {
          const int pos = c.selected_indices[s * 3 + i];
          sum += c.expected_present_value[(kv * 4 + pos) * 8 + d] * c.value_scale[kv * 8 + d];
        }
        c.expected_output[(s * 4 + h) * 8 + d] = sum / c.selected_counts[s];
      }
    }
  }
  return c;
}

TEST(DynamicSparseAttentionTest, Int8CachePerChannelGqaAppendAndAttention_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) GTEST_SKIP();
  RunDynamicSparseAttentionCase<float, int8_t>(MakeInt8CacheCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, Int8CacheBFloat16_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) GTEST_SKIP();
  RunDynamicSparseAttentionCase<BFloat16, int8_t>(MakeInt8CacheCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, Int8CacheSplitAttention_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) GTEST_SKIP();
  auto c = MakeInt8CacheCase();
  c.cache_sequence_length = 1024;
  c.total_sequence_length = 1024;
  c.seqlens_k = {1023};
  c.max_selected = 512;
  c.selected_counts = {512, 512};
  c.selected_indices.resize(1024);
  std::iota(c.selected_indices.begin(), c.selected_indices.end(), 0);
  c.past_key.assign(2 * 1024 * 8, 0.0f);
  c.past_value.assign(2 * 1024 * 8, 3.0f);
  auto key = c.past_key;
  auto value = c.past_value;
  for (int h = 0; h < 2; ++h) {
    for (int s = 0; s < 2; ++s) {
      for (int d = 0; d < 8; ++d) {
        key[(h * 1024 + 1022 + s) * 8 + d] = c.expected_present_key[(h * 4 + 2 + s) * 8 + d];
        value[(h * 1024 + 1022 + s) * 8 + d] = c.expected_present_value[(h * 4 + 2 + s) * 8 + d];
      }
    }
  }
  c.expected_present_key = key;
  c.expected_present_value = value;
  for (int s = 0; s < 2; ++s) {
    for (int h = 0; h < 4; ++h) {
      for (int d = 0; d < 8; ++d) {
        float sum = 0.0f;
        for (int i = 0; i < 512; ++i) {
          sum += value[((h / 2) * 1024 + s * 512 + i) * 8 + d] * c.value_scale[(h / 2) * 8 + d];
        }
        c.expected_output[(s * 4 + h) * 8 + d] = sum / 512.0f;
      }
    }
  }
  RunDynamicSparseAttentionCase<BFloat16, int8_t>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, Int8CacheRejectsMissingOrMalformedScales_CUDA) {
  for (int variant = 0; variant < 3; ++variant) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    if (!cuda_ep) GTEST_SKIP();
    auto c = MakeInt8CacheCase();
    if (variant == 0) {
      c.key_scale.clear();
      c.value_scale.clear();
    } else if (variant == 1) {
      c.value_scale.clear();
    } else {
      c.key_scale.resize(8);
    }
    const char* error = variant == 0   ? "INT8 caches require per-channel"
                        : variant == 1 ? "must be provided together"
                                       : "INT8 cache scales must have shape";
    RunDynamicSparseAttentionCase<float, int8_t>(c, std::move(cuda_ep),
                                                 OpTester::ExpectResult::kExpectFailure, error);
  }
}

DynamicSparseAttentionCase MakeGroupedCacheCase(int sequence_length, bool auxiliary,
                                                bool floating_cache = false, int batch_size = 2,
                                                int num_heads = 20, int kv_num_heads = 2,
                                                bool varied_channels = false, int selected_count = 509) {
  DynamicSparseAttentionCase c;
  c.batch_size = batch_size;
  c.sequence_length = sequence_length;
  c.num_heads = num_heads;
  c.kv_num_heads = kv_num_heads;
  c.head_size = 256;
  c.cache_sequence_length = 520;
  c.max_selected = 512;
  c.attention_mode = auxiliary ? "local_plus_selected" : "selected_only";
  c.local_window_size = auxiliary ? 16 : -1;
  c.smooth_softmax = 1;
  c.scale = 0.0625f;
  c.head_sink.assign(c.num_heads, 0.25f);
  c.seqlens_k.resize(batch_size);
  for (int b = 0; b < batch_size; ++b) c.seqlens_k[b] = 519 - b % 2;
  c.total_sequence_length = 520;
  const int channels = static_cast<int>(c.kv_num_heads * c.head_size);
  const int rows = static_cast<int>(c.batch_size * c.sequence_length);
  c.query.resize(rows * c.num_heads * c.head_size);
  c.key.resize(rows * channels);
  c.value.resize(rows * channels);
  c.key_scale.resize(channels);
  c.value_scale.resize(channels);
  for (int d = 0; d < channels; ++d) {
    c.key_scale[d] = floating_cache ? 1.0f : (varied_channels ? (d % 8 + 1) * 0.03125f : (d % 2 ? 0.125f : 0.0625f));
    c.value_scale[d] = floating_cache ? 1.0f : (varied_channels ? (8 - d % 8) * 0.0625f : (d % 2 ? 0.25f : 0.125f));
  }
  for (size_t i = 0; i < c.query.size(); ++i) {
    c.query[i] = float(int(i % 7) - 3) * (varied_channels ? 0.00390625f : 0.125f);
  }
  for (size_t i = 0; i < c.key.size(); ++i) {
    c.key[i] = float(int(i % 11) - 5) * c.key_scale[i % channels];
    c.value[i] = float(int(i % 13) - 6) * c.value_scale[i % channels];
  }
  c.past_key.resize(c.batch_size * c.cache_sequence_length * channels);
  c.past_value.resize(c.past_key.size());
  constexpr int8_t kSignedBytes[] = {-128, -127, -65, -1, 0, 1, 64, 127};
  for (size_t i = 0; i < c.past_key.size(); ++i) {
    c.past_key[i] = varied_channels ? float(kSignedBytes[i % 8]) : float(int(i % 17) - 8);
    c.past_value[i] = varied_channels ? float(kSignedBytes[7 - i % 8]) : float(int(i % 19) - 9);
  }
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;
  for (int row = 0; row < rows; ++row) {
    const int b = row / sequence_length;
    const int pos = c.seqlens_k[b] + 1 - sequence_length + row % sequence_length;
    for (int d = 0; d < channels; ++d) {
      const int64_t dst =
          ((b * c.kv_num_heads + d / c.head_size) * c.cache_sequence_length + pos) * c.head_size +
          d % c.head_size;
      c.expected_present_key[dst] = c.key[row * channels + d] / c.key_scale[d];
      c.expected_present_value[dst] = c.value[row * channels + d] / c.value_scale[d];
    }
  }
  if (auxiliary) {
    c.selected_kv_source = "auxiliary";
    c.auxiliary_sequence_length = 512;
    c.auxiliary_key.resize(c.batch_size * c.auxiliary_sequence_length * channels);
    c.auxiliary_value.resize(c.auxiliary_key.size());
    for (size_t i = 0; i < c.auxiliary_key.size(); ++i) {
      c.auxiliary_key[i] = float(int(i % 17) - 8) * 0.125f;
      c.auxiliary_value[i] = float(int(i % 19) - 9) * 0.25f;
    }
  }
  c.selected_counts.resize(rows);
  c.selected_indices.assign(rows * c.max_selected, -1);
  c.expected_output.assign(c.query.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const int b = row / sequence_length;
    const int pos = c.seqlens_k[b] + 1 - sequence_length + row % sequence_length;
    c.selected_counts[row] = row == 0 ? 0 : selected_count;
    std::iota(c.selected_indices.begin() + row * c.max_selected,
              c.selected_indices.begin() + row * c.max_selected + c.selected_counts[row], 0);
    for (int h = 0; h < c.num_heads; ++h) {
      const int64_t kv = h / (c.num_heads / c.kv_num_heads);
      const int64_t out = (row * c.num_heads + h) * c.head_size;
      std::vector<float> numerator(c.head_size, 0.0f);
      double denominator = std::exp(double(c.head_sink[h]));
      const int local_count = auxiliary ? static_cast<int>(c.local_window_size) : 0;
      const int local_start = pos - local_count + 1;
      for (int i = 0; i < local_count + c.selected_counts[row]; ++i) {
        const bool selected = i >= local_count;
        const int index = selected ? i - local_count : local_start + i;
        if (selected && !auxiliary && index >= local_start) continue;
        const bool from_auxiliary = selected && auxiliary;
        const int64_t length = from_auxiliary ? c.auxiliary_sequence_length : c.cache_sequence_length;
        const int64_t offset = ((b * c.kv_num_heads + kv) * length + index) * c.head_size;
        double dot = 0.0;
        for (int d = 0; d < c.head_size; ++d) {
          const float key = from_auxiliary ? c.auxiliary_key[offset + d]
                                           : c.expected_present_key[offset + d] * c.key_scale[kv * c.head_size + d];
          dot += c.query[out + d] * key;
        }
        const double weight = std::exp(dot * c.scale);
        denominator += weight;
        for (int d = 0; d < c.head_size; ++d) {
          const float value = from_auxiliary ? c.auxiliary_value[offset + d]
                                             : c.expected_present_value[offset + d] * c.value_scale[kv * c.head_size + d];
          numerator[d] += static_cast<float>(weight * value);
        }
      }
      for (int d = 0; d < c.head_size; ++d) {
        c.expected_output[out + d] = static_cast<float>(numerator[d] / denominator);
      }
    }
  }
  if (floating_cache) {
    c.key_scale.clear();
    c.value_scale.clear();
  }
  return c;
}

TEST(DynamicSparseAttentionTest, FloatingCacheGroupedReferenceParity_CUDA) {
  for (int sequence_length : {1, 3}) {
    for (bool auxiliary : {false, true}) {
      for (bool bf16 : {false, true}) {
        auto cuda_ep = DefaultCudaExecutionProvider();
        if (!cuda_ep) GTEST_SKIP() << "CUDA EP not available.";
        SCOPED_TRACE("sequence_length=" + std::to_string(sequence_length) +
                     ", auxiliary=" + std::to_string(auxiliary) + ", bf16=" + std::to_string(bf16));
        auto c = MakeGroupedCacheCase(sequence_length, auxiliary, true);
        // Ten heads per KV head exercise partial groups of both six (prefill) and eight (decode).
        if (bf16) {
          RunDynamicSparseAttentionCase<BFloat16>(c, std::move(cuda_ep));
        } else {
          RunDynamicSparseAttentionCase<MLFloat16>(c, std::move(cuda_ep));
        }
      }
    }
  }
}

TEST(DynamicSparseAttentionTest, FloatingCacheGroupedSingleSplitWorkspace_CUDA) {
  for (bool bf16 : {false, true}) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    if (!cuda_ep) GTEST_SKIP() << "CUDA EP not available.";
    auto c = MakeGroupedCacheCase(1, false, true, 128, 4, 1);
    if (bf16) {
      RunDynamicSparseAttentionCase<BFloat16>(c, std::move(cuda_ep));
    } else {
      RunDynamicSparseAttentionCase<MLFloat16>(c, std::move(cuda_ep));
    }
  }
}

TEST(DynamicSparseAttentionTest, Int8CacheGroupedDecode_CUDA) {
  for (int sequence_length : {1, 2, 3, 4}) {
    for (bool auxiliary : {false, true}) {
      SCOPED_TRACE("sequence_length=" + std::to_string(sequence_length) +
                   ", auxiliary=" + std::to_string(auxiliary));
      auto cuda_ep = DefaultCudaExecutionProvider();
      if (!cuda_ep) GTEST_SKIP();
      RunDynamicSparseAttentionCase<BFloat16, int8_t>(
          MakeGroupedCacheCase(sequence_length, auxiliary), std::move(cuda_ep));
    }
  }
}

TEST(DynamicSparseAttentionTest, Int8CacheGroupedDecodeFloat16_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) GTEST_SKIP();
  RunDynamicSparseAttentionCase<MLFloat16, int8_t>(
      MakeGroupedCacheCase(2, false), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, Int8CacheGroupedDecodeVectorStagingChannels_CUDA) {
  for (int sequence_length : {1, 2, 3, 4}) {
    for (bool auxiliary : {false, true}) {
      SCOPED_TRACE("sequence_length=" + std::to_string(sequence_length) +
                   ", auxiliary=" + std::to_string(auxiliary));
      const auto c = MakeGroupedCacheCase(sequence_length, auxiliary, false, 2, 20, 2, true);
      auto cuda_ep = DefaultCudaExecutionProvider();
      if (!cuda_ep) GTEST_SKIP();
      RunDynamicSparseAttentionCase<BFloat16, int8_t>(c, std::move(cuda_ep));
      cuda_ep = DefaultCudaExecutionProvider();
      RunDynamicSparseAttentionCase<MLFloat16, int8_t>(c, std::move(cuda_ep));
    }
  }
}

TEST(DynamicSparseAttentionTest, GroupedMergeActiveSplitBoundaries_CUDA) {
  for (int count : {1, 31, 32, 33, 255, 256, 257, 511}) {
    for (bool auxiliary : {false, true}) {
      SCOPED_TRACE("selected_count=" + std::to_string(count) +
                   ", auxiliary=" + std::to_string(auxiliary));
      auto cuda_ep = DefaultCudaExecutionProvider();
      if (!cuda_ep) GTEST_SKIP();
      RunDynamicSparseAttentionCase<BFloat16, int8_t>(
          MakeGroupedCacheCase(2, auxiliary, false, 2, 20, 2, true, count), std::move(cuda_ep));
    }
  }
}

TEST(DynamicSparseAttentionTest, Int8CacheRejectsNonpositiveOrNonfiniteScales_CUDA) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION", "1"}}};
  for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    if (!cuda_ep) GTEST_SKIP();
    auto c = MakeInt8CacheCase();
    c.value_scale[15] = invalid;
    RunDynamicSparseAttentionCase<float, int8_t>(
        c, std::move(cuda_ep), OpTester::ExpectResult::kExpectFailure,
        "INT8 cache scales must be positive and finite");
  }
}

Status ResolveDynamicSparseAttentionGraph(size_t output_count, int64_t query_width = 8,
                                          int64_t num_heads = 1, int64_t kv_num_heads = 1,
                                          bool packed_qkv = false, const char* query_width_symbol = nullptr,
                                          bool* output_width_is_unknown = nullptr,
                                          bool unshaped_past = false,
                                          bool initialize_total_length = false,
                                          bool* present_capacity_is_unknown = nullptr) {
  Model model("dynamic_sparse_attention_shape_inference", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();

  std::vector<ONNX_NAMESPACE::TypeProto> tensor_types;
  tensor_types.reserve(10);
  auto add_tensor_type = [&](int elem_type, std::initializer_list<int64_t> dims) {
    tensor_types.emplace_back();
    auto* type = &tensor_types.back();
    type->mutable_tensor_type()->set_elem_type(elem_type);
    auto* shape = type->mutable_tensor_type()->mutable_shape();
    for (const int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
    return type;
  };

  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  auto* query_type = add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, 1, query_width});
  if (query_width_symbol != nullptr) {
    auto* width_dim = query_type->mutable_tensor_type()->mutable_shape()->mutable_dim(2);
    width_dim->clear_dim_value();
    width_dim->set_dim_param(query_width_symbol);
  }
  auto* past_type = unshaped_past
                        ? add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {})
                        : add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, 1, 2, 8});
  if (unshaped_past) {
    past_type->mutable_tensor_type()->clear_shape();
  }
  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("query", query_type),
      packed_qkv
          ? &empty
          : &graph.GetOrCreateNodeArg(
                "key", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, 1, 8})),
      packed_qkv
          ? &empty
          : &graph.GetOrCreateNodeArg(
                "value", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, 1, 8})),
      &graph.GetOrCreateNodeArg("past_key", past_type),
      &graph.GetOrCreateNodeArg("past_value", past_type),
      &empty,
      &empty,
      &graph.GetOrCreateNodeArg("selected_indices",
                                add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {1, 1})),
      &graph.GetOrCreateNodeArg("selected_counts",
                                add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {1})),
      &graph.GetOrCreateNodeArg("seqlens_k",
                                add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {1})),
      &graph.GetOrCreateNodeArg("total_sequence_length",
                                add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {}))};

  std::vector<NodeArg*> outputs;
  outputs.reserve(output_count);
  outputs.push_back(&graph.GetOrCreateNodeArg("output", nullptr));
  if (output_count > 1) {
    outputs.push_back(&graph.GetOrCreateNodeArg("present_key", nullptr));
  }
  if (output_count > 2) {
    outputs.push_back(&graph.GetOrCreateNodeArg("present_value", nullptr));
  }

  auto& node = graph.AddNode("dynamic_sparse_attention", "DynamicSparseAttention", "",
                             inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", num_heads);
  node.AddAttribute("kv_num_heads", kv_num_heads);
  if (initialize_total_length) {
    ONNX_NAMESPACE::TensorProto total_length;
    total_length.set_name("total_sequence_length");
    total_length.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    total_length.add_int32_data(2);
    graph.AddInitializedTensor(total_length);
  }
  auto status = graph.Resolve();
  if (status.IsOK() && output_width_is_unknown != nullptr) {
    const auto& width_dim = outputs[0]->Shape()->dim(2);
    *output_width_is_unknown = !width_dim.has_dim_value() && !width_dim.has_dim_param();
  }
  if (status.IsOK() && present_capacity_is_unknown != nullptr) {
    const auto& capacity_dim = outputs[1]->Shape()->dim(2);
    *present_capacity_is_unknown = !capacity_dim.has_dim_value() && !capacity_dim.has_dim_param();
  }
  return status;
}
#endif

}  // namespace
#endif  // defined(USE_CUDA) || defined(USE_WEBGPU)

#ifdef USE_CUDA

TEST(DynamicSparseAttentionTest, ShapeInferenceSupportsOptionalCacheOutputs_CUDA) {
  ASSERT_STATUS_OK(ResolveDynamicSparseAttentionGraph(1));
  ASSERT_STATUS_OK(ResolveDynamicSparseAttentionGraph(2));
}

TEST(DynamicSparseAttentionTest, ShapeInferenceDoesNotCopyPackedSymbolicWidth_CUDA) {
  bool output_width_is_unknown = false;
  ASSERT_STATUS_OK(ResolveDynamicSparseAttentionGraph(1, 8, 4, 2, true, "packed_width",
                                                      &output_width_is_unknown));
  EXPECT_TRUE(output_width_is_unknown);
}

TEST(DynamicSparseAttentionTest, ShapeInferenceLeavesUnshapedPastCapacityUnknown_CUDA) {
  bool present_capacity_is_unknown = false;
  ASSERT_STATUS_OK(ResolveDynamicSparseAttentionGraph(3, 8, 1, 1, false, nullptr, nullptr,
                                                      true, true, &present_capacity_is_unknown));
  EXPECT_TRUE(present_capacity_is_unknown);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(DynamicSparseAttentionTest, ShapeInferenceRejectsNonDivisibleSeparateQueryWidth_CUDA) {
  const auto status = ResolveDynamicSparseAttentionGraph(3, 9, 2);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("Query hidden size must be divisible"), std::string::npos);
}

TEST(DynamicSparseAttentionTest, ShapeInferenceRejectsHeadCountsOutsideKernelBounds_CUDA) {
  const auto num_heads_status =
      ResolveDynamicSparseAttentionGraph(3, 8, std::numeric_limits<int64_t>::max(), 1);
  ASSERT_FALSE(num_heads_status.IsOK());
  EXPECT_NE(num_heads_status.ErrorMessage().find("num_heads must be a positive int"), std::string::npos);

  const auto kv_num_heads_status =
      ResolveDynamicSparseAttentionGraph(3, 8, 2, std::numeric_limits<int64_t>::max());
  ASSERT_FALSE(kv_num_heads_status.IsOK());
  EXPECT_NE(kv_num_heads_status.ErrorMessage().find("kv_num_heads must be positive"), std::string::npos);
}
#endif

TEST(DynamicSparseAttentionTest, SelectedOnlyMainVariableCountsGqaAndCacheAppend_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.sequence_length = 2;
  c.num_heads = 2;
  c.cache_sequence_length = 3;
  c.max_selected = 3;
  c.total_sequence_length = 3;

  c.query.assign(2 * 2 * 8, 0.0f);
  c.key.assign(2 * 8, 0.0f);
  c.value.insert(c.value.end(), 8, 3.0f);
  c.value.insert(c.value.end(), 8, 5.0f);
  c.past_key.assign(3 * 8, 0.0f);
  c.past_value.assign(3 * 8, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 1.0f);
  c.selected_indices = {0, -1, -1,
                        0, 2, -1};
  c.selected_counts = {1, 2};
  c.seqlens_k = {2};

  c.expected_output.insert(c.expected_output.end(), 2 * 8, 1.0f);
  c.expected_output.insert(c.expected_output.end(), 2 * 8, 3.0f);
  c.expected_present_key.assign(3 * 8, 0.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 1.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 3.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 5.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, ValidationStatusReportsCommittedCursor_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.cache_sequence_length = 2;
  c.total_sequence_length = 2;
  c.past_key.assign(16, 0.0f);
  c.past_value.assign(16, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 0.0f);
  c.expected_present_key.assign(16, 0.0f);
  c.expected_present_value.assign(16, 0.0f);
  c.committed_length = 0;
  c.expected_validation_status = 0;
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));

  cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  c.committed_length = 1;
  c.expected_validation_status = 9;
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, LocalPlusSelectedAuxiliaryJointSoftmaxSinkSharedKv_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.auxiliary_kv_shared = 1;
  c.auxiliary_sequence_length = 2;

  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 2.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 4.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 8.0f);
  c.selected_indices = {1};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.head_sink = {0.0f};

  // Q is zero, so the local, selected auxiliary, and sink logits are all zero.
  // They share one denominator: (2 + 8 + 0) / 3.
  c.expected_output.assign(8, 10.0f / 3.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, LocalPlusSelectedDistinctAuxiliaryValue_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.auxiliary_sequence_length = 1;
  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 2.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.auxiliary_key.assign(8, 0.0f);
  c.auxiliary_value.assign(8, 7.0f);
  c.selected_indices = {0};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.expected_output.assign(8, 4.5f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyNoPastPrefill_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(6.0f);
  c.has_past = false;
  c.past_key.clear();
  c.past_value.clear();
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyEmptySetProducesZeroAndAppendsCache_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  RunDynamicSparseAttentionCase(MakeSingleTokenSelectedOnlyCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, PackedQkvSelectedOnlyGqa_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.packed_qkv = true;
  c.num_heads = 2;
  c.query.assign(2 * 8, 0.0f);             // Q
  c.query.insert(c.query.end(), 8, 0.0f);  // K
  c.query.insert(c.query.end(), 8, 6.0f);  // V
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.selected_indices = {0};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.expected_output.assign(2 * 8, 6.0f);
  c.expected_present_key.assign(8, 0.0f);
  c.expected_present_value.assign(8, 6.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyFloat32_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  RunDynamicSparseAttentionCase<float>(
      MakeSingleTokenSelectedValueCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, FiniteLowestLogitRemainsValid_CUDA) {
  for (const int64_t candidate_capacity : {int64_t{1}, int64_t{257}}) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    if (!cuda_ep) {
      GTEST_SKIP() << "CUDA EP not available.";
    }

    DynamicSparseAttentionCase c;
    c.cache_sequence_length = 1;
    c.max_selected = candidate_capacity;
    c.total_sequence_length = 1;
    c.query.assign(8, 0.0f);
    c.query[0] = 1.0f;
    c.key.assign(8, 0.0f);
    c.key[0] = -std::numeric_limits<float>::max();
    c.value.assign(8, 3.0f);
    c.past_key.assign(8, 0.0f);
    c.past_value.assign(8, 0.0f);
    c.selected_indices.assign(candidate_capacity, -1);
    c.selected_indices.back() = 0;
    c.selected_counts = {static_cast<int32_t>(candidate_capacity)};
    c.seqlens_k = {0};
    c.expected_output.assign(8, 3.0f);
    c.expected_present_key = c.key;
    c.expected_present_value = c.value;

    SCOPED_TRACE(candidate_capacity == 1 ? "fused" : "split");
    RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
  }
}

TEST(DynamicSparseAttentionTest, ZeroWeightCandidatePreservesNonFiniteValue_CUDA) {
  for (const int64_t candidate_capacity : {int64_t{2}, int64_t{257}}) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    if (!cuda_ep) {
      GTEST_SKIP() << "CUDA EP not available.";
    }

    DynamicSparseAttentionCase c;
    c.cache_sequence_length = 2;
    c.max_selected = candidate_capacity;
    c.total_sequence_length = 2;
    c.query.assign(8, 0.0f);
    c.query[0] = 1.0f;
    c.key.assign(8, 0.0f);
    c.key[0] = -200.0f;
    c.value.assign(8, std::numeric_limits<float>::infinity());
    c.past_key.assign(16, 0.0f);
    c.past_value.assign(16, 0.0f);
    std::fill_n(c.past_value.begin(), 8, 1.0f);
    c.selected_indices.assign(candidate_capacity, -1);
    c.selected_indices[0] = 0;
    c.selected_indices[1] = 1;
    c.selected_counts = {static_cast<int32_t>(candidate_capacity)};
    c.seqlens_k = {1};
    // Preserve the reference FP32 arithmetic: exp(-200) underflows, so the valid candidate contributes 0 * Inf.
    c.expected_output.assign(8, std::numeric_limits<float>::quiet_NaN());
    c.expected_present_key = c.past_key;
    c.expected_present_key[8] = -200.0f;
    c.expected_present_value = c.past_value;
    std::fill(c.expected_present_value.begin() + 8, c.expected_present_value.end(),
              std::numeric_limits<float>::infinity());

    SCOPED_TRACE(candidate_capacity == 2 ? "fused" : "split");
    RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
  }
}

TEST(DynamicSparseAttentionTest, SelectedOnlyLargeDecodeSet_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t cache_sequence_length = 257;
  constexpr int64_t head_size = 64;
  DynamicSparseAttentionCase c;
  c.head_size = head_size;
  c.cache_sequence_length = cache_sequence_length;
  c.max_selected = cache_sequence_length;
  c.total_sequence_length = cache_sequence_length;
  c.query.assign(head_size, 0.0f);
  c.key.assign(head_size, 0.0f);
  c.value.assign(head_size, static_cast<float>(cache_sequence_length));
  c.past_key.assign(cache_sequence_length * head_size, 0.0f);
  c.past_value.resize(cache_sequence_length * head_size);
  c.selected_indices.resize(cache_sequence_length);
  for (int32_t index = 0; index < cache_sequence_length; ++index) {
    c.selected_indices[index] = index;
    std::fill_n(c.past_value.begin() + static_cast<int64_t>(index) * head_size,
                head_size, static_cast<float>(index + 1));
  }
  c.selected_counts = {static_cast<int32_t>(cache_sequence_length)};
  c.seqlens_k = {static_cast<int32_t>(cache_sequence_length - 1)};
  c.expected_output.assign(head_size, 129.0f);
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;

  RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, LocalPlusSelectedLargeDecodeSet_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t candidate_count = 513;
  constexpr int64_t head_size = 64;
  DynamicSparseAttentionCase c;
  c.batch_size = 2;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = candidate_count;
  c.head_size = head_size;
  c.cache_sequence_length = candidate_count;
  c.auxiliary_sequence_length = candidate_count;
  c.max_selected = candidate_count;
  c.total_sequence_length = candidate_count;
  c.query.assign(2 * head_size, 0.0f);
  c.key.assign(2 * head_size, 0.0f);
  c.value.assign(head_size, 2.0f);
  c.value.insert(c.value.end(), head_size, 3.0f);
  c.past_key.assign(2 * candidate_count * head_size, 0.0f);
  c.past_value.assign(candidate_count * head_size, 2.0f);
  c.past_value.insert(c.past_value.end(), candidate_count * head_size, 3.0f);
  c.auxiliary_key.assign(2 * candidate_count * head_size, 0.0f);
  c.auxiliary_value.assign(candidate_count * head_size, 6.0f);
  c.auxiliary_value.insert(c.auxiliary_value.end(), candidate_count * head_size, 9.0f);
  c.selected_indices.resize(2 * candidate_count);
  std::iota(c.selected_indices.begin(), c.selected_indices.begin() + candidate_count, 0);
  std::iota(c.selected_indices.begin() + candidate_count, c.selected_indices.end(), 0);
  c.selected_counts = {static_cast<int32_t>(candidate_count),
                       static_cast<int32_t>(candidate_count)};
  c.seqlens_k = {static_cast<int32_t>(candidate_count - 1),
                 static_cast<int32_t>(candidate_count - 1)};
  c.expected_output.assign(head_size, 4.0f);
  c.expected_output.insert(c.expected_output.end(), head_size, 6.0f);
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, MultiBatchGqaMainAndAuxiliaryReads_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t head_size = 8;
  constexpr int64_t cache_capacity = 3;
  DynamicSparseAttentionCase c;
  c.batch_size = 2;
  c.num_heads = 4;
  c.kv_num_heads = 2;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.head_size = head_size;
  c.cache_sequence_length = cache_capacity;
  c.auxiliary_sequence_length = cache_capacity;
  c.max_selected = 1;
  c.total_sequence_length = cache_capacity;
  c.query.assign(c.batch_size * c.num_heads * head_size, 0.0f);
  c.key.resize(c.batch_size * c.kv_num_heads * head_size);
  c.value.resize(c.batch_size * c.kv_num_heads * head_size);
  for (int64_t batch = 0; batch < c.batch_size; ++batch) {
    for (int64_t kv_head = 0; kv_head < c.kv_num_heads; ++kv_head) {
      const int64_t offset = (batch * c.kv_num_heads + kv_head) * head_size;
      std::fill_n(c.key.begin() + offset, head_size, static_cast<float>(100 + 10 * batch + kv_head));
      std::fill_n(c.value.begin() + offset, head_size, static_cast<float>(10 + 20 * batch + 10 * kv_head));
    }
  }

  const size_t cache_elements = c.batch_size * c.kv_num_heads * cache_capacity * head_size;
  c.past_key.resize(cache_elements);
  c.past_value.resize(cache_elements);
  std::iota(c.past_key.begin(), c.past_key.end(), 1.0f);
  std::iota(c.past_value.begin(), c.past_value.end(), 1001.0f);
  c.auxiliary_key.assign(cache_elements, 0.0f);
  c.auxiliary_value.resize(cache_elements);
  for (int64_t batch = 0; batch < c.batch_size; ++batch) {
    for (int64_t kv_head = 0; kv_head < c.kv_num_heads; ++kv_head) {
      for (int64_t position = 0; position < cache_capacity; ++position) {
        const int64_t offset =
            ((batch * c.kv_num_heads + kv_head) * cache_capacity + position) * head_size;
        const float value = static_cast<float>(2 + 4 * batch + 2 * kv_head + position);
        std::fill_n(c.auxiliary_value.begin() + offset, head_size, value);
      }
    }
  }
  c.selected_indices = {0, 1};
  c.selected_counts = {1, 1};
  c.seqlens_k = {1, 2};

  c.expected_output.reserve(c.batch_size * c.num_heads * head_size);
  for (int64_t batch = 0; batch < c.batch_size; ++batch) {
    for (int64_t query_head = 0; query_head < c.num_heads; ++query_head) {
      const int64_t kv_head = query_head / (c.num_heads / c.kv_num_heads);
      const float main_value = static_cast<float>(10 + 20 * batch + 10 * kv_head);
      const float auxiliary_value = static_cast<float>(2 + 4 * batch + 2 * kv_head + batch);
      c.expected_output.insert(c.expected_output.end(), head_size, (main_value + auxiliary_value) / 2.0f);
    }
  }
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;
  for (int64_t batch = 0; batch < c.batch_size; ++batch) {
    for (int64_t kv_head = 0; kv_head < c.kv_num_heads; ++kv_head) {
      const int64_t input_offset = (batch * c.kv_num_heads + kv_head) * head_size;
      const int64_t cache_offset =
          ((batch * c.kv_num_heads + kv_head) * cache_capacity + c.seqlens_k[batch]) * head_size;
      std::copy_n(c.key.begin() + input_offset, head_size, c.expected_present_key.begin() + cache_offset);
      std::copy_n(c.value.begin() + input_offset, head_size, c.expected_present_value.begin() + cache_offset);
    }
  }

  RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SplitSoftmaxRescalesDistinctMaxima_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t candidate_count = 257;
  constexpr int64_t head_size = 64;
  DynamicSparseAttentionCase c;
  c.head_size = head_size;
  c.cache_sequence_length = candidate_count;
  c.max_selected = candidate_count;
  c.total_sequence_length = candidate_count;
  c.query.assign(head_size, 0.0f);
  c.query[0] = 1.0f;
  c.key.assign(head_size, 0.0f);
  c.key[0] = std::log(256.0f);
  c.value.assign(head_size, 10.0f);
  c.past_key.assign(candidate_count * head_size, 0.0f);
  c.past_value.assign(candidate_count * head_size, 0.0f);
  c.selected_indices.resize(candidate_count);
  std::iota(c.selected_indices.begin(), c.selected_indices.end(), 0);
  c.selected_counts = {static_cast<int32_t>(candidate_count)};
  c.seqlens_k = {static_cast<int32_t>(candidate_count - 1)};
  c.head_sink = {std::log(256.0f)};
  c.expected_output.assign(head_size, 10.0f / 3.0f);
  c.expected_present_key = c.past_key;
  c.expected_present_key[(candidate_count - 1) * head_size] = c.key[0];
  c.expected_present_value = c.past_value;
  std::fill_n(c.expected_present_value.begin() + (candidate_count - 1) * head_size,
              head_size, 10.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SplitPathModelShapeFloat16_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t candidate_count = 2048;
  constexpr int64_t head_size = 256;
  DynamicSparseAttentionCase c;
  c.num_heads = 2;
  c.head_size = head_size;
  c.cache_sequence_length = candidate_count;
  c.max_selected = candidate_count;
  c.total_sequence_length = candidate_count;
  c.query.assign(c.num_heads * head_size, 0.0f);
  c.key.assign(head_size, 0.0f);
  c.value.assign(head_size, 3.0f);
  c.past_key.assign(candidate_count * head_size, 0.0f);
  c.past_value.assign(candidate_count * head_size, 3.0f);
  c.selected_indices.resize(candidate_count);
  std::iota(c.selected_indices.begin(), c.selected_indices.end(), 0);
  c.selected_counts = {static_cast<int32_t>(candidate_count)};
  c.seqlens_k = {static_cast<int32_t>(candidate_count - 1)};
  c.expected_output.assign(c.num_heads * head_size, 3.0f);
  c.expected_present_key = c.past_key;
  c.expected_present_value = c.past_value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SplitPathStreamsMultipleTiles_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t candidate_count = 2049;
  DynamicSparseAttentionCase c;
  c.sequence_length = 2;
  c.num_heads = 64;
  c.cache_sequence_length = candidate_count;
  c.max_selected = candidate_count;
  c.total_sequence_length = candidate_count;
  c.query.assign(c.sequence_length * c.num_heads * c.head_size, 0.0f);
  for (int64_t row = 0; row < c.sequence_length; ++row) {
    for (int64_t head = 0; head < c.num_heads; ++head) {
      c.query[(row * c.num_heads + head) * c.head_size] = 1.0f;
    }
  }
  c.key.assign(c.sequence_length * c.head_size, 0.0f);
  c.key[c.head_size] = std::log(2048.0f);
  c.value.assign(c.head_size, 1.0f);
  c.value.insert(c.value.end(), c.head_size, 3.0f);
  c.past_key.assign(candidate_count * c.head_size, 0.0f);
  c.past_value.assign(candidate_count * c.head_size, 1.0f);
  c.selected_indices.resize(2 * candidate_count);
  std::iota(c.selected_indices.begin(), c.selected_indices.begin() + candidate_count - 1, 0);
  c.selected_indices[candidate_count - 1] = -1;
  std::iota(c.selected_indices.begin() + candidate_count, c.selected_indices.end(), 0);
  c.selected_counts = {static_cast<int32_t>(candidate_count - 1),
                       static_cast<int32_t>(candidate_count)};
  c.seqlens_k = {static_cast<int32_t>(candidate_count - 1)};
  c.expected_output.assign(c.num_heads * c.head_size, 1.0f);
  c.expected_output.insert(c.expected_output.end(), c.num_heads * c.head_size, 2.0f);
  c.expected_present_key = c.past_key;
  c.expected_present_key[(candidate_count - 1) * c.head_size] = c.key[c.head_size];
  c.expected_present_value = c.past_value;
  std::fill_n(c.expected_present_value.begin() + (candidate_count - 1) * c.head_size,
              c.head_size, 3.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SplitPathPrefillStreamsWithoutWorkspace_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t candidate_count = 513;
  DynamicSparseAttentionCase c;
  c.sequence_length = 32;
  c.num_heads = 64;
  c.cache_sequence_length = candidate_count;
  c.max_selected = candidate_count;
  c.total_sequence_length = candidate_count;
  c.query.assign(c.sequence_length * c.num_heads * c.head_size, 0.0f);
  for (int64_t row = 0; row < c.sequence_length; ++row) {
    for (int64_t head = 0; head < c.num_heads; ++head) {
      c.query[(row * c.num_heads + head) * c.head_size] = 1.0f;
    }
  }
  c.key.assign(c.sequence_length * c.head_size, 0.0f);
  c.key[(c.sequence_length - 1) * c.head_size] = std::log(512.0f);
  c.value.assign(c.sequence_length * c.head_size, 1.0f);
  std::fill_n(c.value.end() - c.head_size, c.head_size, 3.0f);
  c.past_key.assign(candidate_count * c.head_size, 0.0f);
  c.past_value.assign(candidate_count * c.head_size, 1.0f);
  c.selected_indices.reserve(c.sequence_length * candidate_count);
  c.selected_counts.reserve(c.sequence_length);
  for (int32_t row = 0; row < c.sequence_length; ++row) {
    const int32_t count =
        static_cast<int32_t>(candidate_count - c.sequence_length + row + 1);
    for (int32_t index = 0; index < count; ++index) {
      c.selected_indices.push_back(index);
    }
    c.selected_indices.insert(
        c.selected_indices.end(), candidate_count - count, -1);
    c.selected_counts.push_back(count);
  }
  c.seqlens_k = {static_cast<int32_t>(candidate_count - 1)};
  c.head_sink.assign(c.num_heads, std::log(512.0f));
  c.expected_output.reserve(c.sequence_length * c.num_heads * c.head_size);
  for (int64_t row = 0; row < c.sequence_length; ++row) {
    const float count = static_cast<float>(candidate_count - c.sequence_length + row + 1);
    const float expected = row == c.sequence_length - 1
                               ? 4.0f / 3.0f
                               : count / (count + 512.0f);
    c.expected_output.insert(
        c.expected_output.end(), c.num_heads * c.head_size, expected);
  }
  c.expected_present_key = c.past_key;
  c.expected_present_key[(candidate_count - 1) * c.head_size] =
      c.key[(c.sequence_length - 1) * c.head_size];
  c.expected_present_value = c.past_value;
  std::fill_n(c.expected_present_value.end() - c.head_size, c.head_size, 3.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyBFloat16_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "CUDA device does not support BFloat16.";
  }

  RunDynamicSparseAttentionCase<BFloat16>(
      MakeSingleTokenSelectedValueCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyPrefillBFloat16_CUDA) {
  auto cuda_ep_check = DefaultCudaExecutionProvider();
  if (!cuda_ep_check) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "CUDA device does not support BFloat16.";
  }

  constexpr int64_t sequence_length = 80;
  constexpr int64_t num_heads = 6;
  constexpr int64_t kv_num_heads = 2;
  constexpr int64_t group_size = num_heads / kv_num_heads;
  constexpr int64_t head_size = 256;
  constexpr int64_t max_selected = 72;

  DynamicSparseAttentionCase c;
  c.sequence_length = sequence_length;
  c.num_heads = num_heads;
  c.kv_num_heads = kv_num_heads;
  c.head_size = head_size;
  c.cache_sequence_length = sequence_length;
  c.max_selected = max_selected;
  c.total_sequence_length = sequence_length;
  c.scale = 0.0625f;
  c.query.resize(sequence_length * num_heads * head_size);
  c.key.resize(sequence_length * kv_num_heads * head_size);
  c.value.resize(sequence_length * kv_num_heads * head_size);
  c.past_key.assign(sequence_length * kv_num_heads * head_size, 0.0f);
  c.past_value.assign(sequence_length * kv_num_heads * head_size, 0.0f);
  c.selected_indices.assign(sequence_length * max_selected, -1);
  c.selected_counts.resize(sequence_length);
  c.seqlens_k = {static_cast<int32_t>(sequence_length - 1)};
  c.expected_output.assign(sequence_length * num_heads * head_size, 0.0f);

  auto round_bf16 = [](float x) -> float {
    return BFloat16(x).ToFloat();
  };

  for (int64_t s = 0; s < sequence_length; ++s) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t d = 0; d < head_size; ++d) {
        const float q_val = 0.125f * static_cast<float>(((s * 13 + h * 7 + d * 3) % 9) - 4);
        c.query[(s * num_heads + h) * head_size + d] = round_bf16(q_val);
      }
    }
    for (int64_t kh = 0; kh < kv_num_heads; ++kh) {
      for (int64_t d = 0; d < head_size; ++d) {
        if (s == 0) {
          // Unselected cache entry 0 holds NaN to verify padded candidates zero-fill instead of clamping to index 0.
          c.key[(s * kv_num_heads + kh) * head_size + d] = std::numeric_limits<float>::quiet_NaN();
          c.value[(s * kv_num_heads + kh) * head_size + d] = std::numeric_limits<float>::quiet_NaN();
        } else {
          const float k_val = 0.125f * static_cast<float>(((s * 11 + kh * 5 + d * 5) % 9) - 4);
          const float v_val = 0.125f * static_cast<float>(((s * 17 + kh * 19 + d * 7) % 13) - 6);
          c.key[(s * kv_num_heads + kh) * head_size + d] = round_bf16(k_val);
          c.value[(s * kv_num_heads + kh) * head_size + d] = round_bf16(v_val);
        }
      }
    }

    if (s == 0) {
      c.selected_counts[s] = 0;
      continue;
    }

    const int32_t count = static_cast<int32_t>(std::min<int64_t>(s, 70));
    c.selected_counts[s] = count;
    for (int32_t j = 0; j < count; ++j) {
      c.selected_indices[s * max_selected + j] = static_cast<int32_t>(s - count + 1 + j);
    }

    for (int64_t h = 0; h < num_heads; ++h) {
      const int64_t kh = h / group_size;
      const float* q_row = c.query.data() + (s * num_heads + h) * head_size;
      std::vector<float> logits(count);
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int32_t j = 0; j < count; ++j) {
        const int32_t tok = c.selected_indices[s * max_selected + j];
        const float* k_row = c.key.data() + (static_cast<int64_t>(tok) * kv_num_heads + kh) * head_size;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_size; ++d) {
          dot += q_row[d] * k_row[d];
        }
        logits[j] = dot * c.scale;
        max_logit = std::max(max_logit, logits[j]);
      }
      float sum_prob = 0.0f;
      std::vector<float> probs(count);
      for (int32_t j = 0; j < count; ++j) {
        const float p = round_bf16(std::exp(logits[j] - max_logit));
        probs[j] = p;
        sum_prob += p;
      }
      float* out_row = c.expected_output.data() + (s * num_heads + h) * head_size;
      for (int64_t d = 0; d < head_size; ++d) {
        float acc = 0.0f;
        for (int32_t j = 0; j < count; ++j) {
          const int32_t tok = c.selected_indices[s * max_selected + j];
          const float v_val = c.value[(static_cast<int64_t>(tok) * kv_num_heads + kh) * head_size + d];
          acc += probs[j] * v_val;
        }
        out_row[d] = sum_prob > 0.0f ? round_bf16(acc / sum_prob) : 0.0f;
      }
    }
  }
  c.expected_present_key.resize(c.key.size());
  c.expected_present_value.resize(c.value.size());
  for (int64_t kh = 0; kh < kv_num_heads; ++kh) {
    for (int64_t s = 0; s < sequence_length; ++s) {
      std::copy_n(c.key.begin() + (s * kv_num_heads + kh) * head_size, head_size,
                  c.expected_present_key.begin() + (kh * sequence_length + s) * head_size);
      std::copy_n(c.value.begin() + (s * kv_num_heads + kh) * head_size, head_size,
                  c.expected_present_value.begin() + (kh * sequence_length + s) * head_size);
    }
  }

  for (const char* prefill_env : {"1", "0"}) {
    SCOPED_TRACE(prefill_env);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_DSA_PREFILL_KERNEL", prefill_env}}};
    RunDynamicSparseAttentionCase<BFloat16>(c, DefaultCudaExecutionProvider());
  }
}

TEST(DynamicSparseAttentionTest, QkRmsNormPartialInterleavedRotary_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.head_size = 16;
  c.cache_sequence_length = 2;
  c.max_selected = 2;
  c.total_sequence_length = 2;
  c.scale = 0.1f;
  c.do_rotary = 1;
  c.rotary_interleaved = 1;
  c.rotary_offset = 8;
  c.rotary_cache_length = 2;
  c.rotary_half_dim = 4;

  c.query.assign(16, 0.0f);
  c.key.assign(16, 0.0f);
  std::fill_n(c.query.begin(), 8, 1.0f);
  std::fill_n(c.key.begin(), 8, 1.0f);
  for (int i = 8; i < 16; i += 2) {
    c.query[i] = 1.0f;
    c.key[i] = 1.0f;
  }
  c.value.assign(16, 3.0f);

  c.past_key.assign(2 * 16, 0.0f);
  for (int i = 8; i < 16; i += 2) {
    c.past_key[i] = 2.0f;
  }
  c.past_value.assign(2 * 16, 0.0f);
  std::fill_n(c.past_value.begin(), 16, 1.0f);

  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.position_ids = {1};
  c.q_norm_weight.assign(16, 1.0f);
  c.k_norm_weight.assign(16, 1.0f);
  c.cos_cache = {1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f};
  c.sin_cache = {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f};

  const float inverse_rms = 1.0f / std::sqrt(0.75f + c.qk_norm_epsilon);
  const float current_logit = c.scale * 12.0f * inverse_rms * inverse_rms;
  const float current_weight = std::exp(current_logit);
  const float expected_value = (1.0f + 3.0f * current_weight) / (1.0f + current_weight);
  c.expected_output.assign(16, expected_value);

  c.expected_present_key = c.past_key;
  for (int i = 0; i < 8; ++i) {
    c.expected_present_key[16 + i] = inverse_rms;
  }
  for (int i = 8; i < 16; i += 2) {
    c.expected_present_key[16 + i] = 0.0f;
    c.expected_present_key[16 + i + 1] = inverse_rms;
  }
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 16,
            c.expected_present_value.end(), 3.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, PartialHalfSplitRotary_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(5.0f);
  c.head_size = 16;
  c.has_past = false;
  c.query.assign(16, 0.0f);
  c.key = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
           1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  c.value.assign(16, 5.0f);
  c.past_key.clear();
  c.past_value.clear();
  c.do_rotary = 1;
  c.rotary_offset = 8;
  c.rotary_cache_length = 1;
  c.rotary_half_dim = 2;
  c.position_ids = {0};
  c.cos_cache = {0.0f, 0.0f};
  c.sin_cache = {1.0f, 1.0f};
  c.expected_output.assign(16, 5.0f);
  c.expected_present_key = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            -3.0f, -4.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyExplicitSinkSharesSoftmax_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(4.0f);
  c.head_sink = {0.0f};
  // The selected value and zero-valued sink have equal logits.
  c.expected_output.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlySmoothSoftmaxAddsImplicitSink_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(4.0f);
  c.smooth_softmax = 1;
  c.expected_output.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, ExplicitZeroScaleProducesUniformLogits_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.cache_sequence_length = 2;
  c.total_sequence_length = 2;
  c.max_selected = 2;
  c.scale = 0.0f;
  c.query.assign(8, 1.0f);
  c.key.assign(8, 1.0f);
  c.value.assign(8, 6.0f);
  c.past_key.assign(16, 0.0f);
  c.past_value.assign(16, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 2.0f);
  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.expected_output.assign(8, 4.0f);
  c.expected_present_key.assign(16, 0.0f);
  std::fill(c.expected_present_key.begin() + 8, c.expected_present_key.end(), 1.0f);
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 8, c.expected_present_value.end(), 6.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, OmittedScaleUsesHeadSizeDefault_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.cache_sequence_length = 2;
  c.total_sequence_length = 2;
  c.max_selected = 2;
  c.has_scale = false;
  c.query.assign(8, 1.0f);
  c.key.assign(8, 1.0f);
  c.value.assign(8, 6.0f);
  c.past_key.assign(16, 0.0f);
  c.past_value.assign(16, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 2.0f);
  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  const float current_weight = std::exp(std::sqrt(8.0f));
  c.expected_output.assign(8, (2.0f + 6.0f * current_weight) / (1.0f + current_weight));
  c.expected_present_key.assign(16, 0.0f);
  std::fill(c.expected_present_key.begin() + 8, c.expected_present_key.end(), 1.0f);
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 8, c.expected_present_value.end(), 6.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, PrefillAndTokenDecodeFixedCapacityCache_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  {
    SCOPED_TRACE("prefill");
    DynamicSparseAttentionCase c;
    c.sequence_length = 2;
    c.cache_sequence_length = 2;
    c.max_selected = 2;
    c.total_sequence_length = 2;
    c.query.assign(2 * 8, 0.0f);
    c.key.assign(2 * 8, 0.0f);
    c.value.insert(c.value.end(), 8, 2.0f);
    c.value.insert(c.value.end(), 8, 4.0f);
    c.past_key.assign(2 * 8, 0.0f);
    c.past_value.assign(2 * 8, 0.0f);
    c.selected_indices = {0, -1,
                          1, -1};
    c.selected_counts = {1, 1};
    c.seqlens_k = {1};
    c.expected_output = c.value;
    c.expected_present_key = c.key;
    c.expected_present_value = c.value;
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider());
  }

  {
    SCOPED_TRACE("token decode");
    DynamicSparseAttentionCase c;
    c.cache_sequence_length = 3;
    c.max_selected = 2;
    c.total_sequence_length = 3;
    c.query.assign(8, 0.0f);
    c.key.assign(8, 0.0f);
    c.value.assign(8, 6.0f);
    c.past_key.assign(3 * 8, 0.0f);
    c.past_value.assign(3 * 8, 0.0f);
    std::fill_n(c.past_value.begin(), 8, 2.0f);
    std::fill_n(c.past_value.begin() + 8, 8, 4.0f);
    c.selected_indices = {0, 2};
    c.selected_counts = {2};
    c.seqlens_k = {2};
    c.expected_output.assign(8, 4.0f);
    c.expected_present_key.assign(3 * 8, 0.0f);
    c.expected_present_value = c.past_value;
    std::fill(c.expected_present_value.begin() + 2 * 8,
              c.expected_present_value.end(), 6.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider());
  }
}

#endif  // USE_CUDA

#if defined(USE_CUDA) || defined(USE_WEBGPU)
namespace {

void RunTokenDecodeWithAliasedCache(std::unique_ptr<IExecutionProvider> ep, bool enable_cuda_graph = false) {
  ASSERT_NE(ep, nullptr);

  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_length = 1;
  constexpr int64_t num_heads = 1;
  constexpr int64_t kv_num_heads = 1;
  constexpr int64_t head_size = 8;
  constexpr int64_t cache_capacity = 2;

  Model model("dynamic_sparse_attention_cache_alias", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto fp16_type;
  fp16_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  ONNX_NAMESPACE::TypeProto int32_type;
  int32_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);

  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("query", &fp16_type),
      &graph.GetOrCreateNodeArg("key", &fp16_type),
      &graph.GetOrCreateNodeArg("value", &fp16_type),
      &graph.GetOrCreateNodeArg("past_key", &fp16_type),
      &graph.GetOrCreateNodeArg("past_value", &fp16_type),
      &empty,
      &empty,
      &graph.GetOrCreateNodeArg("selected_indices", &int32_type),
      &graph.GetOrCreateNodeArg("selected_counts", &int32_type),
      &graph.GetOrCreateNodeArg("seqlens_k", &int32_type),
      &graph.GetOrCreateNodeArg("total_sequence_length", &int32_type),
      &empty,
      &empty,
      &empty,
      &empty,
      &empty,
      &empty};
  std::vector<NodeArg*> outputs{
      &graph.GetOrCreateNodeArg("output", &fp16_type),
      &graph.GetOrCreateNodeArg("present_key", &fp16_type),
      &graph.GetOrCreateNodeArg("present_value", &fp16_type)};
  auto& node = graph.AddNode("dynamic_sparse_attention", "DynamicSparseAttention", "",
                             inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", num_heads);
  node.AddAttribute("kv_num_heads", kv_num_heads);
  node.AddAttribute("scale", 1.0f);
  node.AddAttribute("attention_mode", std::string{"selected_only"});
  node.AddAttribute("selected_kv_source", std::string{"main"});
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  IExecutionProvider* ep_ptr = ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto gpu_allocators = ep_ptr->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu_allocator, gpu_allocators.end());
  auto allocator = session.GetAllocator((*gpu_allocator)->Info());
  ASSERT_NE(allocator, nullptr);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Tensor cpu_tensor(DataTypeImpl::GetType<Element>(), shape,
                      const_cast<Element*>(values.data()), cpu_allocator->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, allocator);
    ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };

  const TensorShape qkv_shape{batch_size, sequence_length, head_size};
  const TensorShape cache_shape{batch_size, kv_num_heads, cache_capacity, head_size};
  auto query_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{0.0f}), qkv_shape);
  auto key_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{2.0f}), qkv_shape);
  auto value_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{5.0f}), qkv_shape);
  auto past_key_value =
      make_gpu_value(std::vector<MLFloat16>(cache_capacity * head_size, MLFloat16{0.0f}), cache_shape);
  std::vector<MLFloat16> past_value_data(cache_capacity * head_size, MLFloat16{0.0f});
  std::fill_n(past_value_data.begin(), head_size, MLFloat16{1.0f});
  auto past_value_value = make_gpu_value(past_value_data, cache_shape);
  auto selected_indices_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1, 1});
  auto selected_counts_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1});
  auto seqlens_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1});

  std::vector<int32_t> total_length_data{2};
  OrtValue total_length_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), TensorShape{},
                       total_length_data.data(), cpu_allocator->Info(), total_length_value);
  Tensor output_tensor(DataTypeImpl::GetType<MLFloat16>(), qkv_shape, allocator);
  OrtValue output_value;
  Tensor::InitOrtValue(std::move(output_tensor), output_value);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("query", query_value));
  ASSERT_STATUS_OK(binding->BindInput("key", key_value));
  ASSERT_STATUS_OK(binding->BindInput("value", value_value));
  ASSERT_STATUS_OK(binding->BindInput("past_key", past_key_value));
  ASSERT_STATUS_OK(binding->BindInput("past_value", past_value_value));
  ASSERT_STATUS_OK(binding->BindInput("selected_indices", selected_indices_value));
  ASSERT_STATUS_OK(binding->BindInput("selected_counts", selected_counts_value));
  ASSERT_STATUS_OK(binding->BindInput("seqlens_k", seqlens_value));
  ASSERT_STATUS_OK(binding->BindInput("total_sequence_length", total_length_value));
  ASSERT_STATUS_OK(binding->BindOutput("output", output_value));
  ASSERT_STATUS_OK(binding->BindOutput("present_key", past_key_value));
  ASSERT_STATUS_OK(binding->BindOutput("present_value", past_value_value));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());
  RunOptions run_options;
  if (enable_cuda_graph) {
    ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
  }
  const int run_count = enable_cuda_graph ? 4 : 1;
  for (int i = 0; i < run_count; ++i) {
    if (enable_cuda_graph && i == run_count - 1) {
      std::vector<MLFloat16> replay_value_data(head_size, MLFloat16{7.0f});
      Tensor replay_value_tensor(DataTypeImpl::GetType<MLFloat16>(), qkv_shape,
                                 replay_value_data.data(), cpu_allocator->Info());
      ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(
          replay_value_tensor, *value_value.GetMutable<Tensor>()));
    }
    ASSERT_STATUS_OK(session.Run(run_options, *binding));
  }
  if (enable_cuda_graph) {
    EXPECT_TRUE(ep_ptr->IsGraphCaptured(1));
  }
  ASSERT_STATUS_OK(binding->SynchronizeOutputs());

  const auto& results = binding->GetOutputs();
  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[1].Get<Tensor>().Data<MLFloat16>(),
            past_key_value.Get<Tensor>().Data<MLFloat16>());
  EXPECT_EQ(results[2].Get<Tensor>().Data<MLFloat16>(),
            past_value_value.Get<Tensor>().Data<MLFloat16>());

  Tensor cpu_output(DataTypeImpl::GetType<MLFloat16>(), qkv_shape, cpu_allocator);
  ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(results[0].Get<Tensor>(), cpu_output));
  const float expected_value = enable_cuda_graph ? 7.0f : 5.0f;
  for (MLFloat16 value : cpu_output.DataAsSpan<MLFloat16>()) {
    EXPECT_FLOAT_EQ(value.ToFloat(), expected_value);
  }
  Tensor cpu_present_key(DataTypeImpl::GetType<MLFloat16>(), cache_shape, cpu_allocator);
  ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(results[1].Get<Tensor>(), cpu_present_key));
  Tensor cpu_present_value(DataTypeImpl::GetType<MLFloat16>(), cache_shape, cpu_allocator);
  ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(results[2].Get<Tensor>(), cpu_present_value));
  for (int64_t i = 0; i < head_size; ++i) {
    EXPECT_FLOAT_EQ(cpu_present_key.Data<MLFloat16>()[i].ToFloat(), 0.0f);
    EXPECT_FLOAT_EQ(cpu_present_key.Data<MLFloat16>()[head_size + i].ToFloat(), 2.0f);
    EXPECT_FLOAT_EQ(cpu_present_value.Data<MLFloat16>()[i].ToFloat(), 1.0f);
    EXPECT_FLOAT_EQ(cpu_present_value.Data<MLFloat16>()[head_size + i].ToFloat(), expected_value);
  }
}

}  // namespace
#endif  // defined(USE_CUDA) || defined(USE_WEBGPU)

#ifdef USE_CUDA

TEST(DynamicSparseAttentionTest, TokenDecodeWithAliasedCache_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  RunTokenDecodeWithAliasedCache(std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, TokenDecodeCudaGraphCaptureAndReplay_CUDA) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION", std::nullopt}}};
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.enable_cuda_graph = true;
  RunTokenDecodeWithAliasedCache(CudaExecutionProviderWithOptions(&provider_options), true);
}

TEST(DynamicSparseAttentionTest, RejectsAuxiliaryInputsInMainMode_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.auxiliary_sequence_length = 1;
  c.auxiliary_key.assign(8, 1.0f);
  c.auxiliary_value.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(
      c, std::move(cuda_ep), OpTester::ExpectResult::kExpectFailure,
      "auxiliary KV inputs are not allowed when selected_kv_source is main");
}

TEST(DynamicSparseAttentionTest, RejectsMissingAuxiliaryInputInAuxiliaryMode_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  RunDynamicSparseAttentionCase(
      c, std::move(cuda_ep), OpTester::ExpectResult::kExpectFailure,
      "auxiliary KV inputs are required when selected_kv_source is auxiliary");
}

TEST(DynamicSparseAttentionTest, NormalPathSkipsDeviceMetadataValidation_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION", std::nullopt}}};

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.selected_counts = {3};
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, RejectsInvalidSelectionMetadata_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION", "1"}}};

  {
    SCOPED_TRACE("selected_counts shape");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_counts = {0, 0};
    c.selected_counts_shape = {2};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected count exceeds row capacity");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_counts = {3};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("non-padding entry after selected count");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {0, -1};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("duplicate selected entry");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {0, 0};
    c.selected_counts = {2};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("duplicate selected entry in large set");
    constexpr int64_t selected_count = 4096;
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.cache_sequence_length = selected_count;
    c.max_selected = selected_count;
    c.total_sequence_length = selected_count;
    c.past_key.assign(selected_count * c.head_size, 0.0f);
    c.past_value.assign(selected_count * c.head_size, 0.0f);
    c.selected_indices.resize(selected_count);
    std::iota(c.selected_indices.begin(), c.selected_indices.end(), 0);
    c.selected_indices.back() = c.selected_indices[c.selected_indices.size() - 2];
    c.selected_counts = {static_cast<int32_t>(selected_count)};
    c.seqlens_k = {static_cast<int32_t>(selected_count - 1)};
    c.expected_present_key = c.past_key;
    c.expected_present_value = c.past_value;
    RunDynamicSparseAttentionCase(
        c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure,
        "contains a duplicate active index");
  }
  {
    SCOPED_TRACE("selected entry outside main sequence");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {1, -1};
    c.selected_counts = {1};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected entry is non-causal");
    DynamicSparseAttentionCase c;
    c.sequence_length = 2;
    c.cache_sequence_length = 2;
    c.max_selected = 1;
    c.total_sequence_length = 2;
    c.query.assign(2 * 8, 0.0f);
    c.key.assign(2 * 8, 0.0f);
    c.value.assign(2 * 8, 1.0f);
    c.past_key.assign(2 * 8, 0.0f);
    c.past_value.assign(2 * 8, 0.0f);
    c.selected_indices = {1, 1};
    c.selected_counts = {1, 1};
    c.seqlens_k = {1};
    c.expected_output.assign(2 * 8, 0.0f);
    c.expected_present_key.assign(2 * 8, 0.0f);
    c.expected_present_value.assign(2 * 8, 1.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure,
                                  "must not refer to a future key");
  }
  {
    SCOPED_TRACE("per-batch length exceeds declared total length");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.cache_sequence_length = 3;
    c.total_sequence_length = 2;
    c.past_key.assign(3 * 8, 0.0f);
    c.past_value.assign(3 * 8, 0.0f);
    c.seqlens_k = {2};
    c.expected_present_key.assign(3 * 8, 0.0f);
    c.expected_present_value.assign(3 * 8, 0.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure,
                                  "incompatible with the current sequence");
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(DynamicSparseAttentionTest, RejectsUnknownModeAndSource_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  {
    SCOPED_TRACE("attention_mode");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.attention_mode = "dense";
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected_kv_source");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_kv_source = "paged";
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
}
#endif

#endif  // USE_CUDA

#ifdef USE_WEBGPU

TEST(DynamicSparseAttentionTest, RejectsCudaValidationOptions_WebGPU) {
  for (bool committed : {false, true}) {
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (!webgpu_ep) GTEST_SKIP() << "WebGPU EP not available.";
    auto c = MakeSingleTokenSelectedOnlyCase();
    if (committed) {
      c.committed_length = 0;
    } else {
      c.expected_validation_status = 0;
    }
    RunDynamicSparseAttentionCase<float>(
        c, std::move(webgpu_ep), OpTester::ExpectResult::kExpectFailure,
        committed ? "committed_length is only supported by CUDA" : "validation_status is only supported by CUDA");
  }
}

TEST(DynamicSparseAttentionTest, TokenDecodeWithAliasedCache_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  RunTokenDecodeWithAliasedCache(std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyGqaCacheAppend_Float16_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.sequence_length = 2;
  c.num_heads = 2;
  c.cache_sequence_length = 3;
  c.max_selected = 3;
  c.total_sequence_length = 3;
  c.query.assign(2 * 2 * 8, 0.0f);
  c.key.assign(2 * 8, 0.0f);
  c.value.insert(c.value.end(), 8, 3.0f);
  c.value.insert(c.value.end(), 8, 5.0f);
  c.past_key.assign(3 * 8, 0.0f);
  c.past_value.assign(3 * 8, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 1.0f);
  c.selected_indices = {0, -1, -1,
                        0, 2, -1};
  c.selected_counts = {1, 2};
  c.seqlens_k = {2};
  c.expected_output.insert(c.expected_output.end(), 2 * 8, 1.0f);
  c.expected_output.insert(c.expected_output.end(), 2 * 8, 3.0f);
  c.expected_present_key.assign(3 * 8, 0.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 1.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 3.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 5.0f);

  RunDynamicSparseAttentionCase(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnly_Float32_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  RunDynamicSparseAttentionCase<float>(MakeSingleTokenSelectedValueCase(), std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlySmoothSoftmax_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(4.0f);
  c.smooth_softmax = 1;
  c.expected_output.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, ZeroWidthSelectionProducesZero_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.max_selected = 0;
  c.selected_indices.clear();
  c.selected_counts = {0};
  RunDynamicSparseAttentionCase(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, LocalPlusSelectedAuxiliaryJointSoftmaxSharedKv_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.auxiliary_kv_shared = 1;
  c.auxiliary_sequence_length = 2;
  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 2.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 4.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 8.0f);
  c.selected_indices = {1};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.head_sink = {0.0f};
  c.expected_output.assign(8, 10.0f / 3.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, RejectsAttentionStageAboveStorageBufferLimit_WebGPU) {
  ConfigOptions config_options;
  ASSERT_STATUS_OK(
      config_options.AddConfigEntry(webgpu::options::kMaxStorageBuffersPerShaderStage, "8"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.auxiliary_kv_shared = 1;
  c.auxiliary_sequence_length = 2;
  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 2.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.auxiliary_key.assign(16, 4.0f);
  c.selected_indices = {1};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.head_sink = {0.0f};
  c.expected_output.assign(8, 0.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(
      c, std::move(webgpu_ep), OpTester::ExpectResult::kExpectFailure,
      "requires 9 storage buffers, but the device supports 8");
}

TEST(DynamicSparseAttentionTest, PaddingZeroAndOutOfRangeSelectionsAreGuarded_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.sequence_length = 3;
  c.cache_sequence_length = 3;
  c.max_selected = 2;
  c.total_sequence_length = 3;
  c.query.assign(3 * 8, 0.0f);
  c.key.assign(3 * 8, 0.0f);
  c.value.insert(c.value.end(), 8, 2.0f);
  c.value.insert(c.value.end(), 8, 4.0f);
  c.value.insert(c.value.end(), 8, 6.0f);
  c.past_key.assign(3 * 8, 0.0f);
  c.past_value.assign(3 * 8, 0.0f);
  c.selected_indices = {0, -1,
                        -1, -1,
                        99, -1};
  c.selected_counts = {1, 0, 1};
  c.seqlens_k = {2};
  c.expected_output.insert(c.expected_output.end(), 8, 2.0f);
  c.expected_output.insert(c.expected_output.end(), 2 * 8, 0.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase<float>(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, PackedQkvSelectedOnly_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.packed_qkv = true;
  c.num_heads = 2;
  c.query.assign(2 * 8, 0.0f);
  c.query.insert(c.query.end(), 8, 0.0f);
  c.query.insert(c.query.end(), 8, 6.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.selected_indices = {0};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.expected_output.assign(2 * 8, 6.0f);
  c.expected_present_key.assign(8, 0.0f);
  c.expected_present_value.assign(8, 6.0f);

  RunDynamicSparseAttentionCase(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, QkRmsNormPartialInterleavedRotary_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.head_size = 16;
  c.cache_sequence_length = 2;
  c.max_selected = 2;
  c.total_sequence_length = 2;
  c.scale = 0.1f;
  c.do_rotary = 1;
  c.rotary_interleaved = 1;
  c.rotary_offset = 8;
  c.rotary_cache_length = 2;
  c.rotary_half_dim = 4;
  c.query.assign(16, 0.0f);
  c.key.assign(16, 0.0f);
  std::fill_n(c.query.begin(), 8, 1.0f);
  std::fill_n(c.key.begin(), 8, 1.0f);
  for (int i = 8; i < 16; i += 2) {
    c.query[i] = 1.0f;
    c.key[i] = 1.0f;
  }
  c.value.assign(16, 3.0f);
  c.past_key.assign(2 * 16, 0.0f);
  for (int i = 8; i < 16; i += 2) {
    c.past_key[i] = 2.0f;
  }
  c.past_value.assign(2 * 16, 0.0f);
  std::fill_n(c.past_value.begin(), 16, 1.0f);
  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.q_norm_weight.assign(16, 1.0f);
  c.k_norm_weight.assign(16, 1.0f);
  c.cos_cache = {1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f};
  c.sin_cache = {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f};
  const float inverse_rms = 1.0f / std::sqrt(0.75f + c.qk_norm_epsilon);
  const float current_logit = c.scale * 12.0f * inverse_rms * inverse_rms;
  const float current_weight = std::exp(current_logit);
  c.expected_output.assign(16, (1.0f + 3.0f * current_weight) / (1.0f + current_weight));
  c.expected_present_key = c.past_key;
  for (int i = 0; i < 8; ++i) {
    c.expected_present_key[16 + i] = inverse_rms;
  }
  for (int i = 8; i < 16; i += 2) {
    c.expected_present_key[16 + i] = 0.0f;
    c.expected_present_key[16 + i + 1] = inverse_rms;
  }
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 16, c.expected_present_value.end(), 3.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, ExplicitPositionIdsApplyToQueryAndNewKey_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.cache_sequence_length = 2;
  c.max_selected = 2;
  c.total_sequence_length = 2;
  c.do_rotary = 1;
  c.rotary_cache_length = 2;
  c.rotary_half_dim = 4;
  c.query.assign(8, 0.0f);
  c.query[0] = 1.0f;
  c.key.assign(8, 0.0f);
  c.key[0] = 1.0f;
  c.value.assign(8, 6.0f);
  c.past_key.assign(2 * 8, 0.0f);
  c.past_key[4] = 1.0f;
  c.past_value.assign(2 * 8, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 2.0f);
  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.cos_cache = {1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f};
  c.sin_cache = {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f};
  c.position_ids = {0};
  const float current_weight = std::exp(1.0f);
  c.expected_output.assign(8, (2.0f + 6.0f * current_weight) / (1.0f + current_weight));
  c.expected_present_key = c.past_key;
  c.expected_present_key[8] = 1.0f;
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 8, c.expected_present_value.end(), 6.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(webgpu_ep));
}

TEST(DynamicSparseAttentionTest, PositionIdsWithNonzeroHighWordAreIgnored_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.cache_sequence_length = 2;
  c.max_selected = 2;
  c.total_sequence_length = 2;
  c.do_rotary = 1;
  c.rotary_cache_length = 2;
  c.rotary_half_dim = 4;
  c.query.assign(8, 0.0f);
  c.query[0] = 1.0f;
  c.key.assign(8, 0.0f);
  c.key[0] = 1.0f;
  c.value.assign(8, 6.0f);
  c.past_key.assign(2 * 8, 0.0f);
  c.past_key[4] = 1.0f;
  c.past_value.assign(2 * 8, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 2.0f);
  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.cos_cache = {1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f};
  c.sin_cache = {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f};
  c.position_ids = {(int64_t{1} << 32) + 1};
  const float current_weight = std::exp(1.0f);
  c.expected_output.assign(8, (2.0f + 6.0f * current_weight) / (1.0f + current_weight));
  c.expected_present_key = c.past_key;
  c.expected_present_key[8] = 1.0f;
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 8, c.expected_present_value.end(), 6.0f);

  RunDynamicSparseAttentionCase<float>(c, std::move(webgpu_ep));
}

#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
