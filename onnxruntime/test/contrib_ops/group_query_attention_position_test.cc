// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"

#if defined(USE_CUDA) && USE_FLASH_ATTENTION
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>

#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

struct PositionCase {
  std::string name;
  int batch_size;
  int sequence_length;
  int rotary_dim;
  bool interleaved;
  bool explicit_positions;
  int shift;
};

std::vector<PositionCase> PositionCases() {
  constexpr std::array<const char*, 3> suffixes{"Shifted", "Unshifted", "Implicit"};
  std::vector<PositionCase> cases;
  for (int batch : {1, 2}) {
    for (int rotary_dim : {96, 128}) {
      for (bool interleaved : {false, true}) {
        for (int mode = 0; mode < 3; ++mode) {
          const std::string name = "DecodeBatch" + std::to_string(batch) +
                                   (rotary_dim == 96 ? "Partial" : "Full") +
                                   (interleaved ? "Interleaved" : "Split") +
                                   suffixes[mode];
          cases.push_back({name, batch, 1, rotary_dim, interleaved, mode != 2, mode == 0 ? 32 : 0});
        }
      }
    }
  }
  // Subsequent prompts support B=1 only. S>1 is not eligible for FlashDecoding.
  for (int mode = 0; mode < 3; ++mode) {
    cases.push_back({std::string("MultiTokenControl") + suffixes[mode],
                     1, 4, 96, false, mode != 2, mode == 0 ? 32 : 0});
  }
  return cases;
}

struct PositionData {
  static constexpr int num_heads = 4;
  static constexpr int kv_num_heads = 2;
  static constexpr int head_size = 128;
  static constexpr int past_length = 8;
  static constexpr int capacity = 32;
  static constexpr int table_length = 128;
  std::vector<MLFloat16> query, key, value, past_key, past_value, cos, sin;
  std::vector<int64_t> positions;

  static std::vector<MLFloat16> Values(size_t count, int seed) {
    std::vector<MLFloat16> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = MLFloat16(static_cast<float>(static_cast<int>((i * 13 + i / 7 + seed) % 33) - 16) / 16.0f);
    }
    return values;
  }

  explicit PositionData(const PositionCase& c)
      : query(Values(c.batch_size * c.sequence_length * num_heads * head_size, 1)),
        key(Values(c.batch_size * c.sequence_length * kv_num_heads * head_size, 7)),
        value(Values(key.size(), 11)),
        past_key(Values(c.batch_size * kv_num_heads * capacity * head_size, 17)),
        past_value(Values(past_key.size(), 23)),
        cos(table_length * c.rotary_dim / 2),
        sin(cos.size()),
        positions(c.batch_size * c.sequence_length) {
    for (int p = 0; p < table_length; ++p) {
      for (int d = 0; d < c.rotary_dim / 2; ++d) {
        const float angle = static_cast<float>(p * (1 + d % 8)) / 64.0f;
        cos[p * (c.rotary_dim / 2) + d] = MLFloat16(std::cos(angle));
        sin[p * (c.rotary_dim / 2) + d] = MLFloat16(std::sin(angle));
      }
    }
    for (int b = 0; b < c.batch_size; ++b) {
      for (int s = 0; s < c.sequence_length; ++s) {
        positions[b * c.sequence_length + s] = past_length + s + c.shift + (c.shift == 0 ? 0 : 16 * b);
      }
    }
  }

  static size_t CacheIndex(int b, int h, int s, int d) {
    return ((b * kv_num_heads + h) * capacity + s) * head_size + d;
  }
};

struct PositionReference {
  std::vector<MLFloat16> key, value;
  std::vector<float> output;
};

// Scalar oracle independent of ORT's attention/rotary implementations. Resident
// keys and RoPE tables stay fixed when positions shift; append offsets do not shift.
PositionReference ReferencePositions(const PositionCase& c, const PositionData& data) {
  using D = PositionData;
  auto rotate = [&](const std::vector<MLFloat16>& input, int heads) {
    auto result = input;
    for (int b = 0; b < c.batch_size; ++b) {
      for (int s = 0; s < c.sequence_length; ++s) {
        const int64_t position = c.explicit_positions ? data.positions[b * c.sequence_length + s] : D::past_length + s;
        for (int h = 0; h < heads; ++h) {
          const size_t base = ((b * c.sequence_length + s) * heads + h) * D::head_size;
          for (int pair = 0; pair < c.rotary_dim / 2; ++pair) {
            const int left = c.interleaved ? 2 * pair : pair;
            const int right = c.interleaved ? left + 1 : pair + c.rotary_dim / 2;
            const size_t index = static_cast<size_t>(position) * (c.rotary_dim / 2) + pair;
            const float x = input[base + left].ToFloat();
            const float y = input[base + right].ToFloat();
            const float cos = data.cos[index].ToFloat();
            const float sin = data.sin[index].ToFloat();
            result[base + left] = MLFloat16(x * cos - y * sin);
            result[base + right] = MLFloat16(x * sin + y * cos);
          }
        }
      }
    }
    return result;
  };
  const auto query = rotate(data.query, D::num_heads);
  const auto key = rotate(data.key, D::kv_num_heads);
  PositionReference result{data.past_key, data.past_value, std::vector<float>(query.size())};
  for (int b = 0; b < c.batch_size; ++b) {
    for (int s = 0; s < c.sequence_length; ++s) {
      for (int h = 0; h < D::kv_num_heads; ++h) {
        for (int d = 0; d < D::head_size; ++d) {
          const size_t src = ((b * c.sequence_length + s) * D::kv_num_heads + h) * D::head_size + d;
          const size_t dst = D::CacheIndex(b, h, D::past_length + s, d);
          result.key[dst] = key[src];
          result.value[dst] = data.value[src];
        }
      }
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(D::head_size));
  for (int b = 0; b < c.batch_size; ++b) {
    for (int s = 0; s < c.sequence_length; ++s) {
      for (int h = 0; h < D::num_heads; ++h) {
        const int kv_head = h / (D::num_heads / D::kv_num_heads);
        const size_t base = ((b * c.sequence_length + s) * D::num_heads + h) * D::head_size;
        // Bottom-right causal masking includes the past and current tokens through s.
        std::vector<float> scores(D::past_length + s + 1, 0.0f);
        for (size_t t = 0; t < scores.size(); ++t) {
          for (int d = 0; d < D::head_size; ++d) {
            scores[t] += query[base + d].ToFloat() *
                         result.key[D::CacheIndex(b, kv_head, static_cast<int>(t), d)].ToFloat();
          }
          scores[t] *= scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (int d = 0; d < D::head_size; ++d) {
          float sum = 0.0f;
          for (size_t t = 0; t < scores.size(); ++t) {
            sum += (scores[t] / denominator) *
                   result.value[D::CacheIndex(b, kv_head, static_cast<int>(t), d)].ToFloat();
          }
          result.output[base + d] = MLFloat16(sum).ToFloat();
        }
      }
    }
  }
  return result;
}

TEST(GroupQueryAttentionPositionTest, ReferenceSanityCpu) {
  using D = PositionData;
  for (auto c : PositionCases()) {
    SCOPED_TRACE(c.name);
    PositionData data(c);
    const auto expected = ReferencePositions(c, data);
    c.explicit_positions = false;
    const auto implicit = ReferencePositions(c, data);
    if (c.shift != 0) {
      float maximum_difference = 0.0f;
      for (size_t i = 0; i < expected.output.size(); ++i) {
        maximum_difference = std::max(maximum_difference, std::abs(expected.output[i] - implicit.output[i]));
      }
      EXPECT_GT(maximum_difference, 0.02f) << "Ignored positions must exceed the GPU comparison tolerance";
    } else {
      EXPECT_EQ(expected.output, implicit.output);
      for (size_t i = 0; i < expected.key.size(); ++i) {
        EXPECT_EQ(expected.key[i].val, implicit.key[i].val);
      }
    }
    c.explicit_positions = true;
    std::fill(data.query.begin(), data.query.end(), MLFloat16(0.0f));
    // Zero queries give uniform attention; a quarter turn gives exact signed swaps.
    std::fill(data.cos.begin(), data.cos.end(), MLFloat16(0.0f));
    std::fill(data.sin.begin(), data.sin.end(), MLFloat16(1.0f));
    const auto uniform = ReferencePositions(c, data);
    for (int b = 0; b < c.batch_size; ++b) {
      for (int s = 0; s < c.sequence_length; ++s) {
        for (int h = 0; h < D::num_heads; ++h) {
          const int kh = h / (D::num_heads / D::kv_num_heads);
          for (int d = 0; d < D::head_size; ++d) {
            float sum = 0.0f;
            for (int t = 0; t < D::past_length; ++t) {
              sum += data.past_value[D::CacheIndex(b, kh, t, d)].ToFloat();
            }
            for (int t = 0; t <= s; ++t) {
              sum += data.value[((b * c.sequence_length + t) * D::kv_num_heads + kh) * D::head_size + d].ToFloat();
            }
            const size_t i = ((b * c.sequence_length + s) * D::num_heads + h) * D::head_size + d;
            EXPECT_NEAR(uniform.output[i], MLFloat16(sum / (D::past_length + s + 1)).ToFloat(), 0.00025f);
          }
        }
        for (int h = 0; h < D::kv_num_heads; ++h) {
          for (int d = 0; d < D::head_size; ++d) {
            int partner = d;
            float sign = 1.0f;
            if (d < c.rotary_dim) {
              const bool left = c.interleaved ? d % 2 == 0 : d < c.rotary_dim / 2;
              partner = c.interleaved ? (d ^ 1) : (d + c.rotary_dim / 2) % c.rotary_dim;
              sign = left ? -1.0f : 1.0f;
            }
            const size_t src = ((b * c.sequence_length + s) * D::kv_num_heads + h) * D::head_size + partner;
            EXPECT_EQ(uniform.key[D::CacheIndex(b, h, D::past_length + s, d)].ToFloat(), sign * data.key[src].ToFloat());
          }
        }
      }
    }
  }
}

#if defined(USE_CUDA) && USE_FLASH_ATTENTION
class GqaSharedCachePositionTest : public ::testing::TestWithParam<PositionCase> {};

TEST_P(GqaSharedCachePositionTest, MatchesReference) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "FlashAttention requires CUDA SM80 or later";
  }
  // Select Flash over XQA/cuDNN, without adding runtime observation or changing global defaults.
  ScopedEnvironmentVariables selected_backend{{
      {"ORT_DISABLE_FLASH_ATTENTION", "0"},
      {"ORT_DISABLE_FLASH_DECODE", "0"},
      {"ORT_ENABLE_XQA", "0"},
      {"ORT_ENABLE_CUDNN_FLASH_ATTENTION", "0"},
  }};
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  using D = PositionData;
  const auto& c = GetParam();
  const PositionData data(c);
  const auto expected = ReferencePositions(c, data);
  Model model("gqa_shared_cache_positions", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto fp16_type, int32_type, int64_type;
  fp16_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  int32_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  int64_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  std::vector<NodeArg*> inputs;
  for (const char* name : {"query", "key", "value", "past_key", "past_value"}) {
    inputs.push_back(&graph.GetOrCreateNodeArg(name, &fp16_type));
  }
  inputs.push_back(&graph.GetOrCreateNodeArg("seqlens_k", &int32_type));
  inputs.push_back(&graph.GetOrCreateNodeArg("total_sequence_length", &int32_type));
  inputs.push_back(&graph.GetOrCreateNodeArg("cos_cache", &fp16_type));
  inputs.push_back(&graph.GetOrCreateNodeArg("sin_cache", &fp16_type));
  if (c.explicit_positions) {
    inputs.push_back(&graph.GetOrCreateNodeArg("position_ids", &int64_type));
  }
  std::vector<NodeArg*> outputs;
  for (const char* name : {"output", "present_key", "present_value"}) {
    outputs.push_back(&graph.GetOrCreateNodeArg(name, &fp16_type));
  }
  auto& node = graph.AddNode(c.name, "GroupQueryAttention", "", inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", int64_t{D::num_heads});
  node.AddAttribute("kv_num_heads", int64_t{D::kv_num_heads});
  node.AddAttribute("do_rotary", int64_t{1});
  node.AddAttribute("rotary_interleaved", int64_t{c.interleaved});
  ASSERT_STATUS_OK(graph.Resolve());
  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSession session(options, GetEnvironment());
  IExecutionProvider* ep = cuda_ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());
  auto gpu_allocators = ep->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
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
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };
  const TensorShape query_shape{c.batch_size, c.sequence_length, D::num_heads * D::head_size};
  const TensorShape kv_shape{c.batch_size, c.sequence_length, D::kv_num_heads * D::head_size};
  const TensorShape cache_shape{c.batch_size, D::kv_num_heads, D::capacity, D::head_size};
  const TensorShape table_shape{D::table_length, c.rotary_dim / 2};
  auto past_key = make_gpu_value(data.past_key, cache_shape);
  auto past_value = make_gpu_value(data.past_value, cache_shape);
  auto output = make_gpu_value(std::vector<MLFloat16>(data.query.size()), query_shape);
  std::vector<int32_t> total_data{D::past_length + c.sequence_length};
  OrtValue total;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), TensorShape{1}, total_data.data(),
                       cpu_allocator->Info(), total);
  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("query", make_gpu_value(data.query, query_shape)));
  ASSERT_STATUS_OK(binding->BindInput("key", make_gpu_value(data.key, kv_shape)));
  ASSERT_STATUS_OK(binding->BindInput("value", make_gpu_value(data.value, kv_shape)));
  ASSERT_STATUS_OK(binding->BindInput("past_key", past_key));
  ASSERT_STATUS_OK(binding->BindInput("past_value", past_value));
  auto seqlens = make_gpu_value(std::vector<int32_t>(c.batch_size, total_data[0] - 1), {c.batch_size});
  ASSERT_STATUS_OK(binding->BindInput("seqlens_k", seqlens));
  ASSERT_STATUS_OK(binding->BindInput("total_sequence_length", total));
  ASSERT_STATUS_OK(binding->BindInput("cos_cache", make_gpu_value(data.cos, table_shape)));
  ASSERT_STATUS_OK(binding->BindInput("sin_cache", make_gpu_value(data.sin, table_shape)));
  if (c.explicit_positions) {
    auto positions = make_gpu_value(data.positions, {c.batch_size, c.sequence_length});
    ASSERT_STATUS_OK(binding->BindInput("position_ids", positions));
  }
  ASSERT_STATUS_OK(binding->BindOutput("output", output));
  // Exact OrtValue reuse guarantees aliasing, rather than relying on the memory planner.
  ASSERT_STATUS_OK(binding->BindOutput("present_key", past_key));
  ASSERT_STATUS_OK(binding->BindOutput("present_value", past_value));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());
  ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
  ASSERT_STATUS_OK(binding->SynchronizeOutputs());
  const auto& fetched = binding->GetOutputs();
  ASSERT_EQ(fetched.size(), 3u);
  ASSERT_EQ(fetched[1].Get<Tensor>().DataRaw(), past_key.Get<Tensor>().DataRaw());
  ASSERT_EQ(fetched[2].Get<Tensor>().DataRaw(), past_value.Get<Tensor>().DataRaw());
  std::vector<std::vector<MLFloat16>> actual;
  for (const auto& result : fetched) {
    const auto& gpu_tensor = result.Get<Tensor>();
    Tensor cpu_tensor(DataTypeImpl::GetType<MLFloat16>(), gpu_tensor.Shape(), cpu_allocator);
    ASSERT_STATUS_OK(ep->GetDataTransfer()->CopyTensor(gpu_tensor, cpu_tensor));
    const auto values = cpu_tensor.DataAsSpan<MLFloat16>();
    actual.emplace_back(values.begin(), values.end());
  }
  ASSERT_EQ(actual[0].size(), expected.output.size());
  ASSERT_EQ(actual[1].size(), expected.key.size());
  ASSERT_EQ(actual[2].size(), expected.value.size());
  for (size_t i = 0; i < expected.output.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[0][i].ToFloat())) << "attention index=" << i;
    EXPECT_NEAR(actual[0][i].ToFloat(), expected.output[i], 0.002f) << "attention index=" << i;
  }
  for (int b = 0; b < c.batch_size; ++b) {
    for (int h = 0; h < D::kv_num_heads; ++h) {
      for (int s = 0; s < D::capacity; ++s) {
        for (int d = 0; d < D::head_size; ++d) {
          const size_t i = D::CacheIndex(b, h, s, d);
          EXPECT_EQ(actual[1][i].val, expected.key[i].val) << "K index=" << i;
          EXPECT_EQ(actual[2][i].val, expected.value[i].val) << "V index=" << i;
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GqaSharedCachePositions, GqaSharedCachePositionTest,
                         ::testing::ValuesIn(PositionCases()),
                         [](const ::testing::TestParamInfo<PositionCase>& info) { return info.param.name; });
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
