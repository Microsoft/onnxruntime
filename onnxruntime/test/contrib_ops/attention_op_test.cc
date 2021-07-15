// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
enum MaskIndexType {
  kMaskIndexEnd = 0,
  kMaskIndexEndAndStart,
  kMaskRaw,
  kMask3D,
  kMaskDummy,  // Dummy mask with shape [1, 1] or [batch_size, 1]
  kMask4D      // Megatron GPT2 mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
};

static void RunAttentionTest(
    const std::vector<float>& input_data,    // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,  // weights:    [hidden_size, 3 * hidden_size]
    bool is_weights_constant,
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index: [batch_size] or [batch_size, past_sequence_length + sequence_length] or empty
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    bool only_enable_cuda = false,
    bool only_enable_cpu = false,
    std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {}) {
  input_hidden_size = (input_hidden_size == 0 ? hidden_size : input_hidden_size);  // By default, no pruning.

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !is_weights_constant && !only_enable_cpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !only_enable_cuda;

  int head_size = hidden_size / number_of_heads;
  if (enable_cpu || enable_cuda) {
    OpTester tester("Attention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(is_unidirectional ? 1 : 0));

    int32_t matrix_size;
    int32_t output_hidden_size;
    if (qkv_sizes.size() != 0) {
      matrix_size = qkv_sizes[0] + qkv_sizes[1] + qkv_sizes[2];
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
      output_hidden_size = qkv_sizes[2];
    } else {
      matrix_size = 3 * hidden_size;
      output_hidden_size = hidden_size;
    }

    std::vector<int64_t> input_dims = {batch_size, sequence_length, input_hidden_size};
    std::vector<int64_t> weights_dims = {input_hidden_size, matrix_size};
    std::vector<int64_t> bias_dims = {matrix_size};

    std::vector<int64_t> mask_index_dims_1 = {batch_size};
    std::vector<int64_t> mask_index_dims_2 = {2 * batch_size};
    std::vector<int64_t> mask_index_dims_3 = {batch_size, past_sequence_length + sequence_length};
    std::vector<int64_t> mask_index_dims_4 = {batch_size, 1};
    std::vector<int64_t> mask_index_dims_5 = {batch_size, sequence_length, past_sequence_length + sequence_length};
    std::vector<int64_t> mask_index_dims_6 = {batch_size, 1, max_sequence_length, max_sequence_length};
    std::vector<int64_t> mask_index_dims;
    switch (mask_index_type) {
      case kMaskIndexEnd:
        mask_index_dims = mask_index_dims_1;
        break;
      case kMaskIndexEndAndStart:
        mask_index_dims = mask_index_dims_2;
        break;
      case kMaskRaw:
        mask_index_dims = mask_index_dims_3;
        break;
      case kMaskDummy:
        mask_index_dims = mask_index_dims_4;
        break;
      case kMask3D:
        mask_index_dims = mask_index_dims_5;
        break;
      case kMask4D:
        mask_index_dims = mask_index_dims_6;
        break;
      default:
        assert(0);  // shall not reach here.
        break;
    }

    std::vector<int64_t> past_dims = {2, batch_size, number_of_heads, past_sequence_length, head_size};
    std::vector<int64_t> present_dims = {2, batch_size, number_of_heads, past_sequence_length + sequence_length, head_size};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, output_hidden_size};

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data), is_weights_constant);
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data, is_weights_constant);
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (mask_index_data.size() > 0) {  // mask index is optional.
      tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    if (use_past_state) {
      if (use_float16) {
        if (past_sequence_length > 0) {
          tester.AddInput<MLFloat16>("past", past_dims, ToFloat16(*past_data));
        }
        tester.AddOutput<MLFloat16>("present", present_dims, ToFloat16(*present_data));
      } else {
        if (past_sequence_length > 0) {
          tester.AddInput<float>("past", past_dims, *past_data);
        }
        tester.AddOutput<float>("present", present_dims, *present_data);
      }
    }

    if (extra_add_data.size() > 0) {
      if (!use_past_state) {
        tester.AddOptionalInputEdge<float>();
      }
      std::vector<int64_t> extra_add_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
      tester.AddInput<float>("extra_add_qk", extra_add_data_dims, extra_add_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index: [batch_size] or [batch_size, past_sequence_length + sequence_length] or empty
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    bool only_enable_cuda = false,
    bool only_enable_cpu = false,
    const std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {}) {
    RunAttentionTest(input_data, weights_data, false, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads,
                     use_float16, is_unidirectional, use_past_state, past_sequence_length,
                     past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                     only_enable_cuda, only_enable_cpu, qkv_sizes, extra_add_data);
    RunAttentionTest(input_data, weights_data, true, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads,
                     use_float16, is_unidirectional, use_past_state, past_sequence_length,
                     past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                     only_enable_cuda, only_enable_cpu, qkv_sizes, extra_add_data);
}

TEST(AttentionTest, AttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;
  
  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {
      6, 6, 4};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f, 0.5f, 0.1f, 0.4f, 1.6f,
      0.4f, 0.8f, 0.9f, 0.1f
  };

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f, 0.2f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, 
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, true, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.031707365f, 0.053643607f, 0.057394292f, -0.019800574f, 0.075466447f, -0.0034214978f, 0.012995008f, -0.019587509f};

  std::vector<int32_t> qkv_sizes = {
     6, 6, 2};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      0.64932525157928467f, 0.79390722513198853f, 0.64932847023010254f, 0.79375863075256348f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, true, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1ExtraAdd) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> extra_add_qk = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, true, qkv_sizes, extra_add_qk);
}

TEST(AttentionTest, AttentionBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.154296875, 0.1082763671875, 4.25, 5.6484375,
      3.970703125, 0.072998046875, 4.25, 5.6484375};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, true);
}

TEST(AttentionTest, AttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskExceedSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index > sequence_length
  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionNoMaskIndex) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionUnidirectional) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  bool is_unidirectional = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional);
}

TEST(AttentionTest, AttentionEmptyPastState) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  std::vector<float> past_data = {};

  std::vector<float> present_data = {
      0.053175069391727448f, 0.12795503437519073f, 0.11125634610652924f, -0.0510881207883358f, -0.55345797538757324f, -0.3045809268951416f, -0.36920222640037537f, 0.060108467936515808f, 0.28109729290008545f, 0.069518551230430603f, 0.45718482136726379f, -0.010400654748082161f, 0.0038009658455848694f, 0.29213353991508484f, -0.17697516083717346f, 0.27086889743804932f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 0;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch1) {
  int batch_size = 1;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.019333266f, -0.21813886f, 0.16212955f, -0.015626367f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.20141591f, 0.43005896f, 0.35745093f, 0.19957167f};

  std::vector<float> past_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f};

  std::vector<float> present_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, -0.30182117f, -0.12330482f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f, -0.36450946f, -0.19483691f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, -0.027254611f, -0.096526355f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f, -0.025281552f, -0.25482416f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch2) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.26993567f, 0.23553593f, 0.43190649f, 0.086044826f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f, 0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f, 0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch2WithPadding) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // One sequence has both left padding and right padding
  std::vector<int32_t> mask_index_data = {4, 3, 0, 2};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.18029204f, 0.07451740f, 0.73694098f, 0.17766341f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f, 0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f, 0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionBatch2MaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2, 2, 0, 0};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionRightPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1, 0};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionLeftPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionBatch2LeftPaddingMaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 2, 1, 0};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, Attention3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionBatch2AttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionUnidirectional3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      3.967245340f, 0.07324841f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionUnidirectionalAttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      3.967245340f, 0.07324841f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask1DEndNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEnd);
}

TEST(AttentionTest, AttentionMask1DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionMask2DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask3DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionDummyMask2D) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1, 1};

  std::vector<float> output_data = {
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskDummy);
}

TEST(AttentionTest, Attention4DMask) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 4D mask Bx1xmax_Sxmax_S
  std::vector<int32_t> mask_index_data = {
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 1, 1, 0,
      0, 1, 1, 1};

  std::vector<float> output_data = {
      3.97f, 0.073f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  int input_hidden_size = 0;
  int max_sequence_length = 4;
  bool only_enable_cuda = true;  // only support 4D mask in cuda
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, kMask4D, input_hidden_size, max_sequence_length,
                   only_enable_cuda);
}

TEST(AttentionTest, AttentionMaskIndexOutOfRange) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test end_position > sequence length, or start_position < 0
  std::vector<int32_t> mask_index_data = {3, 2, 0, -1};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionPastState_dynamic) {
  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> input_dims{2, 5, 768};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);

  std::vector<int64_t> weight_dims{768, 2304};
  std::vector<float> weight_data = random.Gaussian<float>(weight_dims, 0.0f, 0.3f);

  std::vector<int64_t> bias_dims{2304};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);

  std::vector<int64_t> past_dims{2, 2, 12, 15, 64};
  std::vector<float> past_data = random.Gaussian<float>(past_dims, 0.0f, 0.3f);

  OpTester test("Attention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 12);
  test.AddAttribute<int64_t>("unidirectional", 1);
  test.AddInput<float>("input", input_dims, input_data);
  test.AddInput<float>("weight", weight_dims, weight_data);
  test.AddInput<float>("bias", bias_dims, bias_data);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("past", past_dims, past_data);

  test.AddReferenceOutputs("testdata/attention_past_state.onnx");
  test.Run();
}

TEST(AttentionTest, AttentionPrunedModel) {
  int batch_size = 2;
  int sequence_length = 2;
  // test input_hidden_size > hidden_size
  int input_hidden_size = 6;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f, 0.0f, 1.0f,
      0.8f, -0.5f, 0.0f, 1.f, 2.0f, 3.0f,
      0.8f, -0.5f, 0.0f, 1.f, 4.0f, 5.0f,
      0.5f, 0.2f, 0.3f, -0.6f, 6.0f, 7.0f};

  std::vector<float> weight_data = {
      0.1f,
      -0.2f,
      0.3f,
      1.0f,
      1.1f,
      0.3f,
      0.5f,
      0.2f,
      0.3f,
      -0.6f,
      1.5f,
      2.0f,
      0.5f,
      0.1f,
      0.4f,
      1.6f,
      1.0f,
      2.0f,
      0.4f,
      0.8f,
      0.9f,
      0.1f,
      -1.3f,
      0.7f,
      0.3f,
      0.2f,
      4.0f,
      2.2f,
      1.6f,
      1.1f,
      0.7f,
      0.2f,
      0.4f,
      1.0f,
      1.2f,
      0.5f,
      0.2f,
      0.1f,
      0.4f,
      1.6f,
      2.4f,
      3.3f,
      2.1f,
      4.2f,
      8.4f,
      0.0f,
      2.1f,
      3.2f,
      0.1f,
      0.2f,
      0.3f,
      0.4f,
      0.5f,
      0.6f,
      0.7f,
      0.8f,
      0.9f,
      1.0f,
      1.1f,
      1.2f,
      1.2f,
      1.1f,
      1.0f,
      0.9f,
      0.8f,
      0.7f,
      0.6f,
      0.5f,
      0.4f,
      0.3f,
      0.2f,
      0.1f,
  };

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_data = {1, 1, 1, 1};

  std::vector<float> output_data = {
      11.689527f, 2.769937f, 7.05f, 8.350000f,
      11.690000f, 2.770000f, 7.05f, 8.350000f,
      14.276558f, 5.374159f, 9.650001f, 10.95f,
      14.289073f, 5.370287f, 9.650001f, 10.95f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw, input_hidden_size);
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(AttentionTest, SharedPrepackedWeights) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  OpTester tester("Attention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
  tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(0));

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
  std::vector<int64_t> bias_dims = {3 * hidden_size};

  std::vector<int64_t> mask_index_dims = {2 * batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  tester.AddInput<float>("input", input_dims, input_data);
  tester.AddInput<float>("weight", weights_dims, weight_data, true);  // Trigger pre-packing
  tester.AddInput<float>("bias", bias_dims, bias_data);
  tester.AddOutput<float>("output", output_dims, output_data);
  tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<float>(), TensorShape(weights_dims),
                                           weight_data.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue weight;

  weight.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;

  // Set up weight as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("weight", &weight), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  tester.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      tester.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
