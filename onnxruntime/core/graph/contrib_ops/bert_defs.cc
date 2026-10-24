// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/graph/contrib_ops/shape_inference_functions.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/sparse/sparse_attention_indexer_common.h"
#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif
using namespace ::ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace defs::math::utils {
void MatMulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);
}  // namespace defs::math::utils
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {

void DecoderAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 2) {  // has new_key_cache and new_value_cache outputs; a pair, so present only when > 2
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 2);
  }
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    updateOutputShape(ctx, 0, query_shape);
  }
  if (ctx.getNumOutputs() > 2) {  // has new_key_cache and new_value_cache outputs; a pair, so present only when > 2
    if (hasInputShape(ctx, 6) && hasInputShape(ctx, 7)) {
      auto& cache_shape = getInputShape(ctx, 6);
      auto& cache_dims = cache_shape.dim();
      if (cache_dims.size() != 4) {
        fail_shape_inference("key and value cache shall be 4 dimensions");
      }
      // has_dim_value() will return false if value is dynamic
      if (cache_dims[0].has_dim_value() &&
          cache_dims[1].has_dim_value() &&
          cache_dims[2].has_dim_value() &&
          cache_dims[3].has_dim_value()) {
        ONNX_NAMESPACE::TensorShapeProto new_cache_shape;
        *new_cache_shape.add_dim() = cache_shape.dim(0);
        *new_cache_shape.add_dim() = cache_shape.dim(1);
        new_cache_shape.add_dim();
        *new_cache_shape.add_dim() = cache_shape.dim(3);

        updateOutputShape(ctx, 1, new_cache_shape);
        updateOutputShape(ctx, 2, new_cache_shape);
      }
    }
  }
}

void RemovePaddingTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Input 0: (batch_size, sequence_length, hidden_size)
  // Output 0: (total_tokens, hidden_size)
  // Output 1: (batch_size, sequence_length)
  // Output 2: (batch_size + 1)
  // Output 3: (1)
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 1);

  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim().size() != 3) {
      fail_shape_inference("input shall be 3 dimensions");
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    output_shape.add_dim();
    *output_shape.add_dim() = input_shape.dim(2);
    updateOutputShape(ctx, 0, output_shape);

    ONNX_NAMESPACE::TensorShapeProto token_offset_shape;
    *token_offset_shape.add_dim() = input_shape.dim(0);
    *token_offset_shape.add_dim() = input_shape.dim(1);
    updateOutputShape(ctx, 1, token_offset_shape);

    ONNX_NAMESPACE::TensorShapeProto cumulated_seq_len_shape;
    auto dim = cumulated_seq_len_shape.add_dim();
    if (input_shape.dim(0).has_dim_value()) {
      dim->set_dim_value(1 + input_shape.dim(0).dim_value());
    }
    updateOutputShape(ctx, 2, cumulated_seq_len_shape);

    ONNX_NAMESPACE::TensorShapeProto max_seq_len_shape;
    max_seq_len_shape.add_dim()->set_dim_value(1);
    updateOutputShape(ctx, 3, max_seq_len_shape);
  }
}

void RestorePaddingTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Input 0:  (total_tokens, hidden_size)
  // Input 1:  (batch_size, sequence_length)
  // Output 0: (batch_size, sequence_length, hidden_size)
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
    auto& input_shape = getInputShape(ctx, 0);
    auto& token_offset_shape = getInputShape(ctx, 1);

    if (input_shape.dim().size() != 2) {
      fail_shape_inference("input shall be 2 dimensions");
    }

    if (token_offset_shape.dim().size() != 2) {
      fail_shape_inference("token_offset shall be 2 dimensions");
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    *output_shape.add_dim() = token_offset_shape.dim(0);
    *output_shape.add_dim() = token_offset_shape.dim(1);
    *output_shape.add_dim() = input_shape.dim(1);
    updateOutputShape(ctx, 0, output_shape);
  }
}

void MultiHeadAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx,
                                             int past_key_index,
                                             bool dmmha_packing = false) {
  // Output 0 has shape (batch_size, sequence_length, v_hidden_size)

  // Q, K and V without packing:
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (key) has shape (batch_size, kv_sequence_length, hidden_size)
  //   Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size)

  // Q, K and V without packing and past (cross attention):
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (key) has shape (batch_size, num_head, kv_sequence_length, head_size)
  //   Input 2 (value) has shape (batch_size, num_head, kv_sequence_length, head_size)

  // Packed KV:
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (batch_size, kv_sequence_length, num_heads, 2, head_size)
  //   Input 2  nullptr

  // Packed QKV:
  //   Input 0 (batch_size, sequence_length, num_heads, 3, head_size) or
  //           (batch_size, sequence_length, 3 * hidden_size))
  //           for DecoderMaskedMultiHeadAttention.
  //   Input 1  nullptr
  //   Input 2  nullptr

  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  int64_t sequence_length = 0;
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 3 && query_dims.size() != 5) {
      fail_shape_inference("Inputs 0 (query) shall be 3 or 5 dimensions");
    }

    if (query_dims.size() == 5) {  // packed QKV
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      *output_shape.add_dim() = query_dims[2] * query_dims[4];
      updateOutputShape(ctx, 0, output_shape);
    } else if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
      if (value_dims.size() != 3 && value_dims.size() != 4) {
        fail_shape_inference("Inputs 2 (value) shall be 3 or 4 dimensions");
      }

      if (value_dims.size() == 3) {
        sequence_length = value_dims[1].dim_value();
      }

      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      *output_shape.add_dim() = value_dims.size() == 3
                                    ? (dmmha_packing ? value_dims[2] / 3 : value_dims[2])
                                    : value_dims[1] * value_dims[3];
      updateOutputShape(ctx, 0, output_shape);
    } else if (hasInputShape(ctx, 1)) {
      auto& key_shape = getInputShape(ctx, 1);
      if (key_shape.dim().size() == 5) {  // packed KV
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
      }
    }
  }

  if (ctx.getNumOutputs() > 2) {  // has present_key and present_value outputs; a pair, so present only when > 2
    if (hasInputShape(ctx, past_key_index)) {
      auto& past_shape = getInputShape(ctx, past_key_index);
      auto& past_dims = past_shape.dim();
      if (past_dims.size() != 4) {
        fail_shape_inference("The past_key input shall be 4 dimensions");
      }

      auto past_present_share_buffer = getAttribute(ctx, "past_present_share_buffer", 0);
      bool mha_buffer_sharing = hasInputShape(ctx, 6) && hasInputShape(ctx, 8);  // equal to MHA op's definition for past_present_share_buffer
      if (past_present_share_buffer || mha_buffer_sharing) {
        propagateElemTypeFromInputToOutput(ctx, past_key_index, 1);
        propagateElemTypeFromInputToOutput(ctx, static_cast<size_t>(past_key_index) + 1, 2);
      } else {
        if (sequence_length > 0 && past_dims[2].has_dim_value()) {
          int64_t total_sequence_length = sequence_length + past_dims[2].dim_value();

          ONNX_NAMESPACE::TensorShapeProto present_shape;
          for (auto& dim : past_dims) {
            *present_shape.add_dim() = dim;
          }
          present_shape.mutable_dim(2)->set_dim_value(total_sequence_length);

          updateOutputShape(ctx, 1, present_shape);
          updateOutputShape(ctx, 2, present_shape);
        }
      }
    }
  }
}

// Type and shape inference for group query attention and sparse attention.
void BaseGroupQueryAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx,
                                                  int past_key_index = -1,
                                                  int use_max_past_present_buffer = -1,
                                                  int output_qk_index = -1,
                                                  int total_sequence_length_index = -1) {
  // Type inference for outputs
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);  // output

  if (ctx.getNumOutputs() >= 3) {  // has present output
    const auto* past_key_type = ctx.getInputType(past_key_index);
    if (past_key_type != nullptr) {
      // present_key and present_value have the same type as past_key/past_value.
      // This allows them to be int8 or packed uint8 when quantization is enabled.
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, past_key_index, 1);      // present_key
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, past_key_index + 1, 2);  // present_value
    } else {
      // If no past state, present is the same type as query.
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 2);
    }
  }

  int64_t kv_sequence_length = -1;
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 3) {
      fail_shape_inference("Inputs 0 (query) shall be 3 dimensions");
    }

    if (ctx.hasInput(2)) {
      //   Input 0 (query) has shape (batch_size, sequence_length, num_heads * head_size)
      //   Input 1 (key) has shape (batch_size, kv_sequence_length, kv_num_heads * head_size)
      //   Input 2 (value) has shape (batch_size, kv_sequence_length, kv_num_heads * head_size)
      //   Output 0 has shape (batch_size, sequence_length, num_heads * head_size)
      ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 0);

      if (hasInputShape(ctx, 2)) {
        auto& value_shape = getInputShape(ctx, 2);
        auto& value_dims = value_shape.dim();
        if (value_dims.size() == 3 && value_dims[1].has_dim_value()) {
          kv_sequence_length = value_dims[1].dim_value();
        }
      }
    } else {
      // Packed QKV:
      //   Input 0 (query) has shape (batch_size, sequence_length, (num_heads + 2 * kv_num_heads) * head_size)
      //   Input 1 (key) is not present
      //   Input 2 (value) is not present
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      int64_t num_heads = getAttribute(ctx, "num_heads", 0);
      int64_t kv_num_heads = getAttribute(ctx, "kv_num_heads", 0);
      if (num_heads <= 0 || kv_num_heads <= 0) {
        fail_shape_inference("num_heads and kv_num_heads must be positive.");
      }
      if (kv_num_heads > (std::numeric_limits<int64_t>::max() - num_heads) / 2) {
        fail_shape_inference("num_heads + 2 * kv_num_heads must not overflow.");
      }

      const int64_t grouped_heads = num_heads + 2 * kv_num_heads;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      auto* output_hidden_size = output_shape.add_dim();
      if (query_dims[2].has_dim_value()) {
        const int64_t hidden_size = query_dims[2].dim_value();
        if (hidden_size % grouped_heads != 0) {
          fail_shape_inference("Packed query hidden size must be divisible by the grouped head count.");
        }
        output_hidden_size->set_dim_value((hidden_size / grouped_heads) * num_heads);
      }
      updateOutputShape(ctx, 0, output_shape);

      if (query_dims[1].has_dim_value()) {
        kv_sequence_length = query_dims[1].dim_value();
      }
    }
  }

  if (ctx.getNumOutputs() >= 3) {  // has present output
    int64_t total_sequence_length_value = 0;
    const auto* total_sequence_length_data =
        total_sequence_length_index >= 0 ? ctx.getInputData(total_sequence_length_index) : nullptr;
    if (total_sequence_length_data != nullptr) {
      const auto& data = ParseData<int32_t>(total_sequence_length_data);
      if (data.size() != 1) {
        fail_shape_inference("total_sequence_length input must contain a single element");
      }
      total_sequence_length_value = static_cast<int64_t>(data[0]);
    }

    if (past_key_index >= 0 && hasInputShape(ctx, past_key_index)) {
      auto& past_shape = getInputShape(ctx, past_key_index);
      auto& past_dims = past_shape.dim();

      // past key has shape (batch_size, kv_num_heads, max_cache_sequence_length, head_size)
      if (past_dims.size() != 4) {
        fail_shape_inference("The past_key input shall be 4 dimensions");
      }

      if (use_max_past_present_buffer == 1) {
        // When past and present use max buffer, they have the same shape
        ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, past_key_index, 1);
        ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, static_cast<size_t>(past_key_index) + 1, 2);
      } else if (use_max_past_present_buffer == 0) {
        if (kv_sequence_length > 0 && past_dims[2].has_dim_value()) {
          const int64_t present_sequence_length = kv_sequence_length + past_dims[2].dim_value();

          ONNX_NAMESPACE::TensorShapeProto present_shape;
          for (auto& dim : past_dims) {
            *present_shape.add_dim() = dim;
          }

          // shape of present key/value is (batch_size, kv_num_heads, present_sequence_length, head_size)
          present_shape.mutable_dim(2)->set_dim_value(present_sequence_length);

          updateOutputShape(ctx, 1, present_shape);
          updateOutputShape(ctx, 2, present_shape);
        }
      } else if (use_max_past_present_buffer == -1) {
        // shape of present key/value is (batch_size, kv_num_heads, present_sequence_length, head_size)
        ONNX_NAMESPACE::TensorShapeProto present_shape;
        *present_shape.add_dim() = past_dims[0];  // batch_size
        *present_shape.add_dim() = past_dims[1];  // kv_num_heads
        if (total_sequence_length_value > 0 && past_dims[2].has_dim_value()) {
          // present_sequence_length = max(past_sequence_length, total_sequence_length)
          const int64_t present_sequence_length = total_sequence_length_value > past_dims[2].dim_value()
                                                      ? total_sequence_length_value
                                                      : past_dims[2].dim_value();
          present_shape.add_dim()->set_dim_value(present_sequence_length);
        } else {
          // total_sequence_length is a runtime input here, so neither a symbolic nor a fixed past
          // length bounds present. Copying the past dimension would let the allocation planner give
          // present a past-sized buffer ("Shape mismatch attempting to re-use buffer").
          present_shape.add_dim();
        }
        *present_shape.add_dim() = past_dims[3];  // head_size

        updateOutputShape(ctx, 1, present_shape);
        updateOutputShape(ctx, 2, present_shape);
      }

      if (output_qk_index >= 0) {
        // An output is considered "supplied" only if it's present AND has a meaningful type definition.
        // An empty string placeholder for an optional output will not have a tensor type proto.
        bool did_supply_qk_buffer = false;
        if (ctx.hasOutput(output_qk_index)) {
          // The output is considered "supplied" if it is present in the node.
          // Note: TypeProto might not be fully populated yet during initial inference.
          did_supply_qk_buffer = true;
        }

        const int64_t qk_output_type = getAttribute(ctx, "qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT));

        if (qk_output_type == static_cast<int64_t>(QKOutputType::NO_OUTPUT) && did_supply_qk_buffer) {
          fail_shape_inference("Output QK buffer was provided but qk_output attribute was not configured");
        }

        if (qk_output_type != static_cast<int64_t>(QKOutputType::NO_OUTPUT) && !did_supply_qk_buffer) {
          fail_shape_inference("Output QK buffer was not provided but qk_output attribute was configured");
        }

        int64_t num_heads = getAttribute(ctx, "num_heads", 0);
        if (did_supply_qk_buffer && hasInputShape(ctx, 0) && total_sequence_length_value > 0 && num_heads > 0) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, output_qk_index);

          auto& query_shape = getInputShape(ctx, 0);
          auto& query_dims = query_shape.dim();

          if (query_dims[0].has_dim_value() && query_dims[1].has_dim_value()) {
            ONNX_NAMESPACE::TensorShapeProto output_qk_shape;
            *output_qk_shape.add_dim() = query_dims[0];                             // batch_size
            output_qk_shape.add_dim()->set_dim_value(num_heads);                    // num_heads
            *output_qk_shape.add_dim() = query_dims[1];                             // sequence_length
            output_qk_shape.add_dim()->set_dim_value(total_sequence_length_value);  // total_sequence_length
            updateOutputShape(ctx, output_qk_index, output_qk_shape);
          }
        }
      }
    } else if (hasInputShape(ctx, 0)) {
      // Handle the case when past_key/past_value is not provided (first token/prefill mode).
      // We still need to infer present_key/present_value output shapes from query and attributes.
      auto& query_shape = getInputShape(ctx, 0);
      auto& query_dims = query_shape.dim();

      int64_t num_heads = getAttribute(ctx, "num_heads", 0);
      int64_t kv_num_heads = getAttribute(ctx, "kv_num_heads", 0);

      if (num_heads > 0 && kv_num_heads > 0 && query_dims.size() == 3 && query_dims[2].has_dim_value()) {
        int64_t hidden_size = query_dims[2].dim_value();
        int64_t head_size = 0;

        if (ctx.hasInput(2)) {
          // query shape is (batch_size, sequence_length, num_heads * head_size)
          head_size = hidden_size / num_heads;
        } else {
          // Packed QKV: query shape is (batch_size, sequence_length, (num_heads + 2 * kv_num_heads) * head_size)
          head_size = hidden_size / (num_heads + 2 * kv_num_heads);
        }

        if (head_size > 0) {
          // Determine present_sequence_length from total_sequence_length or kv_sequence_length
          int64_t present_sequence_length = 0;
          if (total_sequence_length_value > 0) {
            present_sequence_length = total_sequence_length_value;
          } else if (kv_sequence_length > 0) {
            present_sequence_length = kv_sequence_length;
          }

          // present key/value shape is (batch_size, kv_num_heads, present_sequence_length, head_size)
          ONNX_NAMESPACE::TensorShapeProto present_shape;
          *present_shape.add_dim() = query_dims[0];  // batch_size
          present_shape.add_dim()->set_dim_value(kv_num_heads);
          if (present_sequence_length > 0) {
            present_shape.add_dim()->set_dim_value(present_sequence_length);
          } else {
            // Fallback: use query sequence_length (dim 1) as present_sequence_length for prefill
            *present_shape.add_dim() = query_dims[1];
          }
          present_shape.add_dim()->set_dim_value(head_size);

          updateOutputShape(ctx, 1, present_shape);
          updateOutputShape(ctx, 2, present_shape);
        }
      }
    }
  }
}

void GroupQueryAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int past_key_index, int qk_output_index) {
  // TODO(aciddelgado): propagate output shapes depending if kv-share buffer is on or not
  constexpr int use_max_past_present_buffer = -1;
  // With a windowed (sliding_window_cache) KV cache, the bound past/present buffer is the cache
  // capacity C, which is deliberately smaller than total_sequence_length. present therefore keeps
  // the past buffer's own sequence dimension instead of growing with the total sequence length.
  const int64_t sliding_window_cache = getAttribute(ctx, "sliding_window_cache", 0);
  constexpr int total_sequence_length_index = 6;
  BaseGroupQueryAttentionTypeAndShapeInference(
      ctx, past_key_index, sliding_window_cache == 1 ? 1 : use_max_past_present_buffer, qk_output_index,
      total_sequence_length_index);
}

void DynamicSparseAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const int64_t num_heads = getAttribute(ctx, "num_heads", 0);
  const int64_t kv_num_heads = getAttribute(ctx, "kv_num_heads", 0);
  if (num_heads <= 0 || num_heads > std::numeric_limits<int>::max()) {
    fail_shape_inference("num_heads must be a positive int");
  }
  if (kv_num_heads <= 0 || kv_num_heads > std::numeric_limits<int>::max() ||
      num_heads % kv_num_heads != 0) {
    fail_shape_inference("kv_num_heads must be positive and divide num_heads");
  }

  if (hasInputShape(ctx, 0)) {
    const auto& query_shape = getInputShape(ctx, 0);
    const auto& query_dims = query_shape.dim();
    if (query_dims.size() != 3) {
      fail_shape_inference("The query input shall be 3 dimensions");
    }

    auto output_shape = query_shape;
    if (ctx.getInputType(2) == nullptr) {
      output_shape.mutable_dim(2)->Clear();
      if (query_dims[2].has_dim_value()) {
        const int64_t packed_heads = num_heads + 2 * kv_num_heads;
        const int64_t packed_hidden_size = query_dims[2].dim_value();
        if (packed_hidden_size % packed_heads != 0) {
          fail_shape_inference("Packed QKV hidden size must be divisible by num_heads + 2 * kv_num_heads");
        }
        output_shape.mutable_dim(2)->set_dim_value(num_heads * (packed_hidden_size / packed_heads));
      }
    } else if (query_dims[2].has_dim_value() && query_dims[2].dim_value() % num_heads != 0) {
      fail_shape_inference("Query hidden size must be divisible by the number of query heads");
    }
    updateOutputShape(ctx, 0, output_shape);
  }

  for (int output_index = 1; output_index <= 2; ++output_index) {
    const int input_index = output_index + 2;
    if (ctx.getNumOutputs() <= static_cast<size_t>(output_index) || !ctx.hasOutput(output_index)) {
      continue;
    }

    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(
        ctx, ctx.getInputType(input_index) != nullptr ? input_index : 0, output_index);
    if (hasInputShape(ctx, input_index)) {
      ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, input_index, output_index);
      continue;
    }

    if (hasInputShape(ctx, 0)) {
      const auto& query_dims = getInputShape(ctx, 0).dim();
      if (query_dims.size() == 3 && query_dims[2].has_dim_value()) {
        const bool is_packed = ctx.getInputType(2) == nullptr;
        const int64_t divisor = is_packed ? num_heads + 2 * kv_num_heads : num_heads;
        if (query_dims[2].dim_value() % divisor != 0) {
          fail_shape_inference("Query hidden size must be divisible by the number of query heads");
        }
        const int64_t head_size = query_dims[2].dim_value() / divisor;
        ONNX_NAMESPACE::TensorShapeProto present_shape;
        *present_shape.add_dim() = query_dims[0];
        present_shape.add_dim()->set_dim_value(kv_num_heads);
        const bool past_inputs_omitted = ctx.getInputType(3) == nullptr && ctx.getInputType(4) == nullptr;
        const auto* total_length_data = ctx.getInputData(10);
        if (past_inputs_omitted && total_length_data != nullptr) {
          const auto total_lengths = ParseData<int32_t>(total_length_data);
          if (total_lengths.size() != 1) {
            fail_shape_inference("total_sequence_length must contain exactly one element");
          }
          present_shape.add_dim()->set_dim_value(total_lengths[0]);
        } else {
          present_shape.add_dim();
        }
        present_shape.add_dim()->set_dim_value(head_size);
        updateOutputShape(ctx, output_index, present_shape);
      }
    }
  }
}

void SparseAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int past_key_index) {
  constexpr int use_max_past_present_buffer = 1;
  constexpr int qk_output_index = -1;
  constexpr int total_sequence_length_index = 7;
  BaseGroupQueryAttentionTypeAndShapeInference(ctx, past_key_index, use_max_past_present_buffer, qk_output_index,
                                               total_sequence_length_index);
}

constexpr const char* DynamicSparseAttention_ver1_doc = R"DOC(
Model-neutral sparse grouped-query attention with a contiguous main KV cache.

`selected_indices` and `selected_counts` are produced by an external selector. The operator never computes selection
scores or TopK indices. The first `selected_counts[q]` entries in each selected-index row are valid; remaining entries
must be -1. Valid entries must be unique, non-negative request-local positions in the selected source.

`attention_mode="selected_only"` attends only to selected entries. `attention_mode="local_plus_selected"` jointly
normalizes causally valid main-cache entries in `local_window_size` and selected auxiliary entries. An optional
per-query-head sink can participate in either mode; it contributes to the shared softmax denominator but has no value
vector. A row with no entries and no sink produces zero output.

Supported mode/source combinations:

- `selected_only` + `main`: Qwen4-Exp QSA-compatible execution.
- `local_plus_selected` + `auxiliary`: DeepSeek V4 CSA-compatible execution and requires `local_window_size > 0`.

The main cache uses BNSH layout and ordinary contiguous append semantics. Selection changes reads, not cache writes.
Auxiliary K/V use BNSH layout and are read-only. When `auxiliary_kv_shared=1`, auxiliary_value may be omitted and
auxiliary_key is used as both K and V. DeepSeek-style post-attention inverse/conjugate RoPE and output projection remain
the exporter's responsibility.

Main caches may instead use INT8 with positive finite float32 key_scale and value_scale vectors, one scale per
KV-head channel. Newly appended keys (after optional normalization and RoPE) and values are divided by their scales,
rounded to nearest with ties to even, and saturated to [-128, 127]. Attention reads the quantized representation,
including newly appended rows. Auxiliary caches remain floating point. Invalid scale values produce validation_status=10
when validation is requested; CUDA strict validation also reports them as an error.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DynamicSparseAttention, 1,
    OpSchema()
        .SetDoc(DynamicSparseAttention_ver1_doc)
        .Attr("num_heads", "Number of query heads.", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of main and auxiliary KV heads.", AttributeProto::INT)
        .Attr("scale", "Scaling factor applied to QK. Defaults to 1/sqrt(head_size).",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("is_causal", "Must be 1. DynamicSparseAttention version 1 supports causal attention only.",
              AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("local_window_size", "Number of causally visible main-cache entries in local_plus_selected mode.",
              AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("attention_mode", "One of 'selected_only' or 'local_plus_selected'.",
              AttributeProto::STRING, std::string("selected_only"))
        .Attr("selected_kv_source", "Source addressed by selected indices: 'main' or 'auxiliary'.",
              AttributeProto::STRING, std::string("main"))
        .Attr("do_rotary", "Whether to apply rotary embedding to Q and newly appended K.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("rotary_interleaved", "Whether rotary pairs use interleaved layout.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("rotary_offset",
              "First head channel covered by rotary embedding. Must be nonnegative, a multiple of 8, and leave room "
              "for the rotary dimensions within head_size.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("qk_norm_epsilon", "Epsilon for optional per-head Q/K RMS normalization.",
              AttributeProto::FLOAT, 1e-6f)
        .Attr("smooth_softmax", "Add a zero-valued sink logit when no explicit head_sink is supplied.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("auxiliary_kv_shared", "Use auxiliary_key as both key and value when auxiliary_value is omitted.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "query",
               "Query [batch, sequence, num_heads * head_size]. When key and value are omitted, each row contains "
               "Q, K, then V concatenated with total width (num_heads + 2 * kv_num_heads) * head_size.",
               "T")
        .Input(1, "key", "Current main key [batch, sequence, kv_num_heads * head_size].", "T",
               OpSchema::Optional)
        .Input(2, "value", "Current main value [batch, sequence, kv_num_heads * head_size].", "T",
               OpSchema::Optional)
        .Input(3, "past_key", "Main key cache in BNSH layout, same type as query or INT8.", "TCache", OpSchema::Optional)
        .Input(4, "past_value", "Main value cache in BNSH layout, same type as past_key.", "TCache", OpSchema::Optional)
        .Input(5, "auxiliary_key", "Read-only auxiliary key sequence in BNSH layout.", "T",
               OpSchema::Optional)
        .Input(6, "auxiliary_value", "Read-only auxiliary value sequence in BNSH layout.", "T",
               OpSchema::Optional)
        .Input(7, "selected_indices", "Selected request-local source positions [batch * sequence, max_selected].",
               "M")
        .Input(8, "selected_counts", "Number of valid entries per selected-index row [batch * sequence].", "M")
        .Input(9, "seqlens_k", "Total logical main sequence length minus one for each batch entry.", "M")
        .Input(10, "total_sequence_length", "Maximum total main sequence length, as a scalar or one-element tensor.",
               "M")
        .Input(11, "cos_cache", "Rotary cosine cache [max_sequence_length, rotary_dim / 2].", "T",
               OpSchema::Optional)
        .Input(12, "sin_cache", "Rotary sine cache [max_sequence_length, rotary_dim / 2].", "T",
               OpSchema::Optional)
        .Input(13, "position_ids", "Optional rotary positions [batch, sequence].", "tensor(int64)",
               OpSchema::Optional)
        .Input(14, "q_norm_weight", "Optional Q RMSNorm weight [head_size].", "T", OpSchema::Optional)
        .Input(15, "k_norm_weight", "Optional K RMSNorm weight [head_size].", "T", OpSchema::Optional)
        .Input(16, "head_sink",
               "Optional sink logit [num_heads]. Participates in the shared softmax denominator in both attention modes.",
               "T", OpSchema::Optional)
        .Input(17, "committed_length",
               "Optional single-request committed token count for cursor validation. When supplied, validation_status "
               "is 9 if seqlens_k, sequence_length, or cache_capacity is inconsistent. CUDA only.",
               "M", OpSchema::Optional)
        .Input(18, "key_scale", "Positive finite per-channel INT8 key-cache scales [kv_num_heads * head_size].",
               "S", OpSchema::Optional)
        .Input(19, "value_scale", "Positive finite per-channel INT8 value-cache scales [kv_num_heads * head_size].",
               "S", OpSchema::Optional)
        .Output(0, "output", "Attention output [batch, sequence, num_heads * head_size].", "T")
        .Output(1, "present_key", "Updated main key cache in BNSH layout.", "TCache", OpSchema::Optional)
        .Output(2, "present_value", "Updated main value cache in BNSH layout.", "TCache", OpSchema::Optional)
        .Output(3, "validation_status", "Optional CUDA-only int32 status: 0 when valid, a validation code when invalid.",
                "M", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain all floating-point inputs and outputs to one element type.")
        .TypeConstraint("TCache", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int8)"},
                        "Cache type must match T unless INT8 per-channel scales are supplied.")
        .TypeConstraint("S", {"tensor(float)"}, "INT8 cache scales use float32.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain selection and sequence metadata to int32.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          DynamicSparseAttentionTypeAndShapeInference(ctx);
          if (ctx.getNumOutputs() > 3 && ctx.hasOutput(3)) {
            updateOutputElemType(ctx, 3, ONNX_NAMESPACE::TensorProto_DataType_INT32);
            ONNX_NAMESPACE::TensorShapeProto status_shape;
            status_shape.add_dim()->set_dim_value(1);
            updateOutputShape(ctx, 3, status_shape);
          }
        }));

constexpr const char* Attention_ver1_doc = R"DOC(
Multi-Head Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT).

The weights for input projection of Q, K and V are merged. The data is stacked on the second dimension. Its shape
is (input_hidden_size, hidden_size + hidden_size + v_hidden_size). Here hidden_size is the hidden dimension of Q and K,
and v_hidden_size is that of V.
Q, K and V hidden sizes must be positive multiples of num_heads. Their sum must match the weights' second dimension.
When qkv_hidden_sizes is omitted, the weights' second dimension must be divisible by three.

The mask_index is optional. Besides raw attention mask with shape (batch_size, total_sequence_length)
or (batch_size, sequence_length, total_sequence_length) with value 0 for masked and 1 otherwise,
we support other two formats: When input has right-side padding, mask_index is one dimension with shape (batch_size),
where value is actual sequence length excluding padding. When input has left-side padding, mask_index has
shape (2 * batch_size), where the values are the exclusive end positions followed by the inclusive start positions.

When unidirectional is 1, each token only attends to previous tokens.

Both past and present state are optional. They shall be used together, and not allowed to use only one of them.
The qkv_hidden_sizes is required only when K and V have different hidden sizes.

When there is past state, hidden dimension for Q, K and V shall be the same.

The total_sequence_length is past_sequence_length + kv_sequence_length. Here kv_sequence_length is the length of K or V.
For self attention, kv_sequence_length equals to sequence_length (sequence length of Q).
For cross attention, query and key might have different lengths.
)DOC";

// Currently, the `convert_generation.py` script renames the `Attention` nodes to `DecoderMaskedSelfAttention`
// if the user requests it. Hence, the schemas of `DecoderMaskedSelfAttention` and `Attention` schemas
// are tightly coupled. A change in Attention also needs corresponding schema updates in `DecoderMaskedSelfAttention`
// and its kernel.
// TODO(hasesh): Decouple the schema of `DecoderMaskedSelfAttention` from the schema of the `Attention` operator
// by making appropriate tool changes.

ONNX_MS_OPERATOR_SET_SCHEMA(
    Attention, 1,
    OpSchema()
        .SetDoc(Attention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("unidirectional",
              "Whether every token can only attend to previous tokens. Default value is 0.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("qkv_hidden_sizes",
              "Hidden dimension of Q, K, V: hidden_size, hidden_size and v_hidden_size",
              AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(2, batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_embedding_dim",
              "Dimension of rotary embedding. Limited to 32, 64 or 128. Default value is head_size",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, sequence_length, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T",
               OpSchema::Optional)
        .Input(3,
               "mask_index",
               "Attention mask with shape (batch_size, 1, max_sequence_length, max_sequence_length), "
               "(batch_size, total_sequence_length) or (batch_size, sequence_length, total_sequence_length), "
               "or index with shape (batch_size) or (2 * batch_size) or (3 * batch_size + 2)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)"
               "When past_present_share_buffer is set, "
               "its shape is (2, batch_size, num_heads, max_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(5,
               "attention_bias",
               "additional add to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0).",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present",
                "past state for key and value with shape (2, batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (2, batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          constexpr int past_input_index = 4;
          AttentionTypeAndShapeInference(ctx, past_input_index);
        }));

constexpr const char* PackingAttention_ver1_doc = R"DOC(
This is the packed version of Attention.

Sequences in one batch usually don't have same length and they are padded to have same length,
e.g., below is a batch with 3 sequences and tokens* are padded.
  Sequence_0:   0,  1*, 2*,  3*
  Sequence_1:   4,  5,  6*,  7*
  Sequence_2:   8,  9,  10,  11

PackedAttention is designed to takes in packed input, i.e., only the real tokens without padding.
An input as above will be packed into 3 tensors like below:
 - input ([h0, h4, h5, h8, h9, h10, h11])
 - token_offset: 0, 4, 5, 8, 9, 10, 11,  1*, 2*, 3*, 6*, 7*
 - cumulated_token_count: 0, 1, 1+2, 1+2+4

Input tensors contains the hidden embedding of real tokens.
Token_offset records the offset of token in the unpacked input.
cumulated_token_count records cumulated length of each sequence length.

The operator only supports BERT like model with padding on right now.

)DOC";

// Shape inference for PackedAttention. Here are the shapes of inputs and output:
// Input 'input':                      (token_count, input_hidden_size)
// Input 'weights':                    (input_hidden_size, hidden_size + hidden_size + v_hidden_size)
// Input 'bias':                       (hidden_size + hidden_size + v_hidden_size)
// Input 'token_offset':               (batch_size, sequence_length)
// Input 'cumulative_sequence_length': (batch_size + 1)
// Input 'attention_bias':     (batch_size or 1, num_heads or 1, sequence_length, sequence_length)
// Output 'output':                    (token_count, v_hidden_size)
void PackedAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
    auto& input_shape = getInputShape(ctx, 0);
    auto& input_dims = input_shape.dim();
    int input_dim_size = input_dims.size();
    if (input_dim_size != 2) {
      fail_shape_inference("Inputs 0 shall be 2 dimensions");
    }

    auto& bias_shape = getInputShape(ctx, 2);
    auto& bias_dims = bias_shape.dim();
    if (bias_dims.size() != 1) {
      fail_shape_inference("Invalid bias shape");
    }

    int64_t v_hidden_size = -1;
    std::vector<int64_t> qkv_hidden_sizes;
    getRepeatedAttribute(ctx, "qkv_hidden_sizes", qkv_hidden_sizes);

    if (qkv_hidden_sizes.size() != 0) {
      if (qkv_hidden_sizes.size() != 3) {
        fail_shape_inference("qkv_hidden_sizes should have 3 elements")
      }
      v_hidden_size = qkv_hidden_sizes[2];
    } else {
      v_hidden_size = bias_shape.dim(0).dim_value() / 3;
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    for (auto& dim : input_dims) {
      *output_shape.add_dim() = dim;
    }

    output_shape.mutable_dim(input_dim_size - 1)->set_dim_value(v_hidden_size);
    updateOutputShape(ctx, 0, output_shape);
  }
}

ONNX_MS_OPERATOR_SET_SCHEMA(
    PackedAttention, 1,
    OpSchema()
        .SetDoc(PackingAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("qkv_hidden_sizes",
              "Hidden dimension of Q, K, V: hidden_size, hidden_size and v_hidden_size",
              AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (token_count, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T")
        .Input(3,
               "token_offset",
               "In packing mode, it specifies the offset of each token(batch_size, sequence_length).",
               "M")
        .Input(4,
               "cumulative_sequence_length",
               "A tensor with shape (batch_size + 1). It specifies the cumulative sequence length.",
               "M")
        .Input(5,
               "attention_bias",
               "A tensor with shape (batch_size or 1, num_heads or 1, sequence_length, sequence_length)."
               "It specifies the additional bias to QxK'",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "2D output tensor with shape (token_count, v_hidden_size)",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PackedAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* PackedMultiHeadAttention_ver1_doc = R"DOC(
This is the packed version of MultiHeadAttention.

Sequences in one batch usually don't have same length and they are padded to have same length,
e.g., below is a batch with 3 sequences and * is padding token.
  Sequence_0:   0,  1*, 2*,  3*
  Sequence_1:   4,  5,  6*,  7*
  Sequence_2:   8,  9,  10,  11

PackedMultiHeadAttention is designed to takes in packed input, i.e., only the real tokens without padding.
An input as above will be packed into 3 tensors like below:
 - query ([q0, q4, q5, q8, q9, q10, q11])
 - key ([k0, k4, k5, k8, k9, k10, k11])
 - value ([v0, v4, v5, v8, v9, v10, v11])
 - token_offset: 0, 4, 5, 8, 9, 10, 11,  1*, 2*, 3*, 6*, 7*
 - cumulative_sequence_length: 0, 1, 1+2, 1+2+4

The query, key and value tensors contain result of hidden embedding of real tokens after input projections.
Token_offset records the offset of token in the unpacked input.
cumulative_sequence_length records cumulated length of each sequence length.

The operator only supports BERT like model with padding on right now.
)DOC";

// Shape inference for PackedMultiHeadAttention. Here are the shapes of inputs and output:
// When Q, K and V are not packed:
//   Input 'query':                      (token_count, hidden_size)
//   Input 'key':                        (token_count, hidden_size)
//   Input 'value':                      (token_count, v_hidden_size)
// When Q, K and V are packed:
//   Input 'query':                      (token_count, num_heads, 3, head_size)
//   Input 'key':                        None
//   Input 'value':                      None
// Input 'bias':                         (hidden_size + hidden_size + v_hidden_size)
// Input 'token_offset':                 (batch_size, sequence_length)
// Input 'cumulative_sequence_length':   (batch_size + 1)
// Input 'attention_bias':       (batch_size or 1, num_heads or 1, sequence_length, sequence_length) or None
// Output 'output':                      (token_count, v_hidden_size)
void PackedMultiHeadAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 2 && query_dims.size() != 4) {
      fail_shape_inference("Inputs 0 (query) shall be 2 or 4 dimensions");
    }

    if (query_dims.size() == 4) {  // packed QKV
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1] * query_dims[3];
      updateOutputShape(ctx, 0, output_shape);
      return;
    }

    if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
      if (value_dims.size() != 2) {
        fail_shape_inference("Inputs 2 (value) shall be 2 dimensions");
      }

      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = value_dims[1];
      updateOutputShape(ctx, 0, output_shape);
      return;
    }
  }
}

ONNX_MS_OPERATOR_SET_SCHEMA(
    PackedMultiHeadAttention, 1,
    OpSchema()
        .SetDoc(PackedMultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (token_count, hidden_size) or packed qkv with shape (token_count, num_heads, 3, head_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (token_count, hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (token_count, v_hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Input(4,
               "token_offset",
               "Offset of each token before packing, with shape (batch_size, sequence_length).",
               "M")
        .Input(5,
               "cumulative_sequence_length",
               "A tensor with shape (batch_size + 1). It specifies the cumulative sequence length.",
               "M")
        .Input(6,
               "attention_bias",
               "It specifies the additional bias to QxK'. "
               "The shape is (batch_size or 1, num_heads or 1, sequence_length, sequence_length)",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "output tensor with shape (token_count, v_hidden_size)",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask, offset and sequence length to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PackedMultiHeadAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* DecoderMaskedSelfAttention_ver1_doc = R"DOC(
Self attention that supports input sequence length of 1.

The weights for input projection of Q, K and V are merged. The data is stacked on the second dimension. Its shape
is (input_hidden_size, hidden_size + hidden_size + v_hidden_size). Here hidden_size is the hidden dimension of Q and K,
and v_hidden_size is that of V.

The mask_index is optional. If it is provided, only raw attention mask with shape (batch_size, total_sequence_length) is supported currently.

Both past and present state need to be provided.

The qkv_hidden_sizes is required only when K and V have different hidden sizes.

The total_sequence_length is past_sequence_length + kv_sequence_length. Here kv_sequence_length is the length of K or V.
Currently, only self attention is supported which means that kv_sequence_length equals to sequence_length (sequence length of Q).
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderMaskedSelfAttention, 1,
    OpSchema()
        .SetDoc(DecoderMaskedSelfAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(2, batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, 1, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T")
        .Input(3,
               "mask_index",
               "Mask values of shape (batch_size, total_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)"
               "When past_present_share_buffer is set, "
               "its shape is (2, batch_size, num_heads, max_sequence_length, head_size). "
               "The first `batch_size * num_heads * max_sequence_length * head_size` elements correspond to keys "
               "and the next `batch_size * num_heads * max_sequence_length * head_size` elements correspond to values. "
               "The keys buffer is re-ordered in such a way that its virtual sub-tensor of shape "
               "(batch_size, num_heads, max_sequence_length, head_size) which may be perceived as being of shape "
               "(batch_size, num_heads, max_sequence_length, head_size / x, x) is reordered to "
               "become (batch_size, num_heads, head_size / x, max_sequence_length, x) where `x = 16 / sizeof(T)`.",
               "T")
        .Input(5,
               "attention_bias",
               "additional add to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0).",
               "M")
        .Input(7,
               "beam_width",
               "The beam width that is being used while decoding. "
               "If not provided, the beam width will be assumed to be 1.",
               "M",
               OpSchema::Optional)
        .Input(8,
               "cache_indirection",
               "A buffer of shape [batch_size, beam_width, max_output_length] where an `[i, j, k]` entry specifies "
               "which beam the `k`-th token came from for the `j`-th beam for batch `i` in the current iteration",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present",
                "past state for key and value with shape (2, batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (2, batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          constexpr int past_input_index = 4;
          AttentionTypeAndShapeInference(ctx, past_input_index);
        }));

constexpr const char* DecoderMaskedMultiHeadAttention_ver1_doc = R"DOC(
Multihead attention that supports input sequence length of 1.
Similar to DecoderMaskedSelfAttention but this op excludes QKV MatMul and Bias.
This op supports both Self and Cross Attention.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderMaskedMultiHeadAttention, 1,
    OpSchema()
        .SetDoc(DecoderMaskedMultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("output_qk",
              "Need output the cross attention MatMul(Q, K)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, 1, hidden_size) or packed QKV with shape "
               "(batch_size, 1, 2 * hidden_size + v_hidden_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, 1, hidden_size) for self attention "
               "or past_key with shape (batch_size, num_heads, kv_sequence_length, head_size) for cross attention",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, 1, v_hidden_size) for self attention "
               "or past_value with shape (batch_size, num_heads, kv_sequence_length, head_size) for cross attention",
               "T",
               OpSchema::Optional)
        .Input(3,
               "mask_index",
               "Mask values of shape (batch_size, total_sequence_length) or (batch_size, kv_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "attention_bias",
               "additional add to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(5,
               "past_key",
               "past state for key with shape (batch_size, num_heads, past_sequence_length, head_size) for self attention"
               "When past_present_share_buffer is set, "
               "its shape is (batch_size, num_heads, max_sequence_length, head_size). "
               // The re-ordering happens only for CUDA EP at the moment. We probably shall support 4D or 5D shape or
               // attribute to distinguish whether it is re-ordered or not.
               "The keys buffer is re-ordered in such a way that its virtual sub-tensor of shape "
               "(batch_size, num_heads, max_sequence_length, head_size) which may be perceived as being of shape "
               "(batch_size, num_heads, max_sequence_length, head_size / x, x) is reordered to "
               "become (batch_size, num_heads, head_size / x, max_sequence_length, x) where `x = 16 / sizeof(T)`.",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_value",
               "past state for value with shape (batch_size, num_heads, past_sequence_length, head_size) for self attention"
               "When past_present_share_buffer is set, "
               "its shape is (batch_size, num_heads, max_sequence_length, head_size). ",
               "T",
               OpSchema::Optional)
        .Input(7,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0)."
               "Cross Attention doesn't need this input.",
               "M",
               OpSchema::Optional)
        .Input(8,
               "beam_width",
               "The beam width that is being used while decoding. "
               "If not provided, the beam width will be assumed to be 1.",
               "M",
               OpSchema::Optional)
        .Input(9,
               "cache_indirection",
               "A buffer of shape [batch_size, beam_width, max_output_length] where an `[i, j, k]` entry specifies "
               "which beam the `k`-th token came from for the `j`-th beam for batch `i` in the current iteration",
               "M",
               OpSchema::Optional)
        .Input(10,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state for key with shape (batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state for value with shape (batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .Output(3,
                "qk",
                "normalized Q * K, of shape (batch_size, num_heads, 1, total_sequence_length). ",
                "QK",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("QK",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain QK output to float32 or float16 tensors, independent of input type or output type.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          bool is_dmmha_packing = !hasInputShape(ctx, 1) && !hasInputShape(ctx, 2);
          MultiHeadAttentionTypeAndShapeInference(ctx, 5, is_dmmha_packing);
        }));

constexpr const char* MultiHeadAttention_ver1_doc = R"DOC(
Multi-Head Self/Cross Attention. Bias from input projection is included.

The key padding mask is optional. When its shape is (batch_size, kv_sequence_length), value 0
means padding or 1 otherwise. When key has right-side padding, its shape could be (batch_size): it is actual length of
each key sequence excluding paddings.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    MultiHeadAttention, 1,
    OpSchema()
        .SetDoc(MultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("unidirectional",
              "Whether every token can only attend to previous tokens. Default value is 0.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "query",
               "Query with shape (batch_size, sequence_length, hidden_size), or packed QKV with shape (batch_size, kv_sequence_length, num_heads, 3, head_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, kv_sequence_length, hidden_size), or packed KV with shape (batch_size, kv_sequence_length, num_heads, 2, head_size), "
               "or past_key with shape (batch_size, num_heads, kv_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, kv_sequence_length, v_hidden_size), or past_value with shape (batch_size, num_heads, kv_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Input(4,
               "key_padding_mask",
               "Key padding mask with shape (batch_size), (3 * batch_size + 2), (batch_size, kv_sequence_length), (batch_size, total_sequence_length), "
               "or (batch_size, sequence_length, total_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(5,
               "attention_bias",
               "bias added to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_key",
               "past state for key with shape (batch_size, num_heads, past_sequence_length, head_size) "
               "or (batch_size, num_heads, max_sequence_length, head_size) when buffer sharing is used",
               "T",
               OpSchema::Optional)
        .Input(7,
               "past_value",
               "past state for value with shape (batch_size, num_heads, past_sequence_length, head_size) "
               "or (batch_size, num_heads, max_sequence_length, head_size) when buffer sharing is used",
               "T",
               OpSchema::Optional)
        .Input(8,
               "past_sequence_length",
               "The past_sequence_length buffer sharing is used with",
               "M",
               OpSchema::Optional)
        .Input(9,
               "cache_indirection",
               "A buffer of shape [batch_size, beam_width, max_sequence_length] where an [i, j, k] entry specifies"
               "which beam the 'k' th token came from for the 'j' th beam for batch 'i' in the current iteration",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state for key with shape (batch_size, num_heads, total_sequence_length, head_size) "
                "or (batch_size, num_heads, max_sequence_length, head_size) when buffer sharing is used",
                "T",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state for value with shape (batch_size, num_heads, total_sequence_length, head_size) "
                "or (batch_size, num_heads, max_sequence_length, head_size) when buffer sharing is used",
                "T",
                OpSchema::Optional)
        .Output(3,
                "qk",
                "normalized Q * K, of shape (batch_size, num_heads, sequence_length, total_sequence_length). ",
                "QK",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("QK", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain QK output to float32 or float16 tensors, independent of input type or output type.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          MultiHeadAttentionTypeAndShapeInference(ctx, 6);
        }));

constexpr const char* GroupQueryAttention_ver1_doc = R"DOC(
Group Query Self/Cross Attention with KV Cache Quantization Support.

This operator implements grouped-query attention with past state (KV cache) support.
It also supports optional float8, int8 or int4 quantization for the KV cache to reduce memory footprint.

**Cache Format:**
The past and present KV cache tensors are expected in a BNSH format: `(batch_size, num_heads, cache_sequence_length, head_size)`, where `cache_sequence_length` is the length of the cached key/value sequences, or the maximum sequence length when past and present buffer sharing is used.

**Windowed KV Cache (`sliding_window_cache` attribute):**
When `sliding_window_cache` is 1, the past/present buffers are window-sized instead of full-length and the operator evicts internally. Let `C` be the cache capacity (dimension 2 of `past_key`, which is also the sequence dimension of `present_key`), `W` be `local_window_size`, and `T` be the absolute number of tokens processed so far by this batch entry, i.e. `seqlens_k[b] + 1`. The scalar `total_sequence_length` input is only the batch maximum of `T`; the layout below is per batch entry, so a ragged batch gets a different resident range per entry. `C` must be at least `W`.

After a step, rows `[0, L)` of `present_key` and `present_value` hold the `L` most recent positions in increasing position order, so row `i` holds absolute position `T - L + i`. The retained positions are always physically contiguous and start at row 0; the layout never wraps around, so a ring-buffer layout cannot be exposed through these outputs. Rows `[L, C)` are unspecified. The resident count `L` is a function of `T` alone:

```
G = C - W + 1
L(T) = T                            if T <= C
L(T) = T - G * ceil((T - C) / G)    otherwise
```

Hence `min(T, W) <= L(T) <= min(T, C)`: the whole window stays resident, and eviction reclaims `G` positions at once rather than one position per step, so consumers must not assume that the cache is kept full at `min(T, C)`.

  Because `L` depends only on `T`, the resulting layout is independent of how the tokens were split into steps: a multi-token step of `S` tokens (speculative decoding, chunked prefill) leaves exactly the layout that the same tokens would produce one at a time. Any `S >= 1` is accepted, including `S > C`; a step that would evict positions it still has to read is staged internally, so the capacity does not have to cover the step. When past context is present, the existing operator restriction still applies: `sequence_length > 1` requires `batch_size == 1`.

  An execution provider may accept only part of the `C >= W` range. A configuration with `C < W` (equivalently, `W > C`) is invalid and is rejected with `INVALID_ARGUMENT`. The CUDA implementation requires `C == W`, so there `G` is 1 and `L(T)` is `min(T, C)`; a larger capacity is rejected. The CPU implementation accepts any `C >= W`, and slack above the window amortizes compaction over `G` steps.

To drop the last `k` tokens, for example after rejecting speculative draft tokens, re-run with the smaller `total_sequence_length` and `seqlens_k` and leave the buffer untouched. That is exact when `L(T - k) == L(T) - k`, which callers can evaluate with the formula above. Otherwise the shorter layout needs positions that have already been evicted, and the window has to be re-materialized.

**Quantization:**
When quantization is enabled, `past_key` and `past_value` inputs can be of type `float8e4m3fn`, `uint8` or `int8`. The corresponding `k_scale` and `v_scale` tensors must be provided.
The operator will output `present_key` and `present_value` in same format as the `past_key` and `past_value`.

For 4-bit quantization, the data type is uint8 where each byte contains two 4-bit values. The bit width of quantized KV cache can be set using `kv_cache_bit_width` attribute.

The shapes of the k_scale, v_scale tensors shall be broadcastable to present_key shape.

**Quantization Modes (`k_quant_type`, `v_quant_type` attributes):**
- **"NONE"**: No quantization.
- **"PER_TENSOR"**: A single scale for the entire tensor. Scale example shape: `[1]`.
- **"PER_CHANNEL"**: A scale for each channel. Scale example shape: `[1, num_heads_k, 1, head_size]`.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GroupQueryAttention, 1,
    OpSchema()
        .SetDoc(GroupQueryAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads for q", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of attention heads for k and v", AttributeProto::INT)
        .Attr("causal",
              "Whether to apply a causal mask. Must be 0 or 1. Default value is 1. Set to 0 for bidirectional attention.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("softcap",
              "Softcap value for attention weights. Default value is 0.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("local_window_size",
              "left_window_size for causal local attention (like Mistral). Must be -1 when causal is 0. "
              "Default value is -1 meaning unused.",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("sliding_window_cache",
              "Set to 1 when the past/present KV buffers are window-sized instead of holding the whole "
              "sequence. The op then evicts internally and indexes the buffers in cache-relative "
              "coordinates, keeping the most recent positions contiguously at rows [0, L) with "
              "min(T, local_window_size) <= L <= min(T, capacity), where T is seqlens_k[b] + 1 for that "
              "batch entry. Requires local_window_size > 0 and a cache capacity of at least "
              "local_window_size; a smaller capacity (W > C) is rejected with INVALID_ARGUMENT. The CUDA "
              "implementation additionally requires the capacity to equal local_window_size. Multi-token "
              "steps of any length are supported and produce the same layout as single-token steps, so the "
              "capacity need not cover the entire step. When past context is present, sequence_length > 1 "
              "requires batch_size == 1. See the "
              "Windowed KV Cache section of the operator description for the exact resident-range, "
              "eviction and rollback contract. Default value is 0 (full-length cache).",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_interleaved",
              "Rotate using interleaved pattern. Default value is 0 (False).",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("smooth_softmax",
              "Use a smooth factor in softmax.",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("qk_output",
              "Output values of QK matrix multiplication before (1) or after (2) softmax normalization. Default value is 0 (don't output).",
              AttributeProto::INT,
              static_cast<int64_t>(QKOutputType::NO_OUTPUT))
        .Attr("k_quant_type", "Quantization type for K cache. One of 'NONE', 'PER_TENSOR', 'PER_CHANNEL'.", AttributeProto::STRING, std::string("NONE"))
        .Attr("v_quant_type", "Quantization type for V cache. One of 'NONE', 'PER_TENSOR', 'PER_CHANNEL'.", AttributeProto::STRING, std::string("NONE"))
        .Attr("kv_cache_bit_width", "Bit width of quantized KV cache. Supported values are 8 and 4.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("qk_norm_epsilon",
              "Epsilon used by the per-head RMS norm applied to Q and K when q_norm_weight and k_norm_weight inputs are provided. "
              "Default value is 1e-6.",
              AttributeProto::FLOAT,
              1e-6f)
        .Attr("q_norm_epsilon",
              "Epsilon for per-head Q RMS normalization. Overrides qk_norm_epsilon for Q when provided. "
              "Must be finite and positive.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("k_norm_epsilon",
              "Epsilon for per-head K RMS normalization. Overrides qk_norm_epsilon for K when provided. "
              "Must be finite and positive.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, sequence_length, hidden_size), or packed QKV with shape"
               "(batch_size, sequence_length, d) where d is (num_heads * head_size + 2 * kv_num_heads * head_size).",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, kv_sequence_length, kv_hidden_size) ",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, kv_sequence_length, kv_hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "past_key",
               "past state key with support for format BNSH. When past_key uses same tensor as present_key"
               "(k-v cache), it is of length max_sequence_length... otherwise of length past_sequence_length. "
               "When sliding_window_cache is 1 this length is the window cache capacity C, which is chosen by the "
               "caller independently of the sequence length and must be at least local_window_size.",
               "T_CACHE",
               OpSchema::Optional)
        .Input(4,
               "past_value",
               "past state value with support for format BNSH. When past_value uses same tensor as present_value"
               "(k-v cache), it is of length max_sequence_length... otherwise of length past_sequence_length. "
               "When sliding_window_cache is 1 this length is the window cache capacity C, which is chosen by the "
               "caller independently of the sequence length and must be at least local_window_size.",
               "T_CACHE",
               OpSchema::Optional)
        .Input(5,
               "seqlens_k",
               "1D Tensor of shape (batch_size). Equivalent to (total_sequence_lengths - 1).",
               "M")
        .Input(6,
               "total_sequence_length",
               "Scalar tensor equivalent to the maximum total sequence length (past + new) of the batch. Used for "
               "checking inputs and determining prompt vs token generation case.",
               "M")
        .Input(7,
               "cos_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Input(8,
               "sin_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Input(9,
               "position_ids",
               "2D tensor with shape (batch_size, sequence_length). When processing the first prompt the kernel "
               "uses only the first element",
               "tensor(int64)",
               OpSchema::Optional)
        .Input(10,
               "attention_bias",
               "additional add to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length). "
               "The last dimension is indexed by absolute key position and stays total_sequence_length when "
               "sliding_window_cache is 1: it is not reduced to the cache capacity or to local_window_size. The "
               "operator reads the columns of the positions that are resident in the cache and ignores the rest. "
               "CPU supports this windowed absolute-column indexing; CUDA rejects attention_bias when "
               "sliding_window_cache is 1.",
               "T",
               OpSchema::Optional)
        .Input(11,
               "head_sink",
               "1D tensor with shape (num_heads). Each head has a smooth factor adding to the denominator of softmax.",
               "T",
               OpSchema::Optional)
        .Input(12, "k_scale", "Scale tensor for past_key.", "T_KV_SCALE", OpSchema::Optional)
        .Input(13, "v_scale", "Scale tensor for past_value.", "T_KV_SCALE", OpSchema::Optional)
        .Input(14,
               "q_norm_weight",
               "Optional 1D tensor of shape (head_size). When provided together with k_norm_weight, the kernel applies a "
               "per-head RMS normalization to Q (and K) before any rotary embedding. Used by Qwen3-style models that wrap "
               "their Q/K projections in a Reshape -> SimplifiedLayerNormalization -> Reshape stack; downstream graph fusion "
               "folds that pattern into this input. Currently honored by the CUDA and native WebGPU execution providers; "
               "JSEP WebGPU/JS and other EPs must reject the node when this input is set.",
               "T",
               OpSchema::Optional)
        .Input(15,
               "k_norm_weight",
               "Optional 1D tensor of shape (head_size). See q_norm_weight. Must be provided together with q_norm_weight.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state key with support for format BNSH. When past_key uses same tensor as present_key"
                "(k-v buffer), it is of length max_sequence_length... otherwise of length past_sequence_length +"
                "kv_sequence_length.",
                "T_CACHE",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state value with support for format BNSH. When past_value uses same tensor as present_value"
                "(k-v buffer), it is of length max_sequence_length... otherwise of length past_sequence_length +"
                "kv_sequence_length.",
                "T_CACHE",
                OpSchema::Optional)
        .Output(3,
                "output_qk",
                "Values of QK matrix multiplication, either before or after softmax normalization",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("T_CACHE", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(uint8)", "tensor(int8)", "tensor(float8e4m3fn)"}, "Constrain KV cache types.")
        .TypeConstraint("T_KV_SCALE", {"tensor(float)"}, "Constrain KV cache scale types.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to int tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // The 'output_qk' is an optional output at index 3.
          // Pass its index to the shape inference logic only if the node instance actually has more than 3 outputs.
          // Otherwise, pass -1 to signal that the optional output is not present and validation should be skipped.
          int qk_output_index = ctx.getNumOutputs() > 3 ? 3 : -1;
          GroupQueryAttentionTypeAndShapeInference(ctx, 3, qk_output_index);
        }));

constexpr const char* PagedAttention_ver1_doc = R"DOC(
Paged Attention.

This op leverages a block-based KV cache to enable continuous batching for LLMs. Currently, it is designed to work with
the CUDA Execution Provider only.

In other attention ops, batch entries typically aren't of the same length, so they are padded.
Below is a batch with 3 sequences where * denotes a padding token.
  Sequence_0:   0,  1*, 2*,  3*
  Sequence_1:   4,  5,  6*,  7*
  Sequence_2:   8,  9,  10,  11

PagedAttention is designed to take in packed input, i.e., only the real tokens without padding.
For example, the input shown above will be packed into 3 tensors like below:
 - query ([q0, q4, q5, q8, q9, q10, q11])
 - key ([k0, k4, k5, k8, k9, k10, k11])
 - value ([v0, v4, v5, v8, v9, v10, v11])
 - cumulative_sequence_length: 0, 1, 1+2, 1+2+4
This packing omits padding tokens.

The query, key and value tensors contain result of hidden embedding of real tokens after input projections.
cumulative_sequence_length records cumulated length of each sequence length.

)DOC";

// Shape inference for PagedAttention. Here are the shapes of inputs and output:
// When Q, K and V are not packed:
//   Input 'query':                      (token_count, hidden_size)
//   Input 'key':                        (token_count, kv_hidden_size)
//   Input 'value':                      (token_count, kv_hidden_size)
// When Q, K and V are packed:
//   Input 'query':                      (token_count, (num_heads + 2 * kv_num_heads) * head_size)
//   Input 'key':                        None
//   Input 'value':                      None
// Input 'key_cache':                    (num_blocks, block_size, kv_num_heads, head_size)
// Input 'value_cache':                  (num_blocks, block_size, kv_num_heads, head_size)
// Input 'cumulative_sequence_length':   (batch_size + 1)
// Input 'seqlens':                      (batch_size)
// Input 'block_table':                  (batch_size, max_blocks_per_sequence)
// Input 'cos_cache':                    (max_seq_len, head_size / 2)
// Input 'sin_cache':                    (max_seq_len, head_size / 2)
// Input 'slot_mapping':                 (token_count)
// Input 'head_sink':                    (num_heads)
// Input 'q_norm_weight':                (head_size)
// Input 'k_norm_weight':                (head_size)
// Input 'k_scale':                      (1) for PER_TENSOR, (kv_num_heads, 1, head_size) for PER_CHANNEL
// Input 'v_scale':                      (1) for PER_TENSOR, (kv_num_heads, 1, head_size) for PER_CHANNEL
// Input 'attention_metadata':           (2) or (3), CPU memory:
//                                       [max_query_len_bound, max_kv_len_bound,
//                                        optional max_kv_len_lower_bound]
// Output 'output':                      (token_count, num_heads * v_head_size)
// Output 'key_cache_out':               (num_blocks, block_size, kv_num_heads, head_size)
// Output 'value_cache_out':             (num_blocks, block_size, kv_num_heads, head_size), absent for LATENT
void PagedAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  const std::string kv_cache_layout = getAttribute(ctx, "kv_cache_layout", "SEPARATE");
  if (kv_cache_layout != "SEPARATE" && kv_cache_layout != "LATENT") {
    fail_shape_inference("kv_cache_layout must be 'SEPARATE' or 'LATENT'.");
  }
  const bool is_latent_kv = kv_cache_layout == "LATENT";
  const int64_t v_head_size_attr = getAttribute(ctx, "v_head_size", 0);

  // Shape inference for output tensor
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 2) {
      fail_shape_inference("Input 0 (query) shall be 2 dimensions");
    }

    if (is_latent_kv) {
      // Absorbed MLA: query is unpacked and (unlike SEPARATE mode) the output head width is
      // v_head_size rather than head_size, so the output shape cannot simply be copied from query.
      int64_t num_heads = getAttribute(ctx, "num_heads", 0);
      int64_t q_hidden_size = query_dims[1].has_dim_value() ? query_dims[1].dim_value() : 0;
      if (num_heads <= 0) {
        fail_shape_inference("num_heads must be a positive integer.");
      }
      if (v_head_size_attr == 0) {
        // V is as wide as the latent row, so the output width matches the query width.
        propagateShapeFromInputToOutput(ctx, 0, 0);
      } else if (q_hidden_size > 0) {
        if (q_hidden_size % num_heads != 0) {
          fail_shape_inference("Query hidden size must be divisible by num_heads.");
        }
        int64_t head_size = q_hidden_size / num_heads;
        if (v_head_size_attr > head_size) {
          fail_shape_inference("v_head_size must not exceed head_size.");
        }
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = query_dims[0];
        output_shape.add_dim()->set_dim_value(num_heads * v_head_size_attr);
        updateOutputShape(ctx, 0, output_shape);
      } else {
        // Symbolic query hidden size: only the token dimension is known.
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = query_dims[0];
        output_shape.add_dim();
        updateOutputShape(ctx, 0, output_shape);
      }
    } else if (ctx.hasInput(2)) {
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      propagateShapeFromInputToOutput(ctx, 0, 0);
    } else {  // packed QKV
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      int64_t num_heads = getAttribute(ctx, "num_heads", 0);
      int64_t kv_num_heads = getAttribute(ctx, "kv_num_heads", 0);
      int64_t hidden_size = query_dims[1].dim_value();
      if (hidden_size <= 0 || num_heads <= 0 || kv_num_heads < 0) {
        fail_shape_inference("Invalid hidden size or number of heads. Hidden size, num_heads and kv_num_heads must be positive integers.");
      } else if (hidden_size % (num_heads + 2 * kv_num_heads) != 0) {
        fail_shape_inference("Hidden size must be divisible by (num_heads + 2 * kv_num_heads).");
      }
      int64_t head_size = hidden_size / (num_heads + 2 * kv_num_heads);
      output_shape.add_dim()->set_dim_value(head_size * num_heads);
      updateOutputShape(ctx, 0, output_shape);
    }
  }

  // Shape inference for KV Cache output tensors
  if (ctx.getNumOutputs() > 1) {  // has kv cache output
    if (is_latent_kv) {
      // A single physical cache: there is no value cache to alias out.
      if (ctx.getNumOutputs() > 2) {
        fail_shape_inference("value_cache_out must be absent when kv_cache_layout is 'LATENT'.");
      }
    } else if (ctx.getNumOutputs() != 3) {
      fail_shape_inference("Key cache and value cache output tensors must be both present or both absent.");
    } else if (!ctx.hasInput(4)) {
      // value_cache is schema-optional (it must be absent for LATENT), so a SEPARATE node could omit it
      // while still declaring value_cache_out. Fail with a clear message instead of reading input 4.
      fail_shape_inference("value_cache is required when value_cache_out is present.");
    }
    // Types: the cache outputs alias the cache inputs, so their element type comes from inputs 3/4
    // (T_CACHE) rather than from the query (T) — the two differ for a quantized cache. This has to
    // run before the shape propagation below, which requires the output TypeProto to already be a
    // tensor type.
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 3, 1);
    // shapes
    auto& key_cache_shape = getInputShape(ctx, 3);
    auto& key_cache_dims = key_cache_shape.dim();
    if (key_cache_dims.size() != 4) {
      fail_shape_inference("The block-based KV cache inputs shall be 4 dimensions");
    }
    // KV cache in and out share the same buffer, thus they have the same shape
    ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 3, 1);

    if (ctx.getNumOutputs() > 2) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 4, 2);
      ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 4, 2);
    }
  }
}

ONNX_MS_OPERATOR_SET_SCHEMA(
    PagedAttention, 1,
    OpSchema()
        .SetDoc(PagedAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads for q", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of attention heads for k and v", AttributeProto::INT)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("softcap",
              "Softcap value for attention weights. Default value is 0.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("local_window_size",
              "left_window_size for local attention (like Mistral). Default value is -1 meaning unused.",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("is_causal",
              "Whether the attention mask is causal (bottom-right aligned). Default value is 1. "
              "Set to 0 for a block drafter whose query tokens attend to each other bidirectionally; "
              "local_window_size then bounds the mask on the left only.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_interleaved",
              "Rotate using interleaved pattern. Default value is 0 (False).",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("qk_norm_epsilon",
              "Epsilon used by the Q/K RMSNorm when 'q_norm_weight' and 'k_norm_weight' are provided. "
              "Default value is 1e-6.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("k_quant_type",
              "Quantization granularity of the key cache: 'NONE', 'PER_TENSOR' or 'PER_CHANNEL'. "
              "Must be non-'NONE' exactly when 'key_cache' has a quantized element type, and then "
              "'k_scale' is required. Default value is 'NONE'.",
              AttributeProto::STRING,
              std::string("NONE"))
        .Attr("v_quant_type",
              "Quantization granularity of the value cache: 'NONE', 'PER_TENSOR' or 'PER_CHANNEL'. "
              "Must be non-'NONE' exactly when 'value_cache' has a quantized element type, and then "
              "'v_scale' is required. Default value is 'NONE'.",
              AttributeProto::STRING,
              std::string("NONE"))
        .Attr("k_cache_dtype",
              "Logical element type stored in 'key_cache', named after the ONNX element type it denotes: '' "
              "(the default) means the cache tensor's own element type is also the logical type. 'float16', "
              "'bfloat16', 'int8' and 'float8e4m3fn' name that same type explicitly and must agree with the "
              "tensor. 'int4' and 'float4e2m1' name sub-byte types packed two per byte into a uint8 cache, "
              "where the last cache dimension holds (head_size + 1) / 2 bytes and logical element 2*i "
              "occupies the low-order bits of byte i. Every value is a signed, zero-symmetric type: "
              "quantization uses a scale with no zero point, so unsigned logical types are not expressible.",
              AttributeProto::STRING,
              std::string(""))
        .Attr("v_cache_dtype",
              "Logical element type stored in 'value_cache', with the same values and packing rule as "
              "'k_cache_dtype'. Default value is '' (use the cache tensor's element type).",
              AttributeProto::STRING,
              std::string(""))
        .Attr("kv_cache_layout",
              "Physical layout of the KV cache: 'SEPARATE' or 'LATENT'. 'SEPARATE' (the default) uses "
              "distinct 'key_cache' and 'value_cache' tensors. 'LATENT' selects absorbed Multi-head Latent "
              "Attention: there is a single cache, 'value' and 'value_cache' must be absent, 'kv_num_heads' "
              "must be 1, and V for every head is the leading 'v_head_size' channels of the same 'key_cache' "
              "row that supplies K. Default value is 'SEPARATE'.",
              AttributeProto::STRING,
              std::string("SEPARATE"))
        .Attr("v_head_size",
              "Width of the value head, which may be narrower than head_size. Only valid when "
              "'kv_cache_layout' is 'LATENT' (DeepSeek-V3 uses head_size=576 and v_head_size=512). When "
              "v_head_size differs from head_size the 'scale' attribute is required, because the "
              "1/sqrt(head_size) default no longer matches the pre-absorption head width. Default value is 0, "
              "meaning the same as head_size.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_offset",
              "First channel within head_size covered by rotary embedding, so RoPE is applied to "
              "[rotary_offset, rotary_offset + rotary_dim) and channels outside that range are copied "
              "through. Must be a multiple of 8. MLA sets this to kv_lora_rank so that RoPE only touches the "
              "positional suffix of the latent row. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (num_tokens, hidden_size), or packed QKV with shape (num_tokens, d) "
               "where d is (num_heads * head_size + 2 * kv_num_heads * head_size).",
               "T")
        .Input(1,
               "key",
               "Key with shape (num_tokens, kv_hidden_size) ",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (num_tokens, kv_hidden_size). Must be absent when 'kv_cache_layout' is 'LATENT'.",
               "T",
               OpSchema::Optional)
        .Input(3,
               "key_cache",
               "Block-based key cache with shape (num_blocks, block_size, kv_num_heads, cache_head_size), where "
               "cache_head_size is (head_size + 1) / 2 for packed INT4 and head_size otherwise. This is updated in "
               "place within the op. When 'kv_cache_layout' is 'LATENT' this is the only cache, and V is read from its "
               "leading v_head_size channels.",
               "T_CACHE")
        .Input(4,
               "value_cache",
               "Block-based value cache with shape (num_blocks, block_size, kv_num_heads, cache_head_size), where "
               "cache_head_size is (head_size + 1) / 2 for packed INT4 and head_size otherwise. This is updated "
               "in place within the op. This should be the same shape as key_cache. Must be absent when "
               "'kv_cache_layout' is 'LATENT'.",
               "T_CACHE",
               OpSchema::Optional)
        .Input(5,
               "cumulative_sequence_length",
               "A tensor with shape (batch_size + 1). It specifies the cumulative sequence lengths between the packed "
               "entries in Q/K/V.",
               "S")
        .Input(6,
               "past_seqlens",
               "A tensor with shape (batch_size). It specifies the past lengths of cached sequences. "
               "CUDA sanitizes malformed values on device: an invalid value suppresses that sequence's cache writes "
               "and produces zero output. A backend requiring exact host lengths may instead return INVALID_ARGUMENT "
               "when attention_metadata is absent.",
               "S")
        .Input(7,
               "block_table",
               "2D tensor with shape (batch_size, max_blocks_per_sequence) that maps each sequence in the batch to its"
               "corresponding blocks in the KV cache. -1 denotes an unmapped page. CUDA converts values below -1 or "
               "greater than or equal to num_blocks to the unmapped sentinel before resolving cache addresses; mapped "
               "pages remain attendable.",
               "S")
        .Input(8,
               "cos_cache",
               "2D tensor with shape (max total seqlen, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Input(9,
               "sin_cache",
               "2D tensor with shape (max total seqlen, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Input(10,
               "slot_mapping",
               "1D tensor with shape (num_tokens). For each query token, the flat slot index "
               "(block_id * block_size + offset_in_block) at which its key/value is written into the KV cache. "
               "A value of -1 skips the cache write for that token, which lets a scheduler suppress stores for "
               "prefix-cache hits or rejected speculative tokens. When absent, slots are derived from "
               "'past_seqlens', 'cumulative_sequence_length' and 'block_table' as before. 'block_table' is still "
               "required, because it defines the read path.",
               "S",
               OpSchema::Optional)
        .Input(11,
               "head_sink",
               "1D tensor with shape (num_heads). Each head has a learnable sink logit that participates in the "
               "softmax denominator but contributes no value, so attention can 'do nothing'.",
               "T",
               OpSchema::Optional)
        .Input(12,
               "q_norm_weight",
               "1D tensor with shape (head_size). RMSNorm gain applied to each query head before rotary "
               "embedding. Must be provided together with 'k_norm_weight'.",
               "T",
               OpSchema::Optional)
        .Input(13,
               "k_norm_weight",
               "1D tensor with shape (head_size). RMSNorm gain applied to each key head before rotary embedding "
               "and before the key is written to the KV cache. Must be provided together with 'q_norm_weight'.",
               "T",
               OpSchema::Optional)
        .Input(14,
               "k_scale",
               "Dequantization scale of the key cache. Shape is (1) when 'k_quant_type' is 'PER_TENSOR' and "
               "(kv_num_heads, 1, head_size) when it is 'PER_CHANNEL'. Quantization is symmetric (no zero point).",
               "T_KV_SCALE",
               OpSchema::Optional)
        .Input(15,
               "v_scale",
               "Dequantization scale of the value cache. Shape is (1) when 'v_quant_type' is 'PER_TENSOR' and "
               "(kv_num_heads, 1, head_size) when it is 'PER_CHANNEL'. Quantization is symmetric (no zero point).",
               "T_KV_SCALE",
               OpSchema::Optional)
        .Input(16,
               "attention_metadata",
               "1D tensor with shape (2) or (3) holding [max_query_len_bound, max_kv_len_bound, "
               "optional max_kv_len_lower_bound] in CPU memory. "
               "max_query_len_bound is an upper bound on the number of new tokens any one sequence "
               "contributes; max_kv_len_bound is an upper bound on past_seqlens[i] + query_len[i]. Both are "
               "replay-wide upper bounds, never exact per-step values: they must hold for every step this node "
               "-- or a CUDA Graph capturing it -- will serve, and 0 means 'unknown'. They may only select the "
               "backend and size launch dimensions and workspaces; they never enter a mask comparison, so "
               "over-estimating only costs empty work. max_kv_len_lower_bound is a replay-wide lower bound "
               "on the largest per-sequence KV length in the batch and 0 means 'unknown'. It is a "
               "provider-neutral performance hint; omitting it preserves the shape-(2) contract and disables "
               "optimizations that require a lower bound unless the op reads exact lengths back from the device. "
               "The op can otherwise obtain the upper bounds only by copying "
               "'cumulative_sequence_length' and 'past_seqlens' back from the device and synchronizing the "
               "stream on every call, which stalls the pipeline once per node per step and makes the op "
               "impossible to capture into a CUDA Graph. Schedulers already track these bounds on the host, so "
               "supplying them is normally free. When absent, the op falls back to the device readback. "
               "The upper bounds are trusted: an under-sized bound violates the contract and may omit "
               "attention work.",
               "S",
               OpSchema::Optional)
        .Output(0,
                "output",
                "2D output tensor with shape (num_tokens, num_heads * v_head_size), which is "
                "(num_tokens, hidden_size) unless 'kv_cache_layout' is 'LATENT' with a narrower v_head_size.",
                "T")
        .Output(1,
                "key_cache_out",
                "Aliases key_cache with the same shape and element type, including its packed dimension for INT4.",
                "T_CACHE",
                OpSchema::Optional)
        .Output(2,
                "value_cache_out",
                "Aliases value_cache with the same shape and element type, including its packed dimension for INT4. "
                "Must be absent when 'kv_cache_layout' is 'LATENT'.",
                "T_CACHE",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("T_CACHE",
                        {"tensor(float16)", "tensor(bfloat16)", "tensor(int8)", "tensor(float8e4m3fn)", "tensor(uint8)"},
                        "Constrain the KV cache to float or quantized tensors.")
        .TypeConstraint("T_KV_SCALE", {"tensor(float)"}, "Constrain KV cache scales to float tensors.")
        .TypeConstraint("S", {"tensor(int32)"}, "Constrain Positional inputs to int tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PagedAttentionTypeAndShapeInference(ctx);
        }));

static OpSchema PackedSparseAttentionIndexerMergeSchema() {
  return OpSchema()
      .SetDoc(
          "Stable union of a mapped base selection and either [range_start, range_end) "
          "or an additional selection prefix. "
          "Valid prefixes contain nonnegative indices. Padding is ignored. Outputs preserve "
          "first-occurrence order and are padded with -1. Status is 0 for success, 1 for invalid "
          "device metadata, or 2 for capacity overflow; failures return count 0 and all -1. "
          "The caller supplies a common index namespace and guarantees causal visibility.")
      .Attr("policy_mode", "append_range or append_indices.", AttributeProto::STRING)
      .Attr("max_output_entries", "Fixed output width, in [1, 2^30].", AttributeProto::INT)
      .Input(0, "base_indices", "Cached selections, shape (R, C).", "T")
      .Input(1, "base_counts", "Valid base-prefix lengths, shape (R).", "T")
      .Input(2, "base_row_indices", "Output-query to cached-row mapping, shape (N).", "T")
      .Input(3, "range_starts", "Inclusive nonnegative starts, shape (N).", "T", OpSchema::Optional)
      .Input(4, "range_ends", "Exclusive ends not less than starts, shape (N).", "T", OpSchema::Optional)
      .Input(5, "additional_indices", "Required for append_indices: selections of shape (N, A); absent for append_range.",
             "T", OpSchema::Optional)
      .Input(6, "additional_counts", "Required for append_indices: valid prefix lengths of shape (N); absent for append_range.",
             "T", OpSchema::Optional)
      .Output(0, "selected_indices", "Merged indices, shape (N, max_output_entries).", "T")
      .Output(1, "selected_counts", "Merged prefix lengths, shape (N).", "T")
      .Output(2, "status", "Per-query device validation status, shape (N).", "T")
      .TypeConstraint("T", {"tensor(int32)"}, "Integer selection tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        const auto* policy = ctx.getAttribute("policy_mode");
        const auto* capacity = ctx.getAttribute("max_output_entries");
        if (!policy || (policy->s() != "append_range" && policy->s() != "append_indices") || !capacity || capacity->i() <= 0 ||
            capacity->i() > (int64_t{1} << 30)) {
          fail_shape_inference("Invalid PackedSparseAttentionIndexerMerge policy or capacity");
        }
        for (size_t output = 0; output < 3; ++output) propagateElemTypeFromInputToOutput(ctx, 0, output);
        if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() != 2) {
          fail_shape_inference("base_indices must have rank 2");
        }
        const bool append_indices = policy->s() == "append_indices";
        for (size_t index = 3; index <= 6; ++index) {
          const bool provided = index < ctx.getNumInputs() && ctx.getInputType(index) != nullptr;
          if (provided != (append_indices ? index >= 5 : index <= 4)) {
            fail_shape_inference("Merge inputs must match policy_mode");
          }
        }
        const auto check_rows = [&](size_t input, int rank, size_t reference) {
          if (!hasInputShape(ctx, input)) return;
          const auto& shape = getInputShape(ctx, input);
          if (shape.dim_size() != rank) fail_shape_inference("Merge input ", input, " must have rank ", rank);
          if (hasInputShape(ctx, reference)) {
            const auto& expected = getInputShape(ctx, reference);
            if (expected.dim_size() > 0 && shape.dim(0).has_dim_value() && expected.dim(0).has_dim_value() &&
                shape.dim(0).dim_value() != expected.dim(0).dim_value()) {
              fail_shape_inference("Merge input ", input, " has a mismatched row dimension");
            }
          }
        };
        check_rows(1, 1, 0);
        check_rows(2, 1, 2);
        if (append_indices) {
          check_rows(5, 2, 2);
          check_rows(6, 1, 2);
        } else {
          check_rows(3, 1, 2);
          check_rows(4, 1, 2);
        }
        if (hasInputShape(ctx, 2)) {
          const auto& rows = getInputShape(ctx, 2);
          if (rows.dim_size() != 1) fail_shape_inference("base_row_indices must have rank 1");
          auto* shape = getOutputShape(ctx, 0);
          *shape->add_dim() = rows.dim(0);
          shape->add_dim()->set_dim_value(capacity->i());
          propagateShapeFromInputToOutput(ctx, 2, 1);
          propagateShapeFromInputToOutput(ctx, 2, 2);
        }
      });
}

ONNX_MS_OPERATOR_SET_SCHEMA(PackedSparseAttentionIndexerMerge, 1, PackedSparseAttentionIndexerMergeSchema());

constexpr const char* SparsePagedAttention_ver1_doc = R"DOC(
Selected-index attention over the PagedAttention main K/V cache.

Selection is supplied by an external indexer. selected_only attends selected entries;
local_plus_selected combines a main-cache local window and selected entries in one softmax.
Selected indices are request-local logical positions in the main cache or in a contiguous
auxiliary cache, according to selected_kv_source.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    SparsePagedAttention, 1,
    OpSchema()
        .SetDoc(SparsePagedAttention_ver1_doc)
        .Attr("num_heads", "Number of query heads.", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of main K/V heads.", AttributeProto::INT)
        .Attr("scale", "Attention scale; defaults to 1/sqrt(head_size).",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("softcap", "Optional tanh softcap for attention logits.",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("is_causal", "Apply causal filtering to main-cache reads.",
              AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("local_window_size", "Left main-cache window size; -1 means all visible tokens.",
              AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("attention_mode", "'selected_only' or 'local_plus_selected'.",
              AttributeProto::STRING, std::string("selected_only"))
        .Attr("selected_kv_source", "'main' or 'auxiliary'.",
              AttributeProto::STRING, std::string("main"))
        .Attr("auxiliary_cache_layout", "Version 1 supports only 'contiguous'.",
              AttributeProto::STRING, std::string("contiguous"))
        .Attr("auxiliary_kv_shared", "When 1, auxiliary_key supplies both K and V.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("do_rotary", "Apply rotary embedding to current Q/K.", AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("rotary_interleaved", "Use interleaved rotary embedding.", AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("rotary_offset", "First head channel covered by rotary embedding.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("qk_norm_epsilon", "QK RMSNorm epsilon.", AttributeProto::FLOAT, 1e-6f)
        .Attr("k_quant_type", "Main key-cache quantization: NONE, PER_TENSOR, or PER_CHANNEL.",
              AttributeProto::STRING, std::string("NONE"))
        .Attr("v_quant_type", "Main value-cache quantization: NONE, PER_TENSOR, or PER_CHANNEL.",
              AttributeProto::STRING, std::string("NONE"))
        .Input(0, "query", "Query or packed QKV, shaped as for PagedAttention.", "T")
        .Input(1, "key", "Current key.", "T", OpSchema::Optional)
        .Input(2, "value", "Current value.", "T", OpSchema::Optional)
        .Input(3, "key_cache", "Paged main key cache.", "T_CACHE")
        .Input(4, "value_cache", "Paged main value cache.", "T_CACHE")
        .Input(5, "cumulative_sequence_length", "Packed query offsets.", "S")
        .Input(6, "past_seqlens", "Past main-cache lengths.", "S")
        .Input(7, "block_table", "Logical-to-physical main-cache block table.", "S")
        .Input(8, "slot_mapping", "Optional main-cache write slots; -1 suppresses a write.",
               "S", OpSchema::Optional)
        .Input(9, "selected_indices",
               "Request-local positions, shape (token_count, max_selected_entries); unused entries are -1.", "S")
        .Input(10, "selected_counts", "Selected entry count per query token.", "S")
        .Input(11, "auxiliary_key",
               "Shape (batch_size, capacity, kv_num_heads_or_one, head_size).",
               "T", OpSchema::Optional)
        .Input(12, "auxiliary_value", "Same shape as auxiliary_key.", "T", OpSchema::Optional)
        .Input(13, "auxiliary_lengths", "Valid auxiliary length per request.", "S", OpSchema::Optional)
        .Input(14, "cos_cache", "Rotary cosine cache.", "T", OpSchema::Optional)
        .Input(15, "sin_cache", "Rotary sine cache.", "T", OpSchema::Optional)
        .Input(16, "head_sink", "Optional softmax sink logit per query head.", "T", OpSchema::Optional)
        .Input(17, "q_norm_weight", "QK-Norm query weight.", "T", OpSchema::Optional)
        .Input(18, "k_norm_weight", "QK-Norm key weight.", "T", OpSchema::Optional)
        .Input(19, "k_scale", "Main key-cache quantization scale.", "T_KV_SCALE", OpSchema::Optional)
        .Input(20, "v_scale", "Main value-cache quantization scale.", "T_KV_SCALE", OpSchema::Optional)
        .Input(21, "attention_metadata",
               "CPU tensor [max_query_len, max_local_main_len, max_selected_entries, max_auxiliary_len, "
               "max_combined_attention_len].",
               "S", OpSchema::Optional)
        .Output(0, "output", "Shape (token_count, num_heads * head_size).", "T")
        .Output(1, "key_cache_out", "In-place alias of key_cache.", "T_CACHE", OpSchema::Optional)
        .Output(2, "value_cache_out", "In-place alias of value_cache.", "T_CACHE", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)"}, "Activation type.")
        .TypeConstraint("T_CACHE", {"tensor(float16)", "tensor(bfloat16)", "tensor(int8)"},
                        "Main cache storage type.")
        .TypeConstraint("T_KV_SCALE", {"tensor(float)"}, "Main cache scale type.")
        .TypeConstraint("S", {"tensor(int32)"}, "Index and length type.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PagedAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* SparseAttention_ver1_doc = R"DOC(
Block Sparse Attention used in Phi-3-small (https://arxiv.org/pdf/2404.14219).

It is inspired by Sparse Transformers (https://arxiv.org/pdf/1904.10509) and BigBird (https://arxiv.org/pdf/2007.14062).

block_mask can be used to configure sparse layout for different head.
When number of sparse layout is 1, all heads have same sparse layout. Otherwise, different layouts are used cyclically.
For example, given 4 layouts (S0, S1, S2, S3), 8 heads will have layouts like (S0, S1, S2, S3, S0, S1, S2, S3).

The block_row_indices and block_col_indices are the CSR representation of block mask. The block_col_indices might contain
paddings at the right side when different layout has different number of non-zeros in block mask.

An example of block mask with 2 layouts where each layout is 4 x 4 blocks:
  [[[1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 1]],

   [[1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 1, 1]]]

The corresponding CSR format:
  block_col_indices = [[0,  0,  1,  1,  2,  1,  2,  3, -1], [0,  0,  1,  0,  1,  2,  0,  2,  3]]
  block_row_indices = [[0, 1, 3, 5, 8], [0, 1, 3, 6, 9]]

When do_rotary is True, cos_cache and sin_cache are required. Note that the maximum sequence length supported by cos
or sin cache can be different from the maximum sequence length used by kv cache.

Only supports unidirectional attention with cache of past key and value in linear buffers.

For performance, past_key and present_key share same memory buffer, and past_value and present_value too.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    SparseAttention, 1,
    OpSchema()
        .SetDoc(SparseAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads for query", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of attention heads for key and value", AttributeProto::INT)
        .Attr("scale", "Scaling factor applied prior to softmax. The default value is 1/sqrt(head_size)", AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("sparse_block_size", "Number of tokens per sparse block. Choices: 16, 32, 64, 128", AttributeProto::INT)
        .Attr("do_rotary", "Whether to use rotary position embedding. Default value is 0.", AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_interleaved", "Rotary use interleaved pattern or not. Default value is 0.", AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, sequence_length, num_heads * head_size), or packed QKV with shape is"
               "(batch_size, sequence_length, d) where d is (num_heads + 2 * kv_num_heads) * head_size.",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, sequence_length, kv_num_heads * head_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, sequence_length, kv_num_heads * head_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "past_key",
               "Key cache with shape (batch_size, kv_num_heads, max_cache_sequence_length, head_size)",
               "T")
        .Input(4,
               "past_value",
               "Value cache with shape (batch_size, kv_num_heads, max_cache_sequence_length, head_size)",
               "T")
        .Input(5,
               "block_row_indices",
               "The row indices of CSR format of block mask with shape (num_layout, max_blocks + 1)."
               "The num_heads is divisible by num_layout, and max_blocks is max_sequence_length / sparse_block_size.",
               "M")
        .Input(6,
               "block_col_indices",
               "The col indices of CSR format of block mask with shape (num_layout, max_nnz_blocks)."
               "The max_nnz_blocks is the maximum number of non-zeros per layout in block mask.",
               "M")
        .Input(7,
               "total_sequence_length",
               "Scalar tensor of maximum total sequence length (past_sequence_length + sequence_length) among keys.",
               "M")
        .Input(8,
               "key_total_sequence_lengths",
               "1D tensor with shape (batch_size) where each value is total sequence length of key excluding paddings.",
               "M")
        .Input(9,
               "cos_cache",
               "Cos cache of rotary with shape (max_rotary_sequence_length, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Input(10,
               "sin_cache",
               "Sin cache of rotary with shape (max_rotary_sequence_length, head_size / 2).",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, num_heads * head_size)",
                "T")
        .Output(1,
                "present_key",
                "Updated key cache with shape (batch_size, kv_num_heads, max_cache_sequence_length, head_size).",
                "T")
        .Output(2,
                "present_value",
                "Updated value cache with shape (batch_size, kv_num_heads, max_cache_sequence_length, head_size).",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain integer type.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          SparseAttentionTypeAndShapeInference(ctx, 3);
        }));

namespace sai = ::onnxruntime::contrib::sparse_attention_indexer;

namespace {

bool SparseAttentionIndexerHasInput(ONNX_NAMESPACE::InferenceContext& ctx, int index) {
  return static_cast<size_t>(index) < ctx.getNumInputs() && ctx.getInputType(index) != nullptr;
}

// Copies a dimension (value or symbolic parameter) from an input shape into an output shape.
void SparseAttentionIndexerAppendDim(ONNX_NAMESPACE::TensorShapeProto& shape,
                                     const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  *shape.add_dim() = dim;
}

const ONNX_NAMESPACE::TensorShapeProto* SparseAttentionIndexerShape(ONNX_NAMESPACE::InferenceContext& ctx, int index,
                                                                    int expected_rank) {
  if (!SparseAttentionIndexerHasInput(ctx, index) || !hasInputShape(ctx, index)) {
    return nullptr;
  }
  const auto& shape = getInputShape(ctx, index);
  if (shape.dim_size() != expected_rank) {
    fail_shape_inference("SparseAttentionIndexer: input ", index, " must have rank ", expected_rank,
                         ", got rank ", shape.dim_size());
  }
  return &shape;
}

const ONNX_NAMESPACE::TensorShapeProto* SparseAttentionIndexerRotaryCacheShape(
    ONNX_NAMESPACE::InferenceContext& ctx, int index) {
  if (!SparseAttentionIndexerHasInput(ctx, index) || !hasInputShape(ctx, index)) {
    return nullptr;
  }
  const auto& shape = getInputShape(ctx, index);
  if (shape.dim_size() != 2 && shape.dim_size() != 3) {
    fail_shape_inference("SparseAttentionIndexer: input ", index, " must have rank 2 or 3, got rank ",
                         shape.dim_size());
  }
  return &shape;
}

}  // namespace

void SparseAttentionIndexerTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  const std::string policy_mode = getAttribute(ctx, "policy_mode", std::string());
  sai::Policy policy = sai::Policy::kQsa;
  if (!sai::TryParsePolicy(policy_mode, policy)) {
    fail_shape_inference("SparseAttentionIndexer: policy_mode must be 'qsa' or 'csa', got '", policy_mode, "'");
  }
  const bool is_qsa = policy == sai::Policy::kQsa;

  const int64_t compress_ratio = getAttribute(ctx, "compress_ratio", static_cast<int64_t>(0));
  if (compress_ratio <= 0 || compress_ratio > std::numeric_limits<int>::max()) {
    fail_shape_inference("SparseAttentionIndexer: compress_ratio must be in (0, INT_MAX], got ", compress_ratio);
  }

  const int64_t token_budget = getAttribute(ctx, "token_budget", static_cast<int64_t>(0));
  const int64_t index_topk = getAttribute(ctx, "index_topk", static_cast<int64_t>(0));
  if (is_qsa) {
    if (ctx.getAttribute("index_topk") != nullptr || ctx.getAttribute("head_weight_scale") != nullptr) {
      fail_shape_inference(
          "SparseAttentionIndexer: index_topk and head_weight_scale must not be set when policy_mode is 'qsa'");
    }
    if (token_budget <= 0 || token_budget % compress_ratio != 0 ||
        token_budget > std::numeric_limits<int>::max() - compress_ratio + 1) {
      fail_shape_inference(
          "SparseAttentionIndexer: policy_mode 'qsa' requires token_budget > 0, divisible by "
          "compress_ratio, and a selected capacity no greater than INT_MAX, got token_budget=",
          token_budget, " compress_ratio=", compress_ratio);
    }
  } else {
    if (ctx.getAttribute("token_budget") != nullptr) {
      fail_shape_inference("SparseAttentionIndexer: token_budget must not be set when policy_mode is 'csa'");
    }
    if (index_topk <= 0 || index_topk > std::numeric_limits<int>::max()) {
      fail_shape_inference("SparseAttentionIndexer: policy_mode 'csa' requires index_topk in (0, INT_MAX], got ",
                           index_topk);
    }
  }

  // Strict policy input validation: every slot of the inactive policy must be omitted, and every
  // required slot of the active policy must be provided. The QSA mask is optional because an
  // omitted mask requests prefix-causal visibility.
  constexpr int kCsaOnlyInputs[] = {sai::kGate, sai::kPositionBias, sai::kHeadWeights,
                                    sai::kPositionIds, sai::kPastProjBuffer};
  for (int index = sai::kQuery; index <= sai::kSinCache; ++index) {
    if (is_qsa && index == sai::kKey) {
      continue;
    }
    if (!SparseAttentionIndexerHasInput(ctx, index)) {
      fail_shape_inference("SparseAttentionIndexer: input ", index, " is required for every policy_mode");
    }
  }
  if (!SparseAttentionIndexerHasInput(ctx, sai::kPastKey)) {
    fail_shape_inference("SparseAttentionIndexer: past_key is required for every policy_mode");
  }
  if (!is_qsa && SparseAttentionIndexerHasInput(ctx, sai::kMask)) {
    fail_shape_inference("SparseAttentionIndexer: input ", sai::kMask,
                         " must be omitted when policy_mode is 'csa'");
  }
  for (int index : kCsaOnlyInputs) {
    if (SparseAttentionIndexerHasInput(ctx, index) == is_qsa) {
      fail_shape_inference("SparseAttentionIndexer: input ", index,
                           is_qsa ? " must be omitted when policy_mode is 'qsa'"
                                  : " is required when policy_mode is 'csa'");
    }
  }
  const size_t expected_outputs = is_qsa ? sai::kQsaOutputCount : sai::kCsaOutputCount;
  if (ctx.getNumOutputs() != expected_outputs) {
    fail_shape_inference("SparseAttentionIndexer: policy_mode '", policy_mode, "' requires exactly ",
                         expected_outputs, " declared outputs, got ", ctx.getNumOutputs());
  }
  updateOutputElemType(ctx, sai::kSelectedIndices, ONNX_NAMESPACE::TensorProto_DataType_INT32);

  const bool has_key = SparseAttentionIndexerHasInput(ctx, sai::kKey);
  (void)SparseAttentionIndexerShape(ctx, sai::kKey, 3);
  (void)SparseAttentionIndexerShape(ctx, sai::kKeyNormWeight, 1);
  (void)SparseAttentionIndexerRotaryCacheShape(ctx, sai::kCosCache);
  (void)SparseAttentionIndexerRotaryCacheShape(ctx, sai::kSinCache);

  const auto* query_shape = SparseAttentionIndexerShape(ctx, sai::kQuery, 3);
  if (query_shape == nullptr) {
    return;
  }
  const auto& batch_dim = query_shape->dim(0);
  const auto& sequence_dim = query_shape->dim(1);
  const auto& query_width_dim = query_shape->dim(2);
  const auto* query_norm_shape = SparseAttentionIndexerShape(ctx, sai::kQueryNormWeight, 1);
  if (query_width_dim.has_dim_value() && query_width_dim.dim_value() <= 0) {
    fail_shape_inference("SparseAttentionIndexer: query width must be > 0, got ", query_width_dim.dim_value());
  }
  if (query_width_dim.has_dim_value() && query_width_dim.dim_value() > std::numeric_limits<int>::max()) {
    fail_shape_inference("SparseAttentionIndexer: query width must be no greater than INT_MAX");
  }
  if (query_norm_shape == nullptr) {
    return;
  }
  const auto& head_size_dim = query_norm_shape->dim(0);
  if (head_size_dim.has_dim_value() && head_size_dim.dim_value() <= 0) {
    fail_shape_inference("SparseAttentionIndexer: head_size must be > 0, got ", head_size_dim.dim_value());
  }
  if (query_width_dim.has_dim_value() && head_size_dim.has_dim_value() && head_size_dim.dim_value() > 0) {
    const int64_t query_width = query_width_dim.dim_value();
    const int64_t head_size = head_size_dim.dim_value();
    if (query_width % head_size != 0) {
      fail_shape_inference("SparseAttentionIndexer: query width must be divisible by head_size");
    }
    if (is_qsa && !has_key && query_width / head_size < 2) {
      fail_shape_inference("SparseAttentionIndexer: packed QK input must contain at least one query head and one key");
    }
  }
  if (!is_qsa && head_size_dim.has_dim_value() && head_size_dim.dim_value() > std::numeric_limits<int64_t>::max() / 2) {
    fail_shape_inference("SparseAttentionIndexer: 2 * head_size exceeds INT64_MAX");
  }

  const int64_t capacity = sai::SelectedCapacity(policy, token_budget, index_topk, compress_ratio);
  ONNX_NAMESPACE::TensorShapeProto selected_shape;
  SparseAttentionIndexerAppendDim(selected_shape, batch_dim);
  SparseAttentionIndexerAppendDim(selected_shape, sequence_dim);
  selected_shape.add_dim()->set_dim_value(capacity);
  updateOutputShape(ctx, sai::kSelectedIndices, selected_shape);

  if (is_qsa) {
    (void)SparseAttentionIndexerShape(ctx, sai::kMask, 2);
    // ctx.getNumOutputs() == 2 was enforced above, so index 1 is in range.
    propagateElemTypeFromInputToOutput(ctx, sai::kQuery, sai::kPresentKey);
    const auto* past_key_shape = SparseAttentionIndexerShape(ctx, sai::kPastKey, 3);
    if (past_key_shape != nullptr) {
      (void)SparseAttentionIndexerShape(ctx, sai::kPastSequenceLength, 1);
      ONNX_NAMESPACE::TensorShapeProto present_shape;
      SparseAttentionIndexerAppendDim(present_shape, batch_dim);
      if (SparseAttentionIndexerHasInput(ctx, sai::kPastSequenceLength)) {
        SparseAttentionIndexerAppendDim(present_shape, past_key_shape->dim(1));
      } else {
        auto* total_dim = present_shape.add_dim();
        if (past_key_shape->dim(1).has_dim_value() && sequence_dim.has_dim_value()) {
          const int64_t past_length = past_key_shape->dim(1).dim_value();
          const int64_t sequence_length = sequence_dim.dim_value();
          if (past_length > std::numeric_limits<int64_t>::max() - sequence_length) {
            fail_shape_inference("SparseAttentionIndexer: present key sequence length exceeds INT64_MAX");
          }
          total_dim->set_dim_value(past_length + sequence_length);
        }
      }
      SparseAttentionIndexerAppendDim(present_shape, head_size_dim);
      updateOutputShape(ctx, sai::kPresentKey, present_shape);
    }
    return;
  }

  // ctx.getNumOutputs() == 3 was enforced above, so indices 1 and 2 are in range.
  propagateElemTypeFromInputToOutput(ctx, sai::kQuery, sai::kPresentKey);
  propagateElemTypeFromInputToOutput(ctx, sai::kQuery, sai::kPresentProjBuffer);

  const auto* past_compressed_shape = SparseAttentionIndexerShape(ctx, sai::kPastKey, 3);
  (void)SparseAttentionIndexerShape(ctx, sai::kPastSequenceLength, 1);
  const auto* past_buffer_shape = SparseAttentionIndexerShape(ctx, sai::kPastProjBuffer, 4);
  (void)SparseAttentionIndexerShape(ctx, sai::kGate, 3);
  (void)SparseAttentionIndexerShape(ctx, sai::kPositionBias, 2);
  (void)SparseAttentionIndexerShape(ctx, sai::kHeadWeights, 3);
  (void)SparseAttentionIndexerShape(ctx, sai::kPositionIds, 2);

  if (past_buffer_shape != nullptr && past_buffer_shape->dim(0).has_dim_value() &&
      past_buffer_shape->dim(0).dim_value() != 2) {
    fail_shape_inference("SparseAttentionIndexer: past_proj_buffer dimension 0 must be 2, got ",
                         past_buffer_shape->dim(0).dim_value());
  }
  if (past_buffer_shape != nullptr && past_buffer_shape->dim(1).has_dim_value() &&
      batch_dim.has_dim_value() && past_buffer_shape->dim(1).dim_value() != batch_dim.dim_value()) {
    fail_shape_inference("SparseAttentionIndexer: past_proj_buffer dimension 1 must equal batch_size");
  }
  if (past_buffer_shape != nullptr && past_buffer_shape->dim(3).has_dim_value() &&
      head_size_dim.has_dim_value() &&
      past_buffer_shape->dim(3).dim_value() != 2 * head_size_dim.dim_value()) {
    fail_shape_inference("SparseAttentionIndexer: past_proj_buffer dimension 3 must equal 2 * head_size");
  }

  sai::CsaWindowPlan plan;
  const bool plan_known = past_buffer_shape != nullptr && past_buffer_shape->dim(2).has_dim_value() &&
                          sequence_dim.has_dim_value() &&
                          sai::TryComputeCsaWindowPlan(past_buffer_shape->dim(2).dim_value(),
                                                       sequence_dim.dim_value(), compress_ratio, plan);
  if (past_buffer_shape != nullptr && past_buffer_shape->dim(2).has_dim_value() &&
      sequence_dim.has_dim_value() && !plan_known) {
    fail_shape_inference(
        "SparseAttentionIndexer: past_proj_buffer sequence length must be in [0, 2 * compress_ratio), got ",
        past_buffer_shape->dim(2).dim_value());
  }

  if (past_compressed_shape != nullptr) {
    ONNX_NAMESPACE::TensorShapeProto present_shape;
    SparseAttentionIndexerAppendDim(present_shape, batch_dim);
    auto* entry_dim = present_shape.add_dim();
    if (SparseAttentionIndexerHasInput(ctx, sai::kPastSequenceLength)) {
      *entry_dim = past_compressed_shape->dim(1);
    } else if (plan_known && past_compressed_shape->dim(1).has_dim_value()) {
      const int64_t past_length = past_compressed_shape->dim(1).dim_value();
      if (past_length > std::numeric_limits<int64_t>::max() - plan.new_window_count) {
        fail_shape_inference("SparseAttentionIndexer: present compressed-key length exceeds INT64_MAX");
      }
      entry_dim->set_dim_value(past_length + plan.new_window_count);
    }
    SparseAttentionIndexerAppendDim(present_shape, head_size_dim);
    updateOutputShape(ctx, sai::kPresentKey, present_shape);
  }

  if (past_buffer_shape != nullptr) {
    ONNX_NAMESPACE::TensorShapeProto buffer_shape;
    SparseAttentionIndexerAppendDim(buffer_shape, past_buffer_shape->dim(0));
    SparseAttentionIndexerAppendDim(buffer_shape, batch_dim);
    auto* buffer_dim = buffer_shape.add_dim();
    if (plan_known) {
      buffer_dim->set_dim_value(plan.present_buffer_length);
    }
    SparseAttentionIndexerAppendDim(buffer_shape, past_buffer_shape->dim(3));
    updateOutputShape(ctx, sai::kPresentProjBuffer, buffer_shape);
  }
}

constexpr const char* SparseAttentionIndexer_ver1_doc = R"DOC(
Selects, for every query token, the sparse-attention candidates that the following attention
operator is allowed to read. It covers the two indexer flavours used by recent sparse-attention
decoders, chosen with the policy_mode attribute:

  policy_mode = "qsa" ("query sparse attention" token indexer)
    Groups the tokens that are visible to a query into complete blocks of compress_ratio tokens,
    mean-pools the indexer keys of every block, normalizes and rotates the pooled key, scores it
    against the query heads with sum_h ReLU(q_h . k), keeps the token_budget / compress_ratio
    highest scoring blocks and emits the token indices of those blocks followed by the visible
    tokens of the trailing incomplete block.

  policy_mode = "csa" ("compressed sparse attention" block indexer)
    Compresses every compress_ratio consecutive tokens into one entry with a softmax-gated pooling
    over a window of 2 * compress_ratio slots (the previous window contributes its "Ca" half and
    the current window its "Cb" half), normalizes and rotates the entry, appends it to the
    compressed-key state, scores the queries against every compressed entry with
    sum_h w_h * ReLU(q_h . k), masks the entries a query may not attend to and emits the index_topk
    highest scoring entry indices. If a channel's softmax denominator is non-positive or non-finite
    (for example, all logits are -inf or any logit is +inf), its pooled value is zero.

Common contract:
  * selected_indices is int32 with a fixed capacity that only depends on attributes:
    token_budget + compress_ratio - 1 for "qsa" and index_topk for "csa". Unused entries are -1,
    so no output size depends on the data and no device-to-host synchronization is required.
  * All state is explicit in the graph. Nothing is cached inside the operator.
  * Rotary embeddings reuse the precomputed cos_cache / sin_cache tables, which are indexed by
    absolute key position. "qsa" applies the half-rotation of the model's (M)RoPE to the leading
    rotary_dim = cos_cache.shape[2] channels. "csa" applies its trailing rotary to the last
    2 * cos_cache.shape[2] channels, with each cos/sin entry covering two consecutive channels.
  * query_norm_weight and key_norm_weight are the effective RMSNorm multipliers. Models that store a zero-centered gamma
    (the normalized value is multiplied by 1 + gamma) must fold the addition into this initializer.
  * Accumulation, pooling, softmax, normalization and scoring are performed in float32 and the
    result is rounded once to the tensor element type.
  * Ties in the top-k selection are broken by the smaller entry index, and the emitted entries are
    ordered by decreasing score, so the result is deterministic.

State layout for policy_mode = "csa": the two slices of past_proj_buffer hold the key and gate
projections that have not been folded into a compressed entry yet. When their length is >=
compress_ratio, the first compress_ratio tokens are the previous complete window (the "Ca" operand
of the next window) and the remainder is the current incomplete window; when it is < compress_ratio
there is no previous complete window and the whole buffer is the incomplete window. The length is
therefore always in [0, 2 * compress_ratio), and the number of compressed entries emitted by a call
is known from the input shapes alone. position_bias is re-applied to the buffered gates, so the gate
projection plane stores the raw projection.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    SparseAttentionIndexer, 1,
    OpSchema()
        .SetDoc(SparseAttentionIndexer_ver1_doc)
        .Attr("policy_mode",
              "Indexer policy. Must be exactly 'qsa' (token indexer) or 'csa' (compressed block indexer).",
              AttributeProto::STRING)
        .Attr("compress_ratio",
              "Number of consecutive tokens folded into one compressed block. Must be > 0.",
              AttributeProto::INT)
        .Attr("token_budget",
              "Only for policy_mode 'qsa': maximum number of tokens selected from complete blocks. "
              "Must be > 0 and divisible by compress_ratio. Must be omitted when policy_mode is 'csa'.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("index_topk",
              "Only for policy_mode 'csa': number of compressed entries selected per query. Must be > 0. "
              "Must be omitted when policy_mode is 'qsa'.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("epsilon",
              "Epsilon of the RMS normalization applied to queries and compressed keys. Default is 1e-6.",
              AttributeProto::FLOAT,
              1.0e-6f)
        .Attr("scale",
              "Scale applied to the per-head ReLU scores. Default is 1/sqrt(head_size).",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("head_weight_scale",
              "Only for policy_mode 'csa': scale applied to head_weights. Default is 1/sqrt(num_heads). "
              "Must be omitted when policy_mode is 'qsa'.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Indexer queries with shape (batch_size, sequence_length, num_heads * head_size), before "
               "normalization, logical reshape, and rotary embedding. For policy_mode 'qsa', when key is omitted, "
               "this input instead packs query followed by key along the last dimension and has shape "
               "(batch_size, sequence_length, (num_heads + 1) * head_size).",
               "T")
        .Input(1,
               "key",
               "Indexer key projection of the new tokens. Shape is (batch_size, sequence_length, head_size) "
               "for policy_mode 'qsa' and (batch_size, sequence_length, 2 * head_size) for policy_mode 'csa', "
               "where the first head_size channels are the Ca series and the last head_size channels the Cb series. "
               "May be omitted for policy_mode 'qsa' when query contains packed QK.",
               "T",
               OpSchema::Optional)
        .Input(2,
               "query_norm_weight",
               "Effective RMSNorm multiplier of the queries, with shape (head_size).",
               "T")
        .Input(3,
               "key_norm_weight",
               "Effective RMSNorm multiplier of the compressed keys, with shape (head_size).",
               "T")
        .Input(4,
               "cos_cache",
               "Cosine rotary table indexed by absolute key position, shared across the batch with shape "
               "(max_rotary_sequence_length, rotary_width) or request-specific with shape "
               "(batch_size, max_rotary_sequence_length, rotary_width).",
               "T")
        .Input(5,
               "sin_cache",
               "Sine rotary table with the same shape as cos_cache.",
               "T")
        .Input(6,
               "mask",
               "Only for policy_mode 'qsa': INT64 padding mask with shape "
               "(batch_size, total_sequence_length). Nonzero entries are visible subject to causal masking. "
               "total_sequence_length is past_sequence_length + sequence_length. When omitted, every position "
               "through past_sequence_length plus the current query index is visible.",
               "TB",
               OpSchema::Optional)
        .Input(7,
               "past_key",
               "Cached indexer keys. For policy_mode 'qsa', these are raw keys; for 'csa', they are compressed "
               "keys. Shape is (batch_size, past_sequence_length, head_size), or "
               "(batch_size, max_cache_length, head_size) when a valid past_sequence_length is provided.",
               "T")
        .Input(8,
               "gate",
               "Only for policy_mode 'csa': gate projection of the new tokens with shape "
               "(batch_size, sequence_length, 2 * head_size).",
               "T",
               OpSchema::Optional)
        .Input(9,
               "position_bias",
               "Only for policy_mode 'csa': per-slot gate bias with shape (compress_ratio, 2 * head_size).",
               "T",
               OpSchema::Optional)
        .Input(10,
               "head_weights",
               "Only for policy_mode 'csa': per-head score weights with shape "
               "(batch_size, sequence_length, num_heads).",
               "T",
               OpSchema::Optional)
        .Input(11,
               "position_ids",
               "Only for policy_mode 'csa': absolute position of every query with shape "
               "(batch_size, sequence_length).",
               "I",
               OpSchema::Optional)
        .Input(12,
               "past_sequence_length",
               "Optional one-element CPU tensor containing the number of valid rows in past_key. For policy_mode "
               "'csa', the value is the number of compressed keys. When provided, past_key and present_key have "
               "the same max-capacity shape and may share their buffer.",
               "M",
               OpSchema::Optional)
        .Input(13,
               "past_proj_buffer",
               "Only for policy_mode 'csa': buffered key and gate projections packed along dimension 0, with shape "
               "(2, batch_size, buffer_length, 2 * head_size). Slice 0 contains keys and slice 1 contains gates. "
               "buffer_length is in [0, 2 * compress_ratio).",
               "T",
               OpSchema::Optional)
        .Output(0,
                "selected_indices",
                "Selected entries with shape (batch_size, sequence_length, capacity). capacity is "
                "token_budget + compress_ratio - 1 for policy_mode 'qsa', where the values are token indices "
                "into the key cache, and index_topk for policy_mode 'csa', where the values are compressed "
                "entry indices. Unused entries are -1.",
                "M")
        .Output(1,
                "present_key",
                "Updated raw key cache for policy_mode 'qsa' or compressed-key cache for 'csa'. Without "
                "past_sequence_length its sequence dimension grows by the entries emitted by this call. When "
                "past_sequence_length is provided, its shape matches the max-capacity past_key and the two tensors "
                "may share a buffer.",
                "T")
        .Output(2,
                "present_proj_buffer",
                "Only for policy_mode 'csa': updated packed key/gate projection buffer with shape "
                "(2, batch_size, present_buffer_length, 2 * head_size).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain floating point tensors to float, float16 and bfloat16.")
        .TypeConstraint("TB", {"tensor(int64)"}, "Constrain the mask to 64-bit integer tensors.")
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain position ids to 64-bit integer tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain indices and cache lengths to 32-bit integer tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          SparseAttentionIndexerTypeAndShapeInference(ctx);
        }));

namespace psai = ::onnxruntime::contrib::packed_sparse_attention_indexer;

namespace {

bool PackedSparseAttentionIndexerHasInput(ONNX_NAMESPACE::InferenceContext& ctx, int index) {
  return static_cast<size_t>(index) < ctx.getNumInputs() && ctx.getInputType(index) != nullptr;
}

const ONNX_NAMESPACE::TensorShapeProto* PackedSparseAttentionIndexerShape(ONNX_NAMESPACE::InferenceContext& ctx,
                                                                          int index, int expected_rank) {
  if (!PackedSparseAttentionIndexerHasInput(ctx, index) || !hasInputShape(ctx, index)) {
    return nullptr;
  }
  const auto& shape = getInputShape(ctx, index);
  if (shape.dim_size() != expected_rank) {
    fail_shape_inference("PackedSparseAttentionIndexer: input ", index, " must have rank ", expected_rank,
                         ", got rank ", shape.dim_size());
  }
  return &shape;
}

}  // namespace

void PackedSparseAttentionIndexerTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  const std::string policy_mode = getAttribute(ctx, "policy_mode", std::string());
  psai::Policy policy = psai::Policy::kQsa;
  if (!psai::TryParsePolicy(policy_mode, policy)) {
    fail_shape_inference("PackedSparseAttentionIndexer: policy_mode must be 'qsa' or 'csa', got '", policy_mode,
                         "'");
  }
  const bool is_qsa = policy == psai::Policy::kQsa;
  const std::string state_policy = getAttribute(ctx, "indexshare_state_policy", std::string("default"));
  if ((state_policy != "default" && state_policy != "hold" && state_policy != "advance") ||
      (is_qsa && state_policy == "advance")) {
    fail_shape_inference("PackedSparseAttentionIndexer: invalid indexshare_state_policy for policy_mode");
  }

  const bool has_indexshare = PackedSparseAttentionIndexerHasInput(ctx, psai::kIndexShareMode);
  const int64_t output_capacity = getAttribute(ctx, "max_output_entries", static_cast<int64_t>(0));
  for (int index : {psai::kMergedIndices, psai::kMergedCounts, psai::kMergedStatus}) {
    if (PackedSparseAttentionIndexerHasInput(ctx, index) != has_indexshare) {
      fail_shape_inference("PackedSparseAttentionIndexer: all IndexShare inputs must be provided together");
    }
  }
  if (has_indexshare != (output_capacity > 0) || output_capacity < 0 ||
      output_capacity > std::numeric_limits<int>::max()) {
    fail_shape_inference("PackedSparseAttentionIndexer: IndexShare requires a positive max_output_entries");
  }
  if (has_indexshare != ctx.hasOutput(psai::kIndexShareStatus)) {
    fail_shape_inference("PackedSparseAttentionIndexer: IndexShare requires its status output");
  }

  const int64_t compress_ratio = getAttribute(ctx, "compress_ratio", static_cast<int64_t>(0));
  if (compress_ratio <= 0 ||
      compress_ratio > (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2) {
    fail_shape_inference(
        "PackedSparseAttentionIndexer: compress_ratio must be positive and produce a generic "
        "buffer capacity no greater than INT_MAX, got ",
        compress_ratio);
  }
  const int64_t state_capacity = getAttribute(ctx, "state_capacity", static_cast<int64_t>(0));
  if (state_capacity <= 0 || state_capacity > std::numeric_limits<int>::max()) {
    fail_shape_inference("PackedSparseAttentionIndexer: state_capacity must be in (0, INT_MAX], got ",
                         state_capacity);
  }
  const int64_t state_update_capacity =
      getAttribute(ctx, "state_update_capacity", static_cast<int64_t>(0));
  if (state_update_capacity < 0 || state_update_capacity > 8) {
    fail_shape_inference("PackedSparseAttentionIndexer: state_update_capacity must be in [0, 8], got ",
                         state_update_capacity);
  }

  const int64_t token_budget = getAttribute(ctx, "token_budget", static_cast<int64_t>(0));
  const int64_t index_topk = getAttribute(ctx, "index_topk", static_cast<int64_t>(0));
  if (is_qsa) {
    if (ctx.getAttribute("index_topk") != nullptr || ctx.getAttribute("head_weight_scale") != nullptr) {
      fail_shape_inference(
          "PackedSparseAttentionIndexer: index_topk and head_weight_scale must not be set when policy_mode is "
          "'qsa'");
    }
    if (token_budget <= 0 || token_budget % compress_ratio != 0 ||
        token_budget > std::numeric_limits<int>::max() - compress_ratio + 1) {
      fail_shape_inference(
          "PackedSparseAttentionIndexer: policy_mode 'qsa' requires token_budget > 0, divisible by "
          "compress_ratio, and a selected capacity no greater than INT_MAX, got token_budget=",
          token_budget, " compress_ratio=", compress_ratio);
    }
  } else {
    if (ctx.getAttribute("token_budget") != nullptr) {
      fail_shape_inference("PackedSparseAttentionIndexer: token_budget must not be set when policy_mode is 'csa'");
    }
    if (index_topk <= 0 || index_topk > std::numeric_limits<int>::max()) {
      fail_shape_inference(
          "PackedSparseAttentionIndexer: policy_mode 'csa' requires index_topk in (0, INT_MAX], got ", index_topk);
    }
  }

  // Strict policy input validation: every fixed slot required by every policy must be provided;
  // csa-only slots must be provided iff policy_mode is 'csa'.
  for (int index : {psai::kQuery, psai::kQueryNormWeight, psai::kKeyNormWeight, psai::kCosCache, psai::kSinCache,
                    psai::kCumulativeSequenceLengths, psai::kPastSequenceLengths, psai::kPastKeyState,
                    psai::kPastKvBuffer, psai::kPastStateLengths}) {
    if (!PackedSparseAttentionIndexerHasInput(ctx, index)) {
      fail_shape_inference("PackedSparseAttentionIndexer: input ", index, " is required for every policy_mode");
    }
  }
  const bool has_key = PackedSparseAttentionIndexerHasInput(ctx, psai::kKey);
  if (!is_qsa && !has_key) {
    fail_shape_inference("PackedSparseAttentionIndexer: key is required for policy_mode 'csa'");
  }
  for (int index : {psai::kGate, psai::kPositionBias, psai::kHeadWeights, psai::kPastGateBuffer}) {
    if (PackedSparseAttentionIndexerHasInput(ctx, index) == is_qsa) {
      fail_shape_inference("PackedSparseAttentionIndexer: input ", index,
                           is_qsa ? " must be omitted when policy_mode is 'qsa'"
                                  : " is required when policy_mode is 'csa'");
    }
  }
  if (PackedSparseAttentionIndexerHasInput(ctx, psai::kPositionIds) == is_qsa) {
    fail_shape_inference("PackedSparseAttentionIndexer: input ", psai::kPositionIds,
                         is_qsa ? " (position_ids) must be omitted when policy_mode is 'qsa'"
                                : " (position_ids) is required when policy_mode is 'csa'");
  }
  const bool has_capture_count = PackedSparseAttentionIndexerHasInput(ctx, psai::kStateUpdateCaptureCount);
  const bool has_state_update_active = PackedSparseAttentionIndexerHasInput(ctx, psai::kStateUpdateActive);
  const bool has_state_update_output = ctx.hasOutput(psai::kStateUpdate);
  const bool has_mid_kv_buffer = ctx.hasOutput(psai::kMidKvBuffer);
  const bool has_mid_state_lengths = ctx.hasOutput(psai::kMidStateLengths);
  const bool has_validation_status = ctx.hasOutput(psai::kValidationStatus);
  if (has_mid_kv_buffer != has_mid_state_lengths) {
    fail_shape_inference("PackedSparseAttentionIndexer: mid_kv_buffer and mid_state_lengths must be connected together");
  }
  if (!is_qsa && (has_mid_kv_buffer || has_validation_status)) {
    fail_shape_inference("PackedSparseAttentionIndexer: mid-state and validation outputs are only supported for qsa");
  }
  if (!is_qsa && (state_update_capacity > 0 || has_capture_count ||
                  has_state_update_active || has_state_update_output)) {
    fail_shape_inference("PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
  }
  if (state_update_capacity > 0 && !has_capture_count) {
    fail_shape_inference(
        "PackedSparseAttentionIndexer: state_update_capture_count is required when "
        "state_update_capacity is positive");
  }

  if (ctx.getNumOutputs() < static_cast<size_t>(psai::kFixedOutputCount) ||
      ctx.getNumOutputs() > static_cast<size_t>(psai::kMaxOutputCount)) {
    fail_shape_inference("PackedSparseAttentionIndexer: expected ", psai::kFixedOutputCount, " to ",
                         psai::kMaxOutputCount, " declared outputs, got ", ctx.getNumOutputs());
  }
  if (ctx.hasOutput(psai::kPresentGateBuffer) == is_qsa) {
    fail_shape_inference("PackedSparseAttentionIndexer: output ", psai::kPresentGateBuffer,
                         is_qsa ? " (present_gate_buffer) must be omitted when policy_mode is 'qsa'"
                                : " (present_gate_buffer) is required when policy_mode is 'csa'");
  }
  updateOutputElemType(ctx, psai::kSelectedIndices, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  updateOutputElemType(ctx, psai::kSelectedCounts, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  if (has_indexshare) updateOutputElemType(ctx, psai::kIndexShareStatus, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  updateOutputElemType(ctx, psai::kPresentStateLengths, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  if (has_mid_state_lengths) {
    updateOutputElemType(ctx, psai::kMidStateLengths, ONNX_NAMESPACE::TensorProto_DataType_INT32);
    propagateShapeFromInputToOutput(ctx, psai::kPastStateLengths, psai::kMidStateLengths);
  }
  if (has_validation_status) {
    updateOutputElemType(ctx, psai::kValidationStatus, ONNX_NAMESPACE::TensorProto_DataType_INT32);
    propagateShapeFromInputToOutput(ctx, psai::kPastSequenceLengths, psai::kValidationStatus);
  }
  const bool has_state_status_output = ctx.hasOutput(psai::kStateStatus);
  if (has_state_status_output) {
    updateOutputElemType(ctx, psai::kStateStatus, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  }
  propagateElemTypeFromInputToOutput(ctx, psai::kPastKeyState, psai::kPresentKeyState);
  propagateElemTypeFromInputToOutput(ctx, psai::kPastKvBuffer, psai::kPresentKvBuffer);
  if (has_mid_kv_buffer) {
    propagateElemTypeFromInputToOutput(ctx, psai::kPastKvBuffer, psai::kMidKvBuffer);
    propagateShapeFromInputToOutput(ctx, psai::kPastKvBuffer, psai::kMidKvBuffer);
  }
  for (int prefix = 0; prefix < 2; ++prefix) {
    const int buffer = psai::kMidKvBuffer2 + 2 * prefix;
    const int lengths = buffer + 1;
    if (ctx.hasOutput(buffer) != ctx.hasOutput(lengths))
      fail_shape_inference("PackedSparseAttentionIndexer: prefix buffer and lengths must be connected together");
    if (!ctx.hasOutput(buffer)) continue;
    if (!is_qsa) fail_shape_inference("PackedSparseAttentionIndexer: prefix outputs require policy_mode='qsa'");
    propagateElemTypeFromInputToOutput(ctx, psai::kPastKvBuffer, buffer);
    propagateShapeFromInputToOutput(ctx, psai::kPastKvBuffer, buffer);
    updateOutputElemType(ctx, lengths, ONNX_NAMESPACE::TensorProto_DataType_INT32);
    propagateShapeFromInputToOutput(ctx, psai::kPastStateLengths, lengths);
  }
  if (!is_qsa) {
    // present_gate_buffer keeps its fixed positional slot (with an empty name) for policy_mode
    // 'qsa'; only propagate its type/shape when it is actually produced.
    propagateElemTypeFromInputToOutput(ctx, psai::kPastGateBuffer, psai::kPresentGateBuffer);
  }
  if (has_state_update_output) {
    propagateElemTypeFromInputToOutput(ctx, psai::kPastKvBuffer, psai::kStateUpdate);
  }

  const auto* query_shape = PackedSparseAttentionIndexerShape(ctx, psai::kQuery, 2);
  const auto* key_shape = PackedSparseAttentionIndexerShape(ctx, psai::kKey, 2);
  const auto* query_norm_shape = PackedSparseAttentionIndexerShape(ctx, psai::kQueryNormWeight, 1);
  const auto* key_norm_shape = PackedSparseAttentionIndexerShape(ctx, psai::kKeyNormWeight, 1);
  const auto* cumulative_shape =
      PackedSparseAttentionIndexerShape(ctx, psai::kCumulativeSequenceLengths, 1);
  const auto* past_sequence_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPastSequenceLengths, 1);
  const ONNX_NAMESPACE::TensorShapeProto* cos_shape = nullptr;
  const ONNX_NAMESPACE::TensorShapeProto* sin_shape = nullptr;
  for (const auto& [index, name, shape_out] :
       {std::tuple<int, const char*, const ONNX_NAMESPACE::TensorShapeProto**>{
            psai::kCosCache, "cos_cache", &cos_shape},
        {psai::kSinCache, "sin_cache", &sin_shape}}) {
    if (hasInputShape(ctx, index)) {
      const auto& shape = getInputShape(ctx, index);
      if (shape.dim_size() != 2 && shape.dim_size() != 3) {
        fail_shape_inference("PackedSparseAttentionIndexer: ", name, " must have rank 2 or 3, got rank ",
                             shape.dim_size());
      }
      *shape_out = &shape;
    }
  }
  const auto* key_state_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPastKeyState, 3);
  const auto* kv_buffer_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPastKvBuffer, 3);
  const auto* state_lengths_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPastStateLengths, 2);
  const ONNX_NAMESPACE::TensorShapeProto* gate_shape = nullptr;
  const ONNX_NAMESPACE::TensorShapeProto* position_bias_shape = nullptr;
  const ONNX_NAMESPACE::TensorShapeProto* head_weights_shape = nullptr;
  const ONNX_NAMESPACE::TensorShapeProto* gate_buffer_shape = nullptr;
  if (!is_qsa) {
    gate_shape = PackedSparseAttentionIndexerShape(ctx, psai::kGate, 2);
    position_bias_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPositionBias, 2);
    head_weights_shape = PackedSparseAttentionIndexerShape(ctx, psai::kHeadWeights, 2);
    gate_buffer_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPastGateBuffer, 3);
  }
  const ONNX_NAMESPACE::TensorShapeProto* position_ids_shape = nullptr;
  if (PackedSparseAttentionIndexerHasInput(ctx, psai::kPositionIds)) {
    position_ids_shape = PackedSparseAttentionIndexerShape(ctx, psai::kPositionIds, 1);
  }
  const auto* capture_count_shape =
      PackedSparseAttentionIndexerShape(ctx, psai::kStateUpdateCaptureCount, 1);
  const auto* state_update_active_shape =
      PackedSparseAttentionIndexerShape(ctx, psai::kStateUpdateActive, 1);
  const auto* mode_shape = PackedSparseAttentionIndexerShape(ctx, psai::kIndexShareMode, 1);
  const auto* merged_indices_shape = PackedSparseAttentionIndexerShape(ctx, psai::kMergedIndices, 2);
  const auto* merged_counts_shape = PackedSparseAttentionIndexerShape(ctx, psai::kMergedCounts, 1);
  const auto* merged_status_shape = PackedSparseAttentionIndexerShape(ctx, psai::kMergedStatus, 1);

  auto require_equal_dims = [](const ONNX_NAMESPACE::TensorShapeProto* lhs, int lhs_index,
                               const ONNX_NAMESPACE::TensorShapeProto* rhs, int rhs_index,
                               const char* description) {
    if (lhs != nullptr && rhs != nullptr && lhs->dim(lhs_index).has_dim_value() &&
        rhs->dim(rhs_index).has_dim_value() &&
        lhs->dim(lhs_index).dim_value() != rhs->dim(rhs_index).dim_value()) {
      fail_shape_inference("PackedSparseAttentionIndexer: ", description);
    }
  };
  auto require_dim_value = [](const ONNX_NAMESPACE::TensorShapeProto* shape, int index, int64_t expected,
                              const char* description) {
    if (shape != nullptr && shape->dim(index).has_dim_value() && shape->dim(index).dim_value() != expected) {
      fail_shape_inference("PackedSparseAttentionIndexer: ", description, " (", expected, "), got ",
                           shape->dim(index).dim_value());
    }
  };

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio);
  require_dim_value(mode_shape, 0, 1, "indexshare_mode must have shape {1}");
  require_dim_value(merged_indices_shape, 1, output_capacity, "merged_indices width must equal max_output_entries");
  require_equal_dims(merged_indices_shape, 0, merged_counts_shape, 0, "merged_counts must match merged_indices rows");
  require_equal_dims(merged_indices_shape, 0, merged_status_shape, 0, "merged_status must match merged_indices rows");
  if (merged_indices_shape != nullptr && merged_indices_shape->dim(0).has_dim_value() &&
      merged_indices_shape->dim(0).dim_value() != 0) {
    require_equal_dims(merged_indices_shape, 0, past_sequence_shape, 0,
                       "Nonempty merged_indices must have one row per request");
  }
  if (!has_indexshare) require_equal_dims(key_shape, 0, query_shape, 0, "key dimension 0 must equal query dimension 0");
  require_equal_dims(query_norm_shape, 0, key_norm_shape, 0,
                     "query_norm_weight and key_norm_weight dimensions must match");
  require_equal_dims(past_sequence_shape, 0, key_state_shape, 0,
                     "past_sequence_lengths dimension 0 must equal the state batch dimension");
  require_equal_dims(key_state_shape, 0, kv_buffer_shape, 0,
                     "past_key_state and past_kv_buffer batch dimensions must match");
  require_equal_dims(state_lengths_shape, 0, key_state_shape, 0,
                     "past_state_lengths dimension 0 must equal the state batch dimension");
  require_equal_dims(capture_count_shape, 0, key_state_shape, 0,
                     "state_update_capture_count dimension 0 must equal the state batch dimension");
  require_dim_value(state_update_active_shape, 0, 1, "state_update_active dimension 0 must equal 1");
  require_equal_dims(key_state_shape, 2, query_norm_shape, 0,
                     "past_key_state dimension 2 must equal head_size");
  require_dim_value(key_state_shape, 1, state_capacity, "past_key_state dimension 1 must equal state_capacity");
  require_dim_value(kv_buffer_shape, 1, buffer_capacity,
                    "past_kv_buffer dimension 1 must equal 2 * compress_ratio - 1");
  require_dim_value(state_lengths_shape, 1, psai::kStateLengthColumns,
                    "past_state_lengths dimension 1 must equal 2");
  if (cumulative_shape != nullptr && past_sequence_shape != nullptr &&
      cumulative_shape->dim(0).has_dim_value() && past_sequence_shape->dim(0).has_dim_value() &&
      cumulative_shape->dim(0).dim_value() != past_sequence_shape->dim(0).dim_value() + 1) {
    fail_shape_inference(
        "PackedSparseAttentionIndexer: cumulative_sequence_lengths dimension 0 must equal batch_size + 1");
  }

  if (query_norm_shape != nullptr && query_norm_shape->dim(0).has_dim_value()) {
    const int64_t head_size = query_norm_shape->dim(0).dim_value();
    if (head_size <= 0) {
      fail_shape_inference("PackedSparseAttentionIndexer: head_size must be > 0, got ", head_size);
    }
    if (!is_qsa && head_size > std::numeric_limits<int64_t>::max() / 2) {
      fail_shape_inference("PackedSparseAttentionIndexer: 2 * head_size exceeds INT64_MAX");
    }
    const int64_t width = is_qsa ? head_size : 2 * head_size;
    require_dim_value(key_shape, 1, width, "key dimension 1 must equal the policy-specific width");
    require_dim_value(kv_buffer_shape, 2, width,
                      "past_kv_buffer dimension 2 must equal the policy-specific width");
    if (!is_qsa) {
      require_dim_value(gate_shape, 1, width, "gate dimension 1 must equal 2 * head_size");
      require_dim_value(position_bias_shape, 1, width, "position_bias dimension 1 must equal 2 * head_size");
      require_dim_value(gate_buffer_shape, 2, width, "past_gate_buffer dimension 2 must equal 2 * head_size");
    }
  }
  if (!is_qsa) {
    require_equal_dims(key_shape, 0, gate_shape, 0, "gate dimension 0 must equal key dimension 0");
    require_equal_dims(key_shape, 0, position_ids_shape, 0, "position_ids dimension 0 must equal key dimension 0");
    if (!has_indexshare) require_equal_dims(gate_shape, 0, query_shape, 0, "gate dimension 0 must equal query dimension 0");
    require_equal_dims(head_weights_shape, 0, query_shape, 0,
                       "head_weights dimension 0 must equal query dimension 0");
    if (head_weights_shape != nullptr && query_shape != nullptr &&
        head_weights_shape->dim(1).has_dim_value() && query_shape->dim(1).has_dim_value() &&
        query_norm_shape != nullptr && query_norm_shape->dim(0).has_dim_value() &&
        query_norm_shape->dim(0).dim_value() > 0 &&
        query_shape->dim(1).dim_value() / query_norm_shape->dim(0).dim_value() !=
            head_weights_shape->dim(1).dim_value()) {
      fail_shape_inference("PackedSparseAttentionIndexer: head_weights dimension 1 must equal num_heads");
    }
    require_dim_value(position_bias_shape, 0, compress_ratio,
                      "position_bias dimension 0 must equal compress_ratio");
    require_equal_dims(gate_buffer_shape, 0, kv_buffer_shape, 0,
                       "past_gate_buffer and past_kv_buffer batch dimensions must match");
    require_equal_dims(gate_buffer_shape, 1, kv_buffer_shape, 1,
                       "past_gate_buffer and past_kv_buffer capacities must match");
    require_equal_dims(gate_buffer_shape, 2, kv_buffer_shape, 2,
                       "past_gate_buffer and past_kv_buffer widths must match");
  }
  if (!has_indexshare) require_equal_dims(position_ids_shape, 0, query_shape, 0,
                                          "position_ids dimension 0 must equal query dimension 0");
  if (cos_shape != nullptr && sin_shape != nullptr) {
    if (cos_shape->dim_size() != sin_shape->dim_size()) {
      fail_shape_inference("PackedSparseAttentionIndexer: cos_cache and sin_cache ranks must match");
    }
    for (int i = 0; i < cos_shape->dim_size(); ++i) {
      require_equal_dims(cos_shape, i, sin_shape, i, "cos_cache and sin_cache dimensions must match");
    }
  }
  for (const auto* cache_shape : {cos_shape, sin_shape}) {
    if (cache_shape == nullptr) {
      continue;
    }
    if (cache_shape->dim_size() == 3) {
      require_equal_dims(cache_shape, 0, past_sequence_shape, 0,
                         "batched rotary cache dimension 0 must equal batch_size");
    }
    const int position_dim = cache_shape->dim_size() - 2;
    const int rotary_dim = cache_shape->dim_size() - 1;
    if (cache_shape->dim(position_dim).has_dim_value() && cache_shape->dim(position_dim).dim_value() <= 0) {
      fail_shape_inference("PackedSparseAttentionIndexer: rotary cache max_position must be > 0");
    }
    if (cache_shape->dim(rotary_dim).has_dim_value()) {
      const int64_t rotary_width = cache_shape->dim(rotary_dim).dim_value();
      if (rotary_width <= 0 || (is_qsa && rotary_width % 2 != 0)) {
        fail_shape_inference("PackedSparseAttentionIndexer: invalid rotary cache width ", rotary_width);
      }
      if (query_norm_shape != nullptr && query_norm_shape->dim(0).has_dim_value()) {
        const int64_t head_size = query_norm_shape->dim(0).dim_value();
        if ((is_qsa && rotary_width > head_size) ||
            (!is_qsa && rotary_width > head_size / 2)) {
          fail_shape_inference("PackedSparseAttentionIndexer: rotary cache width is incompatible with head_size");
        }
      }
    }
  }

  if (query_shape != nullptr) {
    const auto& total_tokens_dim = query_shape->dim(0);
    const auto& query_width_dim = query_shape->dim(1);
    const int64_t packed_key_heads = is_qsa && !has_key ? 1 : 0;
    if (query_width_dim.has_dim_value() && query_norm_shape != nullptr &&
        query_norm_shape->dim(0).has_dim_value() &&
        query_width_dim.dim_value() <= packed_key_heads * query_norm_shape->dim(0).dim_value()) {
      fail_shape_inference("PackedSparseAttentionIndexer: query width must contain at least one query head",
                           has_key ? "" : " followed by one packed key head", ", got ",
                           query_width_dim.dim_value());
    }
    if (query_width_dim.has_dim_value() && query_norm_shape != nullptr &&
        query_norm_shape->dim(0).has_dim_value() && query_norm_shape->dim(0).dim_value() > 0 &&
        query_width_dim.dim_value() % query_norm_shape->dim(0).dim_value() != 0) {
      fail_shape_inference("PackedSparseAttentionIndexer: query width must be divisible by head_size");
    }

    const int64_t base_capacity = psai::SelectedCapacity(policy, token_budget, index_topk, compress_ratio);
    if (has_indexshare && output_capacity < base_capacity) {
      fail_shape_inference("PackedSparseAttentionIndexer: max_output_entries is smaller than the selection budget");
    }
    const int64_t capacity = has_indexshare ? output_capacity : base_capacity;
    ONNX_NAMESPACE::TensorShapeProto selected_shape;
    if (has_indexshare)
      selected_shape.add_dim();
    else
      SparseAttentionIndexerAppendDim(selected_shape, total_tokens_dim);
    selected_shape.add_dim()->set_dim_value(capacity);
    updateOutputShape(ctx, psai::kSelectedIndices, selected_shape);

    ONNX_NAMESPACE::TensorShapeProto counts_shape;
    if (has_indexshare)
      counts_shape.add_dim();
    else
      SparseAttentionIndexerAppendDim(counts_shape, total_tokens_dim);
    updateOutputShape(ctx, psai::kSelectedCounts, counts_shape);
    if (has_indexshare) updateOutputShape(ctx, psai::kIndexShareStatus, counts_shape);
  }

  // State never grows: present_* always has exactly the same fixed shape as past_*.
  if (key_state_shape != nullptr) {
    updateOutputShape(ctx, psai::kPresentKeyState, *key_state_shape);
  }
  if (kv_buffer_shape != nullptr) {
    updateOutputShape(ctx, psai::kPresentKvBuffer, *kv_buffer_shape);
  }
  if (gate_buffer_shape != nullptr) {
    updateOutputShape(ctx, psai::kPresentGateBuffer, *gate_buffer_shape);
  }
  if (state_lengths_shape != nullptr) {
    updateOutputShape(ctx, psai::kPresentStateLengths, *state_lengths_shape);
  }
  if (has_state_update_output && key_state_shape != nullptr) {
    ONNX_NAMESPACE::TensorShapeProto state_update_shape;
    *state_update_shape.add_dim() = key_state_shape->dim(0);
    state_update_shape.add_dim()->set_dim_value(state_update_capacity);
    *state_update_shape.add_dim() = key_state_shape->dim(2);
    updateOutputShape(ctx, psai::kStateUpdate, state_update_shape);
  }
  if (has_state_status_output && (key_state_shape != nullptr || past_sequence_shape != nullptr)) {
    ONNX_NAMESPACE::TensorShapeProto state_status_shape;
    *state_status_shape.add_dim() =
        key_state_shape != nullptr ? key_state_shape->dim(0) : past_sequence_shape->dim(0);
    updateOutputShape(ctx, psai::kStateStatus, state_status_shape);
  }
}

constexpr const char* PackedSparseAttentionIndexer_ver1_doc = R"DOC(
Packed/variable-length counterpart of SparseAttentionIndexer, for continuous-batching engines
(such as an OgaEngine-style PagedAttention model) that flatten every request's tokens into one
[total_tokens, ...] axis instead of a dense [batch_size, sequence_length, ...] axis. It selects, for
every packed query token, the sparse-attention candidates that a following SparsePagedAttention (or
similar) operator is allowed to read.

Unlike SparseAttentionIndexer, this operator:
  * takes packed query/key tensors plus cumulative_sequence_lengths (request boundaries) and
    past_sequence_lengths (per-request past length) instead of a dense batch and a dense mask;
  * derives ordinary causal visibility purely from that packed metadata -- there is no mask input;
  * uses a single generic set of state slots (past_key_state / past_kv_buffer / past_gate_buffer /
    past_state_lengths) for both policy_mode values, each with a shape that is fixed across calls
    (state never grows and is never concatenated); a step that would overflow the fixed capacity is
    rejected as a deterministic no-op on state rather than truncated or allowed to corrupt memory
    (the optional state_status output reports which requests were rejected and why);
  * additionally emits selected_counts, the exact number of active (non -1) entries per query, so
    that no downstream consumer needs to scan selected_indices for its query's true count.
  * for QSA, optionally emits the pending-state snapshot after the first request token and a
    per-request validation status for overflow or inconsistent state metadata.

Both policy_mode values keep the semantics of SparseAttentionIndexer, applied independently to each
request's own packed token range and fixed-capacity state slice:

  policy_mode = "qsa" ("query sparse attention" token indexer)
    Processes each request's new tokens sequentially: appends raw indexer keys to the generic
    pending buffer, and whenever it reaches compress_ratio tokens, mean-pools it, applies RMSNorm
    and key_norm_weight, applies the leading/split-half rotary convention at the block's first
    logical token position, and appends the prepared (already normalized and rotated) key to
    key_state. Queries are scored against every causally visible complete block with
    sum_h ReLU(q_h . k), the token_budget / compress_ratio highest scoring blocks are kept, and
    their token indices are emitted (request-local logical positions, i.e. the same numbering as
    past_sequence_lengths + local offset) followed by the causally visible tokens of the trailing
    incomplete block. There is no position_ids input: QSA positions are the request-local logical
    cache positions derived from past_sequence_lengths and cumulative_sequence_lengths.

  policy_mode = "csa" ("compressed sparse attention" block indexer)
    Applies the same window-plan arithmetic as SparseAttentionIndexer (overlap/leftover/new window
    count) independently per request, using that request's own buffer_length and new token count;
    every newly closed window is compressed with the softmax-gated Ca/Cb pooling, normalized,
    rotated and appended to key_state. Queries are scored against every causally visible compressed
    entry with sum_h w_h * ReLU(q_h . k) and the index_topk highest scoring entry indices are
    emitted.

Common contract:
  * selected_indices is int32 with a fixed capacity that only depends on attributes:
    token_budget + compress_ratio - 1 for "qsa" (values are request-local token positions into the
    main key/value cache, directly consumable by SparsePagedAttention configured with
    attention_mode="selected_only", selected_kv_source="main") and index_topk for "csa" (values are
    compressed-entry indices into key_state, directly consumable by SparsePagedAttention configured
    with attention_mode="local_plus_selected", selected_kv_source="auxiliary"; key_state is
    layout-compatible with a [batch_size, capacity, 1, head_size] auxiliary cache when K = V).
    Unused entries are -1 and selected_counts holds the exact number of used entries.
  * query_norm_weight and key_norm_weight are the effective RMSNorm multipliers, exactly as in
    SparseAttentionIndexer.
  * Accumulation, pooling, softmax, normalization and scoring are performed in float32 and the
    result is rounded once to the tensor element type.
  * Ties in the top-k selection are broken by the smaller entry index, and the emitted entries are
    ordered by decreasing score, so the result is deterministic.
  * cos_cache / sin_cache may be shared across the batch ([max_position, rotary_width]) or
    request-specific ([batch_size, max_position, rotary_width]).
  * cumulative_sequence_lengths, past_sequence_lengths and past_state_lengths are read directly by
    the device kernel; a zero-token request row (a repeated cumulative offset) is valid and simply
    contributes no query rows for that request.

OgaEngine integration note: this operator only defines the ORT contrib op; wiring
past_key_state / past_kv_buffer / past_gate_buffer / past_state_lengths as Engine-managed,
per-request fixed-size state (analogous to a paged auxiliary cache) is expected to happen in the
OgaEngine / Model Builder integration, which is out of scope for this operator definition.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    PackedSparseAttentionIndexer, 1,
    OpSchema()
        .SetDoc(PackedSparseAttentionIndexer_ver1_doc)
        .Attr("policy_mode",
              "Indexer policy. Must be exactly 'qsa' (token indexer) or 'csa' (compressed block indexer).",
              AttributeProto::STRING)
        .Attr("compress_ratio",
              "Number of consecutive tokens folded into one compressed/pooled entry. Must be > 0 and "
              "2 * compress_ratio - 1 must not exceed INT_MAX.",
              AttributeProto::INT)
        .Attr("state_capacity",
              "Fixed capacity (number of entries) of past_key_state / present_key_state. Must be > 0.",
              AttributeProto::INT)
        .Attr("state_update_capacity",
              "Only for policy_mode 'qsa': maximum number of leading token transitions captured per request. "
              "Must be in [0, 8]. Default is 0.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("token_budget",
              "Only for policy_mode 'qsa': maximum number of tokens selected from complete blocks. "
              "Must be > 0 and divisible by compress_ratio. Must be omitted when policy_mode is 'csa'.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("index_topk",
              "Only for policy_mode 'csa': number of compressed entries selected per query. Must be > 0. "
              "Must be omitted when policy_mode is 'qsa'.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("epsilon",
              "Epsilon of the RMS normalization applied to queries and compressed keys. Default is 1e-6.",
              AttributeProto::FLOAT,
              1.0e-6f)
        .Attr("scale",
              "Scale applied to the per-head ReLU scores. Default is 1/sqrt(head_size).",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("head_weight_scale",
              "Only for policy_mode 'csa': scale applied to head_weights. Default is 1/sqrt(num_heads). "
              "Must be omitted when policy_mode is 'qsa'.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Packed indexer queries with shape (total_tokens, num_heads * head_size), before normalization, "
               "logical reshape, and rotary embedding. For policy_mode 'qsa', key may be omitted and query then "
               "contains row-wise concatenated query and key projections with shape "
               "(total_tokens, (num_heads + 1) * head_size).",
               "T")
        .Input(1,
               "key",
               "Packed indexer key projection of the new tokens. Shape is (total_tokens, head_size) for "
               "policy_mode 'qsa' and (total_tokens, 2 * head_size) for policy_mode 'csa', where the first "
               "head_size channels are the Ca series and the last head_size channels the Cb series. May be "
               "omitted for policy_mode 'qsa' when query contains the packed query/key projection.",
               "T",
               OpSchema::Optional)
        .Input(2,
               "query_norm_weight",
               "Effective RMSNorm multiplier of the queries, with shape (head_size).",
               "T")
        .Input(3,
               "key_norm_weight",
               "Effective RMSNorm multiplier of the compressed keys, with shape (head_size).",
               "T")
        .Input(4,
               "cos_cache",
               "Cosine rotary table indexed by absolute key position, shared across the batch with shape "
               "(max_rotary_sequence_length, rotary_width) or request-specific with shape "
               "(batch_size, max_rotary_sequence_length, rotary_width).",
               "T")
        .Input(5,
               "sin_cache",
               "Sine rotary table with the same shape as cos_cache.",
               "T")
        .Input(6,
               "cumulative_sequence_lengths",
               "Device-resident packed request boundaries with shape (batch_size + 1); "
               "cumulative_sequence_lengths[0] must be 0 and cumulative_sequence_lengths[batch_size] must equal "
               "total_tokens. Request b owns rows [cumulative_sequence_lengths[b], "
               "cumulative_sequence_lengths[b + 1]) of query/key (a repeated offset is a valid zero-token row).",
               "M")
        .Input(7,
               "past_sequence_lengths",
               "Device-resident number of tokens already processed for each request before this call, with "
               "shape (batch_size). Used to derive request-local logical query positions for policy_mode 'qsa' "
               "and to validate state consistency.",
               "M")
        .Input(8,
               "gate",
               "Only for policy_mode 'csa': gate projection of the new tokens with shape "
               "(total_tokens, 2 * head_size).",
               "T",
               OpSchema::Optional)
        .Input(9,
               "position_bias",
               "Only for policy_mode 'csa': per-slot gate bias with shape (compress_ratio, 2 * head_size).",
               "T",
               OpSchema::Optional)
        .Input(10,
               "head_weights",
               "Only for policy_mode 'csa': per-head score weights with shape (total_tokens, num_heads).",
               "T",
               OpSchema::Optional)
        .Input(11,
               "position_ids",
               "Only for policy_mode 'csa': absolute position of every packed query, with shape (total_tokens).",
               "I",
               OpSchema::Optional)
        .Input(12,
               "past_key_state",
               "Generic fixed-capacity state: policy_mode 'qsa' stores prepared complete-block keys; "
               "policy_mode 'csa' stores compressed keys. Shape is (batch_size, state_capacity, head_size) and "
               "never changes across calls. CUDA QSA also supports FLOAT key state with BFLOAT16 "
               "query/key tensors. WebGPU requires key state and activations to have the same type.",
               "T_KEY")
        .Input(13,
               "past_kv_buffer",
               "Generic fixed-capacity pending-token buffer. Shape is (batch_size, 2 * compress_ratio - 1, "
               "head_size) for policy_mode 'qsa' (which only ever uses up to compress_ratio - 1 of these "
               "entries) and (batch_size, 2 * compress_ratio - 1, 2 * head_size) for policy_mode 'csa'.",
               "T")
        .Input(14,
               "past_gate_buffer",
               "Only for policy_mode 'csa': buffered gate projections with the same shape as past_kv_buffer.",
               "T",
               OpSchema::Optional)
        .Input(15,
               "past_state_lengths",
               "Generic per-request state length with shape (batch_size, 2). Column 0 is the key_state entry "
               "count (policy_mode 'qsa': complete-block count; 'csa': compressed-entry count); column 1 is the "
               "pending-buffer length (policy_mode 'qsa': incomplete-block length in [0, compress_ratio); 'csa': "
               "buffer length in [0, 2 * compress_ratio)).",
               "M")
        .Input(16,
               "state_update_capture_count",
               "Only for policy_mode 'qsa': number of leading token transitions to capture for each request, "
               "with shape (batch_size). Values are clamped to the request length and state_update_capacity. "
               "Required when state_update_capacity is positive.",
               "M",
               OpSchema::Optional)
        .Input(17,
               "state_update_active",
               "Only for policy_mode 'qsa': optional capture gate with shape (1). A zero value disables capture.",
               "M",
               OpSchema::Optional)
        .Input(18, "indexshare_mode", "CPU scalar [1]: 0 refresh, 1 reuse.", "M", OpSchema::Optional)
        .Input(19, "merged_indices", "Packed merged selection [N,max_output_entries].", "M", OpSchema::Optional)
        .Input(20, "merged_counts", "Merged counts [N].", "M", OpSchema::Optional)
        .Input(21, "merged_status", "Merged device status [N].", "M", OpSchema::Optional)
        .Attr("max_output_entries", "Fixed padded output width for the optional IndexShare interface.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("indexshare_state_policy", "Reuse state policy: default holds QSA and advances CSA; hold or advance overrides it.",
              AttributeProto::STRING, std::string("default"))
        .Output(0,
                "selected_indices",
                "Selected entries with shape (total_tokens, capacity). capacity is "
                "token_budget + compress_ratio - 1 for policy_mode 'qsa' (request-local token positions into the "
                "main key/value cache) and index_topk for policy_mode 'csa' (compressed entry indices into "
                "key_state). Unused entries are -1.",
                "M")
        .Output(1,
                "selected_counts",
                "Exact number of used (non -1) entries of selected_indices for every query, with shape "
                "(total_tokens).",
                "M")
        .Output(2,
                "present_key_state",
                "Updated generic key state, with the same fixed shape as past_key_state.",
                "T_KEY")
        .Output(3,
                "present_kv_buffer",
                "Updated generic pending-token buffer, with the same fixed shape as past_kv_buffer.",
                "T")
        .Output(4,
                "present_gate_buffer",
                "Only for policy_mode 'csa': updated gate buffer with the same fixed shape as past_gate_buffer.",
                "T",
                OpSchema::Optional)
        .Output(5,
                "present_state_lengths",
                "Updated generic per-request state length, with the same fixed shape as past_state_lengths.",
                "M")
        .Output(6,
                "state_update",
                "Only for policy_mode 'qsa': compact transition payloads with shape "
                "(batch_size, state_update_capacity, head_size). A token that completes a compression block "
                "stores the prepared block representative; any other captured token stores its raw key. "
                "Inactive and unused slots are zero.",
                "T",
                OpSchema::Optional)
        .Output(7, "indexshare_status", "Selection status [N], forwarded on reuse and zero on refresh.",
                "M", OpSchema::Optional)
        .Output(8,
                "state_status",
                "Device-side per-request state update status with shape (batch_size). 0: the request's state "
                "update was applied (or intentionally held by IndexShare reuse); 1: invalid device metadata "
                "(cumulative_sequence_lengths, past_sequence_lengths or past_state_lengths are inconsistent); "
                "2: the update would exceed state_capacity. A non-zero request leaves its state payloads "
                "unchanged, clamps out-of-range past_state_lengths into present_state_lengths, and emits an "
                "empty selection (selected_counts is 0 and selected_indices is -1). "
                "Intended for CUDA graph replay, where the host checks it after the replay instead of "
                "synchronizing inside the operator. Currently supported by the CUDA provider only.",
                "M",
                OpSchema::Optional)
        .Output(9, "mid_kv_buffer",
                "Only for CUDA policy_mode 'qsa' refresh: pending-buffer snapshot after the first token of each request. "
                "Not supported in IndexShare reuse mode.",
                "T", OpSchema::Optional)
        .Output(10, "mid_state_lengths",
                "Only for CUDA policy_mode 'qsa' refresh: state lengths after the first token of each request. "
                "Not supported in IndexShare reuse mode.",
                "M", OpSchema::Optional)
        .Output(11, "validation_status",
                "Only for CUDA policy_mode 'qsa' refresh: per-request overflow status; two-token rejection is encoded as 0x99. "
                "Not supported in IndexShare reuse mode.",
                "M", OpSchema::Optional)
        .Output(12, "mid_kv_buffer2", "CUDA QSA pending-buffer snapshot after at most two request tokens.",
                "T", OpSchema::Optional)
        .Output(13, "mid_state_lengths2", "CUDA QSA state lengths after at most two request tokens.",
                "M", OpSchema::Optional)
        .Output(14, "mid_kv_buffer3", "CUDA QSA pending-buffer snapshot after at most three request tokens.",
                "T", OpSchema::Optional)
        .Output(15, "mid_state_lengths3", "CUDA QSA state lengths after at most three request tokens.",
                "M", OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain floating point tensors to float, float16 and bfloat16.")
        .TypeConstraint("T_KEY",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain the generic key-state tensor to float, float16 and bfloat16.")
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain position ids to 64-bit integer tensors.")
        .TypeConstraint("M", {"tensor(int32)"},
                        "Constrain packed metadata, generic state lengths and selected indices/counts to 32-bit "
                        "integer tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PackedSparseAttentionIndexerTypeAndShapeInference(ctx);
        }));

constexpr const char* Longformer_Attention_doc = R"DOC(
Longformer Self Attention with a local context and a global context. Tokens attend locally: Each token
attends to its W previous tokens and W succeeding tokens with W being the window length. A selected few tokens
attend globally to all other tokens.

The attention mask is of shape (batch_size, sequence_length), where sequence_length is a multiple of 2W after padding.
Mask value < 0 (like -10000.0) means the token is masked, 0 otherwise.

Global attention flags have value 1 for the tokens attend globally and 0 otherwise.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    LongformerAttention, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(Longformer_Attention_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("window", "One sided attention windows length W, or half of total window length", AttributeProto::INT)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size", "T")
        .Input(1, "weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T")
        .Input(2, "bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(3, "mask", "Attention mask with shape (batch_size, sequence_length)", "T")
        .Input(4, "global_weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T")
        .Input(5, "global_bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(6, "global", "Global attention flags with shape (batch_size, sequence_length)", "G")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("G", {"tensor(int32)"}, "Constrain to integer types")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Decoder_Attention_doc = R"DOC(
This DecoderAttention supports self attention and cross attention, key and value cache, and key_padding_mask. The attention mask is not support at the moment.
Some boolean parameters are passed by runtime input for generic purpose
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderAttention, 1,
    OpSchema()
        .SetDoc(Decoder_Attention_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "query", "3D input tensor with shape (sequence_length, batch_size, hidden_size), hidden_size = num_heads * head_size", "T")
        .Input(1, "key", "3D input tensor with shape (total_sequence_length, batch_size, hidden_size)", "T")
        .Input(2, "q_weight", "2D input tensor with shape (hidden_size, hidden_size)", "T")
        .Input(3, "kv_weight", "2D input tensor with shape (hidden_size, 2 * hidden_size)", "T")
        .Input(4, "bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(5, "key_padding_mask", "2D input tensor with shape (batch_size, total_sequence_length)", "B", OpSchema::Optional)
        .Input(6, "key_cache", "input tensor with shape (batch_size, num_heads, sequence_length or total_sequence_length, head_size)", "T", OpSchema::Optional)    // self & cross
        .Input(7, "value_cache", "input tensor with shape (batch_size, num_heads, sequence_length or total_sequence_length, head_size)", "T", OpSchema::Optional)  // self & cross
        .Input(8, "static_kv", "If static_kv = true, cross-attention; else self-attention", "B")
        .Input(9, "use_past", "If use_past = true, use cache; else no cache", "B")
        .Input(10, "has_layer_state", "If has_layer_state = true, layer_state = {} or [a,b]; else layer_state = None", "B")
        .Input(11, "has_key_padding_mask", "has_key_padding_mask or not", "B")
        .Output(0, "output", "3D output tensor with shape (sequence_length, batch_size, hidden_size)", "T")
        .Output(1, "new_key_cache", "output tensor with shape (batch_size, num_heads, new sequence_length, head_size)", "T", OpSchema::Optional)    // self & cross
        .Output(2, "new_value_cache", "output tensor with shape (batch_size, num_heads, new sequence_length, head_size)", "T", OpSchema::Optional)  // self & cross
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float and float16 tensors.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain key_padding_mask to bool tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          DecoderAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* RotaryEmbedding_ver1_doc = R"DOC(
RotaryEmbedding is the implementation of rotary positional embeddings (RoPE). The positions are represented as rotation matrices
that are multiplied to query and key before the inner product of query and key is taken.
)DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(
    RotaryEmbedding, 1,
    OpSchema()
        .SetDoc(RotaryEmbedding_ver1_doc)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1.0",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("interleaved",
              "Indicates whether the input has real and imaginary parts interleaved. "
              "Default value is 0 (False), meaning the first half of the input consists of real values "
              "and the second half consists of imaginary values.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_embedding_dim",
              "Rotary embedding dimension. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("num_heads",
              "Number of attention heads. Default value is 0. Must use with rotary_embedding_dim",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("is_packed_batching",
              "ragged batch inputs or not. Default value is 0",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "3D tensor with shape (batch_size, sequence_length, hidden_size) or 4D with shape (batch_size, num_heads, sequence_length, head_size)",
               "T")
        .Input(1,
               "position_ids",
               "1D tensor with shape (1) or 2D tensor with shape (batch_size, sequence_length)",
               "M")
        .Input(2,
               "cos_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2) or (max_sequence_length, rotary_embedding_dim / 2)",
               "T")
        .Input(3,
               "sin_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2) or (max_sequence_length, rotary_embedding_dim / 2)",
               "T")
        .Output(0,
                "output",
                "tensor with same shape as input.",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("M", {"tensor(int64)"}, "Constrain input and output types to integer tensors")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* MRotaryEmbedding_ver1_doc = R"DOC(
MRotaryEmbedding is the fused implementation of Multimodal Rotary Positional Embeddings (M-RoPE) used by the
Qwen family of vision-language models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE).

Unlike standard RoPE which uses a single 1D position per token, M-RoPE derives three positions per token
(temporal T, height H, width W), each of which indexes into the same cos/sin cache. The half_rotary_embedding_dim
axis of the cache is partitioned into 3 contiguous or interleaved sections (specified by `mrope_section`); each
section is populated using the cos/sin values gathered with the corresponding T/H/W position, and the sections
are then concatenated (or interleaved) to produce a single per-token cos/sin vector of length
half_rotary_embedding_dim. The standard RoPE rotation (as in RotaryEmbedding) is then applied using this
combined vector.

For text-only tokens, T == H == W (all three position streams collapse to the ordinary sequential position),
so this op is a strict superset of RotaryEmbedding: setting `mrope_section` to a single full-width section
reduces this op to standard RoPE.

`mrope_layout` selects how the three sections are combined:
  - 0 (Sectioned / Chunked): the half_rotary_embedding_dim axis is split into 3 contiguous chunks according to
    `mrope_section` (i.e. [T]*section[0] + [H]*section[1] + [W]*section[2]). This is used by Qwen2-VL and
    Qwen2.5-VL.
  - 1 (Interleaved): the half_rotary_embedding_dim axis is filled starting from T at every position, then H
    overwrites every 3rd position starting at offset 1 for the first `section[1]*3` positions, and W overwrites
    every 3rd position starting at offset 2 for the first `section[2]*3` positions. This is used by Qwen3-VL,
    Qwen3-VL-MoE, Qwen3.5, and Qwen3.5-MoE.
)DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(
    MRotaryEmbedding, 1,
    OpSchema()
        .SetDoc(MRotaryEmbedding_ver1_doc)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1.0",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("interleaved",
              "Indicates whether the input has real and imaginary parts interleaved along the last "
              "(head_size) axis. Default value is 0 (False), meaning the first half of the rotary "
              "portion consists of real values and the second half consists of imaginary values.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("rotary_embedding_dim",
              "Rotary embedding dimension. Default value is 0, meaning the whole head_size is rotated.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("num_heads",
              "Number of attention heads. Default value is 0. Must use with rotary_embedding_dim",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("is_packed_batching",
              "ragged batch inputs or not. Default value is 0",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("mrope_section",
              "3 non-negative integers [section_t, section_h, section_w] describing how the "
              "half_rotary_embedding_dim axis of the cos/sin cache is divided among the temporal, "
              "height, and width position streams. section_t + section_h + section_w must equal "
              "rotary_embedding_dim / 2 (or head_size / 2 when rotary_embedding_dim is 0). Required.",
              AttributeProto::INTS)
        .Attr("mrope_layout",
              "How the 3 sections are combined to form the per-token cos/sin vector: 0 (default) for "
              "Sectioned/Chunked layout (Qwen2-VL, Qwen2.5-VL) or 1 for Interleaved layout (Qwen3-VL, "
              "Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE).",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "3D tensor with shape (batch_size, sequence_length, hidden_size) or 4D with shape "
               "(batch_size, num_heads, sequence_length, head_size)",
               "T")
        .Input(1,
               "position_ids",
               "3D tensor with shape (3, batch_size, sequence_length) containing the temporal, height, "
               "and width position id streams (in that order along dim 0).",
               "M")
        .Input(2,
               "cos_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2) or "
               "(max_sequence_length, rotary_embedding_dim / 2)",
               "T")
        .Input(3,
               "sin_cache",
               "2D tensor with shape (max_sequence_length, head_size / 2) or "
               "(max_sequence_length, rotary_embedding_dim / 2)",
               "T")
        .Output(0,
                "output",
                "tensor with same shape as input.",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("M", {"tensor(int64)"}, "Constrain position_ids to integer tensors")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* GemmaRotaryEmbedding_ver1_doc = R"DOC(
GemmaRotaryEmbedding is the implementation of below part of rotary positional embeddings (RoPE). It implements below from modeling_gemma.py.

Here's onnxscript that was tested

from onnxscript import FLOAT, FLOAT16, script
from onnxscript import opset18 as op

@script()
def gemma_rotary_embedding(emb: FLOAT["bs", "seq_len", "dim"], q: FLOAT16["bs", "num_heads", "seq_len", "dim"], q_rot: FLOAT16["bs", "num_heads", "seq_len", "dim"], k: FLOAT16["bs", "num_heads", "seq_len", "dim"], k_rot: FLOAT16["bs", "num_heads", "seq_len", "dim"]):
  sin_val = op.Sin(emb)
  casted_sin = op.Cast(sin_val, to=10) # for fp16 mix-precision training. Other types are not supported.
  cos_val = op.Cos(emb)
  casted_cos = op.Cast(cos_val, to=10)
  unsqueezed_sin = op.Unsqueeze(casted_sin, [1])
  unsqueezed_cos = op.Unsqueeze(casted_cos, [1])
  q_embed = (q * casted_cos) + (q_rot * casted_sin)
  k_embed = (k * casted_cos) + (k_rot * casted_sin)
  return q_embed, k_embed

onnx_model = gemma_rotary_embedding.to_model_proto()


)DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(
    GemmaRotaryEmbedding, 1,
    OpSchema()
        .SetDoc(GemmaRotaryEmbedding_ver1_doc)
        .Input(0,
               "emb",
               "embedding - 3D tensor with shape (batch_size, seq_len, dim)",
               "U")
        .Input(1,
               "q",
               "q state - 4D tensor with shape (batch_size, num_heads, seq_len, dim)",
               "T")
        .Input(2,
               "q_rot",
               "half rotated q state - 4D tensor with shape (batch_size, num_heads, seq_len, dim)",
               "T")
        .Input(3,
               "k",
               "k state - 4D tensor with shape (batch_size, num_heads, seq_len, dim)",
               "T")
        .Input(4,
               "k_rot",
               "k state - 4D tensor with shape (batch_size, num_heads, seq_len, dim)",
               "T")
        .Output(0,
                "output1",
                "4D tensor with shape (batch_size, num_heads, seq_len, dim)",
                "T")
        .Output(1,
                "output2",
                "4D tensor with shape (batch_size, num_heads, seq_len, dim)",
                "T")
        .TypeConstraint("T", {"tensor(float16)"}, "Constrain input and output types to float16 tensors.")
        .TypeConstraint("U", {"tensor(float)"}, "Constrain input 0 type to float tensors")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          propagateElemTypeFromInputToOutput(ctx, 1, 1);
          propagateShapeFromInputToOutput(ctx, 1, 0);
          propagateShapeFromInputToOutput(ctx, 1, 1);
        }));

constexpr const char* EmbedLayerNormalization_ver1_doc = R"DOC(
EmbedLayerNormalization is the fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors.
The last input mask is optional. If mask is provided, mask index (that is position of first 0 in mask, or number of words)
will be calculated.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    EmbedLayerNormalization, 1,
    OpSchema()
        .SetDoc(EmbedLayerNormalization_ver1_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultEmbedLayerNormEpsilon)
        .Attr("mask_index_type", "The mask index tensor type for shape inference (0: None, 1: 1D mask_index)", AttributeProto::INT, OPTIONAL_VALUE)
        .Input(0, "input_ids", "2D words IDs with shape (batch_size, sequence_length)", "T1")
        .Input(1, "segment_ids", "2D segment IDs with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
        .Input(2, "word_embedding", "2D with shape (,hidden_size)", "T")
        .Input(3, "position_embedding", "2D with shape (, hidden_size)", "T")
        .Input(4, "segment_embedding", "2D with shape (, hidden_size)", "T", OpSchema::Optional)
        .Input(5, "gamma", "1D gamma tensor for layer normalization with shape (hidden_size)", "T")
        .Input(6, "beta", "1D beta tensor for layer normalization  with shape (hidden_size)", "T")
        .Input(7, "mask", "2D attention mask with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
        .Input(8, "position_ids", "2D position ids with shape (batch_size, sequence_length) or (1, sequence_length)", "T1", OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(1, "mask_index", "1D mask_index tensor with shape (batch_size)", "T1", OpSchema::Optional)
        .Output(2, "embedding_sum", "sum of word_embedding and position_embedding without layer normalization", "T", OpSchema::Optional)
        .TypeConstraint("T1", {"tensor(int32)"}, "Constrain input and output integer tensors types")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output float tensors types.")
        .TypeAndShapeInferenceFunction(EmbedLayerNormalizationShapeInference));

constexpr const char* FastGelu_ver1_doc = R"DOC(
GELU (Gaussian Error Linear Unit) approximation: Y=0.5*X*(1+tanh(0.797885*X+0.035677*X*X*X)) with an optional input of bias that will be added to X before GELU.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    FastGelu, 1,
    OpSchema()
        .SetDoc(FastGelu_ver1_doc)
        .Input(0, "X", "input tensor", "T")
        .Input(1, "bias", "bias tensor", "T", OpSchema::Optional)
        .Output(0, "Y", "output tensor", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
          // fastgelu(x) =
          auto* tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type()))
            return false;
          auto elem_type = (TensorProto_DataType)(tp->tensor_type().elem_type());

          // Optional input 1 indicates a bias to be added to input 0.
          auto hasBias = ctx.hasInput(1);

          FunctionBuilder builder(functionProto);
          builder
              .AddOpset("", 13)
              .Const("a", ToTensor(0.5, elem_type))
              .Const("b", ToTensor(0.797885, elem_type))
              .Const("c", ToTensor(0.035677, elem_type))
              .Const("one", ToTensor(1.0, elem_type))
              .Add(hasBias ? "X_bias = Add (X, bias)" : "X_bias = Identity (X)")
              .Add(R"(
                T1 = Mul (X_bias, X_bias)
                T2 = Mul (c, T1)
                T3 = Add (b, T2)
                T4 = Mul (X_bias, T3)
                T5 = Tanh (T4)
                T6 = Add (one, T5)
                T7 = Mul (X_bias, T6)
                Y = Mul (a, T7)
            )");

          schema.BuildFunction(functionProto);
          return true;
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    RelativePositionBias, 1,
    OpSchema()
        .SetDoc("Compute binned relative position bias for T5 model. ref: https://arxiv.org/abs/1803.02155v2")
        .Attr("max_distance", "Max distance", AttributeProto::INT)
        .Attr("is_bidirectional", "Default value is 0.", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "bias_table", "2D input tensor with shape (num_buckets, num_heads), COL-major(See UT for example)", "T")
        .Input(1, "query_length", "The length of query. Self Attention requires query_length = key_length", "U")
        .Input(2, "key_length", "The length of key.", "U")
        .Output(0, "output", "4D output tensor with shape (1, num_heads, sequence_length, sequence_length)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeConstraint("U", {"tensor(int64)"}, "Constrain sequence_length to int tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& bias_table_shape = getInputShape(ctx, 0);
          if (bias_table_shape.dim_size() < 2) {
            fail_shape_inference("RelativePositionBias: bias_table must have rank >= 2");
          }
          TensorShapeProto output_shape;
          output_shape.add_dim()->set_dim_value(1);
          *output_shape.add_dim() = bias_table_shape.dim(1);
          output_shape.add_dim();
          output_shape.add_dim();
          updateOutputShape(ctx, 0, output_shape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    SkipLayerNormalization, 1,
    OpSchema()
        .SetDoc("Skip and Layer Normalization Fusion")
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Input(1, "skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size) or (1, sequence_length, hidden_size) or (sequence_length, hidden_size)", "T")
        .Input(2, "gamma", "1D input tensor with shape (hidden_size)", "T")
        .Input(3, "beta", "1D skip tensor with shape (hidden_size", "T", OpSchema::Optional)
        .Input(4, "bias", "1D bias tensor with shape (hidden_size", "T", OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
        .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
        .Output(3, "input_skip_bias_sum", "Sum of the input and skip inputs (and bias if it exists) with shape (batch_size, sequence_length, hidden_size).", "T", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float or half tensors.")
        .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
        .TypeAndShapeInferenceFunction(SkipLayerNormalizationShapeInference));

ONNX_MS_OPERATOR_SET_SCHEMA(
    SkipSimplifiedLayerNormalization, 1,
    OpSchema()
        .SetDoc("Skip and Root Mean Square Layer Normalization")
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
        .Input(0,
               "input",
               "3D input tensor with shape (batch_size, sequence_length, hidden_size)"
               "Or 2D input tensor with shape (token_count, hidden_size)",
               "T")
        .Input(1,
               "skip",
               "3D input tensor with shape (batch_size, sequence_length, hidden_size)"
               "Or 2D input tensor with shape (token_count, hidden_size)",
               "T")
        .Input(2,
               "gamma",
               "1D input tensor with shape (hidden_size)",
               "T")
        .Input(3,
               "bias",
               "1D bias tensor with shape (hidden_size",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, hidden_size)"
                "Or 2D output tensor with shape (token_count, hidden_size)",
                "T")
        .Output(1,
                "mean",
                "Saved mean used during training to speed up gradient computation",
                "U",
                OpSchema::Optional)
        .Output(2,
                "inv_std_var",
                "Saved inverse standard variance used during training to speed up gradient computation.",
                "U",
                OpSchema::Optional)
        .Output(3,
                "input_skip_bias_sum",
                "Sum of the input and skip inputs (and bias if it exists)"
                "with shape (batch_size, sequence_length, hidden_size) or (token_count, hidden_size).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
        .TypeAndShapeInferenceFunction(SkipLayerNormalizationShapeInference));

constexpr const char* NGramRepeatBlock_ver1_doc = R"DOC(
Enforce no repetition of n-grams. Scores are set to `-inf` for tokens that form a repeated n-gram if added to the back of the input_ids.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    NGramRepeatBlock, 1,
    OpSchema().SetDoc(NGramRepeatBlock_ver1_doc).Attr("ngram_size", "The NGram size.", AttributeProto::INT).Input(0, "input_ids", "2D input tensor with shape (batch_size, sequence_length)", "Tid").Input(1, "scores", "2D input tensor with shape (batch_size, vocab_size)", "T").Output(0, "scores_out", "2D output tensor with shape (batch_size, vocab_size)", "T").TypeConstraint("Tid", {"tensor(int64)"}, "Constrain indices to integer types").TypeConstraint("T", {"tensor(float)"}, "Constrain scores input and output types to float tensors.").TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 1, 0);
      if (!hasInputShape(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 1, 0);
    }));

constexpr const char* BifurcationDetector_ver1_doc = R"DOC(
Component for aggressive decoding. Find the bifurcation index of predicted tokens, between source tokens,
starting from previous suffix match index, and predicted tokens.
Concat predicted tokens, starting from bifurcation index, to the back
of current tokens. This forms the output tokens.
Detect suffix match index in source tokens, between source tokens and output tokens.
Detection is based on finding the appearances of last n-gram in output tokens
in source tokens.
A match is considered found if source tokens contain a single matching n-gram.
Return the index of the start of the n-gram in source tokens.
No matching if found if src tokens contain multiple or zero matching n-grams. Return -1.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    BifurcationDetector, 1,
    OpSchema()
        .SetDoc(BifurcationDetector_ver1_doc)
        .Attr("min_ngram_size", "The minimum NGram size for suffix matching.", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("max_ngram_size", "The maximum NGram size for suffix matching.", AttributeProto::INT, static_cast<int64_t>(3))
        .Input(0, "src_tokens", "Encoder input ids.", "T")
        .Input(1, "cur_tokens", "Decoder input ids.", "T")
        .Input(2, "prev_suffix_match_idx", "Previous suffix match index", "T")
        .Input(3, "pred_tokens", "Predicted token ids from aggressive decoding", "T", OpSchema::Optional)
        .Output(0, "tokens", "Decoder input ids after merging predicted tokens", "T")
        .Output(1, "suffix_match_idx", "new suffix match index", "T")
        .TypeConstraint("T", {"tensor(int64)"}, "Constrain to integer types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          propagateElemTypeFromInputToOutput(ctx, 2, 1);
          if (hasInputShape(ctx, 2)) {
            propagateShapeFromInputToOutput(ctx, 2, 1);
          }
          // output tokens lengths is dynamic as it depends on the bifurcation index of predicted tokens and source tokens,
          // and current tokens length.
          // tokens_length = cur_tokens_length + bifurcation_index + 1.
        }));

constexpr const char* GemmFastGelu_ver1_doc = R"DOC(
It's a fusion of MatMul and FastGelu.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GemmFastGelu, 1,
    OpSchema()
        .SetDoc(GemmFastGelu_ver1_doc)
        .Input(0, "X", "input tensor", "T")
        .Input(1, "W", "input tensor", "T")
        .Input(2, "bias", "bias tensor", "T", OpSchema::Optional)
        .Output(0, "Y", "output tensor", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
          ONNX_NAMESPACE::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
        }));

constexpr const char* RemovePadding_ver1_doc = R"DOC(
Compress transformer input by removing paddings. It assumes padding is on the right side of sequence.

The input has padding with shape (batch_size, sequence_length, hidden_size). This will generate two outputs:
output has shape (total_tokens, hidden_size); token_offset with shape (batch_size, sequence_length).

token_offset has offsets of all non-padding tokens first, then offset of all padding tokens. It is
a list of batch_size * sequence_length elements, which is reshaped to 2D for convenience of shape inference.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    RemovePadding, 1,
    OpSchema()
        .SetDoc(RemovePadding_ver1_doc)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, sequence_length, hidden_size)",
               "T")
        .Input(1,
               "sequence_token_count",
               "Number of non-padding tokens in each sequence with shape (batch_size).",
               "M")
        .Output(0,
                "output",
                "output tensor with shape (total_tokens, hidden_size)",
                "T")
        .Output(1,
                "token_offset",
                "Offset of non-padding tokens, and those of padding tokens. Its shape is (batch_size, sequence_length)",
                "M")
        .Output(2,
                "cumulated_seq_len",
                "Cumulated sequence lengths. Its shape is (batch_size + 1)",
                "M")
        .Output(3,
                "max_seq_len",
                "Max sequence length without padding. Its shape is (1)",
                "M")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain sequence_token_count and token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          RemovePaddingTypeAndShapeInference(ctx);
        }));

constexpr const char* RestorePadding_ver1_doc = R"DOC(
Restore paddings and fill padding with zeros.

The input has padding with shape (total_tokens, hidden_size) and token_offset with shape (batch_size, sequence_length).
The output has shape (batch_size, sequence_length, hidden_size).
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    RestorePadding, 1,
    OpSchema()
        .SetDoc(RestorePadding_ver1_doc)
        .Input(0,
               "input",
               "Input tensor with shape (total_tokens, hidden_size)",
               "T")
        .Input(1,
               "token_offset",
               "Offset of non-padding tokens and paddings. Its shape is (batch_size, sequence_length)",
               "M")
        .Output(0,
                "output",
                "output tensor with shape (batch_size, sequence_length, hidden_size)",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          RestorePaddingTypeAndShapeInference(ctx);
        }));

constexpr const char* GatedRelativePositionBias_ver1_doc = R"DOC(
  query_layer = (query_layer + query_bias).reshape(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
  gate_u, gate_r = torch.sigmoid(
      self.gate_ur_linear(query_layer).view(batch_size, num_head, seq_len, 2, D/2).sum(-1, keepdim=False)
  ).chunk(2, dim=-1)
  gate_u_1 = gate_u * (gate_r * self.eco_a - 1.0) + 2.0
  rel_pos_bias = gate_u_1 * rel_pos
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedRelativePositionBias, 1,
    OpSchema()
        .SetDoc(GatedRelativePositionBias_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Input(0, "query_layer", "tensor with shape (batch_size, seq_len, num_heads x head_size) or (token_count, num_heads x head_size)", "T")
        .Input(1, "query_bias", "1-d tensor with shape (num_heads x head_size)", "T")
        .Input(2, "rel_pos", "tensor with shape (1, num_head, seq_len, seq_len)", "T")
        .Input(3, "weight", "gemm weight for the gated_ur_linear, shape (head_size, D), D is divisible by 2", "T")
        .Input(4, "bias", "bias for the gated_ur_linear, shape (D)", "T")
        .Input(5, "eco_a", "tensor of shape (1, num_heads, 1, 1)", "T")
        .Input(6, "token_offset", "offset of each token with shape (batch_size, seq_len)", "M", OpSchema::Optional)
        .Output(0, "output", "output tensor with shape (batch_size, num_heads, seq_len, seq_len)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          int64_t num_heads = getAttribute(ctx, "num_heads", -1L);

          // When padding is removed:
          //   query_layer: (token_count, num_heads x head_size)
          //   token_offset: (batch_size, seq_len)
          // Otherwise:
          //   query_layer: (batch_size, seq_len, num_heads x head_size)
          //   token_offset: None
          // Output shape: (batch_size, num_heads, seq_len, seq_len)
          if (hasInputShape(ctx, 6)) {
            auto& token_offset_shape = getInputShape(ctx, 6);
            if (token_offset_shape.dim_size() < 2) {
              fail_shape_inference("GatedRelativePositionBias: token_offset must have rank >= 2");
            }
            TensorShapeProto output_shape;
            *output_shape.add_dim() = token_offset_shape.dim(0);
            output_shape.add_dim()->set_dim_value(num_heads);
            *output_shape.add_dim() = token_offset_shape.dim(1);
            *output_shape.add_dim() = token_offset_shape.dim(1);
            updateOutputShape(ctx, 0, output_shape);
          } else if (hasInputShape(ctx, 0)) {
            auto& query_layer_shape = getInputShape(ctx, 0);
            if (query_layer_shape.dim().size() == 3) {
              TensorShapeProto output_shape;
              *output_shape.add_dim() = query_layer_shape.dim(0);
              output_shape.add_dim()->set_dim_value(num_heads);
              *output_shape.add_dim() = query_layer_shape.dim(1);
              *output_shape.add_dim() = query_layer_shape.dim(1);
              updateOutputShape(ctx, 0, output_shape);
            }
          }
        }));

constexpr const char* NGramHashMapping_ver1_doc = R"DOC(
Computes Engram n-gram hash ids from pre-compressed tokenizer ids.

For n in [2, max_ngram_size], the op creates causal shifts of input_ids, padding positions before the
sequence with pad_id, and computes
mix = shifted_0 * multipliers[0] xor ... xor shifted_(n-1) * multipliers[n-1].
For every head of that n-gram order it emits mix modulo the corresponding head vocabulary size.
The output layout is (batch_size, sequence_length, (max_ngram_size - 1) * n_head_per_ngram), with
heads for n=2 first, then n=3, and so on.

An n-gram window reaches max_ngram_size - 1 positions before the current token. To keep the op causal
across invocations (chunked prefill or autoregressive decode), the optional past_ids input carries
those preceding ids and present_ids returns the ids to pass to the next call. Both have shape
(batch_size, max_ngram_size - 1) and are right-aligned, so the last slot is the most recent id.
Positions before the start of the whole sequence use pad_id, or eos_token_id when it is provided.
Running the op once over a full sequence and running it over consecutive chunks while threading
present_ids into past_ids produce identical hash ids, including when reset_on_eos is enabled. When
segment_ids is used, segment boundaries are applied only within the current input_ids chunk and are
not inferred from past_ids. When past_ids is omitted the missing history is pad_id, or eos_token_id
when it is provided.
past_ids and present_ids may use the same allocation. Such in-place execution is transaction-safe
only when the whole operator call is unconditionally committed; a caller that may select a prefix or
roll back must preserve past_ids.

Optional inputs add packed-sequence and Qwen4-Exp-style n-gram embedding support:

- eos_token_id, when provided together with reset_on_eos != 0, causes causal history to reset at EOS
  boundaries: any shifted position at or before the most recent EOS strictly before the current
  position is replaced with eos_token_id instead of the real token.
- segment_ids, when provided, additionally resets causal history at any position whose segment id
  differs from the immediately preceding position's segment id within input_ids. Segment boundaries
  are not checked against past_ids history.
- head_offsets, when provided, adds a fixed per-output-head offset after the modulo by the head's
  vocabulary size, letting all heads across all n-gram orders share one flat embedding table.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    NGramHashMapping, 1,
    OpSchema()
        .SetDoc(NGramHashMapping_ver1_doc)
        .Attr("max_ngram_size",
              "Maximum n-gram order. Must be at least 2.",
              AttributeProto::INT)
        .Attr("n_head_per_ngram",
              "Number of hash heads emitted for each n-gram order.",
              AttributeProto::INT)
        .Attr("pad_id",
              "Compressed tokenizer id used to pad causal shifts before the beginning of a sequence.",
              AttributeProto::INT)
        .Attr("reset_on_eos",
              "When non-zero and the eos_token_id input is provided, reset causal n-gram history at "
              "EOS boundaries as described in the op doc. Default is 0 (disabled), which preserves "
              "the original pad_id-only behavior.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "input_ids",
               "Compressed tokenizer ids with shape (batch_size, sequence_length).",
               "M")
        .Input(1,
               "multipliers",
               "Per-shift hash multipliers with shape at least (max_ngram_size). Conventionally odd, "
               "but any value is accepted.",
               "M")
        .Input(2,
               "vocab_sizes",
               "Per-output-head vocabulary sizes, conventionally prime, with shape "
               "((max_ngram_size - 1) * n_head_per_ngram). Every entry must be strictly positive. "
               "The CPU implementation rejects a non-positive entry; GPU implementations guard the "
               "modulo to avoid a device-side division by zero and emit a hash id of 0 for that head.",
               "M")
        .Input(3,
               "past_ids",
               "Optional compressed tokenizer ids for the max_ngram_size - 1 positions that precede "
               "this call, with shape (batch_size, max_ngram_size - 1). Right-aligned, so the last "
               "slot is the most recent id. If omitted the history is pad_id, or eos_token_id when "
               "provided.",
               "M",
               OpSchema::Optional)
        .Input(4,
               "head_offsets",
               "Optional per-output-head additive offset with shape "
               "((max_ngram_size - 1) * n_head_per_ngram), added after the modulo.",
               "M",
               OpSchema::Optional)
        .Input(5,
               "eos_token_id",
               "Optional scalar end-of-sequence token id, same type as input_ids. Required for "
               "reset_on_eos to take effect and for EOS-based substitution of unavailable prior "
               "context; see the op doc.",
               "M",
               OpSchema::Optional)
        .Input(6,
               "segment_ids",
               "Optional per-token segment id with shape (batch_size, sequence_length), used to reset "
               "causal history at packed-sequence boundaries within input_ids.",
               "tensor(int32)",
               OpSchema::Optional)
        .Output(0,
                "hash_ids",
                "Hash ids with shape (batch_size, sequence_length, "
                "(max_ngram_size - 1) * n_head_per_ngram).",
                "M")
        .Output(1,
                "present_ids",
                "Trailing max_ngram_size - 1 ids of past_ids followed by input_ids, with shape "
                "(batch_size, max_ngram_size - 1). Feed this back as past_ids on the next call.",
                "M",
                OpSchema::Optional)
        .TypeConstraint("M",
                        {"tensor(int32)", "tensor(int64)"},
                        "Constrain ids, multipliers, vocabulary sizes, and output ids to integer tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumOutputs() > 1) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
          }
          const int64_t max_ngram_size = getAttribute(ctx, "max_ngram_size", int64_t{-1});
          const int64_t n_head_per_ngram = getAttribute(ctx, "n_head_per_ngram", int64_t{-1});
          if (max_ngram_size < 2) {
            fail_shape_inference("NGramHashMapping: max_ngram_size must be at least 2");
          }
          if (n_head_per_ngram < 1) {
            fail_shape_inference("NGramHashMapping: n_head_per_ngram must be positive");
          }
          if (max_ngram_size - 1 > std::numeric_limits<int64_t>::max() / n_head_per_ngram) {
            fail_shape_inference("NGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram overflows int64_t");
          }
          const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;

          if (hasInputShape(ctx, 0)) {
            const auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 2) {
              fail_shape_inference("NGramHashMapping: input_ids must have rank 2");
            }
            TensorShapeProto output_shape;
            *output_shape.add_dim() = input_shape.dim(0);
            *output_shape.add_dim() = input_shape.dim(1);
            output_shape.add_dim()->set_dim_value(num_heads);
            updateOutputShape(ctx, 0, output_shape);

            if (ctx.getNumOutputs() > 1) {
              TensorShapeProto present_shape;
              *present_shape.add_dim() = input_shape.dim(0);
              present_shape.add_dim()->set_dim_value(max_ngram_size - 1);
              updateOutputShape(ctx, 1, present_shape);
            }
          }
          if (hasInputShape(ctx, 1)) {
            const auto& multipliers_shape = getInputShape(ctx, 1);
            if (multipliers_shape.dim_size() != 1 ||
                (multipliers_shape.dim(0).has_dim_value() &&
                 multipliers_shape.dim(0).dim_value() < max_ngram_size)) {
              fail_shape_inference("NGramHashMapping: multipliers must have shape at least (max_ngram_size)");
            }
          }
          if (hasInputShape(ctx, 2)) {
            const auto& vocab_sizes_shape = getInputShape(ctx, 2);
            if (vocab_sizes_shape.dim_size() != 1 ||
                (vocab_sizes_shape.dim(0).has_dim_value() &&
                 vocab_sizes_shape.dim(0).dim_value() != num_heads)) {
              fail_shape_inference(
                  "NGramHashMapping: vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
            }
          }
          if (hasInputShape(ctx, 4)) {
            const auto& head_offsets_shape = getInputShape(ctx, 4);
            if (head_offsets_shape.dim_size() != 1 ||
                (head_offsets_shape.dim(0).has_dim_value() && head_offsets_shape.dim(0).dim_value() != num_heads)) {
              fail_shape_inference(
                  "NGramHashMapping: head_offsets must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
            }
          }
          if (hasInputShape(ctx, 5)) {
            const auto& eos_token_id_shape = getInputShape(ctx, 5);
            if (eos_token_id_shape.dim_size() != 0) {
              fail_shape_inference("NGramHashMapping: eos_token_id must be a scalar");
            }
          }
          if (hasInputShape(ctx, 6)) {
            const auto& segment_ids_shape = getInputShape(ctx, 6);
            if (segment_ids_shape.dim_size() != 2) {
              fail_shape_inference("NGramHashMapping: segment_ids must have rank 2");
            }
            if (hasInputShape(ctx, 0)) {
              const auto& input_shape = getInputShape(ctx, 0);
              if ((segment_ids_shape.dim(0).has_dim_value() && input_shape.dim(0).has_dim_value() &&
                   segment_ids_shape.dim(0).dim_value() != input_shape.dim(0).dim_value()) ||
                  (segment_ids_shape.dim(1).has_dim_value() && input_shape.dim(1).has_dim_value() &&
                   segment_ids_shape.dim(1).dim_value() != input_shape.dim(1).dim_value())) {
                fail_shape_inference("NGramHashMapping: segment_ids must have shape (batch_size, sequence_length)");
              }
            }
          }
        }));

constexpr const char* VarlenNGramHashMapping_ver1_doc = R"DOC(
Computes Engram n-gram hash ids from pre-compressed tokenizer ids over a packed, token-major batch
of variable-length sequences.

input_ids has shape (total_tokens) and hash_ids has shape (total_tokens, (max_ngram_size - 1) *
n_head_per_ngram). cumulative_sequence_length is a device-resident int32 tensor of shape
(batch_size + 1); request i occupies [cumulative_sequence_length[i], cumulative_sequence_length[i +
1]) of the packed buffer. Every request contributes at least one token.

For n in [2, max_ngram_size], the op creates causal shifts of each request's own tokens, padding
positions before that request's start with pad_id (or its own past_ids history, see below), and
computes mix = shifted_0 * multipliers[0] xor ... xor shifted_(n-1) * multipliers[n-1]. For every
head of that n-gram order it emits mix modulo the corresponding head vocabulary size. The n-gram
window never reads tokens belonging to a different packed request; it is clamped at each request's
own boundary exactly like VarlenCausalConvWithState clamps its causal convolution.

An n-gram window reaches max_ngram_size - 1 positions before the current token. To keep the op
causal across invocations (chunked prefill or autoregressive decode) for every concurrent request in
the packed batch, the optional past_ids input carries those preceding ids per request and
present_ids returns the ids to pass to the next call. Both have shape (batch_size, max_ngram_size -
1) and are right-aligned, so the last slot is the most recent id, and are indexed by request
(batch_size), not by position in the packed buffer. Positions before the start of a request's whole
sequence use pad_id, or eos_token_id when provided. Running NGramHashMapping once per sequence and running this op once over those
sequences packed together (optionally split into packed chunks with present_ids threaded into
past_ids) produce identical hash ids.

Optional inputs add Qwen4-Exp-style n-gram embedding support:

- eos_token_id, when provided together with reset_on_eos != 0, causes causal history to reset at EOS
  boundaries. Missing history is also filled with eos_token_id.
- segment_ids, when provided, additionally resets causal history when adjacent tokens within one
  packed request have different segment ids. Thread present_segment_ids into past_segment_ids on
  subsequent calls to preserve boundaries across chunked prefill and decode calls.
- head_offsets, when provided, adds a fixed per-output-head offset after the modulo. Addition wraps
  in the input id type on overflow.
- capture_count enables compact prefix state capture when state_update_capacity is positive. For
  each request, state_update contains the trailing state after each captured prefix token.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    VarlenNGramHashMapping, 1,
    OpSchema()
        .SetDoc(VarlenNGramHashMapping_ver1_doc)
        .Attr("max_ngram_size",
              "Maximum n-gram order. Must be at least 2.",
              AttributeProto::INT)
        .Attr("n_head_per_ngram",
              "Number of hash heads emitted for each n-gram order.",
              AttributeProto::INT)
        .Attr("pad_id",
              "Compressed tokenizer id used to pad causal shifts before the beginning of a request's "
              "sequence.",
              AttributeProto::INT)
        .Attr("reset_on_eos",
              "When non-zero and eos_token_id is provided, reset causal n-gram history at EOS "
              "boundaries. Default is 0.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("state_update_capacity",
              "Static number of compact contiguous-prefix states to expose per request. Valid range "
              "is [0, 8]. capture_count is required exactly when this is positive.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "input_ids",
               "Token-major packed compressed tokenizer ids with shape (total_tokens).",
               "M")
        .Input(1,
               "multipliers",
               "Per-shift hash multipliers with at least max_ngram_size elements. Conventionally odd, "
               "but any value is accepted.",
               "M")
        .Input(2,
               "vocab_sizes",
               "Per-output-head vocabulary sizes, conventionally prime, with shape "
               "((max_ngram_size - 1) * n_head_per_ngram). Every entry must be strictly positive. "
               "The CPU implementation rejects a non-positive entry; GPU implementations guard the "
               "modulo to avoid a device-side division by zero and emit a hash id of 0 for that head.",
               "M")
        .Input(3,
               "cumulative_sequence_length",
               "Device tensor with shape (batch_size + 1) giving the half-open packed token range "
               "of each request.",
               "S")
        .Input(4,
               "past_ids",
               "Optional compressed tokenizer ids for the max_ngram_size - 1 positions that precede "
               "this call, with shape (batch_size, max_ngram_size - 1). Right-aligned, so the last "
               "slot is the most recent id, and indexed by request rather than by packed position. "
               "If omitted the history is pad_id, or eos_token_id when provided.",
               "M",
               OpSchema::Optional)
        .Input(5,
               "head_offsets",
               "Optional per-output-head additive offset with shape "
               "((max_ngram_size - 1) * n_head_per_ngram), added after the modulo with wrapping "
               "arithmetic in the input id type.",
               "M",
               OpSchema::Optional)
        .Input(6,
               "eos_token_id",
               "Optional scalar end-of-sequence token id. When provided it replaces pad_id for "
               "missing history and enables reset_on_eos.",
               "M",
               OpSchema::Optional)
        .Input(7,
               "segment_ids",
               "Optional token-major segment ids with shape (total_tokens), used to reset causal "
               "history at segment boundaries within each packed request.",
               "tensor(int32)",
               OpSchema::Optional)
        .Input(8,
               "past_segment_ids",
               "Optional segment ids corresponding to past_ids, with shape "
               "(batch_size, max_ngram_size - 1). Thread present_segment_ids from the previous call "
               "into this input to preserve segment boundaries across calls.",
               "S",
               OpSchema::Optional)
        .Input(9,
               "capture_count",
               "Optional device int32 tensor with shape (batch_size). For each request, captures "
               "that many local prefix states, clamped to the request length and "
               "state_update_capacity. Required exactly when state_update_capacity is positive.",
               "S",
               OpSchema::Optional)
        .Output(0,
                "hash_ids",
                "Token-major packed hash ids with shape (total_tokens, "
                "(max_ngram_size - 1) * n_head_per_ngram).",
                "M")
        .Output(1,
                "present_ids",
                "Trailing max_ngram_size - 1 ids of past_ids followed by each request's own tokens, "
                "with shape (batch_size, max_ngram_size - 1). Feed this back as past_ids on the next "
                "call.",
                "M",
                OpSchema::Optional)
        .Output(2,
                "present_segment_ids",
                "Trailing max_ngram_size - 1 segment ids corresponding to present_ids, with shape "
                "(batch_size, max_ngram_size - 1). Feed this back as past_segment_ids on the next call.",
                "S",
                OpSchema::Optional)
        .Output(3,
                "state_update",
                "Optional trailing id states after each captured prefix token, with shape "
                "(batch_size, state_update_capacity, max_ngram_size - 1). Unused slots are pad_id.",
                "M",
                OpSchema::Optional)
        .TypeConstraint("M",
                        {"tensor(int32)", "tensor(int64)"},
                        "Constrain ids, multipliers, vocabulary sizes, and output ids to integer tensors.")
        .TypeConstraint("S",
                        {"tensor(int32)"},
                        "Constrain cumulative_sequence_length and segment ids to device int32 tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumOutputs() > 1) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
          }
          if (ctx.getNumOutputs() > 2) {
            propagateElemTypeFromInputToOutput(ctx, 3, 2);
          }
          if (ctx.getNumOutputs() > 3) {
            propagateElemTypeFromInputToOutput(ctx, 0, 3);
          }

          const int64_t max_ngram_size = getAttribute(ctx, "max_ngram_size", int64_t{-1});
          const int64_t n_head_per_ngram = getAttribute(ctx, "n_head_per_ngram", int64_t{-1});
          const int64_t state_update_capacity = getAttribute(ctx, "state_update_capacity", int64_t{0});
          if (max_ngram_size < 2) {
            fail_shape_inference("VarlenNGramHashMapping: max_ngram_size must be at least 2");
          }
          if (n_head_per_ngram < 1) {
            fail_shape_inference("VarlenNGramHashMapping: n_head_per_ngram must be positive");
          }
          if (state_update_capacity < 0 || state_update_capacity > 8) {
            fail_shape_inference("VarlenNGramHashMapping: state_update_capacity must be in [0, 8]");
          }
          // max_ngram_size and n_head_per_ngram are model-controlled attributes with only
          // lower-bound checks above, so (max_ngram_size - 1) * n_head_per_ngram must be
          // multiplied with an overflow check: e.g. max_ngram_size == INT64_MAX would otherwise
          // silently wrap this signed multiplication before it ever reaches tensor-shape
          // validation.
          const int64_t state_length = max_ngram_size - 1;
          if (state_length != 0 && n_head_per_ngram > std::numeric_limits<int64_t>::max() / state_length) {
            fail_shape_inference(
                "VarlenNGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram overflows int64_t");
          }
          const int64_t num_heads = state_length * n_head_per_ngram;

          if (hasInputShape(ctx, 0)) {
            const auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 1) {
              fail_shape_inference("VarlenNGramHashMapping: input_ids must have rank 1");
            }
            TensorShapeProto output_shape;
            *output_shape.add_dim() = input_shape.dim(0);
            output_shape.add_dim()->set_dim_value(num_heads);
            updateOutputShape(ctx, 0, output_shape);
          }

          if (hasInputShape(ctx, 3)) {
            const auto& cu_seqlen_shape = getInputShape(ctx, 3);
            if (cu_seqlen_shape.dim_size() != 1) {
              fail_shape_inference("VarlenNGramHashMapping: cumulative_sequence_length must have rank 1");
            }
            const auto& cu_dim = cu_seqlen_shape.dim(0);
            if (cu_dim.has_dim_value() && cu_dim.dim_value() < 2) {
              fail_shape_inference(
                  "VarlenNGramHashMapping: cumulative_sequence_length must have at least 2 elements");
            }
            if (ctx.getNumOutputs() > 1) {
              TensorShapeProto present_shape;
              if (cu_dim.has_dim_value()) {
                // Runtime requires at least 2 elements (batch_size >= 1); a declared static shape of
                // [0] or [1] would otherwise infer a present_ids batch dimension of -1 or 0.
                present_shape.add_dim()->set_dim_value(cu_dim.dim_value() - 1);
              } else {
                present_shape.add_dim();  // unknown batch size
              }
              present_shape.add_dim()->set_dim_value(max_ngram_size - 1);
              updateOutputShape(ctx, 1, present_shape);
              if (ctx.getNumOutputs() > 2) {
                updateOutputShape(ctx, 2, present_shape);
              }
              if (ctx.getNumOutputs() > 3) {
                TensorShapeProto state_update_shape;
                *state_update_shape.add_dim() = present_shape.dim(0);
                state_update_shape.add_dim()->set_dim_value(state_update_capacity);
                state_update_shape.add_dim()->set_dim_value(max_ngram_size - 1);
                updateOutputShape(ctx, 3, state_update_shape);
              }
            }
            if (hasInputShape(ctx, 9)) {
              const auto& capture_count_shape = getInputShape(ctx, 9);
              if (capture_count_shape.dim_size() != 1) {
                fail_shape_inference("VarlenNGramHashMapping: capture_count must have rank 1");
              }
              if (cu_dim.has_dim_value() && capture_count_shape.dim(0).has_dim_value() &&
                  capture_count_shape.dim(0).dim_value() != cu_dim.dim_value() - 1) {
                fail_shape_inference("VarlenNGramHashMapping: capture_count must have shape (batch_size)");
              }
            }
          }
        }));

constexpr const char* CausalConvWithState_ver1_doc = R"DOC(
Stateful causal depthwise convolution, generalized to N spatial dimensions.

Used by Gated DeltaNet (Qwen3.5) and Mamba (Jamba, FalconMamba) as a preprocessing step.
Replaces the 3-op pattern (Concat + Conv + Slice) with a single fused operation.

The convolution is causal (looks only at current and past positions along the last
spatial dimension) and depthwise (each channel is convolved independently with its own kernel).

Input layout is channels-first: (batch_size, channels, ...).
Weight layout: (channels, 1, k_1, ...) for depthwise convolution.
The carry state stores the last (k-1) positions along the causal axis for incremental decode.

The ndim attribute generalizes the op to 1D, 2D, or 3D spatial dimensions. Causality is
enforced on the last spatial dimension only.

The optional activation attribute supports fused SiLU/Swish activation.

The dilation attribute spaces the kernel taps along the causal axis: output position t reads
input positions t - (k_1 - 1 - j) * dilation for tap j. The receptive field therefore spans
(k_1 - 1) * dilation positions before the current one, and the carry state grows to match:
past_state and present_state hold (k_1 - 1) * dilation positions instead of k_1 - 1. Dilation 1
(the default) is the undilated case and keeps the original state length, so models exported
before the attribute existed are unaffected.

The channels_last attribute selects a sequence-major layout for the activations and the carry
state, so a model that already produces channels-last activations does not have to transpose into
and out of the channels-first layout. With channels_last = 1 and ndim = 1, input and output are
(batch_size, sequence_length, d_1, ..., d_n) and the state tensors are
(batch_size, state_length, d_1, ..., d_n), where channels = d_1 * ... * d_n. Any number of trailing
channel axes is accepted, so an activation that keeps hyper-connections and hidden size as separate
axes needs no reshape either. weight and bias keep their channels-first (channels, 1, k_1) and
(channels) shapes because they have no sequence axis. The computed values are identical to the
channels-first layout; only the memory layout differs.
)DOC";

constexpr const char* EngramGate_ver1_doc = R"DOC(
Fuses the Engram gate.

The op consumes already projected keys in (batch_size, sequence_length, hc_mult, hidden_size) layout,
the hidden-state queries in the same layout, an already projected value in
(batch_size, sequence_length, hidden_size) layout that is shared by every hyper-connection, and the two
RMSNorm scales. The key and value projections stay outside the op so they can run on the execution
provider's tuned MatMul (weight prepacking, tensor cores, quantized weights) and so the value
projection is computed once per token instead of once per hyper-connection.

It computes the Engram gate:

gate = sigmoid(sign(dot) * sqrt(max(abs(dot), 1e-6))) where
dot = sum(RMSNorm(key) * RMSNorm(query)) / sqrt(hidden_size).

The output is gate * value, broadcast across the hyper-connections. The optional gated_value_normed
output applies RMSNorm to gate * value with conv_norm_scale, which can feed a following
CausalConvWithState. The final Engram residual value + short_conv(value) is then expressed with
RMSNorm, CausalConvWithState and Add.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    EngramGate, 1,
    OpSchema()
        .SetDoc(EngramGate_ver1_doc)
        .Attr("epsilon",
              "Epsilon used by both RMS normalization steps. Default is 1e-5.",
              AttributeProto::FLOAT,
              1.0e-5f)
        .Input(0,
               "key",
               "Projected Engram keys with shape (batch_size, sequence_length, hc_mult, hidden_size).",
               "T")
        .Input(1,
               "query",
               "Hidden-state queries with shape (batch_size, sequence_length, hc_mult, hidden_size).",
               "T")
        .Input(2,
               "value",
               "Projected Engram value shared by every hyper-connection, with shape "
               "(batch_size, sequence_length, hidden_size).",
               "T")
        .Input(3,
               "key_norm_scale",
               "RMSNorm scale for keys with shape (hc_mult, hidden_size).",
               "T")
        .Input(4,
               "query_norm_scale",
               "RMSNorm scale for queries with shape (hc_mult, hidden_size).",
               "T")
        .Input(5,
               "conv_norm_scale",
               "Optional RMSNorm scale for the gated value, with shape (hc_mult, hidden_size). Required "
               "when gated_value_normed is requested.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "Gated value tensor with shape (batch_size, sequence_length, hc_mult, hidden_size).",
                "T")
        .Output(1,
                "gated_value_normed",
                "Optional RMS-normalized gated value tensor with shape "
                "(batch_size, sequence_length, hc_mult, hidden_size).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumOutputs() > 1) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
          }

          if (hasInputShape(ctx, 0)) {
            const auto& key_shape = getInputShape(ctx, 0);
            if (key_shape.dim_size() != 4) {
              fail_shape_inference("EngramGate: key must have rank 4");
            }
            propagateShapeFromInputToOutput(ctx, 0, 0);
            if (ctx.getNumOutputs() > 1) {
              propagateShapeFromInputToOutput(ctx, 0, 1);
            }
          }
          if (hasInputShape(ctx, 1)) {
            const auto& query_shape = getInputShape(ctx, 1);
            if (query_shape.dim_size() != 4) {
              fail_shape_inference("EngramGate: query must have rank 4");
            }
          }
          if (hasInputShape(ctx, 2)) {
            const auto& value_shape = getInputShape(ctx, 2);
            if (value_shape.dim_size() != 3) {
              fail_shape_inference("EngramGate: value must have rank 3");
            }
          }
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    CausalConvWithState, 1,
    OpSchema()
        .SetDoc(CausalConvWithState_ver1_doc)
        .Attr("activation",
              "Fused activation function. One of: 'silu', 'swish', 'none'. "
              "Default is 'none'.",
              AttributeProto::STRING,
              std::string("none"))
        .Attr("ndim",
              "Spatial dimensionality: 1, 2, or 3. Default is 1.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("dilation",
              "Spacing between kernel taps along the causal (last spatial) axis. The receptive "
              "field spans (k_1 - 1) * dilation positions before the current one, and past_state / "
              "present_state hold that many positions. Must be >= 1. Default is 1 (undilated).",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("channels_last",
              "When 1, input, output, past_state and present_state use a sequence-major, "
              "channels-last layout: input and output are "
              "(batch_size, sequence_length, d_1, ..., d_n) and the state tensors are "
              "(batch_size, state_length, d_1, ..., d_n), where channels = d_1 * ... * d_n. "
              "weight and bias keep their channels-first shapes. Requires ndim = 1. "
              "Default is 0 (channels-first).",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("state_window",
              "Number of trailing per-position carry states held by past_state and present_state. "
              "When 0 (default) the state tensors have no window axis and hold only the state after "
              "the last position, i.e. the backward-compatible (batch_size, channels, state_length) "
              "where state_length = (k_1 - 1) * dilation. "
              "When W > 0 both gain a LEADING axis of extent W, right-aligned: slot j is the state "
              "after position (seq_len - W + j), so slot W-1 is always the state after the last "
              "position (identical to the W = 0 tensor) and is the slot past_state is read from. "
              "The window axis leads the batch axis so that each slot is one contiguous "
              "(batch_size, channels, state_length) block. Slots below max(0, W - seq_len) hold no "
              "position from this call and are filled with zeros. A window lets a speculative "
              "decoder roll the state back to an accepted prefix without replaying the forward. "
              "Valid range is [0, 8].",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, channels, ...) in the default channels-first "
               "layout. Spatial dims: 1D: (L,); 2D: (H, W); 3D: (D, H, W). When channels_last = 1 "
               "the shape is (batch_size, sequence_length, d_1, ..., d_n) instead.",
               "T")
        .Input(1,
               "weight",
               "Depthwise convolution kernel with shape (channels, 1, k_1, ...). "
               "Spatial kernel sizes: (k_1, ..., k_ndim).",
               "T")
        .Input(2,
               "bias",
               "Optional per-channel bias with shape (channels).",
               "T",
               OpSchema::Optional)
        .Input(3,
               "past_state",
               "Carry state from previous step. For ndim=1: (batch_size, channels, state_length), "
               "or (W, batch_size, channels, state_length) when state_window = W > 0, in which case "
               "only slot W-1 is read, where state_length = (k_1 - 1) * dilation. When "
               "channels_last = 1 each slot is (batch_size, state_length, d_1, ..., d_n) instead. "
               "If not provided, padding is zero.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "Convolution output with same shape as input.",
                "T")
        .Output(1,
                "present_state",
                "Updated carry state. For ndim=1: (batch_size, channels, state_length), or "
                "(W, batch_size, channels, state_length) when state_window = W > 0, and "
                "(batch_size, state_length, d_1, ..., d_n) per slot when channels_last = 1. Slot "
                "W-1 contains the last state_length values from the virtual input along the causal "
                "axis; slot j contains the same for the prefix ending at position "
                "(seq_len - W + j).",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateElemTypeFromInputToOutput(ctx, 0, 1);

          const int64_t state_window = getAttribute(ctx, "state_window", 0);
          if (state_window < 0 || state_window > kMaxStateWindow) {
            fail_shape_inference("CausalConvWithState: state_window must be in [0, ", kMaxStateWindow,
                                 "], got ", state_window);
          }

          const int64_t dilation = getAttribute(ctx, "dilation", 1);
          if (dilation < 1) {
            fail_shape_inference("CausalConvWithState: dilation must be >= 1, got ", dilation);
          }

          const int64_t channels_last = getAttribute(ctx, "channels_last", 0);
          if (channels_last != 0 && channels_last != 1) {
            fail_shape_inference("CausalConvWithState: channels_last must be 0 or 1, got ",
                                 channels_last);
          }

          const int64_t ndim = getAttribute(ctx, "ndim", 1);
          if (ndim < 1 || ndim > 3) {
            fail_shape_inference("CausalConvWithState: ndim must be 1, 2, or 3, got ", ndim);
          }
          if (channels_last == 1 && ndim != 1) {
            fail_shape_inference("CausalConvWithState: channels_last requires ndim = 1");
          }

          // Output 0: same shape as input (batch_size, channels, ...)
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // Output 1: state shape is (batch_size, channels, [non-causal spatial dims...], k_last - 1)
          // For ndim=1: (B, C, k_1-1)
          // For ndim=2: (B, C, input_H, k_2-1)
          // For ndim=3: (B, C, input_D, input_H, k_3-1)
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
            auto& input_shape = getInputShape(ctx, 0);
            auto& weight_shape = getInputShape(ctx, 1);
            // weight is always channels-first: (channels, 1, k_1, ..., k_ndim), rank == ndim + 2.
            if (weight_shape.dim_size() != ndim + 2) {
              fail_shape_inference("CausalConvWithState: weight must have rank ndim + 2 (",
                                   ndim + 2, "), got rank ", weight_shape.dim_size());
            }
            if (channels_last == 1) {
              // (batch_size, sequence_length, d_1, ..., d_n). Check the lower bound.
              if (input_shape.dim_size() < 3) {
                fail_shape_inference("CausalConvWithState: channels_last input must have rank >= 3");
              }
            } else if (input_shape.dim_size() != ndim + 2) {
              fail_shape_inference("CausalConvWithState: input must have rank ndim + 2 (",
                                   ndim + 2, "), got rank ", input_shape.dim_size());
            }
            // (kernel_size - 1) * dilation, or an unset dim when kernel_size is symbolic.
            const int last_kernel_dim = weight_shape.dim_size() - 1;
            TensorShapeProto::Dimension state_length;
            if (weight_shape.dim(last_kernel_dim).has_dim_value()) {
              state_length.set_dim_value((weight_shape.dim(last_kernel_dim).dim_value() - 1) *
                                         dilation);
            }

            if (channels_last == 1) {
              // (batch_size, state_length, d_1, ..., d_n), optionally led by the window axis.
              // The trailing channel axes are copied verbatim from the input, so a caller that
              // keeps hyper-connections and hidden size separate gets the same split back.
              TensorShapeProto cl_state_shape;
              if (state_window > 0) {
                cl_state_shape.add_dim()->set_dim_value(state_window);
              }
              *cl_state_shape.add_dim() = input_shape.dim(0);
              *cl_state_shape.add_dim() = state_length;
              for (int i = 2; i < input_shape.dim_size(); ++i) {
                *cl_state_shape.add_dim() = input_shape.dim(i);
              }
              updateOutputShape(ctx, 1, cl_state_shape);
              return;
            }

            // state_window = W > 0 prepends a window axis, holding the carry state after each of
            // the last W positions (slot W-1 == the W = 0 tensor). The window axis leads the batch
            // axis so a slot is one contiguous (batch_size, channels, ...) block. W = 0 keeps the
            // legacy (batch_size, channels, ...) shape for backward compatibility.
            TensorShapeProto state_shape;
            if (state_window > 0) {
              state_shape.add_dim()->set_dim_value(state_window);
            }
            *state_shape.add_dim() = input_shape.dim(0);  // batch_size
            *state_shape.add_dim() = input_shape.dim(1);  // channels
            // Copy non-causal spatial dims from input (dims 2 .. 2+ndim-2)
            for (int64_t i = 0; i < ndim - 1; ++i) {
              *state_shape.add_dim() = input_shape.dim(static_cast<int>(2 + i));
            }
            // Causal (last) spatial dim: (kernel_size - 1) * dilation
            *state_shape.add_dim() = state_length;
            updateOutputShape(ctx, 1, state_shape);
          }
        }));

constexpr const char* VarlenCausalConvWithState_ver1_doc = R"DOC(
Stateful causal depthwise convolution over a packed, token-major batch of variable-length
sequences (CUDA and WebGPU).

input and output have shape (total_tokens, channels). cumulative_sequence_length is a
device-resident int32 tensor of shape (batch_size + 1); sequence i occupies
[cumulative_sequence_length[i], cumulative_sequence_length[i + 1]). Every sequence contributes
at least one token. weight has shape (channels, 1, kernel_size), and optional bias has shape
(channels). The convolution never reads across a sequence boundary.

initial_state is required and has shape (batch_size, channels, state_length), where
state_length = (kernel_size - 1) * dilation. It contains
the committed raw activation samples immediately preceding this call. final_state has the same
shape and type and is fully written with the state after each sequence's final token. State
uses the activation type because it stores raw samples, not accumulated convolution values.
initial_state and final_state may use the same allocation. Such in-place execution is
transaction-safe only when the whole operator call is unconditionally committed; a caller that
may select a prefix or roll back must preserve initial_state and replay the compact state update.

When state_update_capacity is positive, capture_count is required with shape (batch_size), and
state_update has shape (batch_size, state_update_capacity, channels).
For request b, slots [0, clamp(capture_count[b], 0,
min(state_update_capacity, sequence_length[b]))) contain the original local input token values.
These values represent the append component of each shift-left-and-append state transition.
All remaining slots are zero. capture_count is forbidden when state_update_capacity is zero.

For memory-safety containment, each GPU work item validates cumulative_sequence_length[0] == 0,
cumulative_sequence_length[batch_size] == total_tokens, and its local range
0 <= start < end <= total_tokens before accessing input, state, or output.
Malformed offsets cause affected work to return without those accesses; outputs are unspecified.
This device-side containment is not a synchronous validation or rejection mechanism.

The optional activation attribute supports none, SiLU, and Swish.

The dilation attribute spaces the kernel taps along the sequence axis: local token t of a request
reads that request's local positions t - (kernel_size - 1 - j) * dilation for tap j, and positions
before the request's first token come from the carry state. The carry state therefore holds
state_length = (kernel_size - 1) * dilation positions per request instead of kernel_size - 1.
Dilation 1 (the default) is the undilated case and keeps the original state length, so models
exported before the attribute existed are unaffected. input and output are already token-major
(sequence-major, channels-last), so this op needs no separate layout attribute.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    VarlenCausalConvWithState, 1,
    OpSchema()
        .SetDoc(VarlenCausalConvWithState_ver1_doc)
        .Attr("activation",
              "Fused activation function. One of: 'silu', 'swish', 'none'. "
              "Default is 'none'.",
              AttributeProto::STRING,
              std::string("none"))
        .Attr("dilation",
              "Spacing between kernel taps along the sequence axis. The receptive field spans "
              "(kernel_size - 1) * dilation positions before the current token, and "
              "initial_state / final_state hold that many positions per request. "
              "Must be >= 1. Default is 1 (undilated).",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("state_update_capacity",
              "Static number of compact contiguous-prefix transition values to expose per request. "
              "Valid range is [0, 8]. capture_count is required exactly when this is positive.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "input",
               "Token-major packed input with shape (total_tokens, channels).",
               "T")
        .Input(1,
               "weight",
               "Depthwise convolution kernel with shape (channels, 1, kernel_size).",
               "T")
        .Input(2,
               "cumulative_sequence_length",
               "Device tensor with shape (batch_size + 1) giving the half-open packed token "
               "range of each sequence.",
               "M")
        .Input(3,
               "bias",
               "Optional per-channel bias with shape (channels). Because the following "
               "initial_state input is required, an omitted bias must still occupy this "
               "position as an empty input name so initial_state stays at input index 4.",
               "T",
               OpSchema::Optional)
        .Input(4,
               "initial_state",
               "Required committed carry state with shape "
               "(batch_size, channels, (kernel_size - 1) * dilation).",
               "T")
        .Input(5,
               "capture_count",
               "Optional device int32 tensor with shape (batch_size). For each request, captures "
               "that many local tokens from the contiguous prefix, clamped to the sequence length "
               "and state_update_capacity. Required exactly when state_update_capacity is positive.",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "Token-major convolution output with the same shape as input.",
                "T")
        .Output(1,
                "final_state",
                "Fully written state after each sequence's final token, with shape "
                "(batch_size, channels, (kernel_size - 1) * dilation).",
                "T")
        .Output(2,
                "state_update",
                "Optional compact transition values with shape "
                "(batch_size, state_update_capacity, channels). Inactive slots are zero.",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain cumulative_sequence_length and capture_count to device int32 tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
          if (ctx.getNumOutputs() > 2) {
            propagateElemTypeFromInputToOutput(ctx, 0, 2);
          }
          const int64_t state_update_capacity = getAttribute(ctx, "state_update_capacity", 0);
          if (state_update_capacity < 0 || state_update_capacity > kMaxStateWindow) {
            fail_shape_inference("VarlenCausalConvWithState: state_update_capacity must be in [0, ",
                                 kMaxStateWindow, "], got ", state_update_capacity);
          }

          const int64_t dilation = getAttribute(ctx, "dilation", 1);
          if (dilation < 1) {
            fail_shape_inference("VarlenCausalConvWithState: dilation must be >= 1, got ", dilation);
          }

          // Output 0: same shape as input (total_tokens, channels)
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // State shapes use batch_size from cumulative_sequence_length, never total_tokens.
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1) && hasInputShape(ctx, 2)) {
            auto& input_shape = getInputShape(ctx, 0);
            auto& weight_shape = getInputShape(ctx, 1);
            auto& cu_seqlen_shape = getInputShape(ctx, 2);
            if (input_shape.dim_size() != 2) {
              fail_shape_inference(
                  "VarlenCausalConvWithState: input must have rank 2 "
                  "(total_tokens, channels)");
            }
            if (weight_shape.dim_size() != 3) {
              fail_shape_inference(
                  "VarlenCausalConvWithState: weight must have rank 3 "
                  "(channels, 1, kernel_size)");
            }
            if (cu_seqlen_shape.dim_size() != 1) {
              fail_shape_inference(
                  "VarlenCausalConvWithState: cumulative_sequence_length must "
                  "have rank 1");
            }
            auto& cu_dim = cu_seqlen_shape.dim(0);
            if (cu_dim.has_dim_value() && cu_dim.dim_value() < 2) {
              fail_shape_inference(
                  "VarlenCausalConvWithState: cumulative_sequence_length must have at least 2 elements");
            }
            if (input_shape.dim(0).has_dim_value() && cu_dim.has_dim_value() &&
                input_shape.dim(0).dim_value() < cu_dim.dim_value() - 1) {
              fail_shape_inference(
                  "VarlenCausalConvWithState: total_tokens must be at least batch_size");
            }
            if (hasInputShape(ctx, 5)) {
              auto& capture_count_shape = getInputShape(ctx, 5);
              if (capture_count_shape.dim_size() != 1) {
                fail_shape_inference("VarlenCausalConvWithState: capture_count must have rank 1");
              }
              if (cu_dim.has_dim_value() && capture_count_shape.dim(0).has_dim_value() &&
                  capture_count_shape.dim(0).dim_value() != cu_dim.dim_value() - 1) {
                fail_shape_inference("VarlenCausalConvWithState: capture_count must have shape (batch_size)");
              }
            }

            TensorShapeProto state_shape;
            // batch_size = cumulative_sequence_length.dim(0) - 1
            if (cu_dim.has_dim_value()) {
              state_shape.add_dim()->set_dim_value(cu_dim.dim_value() - 1);
            } else {
              state_shape.add_dim();  // unknown batch size
            }
            *state_shape.add_dim() = input_shape.dim(1);  // channels
            if (weight_shape.dim(2).has_dim_value()) {
              state_shape.add_dim()->set_dim_value((weight_shape.dim(2).dim_value() - 1) * dilation);
            } else {
              state_shape.add_dim();  // unknown (kernel_size - 1) * dilation
            }
            updateOutputShape(ctx, 1, state_shape);

            if (ctx.getNumOutputs() > 2) {
              TensorShapeProto state_update_shape;
              *state_update_shape.add_dim() = state_shape.dim(0);
              state_update_shape.add_dim()->set_dim_value(state_update_capacity);
              *state_update_shape.add_dim() = input_shape.dim(1);
              updateOutputShape(ctx, 2, state_update_shape);
            }
          }
        }));

constexpr const char* LinearAttention_ver1_doc = R"DOC(
Unified linear attention operator for autoregressive decoding (T=1) and prefill (T>1).

All inputs use 3D packed format [B, T, H*D]; q_num_heads and kv_num_heads are always
required. The op internally unpacks to 4D for computation.

The update_rule attribute selects the recurrence type:
- "linear": S_t = S_{t-1} + k_t ⊗ v_t; o_t = scale * q_t^T S_t
- "gated": S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t; o_t = scale * q_t^T S_t
- "delta": S_t = S_{t-1} + β_t * k_t ⊗ (v_t - S_{t-1}^T k_t); o_t = scale * q_t^T S_t
- "gated_delta": S_t = exp(g_t) * S_{t-1} + β_t * k_t ⊗ (v_t - exp(g_t) * S_{t-1}^T k_t); o_t = scale * q_t^T S_t

where g_t is the decay (in log-space), β_t is the update rate, and ⊗ denotes outer product.

Semantics: Equivalent to running the recurrent update sequentially for each token,
but may be implemented using chunk-parallel algorithms for GPU efficiency.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    LinearAttention, 1,
    OpSchema()
        .SetDoc(LinearAttention_ver1_doc)
        .Attr("update_rule",
              "The update rule for the linear attention recurrence. "
              "One of: 'linear', 'gated', 'delta', 'gated_delta'. Default is 'gated_delta'.",
              AttributeProto::STRING,
              std::string("gated_delta"))
        .Attr("scale",
              "Output scaling factor. When 0.0 (default), derives d_k = query.shape[-1] / q_num_heads "
              "and uses 1/sqrt(d_k). Set explicitly to override.",
              AttributeProto::FLOAT,
              0.0f)
        .Attr("q_num_heads",
              "Number of query heads. Always required.",
              AttributeProto::INT)
        .Attr("kv_num_heads",
              "Number of key/value heads. Always required.",
              AttributeProto::INT)
        .Attr("chunk_size",
              "Chunk size for the chunk-parallel WY decomposition during prefill (T>1). "
              "Tuning hint; does not affect output correctness.",
              AttributeProto::INT,
              static_cast<int64_t>(64))
        .Attr("state_window",
              "Number of trailing per-token recurrent states held by past_state and present_state. "
              "When 0 (default) the state tensors are 4D and hold only the state after the last "
              "token, i.e. the backward-compatible (B, H_kv, d_k, d_v). When W > 0 both are 5D with "
              "a LEADING axis of extent W, right-aligned: slot j is the state after token "
              "(T - W + j), so slot W-1 is always the state after the last token (identical to the "
              "W = 0 tensor) and is the slot past_state is read from. The window axis leads the "
              "batch axis so that each slot is one contiguous (B, H_kv, d_k, d_v) block. Slots "
              "below max(0, W - T) hold no token from this call and are filled with zeros. A "
              "window lets a speculative decoder roll the state back to an accepted prefix "
              "without replaying the forward. Valid range is [0, 8].",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "query",
               "Query vectors with 3D packed shape (B, T, H_q * d_k). "
               "Heads are packed into the last dimension.",
               "T")
        .Input(1,
               "key",
               "Key vectors with 3D packed shape (B, T, H_kv * d_k). "
               "Should be L2-normalized for delta/gated_delta modes.",
               "T")
        .Input(2,
               "value",
               "Value vectors with 3D packed shape (B, T, H_kv * d_v).",
               "T")
        .Input(3,
               "past_state",
               "Recurrent state from previous step with shape (B, H_kv, d_k, d_v), or "
               "(W, B, H_kv, d_k, d_v) when state_window = W > 0, in which case only slot W-1 is "
               "read. If not provided, defaults to zeros.",
               "S",
               OpSchema::Optional)
        .Input(4,
               "decay",
               "Exponential decay gate in log-space. 3D packed shape: "
               "(B, T, H_kv * d_k) for per-key-dimension decay (GLA/RWKV-6), or "
               "(B, T, H_kv) for per-head scalar decay (DeltaNet/RetNet). "
               "Required for 'gated' and 'gated_delta' modes.",
               "T",
               OpSchema::Optional)
        .Input(5,
               "beta",
               "Update rate (sigmoid output). 3D packed shape: "
               "(B, T, H_kv) or (B, T, 1). "
               "Required for 'delta' and 'gated_delta' modes.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                // Kept free of angle brackets: gen_contrib_doc.py interpolates this
                // description straight into an HTML <dd> element without escaping.
                "Attention output with 3D packed shape (B, T, max(H_q, H_kv) * d_v). "
                "Standard GQA emits one output per query head; inverse GQA, where "
                "H_kv exceeds H_q, emits one per KV head.",
                "T")
        .Output(1,
                "present_state",
                "Updated recurrent state with shape (B, H_kv, d_k, d_v), or (W, B, H_kv, d_k, d_v) "
                "when state_window = W > 0. Slot W-1 is the state after the last token; slot j is "
                "the state after token (T - W + j).",
                "S")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("S",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain state types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateElemTypeFromInputToOutput(ctx, 0, 1);

          const int64_t state_window = getAttribute(ctx, "state_window", 0);
          if (state_window < 0 || state_window > kMaxStateWindow) {
            fail_shape_inference("LinearAttention: state_window must be in [0, ", kMaxStateWindow,
                                 "], got ", state_window);
          }

          // Read required attributes
          auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
          auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
          int64_t q_num_heads = (q_num_heads_attr && q_num_heads_attr->has_i()) ? q_num_heads_attr->i() : 0;
          int64_t kv_num_heads = (kv_num_heads_attr && kv_num_heads_attr->has_i()) ? kv_num_heads_attr->i() : 0;

          // Output 0: (B, T, max(H_q, H_kv) * d_v) — 3D packed
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2) && q_num_heads > 0 && kv_num_heads > 0) {
            auto& query_shape = getInputShape(ctx, 0);
            auto& value_shape = getInputShape(ctx, 2);
            if (query_shape.dim_size() < 3) {
              fail_shape_inference("LinearAttention: query must have rank >= 3");
            }
            if (value_shape.dim_size() < 3) {
              fail_shape_inference("LinearAttention: value must have rank >= 3");
            }
            TensorShapeProto output_shape;
            *output_shape.add_dim() = query_shape.dim(0);  // B
            *output_shape.add_dim() = query_shape.dim(1);  // T
            // Output hidden = max(H_q, H_kv) * d_v, matching Compute: standard GQA (H_q >= H_kv)
            // emits one output per query head; inverse GQA (H_q < H_kv) one per KV head.
            // d_v = value.dim(2) / kv_num_heads.
            if (value_shape.dim(2).has_dim_value()) {
              int64_t d_v = value_shape.dim(2).dim_value() / kv_num_heads;
              int64_t out_heads = q_num_heads > kv_num_heads ? q_num_heads : kv_num_heads;
              output_shape.add_dim()->set_dim_value(out_heads * d_v);
            } else {
              output_shape.add_dim();  // unknown
            }
            updateOutputShape(ctx, 0, output_shape);
          }

          // Output 1: present_state shape (B, H_kv, d_k, d_v) — 4D, or (W, B, H_kv, d_k, d_v) — 5D
          // when state_window = W > 0. W = 0 keeps the legacy 4D shape for backward compatibility.
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2) && q_num_heads > 0 && kv_num_heads > 0) {
            auto& query_shape = getInputShape(ctx, 0);
            auto& value_shape = getInputShape(ctx, 2);
            if (query_shape.dim_size() < 3 || value_shape.dim_size() < 3) {
              // Already validated in Output 0 block above; skip if shapes are invalid.
              return;
            }
            TensorShapeProto state_shape;
            if (state_window > 0) {
              state_shape.add_dim()->set_dim_value(state_window);  // W
            }
            *state_shape.add_dim() = query_shape.dim(0);         // B
            state_shape.add_dim()->set_dim_value(kv_num_heads);  // H_kv
            // d_k = query.dim(2) / q_num_heads
            if (query_shape.dim(2).has_dim_value()) {
              state_shape.add_dim()->set_dim_value(query_shape.dim(2).dim_value() / q_num_heads);
            } else {
              state_shape.add_dim();
            }
            // d_v = value.dim(2) / kv_num_heads
            if (value_shape.dim(2).has_dim_value()) {
              state_shape.add_dim()->set_dim_value(value_shape.dim(2).dim_value() / kv_num_heads);
            } else {
              state_shape.add_dim();
            }
            updateOutputShape(ctx, 1, state_shape);
          } else if (hasInputShape(ctx, 3)) {
            propagateShapeFromInputToOutput(ctx, 3, 1);
          }
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedDeltaNetStateReplay, 1,
    OpSchema()
        .SetDoc(R"DOC(
Replay one request's captured GatedDeltaNet transitions into an inactive recurrent-state row.
The WebGPU implementation uses FP32 backing tensors and CPU metadata; it does not recompute
normalization, gates, beta, or retrieval. It waits for checked GPU completion before returning.

`metadata` is int64 [11], ordered as:
  source_offset, destination_offset, decay_offset, key_offset, delta_offset,
  Hv, Dv, Dk, Hk, capacity, kept_count.
Offsets are FP32-element offsets into the full backing tensors, not storage-binding byte offsets.
All offsets are nonnegative; dimensions and capacity are positive; 1 <= kept_count <= capacity.
The caller must ensure that all kept transitions were actually captured.

The source_state selected by source_offset must be the state snapshot from immediately before
the captured GatedDeltaNet transitions began. Using the producer's final state or another
same-shaped snapshot is invalid because captured delta is state-dependent.
The capsule geometry (Hv, Dv, Dk, Hk) and capacity must match the producer that generated it.
capacity is the original capsule capture/layout capacity. kept_count only selects the leading
transitions to replay; kept_count must NOT be substituted for capacity when deriving the
decay/key/delta section offsets.

State rows have layout [Hv, Dv, Dk]. The capsule contains decay [capacity, Hv],
key [capacity, Hk, Dk], and delta [capacity, Hv, Dv] starting at their respective offsets.
For every state element (h,v,i):
  hk = floor(h * Hk / Hv)
  s = source[h,v,i]
  for t = 0 .. kept_count-1:
    s *= decay[t,h]
    s += key[t,hk,i] * delta[t,h,v]
  destination[h,v,i] = s
Unused capsule tail entries are not read. All other destination elements are preserved.

The output must be explicitly prebound to the same full allocation as destination_backing.
Destination backing must be distinct from source and capsule backing. This first implementation
supports one descriptor, uint32 shader indexing and no graph capture.
Source and capsule may share read-only backing; storage bindings are deduplicated only when
their BufferViews have the same Program binding owner, not merely equal raw buffer handles.
)DOC")
        .Input(0, "source_backing",
               "Full rank-1 FP32 backing tensor containing the state snapshot from immediately before capture.", "T")
        .Input(1, "capsule_backing", "Full rank-1 FP32 compact-update backing tensor.", "T")
        .Input(2, "destination_backing", "Full rank-1 FP32 inactive destination-state backing tensor.", "T")
        .Input(3, "metadata", "CPU int64 [11] containing offsets, geometry, capacity and kept_count.", "TI")
        .Output(0, "destination_out", "Same backing allocation and shape as destination_backing.", "T")
        .TypeConstraint("T", {"tensor(float)"}, "State and capsule tensors are FP32.")
        .TypeConstraint("TI", {"tensor(int64)"}, "CPU replay metadata is int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
          for (size_t index = 0; index < 4; ++index) {
            if (!hasInputShape(ctx, index)) {
              continue;
            }
            const auto& shape = getInputShape(ctx, index);
            if (shape.dim_size() != 1) {
              fail_shape_inference("GatedDeltaNetStateReplay inputs must have rank 1.");
            }
            if (index == 3 && shape.dim(0).has_dim_value() && shape.dim(0).dim_value() != 11) {
              fail_shape_inference("GatedDeltaNetStateReplay metadata must have shape [11].");
            }
          }
          if (hasInputShape(ctx, 2)) {
            propagateShapeFromInputToOutput(ctx, 2, 0);
          }
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    ConvStateReplay, 1,
    OpSchema()
        .SetDoc(R"DOC(
Reconstruct rolling convolution history after retaining a prefix of captured inputs, without
performing convolution arithmetic. The WebGPU-only implementation copies FP16 or FP32 payload
bits exactly, including signed zero and NaN payloads, and waits for checked GPU completion.

Inputs 0-2 are full rank-1 backing allocations of the same element type.
CPU int64 metadata [8] is ordered as:
  source_offset_bytes, destination_offset_bytes, capture_offset_bytes,
  batch_size, channels, history_width, capacity, kept_count.
Offsets are logical byte offsets into the full backing tensors, aligned to the element size
(two bytes for FP16, four for FP32), not WebGPU storage-binding offsets.
All fields are nonnegative; channels must be positive; 0 <= kept_count <= capacity.
Zero batch size, history width, or capture capacity is valid. Zero state size is a no-op.
Geometry, backing byte extents and metadata must fit the implementation's uint32 indexing range.

The source view is the incoming snapshot immediately before capture, with layout
[batch_size, channels, history_width]. Captured inputs have layout [batch_size, capacity, channels].
capacity is the original capture stride, not kept_count. All retained inputs must have been captured.
For each batch b, channel c and position p, with W = history_width and k = kept_count:
  destination[b,c,p] = source[b,c,p+k]                 if p+k < W
                     = capture[b,p+k-W,c]            otherwise.
For k >= W only the last W inputs of the retained prefix are used; unused capture tails are not read.

The output must be explicitly prebound to the same full allocation as destination_backing.
Destination backing must be distinct from source and capture backing, even for disjoint logical views.
Only the designated destination view is modified; neighboring elements and all source/capture data
are preserved. One descriptor is supported per invocation; graph capture is not supported.
)DOC")
        .Input(0, "source_backing", "Full backing tensor containing the incoming convolution history.", "T")
        .Input(1, "capture_backing", "Full backing tensor containing captured token-major inputs.", "T")
        .Input(2, "destination_backing", "Full inactive destination-state backing tensor.", "T")
        .Input(3, "metadata", "CPU int64 [8]: byte offsets, batch, channels, history, capacity, kept_count.", "TI")
        .Output(0, "destination_out", "Same backing allocation and shape as destination_backing.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "State and captured inputs are FP16 or FP32.")
        .TypeConstraint("TI", {"tensor(int64)"}, "CPU replay metadata is int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
          for (size_t index = 0; index < 4; ++index) {
            if (!hasInputShape(ctx, index)) {
              continue;
            }
            const auto& shape = getInputShape(ctx, index);
            if (shape.dim_size() != 1) {
              fail_shape_inference("ConvStateReplay inputs must have rank 1.");
            }
            if (index == 3 && shape.dim(0).has_dim_value() && shape.dim(0).dim_value() != 8) {
              fail_shape_inference("ConvStateReplay metadata must have shape [8].");
            }
          }
          if (hasInputShape(ctx, 2)) {
            propagateShapeFromInputToOutput(ctx, 2, 0);
          }
        }));

constexpr const char* GatedDeltaNet_ver1_doc = R"DOC(
Packed (token-major) gated delta network / linear attention with an explicit recurrent state.
Implemented by CUDA and native WebGPU execution providers. WebGPU supports float and float16
with scalar decay, `head_size_qk <= 256`, and compact state updates.

Layout. Query, key and value are token-major, so head counts are derived from the shapes
rather than from attributes:

  query [total_tokens, num_heads_q, head_size_qk]
  key   [total_tokens, num_heads_k, head_size_qk]
  value [total_tokens, num_heads_v, head_size_v]

Alternatively, key and value may be omitted and query contains packed QKV with shape
`[total_tokens, 2 * num_heads_q * head_size_qk + num_heads_v * head_size_v]`.
`initial_state` is required for packed QKV to determine the head counts and head sizes.

The leading token axis may instead be spelled as an explicit `[batch_size, sequence_length]`
pair, making query/key/value (and the output) rank 4 and decay/beta rank 3. The memory layout
is identical; the rank-4 spelling exists so an exporter can round-trip a `[B, S, H*D]`
activation with static Reshape targets instead of Shape-derived ones. Ragged packing
(`cu_seqlens`) requires the rank-3 spelling.

`num_heads_q` must equal `num_heads_k`, and `num_heads_v` must be a positive multiple of
`num_heads_q` (inverse grouped-query attention: each query/key head is shared by
`num_heads_v / num_heads_q` value heads). Decay, beta, the state and the output are all at
`num_heads_v`.

Sequence packing. When `cu_seqlens` is provided it is a device int32 tensor of length
`batch_size + 1` holding the exclusive prefix sums of the per-request token counts, so
requests may have different lengths. When it is absent the packing is uniform and the batch
size is taken from `initial_state`, which is then required.

State. `initial_state` and `final_state` are V-major, `[batch_size, num_heads_v, head_size_v,
head_size_qk]`, and always float regardless of the query/key/value type: the recurrence
boundary is where reduced precision hurts most. The two may be the same allocation; the
implementation reads the whole incoming state before writing any of it.

Compact state updates. When `state_update_capacity` C is greater than zero, `capture_count`
is required with shape `[batch_size]`. For request b, the first `capture_count[b]` local token
transitions (clamped on device to `[0, min(C, sequence_length)]`) are emitted in one `state_update`
float tensor `[batch_size, C * (num_heads_v + num_heads_k * head_size_qk + num_heads_v * head_size_v)]`.
Each row is struct-of-arrays: all decay values, then all keys, then all deltas. Entries at positions
greater than or equal to `min(capture_count[b], C, sequence_length)` are unspecified; consumers must
read only the captured prefix. The key retains its shared `num_heads_k` representation. For scalar
decay the decoded factors replay one transition as `S *= decay; S += outer(key, delta)`.
Per-key-dimension decay is not supported when compact updates are enabled. `capture_count` is
forbidden when C is zero.

The optional CPU input `state_update_active` has shape `[1]`. When zero, transition capture is
disabled, `capture_count` is ignored, `state_update` is zero-filled, and the planner may use an
engine that cannot emit compact updates. Omitting it preserves the conservative behavior of
treating capture as active.

Recurrence, per value head, with S the [head_size_qk x head_size_v] state:

  S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T
  o_t = scale * S_t^T q_t

`update_rule` selects which terms are present: 'linear' drops both the decay and the delta
retrieval, 'gated' keeps only the decay, 'delta' keeps only the retrieval, and 'gated_delta'
keeps both.

The delta family ('delta' and 'gated_delta') requires L2-normalized keys. Without them the
per-chunk system (I + M) is arbitrarily ill-conditioned and the recurrence diverges. Either
normalize upstream or set `qk_l2_norm=1` to have the operator do it.

Fused activations. `gate_activation='qwen'` computes the effective decay in float32 from the
raw projection carried by `decay`:

  g = -exp(a_log) * Softplus(decay + dt_bias)

`beta_activation='sigmoid'` applies a sigmoid to `beta`, and `qk_l2_norm=1` L2-normalizes each
query and key head vector. Folding these in avoids materializing the intermediates and keeps
the gate arithmetic in float32 independent of the input type.

)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedDeltaNet, 1,
    OpSchema()
        .SetDoc(GatedDeltaNet_ver1_doc)
        .Attr("update_rule",
              "One of: 'linear', 'gated', 'delta', 'gated_delta'. Default is 'gated_delta'.",
              AttributeProto::STRING, std::string("gated_delta"))
        .Attr("scale",
              "Output scaling factor. When 0.0 (default) uses 1/sqrt(head_size_qk).",
              AttributeProto::FLOAT, 0.0f)
        .Attr("gate_activation",
              "'none' (default) treats `decay` as the effective log-space decay. 'qwen' computes "
              "-exp(a_log) * Softplus(decay + dt_bias) in float32.",
              AttributeProto::STRING, std::string("none"))
        .Attr("beta_activation",
              "'none' (default) treats `beta` as the effective update rate. 'sigmoid' applies a "
              "sigmoid.",
              AttributeProto::STRING, std::string("none"))
        .Attr("qk_l2_norm",
              "When 1, L2-normalize each query and key head vector before the recurrence. "
              "Default 0.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("chunk_size",
              "Tuning hint for the chunk-parallel prefill algorithm. On CUDA, 32 pins the narrow "
              "chunk and any other value lets the implementation select the widest chunk the device "
              "can hold. On WebGPU, 16 opts into the chunkwise prefill route on supported Apple silicon; "
              "default 64 retains recurrent execution. Default 64.",
              AttributeProto::INT, static_cast<int64_t>(64))
        .Attr("state_update_capacity",
              "Capacity C for compact contiguous-prefix transition capture, in [0, 8]. "
              "0 (default) disables compact state-update outputs.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "query", "Query or packed QKV, shaped as described above.", "T")
        .Input(1, "key", "Key, shape (total_tokens, num_heads_k, head_size_qk)", "T",
               OpSchema::Optional)
        .Input(2, "value", "Value, shape (total_tokens, num_heads_v, head_size_v)", "T",
               OpSchema::Optional)
        .Input(3, "cu_seqlens",
               "Exclusive prefix sums of the per-request token counts, shape (batch_size + 1). "
               "Absent means uniform packing.",
               "TI", OpSchema::Optional)
        .Input(4, "decay",
               "Log-space decay, shape (total_tokens, num_heads_v) for a scalar per-head decay or "
               "(total_tokens, num_heads_v, head_size_qk) for a per-key-dimension decay.",
               "TS", OpSchema::Optional)
        .Input(5, "beta", "Update rate, shape (total_tokens, num_heads_v)", "TS",
               OpSchema::Optional)
        .Input(6, "initial_state",
               "Recurrent state, shape (batch_size, num_heads_v, head_size_v, head_size_qk), "
               "V-major. Required for packed QKV. May alias final_state.",
               "TS", OpSchema::Optional)
        .Input(7, "a_log",
               "Per-head A_log, shape (num_heads_v). Requires gate_activation=qwen.",
               "TS", OpSchema::Optional)
        .Input(8, "dt_bias",
               "Per-head gate bias, shape (num_heads_v). "
               "Requires gate_activation=qwen.",
               "TS", OpSchema::Optional)
        .Input(9, "capture_count",
               "Number of leading local token transitions to capture for each request, shape "
               "(batch_size). Clamped on device to [0, min(state_update_capacity, sequence_length)]. "
               "Required exactly when state_update_capacity is positive.",
               "TI", OpSchema::Optional)
        .Input(10, "state_update_active",
               "CPU int32 control with shape (1). Zero disables transition capture, ignores "
               "capture_count, and produces a zero-filled state_update. Omission is conservative.",
               "TI", OpSchema::Optional)
        .Output(0, "output",
                "Output, shape (total_tokens, max(num_heads_q, num_heads_v), head_size_v)", "T")
        .Output(1, "final_state",
                "State after the last token of each request, shape "
                "(batch_size, num_heads_v, head_size_v, head_size_qk)",
                "TS", OpSchema::Optional)
        .Output(2, "state_update",
                "Struct-of-arrays compact transition factors, shape "
                "(batch_size, state_update_capacity * (num_heads_v + num_heads_k * "
                "head_size_qk + num_heads_v * head_size_v)).",
                "TS", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain query/key/value/output types.")
        .TypeConstraint("TS", {"tensor(float)"},
                        "State, gate, beta and compact state-update tensors are always float.")
        .TypeConstraint("TI", {"tensor(int32)"}, "Constrain index and count tensors to int32.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumOutputs() > 1) {
            updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          }
          if (ctx.getNumOutputs() > 2) {
            updateOutputElemType(ctx, 2, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          }

          const bool has_key = ctx.getNumInputs() > 1 && ctx.getInputType(1) != nullptr;
          const bool has_value = ctx.getNumInputs() > 2 && ctx.getInputType(2) != nullptr;
          if (has_key != has_value) {
            fail_shape_inference("GatedDeltaNet: key and value must both be present or both be omitted");
          }
          const bool packed_qkv = !has_key;
          const bool has_state = ctx.getNumInputs() > 6 && ctx.getInputType(6) != nullptr;
          if (packed_qkv && !has_state) {
            fail_shape_inference("GatedDeltaNet: initial_state is required for packed QKV");
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          const auto& query_shape = getInputShape(ctx, 0);
          const int rank = query_shape.dim_size();
          const int token_dims = rank - (packed_qkv ? 1 : 2);
          if ((packed_qkv && rank != 2 && rank != 3) ||
              (!packed_qkv && rank != 3 && rank != 4)) {
            fail_shape_inference(
                "GatedDeltaNet: packed QKV must have rank 2 or 3, and separate "
                "query, key and value inputs must have rank 3 or 4");
          }

          const ONNX_NAMESPACE::TensorShapeProto* value_shape = nullptr;
          if (!packed_qkv) {
            if (!hasInputShape(ctx, 2)) {
              return;
            }
            value_shape = &getInputShape(ctx, 2);
            if (value_shape->dim_size() != rank) {
              fail_shape_inference("GatedDeltaNet: query and value must have the same rank");
            }
          }

          const ONNX_NAMESPACE::TensorShapeProto* state_shape = nullptr;
          if (ctx.getNumInputs() > 6 && hasInputShape(ctx, 6)) {
            state_shape = &getInputShape(ctx, 6);
            if (state_shape->dim_size() != 4) {
              fail_shape_inference("GatedDeltaNet: initial_state must have rank 4");
            }
          }

          if (packed_qkv) {
            if (state_shape == nullptr) {
              return;
            }
            if (query_shape.dim(token_dims).has_dim_value() &&
                state_shape->dim(1).has_dim_value() && state_shape->dim(2).has_dim_value() &&
                state_shape->dim(3).has_dim_value()) {
              const int64_t packed_size = query_shape.dim(token_dims).dim_value();
              const int64_t num_heads_v = state_shape->dim(1).dim_value();
              const int64_t head_size_v = state_shape->dim(2).dim_value();
              const int64_t head_size_qk = state_shape->dim(3).dim_value();
              constexpr int64_t max_dimension = std::numeric_limits<int64_t>::max();
              if (num_heads_v <= 0 || head_size_v <= 0 || head_size_qk <= 0) {
                fail_shape_inference("GatedDeltaNet: packed QKV head counts and head sizes must be positive");
              }
              if (num_heads_v > max_dimension / head_size_v || head_size_qk > max_dimension / 2) {
                fail_shape_inference("GatedDeltaNet: packed QKV dimensions overflow int64");
              }
              const int64_t value_size = num_heads_v * head_size_v;
              const int64_t query_key_size = 2 * head_size_qk;
              if (packed_size <= value_size || (packed_size - value_size) % query_key_size != 0) {
                fail_shape_inference(
                    "GatedDeltaNet: packed QKV last dimension must be 2 * num_heads_q * head_size_qk + "
                    "num_heads_v * head_size_v");
              }
              const int64_t num_heads_q = (packed_size - value_size) / query_key_size;
              if (num_heads_v % num_heads_q != 0) {
                fail_shape_inference("GatedDeltaNet: num_heads_v must be a positive multiple of num_heads_q");
              }
            }
            const bool has_cu_seqlens = ctx.getNumInputs() > 3 && ctx.getInputType(3) != nullptr;
            if (!has_cu_seqlens && token_dims == 1 && query_shape.dim(0).has_dim_value() &&
                state_shape->dim(0).has_dim_value()) {
              const int64_t total_tokens = query_shape.dim(0).dim_value();
              const int64_t batch = state_shape->dim(0).dim_value();
              if (batch <= 0 || total_tokens <= 0 || total_tokens % batch != 0) {
                fail_shape_inference(
                    "GatedDeltaNet: total_tokens must be divisible by batch, with positive token count and batch size");
              }
            }
          }

          ONNX_NAMESPACE::TensorShapeProto out_shape;
          for (int i = 0; i < token_dims; ++i) {
            *out_shape.add_dim() = query_shape.dim(i);
          }
          if (packed_qkv) {
            *out_shape.add_dim() = state_shape->dim(1);
            *out_shape.add_dim() = state_shape->dim(2);
          } else {
            if (query_shape.dim(token_dims).has_dim_value() &&
                value_shape->dim(token_dims).has_dim_value()) {
              out_shape.add_dim()->set_dim_value(std::max(query_shape.dim(token_dims).dim_value(),
                                                          value_shape->dim(token_dims).dim_value()));
            } else {
              out_shape.add_dim();
            }
            *out_shape.add_dim() = value_shape->dim(token_dims + 1);
          }
          updateOutputShape(ctx, 0, out_shape);

          auto add_batch_dim = [&](ONNX_NAMESPACE::TensorShapeProto& shape) {
            if (ctx.getNumInputs() > 9 && hasInputShape(ctx, 9) && getInputShape(ctx, 9).dim_size() == 1) {
              *shape.add_dim() = getInputShape(ctx, 9).dim(0);
            } else if (rank == (packed_qkv ? 3 : 4)) {
              *shape.add_dim() = query_shape.dim(0);
            } else if (state_shape != nullptr) {
              *shape.add_dim() = state_shape->dim(0);
            } else {
              shape.add_dim();
            }
          };

          const int64_t state_update_capacity =
              getAttribute(ctx, "state_update_capacity", static_cast<int64_t>(0));
          if (state_update_capacity < 0 || state_update_capacity > kMaxStateWindow) {
            fail_shape_inference("GatedDeltaNet: state_update_capacity must be in [0, ",
                                 kMaxStateWindow, "], got ", state_update_capacity);
          }
          if (ctx.getNumOutputs() > 2) {
            ONNX_NAMESPACE::TensorShapeProto capsule_shape;
            add_batch_dim(capsule_shape);
            auto* width = capsule_shape.add_dim();
            if (!packed_qkv && query_shape.dim(token_dims).has_dim_value() &&
                query_shape.dim(token_dims + 1).has_dim_value() &&
                value_shape->dim(token_dims).has_dim_value() &&
                value_shape->dim(token_dims + 1).has_dim_value()) {
              // num_heads_k is constrained to equal num_heads_q, so query supplies it.
              const int64_t num_heads_k = query_shape.dim(token_dims).dim_value();
              const int64_t head_size_qk = query_shape.dim(token_dims + 1).dim_value();
              const int64_t num_heads_v = value_shape->dim(token_dims).dim_value();
              const int64_t head_size_v = value_shape->dim(token_dims + 1).dim_value();
              if (num_heads_k <= 0 || head_size_qk <= 0 || num_heads_v <= 0 || head_size_v <= 0) {
                fail_shape_inference(
                    "GatedDeltaNet: head counts and head sizes must be positive");
              }

              constexpr int64_t max_dimension = std::numeric_limits<int64_t>::max();
              if (num_heads_k > max_dimension / head_size_qk ||
                  num_heads_v > max_dimension / head_size_v) {
                fail_shape_inference("GatedDeltaNet: state_update width overflows int64");
              }
              const int64_t key_width = num_heads_k * head_size_qk;
              const int64_t value_width = num_heads_v * head_size_v;
              if (num_heads_v > max_dimension - key_width) {
                fail_shape_inference("GatedDeltaNet: state_update width overflows int64");
              }
              const int64_t width_without_capacity = num_heads_v + key_width;
              if (value_width > max_dimension - width_without_capacity) {
                fail_shape_inference("GatedDeltaNet: state_update width overflows int64");
              }
              const int64_t per_token_width = width_without_capacity + value_width;
              if (state_update_capacity > 0 &&
                  per_token_width > max_dimension / state_update_capacity) {
                fail_shape_inference("GatedDeltaNet: state_update width overflows int64");
              }
              width->set_dim_value(state_update_capacity * per_token_width);
            }
            updateOutputShape(ctx, 2, capsule_shape);
          }

          if (state_shape != nullptr) {
            if (ctx.getNumOutputs() > 1) {
              updateOutputShape(ctx, 1, *state_shape);
            }
          }
        }));

constexpr const char* LinearAttentionGate_ver1_doc = R"DOC(
Fuses the gate projections that feed LinearAttention's gated-delta recurrence:

  decay = decay_scale * Softplus(a + dt_bias)
  beta  = Sigmoid(b)                            (only when b is provided)

Reference implementations compute the decay in float32 because exp(decay) inside the
recurrence exponentially amplifies any precision loss. Exporters therefore emit
Cast -> Add -> Softplus -> Mul -> Cast, which is five kernel launches on a tensor with
only num_heads elements per token. This operator keeps the intermediates in float32
registers so a single launch replaces the whole chain.

dt_bias and decay_scale are float32 per-head vectors of length H. decay_scale is the
already-negated -exp(A_log) factor.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    LinearAttentionGate, 1,
    OpSchema()
        .SetDoc(LinearAttentionGate_ver1_doc)
        .Input(0,
               "a",
               "Decay gate projection with shape (B, T, H).",
               "T")
        .Input(1,
               "dt_bias",
               "Per-head float32 bias added to a, with shape (H).",
               "TF")
        .Input(2,
               "decay_scale",
               "Per-head float32 multiplier applied to Softplus(a + dt_bias), with shape (H). "
               "For gated DeltaNet this is -exp(A_log).",
               "TF")
        .Input(3,
               "b",
               "Update-rate projection with shape (B, T, H). Required when the beta output is requested.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "decay",
                "decay_scale * Softplus(a + dt_bias) with shape (B, T, H).",
                "T")
        .Output(1,
                "beta",
                "Sigmoid(b) with shape (B, T, H).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain gate input and output types to float tensors.")
        .TypeConstraint("TF",
                        {"tensor(float)"},
                        "Constrain the per-head parameters to float32.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumOutputs() > 1) {
            if (ctx.getNumInputs() < 4 || ctx.getInputType(3) == nullptr) {
              fail_shape_inference("The b input is required when the beta output is requested.");
            }
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
            if (hasInputShape(ctx, 3)) {
              propagateShapeFromInputToOutput(ctx, 3, 1);
            } else {
              propagateShapeFromInputToOutput(ctx, 0, 1);
            }
          }
        }));

constexpr const char* GatedRMSNorm_ver1_doc = R"DOC(
Gated RMS normalization as used by Mamba2 / gated DeltaNet attention outputs:

  Y = X * rsqrt(mean(X^2) + epsilon) * scale * gate_activation(gate)

where `gate_activation` is one of:
- `silu` or `swish`: `z * sigmoid(z)`
- `sigmoid`: `sigmoid(z)`

The mean of squares is taken over the trailing `C` elements of each row, where `C` is the
length of `scale`; the input's last dimension must be a multiple of `C`, which lets a
per-head norm run on a packed (B, T, H * C) tensor without any surrounding Reshape.
All arithmetic including gate activation is done in float32 regardless of the tensor type, matching
the reference implementation, so this replaces the exported
SimplifiedLayerNormalization -> Cast -> Sigmoid -> Mul -> Cast -> Mul -> Cast chain with a
single launch.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedRMSNorm, 1,
    OpSchema()
        .SetDoc(GatedRMSNorm_ver1_doc)
        .Attr("epsilon",
              "Epsilon added to the mean of squares before the reciprocal square root.",
              AttributeProto::FLOAT,
              1e-5f)
        .Attr("activation",
              "Fused gate activation. One of: 'silu', 'swish', 'sigmoid'. "
              "'swish' is an alias of 'silu'.",
              AttributeProto::STRING,
              std::string("silu"))
        .Input(0,
               "X",
               "Input tensor with shape (..., H * C). Normalization is applied over each "
               "contiguous group of C elements.",
               "T")
        .Input(1,
               "scale",
               "Normalization weight with shape (C).",
               "T")
        .Input(2,
               "gate",
               "Gate tensor with the same shape as X.",
               "T")
        .Output(0,
                "Y",
                "Output tensor with the same shape as X.",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* BranchwiseRMSNorm_ver1_doc = R"DOC(
Applies RMS normalization independently to each branch. X may use grouped shape
(..., C, H), or flattened shape (..., C * H) when num_branches is specified.
The optional scale may have shape (C * H), (C, H), or (H). Arithmetic is
performed in float32 and the result is converted to T.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    BranchwiseRMSNorm, 1,
    OpSchema()
        .SetDoc(BranchwiseRMSNorm_ver1_doc)
        .Attr("epsilon", "Epsilon added before reciprocal square root.",
              AttributeProto::FLOAT, 1e-5f)
        .Attr("num_branches",
              "Number of branches for flattened input. Omit or set to zero for grouped input.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "X", "Grouped (..., C, H) or flattened (..., C * H) input.", "T")
        .Input(1, "scale", "Optional scale with shape (C * H), (C, H), or (H).",
               "M", OpSchema::Optional)
        .Output(0, "Y", "RMS-normalized output with the same shape as X.", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output to floating-point tensors.")
        .TypeConstraint("M",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain scale to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* ScaledSiLU_ver1_doc = R"DOC(
Computes SiLU after scaling, with explicit T rounding:
Z_T = cast_T(effective_scale * X), S_T = cast_T(sigmoid(Z_T)), and
Y = cast_T(Z_T * S_T). effective_scale is the scalar input when present,
otherwise it is the alpha attribute.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    ScaledSiLU, 1,
    OpSchema()
        .SetDoc(ScaledSiLU_ver1_doc)
        .Attr("alpha", "Scale used when the optional scale input is absent.",
              AttributeProto::FLOAT, 1.0f)
        .Input(0, "X", "Input tensor.", "T")
        .Input(1, "scale", "Optional scalar scale that overrides alpha.", "M", OpSchema::Optional)
        .Output(0, "Y", "Output with the same shape as X.", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output to floating-point tensors.")
        .TypeConstraint("M",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain the optional scale to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* HyperConnectionPreMix_ver1_doc = R"DOC(
Reduces C streams to one feature tensor without applying an activation:
Y[..., h] = reduction_scale * sum_c(X[..., c, h] * pre_mix[..., c, h]).
pre_mix may have shape (..., C), (..., C, 1), (..., C, H), or (..., C * H)
for flattened X. X may be grouped (..., C, H), or flattened (..., C * H)
when num_branches is specified.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    HyperConnectionPreMix, 1,
    OpSchema()
        .SetDoc(HyperConnectionPreMix_ver1_doc)
        .Attr("num_branches",
              "Number of branches for flattened streams. Omit or set to zero for grouped streams.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("reduction_scale", "Multiplier applied to the branch reduction.",
              AttributeProto::FLOAT, 1.0f)
        .Input(0, "streams", "Grouped (..., C, H) or flattened (..., C * H) streams.", "T")
        .Input(1, "pre_mix", "Branch or feature gates with shape (..., C), (..., C, 1), (..., C, H), or (..., C * H) for flattened streams.", "M")
        .Output(0, "output", "Reduced feature tensor with shape (..., H).", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain streams and output to floating-point tensors.")
        .TypeConstraint("M",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain mixing weights to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          const auto& input_shape = getInputShape(ctx, 0);
          auto* output_shape = getOutputShape(ctx, 0);
          const auto branches = getAttribute(ctx, "num_branches", 0);
          const int rank = input_shape.dim_size();
          if (branches == 0) {
            if (rank < 2) {
              fail_shape_inference("Grouped streams must have rank at least 2.");
            }
            for (int i = 0; i < rank - 2; ++i) {
              *output_shape->add_dim() = input_shape.dim(i);
            }
            *output_shape->add_dim() = input_shape.dim(rank - 1);
          } else {
            if (branches < 1 || rank < 1) {
              fail_shape_inference("num_branches must be positive for flattened streams.");
            }
            for (int i = 0; i < rank - 1; ++i) {
              *output_shape->add_dim() = input_shape.dim(i);
            }
            auto* last = output_shape->add_dim();
            const auto& input_last = input_shape.dim(rank - 1);
            if (input_last.has_dim_value()) {
              if (input_last.dim_value() % branches != 0) {
                fail_shape_inference("The flattened stream width must be divisible by num_branches.");
              }
              last->set_dim_value(input_last.dim_value() / branches);
            }
          }
        }));

constexpr const char* HyperConnectionPostMix_ver1_doc = R"DOC(
Mixes existing streams and injects one branch output:
Y[..., k, h] = sum_c(stream_mix[..., c, k] * streams[..., c, h])
               + post_mix[..., k, h] * block_output[..., h].
stream_mix is optional and defaults to the identity. post_mix has shape
(..., C), (..., C, 1), or (..., C, H). No activation is applied.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    HyperConnectionPostMix, 1,
    OpSchema()
        .SetDoc(HyperConnectionPostMix_ver1_doc)
        .Attr("num_branches",
              "Number of branches for flattened streams. Omit or set to zero for grouped streams.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "streams", "Grouped (..., C, H) or flattened (..., C * H) streams.", "T")
        .Input(1, "block_output", "Feature tensor with shape (..., H).", "T")
        .Input(2, "post_mix", "Branch or feature injection gates.", "M")
        .Input(3, "stream_mix", "Optional stream matrix with shape (..., C, C).",
               "M", OpSchema::Optional)
        .Output(0, "output", "Mixed streams with the same shape as streams.", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain streams, block output, and output to floating-point tensors.")
        .TypeConstraint("M",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain mixing weights to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* HyperConnectionLowRankMix_ver1_doc = R"DOC(
One hyper-connection block with a low-rank gate network, fused into one operator. C = num_branches streams of
width H are kept as one flattened (..., C * H) tensor. With the optional prologue the operator first applies the
post-mix of the previous block (identity stream mix):

  S[..., c, h] = streams[..., c, h] + prev_post_gate[..., c] * prev_block_output[..., h]

and otherwise S = streams. It then computes, with `rms` taken per branch over H:

  hn[..., c, h] = S[..., c, h] / sqrt(mean_h(S[..., c, :]^2) + epsilon) * norm_scale[c, h]
  d             = ScaledSiLU(hn @ down_weight, silu_alpha)                   (..., R)
  g[..., c, h]  = sigmoid(d @ up_weight)                                    per stream and feature
  pre_output    = reduction_scale * sum_c(hn[..., c, h] * g[..., c, h])     (..., H)
  post_gate     = sigmoid((hn @ inject_weight) / gate_div) * gate_mul         (..., C)

pre_output is the input of the next transformer block and post_gate is prev_post_gate of the next call.
streams_out is S (the streams after the prologue; a copy of streams when there is no prologue).

Numerics contract. Matrix products and reductions accumulate in float32. The result of every step is rounded
to T once, as the output of the equivalent standard operator would be:
S = cast_T(streams + prev_post_gate * prev_block_output), hn = cast_T(S * inv_rms * norm_scale),
down = cast_T(hn @ down_weight), Z = cast_T(silu_alpha * down), SG = cast_T(sigmoid(Z)), hidden = cast_T(Z * SG),
up = cast_T(hidden @ up_weight), g = cast_T(sigmoid(up)), pre_output = cast_T(reduction_scale * sum_c(hn * g)),
inject = cast_T(hn @ inject_weight), post_gate = cast_T(cast_T(sigmoid(cast_T(inject / gate_div))) * gate_mul).
sigmoid is evaluated in float32 with expf and rounded once to T; it does not reproduce a Sigmoid kernel that
rounds to T after every elementary operation. The accumulation order of the matrix products is fixed, so the
results are deterministic, but they are not bitwise identical to cuBLAS.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    HyperConnectionLowRankMix, 1,
    OpSchema()
        .SetDoc(HyperConnectionLowRankMix_ver1_doc)
        .Attr("num_branches", "Number of residual streams C. The last dimension of streams is C * H.",
              AttributeProto::INT, true)
        .Attr("epsilon", "Epsilon added to the mean square of each branch before the reciprocal square root.",
              AttributeProto::FLOAT, 1e-5f)
        .Attr("reduction_scale", "Multiplier applied to the sum over branches of pre_output.",
              AttributeProto::FLOAT, 1.0f)
        .Attr("silu_alpha", "Scale of the ScaledSiLU between the down and up projections.",
              AttributeProto::FLOAT, 1.0f)
        .Attr("gate_div", "Divisor applied to the inject logits before the sigmoid. Must not be zero.",
              AttributeProto::FLOAT, 1.0f)
        .Attr("gate_mul", "Multiplier applied to the sigmoid of the inject logits.",
              AttributeProto::FLOAT, 1.0f)
        .Input(0, "streams",
               "Flattened streams (..., C * H). With the prologue these are the streams before the previous "
               "block's post-mix.",
               "T")
        .Input(1, "norm_scale", "Per-branch RMS norm weight with shape (C * H).", "T")
        .Input(2, "down_weight", "Down projection with shape (C * H, R), row-major [K, N].", "T")
        .Input(3, "up_weight", "Up projection with shape (R, C * H), row-major [K, N].", "T")
        .Input(4, "inject_weight", "Post-gate projection with shape (C * H, C), row-major [K, N].", "T")
        .Input(5, "prev_block_output",
               "Optional output of the previous block with shape (..., H). Provide together with prev_post_gate.",
               "T", OpSchema::Optional)
        .Input(6, "prev_post_gate",
               "Optional post-mix gate of the previous block with shape (..., C). Provide together with "
               "prev_block_output.",
               "T", OpSchema::Optional)
        .Output(0, "pre_output", "Block input with shape (..., H).", "T")
        .Output(1, "post_gate", "Post-mix gate for the next call with shape (..., C).", "T")
        .Output(2, "streams_out",
                "Optional streams after the prologue, with the same shape as streams. Equal to streams when the "
                "prologue inputs are absent.",
                "T", OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain all floating-point inputs and outputs to the same type.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
          if (ctx.getNumOutputs() > 2) {
            propagateElemTypeFromInputToOutput(ctx, 0, 2);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          const auto branches = getAttribute(ctx, "num_branches", 0);
          if (branches < 1) {
            fail_shape_inference("num_branches must be positive.");
          }
          const auto& input_shape = getInputShape(ctx, 0);
          const int rank = input_shape.dim_size();
          if (rank < 1) {
            fail_shape_inference("streams must have rank at least 1.");
          }
          auto* pre_shape = getOutputShape(ctx, 0);
          auto* gate_shape = getOutputShape(ctx, 1);
          for (int i = 0; i < rank - 1; ++i) {
            *pre_shape->add_dim() = input_shape.dim(i);
            *gate_shape->add_dim() = input_shape.dim(i);
          }
          auto* hidden = pre_shape->add_dim();
          const auto& last = input_shape.dim(rank - 1);
          if (last.has_dim_value()) {
            if (last.dim_value() % branches != 0) {
              fail_shape_inference("The last dimension of streams must be divisible by num_branches.");
            }
            hidden->set_dim_value(last.dim_value() / branches);
          }
          gate_shape->add_dim()->set_dim_value(branches);
          if (ctx.getNumOutputs() > 2) {
            propagateShapeFromInputToOutput(ctx, 0, 2);
          }
        }));

constexpr const char* GatedAdd_ver1_doc = R"DOC(
Adds one tensor to another tensor scaled by a per-row gate:

  output = X + round_to_T(Y * gate)

X and Y have shape (..., C), and gate has shape (..., 1). The gate is broadcast
over C. For reduced-precision types, the product is rounded to T before the add,
matching separate ONNX Mul and Add operators.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedAdd, 1,
    OpSchema()
        .SetDoc(GatedAdd_ver1_doc)
        .Input(0, "X", "Unscaled input with shape (..., C).", "T")
        .Input(1, "Y", "Input scaled by gate, with the same shape as X.", "T")
        .Input(2, "gate", "Per-row gate with shape (..., 1).", "T")
        .Output(0, "output", "Gated sum with the same shape as X.", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

}  // namespace contrib
}  // namespace onnxruntime
