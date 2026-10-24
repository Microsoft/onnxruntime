// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Everything the device code needs to know about a PackedSparseAttentionIndexer call. All of it is
// derived from attributes and input *shapes* (never input values), so no device data is ever read
// on the host; per-request quantities (past_sequence_lengths, past_state_lengths, cumulative
// offsets) are read directly by the kernels below from device memory.
struct PackedSparseAttentionIndexerParams {
  int batch_size = 0;
  int total_tokens = 0;
  int num_heads = 0;
  int head_size = 0;
  int query_row_stride = 0;
  int key_row_stride = 0;
  int rotary_width = 0;            // cos_cache.shape[-1]
  int max_rotary_length = 0;       // cos_cache.shape[-2]
  bool cos_cache_batched = false;  // cos_cache rank: 3 = [batch, pos, rot], 2 = [pos, rot]
  bool has_position_ids = false;
  int compress_ratio = 0;
  int state_capacity = 0;   // past_key_state.shape[1]
  int buffer_capacity = 0;  // past_kv_buffer.shape[1] == 2 * compress_ratio - 1
  int capacity = 0;         // selected_indices.shape[1]
  int output_capacity = 0;
  bool state_only = false;
  float epsilon = 1e-6f;
  float scale = 0.0f;

  // policy_mode = "qsa"
  int block_topk = 0;  // token_budget / compress_ratio
  int state_update_capacity = 0;
  void* mid_kv_buffer = nullptr;          // [batch, buffer_capacity, head_size] T
  int32_t* mid_state_lengths = nullptr;   // [batch, 2]
  void* mid_kv_buffer2 = nullptr;         // [batch, buffer_capacity, head_size] T
  int32_t* mid_state_lengths2 = nullptr;  // [batch, 2]
  void* mid_kv_buffer3 = nullptr;         // [batch, buffer_capacity, head_size] T
  int32_t* mid_state_lengths3 = nullptr;  // [batch, 2]
  int32_t* validation_status = nullptr;   // [batch]

  // policy_mode = "csa"
  int index_topk = 0;
  float head_weight_scale = 0.0f;
};

// Scratch requirements, in float elements.
size_t GetQsaPackedWorkspaceFloatCount(const PackedSparseAttentionIndexerParams& params);
size_t GetCsaPackedWorkspaceFloatCount(const PackedSparseAttentionIndexerParams& params);
Status LaunchValidatePackedIndexerReuse(cudaStream_t stream, const int32_t* cumulative_lengths,
                                        const int32_t* past_sequence_lengths, const int32_t* past_state_lengths,
                                        int32_t* indices, int32_t* counts, int32_t* status,
                                        int batch_size, int capacity, int state_capacity, int compress_ratio,
                                        bool is_qsa, int32_t* overflow_flags = nullptr);
// `overflow_flags` is a caller-allocated int32 buffer with at least `batch_size` elements
// (unused when batch_size == 0); it may be the operator's state_status output. The update kernel
// writes, per request, a psai::StateStatusCode: 1 for inconsistent device metadata, 2 when this
// call's new blocks/windows would exceed state_capacity, else 0. When non-zero, the whole step is rejected for that
// request (present_key_state / present_kv_buffer / present_gate_buffer for that request are left
// unchanged and present_state_lengths is left as past_state_lengths clamped to valid bounds) and
// the select kernels force that request's selected_indices/selected_counts to the safe empty
// result (-1 / 0) rather than selecting against a partially updated state.
template <typename T, typename TState>
Status LaunchQsaPackedSparseAttentionIndexer(
    cudaStream_t stream,
    const PackedSparseAttentionIndexerParams& params,
    const T* query,
    const T* key,
    const T* query_norm_weight,
    const T* key_norm_weight,
    const T* cos_cache,
    const T* sin_cache,
    const int32_t* cumulative_sequence_lengths,
    const int32_t* past_sequence_lengths,
    const int64_t* position_ids,
    const TState* past_key_state,
    const T* past_kv_buffer,
    const int32_t* past_state_lengths,
    const int32_t* state_update_capture_count,
    bool state_update_active,
    int32_t* selected_indices,
    int32_t* selected_counts,
    TState* present_key_state,
    T* present_kv_buffer,
    int32_t* present_state_lengths,
    T* state_update,
    float* float_workspace,
    int32_t* overflow_flags);

template <typename T>
Status LaunchCsaPackedSparseAttentionIndexer(
    cudaStream_t stream,
    const PackedSparseAttentionIndexerParams& params,
    const T* query,
    const T* key,
    const T* query_norm_weight,
    const T* key_norm_weight,
    const T* cos_cache,
    const T* sin_cache,
    const T* gate,
    const T* position_bias,
    const T* head_weights,
    const int32_t* cumulative_sequence_lengths,
    const int32_t* past_sequence_lengths,
    const int64_t* position_ids,
    const T* past_key_state,
    const T* past_kv_buffer,
    const T* past_gate_buffer,
    const int32_t* past_state_lengths,
    int32_t* selected_indices,
    int32_t* selected_counts,
    T* present_key_state,
    T* present_kv_buffer,
    T* present_gate_buffer,
    int32_t* present_state_lengths,
    float* float_workspace,
    int32_t* overflow_flags);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
