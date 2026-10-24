// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_parameters.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum DynamicSparseAttentionValidationError : int32_t {
  kDynamicSparseAttentionValidationOk = 0,
  kDynamicSparseAttentionInvalidCount = 1,
  kDynamicSparseAttentionInvalidPadding = 2,
  kDynamicSparseAttentionNegativeIndex = 3,
  kDynamicSparseAttentionIndexOutOfBounds = 4,
  kDynamicSparseAttentionDuplicateIndex = 5,
  kDynamicSparseAttentionNonCausalIndex = 6,
  kDynamicSparseAttentionInvalidSequenceLength = 7,
  kDynamicSparseAttentionInvalidPosition = 8,
  kDynamicSparseAttentionInvalidCacheScale = 10,
};

template <typename T>
struct DynamicSparseAttentionData {
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const T* auxiliary_key = nullptr;
  const T* auxiliary_value = nullptr;
  const int32_t* selected_indices = nullptr;
  const int32_t* selected_counts = nullptr;
  const int32_t* seqlens_k = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  const int64_t* position_ids = nullptr;
  const T* q_norm_weight = nullptr;
  const T* k_norm_weight = nullptr;
  const T* head_sink = nullptr;
  T* prepared_query = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
  T* output = nullptr;
  float* attention_workspace = nullptr;
  bool quantized_cache = false;
  const float* key_scale = nullptr;
  const float* value_scale = nullptr;
};

size_t GetDynamicSparseAttentionValidationWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t max_shared_memory_per_block);

size_t GetDynamicSparseAttentionWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t element_size,
    size_t max_shared_memory_per_block,
    bool quantized_cache = false);

Status ValidateDynamicSparseAttentionOnDevice(
    cudaStream_t stream,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    const DynamicSparseAttentionParameters& parameters,
    int32_t* error_flag,
    uint32_t* validation_bitmap,
    size_t max_shared_memory_per_block,
    bool copy_result_to_host,
    const float* key_scale = nullptr,
    const float* value_scale = nullptr,
    const int32_t* committed_length = nullptr);

template <typename T>
Status LaunchDynamicSparseAttention(
    cudaStream_t stream,
    const DynamicSparseAttentionParameters& parameters,
    const DynamicSparseAttentionData<T>& data,
    bool initialize_key_cache,
    bool initialize_value_cache,
    int max_threads_per_block,
    size_t max_shared_memory_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
