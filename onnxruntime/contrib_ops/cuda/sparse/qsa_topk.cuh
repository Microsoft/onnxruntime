// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_merge_sort.cuh>
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime::contrib::cuda::qsa_topk {

constexpr int kChunkSize = 2048;
constexpr int kTopK = 512;
constexpr int kThreads = 512;

namespace topk = onnxruntime::cuda::topk;

template <int ItemsPerThread>
using ChunkSort = cub::BlockMergeSort<uint64_t, kThreads, ItemsPerThread>;

template <int ItemsPerThread>
__device__ __forceinline__ void SortChunk(const uint64_t* input, uint64_t* output, int input_count,
                                          typename ChunkSort<ItemsPerThread>::TempStorage& temp) {
  uint64_t keys[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x * ItemsPerThread + item;
    keys[item] = index < input_count ? input[index] : topk::kPaddingSortKey;
  }
  ChunkSort<ItemsPerThread>(temp).Sort(keys, topk::Greater<uint64_t>());
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int rank = threadIdx.x * ItemsPerThread + item;
    if (rank < kTopK) output[rank] = keys[item];
  }
}

__global__ void QsaCompactTileTopKKernel(const uint64_t* input_keys, uint64_t* output_keys,
                                         int input_stride, int output_stride) {
  __shared__ union {
    ChunkSort<1>::TempStorage s1;
    ChunkSort<2>::TempStorage s2;
    ChunkSort<4>::TempStorage s4;
  } temp;
  const int chunk_start = static_cast<int>(blockIdx.x) * kChunkSize;
  const uint64_t* input = input_keys + static_cast<int64_t>(blockIdx.y) * input_stride + chunk_start;
  uint64_t* output = output_keys + static_cast<int64_t>(blockIdx.y) * output_stride + blockIdx.x * kTopK;
  const int count = min(kChunkSize, input_stride - chunk_start);

  // Scoring emits sorted 32-key tiles in a live prefix followed by zero padding.
  // Tile-aligned probes adapt to device context growth without graph recapture.
  if (input[0] == topk::kPaddingSortKey) {
    output[threadIdx.x] = topk::kPaddingSortKey;
  } else if (count > 1024 && input[1024] != topk::kPaddingSortKey) {
    SortChunk<4>(input, output, count, temp.s4);
  } else if (count > 512 && input[512] != topk::kPaddingSortKey) {
    SortChunk<2>(input, output, min(count, 1024), temp.s2);
  } else {
    SortChunk<1>(input, output, min(count, 512), temp.s1);
  }
}

}  // namespace onnxruntime::contrib::cuda::qsa_topk
