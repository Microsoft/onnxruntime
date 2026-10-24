
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/moe/qmoe_fused_route.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <type_traits>

#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "core/common/narrow.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cu_inc/cub.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

int Compute1DGridSize(int num_elements, int block_size) {
  return (num_elements + block_size - 1) / block_size;
}

constexpr float kTopKNormalizeEpsilon = onnxruntime::cuda::topk::kTopKNormalizeEpsilon;

template <typename T>
__global__ void QMoEOffloadGatherKernel(
    const T* input, const int* route_ids, T* gathered, int hidden_size, int k) {
  const size_t source_row = static_cast<size_t>(route_ids[blockIdx.x] / k);
  const size_t target_row = blockIdx.x;
  for (size_t column = threadIdx.x; column < static_cast<size_t>(hidden_size); column += blockDim.x) {
    gathered[target_row * hidden_size + column] = input[source_row * hidden_size + column];
  }
}

template <typename T>
__global__ void QMoEOffloadScatterKernel(
    const T* wave_output, const int* route_ids, T* contributions,
    int hidden_size, int tile_rows, int k) {
  const int route_id = route_ids[blockIdx.x];
  const size_t target_row = static_cast<size_t>(route_id % k) * tile_rows + route_id / k;
  const size_t source_row = blockIdx.x;
  for (size_t column = threadIdx.x; column < static_cast<size_t>(hidden_size); column += blockDim.x) {
    contributions[target_row * hidden_size + column] = wave_output[source_row * hidden_size + column];
  }
}

__global__ void QMoEOffloadMetadataKernel(float* unit_scales, int* identity_permutation, int route_count) {
  const size_t route = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (route < static_cast<size_t>(route_count)) {
    unit_scales[route] = 1.0f;
    identity_permutation[route] = static_cast<int>(route);
  }
}

__global__ void QMoEOffloadSourceRowsKernel(int* source_rows, int route_count, int k) {
  const int route = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (route < route_count) {
    source_rows[route] = route / k;
  }
}

template <typename T>
__global__ void QMoEOffloadScatterBiasKernel(
    const T* wave_bias, const int* compact_experts, const int* route_ids, T* route_bias, int hidden_size) {
  const size_t source_row = compact_experts[blockIdx.x];
  const size_t target_row = route_ids[blockIdx.x];
  for (size_t column = threadIdx.x; column < static_cast<size_t>(hidden_size); column += blockDim.x) {
    route_bias[target_row * hidden_size + column] = wave_bias[source_row * hidden_size + column];
  }
}

cudaError_t LaunchQMoEOffloadScatterBias(
    const half* wave_bias, const int* compact_experts, const int* route_ids, half* route_bias,
    int route_count, int hidden_size, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadScatterBiasKernel<<<route_count, 256, 0, stream>>>(
      wave_bias, compact_experts, route_ids, route_bias, hidden_size);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadScatterBias(
    const __nv_bfloat16* wave_bias, const int* compact_experts, const int* route_ids, __nv_bfloat16* route_bias,
    int route_count, int hidden_size, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadScatterBiasKernel<<<route_count, 256, 0, stream>>>(
      wave_bias, compact_experts, route_ids, route_bias, hidden_size);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadGather(
    const half* input, const int* route_ids, half* gathered,
    int route_count, int hidden_size, int k, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadGatherKernel<<<route_count, 256, 0, stream>>>(input, route_ids, gathered, hidden_size, k);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadGather(
    const __nv_bfloat16* input, const int* route_ids, __nv_bfloat16* gathered,
    int route_count, int hidden_size, int k, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadGatherKernel<<<route_count, 256, 0, stream>>>(input, route_ids, gathered, hidden_size, k);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadScatter(
    const half* wave_output, const int* route_ids, half* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadScatterKernel<<<route_count, 256, 0, stream>>>(
      wave_output, route_ids, contributions, hidden_size, tile_rows, k);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadScatter(
    const __nv_bfloat16* wave_output, const int* route_ids, __nv_bfloat16* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadScatterKernel<<<route_count, 256, 0, stream>>>(
      wave_output, route_ids, contributions, hidden_size, tile_rows, k);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadMetadata(
    float* unit_scales, int* identity_permutation, int route_count, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  const int blocks = route_count / 256 + (route_count % 256 != 0);
  QMoEOffloadMetadataKernel<<<blocks, 256, 0, stream>>>(
      unit_scales, identity_permutation, route_count);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadSourceRows(
    int* source_rows, int route_count, int k, cudaStream_t stream) {
  if (route_count <= 0) return cudaSuccess;
  const int blocks = route_count / 256 + (route_count % 256 != 0);
  QMoEOffloadSourceRowsKernel<<<blocks, 256, 0, stream>>>(source_rows, route_count, k);
  return cudaGetLastError();
}

__global__ void QMoEOffloadClassifyKernel(
    const int* topk_indices, const int* expert_slot, int route_count, int row_count,
    int* hit_ids, int* hit_compact, int* hit_flag, int dummy_slot,
    float* unit_scales, int* identity_permutation, int k_for_source_rows,
    int* host_indices_out, const uint4* input_dev, uint4* host_input_out, int input_vec4_count,
    uint64_t* doorbell_seq_dev, volatile uint64_t* doorbell_req_host,
    volatile int32_t* doorbell_tile_rows_host, int tile_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < row_count) {
    if (unit_scales != nullptr) unit_scales[row] = 1.0f;
    if (identity_permutation != nullptr) identity_permutation[row] = row;
    const int route = row < route_count ? row : 0;
    const int exp = topk_indices[route];
    const int slot = (row < route_count) ? expert_slot[exp] : -1;
    hit_ids[row] = (k_for_source_rows > 0 && row < route_count) ? (route / k_for_source_rows) : route;
    hit_compact[row] = slot < 0 ? dummy_slot : slot;
    if (row < route_count) {
      hit_flag[row] = slot >= 0;
      if (host_indices_out != nullptr) {
        host_indices_out[row] = exp;
      }
    }
  }
  if (input_dev != nullptr && host_input_out != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_vec4_count; i += gridDim.x * blockDim.x) {
      host_input_out[i] = input_dev[i];
    }
  }
  if (doorbell_seq_dev != nullptr && doorbell_req_host != nullptr && gridDim.x == 1) {
    __syncthreads();
    if (threadIdx.x == 0) {
      if (doorbell_tile_rows_host != nullptr) {
        *doorbell_tile_rows_host = tile_rows;
      }
      __threadfence_system();
      const uint64_t next_seq = *doorbell_seq_dev + 1;
      *doorbell_seq_dev = next_seq;
      *doorbell_req_host = next_seq;
      __threadfence_system();
    }
  }
}

template <typename T>
__global__ void QMoEOffloadMergeKernel(
    const T* wave_output, const T* cpu_rows, const int* hit_flag, T* contributions,
    int hidden_size, int tile_rows, int k,
    const uint64_t* doorbell_seq_dev, const volatile uint64_t* doorbell_done_host) {
  const int route = blockIdx.x;
  const size_t target_row = static_cast<size_t>(route % k) * tile_rows + route / k;
  const int is_hit = hit_flag[route];
  T* dst = contributions + target_row * hidden_size;
  if (is_hit) {
    const T* source = wave_output + static_cast<size_t>(route) * hidden_size;
    if ((hidden_size & 7) == 0) {
      const uint4* src4 = reinterpret_cast<const uint4*>(source);
      uint4* dst4 = reinterpret_cast<uint4*>(dst);
      const size_t vec_count = static_cast<size_t>(hidden_size) >> 3;
      for (size_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
        dst4[i] = src4[i];
      }
    } else {
      for (size_t column = threadIdx.x; column < static_cast<size_t>(hidden_size); column += blockDim.x) {
        dst[column] = source[column];
      }
    }
    if (route == 0 && doorbell_seq_dev != nullptr && doorbell_done_host != nullptr && threadIdx.x == 0) {
      const uint64_t expected = *doorbell_seq_dev;
      while (*doorbell_done_host < expected) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
      }
      __threadfence_system();
    }
  } else {
    if (doorbell_seq_dev != nullptr && doorbell_done_host != nullptr) {
      if (threadIdx.x == 0) {
        const uint64_t expected = *doorbell_seq_dev;
        while (*doorbell_done_host < expected) {
#if __CUDA_ARCH__ >= 700
          __nanosleep(100);
#endif
        }
        __threadfence_system();
      }
      __syncthreads();
    }
    const T* source = cpu_rows + target_row * hidden_size;
    if ((hidden_size & 7) == 0) {
      const uint4* src4 = reinterpret_cast<const uint4*>(source);
      uint4* dst4 = reinterpret_cast<uint4*>(dst);
      const size_t vec_count = static_cast<size_t>(hidden_size) >> 3;
      for (size_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
        dst4[i] = src4[i];
      }
    } else {
      for (size_t column = threadIdx.x; column < static_cast<size_t>(hidden_size); column += blockDim.x) {
        dst[column] = source[column];
      }
    }
  }
}

cudaError_t LaunchQMoEOffloadClassify(
    const int* topk_indices, const int* expert_slot, int route_count, int row_count,
    int* hit_ids, int* hit_compact, int* hit_flag, cudaStream_t stream, int dummy_slot,
    float* unit_scales, int* identity_permutation, int k_for_source_rows,
    int* host_indices_out, const void* input_dev, void* host_input_out, int input_vec4_count,
    uint64_t* doorbell_seq_dev, volatile uint64_t* doorbell_req_host,
    volatile int32_t* doorbell_tile_rows_host, int tile_rows) {
  if (row_count <= 0) return cudaSuccess;
  if (doorbell_seq_dev != nullptr && doorbell_req_host != nullptr && row_count <= 256) {
    QMoEOffloadClassifyKernel<<<1, 256, 0, stream>>>(
        topk_indices, expert_slot, route_count, row_count, hit_ids, hit_compact, hit_flag, dummy_slot,
        unit_scales, identity_permutation, k_for_source_rows,
        host_indices_out, reinterpret_cast<const uint4*>(input_dev),
        reinterpret_cast<uint4*>(host_input_out), input_vec4_count,
        doorbell_seq_dev, doorbell_req_host, doorbell_tile_rows_host, tile_rows);
    return cudaGetLastError();
  }
  const int blocks = row_count / 128 + (row_count % 128 != 0);
  QMoEOffloadClassifyKernel<<<blocks, 128, 0, stream>>>(
      topk_indices, expert_slot, route_count, row_count, hit_ids, hit_compact, hit_flag, dummy_slot,
      unit_scales, identity_permutation, k_for_source_rows,
      host_indices_out, reinterpret_cast<const uint4*>(input_dev),
      reinterpret_cast<uint4*>(host_input_out), input_vec4_count,
      nullptr, nullptr, nullptr, tile_rows);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadMerge(
    const half* wave_output, const half* cpu_rows, const int* hit_flag, half* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev, const volatile uint64_t* doorbell_done_host) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadMergeKernel<<<route_count, 256, 0, stream>>>(
      wave_output, cpu_rows, hit_flag, contributions, hidden_size, tile_rows, k,
      doorbell_seq_dev, doorbell_done_host);
  return cudaGetLastError();
}

cudaError_t LaunchQMoEOffloadMerge(
    const __nv_bfloat16* wave_output, const __nv_bfloat16* cpu_rows, const int* hit_flag, __nv_bfloat16* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev, const volatile uint64_t* doorbell_done_host) {
  if (route_count <= 0) return cudaSuccess;
  QMoEOffloadMergeKernel<<<route_count, 256, 0, stream>>>(
      wave_output, cpu_rows, hit_flag, contributions, hidden_size, tile_rows, k,
      doorbell_seq_dev, doorbell_done_host);
  return cudaGetLastError();
}

__global__ void QMoEOffloadMergeFinalizeCombineKernel(
    const __nv_bfloat16* wave_output, const __nv_bfloat16* cpu_rows, const int* hit_flag,
    const float* scales, __nv_bfloat16* routed_out, __nv_bfloat16* shared_out,
    int hidden_size, int tile_rows, int k,
    const uint64_t* doorbell_seq_dev, const volatile uint64_t* doorbell_done_host, int trigger) {
  const int original_row = blockIdx.x;
  if (original_row >= tile_rows) return;
  __shared__ int s_hit[16];
  __shared__ float s_scale[16];
  if (threadIdx.x < k && threadIdx.x < 16) {
    const int route = original_row * k + threadIdx.x;
    s_hit[threadIdx.x] = hit_flag[route];
    s_scale[threadIdx.x] = scales[route];
  }
  __syncthreads();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (trigger == 1) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
  const int num_vec4 = hidden_size >> 3;
  const uint4* wave_v = reinterpret_cast<const uint4*>(wave_output);
  const uint4* cpu_v = reinterpret_cast<const uint4*>(cpu_rows);
  uint4* routed_v = reinterpret_cast<uint4*>(routed_out);
  uint4* shared_v = reinterpret_cast<uint4*>(shared_out);

  uint4 preloaded_wave[16];
  uint4 preloaded_shared = make_uint4(0, 0, 0, 0);
  const bool single_pass = (num_vec4 <= static_cast<int>(blockDim.x));
  if (single_pass && static_cast<int>(threadIdx.x) < num_vec4) {
    if (shared_v != nullptr) {
      preloaded_shared = shared_v[static_cast<size_t>(original_row) * num_vec4 + threadIdx.x];
    }
    for (int k_idx = 0; k_idx < k && k_idx < 16; ++k_idx) {
      if (s_hit[k_idx]) {
        const int route = original_row * k + k_idx;
        preloaded_wave[k_idx] = wave_v[static_cast<size_t>(route) * num_vec4 + threadIdx.x];
      }
    }
  }
  if (threadIdx.x == 0 && doorbell_seq_dev != nullptr && doorbell_done_host != nullptr) {
    const uint64_t expected = *doorbell_seq_dev;
    while (*doorbell_done_host < expected) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(100);
#endif
    }
    __threadfence_system();
  }
  __syncthreads();

  for (int elem_idx = threadIdx.x; elem_idx < num_vec4; elem_idx += blockDim.x) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int route = original_row * k + k_idx;
      const size_t target_row = static_cast<size_t>(k_idx) * tile_rows + original_row;
      const uint4 raw = s_hit[k_idx]
                            ? (single_pass ? preloaded_wave[k_idx] : wave_v[static_cast<size_t>(route) * num_vec4 + elem_idx])
                            : cpu_v[target_row * num_vec4 + elem_idx];
      const float scale = s_scale[k_idx];
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f2 = __bfloat1622float2(b2[i]);
        acc[2 * i + 0] = acc[2 * i + 0] + scale * f2.x;
        acc[2 * i + 1] = acc[2 * i + 1] + scale * f2.y;
      }
    }
    uint4 routed_u4;
    __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(&routed_u4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r2[i] = __floats2bfloat162_rn(acc[2 * i + 0], acc[2 * i + 1]);
    }
    if (shared_v != nullptr) {
      const uint4 shared_u4 = single_pass
                                  ? preloaded_shared
                                  : shared_v[static_cast<size_t>(original_row) * num_vec4 + elem_idx];
      const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&shared_u4);
      uint4 comb_u4;
      __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&comb_u4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 rf = __bfloat1622float2(r2[i]);
        const float2 sf = __bfloat1622float2(s2[i]);
        c2[i] = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
      }
      shared_v[static_cast<size_t>(original_row) * num_vec4 + elem_idx] = comb_u4;
    } else if (routed_v != nullptr) {
      routed_v[static_cast<size_t>(original_row) * num_vec4 + elem_idx] = routed_u4;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (trigger == 2 || trigger == 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

__global__ void QMoESharedExpertCombineKernel(
    __nv_bfloat16* shared_out, const __nv_bfloat16* routed_out, int vec4_count, int trigger) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (trigger == 1) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
  uint4* shared_v = reinterpret_cast<uint4*>(shared_out);
  const uint4* routed_v = reinterpret_cast<const uint4*>(routed_out);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec4_count; idx += gridDim.x * blockDim.x) {
    const uint4 r4 = routed_v[idx];
    const uint4 s4 = shared_v[idx];
    uint4 d4;
    const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&r4);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&s4);
    __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(&d4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 rf = __bfloat1622float2(r2[i]);
      const float2 sf = __bfloat1622float2(s2[i]);
      d2[i] = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
    }
    shared_v[idx] = d4;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (trigger == 2) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

cudaError_t LaunchQMoEOffloadMergeFinalizeCombine(
    const __nv_bfloat16* wave_output, const __nv_bfloat16* cpu_rows, const int* hit_flag,
    const float* scales, __nv_bfloat16* routed_out, __nv_bfloat16* shared_out,
    int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev, const volatile uint64_t* doorbell_done_host,
    int trigger) {
  if (tile_rows <= 0 || hidden_size <= 0) return cudaSuccess;
  const int num_vec4 = hidden_size >> 3;
  const int threads = (num_vec4 == 320) ? 320 : 256;
  QMoEOffloadMergeFinalizeCombineKernel<<<tile_rows, threads, 0, stream>>>(
      wave_output, cpu_rows, hit_flag, scales, routed_out, shared_out,
      hidden_size, tile_rows, k, doorbell_seq_dev, doorbell_done_host, trigger);
  return cudaGetLastError();
}

cudaError_t LaunchQMoESharedExpertCombine(
    __nv_bfloat16* shared_out, const __nv_bfloat16* routed_out,
    int count, int trigger, cudaStream_t stream) {
  if (count <= 0) return cudaSuccess;
  const int vec4_count = count >> 3;
  const int blocks = std::min(16, (vec4_count + 255) / 256);
  QMoESharedExpertCombineKernel<<<blocks, 256, 0, stream>>>(shared_out, routed_out, vec4_count, trigger);
  return cudaGetLastError();
}

// Shared with the fused routing prologue in the LLM MoE runner (moe_kernels.cu); see
// core/providers/cuda/cu_inc/topk_warp_sort.cuh for why these live in a header.
using onnxruntime::cuda::topk::SafeInvSum;
using onnxruntime::cuda::topk::SoftmaxScale;
using onnxruntime::cuda::topk::TopKNormalizeDenom;
using onnxruntime::cuda::topk::WarpReduceMax;
using onnxruntime::cuda::topk::WarpReduceSum;

template <typename BlockReduce>
__device__ __forceinline__ float BlockReduceMax(float value, typename BlockReduce::TempStorage& temp_storage) {
#if CUDART_VERSION >= 12090
  return BlockReduce(temp_storage).Reduce(value, ::cuda::maximum());
#else
  return BlockReduce(temp_storage).Reduce(value, cub::Max());
#endif
}

template <typename BlockReduce>
__device__ __forceinline__ float BlockReduceSum(float value, typename BlockReduce::TempStorage& temp_storage) {
#if CUDART_VERSION >= 12090
  return BlockReduce(temp_storage).Reduce(value, ::cuda::std::plus());
#else
  return BlockReduce(temp_storage).Reduce(value, cub::Sum());
#endif
}

template <typename T>
__global__ void SoftmaxTopKKernel(const T* logits, float* topk_scales, int* topk_indices,
                                  int num_rows, int num_experts, int k, bool normalize_scales) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  const T* row_logits = logits + row * num_experts;
  float* row_scales = topk_scales + row * k;
  int* row_indices = topk_indices + row * k;

  // 1. Find max for numerical stability
  float max_val = onnxruntime::cuda::topk::kNegativeInfinity;
  for (int i = 0; i < num_experts; ++i) {
    float val = static_cast<float>(row_logits[i]);
    if (val > max_val) max_val = val;
  }

  // 2. Compute exp sum
  float sum_exp = 0.0f;
  for (int i = 0; i < num_experts; ++i) {
    sum_exp += expf(static_cast<float>(row_logits[i]) - max_val);
  }
  const float inv_sum = SafeInvSum(sum_exp);

  // 3. Compute Softmax and find TopK
  // For small k, we can do a simple selection.
  // Note: This is efficient only for small k and small num_experts.

  // We can compute softmax values on the fly or store them.
  // Given we need topK, let's just compute all softmax values then pick top K.
  // (Optimization: use a heap or similar if K is small and N is large)

  for (int i = 0; i < k; ++i) {
    row_scales[i] = -FLT_MAX;
    row_indices[i] = -1;
  }

  for (int i = 0; i < num_experts; ++i) {
    float prob = SoftmaxScale(static_cast<float>(row_logits[i]), max_val, inv_sum);

    // Insert into top-k logic
    // Simple insertion sort for very small k (e.g. k=2)
    for (int j = 0; j < k; ++j) {
      if (prob > row_scales[j]) {
        // Shift current values down
        for (int m = k - 1; m > j; --m) {
          row_scales[m] = row_scales[m - 1];
          row_indices[m] = row_indices[m - 1];
        }
        row_scales[j] = prob;
        row_indices[j] = i;
        break;
      }
    }
  }

  // 4. Normalize if requested
  if (normalize_scales) {
    float scale_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      scale_sum += row_scales[i];
    }
    if (scale_sum > kTopKNormalizeEpsilon) {
      for (int i = 0; i < k; ++i) {
        row_scales[i] /= scale_sum;
      }
    }
  }
}

// Block-per-row softmax + top-k using a CUB block sort. Each block sorts one
// row's logits (descending) and reads the first k. A full sort of 256 logits is
// ~2.5x faster than k rounds of block-argmax on this size (benchmarked), and is
// the layout onnxruntime-genai's top-k benchmarks also recommend (CUB block
// merge) for sort sizes up to ~1024. The capacity (kBlockSize*kItemsPerThread)
// must be >= num_experts; padding lanes carry (-inf, INT_MAX) so valid -inf
// expert scores sort ahead of padding. Tie-breaking matches the scalar kernel
// (lower expert index first) via the same packed stable sort key used by the
// warp merge path.

template <typename T, int kBlockSize, int kItemsPerThread>
__global__ void SoftmaxTopKMergeKernel(const T* logits, float* topk_scales, int* topk_indices,
                                       int num_rows, int num_experts, int k, bool normalize_scales) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;
  const int tid = threadIdx.x;

  using BlockMergeSort = cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>;
  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ union {
    typename BlockMergeSort::TempStorage merge;
    typename BlockReduce::TempStorage reduce;
  } temp;
  __shared__ float s_topk[64];  // k <= 64
  __shared__ float s_max;
  __shared__ float s_sum;

  // Load this thread's packed (logit, expert index) keys in a blocked
  // arrangement: thread t owns indices [t*ipt, t*ipt+ipt).
  uint64_t keys[kItemsPerThread];
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int idx = tid * kItemsPerThread + j;
    const float logit = (idx < num_experts) ? static_cast<float>(row_logits[idx])
                                            : onnxruntime::cuda::topk::kNegativeInfinity;
    const int index = (idx < num_experts) ? idx : INT_MAX;
    keys[j] = onnxruntime::cuda::topk::PackStableSortKey(logit, index);
    local_max = fmaxf(local_max, logit);
  }

  // Softmax denominator over all experts (needed when normalize_scales is false;
  // when true it cancels in the top-k normalization but is still correct).
  const float block_max = BlockReduceMax<BlockReduce>(local_max, temp.reduce);
  if (tid == 0) s_max = block_max;
  // Single barrier: publishes s_max to all threads and also separates the two
  // BlockReduce uses that share temp.reduce.
  __syncthreads();
  const float max_val = s_max;

  float local_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int idx = tid * kItemsPerThread + j;
    if (idx < num_experts) {
      local_sum += expf(onnxruntime::cuda::topk::UnpackStableSortScore(keys[j]) - max_val);
    }
  }
  const float block_sum = BlockReduceSum<BlockReduce>(local_sum, temp.reduce);
  if (tid == 0) s_sum = block_sum;
  // Single barrier: publishes s_sum and separates temp.reduce from temp.merge.
  __syncthreads();
  const float inv_sum = SafeInvSum(s_sum);

  // Sort packed (logit, index) keys descending. Result stays in a blocked
  // layout, so sorted rank r lives in thread (r / ipt), item (r % ipt). Sort()
  // leaves the sorted keys in each thread's registers and temp.merge is not
  // reused afterwards, so no barrier is needed here; the shared s_topk writes
  // below are published by the barrier that follows them.
  BlockMergeSort(temp.merge).Sort(keys, onnxruntime::cuda::topk::Greater<uint64_t>());

#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int rank = tid * kItemsPerThread + j;
    if (rank < k) {
      const uint64_t key = keys[j];
      topk_indices[static_cast<size_t>(row) * k + rank] =
          onnxruntime::cuda::topk::UnpackStableSortIndex(key);
      s_topk[rank] = SoftmaxScale(onnxruntime::cuda::topk::UnpackStableSortScore(key), max_val, inv_sum);
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (normalize_scales) {
      float scale_sum = 0.0f;
      for (int i = 0; i < k; ++i) scale_sum += s_topk[i];
      const float denom = TopKNormalizeDenom(normalize_scales, scale_sum);
      for (int i = 0; i < k; ++i) topk_scales[static_cast<size_t>(row) * k + i] = s_topk[i] / denom;
    } else {
      for (int i = 0; i < k; ++i) topk_scales[static_cast<size_t>(row) * k + i] = s_topk[i];
    }
  }
}

// Warp-bitonic softmax + top-k for num_experts <= 32. Each warp handles one
// row, with lane `l` owning expert `l`. The whole softmax reduction and the
// sort are done with warp shuffles (no shared memory). This is the fastest path
// for tiny expert counts per the onnxruntime-genai top-k benchmark. Tie-breaking
// (equal scores prefer the lower expert index) matches SoftmaxTopKMergeKernel.
template <typename T, int kWarpsPerBlock>
__global__ void SoftmaxTopKWarpBitonicKernel(const T* logits, float* topk_scales, int* topk_indices,
                                             int num_rows, int num_experts, int k, bool normalize_scales) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.y;
  if (row >= num_rows) return;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;
  const float logit = (lane < num_experts) ? static_cast<float>(row_logits[lane])
                                           : onnxruntime::cuda::topk::kNegativeInfinity;

  const float max_val = WarpReduceMax(logit);

  // Warp-wide exp sum (softmax denominator over all experts).
  const float sum_exp = WarpReduceSum((lane < num_experts) ? expf(logit - max_val) : 0.0f);
  const float inv_sum = SafeInvSum(sum_exp);

  // Sort (logit, expert index) descending; sorting by logit is equivalent to
  // sorting by softmax probability since the mapping is monotonic.
  float score = logit;
  int index = (lane < num_experts) ? lane : INT_MAX;
  onnxruntime::cuda::topk::WarpBitonicSortDescending(score, index);

  // Lane r now holds the rank-r element. Compute the top-k probabilities.
  float prob = (lane < k) ? SoftmaxScale(score, max_val, inv_sum) : 0.0f;

  if (normalize_scales) {
    prob /= TopKNormalizeDenom(normalize_scales, WarpReduceSum(prob));
  }

  if (lane < k) {
    topk_scales[static_cast<size_t>(row) * k + lane] = prob;
    topk_indices[static_cast<size_t>(row) * k + lane] = index;
  }
}

// Warp-per-row softmax + top-k for expert counts in (64, kWarpSize * kItemsPerLane].
// Lane `l` owns experts l, l + 32, l + 64, ... (strided so the logit loads stay
// coalesced), and the whole row - softmax reduction and top-k selection alike - is
// resolved with warp shuffles in registers: no shared memory, no __syncthreads and
// no block-wide merge sort.
//
// This exists because the MoE decode shapes have very few rows, so the grid is a
// handful of blocks and the kernel is bound by its dependency chain rather than by
// throughput. SoftmaxTopKMergeKernel fully sorts all `num_experts` logits through
// cub::BlockMergeSort, whose per-round MergePath searches and barriers dominate that
// chain, while the top-k butterfly here costs a fixed ~5 shuffle rounds. Requires
// k <= kItemsPerLane; tie-breaking matches SoftmaxTopKMergeKernel via the same
// packed stable sort key.
template <typename T, int kItemsPerLane, int kRowsPerBlock>
__global__ void SoftmaxTopKWarpTopKKernel(const T* logits, float* topk_scales, int* topk_indices,
                                          int num_rows, int num_experts, int k, bool normalize_scales) {
  constexpr int kWarpSize = onnxruntime::cuda::topk::kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.y);
  if (row >= num_rows) return;

  const int lane = threadIdx.x;
  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;

  // Padding lanes carry (-inf, INT_MAX) so that valid -inf expert scores sort ahead.
  uint64_t keys[kItemsPerLane];
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
  for (int j = 0; j < kItemsPerLane; ++j) {
    const int idx = j * kWarpSize + lane;
    const float logit = (idx < num_experts) ? static_cast<float>(row_logits[idx])
                                            : onnxruntime::cuda::topk::kNegativeInfinity;
    keys[j] = onnxruntime::cuda::topk::PackStableSortKey(logit, (idx < num_experts) ? idx : INT_MAX);
    local_max = fmaxf(local_max, logit);
  }

  // Softmax denominator over all experts (needed when normalize_scales is false;
  // when true it cancels in the top-k normalization but is still correct).
  const float max_val = WarpReduceMax(local_max);
  float local_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kItemsPerLane; ++j) {
    if (j * kWarpSize + lane < num_experts) {
      local_sum += expf(onnxruntime::cuda::topk::UnpackStableSortScore(keys[j]) - max_val);
    }
  }
  const float inv_sum = SafeInvSum(WarpReduceSum(local_sum));

  // Every lane ends up holding the row's top kItemsPerLane keys, sorted descending.
  onnxruntime::cuda::topk::WarpBitonicTopN<kItemsPerLane>(keys);

  float probs[kItemsPerLane];
  float scale_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kItemsPerLane; ++j) {
    probs[j] = SoftmaxScale(onnxruntime::cuda::topk::UnpackStableSortScore(keys[j]), max_val, inv_sum);
    if (j < k) scale_sum += probs[j];
  }
  const float denom = TopKNormalizeDenom(normalize_scales, scale_sum);

  // The results are replicated across the warp, so lane j simply writes rank j.
#pragma unroll
  for (int j = 0; j < kItemsPerLane; ++j) {
    if (lane == j && j < k) {
      topk_scales[static_cast<size_t>(row) * k + j] = normalize_scales ? (probs[j] / denom) : probs[j];
      topk_indices[static_cast<size_t>(row) * k + j] = onnxruntime::cuda::topk::UnpackStableSortIndex(keys[j]);
    }
  }
}

// Warp CUB merge sort softmax + top-k for num_experts <= kBufferSize (64). One
// warp (32 threads) per block sorts a row's logits held in shared memory. This
// is the genai-recommended path for sort sizes in (32, 64]. Tie-breaking
// matches SoftmaxTopKMergeKernel via a packed stable sort key.
template <typename T, int kBufferSize>
__global__ void SoftmaxTopKWarpMergeKernel(const T* logits, float* topk_scales, int* topk_indices,
                                           int num_rows, int num_experts, int k, bool normalize_scales) {
  constexpr int kWarpSize = onnxruntime::cuda::topk::kWarpSize;
  using WarpMergeSorter = onnxruntime::cuda::topk::WarpMergeSorter<kBufferSize>;

  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int lane = threadIdx.x;

  __shared__ float s_scores[kBufferSize];
  __shared__ int s_indices[kBufferSize];
  __shared__ typename WarpMergeSorter::TempStorage temp_storage;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;

  // Load logits into shared memory and compute the warp-wide max.
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
  for (int i = lane; i < kBufferSize; i += kWarpSize) {
    const float v = (i < num_experts) ? static_cast<float>(row_logits[i])
                                      : onnxruntime::cuda::topk::kNegativeInfinity;
    s_scores[i] = v;
    s_indices[i] = (i < num_experts) ? i : INT_MAX;
    local_max = fmaxf(local_max, v);
  }
  const float max_val = WarpReduceMax(local_max);

  // Warp-wide exp sum over all experts.
  float local_sum = 0.0f;
  for (int i = lane; i < num_experts; i += kWarpSize) {
    local_sum += expf(s_scores[i] - max_val);
  }
  const float inv_sum = SafeInvSum(WarpReduceSum(local_sum));

  // Each lane reads back only the slots it wrote above, so the sort needs no barrier before it.
  WarpMergeSorter::Sort(s_scores, s_indices, temp_storage, num_experts);
  // Sort's blocked write-back must be visible to the strided reads below.
  __syncwarp();

  // s_scores[r]/s_indices[r] now hold the rank-r logit/expert index.
  float scale_sum = 0.0f;
  if (normalize_scales) {
    for (int i = lane; i < k; i += kWarpSize) {
      scale_sum += SoftmaxScale(s_scores[i], max_val, inv_sum);
    }
    scale_sum = WarpReduceSum(scale_sum);
  }
  const float denom = TopKNormalizeDenom(normalize_scales, scale_sum);

  for (int i = lane; i < k; i += kWarpSize) {
    const float prob = SoftmaxScale(s_scores[i], max_val, inv_sum);
    topk_scales[static_cast<size_t>(row) * k + i] = normalize_scales ? (prob / denom) : prob;
    topk_indices[static_cast<size_t>(row) * k + i] = s_indices[i];
  }
}

template <typename KernelPtr>
bool IsFusedRouteKernelSm90Available(KernelPtr kernel) {
  static thread_local int cached_device = -1;
  static thread_local KernelPtr cached_kernel = nullptr;
  static thread_local bool available = false;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (device != cached_device || kernel != cached_kernel) {
    int driver_version = 0;
    int sm_major = 0;
    cudaFuncAttributes attributes{};
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    available = driver_version >= 12000 && sm_major >= 9 &&
                attributes.ptxVersion >= 90 && attributes.binaryVersion >= 90;
    cached_device = device;
    cached_kernel = kernel;
  }
  return available;
}

template <typename T>
bool LaunchFusedRouteMapsT(const T* logits, float* topk_scales, int* topk_indices,
                           int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
                           int* permuted_expert, int64_t* expert_first_token_offset, int num_rows,
                           int num_experts, int k, bool normalize_scales, bool pdl, cudaStream_t stream) {
  // k is limited to the values the single-CTA radix-rank map builder supports (1, 2, 4, 6, 8, 10): the fused path then
  // replaces exactly the launches it is bit-identical to; other k keep the generic route.
  const bool supported_k = k == 1 || k == 2 || k == 4 || k == 6 || k == 8 || k == 10;
  if (num_rows < 1 || num_rows > kFusedRouteMaxRows || num_experts != kFusedRouteExperts || !supported_k ||
      reinterpret_cast<uintptr_t>(logits) % 8 != 0) {
    return false;
  }
  const auto kernel = k == 10 ? FusedSoftmaxTopKMapsKernel<T, 10> : FusedSoftmaxTopKMapsKernel<T, 0>;
  if (!IsFusedRouteKernelSm90Available(kernel)) {
    return false;
  }
  if (pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(kFusedRouteThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1]{};
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    const cudaError_t status = cudaLaunchKernelEx(
        &config, kernel, logits, topk_scales, topk_indices, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, permuted_expert, expert_first_token_offset, num_rows, k, normalize_scales);
    if (status == cudaSuccess) {
      return true;
    }
    if (status != cudaErrorInvalidValue && status != cudaErrorNotSupported) {
      ORT_ENFORCE(status == cudaSuccess, "fused QMoE routing launch failed: ", cudaGetErrorString(status));
    }
    (void)cudaGetLastError();
  }
  kernel<<<1, kFusedRouteThreads, 0, stream>>>(
      logits, topk_scales, topk_indices, permuted_row_to_unpermuted_row,
      unpermuted_row_to_permuted_row, permuted_expert, expert_first_token_offset, num_rows, k, normalize_scales);
  CUDA_CALL_THROW(cudaGetLastError());
  return true;
}

bool LaunchFusedRouteMaps(const half* logits, float* topk_scales, int* topk_indices, int* r2u, int* u2r, int* expert,
                          int64_t* offsets, int num_rows, int num_experts, int k, bool normalize_scales, bool pdl,
                          cudaStream_t stream) {
  return LaunchFusedRouteMapsT<half>(logits, topk_scales, topk_indices, r2u, u2r, expert, offsets, num_rows,
                                     num_experts, k, normalize_scales, pdl, stream);
}

bool LaunchFusedRouteMaps(const __nv_bfloat16* logits, float* topk_scales, int* topk_indices, int* r2u, int* u2r,
                          int* expert, int64_t* offsets, int num_rows, int num_experts, int k, bool normalize_scales,
                          bool pdl, cudaStream_t stream) {
  return LaunchFusedRouteMapsT<__nv_bfloat16>(logits, topk_scales, topk_indices, r2u, u2r, expert, offsets, num_rows,
                                              num_experts, k, normalize_scales, pdl, stream);
}

template <typename T>
void DispatchSoftmaxTopK(const T* logits, float* topk_scales, int* topk_indices,
                         int num_rows, int num_experts, int k, bool normalize_scales,
                         cudaStream_t stream) {
  ORT_ENFORCE(k > 0 && k <= num_experts,
              "SoftmaxTopK requires 0 < k <= num_experts, got k=", k,
              " and num_experts=", num_experts);

  // Block-per-row CUB merge sort is the fastest path for the common decode case
  // (one block fully sorts a row). Pick the smallest capacity that covers
  // num_experts. k must fit in s_topk (<= 64).
  const dim3 grid(static_cast<unsigned>(num_rows));
  if (k <= 64 && num_experts <= 1024) {
    // Tiny expert counts: a single warp sorts a row entirely in registers via
    // an in-register bitonic sort (no shared memory). Multiple warps per block
    // process multiple rows for better occupancy.
    if (num_experts <= onnxruntime::cuda::topk::kWarpBitonicMaxSize) {
      constexpr int kWarpsPerBlock = 8;
      const dim3 block(static_cast<unsigned>(onnxruntime::cuda::topk::kWarpSize), kWarpsPerBlock);
      const dim3 bitonic_grid(static_cast<unsigned>((num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock));
      SoftmaxTopKWarpBitonicKernel<T, kWarpsPerBlock><<<bitonic_grid, block, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= onnxruntime::cuda::topk::kWarpMergeMaxSize) {
      // Single warp per row sorts up to 64 logits in shared memory (CUB warp
      // merge sort), the genai-recommended path for sort sizes in (32, 64].
      SoftmaxTopKWarpMergeKernel<T, 64><<<grid, onnxruntime::cuda::topk::kWarpSize, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 256 && k <= 8) {
      // Warp-per-row register top-k. Pick the smallest per-lane capacity that both
      // covers num_experts across the warp and can hold the requested k.
      constexpr int kRowsPerBlock = 4;
      const dim3 warp_block(static_cast<unsigned>(onnxruntime::cuda::topk::kWarpSize), kRowsPerBlock);
      const dim3 warp_grid(static_cast<unsigned>((num_rows + kRowsPerBlock - 1) / kRowsPerBlock));
      const int items_per_lane =
          std::max(k, (num_experts + onnxruntime::cuda::topk::kWarpSize - 1) / onnxruntime::cuda::topk::kWarpSize);
      if (items_per_lane <= 4) {
        SoftmaxTopKWarpTopKKernel<T, 4, kRowsPerBlock><<<warp_grid, warp_block, 0, stream>>>(
            logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      } else {
        SoftmaxTopKWarpTopKKernel<T, 8, kRowsPerBlock><<<warp_grid, warp_block, 0, stream>>>(
            logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      }
      return;
    } else if (num_experts <= 128) {
      SoftmaxTopKMergeKernel<T, 128, 1><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 256) {
      SoftmaxTopKMergeKernel<T, 128, 2><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 512) {
      SoftmaxTopKMergeKernel<T, 128, 4><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else /*if (num_experts <= 1024)*/ {
      SoftmaxTopKMergeKernel<T, 256, 4><<<grid, 256, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    }
  } else {
    // Fall back to the simple one-thread-per-row kernel.
    const int block = 256;
    const int grid_1d = Compute1DGridSize(num_rows, block);
    SoftmaxTopKKernel<T><<<grid_1d, block, 0, stream>>>(
        logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
  }
}

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<float>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<half>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

void LaunchSoftmaxTopK(
    const __nv_bfloat16* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<__nv_bfloat16>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

template <typename T>
__global__ void QMoEPrePackZPKernel(const uint8_t* zp, const T* scales, T* out, int num_elements, float offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float s = static_cast<float>(scales[idx]);
    float z = static_cast<float>(zp[idx]);
    // Compute bias = (offset - zp) * scale
    // If offset = 0, bias = -zp * scale
    // If offset = 128 (e.g. for uint8 -> int8 shift), bias = (128 - zp) * scale
    out[idx] = static_cast<T>((offset - z) * s);
  }
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<float><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<half><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

template <typename T>
__global__ void QMoEPrePackPacked4BitZPKernel(const uint8_t* packed_zp, const T* scales, T* out, int num_elements, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float s = static_cast<float>(scales[idx]);

    // 4-bit unpacking with stride N
    // row = idx / N; col = idx % N;
    // byte_row = row >> 1; nibble = row & 1;
    // byte_idx = byte_row * N + col;

    int row = idx / N;
    int col = idx % N;
    int byte_idx = (row >> 1) * N + col;

    uint8_t packed_byte = packed_zp[byte_idx];
    uint8_t val = (packed_byte >> ((row & 1) << 2)) & 0x0F;
    float z = static_cast<float>(val);

    // Bias calculation for Cutlass dequantizer: (8.0 - ZP) * Scale
    // Cutlass dequantizer uses formula: (q - 8) * scale + bias
    // We want: (q - zp) * scale
    // (q - 8) * scale + bias = q*scale - 8*scale + bias
    // q*scale - zp*scale = q*scale - zp*scale
    // So: -8*scale + bias = -zp*scale => bias = (8 - zp) * scale
    out[idx] = static_cast<T>((8.0f - z) * s);
  }
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const float* scales,
    float* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<float><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const half* scales,
    half* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<half><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<float><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<half><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

// Batched 4-bit packed ZP scaled bias kernel.
// Grid: (ceil(n/16), ceil(k_blocks/16), experts)
// Each thread computes one output element: output[e][out_row][out_col]
// where out_row in [0, k_blocks), out_col in [0, n).
// ZP layout: [experts, n, packed_k_blocks] with packed_k_blocks = (k_blocks+1)/2
// Scale/Output layout: [experts, k_blocks, n]
template <typename T>
__global__ void QMoEScaledZP4BitBatchedKernel(
    const uint8_t* packed_zp,
    const T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks,
    float default_zero_point) {
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int expert = blockIdx.z;

  if (out_col < n && out_row < k_blocks) {
    int packed_k_blocks = (k_blocks + 1) / 2;
    int64_t expert_zp_offset = static_cast<int64_t>(expert) * n * packed_k_blocks;
    int64_t expert_scale_offset = static_cast<int64_t>(expert) * k_blocks * n;

    // ZP is [n, packed_k_blocks] per expert; in_row = out_col, in_col = out_row
    int in_row = out_col;
    int in_col = out_row;
    int64_t packed_zp_offset = expert_zp_offset + static_cast<int64_t>(in_row) * packed_k_blocks + in_col / 2;
    uint8_t packed_byte = packed_zp[packed_zp_offset];
    float zero_point_val = static_cast<float>((in_col & 0x01) ? (packed_byte >> 4) : (packed_byte & 0x0f));

    int64_t output_offset = expert_scale_offset + static_cast<int64_t>(out_row) * n + out_col;
    T scale_val = transposed_scale[output_offset];
    float result = static_cast<float>(scale_val) * (-zero_point_val + default_zero_point);
    scaled_zero_point[output_offset] = static_cast<T>(result);
  }
}

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const half* transposed_scale,
    half* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
      (k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE,
      experts);
  QMoEScaledZP4BitBatchedKernel<half><<<gridDim, blockDim, 0, stream>>>(
      packed_zp, transposed_scale, scaled_zero_point, n, k_blocks, default_zero_point);
}

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
      (k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE,
      experts);
  QMoEScaledZP4BitBatchedKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      packed_zp, transposed_scale, scaled_zero_point, n, k_blocks, default_zero_point);
}

__global__ void QMoEShiftWeightsKernel(const uint8_t* input, uint8_t* output, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = input[idx] ^ 0x80;
  }
}

void LaunchQMoEShiftWeights(
    const uint8_t* input,
    uint8_t* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEShiftWeightsKernel<<<grid, block, 0, stream>>>(input, output, num_elements);
}

// ====================== Sparse Mixer Kernel ===============================
// Ported from old/moe_kernel.cu

static constexpr int WARP_SIZE = 32;

template <typename T, int TPB, int NUM_EXPERTS>
__launch_bounds__(TPB) __global__
    void sparse_mixer_top2(const T* inputs, float* output, int* indices, int* source_rows, const float jitter_eps) {
  static constexpr int K = 2;

  using cub_kvp = cub::KeyValuePair<int, T>;
  using KVBlockReduce = cub::BlockReduce<cub_kvp, TPB>;

  __shared__ float result_kvp_value[K];
  __shared__ typename KVBlockReduce::TempStorage kvTmpStorage;

  cub_kvp thread_kvp;
  // cub::ArgMax arg_max; // Use default ArgMax

  // Manually define ArgMax functor if not available or to ensure behavior
  struct ArgMax {
    __device__ __forceinline__ cub_kvp operator()(const cub_kvp& a, const cub_kvp& b) const {
      return (b.value > a.value) ? b : a;
    }
  } arg_max;

  int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const int thread_row_offset = blockIdx.x * NUM_EXPERTS;

  float factor[K];
  bool logits_mask[K];

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-FLT_MAX);

    cub_kvp inp_kvp;
#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[K * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = KVBlockReduce(kvTmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = K * block_row + k_idx;
      result_kvp_value[k_idx] = (float)result_kvp.value;
      indices[idx] = result_kvp.key;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();

#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      factor[k_idx] = max(abs((float)inputs[idx]), result_kvp_value[k_idx]);
      logits_mask[k_idx] = (result_kvp_value[k_idx] - (float)inputs[idx]) > (2 * jitter_eps * factor[k_idx]);
      if (k_idx == 1 && expert == indices[K * block_row]) {
        logits_mask[1] = true;
      }
    }
  }

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    float row_sum(0);

#pragma unroll
    for (int ii = threadIdx.x; ii < NUM_EXPERTS; ii += TPB) {
      const int idx = thread_row_offset + ii;
      row_sum += logits_mask[k_idx] ? 0 : exp((static_cast<float>(inputs[idx]) - result_kvp_value[k_idx]));
    }

#pragma unroll
    for (int mask = NUM_EXPERTS / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, NUM_EXPERTS);
    }

    const float normalizing_factor = 1.f / row_sum;

    const int idx = K * block_row + k_idx;
    if (threadIdx.x == indices[idx]) {
      const int input_idx = thread_row_offset + threadIdx.x;
      output[idx] = logits_mask[k_idx] ? 0
                                       : exp((static_cast<float>(inputs[input_idx]) - result_kvp_value[k_idx])) *
                                             normalizing_factor;
    }
  }
}

template <typename T>
void LaunchSparseMixerTop2Impl(
    const T* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int TPB = WARP_SIZE * WARPS_PER_TB;
  static constexpr float jitter_eps = 0.01f;

  switch (num_experts) {
    case 8: {
      sparse_mixer_top2<T, TPB, 8><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_rows, jitter_eps);
      break;
    }
    case 16: {
      sparse_mixer_top2<T, TPB, 16><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_rows, jitter_eps);
      break;
    }
    // Replicate logic for other sizes if needed, or fallback/throw
    default: {
      ORT_THROW("Sparse mixer only supports 8 or 16 experts, got ", num_experts);
    }
  }
}

void LaunchSparseMixerTop2(
    const float* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<float>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

void LaunchSparseMixerTop2(
    const half* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<half>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

void LaunchSparseMixerTop2(
    const __nv_bfloat16* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<__nv_bfloat16>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

template <typename T>
__global__ void QMoETranspose2DKernel(const T* input, T* output, int num_elements_per_batch, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z;

  if (col < cols && row < rows) {
    int in_idx = batch * num_elements_per_batch + row * cols + col;
    int out_idx = batch * num_elements_per_batch + col * rows + row;
    output[out_idx] = input[in_idx];
  }
}

void LaunchQMoETranspose2D(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<float><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const half* input,
    half* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<half><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<uint8_t><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

__device__ __forceinline__ int64_t QMoEBlockScaleInterleaveOffset(
    int batch, int row, int col, int rows_padded, int cols_padded) {
  int64_t num_k_tiles = (cols_padded + 3) / 4;
  int64_t m_tile_idx = row / 128;
  int64_t k_tile_idx = col / 4;
  int64_t tile_offset = ((m_tile_idx * num_k_tiles) + k_tile_idx) * 512;
  int64_t intra_tile_offset = (row % 32) * 16 + ((row % 128) / 32) * 4 + (col % 4);
  int64_t batch_stride = ((rows_padded + 127) / 128) * num_k_tiles * 512;
  return static_cast<int64_t>(batch) * batch_stride + tile_offset + intra_tile_offset;
}

__global__ void QMoEBlockScaleInterleaveKernel(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded) {
  for (int row = blockIdx.x; row < rows_padded; row += gridDim.x) {
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int col = threadIdx.x; col < cols_padded; col += blockDim.x) {
        uint8_t scale = 0;
        if (row < rows && col < cols) {
          scale = input[static_cast<int64_t>(batch) * rows * cols + row * cols + col];
        }
        output[QMoEBlockScaleInterleaveOffset(batch, row, col, rows_padded, cols_padded)] = scale;
      }
    }
  }
}

void LaunchQMoEBlockScaleInterleave(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded,
    int multi_processor_count,
    cudaStream_t stream) {
  dim3 block(std::min(cols_padded, 1024));
  int num_blocks_per_sm = std::max(1, 4096 / static_cast<int>(block.x));
  dim3 grid(std::min(rows_padded, multi_processor_count * num_blocks_per_sm));
  QMoEBlockScaleInterleaveKernel<<<grid, block, 0, stream>>>(
      input, output, batch_size, rows, cols, rows_padded, cols_padded);
}

// E2M1 has only 8 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6), so the obvious implementation is a
// table lookup -- but a runtime-indexed local table compiles to a constant-bank load that the
// hardware replays once per distinct address in a warp, and neighbouring weights rarely share a
// code. Assembling the float bits directly keeps it branch-free and in registers:
//   e != 0: value = 2^(e-1) * (1 + m/2)  -> biased exponent 126 + e, mantissa MSB m
//   e == 0: value = m ? 0.5 : 0          -> biased exponent 126 with zero mantissa, or all zero
__device__ __forceinline__ float DecodeFp4E2M1(uint8_t code) {
  const uint32_t e = (code >> 1) & 0x3u;
  const uint32_t m = code & 0x1u;
  const uint32_t sign = static_cast<uint32_t>(code & 0x8u) << 28;
  const uint32_t normal = ((126u + e) << 23) | (m << 22);
  const uint32_t subnormal = m ? (126u << 23) : 0u;
  return __uint_as_float(sign | (e ? normal : subnormal));
}

__device__ __forceinline__ float DecodeUE8M0(uint8_t code) {
  return code == 0 ? 0.0f : exp2f(static_cast<int>(code) - 127);
}

template <bool NativeDecode = false>
__device__ __forceinline__ float DecodeFloat8E4M3FN(uint8_t code) {
#if defined(ORT_CUDA_TYPE_HELPER_HAS_FP8) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (NativeDecode) {
    return e4m3_to_float(code);
  }
#endif
  // ONNX float8e4m3fn has no infinities. The only NaN payloads are 0x7F/0xFF;
  // finite values, including the max finite code 0x7E, use the normal E4M3 formula.
  const int sign = code & 0x80;
  const int exponent = (code >> 3) & 0x0F;
  const int mantissa = code & 0x07;

  if ((code & 0x7F) == 0) {
    return sign ? -0.0f : 0.0f;
  }
  if (exponent == 0x0F && mantissa == 0x07) {
    return __int_as_float(0x7fffffff);
  }

  float value = 0.0f;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  }
  return sign ? -value : value;
}

// Tile shape for QMoEDequantizeFp4WeightsVecKernel. kTileN = 64 rows is exactly 32 packed bytes,
// so a block consumes every sector of the packed weights it touches; kTileK / kVecK = 8 threads
// per row make each store 128 contiguous bytes. Block is (8, 64) = 512 threads.
constexpr int kQMoEDequantizeFp4VecK = 8;
constexpr int kQMoEDequantizeFp4TileN = 64;
constexpr int kQMoEDequantizeFp4TileK = 64;

// Coalescing-optimized FP4 / NVFP4 weight dequantization.
//
// The scalar kernels below map one thread to one output element with the k index varying
// fastest. The packed weights are stored [E, K, N/2] (n-packed) while the output is [E, N, K],
// so that mapping makes consecutive lanes read packed bytes ``packed_n`` apart -- a separate
// 32-byte sector per lane for half a byte of payload, plus a 64-bit div/mod per element. On the
// 40-layer NVFP4 MoE prefill it ran at ~5% of HBM peak (3.8 ms per launch, 307 ms per forward).
//
// This kernel tiles the output: a block covers ``kTileN`` consecutive rows x ``kTileK``
// consecutive k, with threadIdx.x selecting the k group and threadIdx.y the row. Each thread
// emits ``kVecK`` = 8 values (one 16-byte uint4).
//
// The mapping matters more than the vector width. Nsight Compute on an earlier revision that gave
// each thread 16 (then 32) consecutive k of a *single* row reported ~95% of ``Max Bandwidth`` at
// only ~26% DRAM throughput: the kernel was bound by memory *requests*, not by DRAM. With one row
// per lane a warp's 32 stores land ``k * sizeof(T)`` bytes apart, so every lane needs its own
// 32-byte sector and widening the per-lane store leaves requests-per-byte unchanged (measured: no
// improvement from 16 -> 32 values per thread).
//
// Here threadIdx.x spans kTileK / kVecK = 8 lanes of the same row, so each group of 8 lanes issues
// one 128-byte contiguous store and a warp covers 4 rows in 4 requests instead of 32. The packed
// reads stay efficient because kTileN = 64 rows is exactly 32 packed bytes, so the block consumes
// every sector it touches:
//   * stores: 128 contiguous bytes per 8 lanes, 100% of each sector used;
//   * reads: one 32-byte sector per (expert, k) fully consumed by the block's 64 rows;
//   * the block-scale byte and the per-expert global scale are read once per kVecK values;
//   * the index decomposition is pure grid arithmetic, so there is no integer division.
//
// ``kE4M3Scale`` selects the NVFP4 scale codec (Float8E4M3FN, block 16) over the MXFP4 one
// (Float8E8M0, block 32); ``kBlockSize`` is the matching scale block size. kVecK divides both, so
// each thread's values always share one block scale.
template <typename T, int kBlockSize, bool kE4M3Scale>
__global__ void QMoEDequantizeFp4WeightsVecKernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    T* __restrict__ output,
    int n,
    int k) {
  constexpr int kVecK = kQMoEDequantizeFp4VecK;
  const int row = static_cast<int>(blockIdx.x) * kQMoEDequantizeFp4TileN + static_cast<int>(threadIdx.y);
  if (row >= n) {
    return;
  }
  const int k_base = static_cast<int>(blockIdx.y) * kQMoEDequantizeFp4TileK +
                     static_cast<int>(threadIdx.x) * kVecK;
  const int expert = static_cast<int>(blockIdx.z);

  const int packed_n = n >> 1;
  const int shift = (row & 1) ? 4 : 0;
  const int64_t weight_base = (static_cast<int64_t>(expert) * k + k_base) * packed_n + (row >> 1);

  const int scale_k = k / kBlockSize;
  const uint8_t scale_code = block_scales[(static_cast<int64_t>(expert) * n + row) * scale_k + k_base / kBlockSize];
  const float scale = (kE4M3Scale ? DecodeFloat8E4M3FN(scale_code) : DecodeUE8M0(scale_code)) * global_scales[expert];

  // uint4 storage keeps the staging buffer 16-byte aligned for the vector store below.
  uint4 staged[kVecK * sizeof(T) / sizeof(uint4)];
  T* values = reinterpret_cast<T*>(staged);
#pragma unroll
  for (int j = 0; j < kVecK; ++j) {
    const uint8_t packed = packed_weights[weight_base + static_cast<int64_t>(j) * packed_n];
    values[j] = static_cast<T>(DecodeFp4E2M1(static_cast<uint8_t>((packed >> shift) & 0x0F)) * scale);
  }

  uint4* dst = reinterpret_cast<uint4*>(output + (static_cast<int64_t>(expert) * n + row) * k + k_base);
#pragma unroll
  for (int v = 0; v < static_cast<int>(kVecK * sizeof(T) / sizeof(uint4)); ++v) {
    dst[v] = staged[v];
  }
}

// The vectorized kernel needs an even n (nibble packing), a k that is a whole number of tiles, and
// a scale block that is a multiple of the per-thread vector so every thread's values share one
// block scale. The expert and k-tile counts also have to fit gridDim.z / gridDim.y (65535). Every
// shape produced by the QMoE quantizers satisfies this; the scalar kernels stay as the fallback
// for anything else.
template <int kBlockSize>
inline bool QMoEDequantizeFp4VecApplies(int num_experts, int n, int k) {
  return (n % 2) == 0 && (k % kQMoEDequantizeFp4TileK) == 0 && (k % kBlockSize) == 0 &&
         (kBlockSize % kQMoEDequantizeFp4VecK) == 0 &&
         (k / kQMoEDequantizeFp4TileK) <= 65535 && num_experts <= 65535;
}

template <typename T>
__global__ void QMoEDequantizeFp4WeightsKernel(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k;
  int expert = static_cast<int>(index / expert_stride);
  int64_t offset = index - static_cast<int64_t>(expert) * expert_stride;
  int row = static_cast<int>(offset / k);
  int col = static_cast<int>(offset - static_cast<int64_t>(row) * k);

  int packed_n = n / 2;
  uint8_t packed = packed_weights[(static_cast<int64_t>(expert) * k + col) * packed_n + row / 2];
  uint8_t fp4_code = (row & 1) == 0 ? (packed & 0x0F) : (packed >> 4);

  int scale_k = k / 32;
  uint8_t scale_code = block_scales[(static_cast<int64_t>(expert) * n + row) * scale_k + col / 32];
  float value = DecodeFp4E2M1(fp4_code) * DecodeUE8M0(scale_code) * global_scales[expert];
  output[index] = static_cast<T>(value);
}

template <typename T>
void LaunchQMoEDequantizeFp4WeightsImpl(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int block = 256;
  if (QMoEDequantizeFp4VecApplies<32>(num_experts, n, k)) {
    const dim3 tile_block(kQMoEDequantizeFp4TileK / kQMoEDequantizeFp4VecK, kQMoEDequantizeFp4TileN);
    const dim3 tile_grid((n + kQMoEDequantizeFp4TileN - 1) / kQMoEDequantizeFp4TileN,
                         k / kQMoEDequantizeFp4TileK, num_experts);
    QMoEDequantizeFp4WeightsVecKernel<T, 32, false><<<tile_grid, tile_block, 0, stream>>>(
        packed_weights, block_scales, global_scales, output, n, k);
    CUDA_CALL_THROW(cudaGetLastError());
    return;
  }
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEDequantizeFp4WeightsKernel<<<grid, block, 0, stream>>>(
      packed_weights, block_scales, global_scales, output, num_experts, n, k);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k, stream);
}

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k, stream);
}

// ---------------------------------------------------------------------------
// MXFP4 -> FP8 (e4m3) weight conversion for the QMoE DeepGEMM path.
//
// The FP8 GEMM scales B by one fp32 factor per [128 N, 128 K] block. Factoring the arbitrary
// per-expert global scale out of the FP8 value and rounding the remaining block scale to a
// *power of two* makes the conversion bit-exact: an E2M1 code carries at most two significant
// bits, e4m3 carries four, and a power-of-two scale only shifts exponents. The conventional
// amax/448 scale would give every weight a full mantissa before rounding it to three bits
// (measured 4.76% max relative error).
//
// Losslessness still needs the quantized magnitudes to stay inside e4m3's range, i.e. the
// per-block spread of the MXFP4 group exponents must be at most 14. It is 6 for this model, but
// the second pass verifies the round trip element-by-element rather than assuming it.
// ---------------------------------------------------------------------------
constexpr int kQMoEFp8BlockN = 128;
constexpr int kQMoEFp8BlockK = 128;
constexpr int kQMoEFp8ScaleThreads = 256;
constexpr int kQMoEFp8VecK = 16;
constexpr int kQMoEFp8TileN = 64;
constexpr int kQMoEFp8TileK = 128;
constexpr float kQMoEFp8Max = 448.0f;

// One CUDA block per [128 N, 128 K] weight block, emitting sfb[expert][n / 128][k / 128].
__global__ void QMoEFp4ToFp8BlockScaleKernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    float* __restrict__ output_scales,
    int n,
    int k) {
  using BlockReduce = cub::BlockReduce<float, kQMoEFp8ScaleThreads>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int n_block = static_cast<int>(blockIdx.x);
  const int k_block = static_cast<int>(blockIdx.y);
  const int expert = static_cast<int>(blockIdx.z);
  const int packed_n = n >> 1;
  const int scale_k = k >> 5;  // MXFP4 group size is 32
  constexpr int kHalfRows = kQMoEFp8BlockN / 2;

  float local_max = 0.0f;
  for (int i = static_cast<int>(threadIdx.x); i < kQMoEFp8BlockK * kHalfRows; i += kQMoEFp8ScaleThreads) {
    const int col = k_block * kQMoEFp8BlockK + i / kHalfRows;
    const int half = i % kHalfRows;
    const uint8_t packed = packed_weights[(static_cast<int64_t>(expert) * k + col) * packed_n +
                                          n_block * kHalfRows + half];
    const int row = n_block * kQMoEFp8BlockN + 2 * half;
    const int64_t scale_base = (static_cast<int64_t>(expert) * n + row) * scale_k + (col >> 5);
    const float even = DecodeFp4E2M1(static_cast<uint8_t>(packed & 0x0F)) * DecodeUE8M0(block_scales[scale_base]);
    const float odd = DecodeFp4E2M1(static_cast<uint8_t>(packed >> 4)) * DecodeUE8M0(block_scales[scale_base + scale_k]);
    local_max = fmaxf(local_max, fmaxf(fabsf(even), fabsf(odd)));
  }
  const float amax = BlockReduceMax<BlockReduce>(local_max, temp_storage);

  if (threadIdx.x == 0) {
    // Preserve the arbitrary fp32 global scale and round only the MXFP4 exponent component.
    // frexpf gives amax/448 = m * 2^e with m in [0.5, 1).
    float scale = 1.0f;
    const float global_scale = fabsf(global_scales[expert]);
    if (amax > 0.0f && global_scale > 0.0f) {
      int exponent = 0;
      const float mantissa = frexpf(amax * (1.0f / kQMoEFp8Max), &exponent);
      exponent -= mantissa == 0.5f;
      scale = ldexpf(global_scale, exponent);
    }
    output_scales[(static_cast<int64_t>(expert) * (n / kQMoEFp8BlockN) + n_block) * (k / kQMoEFp8BlockK) + k_block] =
        scale;
  }
}

// Second pass: quantize with the power-of-two block scale from above. Tiling matches
// QMoEDequantizeFp4WeightsVecKernel (8 lanes of one row, 16 bytes each = one 128-byte store).
__global__ void QMoEFp4ToFp8WeightsKernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    const float* __restrict__ output_scales,
    uint8_t* __restrict__ output,
    int* __restrict__ inexact,
    int n,
    int k) {
  const int row = static_cast<int>(blockIdx.x) * kQMoEFp8TileN + static_cast<int>(threadIdx.y);
  if (row >= n) {
    return;
  }
  const int k_base = static_cast<int>(blockIdx.y) * kQMoEFp8TileK + static_cast<int>(threadIdx.x) * kQMoEFp8VecK;
  const int expert = static_cast<int>(blockIdx.z);

  const int packed_n = n >> 1;
  const int shift = (row & 1) ? 4 : 0;
  const int64_t weight_base = (static_cast<int64_t>(expert) * k + k_base) * packed_n + (row >> 1);

  const int scale_k = k >> 5;
  const float group_scale =
      DecodeUE8M0(block_scales[(static_cast<int64_t>(expert) * n + row) * scale_k + (k_base >> 5)]);
  const float block_scale =
      output_scales[(static_cast<int64_t>(expert) * (n / kQMoEFp8BlockN) + row / kQMoEFp8BlockN) *
                        (k / kQMoEFp8BlockK) +
                    k_base / kQMoEFp8BlockK];
  const float scaled_global = global_scales[expert] / block_scale;

  union {
    uint4 vec;
    uint8_t bytes[kQMoEFp8VecK];
  } staged;
  bool exact = true;
#pragma unroll
  for (int j = 0; j < kQMoEFp8VecK; ++j) {
    const uint8_t packed = packed_weights[weight_base + static_cast<int64_t>(j) * packed_n];
    const float fp4_value = DecodeFp4E2M1(static_cast<uint8_t>((packed >> shift) & 0x0F));
    const float value = fp4_value * group_scale * global_scales[expert];
    const __nv_fp8_e4m3 quantized(fp4_value * group_scale * scaled_global);
    staged.bytes[j] = quantized.__x;
    exact = exact && (static_cast<float>(quantized) * block_scale == value);
  }
  *reinterpret_cast<uint4*>(output + (static_cast<int64_t>(expert) * n + row) * k + k_base) = staged.vec;
  if (!exact) {
    *inexact = 1;
  }
}

void LaunchQMoEQuantizeFp4WeightsToFp8(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    uint8_t* output,
    float* output_scales,
    int* inexact_flag,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  ORT_ENFORCE(n % kQMoEFp8BlockN == 0 && k % kQMoEFp8BlockK == 0 && (n % kQMoEFp8TileN) == 0,
              "QMoE MXFP4->FP8 conversion requires n a multiple of ", kQMoEFp8BlockN,
              " and k a multiple of ", kQMoEFp8BlockK, ", got n=", n, " k=", k);
  static_assert(kQMoEFp8VecK * sizeof(uint8_t) == sizeof(uint4), "vector store must be 16 bytes");
  // 32 is the MXFP4 group size; a thread's kQMoEFp8VecK values must share one group scale.
  static_assert(32 % kQMoEFp8VecK == 0);
  static_assert(kQMoEFp8TileK % kQMoEFp8VecK == 0);

  const dim3 scale_grid(n / kQMoEFp8BlockN, k / kQMoEFp8BlockK, num_experts);
  QMoEFp4ToFp8BlockScaleKernel<<<scale_grid, kQMoEFp8ScaleThreads, 0, stream>>>(
      packed_weights, block_scales, global_scales, output_scales, n, k);
  CUDA_CALL_THROW(cudaGetLastError());

  const dim3 grid(n / kQMoEFp8TileN, k / kQMoEFp8TileK, num_experts);
  const dim3 block(kQMoEFp8TileK / kQMoEFp8VecK, kQMoEFp8TileN);
  QMoEFp4ToFp8WeightsKernel<<<grid, block, 0, stream>>>(
      packed_weights, block_scales, global_scales, output_scales, output, inexact_flag, n, k);
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T>
__global__ void QMoECombineFp4ScalesForGemvKernel(
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int experts,
    int n,
    int k_blocks) {
  int64_t total = static_cast<int64_t>(experts) * n * k_blocks;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k_blocks;
  int expert = static_cast<int>(index / expert_stride);
  int64_t offset = index - static_cast<int64_t>(expert) * expert_stride;
  int row = static_cast<int>(offset / k_blocks);
  int k_block = static_cast<int>(offset - static_cast<int64_t>(row) * k_blocks);

  int64_t output_index = (static_cast<int64_t>(expert) * k_blocks + k_block) * n + row;
  float scale = DecodeUE8M0(block_scales[index]) * global_scales[expert];
  output[output_index] = static_cast<T>(scale);
}

template <typename T>
void LaunchQMoECombineFp4ScalesForGemvImpl(
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(experts) * n * k_blocks;
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoECombineFp4ScalesForGemvKernel<<<grid, block, 0, stream>>>(
      block_scales, global_scales, output, experts, n, k_blocks);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  LaunchQMoECombineFp4ScalesForGemvImpl(block_scales, global_scales, output, experts, n, k_blocks, stream);
}

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  LaunchQMoECombineFp4ScalesForGemvImpl(block_scales, global_scales, output, experts, n, k_blocks, stream);
}

__global__ void QMoEPackFp4ScalesForTmaWsKernel(
    const uint8_t* input,
    uint8_t* output,
    int n,
    int k_blocks,
    int k_blocks_padded,
    int packed_scales_per_k_tile,
    int64_t total) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int inner = static_cast<int>(index % packed_scales_per_k_tile);
  int row = static_cast<int>((index / packed_scales_per_k_tile) % n);
  int64_t packed_k_tile = (index / packed_scales_per_k_tile) / n;
  int packed_k_tiles = k_blocks_padded / packed_scales_per_k_tile;
  int expert = static_cast<int>(packed_k_tile / packed_k_tiles);
  int tile = static_cast<int>(packed_k_tile - static_cast<int64_t>(expert) * packed_k_tiles);
  int k_block = tile * packed_scales_per_k_tile + inner;

  // Tail k-blocks added by padding k_blocks up to a multiple of packed_scales_per_k_tile are
  // zero-filled. They are only read by the GEMM's last partial CTA-K-tile, where the matching
  // A/B K elements are TMA-zeroed, so their scale value does not affect the result.
  output[index] = (k_block < k_blocks)
                      ? input[(static_cast<int64_t>(expert) * n + row) * k_blocks + k_block]
                      : static_cast<uint8_t>(0);
}

void LaunchQMoEPackFp4ScalesForTmaWs(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  constexpr int kPackedScalesPerKTile = 8;
  // Pad k_blocks up to a multiple of the packed K-tile so the GEMM's last (possibly partial)
  // CTA-K-tile can read a full packed scale group. The caller must allocate the output buffer
  // for the padded k-block count (experts * n * k_blocks_padded bytes).
  const int k_blocks_padded =
      ((k_blocks + kPackedScalesPerKTile - 1) / kPackedScalesPerKTile) * kPackedScalesPerKTile;
  int64_t total = static_cast<int64_t>(experts) * n * k_blocks_padded;
  if (total <= 0) {
    return;
  }
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEPackFp4ScalesForTmaWsKernel<<<grid, block, 0, stream>>>(
      input, output, n, k_blocks, k_blocks_padded, kPackedScalesPerKTile, total);
  CUDA_CALL_THROW(cudaGetLastError());
}

__global__ void QMoEMarkActiveExpertsKernel(
    const int* expert_indices, int* expert_to_compact, int64_t num_routes) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_routes) {
    atomicExch(expert_to_compact + expert_indices[index], 0);
  }
}

__global__ void QMoECompactExpertMapKernel(
    int* expert_to_compact, int* compact_to_expert, int num_experts, int expert_capacity, int pass_offset) {
  // Expert counts are small; a serial prefix preserves ascending expert order deterministically.
  int active_index = 0;
  int slot = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    if (expert_to_compact[expert] == 0) {
      if (active_index >= pass_offset && slot < expert_capacity) {
        expert_to_compact[expert] = slot;
        compact_to_expert[slot++] = expert;
      } else {
        expert_to_compact[expert] = -1;
      }
      ++active_index;
    }
  }
  for (; slot < expert_capacity; ++slot) {
    compact_to_expert[slot] = -1;
  }
}

__global__ void QMoERemapExpertsKernel(
    const int* expert_indices, int* compact_indices, const int* expert_to_compact, int64_t num_routes) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_routes) {
    compact_indices[index] = expert_to_compact[expert_indices[index]];
  }
}

void LaunchQMoECompactExperts(
    const int* expert_indices,
    int* compact_indices,
    int* expert_to_compact,
    int* compact_to_expert,
    int num_experts,
    int expert_capacity,
    int64_t num_routes,
    cudaStream_t stream,
    int pass_offset) {
  ORT_ENFORCE(num_experts > 0 && num_routes > 0 &&
                  expert_capacity > 0 &&
                  (pass_offset > 0 || expert_capacity >= std::min<int64_t>(num_experts, num_routes) || expert_capacity <= num_experts) &&
                  expert_capacity <= num_experts,
              "Invalid QMoE compact expert capacity.");
  CUDA_CALL_THROW(cudaMemsetAsync(expert_to_compact, 0xff, static_cast<size_t>(num_experts) * sizeof(int), stream));
  constexpr int block = 256;
  const int grid = onnxruntime::narrow<int>((num_routes - 1) / block + 1);
  QMoEMarkActiveExpertsKernel<<<grid, block, 0, stream>>>(expert_indices, expert_to_compact, num_routes);
  CUDA_CALL_THROW(cudaGetLastError());
  QMoECompactExpertMapKernel<<<1, 1, 0, stream>>>(expert_to_compact, compact_to_expert, num_experts, expert_capacity, pass_offset);
  CUDA_CALL_THROW(cudaGetLastError());
  QMoERemapExpertsKernel<<<grid, block, 0, stream>>>(expert_indices, compact_indices, expert_to_compact, num_routes);
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T, bool NativeDecode = false>
__device__ __forceinline__ T QMoEFp8Weight(const QMoEFp8ProjectionParams& params, int expert, int row, int col) {
  const uint8_t* weights = params.weights;
  const void* scales = params.scales;
  int scale_type = params.scale_type;
  int source_n = params.n;
  if (params.up_weights) {
    if (row % 2) {
      weights = params.up_weights;
      scales = params.up_scales;
      scale_type = params.up_scale_type;
    }
    source_n /= 2;
    row /= 2;
  } else if (params.fusion == 2) {
    row = row / 2 + (row % 2) * (source_n / 2);
  }
  const int scale_n = (source_n + params.block_size - 1) / params.block_size;
  const int scale_k = (params.k + params.block_size - 1) / params.block_size;
  const int64_t scale_index = (static_cast<int64_t>(expert) * scale_n + row / params.block_size) * scale_k +
                              col / params.block_size;
  const int64_t weight_index = (static_cast<int64_t>(expert) * source_n + row) * params.k + col;
  const float scale = scale_type == 0   ? static_cast<const float*>(scales)[scale_index]
                      : scale_type == 1 ? __half2float(static_cast<const half*>(scales)[scale_index])
                                        : __bfloat162float(static_cast<const __nv_bfloat16*>(scales)[scale_index]);
  return static_cast<T>(DecodeFloat8E4M3FN<NativeDecode>(weights[weight_index]) * scale);
}

template <typename T, bool NativeDecode = false, typename OutputT = T>
__global__ void QMoEFp8GemvKernel(QMoEFp8ProjectionParams params, const T* input, OutputT* output) {
  const int row = static_cast<int>(blockIdx.y);
  const int expert = params.experts[row];
  const int output_col = static_cast<int>(blockIdx.x) * 4 + threadIdx.x / 32;
  if (output_col >= params.n) {
    return;
  }
  const int input_row = params.row_to_unpermuted ? params.row_to_unpermuted[row] % params.num_rows : row;
  float sum = 0.0f;
  for (int col = threadIdx.x % 32; col < params.k; col += 32) {
    sum = fmaf(static_cast<float>(input[static_cast<int64_t>(input_row) * params.k + col]),
               static_cast<float>(QMoEFp8Weight<T, NativeDecode>(params, expert, output_col, col)), sum);
  }
  sum = WarpReduceSum(sum);
  if (threadIdx.x % 32 == 0) {
    output[static_cast<int64_t>(row) * params.n + output_col] = static_cast<OutputT>(sum);
  }
}

bool QMoEFp8GemvSupportsNativeDecode() {
#if defined(ORT_CUDA_TYPE_HELPER_HAS_FP8)
  cudaFuncAttributes attributes{};
  CUDA_CALL_THROW(cudaFuncGetAttributes(&attributes, QMoEFp8GemvKernel<half, true>));
  return attributes.binaryVersion >= 90 && attributes.ptxVersion >= 90;
#else
  return false;
#endif
}

bool QMoEFp8GemvKernelUsesNativeDecode(const void* kernel) {
  return kernel == reinterpret_cast<const void*>(QMoEFp8GemvKernel<half, true>) ||
         kernel == reinterpret_cast<const void*>(QMoEFp8GemvKernel<__nv_bfloat16, true>) ||
         kernel == reinterpret_cast<const void*>(QMoEFp8GemvKernel<half, true, float>) ||
         kernel == reinterpret_cast<const void*>(QMoEFp8GemvKernel<__nv_bfloat16, true, float>);
}

__global__ void QMoEFp8ExpertTilesKernel(const int64_t* expert_offsets, int* tile_offsets, int num_experts) {
  int tiles = 0;
  tile_offsets[0] = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    tiles += static_cast<int>((expert_offsets[expert + 1] - expert_offsets[expert] + 15) / 16);
    tile_offsets[expert + 1] = tiles;
  }
}

void LaunchQMoEFp8ExpertTiles(const int64_t* expert_offsets, int* tile_offsets,
                              int num_experts, cudaStream_t stream) {
  QMoEFp8ExpertTilesKernel<<<1, 1, 0, stream>>>(expert_offsets, tile_offsets, num_experts);
  CUDA_CALL_THROW(cudaGetLastError());
}

__global__ void QMoERemapExpertRangeKernel(
    const int* __restrict__ expert_indices,
    int* __restrict__ compact_indices,
    int64_t num_routes,
    int pass_start,
    int expert_capacity) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_routes) {
    const int e = expert_indices[index];
    compact_indices[index] = (e >= pass_start && e < pass_start + expert_capacity) ? (e - pass_start) : -1;
  }
}

__global__ void QMoEAccumulateExpertHistogramKernel(
    const int* __restrict__ expert_indices,
    int64_t num_routes,
    int num_experts,
    int* __restrict__ global_hist) {
  __shared__ int smem_hist[1024];
  const int capped = num_experts <= 1024 ? num_experts : 1024;
  for (int i = threadIdx.x; i < capped; i += blockDim.x) {
    smem_hist[i] = 0;
  }
  __syncthreads();
  for (int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       r < num_routes;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int e = expert_indices[r];
    if (e >= 0 && e < capped) {
      atomicAdd(&smem_hist[e], 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < capped; i += blockDim.x) {
    const int c = smem_hist[i];
    if (c > 0) {
      atomicAdd(&global_hist[i], c);
    }
  }
}

void LaunchQMoEMarkActiveExperts(
    const int* expert_indices,
    int* expert_active_mask,
    int num_experts,
    int64_t num_routes,
    cudaStream_t stream) {
  if (num_experts <= 0 || num_routes <= 0) return;
  CUDA_CALL_THROW(cudaMemsetAsync(expert_active_mask, 0xff, static_cast<size_t>(num_experts) * sizeof(int), stream));
  constexpr int block = 256;
  const int grid = onnxruntime::narrow<int>((num_routes - 1) / block + 1);
  QMoEMarkActiveExpertsKernel<<<grid, block, 0, stream>>>(expert_indices, expert_active_mask, num_routes);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoERemapExpertRange(
    const int* expert_indices,
    int* compact_indices,
    int64_t num_routes,
    int pass_start,
    int expert_capacity,
    cudaStream_t stream) {
  if (num_routes <= 0) return;
  constexpr int block = 256;
  const int grid = onnxruntime::narrow<int>((num_routes - 1) / block + 1);
  QMoERemapExpertRangeKernel<<<grid, block, 0, stream>>>(
      expert_indices, compact_indices, num_routes, pass_start, expert_capacity);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoEAccumulateExpertHistogram(
    const int* expert_indices,
    int64_t num_routes,
    int num_experts,
    int* global_hist,
    cudaStream_t stream) {
  if (num_routes <= 0 || num_experts <= 0 || global_hist == nullptr) return;
  constexpr int block = 256;
  const int grid = std::min<int>(onnxruntime::narrow<int>((num_routes + block - 1) / block), 32);
  QMoEAccumulateExpertHistogramKernel<<<grid, block, 0, stream>>>(
      expert_indices, num_routes, num_experts, global_hist);
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T, typename OutputT>
__global__ void QMoEFp8GemmKernel(QMoEFp8ProjectionParams params, const T* input, OutputT* output) {
#if __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  const int tile = static_cast<int>(blockIdx.y);
  if (tile >= params.tile_offsets[params.num_experts]) {
    return;
  }
  int first = 0;
  int last = params.num_experts;
  while (first < last) {
    const int middle = (first + last) / 2;
    if (params.tile_offsets[middle + 1] <= tile) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  const int expert = first;
  const int64_t row_start = params.expert_offsets[expert] + (tile - params.tile_offsets[expert]) * 16;
  const int64_t row_end = params.expert_offsets[expert + 1];
  const int column_start = static_cast<int>(blockIdx.x) * 64;
  __shared__ __align__(32) T activation_tile[16 * 32];
  __shared__ __align__(32) T weight_tile[64 * 32];
  __shared__ __align__(32) float output_tile[16 * 64];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  const int warp = threadIdx.x / 32;
  for (int k_start = 0; k_start < params.k; k_start += 32) {
    for (int index = threadIdx.x; index < 16 * 32; index += 128) {
      const int64_t row = row_start + index / 32;
      const int col = k_start + index % 32;
      T value = static_cast<T>(0.0f);
      if (row < row_end && col < params.k) {
        const int input_row = params.row_to_unpermuted ? params.row_to_unpermuted[row] % params.num_rows
                                                       : static_cast<int>(row);
        value = input[static_cast<int64_t>(input_row) * params.k + col];
      }
      activation_tile[index] = value;
    }
    for (int index = threadIdx.x; index < 64 * 32; index += 128) {
      const int row = column_start + index / 32;
      const int col = k_start + index % 32;
      weight_tile[index] = row < params.n && col < params.k
                               ? QMoEFp8Weight<T>(params, expert, row, col)
                               : static_cast<T>(0.0f);
    }
    __syncthreads();
    for (int inner = 0; inner < 32; inner += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> activation;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> weight;
      wmma::load_matrix_sync(activation, activation_tile + inner, 32);
      wmma::load_matrix_sync(weight, weight_tile + warp * 16 * 32 + inner, 32);
      wmma::mma_sync(accumulator, activation, weight, accumulator);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(output_tile + warp * 16, accumulator, 64, wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x; index < 16 * 64; index += 128) {
    const int64_t row = row_start + index / 64;
    const int col = column_start + index % 64;
    if (row < row_end && col < params.n) {
      output[row * params.n + col] = static_cast<OutputT>(output_tile[index]);
    }
  }
#endif
}

template <typename T, typename OutputT>
void LaunchQMoEFp8Projection(const QMoEFp8ProjectionParams& params,
                             const T* input, OutputT* output, cudaStream_t stream) {
  if (params.tile_offsets) {
    const int tiles = std::min(params.expanded_rows,
                               std::min(params.num_experts, params.expanded_rows) + params.expanded_rows / 16);
    QMoEFp8GemmKernel<T, OutputT><<<dim3((params.n + 63) / 64, tiles), 128, 0, stream>>>(params, input, output);
  } else {
    static const bool native_decode =
        ParseEnvironmentVariableWithDefault<bool>("ORT_QMOE_FP8_GEMV_NATIVE_DECODE", false);
    if (native_decode) {
      QMoEFp8GemvKernel<T, true, OutputT><<<dim3((params.n + 3) / 4, params.expanded_rows), 128, 0, stream>>>(params, input, output);
    } else {
      QMoEFp8GemvKernel<T, false, OutputT><<<dim3((params.n + 3) / 4, params.expanded_rows), 128, 0, stream>>>(params, input, output);
    }
  }
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T, typename InputT>
__global__ void QMoEFp8ActivationKernel(const InputT* input, const T* bias, T* output,
                                        const int* experts, int rows, int inter_size, int fusion,
                                        float alpha, float beta, float limit) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(rows) * inter_size) {
    return;
  }
  const int row = static_cast<int>(index / inter_size);
  const int col = static_cast<int>(index % inter_size);
  const int n = inter_size * (fusion ? 2 : 1);
  const int gate_col = fusion ? 2 * col : col;
  float gate = static_cast<float>(input[static_cast<int64_t>(row) * n + gate_col]);
  float up = fusion ? static_cast<float>(input[static_cast<int64_t>(row) * n + gate_col + 1]) : 1.0f;
  if (bias) {
    const int bias_col = fusion == 2 ? col : gate_col;
    gate += static_cast<float>(bias[static_cast<int64_t>(experts[row]) * n + bias_col]);
    if (fusion) {
      up += static_cast<float>(bias[static_cast<int64_t>(experts[row]) * n +
                                    (fusion == 2 ? col + inter_size : gate_col + 1)]);
    }
  }
  if (fusion) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit) + beta;
  }
  output[index] = static_cast<T>(gate / (1.0f + expf(-alpha * gate)) * up);
}

template <typename T, typename InputT>
void LaunchQMoEFp8Activation(const InputT* input, const T* bias, T* output,
                             const int* experts, int rows, int inter_size, int fusion,
                             float alpha, float beta, float limit, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inter_size;
  QMoEFp8ActivationKernel<<<narrow<int>((count + 255) / 256), 256, 0, stream>>>(
      input, bias, output, experts, rows, inter_size, fusion, alpha, beta, limit);
  CUDA_CALL_THROW(cudaGetLastError());
}

template void LaunchQMoEFp8Projection<half>(const QMoEFp8ProjectionParams&, const half*, half*, cudaStream_t);
template void LaunchQMoEFp8Projection<__nv_bfloat16>(const QMoEFp8ProjectionParams&, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);
template void LaunchQMoEFp8Projection<half, float>(const QMoEFp8ProjectionParams&, const half*, float*, cudaStream_t);
template void LaunchQMoEFp8Projection<__nv_bfloat16, float>(const QMoEFp8ProjectionParams&, const __nv_bfloat16*, float*, cudaStream_t);
template void LaunchQMoEFp8Activation<half>(const half*, const half*, half*, const int*, int, int, int, float, float, float, cudaStream_t);
template void LaunchQMoEFp8Activation<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int*, int, int, int, float, float, float, cudaStream_t);
template void LaunchQMoEFp8Activation<half, float>(const float*, const half*, half*, const int*, int, int, int, float, float, float, cudaStream_t);
template void LaunchQMoEFp8Activation<__nv_bfloat16, float>(const float*, const __nv_bfloat16*, __nv_bfloat16*, const int*, int, int, int, float, float, float, cudaStream_t);

template <typename T>
__global__ void QMoEAccumulatePassOutputKernel(
    float* __restrict__ accum,
    const T* __restrict__ pass_output,
    T* __restrict__ final_output,
    int64_t count,
    bool write_final) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    const float sum = accum[idx] + static_cast<float>(pass_output[idx]);
    accum[idx] = sum;
    if (write_final) {
      final_output[idx] = static_cast<T>(sum);
    }
  }
}

void LaunchQMoEAccumulatePassOutput(
    float* accum,
    const half* pass_output,
    half* final_output,
    int64_t count,
    bool write_final,
    cudaStream_t stream) {
  if (count <= 0) return;
  constexpr int block = 256;
  const int grid = onnxruntime::narrow<int>((count + block - 1) / block);
  QMoEAccumulatePassOutputKernel<half><<<grid, block, 0, stream>>>(accum, pass_output, final_output, count, write_final);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoEAccumulatePassOutput(
    float* accum,
    const __nv_bfloat16* pass_output,
    __nv_bfloat16* final_output,
    int64_t count,
    bool write_final,
    cudaStream_t stream) {
  if (count <= 0) return;
  constexpr int block = 256;
  const int grid = onnxruntime::narrow<int>((count + block - 1) / block);
  QMoEAccumulatePassOutputKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(accum, pass_output, final_output, count, write_final);
  CUDA_CALL_THROW(cudaGetLastError());
}
template <typename T>
__global__ void QMoEDequantizeFp8WeightsKernel(
    const uint8_t* weights,
    const float* scales,
    T* output,
    int num_experts,
    int n,
    int k,
    int block_size,
    bool split_fc1,
    bool block_fused_fc1,
    const int* compact_to_expert,
    const T* bias,
    T* output_bias) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k;
  const int output_expert = static_cast<int>(index / expert_stride);
  const int expert = compact_to_expert ? compact_to_expert[output_expert] : output_expert;
  if (expert < 0) {
    return;
  }
  int64_t local_index = index % expert_stride;
  int64_t scale_index = expert;
  if (block_size > 0) {
    int scale_n = n / block_size + (n % block_size != 0);
    int scale_k = k / block_size + (k % block_size != 0);
    scale_index = (static_cast<int64_t>(expert) * scale_n + local_index / k / block_size) * scale_k +
                  local_index % k / block_size;
  }
  float value = DecodeFloat8E4M3FN(weights[static_cast<int64_t>(expert) * expert_stride + local_index]) *
                scales[scale_index];
  // Normalize separate and block-fused gate/up projections to interleaved rows.
  int64_t output_index = index;
  if (split_fc1) {
    output_index = static_cast<int64_t>(output_expert) * 2 * expert_stride + 2 * (local_index / k) * k + local_index % k;
  } else if (block_fused_fc1) {
    int64_t row = local_index / k;
    int64_t output_row = row < n / 2 ? 2 * row : 2 * (row - n / 2) + 1;
    output_index = static_cast<int64_t>(output_expert) * expert_stride + output_row * k + local_index % k;
  }
  output[output_index] = static_cast<T>(value);
  if (bias && local_index % k == 0) {
    output_bias[output_index / k] = bias[static_cast<int64_t>(expert) * n + local_index / k];
  }
}

template <typename T>
void LaunchQMoEDequantizeFp8WeightsImpl(
    const uint8_t* weights,
    const float* scales,
    T* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    int block_size,
    bool split_fc1,
    bool block_fused_fc1,
    const int* compact_to_expert,
    const T* bias,
    T* output_bias) {
  ORT_ENFORCE(bias == nullptr || output_bias != nullptr, "QMoE FP8 bias gathering requires an output buffer.");
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEDequantizeFp8WeightsKernel<<<grid, block, 0, stream>>>(
      weights, scales, output, num_experts, n, k, block_size, split_fc1, block_fused_fc1,
      compact_to_expert, bias, output_bias);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    int block_size,
    bool split_fc1,
    bool block_fused_fc1,
    const int* compact_to_expert,
    const half* bias,
    half* output_bias) {
  LaunchQMoEDequantizeFp8WeightsImpl(weights, scales, output, num_experts, n, k, stream,
                                     block_size, split_fc1, block_fused_fc1, compact_to_expert, bias, output_bias);
}

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    int block_size,
    bool split_fc1,
    bool block_fused_fc1,
    const int* compact_to_expert,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output_bias) {
  LaunchQMoEDequantizeFp8WeightsImpl(weights, scales, output, num_experts, n, k, stream,
                                     block_size, split_fc1, block_fused_fc1, compact_to_expert, bias, output_bias);
}

// NVFP4 dequantization. Identical structure to QMoEDequantizeFp4WeightsKernel (MXFP4) except:
//   (a) block size is 16 (scale_k = k / 16) instead of 32,
//   (b) the per-block scale is a Float8E4M3FN byte decoded via DecodeFloat8E4M3FN
//       instead of a Float8E8M0 byte decoded via DecodeUE8M0.
// Weight layout [E, K, N/2] (N-packed) and nibble selection are identical to the MXFP4 kernel.
//   value = DecodeFp4E2M1(fp4_code) * DecodeE4M3(block_scale[n, k/16]) * global_scales[expert]
template <typename T>
__global__ void QMoEDequantizeNvfp4WeightsKernel(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k,
    const int* compact_to_expert,
    const T* bias,
    T* output_bias,
    bool weights_row_major) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k;
  const int output_expert = static_cast<int>(index / expert_stride);
  const int expert = compact_to_expert ? compact_to_expert[output_expert] : output_expert;
  if (expert < 0) {
    return;
  }
  int64_t offset = index % expert_stride;
  int row = static_cast<int>(offset / k);
  int col = static_cast<int>(offset - static_cast<int64_t>(row) * k);

  int packed_n = n / 2;
  const int64_t packed_index = weights_row_major
                                   ? (static_cast<int64_t>(expert) * n + row) * (k / 2) + col / 2
                                   : (static_cast<int64_t>(expert) * k + col) * packed_n + row / 2;
  const int nibble = weights_row_major ? col : row;
  uint8_t packed = packed_weights[packed_index];
  uint8_t fp4_code = (nibble & 1) == 0 ? (packed & 0x0F) : (packed >> 4);

  constexpr int kNvfp4BlockSize = 16;
  int scale_k = k / kNvfp4BlockSize;
  uint8_t scale_code = block_scales[(static_cast<int64_t>(expert) * n + row) * scale_k + col / kNvfp4BlockSize];
  float value = DecodeFp4E2M1(fp4_code) * DecodeFloat8E4M3FN(scale_code) * global_scales[expert];
  output[index] = static_cast<T>(value);
  if (bias && col == 0) {
    output_bias[static_cast<int64_t>(output_expert) * n + row] = bias[static_cast<int64_t>(expert) * n + row];
  }
}

// Float8E4M3FN decode built from float bits; returns exactly the values of DecodeFloat8E4M3FN.
__device__ __forceinline__ float DecodeFloat8E4M3FNFast(uint32_t code) {
  const uint32_t exponent = (code >> 3) & 0x0Fu;
  const uint32_t mantissa = code & 0x07u;
  float value;
  if (exponent == 0u) {
    value = static_cast<float>(mantissa) * 0.001953125f;  // 2^-9
  } else if (exponent == 0x0Fu && mantissa == 0x07u) {
    return __int_as_float(0x7fffffff);
  } else {
    value = __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
  }
  return (code & 0x80u) ? -value : value;
}

// Row-major NVFP4 dequantization, [E, n, k/2] packed bytes (low nibble = even column) and
// [E, n, k/16] block scales. One thread converts 32 consecutive columns of one row: a 16-byte load,
// two scale bytes and 64 bytes of output, so loads and stores are fully coalesced (the scalar kernel
// above issues one byte load and one 2-byte store per element with a 64-bit div/mod each).
// Values are computed in the same order as the scalar kernel, (e2m1 * e4m3) * global, so the output is bit identical.
template <typename T>
__global__ void QMoEDequantizeNvfp4RowMajorKernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    T* __restrict__ output,
    int num_experts,
    int n,
    int k,
    const int* __restrict__ compact_to_expert,
    const T* __restrict__ bias,
    T* __restrict__ output_bias,
    const uint8_t* __restrict__ resident_packed_weights = nullptr,
    const uint8_t* __restrict__ resident_block_scales = nullptr,
    const int* __restrict__ resident_expert_to_slot = nullptr,
    int pass_start_expert = -1,
    const int* __restrict__ expert_active_mask = nullptr,
    const uint8_t* __restrict__ host_packed_weights = nullptr,
    const uint8_t* __restrict__ host_block_scales = nullptr,
    size_t host_expert_stride = 0,
    const int* __restrict__ host_expert_to_slot = nullptr) {
  const int groups_per_row = k >> 5;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_experts) * n * groups_per_row;
  if (index >= total) {
    return;
  }
  const int64_t global_row = index / groups_per_row;
  const int group = static_cast<int>(index - global_row * groups_per_row);
  const int output_expert = static_cast<int>(global_row / n);
  const int row = static_cast<int>(global_row - static_cast<int64_t>(output_expert) * n);
  const int expert = (pass_start_expert >= 0)
                         ? (pass_start_expert + output_expert)
                         : (compact_to_expert ? compact_to_expert[output_expert] : output_expert);
  if (expert < 0 || (expert_active_mask != nullptr && expert_active_mask[expert] < 0)) {
    return;
  }
  const int resident_slot = (resident_expert_to_slot != nullptr) ? resident_expert_to_slot[expert] : -1;
  const bool use_resident = (resident_slot >= 0 && resident_packed_weights != nullptr && resident_block_scales != nullptr);
  int host_slot = -1;
  int cold_index = (pass_start_expert >= 0) ? output_expert : expert;
  if (!use_resident && host_expert_to_slot != nullptr) {
    if (pass_start_expert >= 0) {
      const int map_val = host_expert_to_slot[output_expert];
      if (map_val <= -2 && host_packed_weights != nullptr) {
        host_slot = -2 - map_val;
        cold_index = -1;
      } else {
        cold_index = map_val;
      }
    } else if (host_packed_weights != nullptr) {
      host_slot = host_expert_to_slot[expert];
    }
  }
  const bool use_host_cache = (host_slot >= 0 && host_packed_weights != nullptr && host_block_scales != nullptr && host_expert_stride > 0);
  if (!use_resident && !use_host_cache && cold_index < 0) {
    return;
  }
  const size_t w_row_bytes = static_cast<size_t>(k >> 1);
  const size_t s_row_bytes = static_cast<size_t>(k >> 4);
  const uint8_t* w_expert_base = use_resident
                                     ? (resident_packed_weights + static_cast<size_t>(resident_slot) * n * w_row_bytes)
                                     : (use_host_cache
                                            ? (host_packed_weights + static_cast<size_t>(host_slot) * host_expert_stride)
                                            : (packed_weights + static_cast<size_t>(cold_index) * n * w_row_bytes));
  const uint8_t* s_expert_base = use_resident
                                     ? (resident_block_scales + static_cast<size_t>(resident_slot) * n * s_row_bytes)
                                     : (use_host_cache
                                            ? (host_block_scales + static_cast<size_t>(host_slot) * host_expert_stride)
                                            : (block_scales + static_cast<size_t>(cold_index) * n * s_row_bytes));
  const int64_t orig_source_row = static_cast<int64_t>(expert) * n + row;
  const uint4 packed = *reinterpret_cast<const uint4*>(w_expert_base + static_cast<size_t>(row) * w_row_bytes + (static_cast<int64_t>(group) << 4));
  const uint16_t scale_pair = *reinterpret_cast<const uint16_t*>(s_expert_base + static_cast<size_t>(row) * s_row_bytes + (group << 1));
  const float global_scale = global_scales[expert];
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

  uint4 staged[4];
  T* values = reinterpret_cast<T*>(staged);
#pragma unroll
  for (int half_block = 0; half_block < 2; ++half_block) {
    const float block_scale = DecodeFloat8E4M3FNFast((scale_pair >> (8 * half_block)) & 0xFFu);
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      const uint32_t word = words[half_block * 2 + w];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float fp4 = DecodeFp4E2M1(static_cast<uint8_t>((word >> (4 * j)) & 0x0Fu));
        values[half_block * 16 + w * 8 + j] = static_cast<T>((fp4 * block_scale) * global_scale);
      }
    }
  }
  uint4* dst = reinterpret_cast<uint4*>(output + global_row * k + (static_cast<int64_t>(group) << 5));
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    dst[v] = staged[v];
  }
  if (bias && group == 0) {
    output_bias[global_row] = bias[orig_source_row];
  }
}

template <typename T>
void LaunchQMoEDequantizeNvfp4WeightsImpl(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    const int* compact_to_expert,
    const T* bias,
    T* output_bias,
    bool weights_row_major,
    const uint8_t* resident_packed_weights = nullptr,
    const uint8_t* resident_block_scales = nullptr,
    const int* resident_expert_to_slot = nullptr,
    int pass_start_expert = -1,
    const int* expert_active_mask = nullptr,
    const uint8_t* host_packed_weights = nullptr,
    const uint8_t* host_block_scales = nullptr,
    size_t host_expert_stride = 0,
    const int* host_expert_to_slot = nullptr) {
  ORT_ENFORCE(bias == nullptr || output_bias != nullptr, "QMoE NVFP4 bias gathering requires an output buffer.");
  constexpr int block = 256;
  if (!weights_row_major && !compact_to_expert && pass_start_expert < 0 && !bias && QMoEDequantizeFp4VecApplies<16>(num_experts, n, k)) {
    const dim3 tile_block(kQMoEDequantizeFp4TileK / kQMoEDequantizeFp4VecK, kQMoEDequantizeFp4TileN);
    const dim3 tile_grid((n + kQMoEDequantizeFp4TileN - 1) / kQMoEDequantizeFp4TileN,
                         k / kQMoEDequantizeFp4TileK, num_experts);
    QMoEDequantizeFp4WeightsVecKernel<T, 16, true><<<tile_grid, tile_block, 0, stream>>>(
        packed_weights, block_scales, global_scales, output, n, k);
    CUDA_CALL_THROW(cudaGetLastError());
    return;
  }
  if (weights_row_major && (k % 32) == 0) {
    const int64_t threads = static_cast<int64_t>(num_experts) * n * (k / 32);
    const int64_t blocks = (threads + block - 1) / block;
    QMoEDequantizeNvfp4RowMajorKernel<T><<<onnxruntime::narrow<int>(blocks), block, 0, stream>>>(
        packed_weights, block_scales, global_scales, output, num_experts, n, k, compact_to_expert, bias, output_bias,
        resident_packed_weights, resident_block_scales, resident_expert_to_slot, pass_start_expert, expert_active_mask,
        host_packed_weights, host_block_scales, host_expert_stride, host_expert_to_slot);
    CUDA_CALL_THROW(cudaGetLastError());
    return;
  }
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEDequantizeNvfp4WeightsKernel<<<grid, block, 0, stream>>>(
      packed_weights, block_scales, global_scales, output, num_experts, n, k,
      compact_to_expert, bias, output_bias, weights_row_major);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    const int* compact_to_expert,
    const half* bias,
    half* output_bias,
    bool weights_row_major,
    const uint8_t* resident_packed_weights,
    const uint8_t* resident_block_scales,
    const int* resident_expert_to_slot,
    int pass_start_expert,
    const int* expert_active_mask,
    const uint8_t* host_packed_weights,
    const uint8_t* host_block_scales,
    size_t host_expert_stride,
    const int* host_expert_to_slot) {
  LaunchQMoEDequantizeNvfp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k,
                                       stream, compact_to_expert, bias, output_bias, weights_row_major,
                                       resident_packed_weights, resident_block_scales, resident_expert_to_slot,
                                       pass_start_expert, expert_active_mask,
                                       host_packed_weights, host_block_scales, host_expert_stride, host_expert_to_slot);
}

void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    const int* compact_to_expert,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output_bias,
    bool weights_row_major,
    const uint8_t* resident_packed_weights,
    const uint8_t* resident_block_scales,
    const int* resident_expert_to_slot,
    int pass_start_expert,
    const int* expert_active_mask,
    const uint8_t* host_packed_weights,
    const uint8_t* host_block_scales,
    size_t host_expert_stride,
    const int* host_expert_to_slot) {
  LaunchQMoEDequantizeNvfp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k,
                                       stream, compact_to_expert, bias, output_bias, weights_row_major,
                                       resident_packed_weights, resident_block_scales, resident_expert_to_slot,
                                       pass_start_expert, expert_active_mask,
                                       host_packed_weights, host_block_scales, host_expert_stride, host_expert_to_slot);
}

// NVFP4 counterpart of QMoECombineFp4ScalesForGemvKernel. Identical [E, n, k_blocks] ->
// [E, k_blocks, n] transpose and per-expert global-scale fold, but the per-block scale byte is a
// Float8E4M3FN code decoded via DecodeFloat8E4M3FN (block size 16) instead of a Float8E8M0 code
// decoded via DecodeUE8M0 (block size 32). Produces the TypeA folded scales the fused MoE GEMV
// consumes for NVFP4 (quant_type="nvfp4").
template <typename T>
__global__ void QMoECombineNvfp4ScalesForGemvKernel(
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int experts,
    int n,
    int k_blocks) {
  int64_t total = static_cast<int64_t>(experts) * n * k_blocks;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k_blocks;
  int expert = static_cast<int>(index / expert_stride);
  int64_t offset = index - static_cast<int64_t>(expert) * expert_stride;
  int row = static_cast<int>(offset / k_blocks);
  int k_block = static_cast<int>(offset - static_cast<int64_t>(row) * k_blocks);

  int64_t output_index = (static_cast<int64_t>(expert) * k_blocks + k_block) * n + row;
  float scale = DecodeFloat8E4M3FN(block_scales[index]) * global_scales[expert];
  output[output_index] = static_cast<T>(scale);
}

template <typename T>
void LaunchQMoECombineNvfp4ScalesForGemvImpl(
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(experts) * n * k_blocks;
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoECombineNvfp4ScalesForGemvKernel<<<grid, block, 0, stream>>>(
      block_scales, global_scales, output, experts, n, k_blocks);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  LaunchQMoECombineNvfp4ScalesForGemvImpl(block_scales, global_scales, output, experts, n, k_blocks, stream);
}

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream) {
  LaunchQMoECombineNvfp4ScalesForGemvImpl(block_scales, global_scales, output, experts, n, k_blocks, stream);
}

// Repack column-major FP4 packed weights to row-major layout.
// Input: [experts, k, n/2] packed col-major (each byte holds 2 values along n).
// Output: [experts, n, k/2] packed row-major (each byte holds 2 values along k).
// One thread per output byte.
__global__ void QMoERepackFP4ColToRowKernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int experts,
    int64_t k,
    int64_t n) {
  const int64_t k_half = k / 2;
  const int64_t n_half = n / 2;
  const int64_t out_expert_stride = n * k_half;
  const int64_t in_expert_stride = k * n_half;
  const int64_t total = static_cast<int64_t>(experts) * out_expert_stride;

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int64_t expert = idx / out_expert_stride;
  int64_t rem = idx - expert * out_expert_stride;
  int64_t row = rem / k_half;       // output row index [0, n)
  int64_t col_byte = rem % k_half;  // output byte index [0, k/2)

  int64_t col_even = col_byte * 2;
  int64_t col_odd = col_even + 1;

  // Source byte addresses: src[expert][col][row/2]
  int64_t in_base = expert * in_expert_stride;
  uint8_t src_even = input[in_base + col_even * n_half + row / 2];
  uint8_t src_odd = input[in_base + col_odd * n_half + row / 2];

  // Extract nibble based on row parity
  uint8_t low_code = (row % 2 == 0) ? (src_even & 0x0F) : ((src_even >> 4) & 0x0F);
  uint8_t high_code = (row % 2 == 0) ? (src_odd & 0x0F) : ((src_odd >> 4) & 0x0F);

  output[idx] = low_code | static_cast<uint8_t>(high_code << 4);
}

void LaunchQMoERepackFP4ColToRow(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int64_t k,
    int64_t n,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(experts) * n * (k / 2);
  constexpr int kThreads = 256;
  int blocks = onnxruntime::narrow<int>((total + kThreads - 1) / kThreads);
  QMoERepackFP4ColToRowKernel<<<blocks, kThreads, 0, stream>>>(
      input, output, experts, k, n);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

namespace onnxruntime::llm::kernels {

template <typename T>
__global__ void BatchedTransposeKernel(const T* __restrict__ input, T* __restrict__ output, int batch, int rows, int cols) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t matrix_size = static_cast<int64_t>(rows) * cols;
  int64_t total_size = static_cast<int64_t>(batch) * matrix_size;

  if (idx < total_size) {
    int64_t b = idx / matrix_size;
    int64_t rem = idx % matrix_size;
    int r = rem / cols;
    int c = rem % cols;

    int64_t out_idx = b * matrix_size + static_cast<int64_t>(c) * rows + r;
    output[out_idx] = input[idx];
  }
}

void LaunchBatchedTranspose(cudaStream_t stream, const void* input, void* output, int batch, int rows, int cols, int element_size) {
  int64_t total_elements = static_cast<int64_t>(batch) * rows * cols;
  int threads = 256;
  int blocks = onnxruntime::narrow<int>((total_elements + threads - 1) / threads);

  if (element_size == 1) {
    BatchedTransposeKernel<uint8_t><<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output), batch, rows, cols);
  } else if (element_size == 2) {
    BatchedTransposeKernel<uint16_t><<<blocks, threads, 0, stream>>>(static_cast<const uint16_t*>(input), static_cast<uint16_t*>(output), batch, rows, cols);
  } else if (element_size == 4) {
    BatchedTransposeKernel<uint32_t><<<blocks, threads, 0, stream>>>(static_cast<const uint32_t*>(input), static_cast<uint32_t*>(output), batch, rows, cols);
  } else {
    ORT_THROW("LaunchBatchedTranspose: unsupported element_size ", element_size,
              " (supported: 1, 2, 4)");
  }
}

}  // namespace onnxruntime::llm::kernels
