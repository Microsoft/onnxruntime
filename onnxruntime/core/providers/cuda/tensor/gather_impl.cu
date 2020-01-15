// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_impl.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

namespace onnxruntime {
namespace cuda {

static const int WARP_SIZE = 32;

template <typename T, typename Tin>
__global__ void _GatherKernel(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod output_block_size,
    const fast_divmod block_size,
    const Tin* indices_data,
    const T* input_data,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  int input_block_index, block_offset;
  output_block_size.divmod(id, input_block_index, block_offset);
  int indices_index, offset;
  block_size.divmod(block_offset, indices_index, offset);
  int64_t idx = indices_data[indices_index];
  idx = idx < 0 ? idx + indices_max : idx;
  if (idx < 0 || idx >= indices_max) {
    output_data[id] = 0;
    return;
  }

  input_index = input_block_index * input_block_size + idx * block_size.d_ + offset;
  output_data[id] = input_data[input_index];
}

template <typename T, typename Tin>
void GatherImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod* output_block_size,
    const fast_divmod* block_size,
    const Tin* indices_data,
    const T* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GatherKernel<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_block_size, indices_max, *output_block_size, *block_size, indices_data, input_data, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T)                                                                                               \
  template void GatherImpl<T, int32_t>(const int64_t input_block_size, const int64_t indices_max,                         \
                                       const fast_divmod* output_block_size, const fast_divmod* block_size,               \
                                       const int32_t* indices_data, const T* input_data, T* output_data, const size_t N); \
  template void GatherImpl<T, int64_t>(const int64_t input_block_size, const int64_t indices_max,                         \
                                       const fast_divmod* output_block_size, const fast_divmod* block_size,               \
                                       const int64_t* indices_data, const T* input_data, T* output_data, const size_t N);

SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(bool)

template <typename T, typename Tin>
__global__ void _GatherGradImpl(
    Tin* input,
    Tin* indices,
    T* grad_output,
    T* grad_weight,
    int64_t numel,
    int64_t input_numel,
    int64_t param_itrs,
    int64_t stride) {
  int idx = blockIdx.x * 4 + threadIdx.y;

  const int SZ = 4;
  if (idx < numel && (idx == 0 || input[idx] != input[idx - 1])) {
    do {
      for (int itr = 0; itr < param_itrs; ++itr) {
        const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        const int weight_row = itr * input_numel + ((int)input[idx]) * stride;  //the offset of the input
        const int grad_row = (itr * numel + ((int)indices[idx])) * stride;      //the offset of the gradient

        T gradient[SZ];
        T weight[SZ];

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * WARP_SIZE;
          if (feature_dim < stride) {
            gradient[ii] = static_cast<T>(grad_output[grad_row + feature_dim]);
            weight[ii] = static_cast<T>(grad_weight[weight_row + feature_dim]);
          }
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          weight[ii] += gradient[ii];
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * WARP_SIZE;
          if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = static_cast<T>(weight[ii]);
          }
        }
      }
      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T, typename Tin>
void GatherGradImpl(
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    Tin* origin_indices,
    T* output_data,
    const int64_t num_inputs,  //The number of input elements starting from the gathering dimension
    const int64_t param_itrs,  //The size of dimensions of the data before gathering dimension
    ThrustAllocator& allocator) {
  // sort the index
  auto count_iter = thrust::counting_iterator<Tin>(0);
  auto origin_data = thrust::device_ptr<Tin>(origin_indices);

  auto policy = thrust::cuda::par(allocator);
  thrust::copy(policy, count_iter, count_iter + num_indices, origin_data);
  //TODO: remove the const_cast
  auto sorted_data = thrust::device_ptr<Tin>(const_cast<Tin*>(indices_data));
  thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, origin_data, thrust::less<Tin>());

  dim3 grid(CeilDiv(num_indices, (int64_t)4), CeilDiv(stride, (int64_t)128));
  dim3 block(WARP_SIZE, 4);

  _GatherGradImpl<<<grid, block>>>(
      const_cast<Tin*>(indices_data),
      origin_indices,
      const_cast<T*>(grad_data),
      output_data,
      num_indices,
      num_inputs,
      param_itrs,
      stride);
}

#define SPECIALIZED_GRAD_IMPL2(T)                                                                                      \
  template void GatherGradImpl<T, int64_t>(const T* grad_data, const int64_t* indices_data,                            \
                                           const int64_t num_indices, const int64_t num_weights, const int64_t stride, \
                                           int64_t* origin_indices, T* output_data, const int64_t num_inputs,          \
                                           const int64_t params_itrs, ThrustAllocator& allocator);                     \
  template void GatherGradImpl<T, int32_t>(const T* grad_data, const int32_t* indices_data,                            \
                                           const int64_t num_indices, const int64_t num_weights, const int64_t stride, \
                                           int32_t* origin_indices, T* output_data, const int64_t num_inputs,          \
                                           const int64_t params_itrs, ThrustAllocator& allocator);

SPECIALIZED_GRAD_IMPL2(float)
SPECIALIZED_GRAD_IMPL2(half)

}  // namespace cuda
}  // namespace onnxruntime
