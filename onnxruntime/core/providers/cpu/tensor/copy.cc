//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/copy.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

template <typename T>
void print_vec(const std::vector<T>& vec) {
  std::cout << "[ ";
  for (auto& v : vec) {
    std::cout << v << " ";
  }
  std::cout << "]" << std::endl;
}

template <typename T>
void StridedCopy(concurrency::ThreadPool* thread_pool,
                 T* dst,
                 const std::vector<int64_t> dst_shape,
                 const std::vector<int64_t> dst_strides,
                 T* src,
                 const std::vector<int64_t> src_strides) {
  const size_t dims = dst_shape.size();
  // We will iterate over the output dimensions
  int64_t num_iterations = 1;
  for (size_t dim = 0; dim < dims; dim++) {
    num_iterations *= dst_shape[dim];
  }

  // TODO(orausch): Reorder dimensions so that we iterate along the smallest strides first
  // TODO(orausch): remove single size dimensions
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, num_iterations,
      {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
      [dst_shape, dst_strides, dst, src, src_strides, dims](std::ptrdiff_t first, std::ptrdiff_t last) {
        // Compute the initial n-dimensional index and addresses
        std::vector<int64_t> current_nd_idx(dims);
        {
          size_t current_index = first;
          for (size_t dim = dims; dim > 0; dim--) {
            // Iterate from dims to 1 so we don't roll over to positive on the bounds check
            current_nd_idx[dim - 1] = current_index % dst_shape[dim - 1];
            current_index /= dst_shape[dim - 1];
          }
        }
        for (std::ptrdiff_t outer_i = first; outer_i < last;) {
          // Compute the src and dst addresses
          size_t dst_idx = 0;
          size_t src_idx = 0;
          for (size_t dim = 0; dim < dims; dim++) {
            dst_idx += current_nd_idx[dim] * dst_strides[dim];
            src_idx += current_nd_idx[dim] * src_strides[dim];
          }
          // 1d vectorizable inner loop along last dimension
          std::ptrdiff_t inner_end = std::min(last, outer_i + dst_shape[dims - 1] - current_nd_idx[dims - 1]);
          for (std::ptrdiff_t i = outer_i; i < inner_end; i++) {
            dst[dst_idx] = src[src_idx];
            dst_idx += dst_strides[dims - 1];
            src_idx += src_strides[dims - 1];
          }
          current_nd_idx[dims - 1] += inner_end - outer_i;

          outer_i = inner_end;

          // update the current_nd_idx if needed
          size_t dim = dims - 1;
          while (dim > 0 && current_nd_idx[dim] >= dst_shape[dim]) {
            current_nd_idx[dim] = 0;
            dim--;
            current_nd_idx[dim]++;
          }
        }
      });
}

#define STRIDED_COPY_IMPL(T)                                          \
  template void StridedCopy<T>(concurrency::ThreadPool * thread_pool, \
                               T * dst,                               \
                               std::vector<int64_t> dst_shape,        \
                               std::vector<int64_t> dst_strides,      \
                               T * src,                               \
                               std::vector<int64_t> src_strides);

STRIDED_COPY_IMPL(int32_t)
STRIDED_COPY_IMPL(int64_t)
STRIDED_COPY_IMPL(float)
STRIDED_COPY_IMPL(double)
}  // namespace onnxruntime
