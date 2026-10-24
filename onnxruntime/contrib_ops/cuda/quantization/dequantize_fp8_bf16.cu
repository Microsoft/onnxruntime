// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dequantize_fp8_bf16.h"

#if !defined(DISABLE_FLOAT8_TYPES)
namespace onnxruntime::contrib::cuda {
namespace {

__global__ void DequantizeFp8Bf16Kernel(const Float8E4M3FN* input, const float* scale,
                                        const Float8E4M3FN* zero_point, BFloat16* output,
                                        size_t count, size_t axis_stride, size_t scale_count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const size_t scale_index = (index / axis_stride) % scale_count;
    const float zero = zero_point == nullptr ? 0.0f : zero_point[scale_index].ToFloat();
    output[index] = BFloat16((input[index].ToFloat() - zero) * scale[scale_index]);
  }
}

}  // namespace

void DequantizeFp8Bf16(cudaStream_t stream, const Float8E4M3FN* input, const float* scale,
                       const Float8E4M3FN* zero_point, BFloat16* output, size_t count,
                       size_t axis_stride, size_t scale_count) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  DequantizeFp8Bf16Kernel<<<blocks, threads, 0, stream>>>(
      input, scale, zero_point, output,
      count, axis_stride, scale_count);
}

}  // namespace onnxruntime::contrib::cuda
#endif
