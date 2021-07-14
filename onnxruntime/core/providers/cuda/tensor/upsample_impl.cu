// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "upsample_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int RANK>
__global__ void _UpampleNearestKernel(const TArray<int64_t> input_pitches,
                                      const TArray<fast_divmod> output_div_pitches,
                                      const TArray<fast_divmod> scales_div,
                                      const T* __restrict__ input_data,
                                      T* __restrict__ output_data,
                                      const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  int div, mod;
  for (int dim = 0; dim < RANK; ++dim) {
    output_div_pitches[dim].divmod(output_index, div, mod);
    output_index = mod;
    if (scales_div[dim].d_ != 1 && div > 0) {
      scales_div[dim].divmod(div, div, mod); 
    }
    input_index += input_pitches[dim] * div;
  }
  output_data[id] = input_data[input_index];
}

// The following method supports a 4-D input in 'Linear mode' 
// that amounts to 'Bilinear' Upsampling/Resizing in the sense that it assumes
// the scale values for the outermost 2 dimensions are 1.
// This is the common use-case where the 4-D input (batched multi-channel images) 
// is usually of shape [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
template <typename T>
__global__ void _UpampleBilinear4DInputKernel(const int64_t input_dim2,
                                       const TArray<int64_t> input_pitches,
                                       const TArray<fast_divmod> output_div_pitches,
                                       const TArray<fast_divmod> scales_div,
                                       const T* __restrict__ input_data,
                                       T* __restrict__ output_data,
                                       const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;

  // For bilinear mode, scales[0]=scales[1]=1
  int mod;
  int index_of_dim0, index_of_dim1, index_of_dim2, index_of_dim3;
  output_div_pitches[0].divmod(id, index_of_dim0, mod);
  output_div_pitches[1].divmod(mod, index_of_dim1, mod);
  output_div_pitches[2].divmod(mod, index_of_dim2, mod);
  index_of_dim3 = mod;
  int index_of_input_dim2, index_of_input_dim3, x_offset, y_offset;
  scales_div[2].divmod(index_of_dim2, index_of_input_dim2, y_offset);
  scales_div[3].divmod(index_of_dim3, index_of_input_dim3, x_offset);

  input_index = index_of_dim0 * input_pitches[0] +
                index_of_dim1 * input_pitches[1] +
                index_of_input_dim2 * input_pitches[2] +
                index_of_input_dim3;
  
  T x00 = input_data[input_index];
  T x10, x01, x11;

  bool end_of_dim2 = false;
  if (index_of_input_dim2 == (input_dim2 - 1)) {
    // It's the end in dimension 2
    x01 = x00;
    end_of_dim2 = true;
  } else {
    x01 = input_data[input_index + input_pitches[2]];
  }

  if (index_of_input_dim3 == (input_pitches[2] - 1)) {
    // It's the end in dimension 3
    x10 = x00;
    x11 = x01;
  }
  else {
    x10 = input_data[input_index + 1];
    x11 = end_of_dim2 ? x10 : input_data[input_index + input_pitches[2] + 1];
  }

  T y_offset_T = static_cast<T>(y_offset);
  T x_offset_T = static_cast<T>(x_offset);
  T scales_div2_T = static_cast<T>(scales_div[2].d_);
  T scales_div3_T = static_cast<T>(scales_div[3].d_);
  T y0 = x00 + static_cast<T>(y_offset_T * (x01 - x00) / scales_div2_T);
  T y1 = x10 + static_cast<T>(y_offset_T * (x11 - x10) / scales_div2_T);

  output_data[id] = y0 + static_cast<T>(x_offset_T * (y1 - y0) / scales_div3_T);
}

// The following method supports a 2-D input in 'Linear mode'
template <typename T>
__global__ void _UpampleBilinear2DInputKernel(const int64_t input_dim0,
                                              const TArray<int64_t> input_pitches,
                                              const TArray<fast_divmod> output_div_pitches,
                                              const TArray<fast_divmod> scales_div,
                                              const T* __restrict__ input_data,
                                              T* __restrict__ output_data,
                                              const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;

  int mod;
  int index_of_dim0, index_of_dim1;
  output_div_pitches[0].divmod(id, index_of_dim0, mod);
  index_of_dim1 = mod;
  int index_of_input_dim0, index_of_input_dim1, x_offset, y_offset;
  scales_div[0].divmod(index_of_dim0, index_of_input_dim0, y_offset);
  scales_div[1].divmod(index_of_dim1, index_of_input_dim1, x_offset);

  input_index = index_of_input_dim0 * input_pitches[0] + index_of_input_dim1;

  T x00 = input_data[input_index];
  T x10, x01, x11;

  bool end_of_dim0 = false;
  if (index_of_input_dim0 == (input_dim0 - 1)) {
    // It's the end in dimension 0
    x01 = x00;
    end_of_dim0 = true;
  } else {
    x01 = input_data[input_index + input_pitches[0]];
  }

  if (index_of_input_dim1 == (input_pitches[0] - 1)) {
    // It's the end in dimension 1
    x10 = x00;
    x11 = x01;
  } else {
    x10 = input_data[input_index + 1];
    x11 = end_of_dim0 ? x10 : input_data[input_index + input_pitches[0] + 1];
  }

  T y_offset_T = static_cast<T>(y_offset);
  T x_offset_T = static_cast<T>(x_offset);
  T scales_div0_T = static_cast<T>(scales_div[0].d_);
  T scales_div1_T = static_cast<T>(scales_div[1].d_);
  T y0 = x00 + static_cast<T>(y_offset_T * (x01 - x00) / scales_div0_T);
  T y1 = x10 + static_cast<T>(y_offset_T * (x11 - x10) / scales_div0_T);

  output_data[id] = y0 + static_cast<T>(x_offset_T * (y1 - y0) / scales_div1_T);
}

template <typename T>
void UpampleImpl(cudaStream_t stream,
                 const onnxruntime::UpsampleMode upsample_mode,
                 const size_t rank,
                 const int64_t input_dim2,
                 const TArray<int64_t>& input_pitches,
                 const TArray<fast_divmod>& output_div_pitches,
                 const TArray<fast_divmod>& scales_div,
                 const T* input_data,
                 T* output_data,
                 const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  if (onnxruntime::UpsampleMode::NN == upsample_mode) {
    if (rank == 4) {
      _UpampleNearestKernel<T,4><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else if (rank == 3) {
      _UpampleNearestKernel<T,3><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else if (rank == 2) {
      _UpampleNearestKernel<T,2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else if (rank == 1) {
      _UpampleNearestKernel<T,1><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else {
      ORT_THROW("Unsupported rank by the Upsample CUDA kernel");
    }
  } else if (onnxruntime::UpsampleMode::LINEAR) {
    if (rank == 4) {
      _UpampleBilinear4DInputKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_dim2, input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else if (rank == 2) {
      _UpampleBilinear2DInputKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_dim2, input_pitches, output_div_pitches, scales_div,
          input_data, output_data, N);
    } else {
      ORT_THROW("Unsupported rank by the Upsample CUDA kernel");
    }
  }
}

#define SPECIALIZED_IMPL(T)                                                     \
  template void UpampleImpl<T>(cudaStream_t stream,                       \
                               const onnxruntime::UpsampleMode upsample_mode,   \
                               const size_t rank,                               \
                               const int64_t input_dim2,                        \
                               const TArray<int64_t>& input_pitches,                    \
                               const TArray<fast_divmod>& output_div_pitches,           \
                               const TArray<fast_divmod>& scales_div,                   \
                               const T* input_data,                             \
                               T* output_data,                                  \
                               const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime
