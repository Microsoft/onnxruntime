/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// NVIDIA/apex is licensed under the
// BSD 3 - Clause "New" or "Revised" License
//

/* Modifications Copyright (c) Microsoft. */
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include "orttraining/training_ops/cuda/nn/layer_norm_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

using namespace onnxruntime::cuda;

namespace {
  // This is the un-specialized struct.  Note that we prevent instantiation of this
  // struct by putting an undefined symbol in the function body so it won't compile.
  //  template <typename T>
  //  struct SharedMemory
  //  {
  //      // Ensure that we won't compile any un-specialized types
  //      __device__ T *getPointer()
  //      {
  //          extern __device__ void error(void);
  //          error();
  //          return NULL;
  //      }
  //  };
  // https://github.com/NVIDIA/apex/issues/246
  template <typename T>
  struct SharedMemory;
  
  template <>
  struct SharedMemory<float> {
    __device__ float* getPointer() {
      extern __shared__ float s_float[];
      return s_float;
    }
  };
  
  template <>
  struct SharedMemory<double> {
    __device__ double* getPointer() {
      extern __shared__ double s_double[];
      return s_double;
    }
  };
  }  // namespace

template <typename T, typename U, bool use_mean, bool simplified>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* output,
    const T* dout,
    const int i1_end,
    const int n2,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const U* __restrict__ mean,
    const U* __restrict__ invvar) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = (use_mean && !simplified) ? mean[i1] : U(0);
    U curr_invvar = use_mean ? invvar[i1] : U(0);
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf1[write_idx] = curr_dout;
        if (use_mean) {
          U curr_input = static_cast<U>(input[load_idx]);
          warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          U curr_gamma = static_cast<U>(gamma[i2]);
          U curr_beta = static_cast<U>(beta[i2]);
          U curr_output = static_cast<U>(output[load_idx]);
          warp_buf2[write_idx] = curr_dout * (curr_output - curr_beta) / curr_gamma;
        }
      } else {
        warp_buf1[write_idx] = U(0);
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      warp_buf1[write_idx] = U(0);
      warp_buf2[write_idx] = U(0);
    }
  }
}

template <typename T, typename U, bool use_mean, bool simplified>
__device__ void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* output,
    const T* dout,
    const int i1_end,
    const int n2,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const U* __restrict__ mean,
    const U* __restrict__ invvar) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = (use_mean && !simplified) ? mean[i1] : U(0);
    U curr_invvar = use_mean ? invvar[i1] : U(0);
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf1[write_idx] += curr_dout;
        if (use_mean) {
          U curr_input = static_cast<U>(input[load_idx]);
          warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          U curr_gamma = static_cast<U>(gamma[i2]);
          U curr_beta = static_cast<U>(beta[i2]);
          U curr_output = static_cast<U>(output[load_idx]);
          warp_buf2[write_idx] += curr_dout * (curr_output - curr_beta) / curr_gamma;
        }
      }
    }
  }
}

template <typename T, typename U, bool use_mean, bool simplified>
__global__ void cuComputePartGradGammaBeta(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const int n1,
    const int n2,
    U* part_grad_gamma,
    U* part_grad_beta) {
  const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
  const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
  const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
  const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
  const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
  const int row_stride = blockDim.x + 1;
  const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
  const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
  const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
  SharedMemory<U> shared;
  U* buf = shared.getPointer();  // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
  U* warp_buf1 = (U*)buf;
  U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
  // compute partial sums from strided inputs
  // do this to increase number of loads in flight
  cuLoadWriteStridedInputs<T, U, use_mean, simplified>(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, input, output, dout, i1_end, n2, gamma, beta, mean, invvar);
  for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end; i1_block += blockDim.y * blockDim.y) {
    cuLoadAddStridedInputs<T, U, use_mean, simplified>(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, input, output, dout, i1_end, n2, gamma, beta, mean, invvar);
  }
  __syncthreads();
  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < blockDim.y; ++k) {
    int row1 = threadIdx.y + k * blockDim.y;
    int idx1 = row1 * row_stride + threadIdx.x;
    acc1 += warp_buf1[idx1];
    acc2 += warp_buf2[idx1];
  }
  warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      warp_buf1[idx1] += warp_buf1[idx2];
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename T, typename U, bool simplified>
__global__ void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    T* grad_gamma,
    T* grad_beta) {
  // sum partial gradients for gamma and beta
  SharedMemory<U> shared;
  U* buf = shared.getPointer();
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      sum_beta += part_grad_beta_ptr[warp_offset * n2];
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        buf[write_idx + nbsize3] = sum_beta;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
        sum_beta += buf[read_idx + nbsize3];
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
      if (!simplified) {
        grad_beta[i2] = sum_beta;
      }
    }
  }
}

template <typename T, typename U, bool use_mean, bool use_gamma, bool simplified>
__global__ void cuComputeGradInput(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const T* __restrict__ output,
    const T* gamma,
    const T* beta,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const int n1,
    const int n2,
    T* grad_input) {
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = (use_mean && !simplified) ? mean[i1] : U(0);
    const U c_invvar = invvar[i1];
    const T* k_input = use_mean ? input + i1 * n2 : nullptr;
    const T* k_output = use_mean ? nullptr: output + i1 * n2;
    const T* k_dout = dout + i1 * n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (use_gamma) {
#ifndef __HIP_PLATFORM_HCC__
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss * U(gamma[l + k]);
          if (use_mean) {
            const U c_h = static_cast<U>(k_input[l + k]);
            sum_loss2 += c_loss * U(gamma[l + k]) * (c_h - c_mean) * c_invvar;
          } else {
            const U c_output = static_cast<U>(k_output[l + k]);
            sum_loss2 += c_loss * (c_output - U(beta[l + k]));
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss * U(gamma[l]);
        if (use_mean) {
          const U c_h = static_cast<U>(k_input[l]);
          sum_loss2 += c_loss * U(gamma[l]) * (c_h - c_mean) * c_invvar;
        } else {
          const U c_output = static_cast<U>(k_output[l]);
          sum_loss2 += c_loss * (c_output - U(beta[l]));
        }
      }
#else
      // Optimization for ROCm MI100
      using IT = typename std::conditional<sizeof(T)==2, _Float16, T>::type;
      using T2 = IT __attribute__((ext_vector_type(2)));

      for( int l = 0; l < n2 ; l += 2*numx) {
        int idx = l + 2*thrx;
        if( idx+1 < n2) { 
                T2 gamma_idx = *(const T2*)(gamma + idx ); 
                const T2 c_loss = *(const T2*)(k_dout + idx);

                if (use_mean) {
                  const T2 c_h = *(const T2*)( k_input + idx );

                  for(int k = 0; k < 2; k++) {
                      sum_loss1 += U(c_loss[k]) * U( gamma_idx[k] );
                      sum_loss2 += U(c_loss[k]) * U(gamma_idx[k]) * ( U(c_h[k]) - c_mean) * c_invvar;
                  }
                } else {
                  const T2 c_output = *(const T2*)( k_output + idx );
                  const T2 beta_idx = *(const T2*)( beta + idx );
  
                  for(int k = 0; k < 2; k++) {
                      sum_loss1 += U(c_loss[k]) * U( gamma_idx[k] );
                      sum_loss2 += U(c_loss[k]) * ( U(c_output[k]) - U( beta_idx[k] ));
                    }
                }
        } 
        else if( idx < n2) { 
                T gamma_idx = gamma[ idx ]; 
                const U c_loss = static_cast<U>( k_dout[ idx ] );

                if (use_mean) {
                  const U c_h = static_cast<U>( k_input[ idx ] );

                  sum_loss1 += c_loss * U( gamma_idx );
                  sum_loss2 += c_loss * U(gamma_idx) * (c_h - c_mean) * c_invvar;
                } else {
                  const U c_output = static_cast<U>( k_output[idx] );

                  sum_loss1 += c_loss * U( gamma_idx );
                  sum_loss2 += c_loss * (c_output - U( beta[idx] ));
                }
        } 

      }
#endif
    } else {
#ifndef __HIP_PLATFORM_HCC__
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss;
          if (use_mean) {
            const U c_h = static_cast<U>(k_input[l + k]);
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          } else {
            const U c_output = static_cast<U>(k_output[l + k]);
            sum_loss2 += c_loss * c_output;
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        if (use_mean) {
          const U c_h = static_cast<U>(k_input[l]);
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        } else {
          const U c_output = static_cast<U>(k_output[l]);
          sum_loss2 += c_loss * c_output;
        }
      }
#else
      // Optimization for ROCm MI100
      using IT = typename std::conditional<sizeof(T)==2, _Float16, T>::type;
      using T2 = IT __attribute__((ext_vector_type(2)));

      for( int l = 0; l < n2 ; l += 2*numx) {
        int idx = l + 2*thrx;
        if( idx+1 < n2) { 
                const T2 c_loss = *(const T2*)(k_dout + idx);

                if (use_mean) {
                  const T2 c_h = *(const T2*)( k_input + idx );

                  for(int k = 0; k < 2; k++) {
                      sum_loss1 += U(c_loss[k]);
                      sum_loss2 += U(c_loss[k]) * ( U(c_h[k]) - c_mean) * c_invvar;
                  }
                } else {
                  const T2 c_output = *(const T2*)( k_output + idx );
                  const T2 beta_idx = *(const T2*)( beta + idx );
  
                  for(int k = 0; k < 2; k++) {
                      sum_loss1 += U(c_loss[k]);
                      sum_loss2 += U(c_loss[k]) * U(c_output[k]);
                    }
                }
        } 
        else if( idx < n2) { 
                const U c_loss = static_cast<U>( k_dout[ idx ] );

                if (use_mean) {
                  const U c_h = static_cast<U>( k_input[ idx ] );

                  sum_loss1 += c_loss;
                  sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
                } else {
                  const U c_output = static_cast<U>( k_output[idx] );

                  sum_loss1 += c_loss;
                  sum_loss2 += c_loss * c_output;
                }
        }
      } 

#endif
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    // U sum_loss2 = X_mean_difference_over_std_var in cpu kernel
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1 * n2;
    if (use_gamma) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * U(gamma[l]);
        if (!simplified) {
          f_grad_input -= sum_loss1;
        }
        if (use_mean) {
          const U c_h = static_cast<U>(k_input[l]);
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          const U c_output = static_cast<U>(k_output[l]);
          f_grad_input -= (c_output - U(beta[l])) / U(gamma[l]) * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!simplified) {
          f_grad_input -= sum_loss1;
        }
        if (use_mean) {
          const U c_h = static_cast<U>(k_input[l]);
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          const U c_output = static_cast<U>(k_output[l]);
          f_grad_input -= c_output * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
  }
}

template <typename T, typename U, bool simplified>
void HostLayerNormGradient(
  const cudaDeviceProp& prop,
  cudaStream_t stream,
  const T* dout,
  const T* input,
  const T* output,
  const T* gamma,
  const T* beta,
  const U* mean,
  const U* invvar,
  int64_t n1,
  int64_t n2,
  T* grad_input,
  T* grad_gamma,
  T* grad_beta,
  U* part_grad_gamma,
  U* part_grad_beta,
  const int part_size) {
  const int warp_size = prop.warpSize;
  ORT_ENFORCE(warp_size == GPU_WARP_SIZE);

  const dim3 threads2(warp_size, 4, 1);
  const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
  const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
  const int nshared2_b = threads2.x * threads2.y * sizeof(U);
  const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
  if (mean == nullptr && !simplified) {
    // use_mean == false, simplified == false -> Inverted Layer Norm
    cuComputePartGradGammaBeta<T, U, false, false><<<blocks2, threads2, nshared2, stream>>>(
      dout,
      input,
      output,
      gamma,
      beta,
      mean,
      invvar,
      n1, n2,
      part_grad_gamma,
      part_grad_beta);
  } else {
    // use_mean == true, simplified == false -> Layer Norm
    // use_mean == true, simplified == true -> Simplified Layer Norm
    cuComputePartGradGammaBeta<T, U, true, simplified><<<blocks2, threads2, nshared2, stream>>>(
      dout,
      input,
      output,
      gamma,
      beta,
      mean,
      invvar,
      n1, n2,
      part_grad_gamma,
      part_grad_beta);
  }
  const dim3 threads3(warp_size, 8, 1);
  const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
  const int nshared3 = threads3.x * threads3.y * sizeof(U);
  cuComputeGradGammaBeta<T, U, simplified><<<blocks3, threads3, nshared3, stream>>>(
      part_grad_gamma,
      part_grad_beta,
      part_size,
      n1, n2,
      grad_gamma,
      grad_beta);
  // compute grad_input
  const uint64_t maxGridY = prop.maxGridSize[1];
  const dim3 blocks1(1, std::min<unsigned int>(static_cast<unsigned int>(n1), static_cast<unsigned int>(maxGridY)), 1);
  dim3 threads1(warp_size, 4, 1);
#ifdef __HIP_PLATFORM_HCC__
  // Optimization for ROCm MI100
  threads1.y = 2;
#endif
  int nshared =
      threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
  if (mean == nullptr && !simplified) {
    if (gamma == nullptr) {
      cuComputeGradInput<T, U, false, false, false><<<blocks1, threads1, nshared, stream>>>(
        dout,
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        n1, n2,
        grad_input);
    } else {
      cuComputeGradInput<T, U, false, true, false><<<blocks1, threads1, nshared, stream>>>(
        dout,
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        n1, n2,
        grad_input);
    }
  } else {
    if (gamma == nullptr) {
      cuComputeGradInput<T, U, true, false, simplified><<<blocks1, threads1, nshared, stream>>>(
        dout,
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        n1, n2,
        grad_input);
    } else {
      cuComputeGradInput<T, U, true, true, simplified><<<blocks1, threads1, nshared, stream>>>(
        dout,
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        n1, n2,
        grad_input);
    }
  }
}

#define LAYERNORMGRAD_IMPL(T, U, simplified)                                                                                                  \
  template void HostLayerNormGradient<T, U, simplified>(const cudaDeviceProp& prop, cudaStream_t stream, const T* dout, const T* input, const T* output,           \
                                      const T* gamma, const T* beta, const U* mean, const U* invvar, int64_t n1, int64_t n2,                  \
                                      T* grad_input, T* grad_gamma, T* grad_beta, U* part_grad_gamma, U* part_grad_beta, const int part_size);

LAYERNORMGRAD_IMPL(float, float, true)
LAYERNORMGRAD_IMPL(double, double, true)
LAYERNORMGRAD_IMPL(half, float, true)
LAYERNORMGRAD_IMPL(float, float, false)
LAYERNORMGRAD_IMPL(double, double, false)
LAYERNORMGRAD_IMPL(half, float, false)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
LAYERNORMGRAD_IMPL(nv_bfloat16, float, true)
LAYERNORMGRAD_IMPL(nv_bfloat16, float, false)
#endif

}  // namespace cuda
}  // namespace onnxruntime
