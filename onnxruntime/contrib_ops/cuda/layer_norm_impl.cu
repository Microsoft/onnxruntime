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

#include "core/providers/cuda/cu_inc/common.cuh"

#include "layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename U, bool simplified>
__device__ void cuWelfordOnlineSum(
    const U curr,
    U& mu,
    U& sigma2,
    U& count) {
  count = count + U(1);
  U delta = curr - mu;
  mu += delta / count;
  if (simplified) {
    sigma2 = sigma2 + curr * curr;
  } else {
    U delta2 = curr - mu;
    sigma2 = sigma2 + delta * delta2;
  }
}

template <typename U, bool simplified, bool large_count>
__device__ void cuChanOnlineSum(
    const U muB,
    const U sigma2B,
    const U countB,
    U& mu,
    U& sigma2,
    U& count) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  if (count > U(0)) {
    if (large_count) {
      nB = nB / count;
      mu = nA * mu / count + nB * muB;
    } else {
      nB = nB / count;
      mu += delta * nB;
    }
    if (simplified) {
      sigma2 = sigma2 + sigma2B;
    } else {
      sigma2 = sigma2 + sigma2B + delta * delta * nA * nB;
    }
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename T, typename U, bool simplified, bool large_count>
__device__ void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    U& mu,
    U& sigma2,
    U* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        cuWelfordOnlineSum<U, simplified>(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U, simplified>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      U muB = WARP_SHFL_DOWN(mu, stride);
      U countB = WARP_SHFL_DOWN(count, stride);
      U sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum<U, simplified, large_count>(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U, simplified, large_count>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
    }
  }
}

template <bool simplified, bool large_count>
__device__ void cuWelfordMuSigma2(
    const half* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    float& mu,
    float& sigma2,
    float* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const half* lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum<float, simplified>(curr, mu, sigma2, count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2*)(lvals + l + k)));
        cuWelfordOnlineSum<float, simplified>(static_cast<float>(curr.x), mu, sigma2, count);
        cuWelfordOnlineSum<float, simplified>(static_cast<float>(curr.y), mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum<float, simplified>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      float muB = WARP_SHFL_DOWN(mu, stride);
      float countB = WARP_SHFL_DOWN(count, stride);
      float sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum<float, simplified, large_count>(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum<float, simplified, large_count>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);
    }
  }
}

template <typename U>
__device__ U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template <>
__device__ float rsqrt(float v) {
  return rsqrtf(v);
}
template <>
__device__ double rsqrt(double v) {
  return rsqrt(v);
}

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

template <typename T, typename U, bool simplified, bool large_count>
__global__ void cuApplyLayerNorm(
    T* __restrict__ output_vals,
    U* __restrict__ mean,
    U* __restrict__ inv_std_dev,
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const U epsilon,
    const T* __restrict__ gamma,
    const T* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2<T, U, simplified, large_count>(vals, n1, n2, i1, mu, sigma2, buf);
    const T* lvals = vals + i1 * n2;
    T* ovals = output_vals + i1 * n2;
    U c_inv_std_dev = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = thrx; i < n2; i += numx) {
      U curr = static_cast<U>(lvals[i]);
      T gamma_i = (gamma != NULL) ? gamma[i]: (T)1;
      T beta_i = (beta != NULL) ? beta[i] : (T) 0;
      if (simplified) {
        ovals[i] = gamma_i * static_cast<T>(c_inv_std_dev * curr);
      } else {
        ovals[i] = gamma_i * static_cast<T>(c_inv_std_dev * (curr - mu)) + beta_i;
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (mean != nullptr) mean[i1] = mu;
      if (inv_std_dev != nullptr) inv_std_dev[i1] = c_inv_std_dev;
    }
  }
}

template <typename T, typename U, bool simplified>
void HostApplyLayerNorm(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    T* output,
    U* mean,
    U* inv_std_dev,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const T* gamma,
    const T* beta) {
  const int maxGridY = prop.maxGridSize[1];
  const int warp_size = prop.warpSize;
  bool large_count = n2 > 16384;
  ORT_ENFORCE(warp_size == GPU_WARP_SIZE);

  dim3 threads(warp_size, 4, 1);
#ifdef __HIP_PLATFORM_HCC__
  // Optimization for ROCm MI100
  threads.y = 1;
#endif
  const dim3 blocks(1, std::min<unsigned int>(n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  if (large_count) {
    cuApplyLayerNorm<T, U, simplified, true><<<blocks, threads, nshared, stream>>>(
        output,
        mean,
        inv_std_dev,
        input,
        n1, n2,
        U(epsilon),
        gamma, beta);
  } else {
    cuApplyLayerNorm<T, U, simplified, false><<<blocks, threads, nshared, stream>>>(
        output,
        mean,
        inv_std_dev,
        input,
        n1, n2,
        U(epsilon),
        gamma, beta);
  }
}

#define LAYERNORM_LINEAR_IMPL(T, U, simplified)                                                                                                 \
  template void HostApplyLayerNorm<T, U, simplified>(const cudaDeviceProp& prop, cudaStream_t stream, T* output, U* mean, U* inv_std_dev, const T* input, int n1, int n2, \
                                                     double epsilon, const T* gamma, const T* beta);

LAYERNORM_LINEAR_IMPL(float, float, true)
LAYERNORM_LINEAR_IMPL(half, float, true)
LAYERNORM_LINEAR_IMPL(double, double, true)
LAYERNORM_LINEAR_IMPL(float, float, false)
LAYERNORM_LINEAR_IMPL(half, float, false)
LAYERNORM_LINEAR_IMPL(double, double, false)

//LAYERNORM_LINEAR_IMPL(half, half)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
LAYERNORM_LINEAR_IMPL(nv_bfloat16, float, true)
LAYERNORM_LINEAR_IMPL(nv_bfloat16, float, false)
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
