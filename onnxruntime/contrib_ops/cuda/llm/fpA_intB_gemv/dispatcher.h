/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/details.h"
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

#include <atomic>
#include <cstdio>
#include <type_traits>

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// SM count of the current device if it is sm_12x (consumer / workstation Blackwell), else 0.
inline int GemvPickSmCount() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }
  static constexpr int kMaxDevices = 16;
  static std::atomic<int> cache[kMaxDevices];  // 0 = unknown, -1 = not sm_12x
  int count = device >= 0 && device < kMaxDevices ? cache[device].load(std::memory_order_relaxed) : 0;
  if (count == 0) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
      return 0;
    }
    if (major != 12) {
      count = -1;
    }
    if (device >= 0 && device < kMaxDevices) {
      cache[device].store(count, std::memory_order_relaxed);
    }
  }
  return count > 0 ? count : 0;
}

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// This code are only relevant for CUDA architectures where the fpA_intB_gemv is intended to run (sm_75 and above).
// Therefore, we conditionally compile this block only when __CUDA_ARCH__ >= 750.
// This prevents compilation errors that half2 requires target sm_53 or higher.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)

template <typename DetailsA>
struct MathWrapper {
};

template <>
struct MathWrapper<FP16DetailsA> {
  using Type = typename FP16DetailsA::Type;
  using Type2 = typename FP16DetailsA::Type2;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) {
    return __half2half2(v);
  }

  __device__ __forceinline__ static float2 to_float2(Type2 const& v) {
    return __half22float2(v);
  }

  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) {
    return __hmul2(a, b);
  }
};

template <>
struct MathWrapper<BF16DetailsA> {
  using Type = typename BF16DetailsA::Type;
  using Type2 = typename BF16DetailsA::Type2;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __bfloat162bfloat162(v);
#else
    uint32_t val = 0;
    Type2 ret = reinterpret_cast<Type2&>(val);
    return ret;
#endif
  }

  __device__ __forceinline__ static float2 to_float2(Type2 const& v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __bfloat1622float2(v);
#else
    return float2{0.f, 0.f};
#endif
  }

  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hfma2(a, b, c);
#else
    return to_vec2(static_cast<Type>(0.f));
#endif
  }

  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmul2(a, b);
#else
    return to_vec2(static_cast<Type>(0.f));
#endif
  }
};

template <typename Details, int M, int K, bool Enable>
__device__ __forceinline__ void apply_scale(void* act, void* act_scale) {
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  static_assert(K % 2 == 0);
  [[maybe_unused]] static constexpr int VecK = K / 2;
  if constexpr (Enable) {
    Type2* pa = reinterpret_cast<Type2*>(act);
    Type2* pb = reinterpret_cast<Type2*>(act_scale);
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int k = 0; k < VecK; ++k) {
        pa[m * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::mul2(pa[m * VecK + k], pb[k]);
      }
    }
  }
}

template <typename Details, int N, int K, bool EnableZero, bool ApplyAlphaInAdvance>
__device__ __forceinline__ void dequantize(void* w, void* quantized_w, void* scales, void* zeros, float alpha) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  using Converter = typename ConverterWrapper<Details>::Converter;
  static_assert(K % 2 == 0);
  static constexpr int VecK = K / 2;
#pragma unroll
  for (int n = 0; n < N; ++n) {
    Converter::convert<K>(reinterpret_cast<uint8_t*>(quantized_w) + n * K / Details::kElemsPerByteW,
                          reinterpret_cast<Type*>(w) + n * K);
    Type2 vec_scale, vec_zero;
    if constexpr (ApplyAlphaInAdvance) {
      // For W4A8, we assume scales/zero is always half data type, no matter activation dtype is bf16 or fp16
      Type scales_ = static_cast<float>(reinterpret_cast<half*>(scales)[n]) * alpha;
      vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(scales_);
      vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
      if constexpr (EnableZero) {
        vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(
            static_cast<float>(reinterpret_cast<half*>(zeros)[n]) * alpha);
      }
    } else {
      vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(scales)[n]);
      vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
      if constexpr (EnableZero) {
        vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(zeros)[n]);
      }
    }
#pragma unroll
    for (int k = 0; k < VecK; ++k) {
      reinterpret_cast<Type2*>(w)[n * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::fma2(
          reinterpret_cast<Type2*>(w)[n * VecK + k], vec_scale, vec_zero);
    }
  }
}

template <typename Details, int K>
__device__ __forceinline__ void pack_to_vec2(void* dst, void* src, int n) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  typename Details::LayoutDetails::Mapper mapper;
  int n0 = n & ~0x1, n1 = n & 0x1;
  for (int k = 0; k < K; ++k) {
    int physical_idx = mapper(k);
    reinterpret_cast<Type*>(dst)[n0 * K + k * 2 + n1] = reinterpret_cast<Type*>(src)[physical_idx];
  }
}

template <typename Details, int M, int N, int K,
          typename AccT = typename MathWrapper<typename Details::TypeDetailsA>::Type>
__device__ __forceinline__ void mma(void* acc, void* w_pack2, void* act) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  static_assert(N % 2 == 0);
  static constexpr int VecN = N / 2;
  if constexpr (std::is_same_v<AccT, float>) {
    // fp32 accumulation: keep the per-thread K-dimension dot product in float to avoid
    // the precision loss of accumulating a long chain in 16-bit (especially bf16).
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int n = 0; n < VecN; ++n) {
        float2 acc2 = reinterpret_cast<float2*>(acc)[m * VecN + n];
#pragma unroll
        for (int k = 0; k < K; ++k) {
          float const a = static_cast<float>(reinterpret_cast<Type*>(act)[m * K + k]);
          float2 const w = MathWrapper<typename Details::TypeDetailsA>::to_float2(
              reinterpret_cast<Type2*>(w_pack2)[n * K + k]);
          acc2.x += w.x * a;
          acc2.y += w.y * a;
        }
        reinterpret_cast<float2*>(acc)[m * VecN + n] = acc2;
      }
    }
  } else {
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int n = 0; n < VecN; ++n) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
          reinterpret_cast<Type2*>(acc)[m * VecN + n] = MathWrapper<typename Details::TypeDetailsA>::fma2(
              reinterpret_cast<Type2*>(w_pack2)[n * K + k],
              MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(act)[m * K + k]),
              reinterpret_cast<Type2*>(acc)[m * VecN + n]);
        }
      }
    }
  }
}

// Reduces the per-lane partial dot product across the lanes that share an output column.
//
// A lane's column is ((tid / ThreadsPerInterleavedTile) % Interleave), so the lane-id bits in
// [ThreadsPerInterleavedTile, ThreadsPerInterleavedTile * Interleave) select the column and must
// NOT be summed across; every other bit is a K offset and must be. Descending masks keep the
// float accumulation order of the hand-written Interleave in {1,2,4} sequence this replaces.
template <int Interleave, int ThreadsPerInterleavedTile, typename T>
__device__ __forceinline__ T warp_reduce_sum(T& val) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    if (mask < ThreadsPerInterleavedTile || mask >= ThreadsPerInterleavedTile * Interleave) {
      val += __shfl_xor_sync(~0, val, mask);
    }
  }
  return val;
}

template <typename Details, int CtaM, int CtaN, int Threads, bool EnableBias, bool ApplyAlphaInAdvance,
          typename AccT = typename MathWrapper<typename Details::TypeDetailsA>::Type>
__device__ __forceinline__ void epilogue(void* out, int stride, void* tile_acc, void* bias, float alpha) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(Threads % WarpSize == 0);
  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int n = 0; n < CtaN; ++n) {
      float v = static_cast<float>(reinterpret_cast<AccT*>(tile_acc)[m * CtaN + n]);
      v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
      if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
        shmem[warp_id * CtaM * CtaN * Interleave + m * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int ii = tid; ii < CtaM * CtaN * Interleave; ii += Threads) {
    int m = ii / (CtaN * Interleave), n = ii % (CtaN * Interleave);
    float val = 0.f, v_bias = 0.f;
    if constexpr (EnableBias) {
      v_bias = static_cast<float>(reinterpret_cast<Type*>(bias)[n]);
    }
#pragma unroll
    for (int jj = 0; jj < WarpNum; ++jj) {
      val += shmem[jj * CtaM * CtaN * Interleave + ii];
    }
    if constexpr (ApplyAlphaInAdvance) {
      reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(val + v_bias);
    } else {
      reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(alpha * val + v_bias);
    }
  }
}

template <int N, typename T>
__device__ __forceinline__ void fill(void* tile, T v) {
#pragma unroll
  for (int ii = 0; ii < N; ++ii) {
    reinterpret_cast<T*>(tile)[ii] = v;
  }
}
#endif  // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
          bool EnableBias, bool ApplyAlphaInAdvance, typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* scales, TypeA* zeros, TypeA* bias,
                       TypeA* out, float alpha, int m, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  // ArgType          ArgName          DataType           Shape                 Layout
  // input            act              fp16/bf16          [m, k]                RowMajor
  // input            act_scale        fp16/bf16          [1, k]                RowMajor
  // input            weight           int4b/int8b        [k, n]                ColumnMajor or ColumnMajorInterleaved
  // input            scales           fp16/bf16          [k / GroupSize, n]    RowMajor
  // input            zeros            fp16/bf16          [k / GroupSize, n]    RowMajor
  // input            bias             fp16/bf16          [1, n]                RowMajor
  // output           out              fp16/bf16          [m, n]                RowMajor

  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
    // One scale per thread covers its whole StepK run of weights.
    static_assert(GroupSize % StepK == 0);
  }

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = blockIdx.x, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave + ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize + ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<EnableActScale, AccessTypeA, 1, Details::kAccessNumA, TypeA> act_scale_iterator(
      act_scale, real_offset_k, CtaK / Details::kInterleave, 0);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight,
      (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
      interleaved_k / Details::kElemsPerByteW);

  // Kept as CtaN scalar loads rather than the vectorized ScalesAccess/load_scales form used by
  // the FP4 MoE GEMV: that form needs kInterleave == 1 to make the CtaN scales contiguous, and
  // every layout reaching this dense kernel is ColumnMajorInterleaved (kInterleave 2, 4 or 8).
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  GMemIterator<EnableZero, TypeA, CtaN, 1, TypeA> zeros_iterator(
      zeros,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  TypeA tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA vec_act_scale[StepK];
    TypeA vec_scale[CtaN], vec_zero[CtaN];
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, iter, i);
      zeros_iterator.load(vec_zero + i, iter, i);
    }
    act_scale_iterator.load(vec_act_scale, iter);
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, EnableZero, ApplyAlphaInAdvance>(
          tile_w, tile_w_quantized, vec_scale + i, vec_zero + i, alpha);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      apply_scale<Details, 1, StepK, EnableActScale>(tile_a, vec_act_scale);
      mma<Details, 1, CtaN, StepK>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, ApplyAlphaInAdvance>(out, n, tile_acc, bias, alpha);
#endif
}

// Physical half2 index, in the converter's output order, of the activation half2 pair `q` (logical K
// elements 2q and 2q + 1). The layout mapper keeps element pairs adjacent (group size is even).
template <typename Details>
__host__ __device__ constexpr int PairedPhysicalPair(int q) {
  constexpr int g = Details::LayoutDetails::kElementGroupSizeA;
  constexpr int w = Details::LayoutDetails::kElementGroupSizeW;
  constexpr int off = Details::LayoutDetails::kGroupOffsetA;
  static_assert(g % 2 == 0, "K pairs must stay adjacent under the layout mapper");
  int const i = 2 * q;
  return (i % g + (i % off) / g * w + i / off * g) / 2;
}

// fp16 variant of `kernel` that multiplies converter-order weight pairs (K, K+1) by the activation pairs
// directly. It needs no scalar repack of the weights and no broadcast of the activations, which removes
// about a third of the per-tile instructions. Each accumulator half2 holds two partial K sums that are added
// in float at the end. Scale-only (no zero point, bias or activation scale) kernels only.
template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize>
__global__ void kernel_paired(half* act, uint8_t* weight, half* scales, half* out, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr int StepK = Details::kStepK;
  static constexpr int NumPairs = StepK / 2;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  static_assert(GroupSize % StepK == 0);

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = blockIdx.x, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave + ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize + ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<true, AccessTypeA, CtaM, Details::kAccessNumA, half> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<true, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight,
      (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
      interleaved_k / Details::kElemsPerByteW);
  GMemIterator<true, half, CtaN, 1, half> scales_iterator(
      scales, real_offset_k / GroupSize * n + real_offset_n, CtaK / Details::kInterleave / GroupSize * n,
      Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;

  half2 acc[CtaM][CtaN];
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int j = 0; j < CtaN; ++j) {
      acc[m][j] = __half2half2(__float2half(0.f));
    }
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    half vec_scale[CtaN];
    half2 w2[CtaN][NumPairs];
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, iter, i);
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      uint8_t quantized[StepK / Details::kElemsPerByteW];
      half tile_w[StepK];
      weight_iterator.load(quantized, iter, i);
      ConverterWrapper<Details>::Converter::template convert<StepK>(quantized, tile_w);
      half2 const scale2 = __half2half2(vec_scale[i]);
#pragma unroll
      for (int p = 0; p < NumPairs; ++p) {
        w2[i][p] = __hmul2(reinterpret_cast<half2*>(tile_w)[p], scale2);
      }
    }
#pragma unroll
    for (int m = 0; m < CtaM; ++m) {
      half tile_a[StepK];
      act_iterator.load(tile_a, iter, m);
#pragma unroll
      for (int j = 0; j < CtaN; ++j) {
#pragma unroll
        for (int q = 0; q < NumPairs; ++q) {
          acc[m][j] = __hfma2(w2[j][PairedPhysicalPair<Details>(q)], reinterpret_cast<half2*>(tile_a)[q], acc[m][j]);
        }
      }
    }
  }

  half tile_acc[CtaM * CtaN];
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int j = 0; j < CtaN; ++j) {
      tile_acc[m * CtaN + j] = __float2half(__low2float(acc[m][j]) + __high2float(acc[m][j]));
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, false, false>(out, n, tile_acc, nullptr, 1.f);
#endif
}

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize>
void exec_kernel_paired(Params& params, cudaStream_t s) {
  if (params.m % CtaM || params.n % (CtaN * Details::kInterleave)) {
    ORT_THROW("launch failed");
  }
  dim3 grid(params.m / CtaM, params.n / (CtaN * Details::kInterleave));
  dim3 block(Threads);
  kernel_paired<Details, CtaM, CtaN, Threads, GroupSize><<<grid, block, 0, s>>>(
      reinterpret_cast<half*>(params.act),
      reinterpret_cast<uint8_t*>(params.weight),
      reinterpret_cast<half*>(params.scales),
      reinterpret_cast<half*>(params.out),
      params.n, params.k);
  if (params.debug) {
    std::printf("[fpA_intB_debug] GEMV launch: paired_k=1 M=%d\n", params.m);
    std::fflush(stdout);
  }
}

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
          bool EnableBias, bool ApplyAlphaInAdvance>
void exec_kernel(Params& params, cudaStream_t s) {
  using T = typename Details::TypeDetailsA::Type;
  if (params.m % CtaM || params.n % (CtaN * Details::kInterleave)) {
    ORT_THROW("launch failed");
  }
  dim3 grid(params.m / CtaM, params.n / (CtaN * Details::kInterleave));
  dim3 block(Threads);
  kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, ApplyAlphaInAdvance><<<grid, block, 0, s>>>(
      reinterpret_cast<T*>(params.act),
      reinterpret_cast<T*>(params.act_scale),
      reinterpret_cast<uint8_t*>(params.weight),
      reinterpret_cast<T*>(params.scales),
      reinterpret_cast<T*>(params.zeros),
      reinterpret_cast<T*>(params.bias),
      reinterpret_cast<T*>(params.out),
      params.alpha,
      params.m, params.n, params.k);
  if (params.debug) {
    std::printf("[fpA_intB_debug] GEMV launch: paired_k=0 M=%d\n", params.m);
    std::fflush(stdout);
  }
}

template <typename Details, int GroupSize, bool EnableActScale, bool EnableZero, bool EnableBias, bool ApplyAlphaInAdvance>
void dispatcher(Params& params, cudaStream_t s) {
  // Each block keeps a CtaN * StepK half tile of dequantized weights in registers, so CtaN is
  // halved for the 2-bit layout, whose StepK is 64 instead of the 4/8-bit 32/16. That holds the
  // register footprint (and the columns covered per block, CtaN * kInterleave) at the value the
  // 4-bit kernel was tuned for.
  static constexpr int CtaN = Details::kStepK >= 64 ? (EnableZero ? 2 : 4) : (EnableZero ? 4 : 8);
  // From M = 4 on, the M x StepK activation tile and the CtaM x CtaN accumulators push the 4/8-bit
  // kernel to ~250 registers (two 128-thread blocks per SM), and the N = 5120 decode projections
  // then launch only 160 blocks. Halving CtaN frees registers and doubles the block count; on
  // RTX 4090 it makes the M = 4..8 GEMVs 2-12% faster with DRAM-resident weights. The 2-bit layout
  // already uses the narrow tile.
  static constexpr int CtaNLargeM = Details::kStepK >= 64 ? CtaN : (CtaN / 2 < 2 ? 2 : CtaN / 2);
  // For single-row INT4 decode with group size 32 and no explicit zero points, use a narrower
  // N tile to launch more blocks and more threads per block to parallelize the K reduction.
  // Other configurations retain the default tile and 128-thread launch.
  static constexpr bool NarrowInt4Decode = !EnableZero && Details::kElemsPerByteW == 2 && GroupSize == 32;
  static constexpr int CtaNDecode = NarrowInt4Decode ? 2 : CtaN;
  static constexpr int DecodeThreads = NarrowInt4Decode ? 256 : 128;

  // Paired-K kernel (fp16, int4, scale-only, SM80-interleaved layout). It is only requested through the
  // profiler's optional tactic, and covers the M = 5..8 range that one profiled M bucket serves.
  if constexpr (Details::kStepK == 32 && Details::kInterleave == 4 && !EnableZero && !EnableBias &&
                !EnableActScale && !ApplyAlphaInAdvance && CtaNLargeM == 4 &&
                std::is_same_v<typename Details::TypeDetailsA, FP16DetailsA>) {
    if (params.paired_k && params.m >= 5 && params.m <= 8) {
      if (params.m == 8) {
        exec_kernel_paired<Details, 8, CtaNLargeM, 128, GroupSize>(params, s);
      } else if (params.m == 7) {
        exec_kernel_paired<Details, 7, CtaNLargeM, 128, GroupSize>(params, s);
      } else if (params.m == 6) {
        exec_kernel_paired<Details, 6, CtaNLargeM, 128, GroupSize>(params, s);
      } else {
        exec_kernel_paired<Details, 5, CtaNLargeM, 128, GroupSize>(params, s);
      }
      return;
    }
  }

#define DISPATCHER_FOR_M(target_m, CtaM, TileN, Threads)                                            \
  do {                                                                                              \
    if (params.m == target_m) {                                                                     \
      exec_kernel<Details, CtaM, TileN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, \
                  ApplyAlphaInAdvance>(params, s);                                                  \
      return;                                                                                       \
    }                                                                                               \
  } while (0);

  if (params.decode_variant != 0) {
    if constexpr (NarrowInt4Decode && Details::kStepK == 32 && Details::kInterleave == 4 &&
                  !EnableActScale && !EnableBias && !ApplyAlphaInAdvance) {
      ORT_ENFORCE(params.m == 1 && IsInt4DecodeGeometryLegal(params.decode_variant, params.n, params.k,
                                                             Details::kInterleave),
                  "Unsupported INT4 decode geometry");
      switch (params.decode_variant) {
        case 2:
          DISPATCHER_FOR_M(1, 1, 2, 128);
          break;
        case 3:
          DISPATCHER_FOR_M(1, 1, 2, 256);
          break;
        case 4:
          DISPATCHER_FOR_M(1, 1, 4, 128);
          break;
        case 5:
          DISPATCHER_FOR_M(1, 1, 4, 256);
          break;
        case 6:
          DISPATCHER_FOR_M(1, 1, 8, 128);
          break;
        case 7:
          DISPATCHER_FOR_M(1, 1, 8, 256);
          break;
      }
    } else {
      ORT_THROW("INT4 decode geometry requires the symmetric group32 interleave4 layout");
    }
  }

  if constexpr (NarrowInt4Decode) {
    if (params.m == 1) {
      if (params.n >= 64 && params.n <= 8192 && params.k >= 64 && params.k <= 4096 &&
          (params.n <= 256 || params.k >= 2560) &&
          params.n % (CtaNDecode * Details::kInterleave) == 0 &&
          params.n / (CtaNDecode * Details::kInterleave) <= 65535) {
        exec_kernel<Details, 1, CtaNDecode, DecodeThreads, GroupSize, EnableActScale, EnableZero, EnableBias,
                    ApplyAlphaInAdvance>(params, s);
        return;
      }
      exec_kernel<Details, 1, CtaN, 128, GroupSize, EnableActScale, EnableZero, EnableBias,
                  ApplyAlphaInAdvance>(params, s);
      return;
    }
  }
  DISPATCHER_FOR_M(1, 1, CtaNDecode, DecodeThreads);
  DISPATCHER_FOR_M(2, 2, CtaN, 128);
  DISPATCHER_FOR_M(3, 3, CtaN, 128);
  DISPATCHER_FOR_M(4, 4, CtaNLargeM, 128);
  DISPATCHER_FOR_M(5, 5, CtaNLargeM, 128);
  DISPATCHER_FOR_M(6, 6, CtaNLargeM, 128);
  DISPATCHER_FOR_M(7, 7, CtaNLargeM, 128);
  // M = 8 is the DFlash2 verify batch; its projections are sensitive to wave quantization. Only the
  // fp16 int4 SM80-interleaved kernel was measured, so other types and layouts keep CtaNLargeM.
  if constexpr (Details::kStepK == 32 && Details::kInterleave == 4 && !EnableZero && CtaNLargeM == 4 &&
                std::is_same_v<typename Details::TypeDetailsA, FP16DetailsA>) {
    if (params.m == 8 && params.n / (CtaNLargeM * Details::kInterleave) <= 320 &&
        params.n % ((CtaNLargeM / 2) * Details::kInterleave) == 0) {
      exec_kernel<Details, 8, CtaNLargeM / 2, 128, GroupSize, EnableActScale, EnableZero, EnableBias,
                  ApplyAlphaInAdvance>(params, s);
      return;
    }
    int const sm_count = params.wave_aware && params.m == 8 ? GemvPickSmCount() : 0;
    int const pick = PickGemvCtaN(params.wave_aware, params.m, params.n, Details::kInterleave, CtaNLargeM,
                                  sm_count);
    if (pick == CtaNLargeM / 2) {
      exec_kernel<Details, 8, CtaNLargeM / 2, 128, GroupSize, EnableActScale, EnableZero, EnableBias,
                  ApplyAlphaInAdvance>(params, s);
      return;
    }
    if (pick == CtaNLargeM * 2) {
      exec_kernel<Details, 8, CtaNLargeM * 2, 128, GroupSize, EnableActScale, EnableZero, EnableBias,
                  ApplyAlphaInAdvance>(params, s);
      return;
    }
  }
  DISPATCHER_FOR_M(8, 8, CtaNLargeM, 128);
  DISPATCHER_FOR_M(9, 9, CtaNLargeM, 128);
  DISPATCHER_FOR_M(10, 10, CtaNLargeM, 128);
  DISPATCHER_FOR_M(11, 11, CtaNLargeM, 128);
  DISPATCHER_FOR_M(12, 12, CtaNLargeM, 128);
  DISPATCHER_FOR_M(13, 13, CtaNLargeM, 128);
  DISPATCHER_FOR_M(14, 14, CtaNLargeM, 128);
  DISPATCHER_FOR_M(15, 15, CtaNLargeM, 128);
  ORT_THROW("unsupported m");
#undef DISPATCHER_FOR_M
}

template <typename Details, int GroupSize>
void check_pointer(Params& params, cudaStream_t s) {
  assert(!params.act_scale);               // act_scale is not supported for now.
  assert(!params.apply_alpha_in_advance);  // apply_alpha_in_advance is not supported for now.

#if USE_COMPACT_FPA_INTB_GEMM
  ORT_ENFORCE(!params.zeros && !params.bias,
              "Compact fpA_intB GEMV does not support zero points or bias");
  dispatcher<Details, GroupSize, false, false, false, false>(params, s);
#else
  if (params.zeros && params.bias) {
    dispatcher<Details, GroupSize, false, true, true, false>(params, s);
  } else if (!params.zeros && params.bias) {
    dispatcher<Details, GroupSize, false, false, true, false>(params, s);
  } else if (params.zeros && !params.bias) {
    dispatcher<Details, GroupSize, false, true, false, false>(params, s);
  } else {
    dispatcher<Details, GroupSize, false, false, false, false>(params, s);
  }
#endif
}

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s) {
#if USE_COMPACT_FPA_INTB_GEMM
  constexpr bool kCompact = true;
#else
  constexpr bool kCompact = false;
#endif
  if constexpr (isGroupwise) {
    // A thread dequantizes kStepK consecutive weights with a single scale, so a group smaller
    // than kStepK has no valid instantiation (kStepK is 64 for 2-bit weights).
    if (params.groupsize == 32) {
      if constexpr (Details::kStepK <= 32) {
        check_pointer<Details, 32>(params, s);
        return;
      }
    } else if (params.groupsize == 64) {
      // The compact set carries block_size 64 for 2-bit only, since 2-bit cannot use 32.
      if constexpr (!kCompact || Details::kStepK >= 64) {
        check_pointer<Details, 64>(params, s);
        return;
      }
    } else if (params.groupsize == 128) {
      if constexpr (!kCompact) {
        check_pointer<Details, 128>(params, s);
        return;
      }
    }
  }

  ORT_THROW("unsupported block_size: ", params.groupsize);
}

#define INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(KType, A, B, Layout, ConverterInterleave, KTile) \
  template void select_gs<kernel_type_traits<KType>::isGroupwise,                                 \
                          KernelDetails<A, B, Layout, ConverterInterleave, KTile>>(Params & params, cudaStream_t s);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
