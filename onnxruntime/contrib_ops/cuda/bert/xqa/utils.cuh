/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_hint.cuh"

#ifdef __CUDA_ARCH__
#define XQA_UNROLL _Pragma("unroll")
#else
#define XQA_UNROLL
#endif
#include "utils.h"

#ifndef GENERATE_CUBIN
#include <cstdint>
#else
#include "mha_stdheaders.cuh"
#endif

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "barriers.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

inline constexpr float log2e = 1.4426950408889634f;  // std::log2(M_E)
// we used an optimization where exp(x-rowMax) is computed as:
/*  bias = rowMax * log2e  // shared for the whole row
    exp(x-rowMax) = exp2f(x * log2e - bias)
*/
// this reason, don't set safeInitRowMax with a huge absolute value.
// #define SAFE_INIT_ROW_MAX (-1e+5F)  // moved to defines.h
// Marked __constant__ (like kE4M3_MAX below) because the paged-KV kernels odr-use it from device
// code (Vec::filled takes a const reference), which a plain host constexpr variable cannot satisfy.
__constant__ constexpr int32_t kBAD_PAGE_INDEX = -1;
__constant__ constexpr float kE4M3_MAX = 448.F;

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
constexpr uint32_t kMAX_SMEM_SIZE = (99u << 10);
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 870
constexpr uint32_t kMAX_SMEM_SIZE = (163u << 10);
#elif __CUDA_ARCH__ == 900
constexpr uint32_t kMAX_SMEM_SIZE = (227u << 10);
#else
constexpr uint32_t kMAX_SMEM_SIZE = (48u << 10);  // Default for older architectures
#endif
#endif

__device__ inline void assertWarpConverged() {
  // assert(__activemask() == ~0U);
}

#define DEFINE_VEC_BINARY_FUNC(func)                                                               \
  template <typename T, uint32_t size>                                                             \
  __device__ __host__ inline Vec<decltype(func(mha::declval<T>(), mha::declval<T>())), size> func( \
      const Vec<T, size>& a, const Vec<T, size>& b) {                                              \
    Vec<decltype(func(mha::declval<T>(), mha::declval<T>())), size> result;                        \
    XQA_UNROLL for (uint32_t i = 0; i < size; i++) {                                               \
      result[i] = func(a[i], b[i]);                                                                \
    }                                                                                              \
    return result;                                                                                 \
  }
DEFINE_VEC_BINARY_FUNC(max)
DEFINE_VEC_BINARY_FUNC(fmaxf)
DEFINE_VEC_BINARY_FUNC(__hadd2_rn)

__device__ __host__ inline float2 addFloat2(float2 a, float2 b) {
  return float2{a.x + b.x, a.y + b.y};
}
DEFINE_VEC_BINARY_FUNC(addFloat2)
#undef DEFINE_VEC_BINARY_FUNC
#define DEFINE_VEC_BINARY_OP(op)                                                                      \
  template <typename T, uint32_t size>                                                                \
  __device__ __host__ inline Vec<decltype(mha::declval<T>() op mha::declval<T>()), size> operator op( \
      const Vec<T, size>& a, const Vec<T, size>& b) {                                                 \
    Vec<decltype(mha::declval<T>() op mha::declval<T>()), size> result;                               \
    XQA_UNROLL for (uint32_t i = 0; i < size; i++) {                                                  \
      result[i] = a[i] op b[i];                                                                       \
    }                                                                                                 \
    return result;                                                                                    \
  }                                                                                                   \
  template <typename T, uint32_t size, typename Scalar>                                               \
  __device__ __host__ inline Vec<decltype(mha::declval<T>() op mha::declval<T>()), size> operator op( \
      const Vec<T, size>& a, const Scalar & b) {                                                      \
    Vec<decltype(mha::declval<T>() op mha::declval<Scalar>()), size> result;                          \
    XQA_UNROLL for (uint32_t i = 0; i < size; i++) {                                                  \
      result[i] = a[i] op b;                                                                          \
    }                                                                                                 \
    return result;                                                                                    \
  }                                                                                                   \
  template <typename Scalar, typename T, uint32_t size>                                               \
  __device__ __host__ inline Vec<decltype(mha::declval<T>() op mha::declval<T>()), size> operator op( \
      const Scalar & a, const Vec<T, size>& b) {                                                      \
    Vec<decltype(mha::declval<Scalar>() op mha::declval<T>()), size> result;                          \
    XQA_UNROLL for (uint32_t i = 0; i < size; i++) {                                                  \
      result[i] = a op b[i];                                                                          \
    }                                                                                                 \
    return result;                                                                                    \
  }
// Don't use DEFINE_VEC_BINARY_FUNC(operator+), as operator+(float, float) is undefined,
// and float will be converted into half to perform the operation, which results in much
// lower precision. It's a defect of C++ that operator+(1.F, 2.F) does not work!
DEFINE_VEC_BINARY_OP(+)
DEFINE_VEC_BINARY_OP(-)
DEFINE_VEC_BINARY_OP(*)
DEFINE_VEC_BINARY_OP(/)
DEFINE_VEC_BINARY_OP(==)
DEFINE_VEC_BINARY_OP(!=)
DEFINE_VEC_BINARY_OP(>)
DEFINE_VEC_BINARY_OP(<)
DEFINE_VEC_BINARY_OP(>=)
DEFINE_VEC_BINARY_OP(<=)
#undef DEFINE_VEC_BINARY_OP

template <uint32_t size>
HOST_DEVICE_FUNC inline bool all(const Vec<bool, size>& src) {
  bool ret = true;
  XQA_UNROLL
  for (uint32_t i = 0; i < size; i++) {
    ret = ret && src[i];
  }
  return ret;
}

template <uint32_t size>
HOST_DEVICE_FUNC inline bool any(const Vec<bool, size>& src) {
  bool ret = false;
  XQA_UNROLL
  for (uint32_t i = 0; i < size; i++) {
    ret = ret || src[i];
  }
  return ret;
}

#define DEFINE_VEC_UNARY_OP(op)                                                                     \
  template <typename T, uint32_t size>                                                              \
  __device__ __host__ inline Vec<decltype(op(mha::declval<T>())), size> op(const Vec<T, size>& a) { \
    Vec<decltype(op(mha::declval<T>())), size> result;                                              \
    XQA_UNROLL for (uint32_t i = 0; i < size; i++) {                                                \
      result[i] = op(a[i]);                                                                         \
    }                                                                                               \
    return result;                                                                                  \
  }
DEFINE_VEC_UNARY_OP(expf)
DEFINE_VEC_UNARY_OP(exp2f)
DEFINE_VEC_UNARY_OP(__float2bfloat162_rn)
DEFINE_VEC_UNARY_OP(__float2half2_rn)
DEFINE_VEC_UNARY_OP(__float22half2_rn)
DEFINE_VEC_UNARY_OP(__bfloat1622float2)
DEFINE_VEC_UNARY_OP(__half22float2)
DEFINE_VEC_UNARY_OP(__frcp_rn)
#undef DEFINE_VEC_UNARY_OP

template <typename Dst, typename Src, uint32_t size>
__device__ __host__ inline Vec<Dst, size> convert(const Vec<Src, size>& src) {
  if constexpr (mha::is_same_v<mha::decay_t<Dst>, mha::decay_t<Src>>) {
    return src;
  }
  Vec<Dst, size> dst;
  if constexpr (mha::is_same_v<Src, half> && mha::is_same_v<Dst, float>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<float2&>(dst[i]) = __half22float2(reinterpret_cast<const half2&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, float> && mha::is_same_v<Dst, half>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<half2&>(dst[i]) = __float22half2_rn(reinterpret_cast<const float2&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  }
  if constexpr (mha::is_same_v<Src, __nv_bfloat16> && mha::is_same_v<Dst, float>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<float2&>(dst[i]) = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, float> && mha::is_same_v<Dst, __nv_bfloat16>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<__nv_bfloat162&>(dst[i]) = __float22bfloat162_rn(reinterpret_cast<const float2&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, __nv_fp8_e4m3> && mha::is_same_v<Dst, float>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<float2&>(dst[i]) = float2(reinterpret_cast<const __nv_fp8x2_e4m3&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, float> && mha::is_same_v<Dst, __nv_fp8_e4m3>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<__nv_fp8x2_e4m3&>(dst[i]) = __nv_fp8x2_e4m3{float2{src[i], src[i + 1]}};
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, __nv_fp8_e4m3> && mha::is_same_v<Dst, half>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<half2&>(dst[i]) = half2(reinterpret_cast<const __nv_fp8x2_e4m3&>(src[i]));
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else if constexpr (mha::is_same_v<Src, half> && mha::is_same_v<Dst, __nv_fp8_e4m3>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<__nv_fp8x2_e4m3&>(dst[i]) = __nv_fp8x2_e4m3{reinterpret_cast<const half2&>(src[i])};
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  }
  // else if constexpr (mha::is_same_v<Src, __nv_fp8_e4m3> && mha::is_same_v<Dst, __nv_bfloat16>) {
  //     static_assert("not implemented");
  // }
  else if constexpr (mha::is_same_v<Src, __nv_bfloat16> && mha::is_same_v<Dst, __nv_fp8_e4m3>) {
    for (uint32_t i = 0; i < size - 1; i += 2) {
      reinterpret_cast<__nv_fp8x2_e4m3&>(dst[i]) = __nv_fp8x2_e4m3{reinterpret_cast<const __nv_bfloat162&>(src[i])};
    }
    if constexpr (size % 2 != 0) {
      dst[size - 1] = Dst{src[size - 1]};
    }
  } else {
    for (uint32_t i = 0; i < size; i++) {
      dst[i] = Dst{src[i]};
    }
  }
  return dst;
}

__device__ inline uint32_t laneId() {
  uint32_t id;
  asm("mov.u32 %0, %%laneid;\n" : "=r"(id));
  return id;
}

__device__ inline uint32_t dynamicSmemSize() {
  uint32_t size;
  asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(size));
  return size;
}

__device__ inline void trap() {
  asm volatile("trap;\n");
}

inline constexpr uint32_t warp_size = 32;

struct Warp {
};

__device__ inline Warp this_warp() {
  return {};
}

// @fixme: check asm code to make sure UR is used and SHFL is not generated.
template <typename T>
__device__ inline T makeWarpUniform(const Warp& warp, const T& val) {
  const T val0 = __shfl_sync(~0U, val, 0);
  assert(val == val0);
  return val0;
}

__device__ inline uint3 getWarpIdx(uint3 ctaShapeInWarps, const Warp& warp = this_warp()) {
  assert(ctaShapeInWarps.x % 128 == 0);
  return uint3{ctaShapeInWarps.x == 1 ? 0 : makeWarpUniform(warp, threadIdx.x / warp_size),
               ctaShapeInWarps.y == 1 ? 0 : makeWarpUniform(warp, threadIdx.y),
               ctaShapeInWarps.z == 1 ? 0 : makeWarpUniform(warp, threadIdx.z)};
}

constexpr uint32_t cacheLineSize = 128;

template <uint32_t x>
__device__ __host__ inline void assertIsPowerOf2() {
  static_assert((x & (x - 1)) == 0);
}

template <typename T>
__device__ inline bool hasBankConflict(T* p) {
  static_assert(sizeof(T) % 4 == 0 && sizeof(T) <= 16 && alignof(T) == sizeof(T));
  constexpr uint32_t grpSize = 128 / sizeof(T);
  const uint32_t grpMask = static_cast<uint32_t>(((1ULL << grpSize) - 1ULL) << (laneId() / grpSize * grpSize));
  const uint32_t x = reinterpret_cast<uintptr_t>(p) / sizeof(T) % grpSize;
  const auto match = __match_any_sync(grpMask, x);
  const bool conflict = __popc(match) > 1;
  if (grpSize <= 8 && conflict) {
    char str[grpSize * 2 + 1] = {};
    for (uint32_t i = 0; i < grpSize; i++) {
      str[i * 2] = __shfl_sync(grpMask, x, i, grpSize) + '0';
      str[i * 2 + 1] = ' ';
    }
    if (laneId() % grpSize == 0) {
      printf("bank conflict (%u): %s\n", match, str);
    }
  }
  return conflict;
}

__device__ inline float atomicMax(float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax(reinterpret_cast<int32_t*>(addr), __float_as_int(value)))
                     : __uint_as_float(atomicMin(reinterpret_cast<uint32_t*>(addr), __float_as_uint(value)));
  return old;
}

__device__ inline bool isInInt32Range(uint32_t x) {
  return x <= static_cast<uint32_t>(mha::numeric_limits<int32_t>::max());
}

// struct of arrays instead of array of structs for compact storage
template <typename Pointer, size_t length>
struct CompactRangeList {
  mha::array<Pointer, length> pointerList;
  mha::array<uint32_t, length> sizeList;

  struct Range {
    const Pointer& data;
    const uint32_t& size;
  };

  __device__ inline Range operator[](uint32_t i) const {
    return Range{pointerList[i], sizeList[i]};
  }
};

// alignedForSwizzle is for case when you need to mix TMA+LDS/LDSM, or LDGSTS/STS/STSM+GMMA
template <typename T, uint32_t rows_, uint32_t cols_, bool alignedForSwizzle = true>
struct alignas(mha::min<uint32_t>(maxArrayAlign<T>(rows_* cols_), cacheLineSize)) Array2D {
  using Elem = T;
  static constexpr uint32_t rows = rows_;
  static constexpr uint32_t cols = cols_;
  static constexpr uint32_t size = rows * cols;
  static constexpr uint32_t rowBytes = sizeof(T) * cols;

  template <bool swizzle = false>
  const __device__ inline T& at(uint32_t r, uint32_t c) const {
    assert(r < rows && c < cols);
    // two different swizzle styles
#if 1
    const uint32_t c_swizzled = [&] {
      if constexpr (swizzle) {
        static_assert(rowBytes % cacheLineSize == 0 || cacheLineSize % rowBytes == 0);
        static constexpr uint32_t rowsPerSliding = exactDiv(cacheLineSize, rowBytes % cacheLineSize == 0 ? cacheLineSize : rowBytes % cacheLineSize);
        constexpr uint32_t swizzleRowsRepeat = exactDiv(cacheLineSize, sizeof(Elem));
        const auto runtimeBaseOffset = static_cast<uint32_t>(__cvta_generic_to_shared(this->data)) / rowBytes % rows;
        const uint32_t baseOffset = alignedForSwizzle
                                        ? 0
                                        : runtimeBaseOffset;  // To match TMA when array is not aligned to pattern boundary
        const uint32_t xorMask = alignedForSwizzle
                                     ? BoundedVal<rows>{r}
                                           .template divBy<rowsPerSliding>()
                                           .template mod<exactDiv(swizzleRowsRepeat, rowsPerSliding)>()
                                           .get()
                                     : (r + baseOffset) / rowsPerSliding % exactDiv(swizzleRowsRepeat, rowsPerSliding);
        return c ^ xorMask;
      }
      return c;
    }();
#else
    const uint32_t c_swizzled = swizzle ? (c + r / rowsPerSliding) % cols : c;
#endif
    const T& ret = (&data[0][0])[r * cols + c_swizzled];
    assert(&data[r][c_swizzled] == &ret);
    return ret;
  }

  template <bool swizzle = false>
  __device__ inline T& at(uint32_t r, uint32_t c) {
    return const_cast<T&>(static_cast<const Array2D*>(this)->at<swizzle>(r, c));
  }

  const __device__ inline T& operator()(uint32_t r, uint32_t c) const {
    return at<false>(r, c);
  }

  __device__ inline T& operator()(uint32_t r, uint32_t c) {
    return at<false>(r, c);
  }

  template <typename T2>
  __device__ inline Array2D<T2, rows, exactDiv(sizeof(T) * cols, sizeof(T2))>& as() {
    return reinterpret_cast<Array2D<T2, rows, exactDiv(sizeof(T) * cols, sizeof(T2))>&>(*this);
  }

  __device__ inline void fill(T val) {
    XQA_UNROLL
    for (uint32_t i = 0; i < rows * cols; i++) {
      (&data[0][0])[i] = val;
    }
  }

  __device__ inline static Array2D<T, rows, cols> filled(T val) {
    Array2D<T, rows, cols> ret;
    ret.fill(val);
    return ret;
  }

  T data[rows][cols];
};

#define DEFINE_ARRAY2D_BINARY_OP(op)                                                                            \
  template <typename T, uint32_t rows, uint32_t cols>                                                           \
  __device__ __host__ inline Array2D<decltype(mha::declval<T>() op mha::declval<T>()), rows, cols> operator op( \
      const Array2D<T, rows, cols>& a, const Array2D<T, rows, cols>& b) {                                       \
    Array2D<decltype(mha::declval<T>() op mha::declval<T>()), rows, cols> result;                               \
    XQA_UNROLL for (uint32_t i = 0; i < rows; i++) {                                                            \
      for (uint32_t j = 0; j < cols; j++) {                                                                     \
        result(i, j) = a(i, j) op b(i, j);                                                                      \
      }                                                                                                         \
    }                                                                                                           \
    return result;                                                                                              \
  }                                                                                                             \
  template <typename T, uint32_t rows, uint32_t cols, typename Scalar>                                          \
  __device__ __host__ inline Array2D<decltype(mha::declval<T>() op mha::declval<T>()), rows, cols> operator op( \
      const Array2D<T, rows, cols>& a, const Scalar & b) {                                                      \
    Array2D<decltype(mha::declval<T>() op mha::declval<Scalar>()), rows, cols> result;                          \
    XQA_UNROLL for (uint32_t i = 0; i < rows; i++) {                                                            \
      for (uint32_t j = 0; j < cols; j++) {                                                                     \
        result(i, j) = a(i, j) op b;                                                                            \
      }                                                                                                         \
    }                                                                                                           \
    return result;                                                                                              \
  }                                                                                                             \
  template <typename Scalar, typename T, uint32_t rows, uint32_t cols>                                          \
  __device__ __host__ inline Array2D<decltype(mha::declval<T>() op mha::declval<T>()), rows, cols> operator op( \
      const Scalar & a, const Array2D<T, rows, cols>& b) {                                                      \
    Array2D<decltype(mha::declval<Scalar>() op mha::declval<T>()), rows, cols> result;                          \
    XQA_UNROLL for (uint32_t i = 0; i < rows; i++) {                                                            \
      for (uint32_t j = 0; j < cols; j++) {                                                                     \
        result(i, j) = a op b(i, j);                                                                            \
      }                                                                                                         \
    }                                                                                                           \
    return result;                                                                                              \
  }
// Don't use DEFINE_VEC_BINARY_FUNC(operator+), as operator+(float, float) is undefined,
// and float will be converted into half to perform the operation, which results in much
// lower precision. It's a defect of C++ that operator+(1.F, 2.F) does not work!
DEFINE_ARRAY2D_BINARY_OP(+)
DEFINE_ARRAY2D_BINARY_OP(-)
DEFINE_ARRAY2D_BINARY_OP(*)

using LdGrain = Vec<uint32_t, 4>;
constexpr uint32_t grainBytes = sizeof(LdGrain);

// wrapper for PTX ldmatrix
template <bool transpose, uint32_t nbMat>
__device__ inline Vec<uint32_t, nbMat> ldmatrix(const LdGrain* row) {
  assertWarpConverged();
  uint32_t a, b, c, d;
  if constexpr (nbMat == 4) {
    if (transpose) {
      asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    } else {
      asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    }
#if 0
        auto checkMat = [&](uint32_t val, uint32_t idxMat) -> const Vec<uint16_t, 8>& {
            const auto v = (const Vec<uint16_t, 2>&)val;
            const uint32_t lane = laneId();
            auto getRow = [&](uint32_t r) {
                assert(r<8);
                const auto ret = __shfl_sync(~0U, reinterpret_cast<const uint64_t&>(row), 8*idxMat+r);
                return *reinterpret_cast<const Vec<uint16_t, 8>*>(ret);
            };
            auto checkEq = [](uint16_t x, uint16_t y) {
                if (!(x==y)) {
                    printf("x=%u, y= %u\n", (unsigned)x, (unsigned)y);
                }
            };
            if (transpose) {
                checkEq(v[0], getRow(lane % 4 * 2)[lane / 4]);
                checkEq(v[1], getRow(lane % 4 * 2 + 1)[lane / 4]);
            }
            else {
                checkEq(v[0], getRow(lane / 4)[lane % 4 * 2]);
                checkEq(v[1], getRow(lane / 4)[lane % 4 * 2 + 1]);
            }
        };
        checkMat(a, 0);
        checkMat(b, 1);
        checkMat(c, 2);
        checkMat(d, 3);
#endif
    return Vec<uint32_t, 4>{a, b, c, d};
  } else if constexpr (nbMat == 2) {
    if (transpose) {
      asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
          : "=r"(a), "=r"(b)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    } else {
      asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
          : "=r"(a), "=r"(b)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    }
    return Vec<uint32_t, 2>{a, b};
  } else if constexpr (nbMat == 1) {
    if (transpose) {
      asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 %0, [%1];\n"
          : "=r"(a)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    } else {
      asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 %0, [%1];\n"
          : "=r"(a)
          : "l"(__cvta_generic_to_shared(row))
          : "memory");
    }
    return Vec<uint32_t, 1>{a};
  } else {
    static_assert(nbMat == 1 || nbMat == 2 || nbMat == 4);
  }
}

template <bool transpose>
__device__ inline Vec<uint32_t, 4> ldmatrix_4x(const Warp& warp, const LdGrain* row) {
  return ldmatrix<transpose, 4>(row);
}

template <uint32_t nbMat>
__device__ inline Vec<uint32_t, nbMat * 2> ldmatrix_16x16_trans(const LdGrain* row) {
  uint32_t a, b, c, d;
  if constexpr (nbMat == 1) {
    asm("ldmatrix.sync.aligned.m16n16.x1.trans.shared::cta.b8 {%0, %1}, [%2];\n"
        : "=r"(a), "=r"(b)
        : "l"(__cvta_generic_to_shared(row))
        : "memory");
    return Vec<uint32_t, 2>{a, b};
  } else if constexpr (nbMat == 2) {
    asm("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "l"(__cvta_generic_to_shared(row))
        : "memory");
    return Vec<uint32_t, 4>{a, b, c, d};
  } else {
    static_assert(nbMat == 1 || nbMat == 2);
  }
}

template <bool transpose, uint32_t nbMat>
__device__ inline void stmatrix(LdGrain* row, const Vec<uint32_t, nbMat>& data) {
#if __CUDA_ARCH__ >= 900
  assertWarpConverged();
  if constexpr (nbMat == 4) {
    if constexpr (transpose) {
      asm("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(
              __cvta_generic_to_shared(row)),
          "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3])
          : "memory");
    } else {
      asm("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(
              __cvta_generic_to_shared(row)),
          "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3])
          : "memory");
    }
  } else if constexpr (nbMat == 2) {
    if constexpr (transpose) {
      asm("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" ::"l"(__cvta_generic_to_shared(row)),
          "r"(data[0]), "r"(data[1])
          : "memory");
    } else {
      asm("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n" ::"l"(__cvta_generic_to_shared(row)),
          "r"(data[0]), "r"(data[1])
          : "memory");
    }
  } else if constexpr (nbMat == 1) {
    if constexpr (transpose) {
      asm("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};\n" ::"l"(__cvta_generic_to_shared(row)),
          "r"(data[0])
          : "memory");
    } else {
      asm("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"l"(__cvta_generic_to_shared(row)),
          "r"(data[0])
          : "memory");
    }
  } else {
    static_assert(nbMat == 1 || nbMat == 2 || nbMat == 4);
  }
#else
  trap();
#endif
}

template <bool transpose>
__device__ inline void stmatrix_4x(const Warp& warp, LdGrain* row, const Vec<uint32_t, 4>& data) {
  stmatrix<transpose, 4>(row, data);
}

struct None {
};

template <bool real, typename T>
using RealTypeOrNone = mha::conditional_t<real, T, None>;

template <Scope producerScope = Scope::CTA, Scope consumerScope = Scope::CTA>
struct MBarrierPair {
  MBarrier<producerScope> produced;
  MBarrier<consumerScope> consumed;

  __device__ inline void initialize(uint32_t producedCount, uint32_t consumedCount) {
    init(&produced, producedCount);
    init(&consumed, consumedCount);
  }
};

using CtaBarrierPair = MBarrierPair<Scope::CTA, Scope::CTA>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <Scope scope>
__device__ inline auto arrive_tx(MBarrier<scope>& bar, uint32_t txCount, uint32_t arriveCount = 1) {
#if USE_CUSTOM_BARRIER
  return bar.arrive_tx(txCount, arriveCount);
#else
  return cuda::device::barrier_arrive_tx(bar, arriveCount, txCount);
#endif
}

template <Scope scope>
__device__ inline void arrive_tx_and_wait(MBarrier<scope>& bar, uint32_t txCount, uint32_t arriveCount = 1) {
  bar.wait(arrive_tx(bar, txCount, arriveCount));
}
#endif

template <uint32_t bound0>
__device__ inline mha::tuple<uint32_t, uint32_t> carryLE(uint32_t i0, uint32_t iLast) {
  return mha::tuple<uint32_t, uint32_t>{i0 % bound0, iLast + i0 / bound0};
}

template <uint32_t bound0, uint32_t bound1, uint32_t... bounds>
__device__ inline mha::tuple<uint32_t, uint32_t, decltype(bounds)..., uint32_t> carryLE(
    uint32_t i0, uint32_t i1, decltype(bounds)... i, uint32_t iLast) {
  return mha::tuple_cat(mha::tuple<uint32_t>(i0 % bound0), carryLE<bound1, bounds...>(i1 + i0 / bound0, i..., iLast));
}

__device__ __host__ inline void assertClose([[maybe_unused]] float a, [[maybe_unused]] float b, [[maybe_unused]] float threshold = 0.01f) {
  assert(abs(a - b) < threshold);
}

__device__ __host__ inline void assertClose([[maybe_unused]] half a, [[maybe_unused]] half b, [[maybe_unused]] float threshold = 0.01f) {
  assertClose(__half2float(a), __half2float(b), threshold);
}

// Converts four packed signed int8 values into four half values without using I2F.
//
// A half whose bit pattern is 0x64XX is exactly 1024 + XX: the exponent field of 0x6400
// selects 2^10, where the mantissa ULP is 1, so the low 8 mantissa bits hold the byte
// verbatim. Splicing a byte in with prmt and subtracting a magic constant therefore
// performs the conversion using only full-rate integer/half2 ALU instructions. The
// generic path instead costs one quarter-rate I2F per element plus a sign-extension
// sequence and a repack, which is the bulk of the int8-vs-fp8 gap in this kernel.
//
// Source bytes are biased first (XOR 0x80 maps two's-complement s to unsigned s + 128),
// so the spliced value is 1152 + s and the constant to subtract is half(1152) == 0x6480.
// 1152 + s lies in [1024, 1279] and the result s lies in [-128, 127]; both are exactly
// representable in half, so this is bit-identical to the I2F path.
//
// selector0 / selector1 are prmt byte selectors choosing which source byte feeds the low
// and high half of each output word, which lets callers fold a byte permutation in.
template <uint32_t selector0, uint32_t selector1>
__device__ inline Vec<uint32_t, 2> cvtS8x4ToF16x4(uint32_t i8data) {
  static constexpr uint32_t kSignFlip = 0x80808080U;  // two's complement -> biased unsigned
  static constexpr uint32_t kExpBytes = 0x64646464U;  // half exponent byte for 1024.0
  static constexpr uint32_t kMagic = 0x64806480U;     // half2(1152.0, 1152.0)
  uint32_t const biased = i8data ^ kSignFlip;
  Vec<uint32_t, 2> ret;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(ret.data[0]) : "r"(biased), "n"(kExpBytes), "n"(selector0));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(ret.data[1]) : "r"(biased), "n"(kExpBytes), "n"(selector1));
  // kMagic must be a register operand: f16x2 immediates are not encodable.
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(ret.data[0]) : "r"(ret.data[0]), "r"(kMagic));
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(ret.data[1]) : "r"(ret.data[1]), "r"(kMagic));
  return ret;
}

template <typename InputElem, typename CacheElem>
__device__ inline Vec<uint32_t, 2> convertKCacheWordToF16(uint32_t i8data) {
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert(sizeof(CacheElem) == 1);
  Vec<uint32_t, 2> ret;
#if (defined __CUDA_ARCH__)
  if constexpr (mha::is_same_v<InputElem, half> && mha::is_same_v<CacheElem, int8_t>) {
    // dst[i] = src[i], so byte i of the input feeds output half i.
    return cvtS8x4ToF16x4<0x5150, 0x5352>(i8data);
  }
#endif
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  if constexpr (mha::is_same_v<InputElem, half> && mha::is_same_v<CacheElem, __nv_fp8_e4m3>) {
    uint16_t (&src)[2] = reinterpret_cast<uint16_t (&)[2]>(i8data);
    uint32_t (&dst)[2] = reinterpret_cast<uint32_t (&)[2]>(ret);
    asm("{\n"
        "cvt.rn.f16x2.e4m3x2 %0, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %1, %3;\n"
        "}"
        : "=r"(dst[0]), "=r"(dst[1])
        : "h"(src[0]), "h"(src[1]));
    return ret;
  }
#endif
  const CacheElem(&src)[4] = reinterpret_cast<CacheElem(&)[4]>(i8data);
  InputElem(&dst)[4] = reinterpret_cast<InputElem(&)[4]>(ret);
  XQA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    dst[i] = InputElem(src[i]);
  }
  return ret;
}

template <typename InputElem, typename CacheElem>
__device__ inline Vec<uint32_t, 2> convertVCacheWordToF16(uint32_t i8data) {
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert(sizeof(CacheElem) == 1);
  Vec<uint32_t, 2> ret;
#if (defined __CUDA_ARCH__)
  if constexpr (mha::is_same_v<InputElem, half> && mha::is_same_v<CacheElem, int8_t>) {
    // dst[i][j] = src[j][i], i.e. the 2x2 byte transpose {b0,b1,b2,b3} -> {b0,b2,b1,b3}.
    // Folded into the prmt selectors, so it costs nothing extra here.
    return cvtS8x4ToF16x4<0x5250, 0x5351>(i8data);
  }
#endif
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  if constexpr (mha::is_same_v<InputElem, half> && mha::is_same_v<CacheElem, __nv_fp8_e4m3>) {
    uint32_t (&dst)[2] = reinterpret_cast<uint32_t (&)[2]>(ret);
    asm("{\n"
        ".reg .b32 dst0;\n"
        ".reg .b32 dst1;\n"
        ".reg .b32 src;\n"
        ".reg .b16 src0;\n"
        ".reg .b16 src1;\n"
        "prmt.b32 src, %2, 0x0, 0x3120;\n"
        "mov.b32 {src0, src1}, src;\n"
        "cvt.rn.f16x2.e4m3x2 %0, src0;\n"
        "cvt.rn.f16x2.e4m3x2 %1, src1;\n"
        "}"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(i8data));
    return ret;
  }
#endif
  const CacheElem(&src)[2][2] = reinterpret_cast<CacheElem(&)[2][2]>(i8data);
  InputElem(&dst)[2][2] = reinterpret_cast<InputElem(&)[2][2]>(ret);
  XQA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    XQA_UNROLL
    for (uint32_t j = 0; j < 2; j++) {
      dst[i][j] = InputElem(src[j][i]);
    }
  }

  return ret;
}

struct PermuteOrder {
  uint16_t x0 : 4;
  uint16_t x1 : 4;
  uint16_t x2 : 4;
  uint16_t x3 : 4;
};

static_assert(sizeof(PermuteOrder) == 2);

__device__ inline uint32_t prmt(uint32_t a, uint32_t b, PermuteOrder order) {
  uint32_t d;
  const uint32_t c = reinterpret_cast<const uint16_t&>(order);
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ inline uint32_t movmatrix(uint32_t src) {
  uint32_t dst;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(dst) : "r"(src));
  return dst;
}

__device__ inline bool warpElectSync() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t pred = 0;
  asm volatile(
      "{\n"
      "    .reg .b32 d;\n"
      "    .reg .pred p;\n"
      "    elect.sync d|p, 0xFFFFFFFF;\n"
      "    selp.b32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(pred));
  return pred != 0;
#else
  assert("not available");
  return false;
#endif
}

__device__ inline void preExit() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
}

__device__ inline void acqBulk() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;\n");
#endif
}

__device__ inline uint3 nbClusters() {
  uint3 id;
  asm("mov.v4.u32 {%0, %1, %2, _}, %%nclusterid;\n" : "=r"(id.x), "=r"(id.y), "=r"(id.z));
  return id;
}

__device__ inline uint3 clusterId() {
  uint3 id;
  asm("mov.v4.u32 {%0, %1, %2, _}, %%clusterid;\n" : "=r"(id.x), "=r"(id.y), "=r"(id.z));
  return id;
}

__device__ inline uint32_t clusterCtaRank() {
#if __CUDA_ARCH__ >= 900
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  return rank;
#else
  return 0;
#endif
}

__device__ inline uint3 clusterCtaId() {
  uint3 id;
  asm("mov.v4.u32 {%0, %1, %2, _}, %%cluster_ctaid;\n" : "=r"(id.x), "=r"(id.y), "=r"(id.z));
  return id;
}

// src and return are both generic address
template <typename T>
__device__ inline T* mapa(T* src, uint32_t clusterCtaRank) {
  uint64_t dst;
  asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(dst) : "l"(reinterpret_cast<uint64_t>(src)), "r"(clusterCtaRank));
  return reinterpret_cast<T*>(dst);
}

template <typename T>
__device__ inline T& mapa(T& src, uint32_t clusterCtaRank) {
  return *mapa(&src, clusterCtaRank);
}

__device__ inline void clusterBarArrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n");
}

__device__ inline void clusterBarWait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n");
}

__device__ inline uint32_t clock32() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%clock;\n" : "=r"(ret)::"memory");
  return ret;
}

template <uint32_t nbBufs, Scope producerScope = Scope::CTA, Scope consumerScope = Scope::CTA>
struct BarWaiter {
  MBarrierPair<producerScope, consumerScope> (*bars)[nbBufs];
  uint32_t idx;
  uint32_t idxBuf;
  bool skipBarWait = false;

  __device__ inline BarWaiter(MBarrierPair<producerScope, consumerScope> (&bars)[nbBufs], uint32_t idx)
      : bars{&bars}, idx{idx}, idxBuf{idx % nbBufs} {
  }

  __device__ inline bool testWait() {
    const bool parity = toParity<nbBufs>(idx);
    skipBarWait = bar().produced.test_wait_parity(parity);
    return skipBarWait;
  }

  __device__ inline BarWaiter next(uint32_t step = 1) {
    return BarWaiter{*bars, idx + step};
  }

  __device__ inline void wait() {
    if (!skipBarWait) {
      bar().produced.wait_parity(toParity<nbBufs>(idx));
    }
  }

  __device__ inline MBarrierPair<producerScope, consumerScope>& bar() {
    return (*bars)[idxBuf];
  }

  __device__ inline void consumed() {
    bar().consumed.arrive();
  }
};

class Timer {
 public:
  __device__ inline Timer() {
    reset();
  }

  __device__ inline void print(const char* name = "unnamed", bool reset = false) {
    const auto toc = clock32();
    printf("%s: %u (block={%u, %u, %u})\n", name, toc - mTic, blockIdx.x, blockIdx.y, blockIdx.z);
    if (reset) {
      this->reset();
    }
  }

  __device__ inline void reset() {
    mTic = clock32();
  }

 private:
  uint32_t mTic;
};

// [beg, end)
struct Range {
  uint32_t beg, end;
};

constexpr bool overlap(Range a, Range b) {
  return a.beg < b.end && b.beg < a.end;
}
